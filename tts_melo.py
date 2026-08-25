# -*- coding: utf-8 -*-
"""MeloTTS 离线实时中文 TTS 核心模块。

架构：句子级分块 + 生产者-消费者流式（"模拟"流式，因 MeloTTS/VITS 无原生流式接口）。

- 合成线程逐句合成音频数组，推入有界队列（maxsize=8，天然背压）；
- 播放线程取块，sounddevice 写声卡，边合成边播。
- TTFA = 首帧写进声卡时刻 - speak() 入口时刻，硬指标 <1s。
- 插桩全走开关：profile（时序分析）/ debug（详细日志），生产关闭时热路径零计时埋点。
- HF_HOME 重定向到本项目 .cache/hf，所有文件均在本项目路径下。

典型用法：
    tts = RealtimeTTS(device="cuda", speed=1.0, profile=True, debug=True)
    tts.speak("第一句话。第二句话。")          # 流式播放到扬声器
    tts.speak_to_file("整段文本", "audio/out.wav")  # 非流式落盘
"""
import os
import re
import queue
import threading
import time
import wave

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 必须在 import melo / torch 之前设置，确保权重缓存落在项目内
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))
# huggingface.co 直连被墙时走镜像（仅首次下载用，运行期零网络）；可用环境变量覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# NLTK 语料（g2p_en 的 cmudict）落到项目内，导入期零联网
os.environ.setdefault("NLTK_DATA", os.path.join(_PROJECT_DIR, ".cache", "nltk_data"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from melo.api import TTS  # noqa: E402


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def save_wav_np(audio, path, samplerate):
    """把 float32 音频数组写成 16bit PCM WAV（无需 soundfile/scipy 依赖）。"""
    a = np.asarray(audio)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if np.abs(a).max() > 1.0:
        a = a / 32767.0
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(samplerate))
        w.writeframes(pcm.tobytes())


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device 只支持 auto/cpu/cuda，收到: %r" % device)
    return device


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class RealtimeTTS:
    def __init__(self, device="auto", speed=1.0, profile=False, debug=False):
        self._profile = profile
        self._debug = debug
        self._speed = speed
        self._device = _resolve_device(device)
        self._stop_flag = threading.Event()
        self.last_ttfa = None
        self.last_timing = []

        if self._debug:
            self._env_banner()

        t0 = time.perf_counter()
        if self._debug:
            self._ensure_weights()
        self._model = TTS(language="ZH", device=self._device)
        self._sr = int(getattr(self._model.hps.data, "sampling_rate", 44100))
        self._spk = self._model.hps.data.spk2id["ZH"]
        t_load = time.perf_counter() - t0

        # 预热：cudnn / 首次推理缓存，不计入验收
        t0 = time.perf_counter()
        self._synth("你好。")
        t_warm = time.perf_counter() - t0

        if self._debug:
            print("[debug] 模型加载 %.2fs（含预热 %.2fs） device=%s sr=%d"
                  % (t_load, t_warm, self._device, self._sr))
            print("[debug] HF 权重缓存目录: %s" % os.environ["HF_HOME"])

    # ---------------- debug 辅助 ----------------
    def _env_banner(self):
        import platform
        print("[debug] ==== 环境/设备 ====")
        print("[debug] python: %s | torch: %s" % (platform.python_version(), torch.__version__))
        print("[debug] cuda.is_available=%s device选择=%s"
              % (torch.cuda.is_available(), self._device))
        if self._device == "cuda" and torch.cuda.is_available():
            print("[debug] 显卡型号: %s" % torch.cuda.get_device_name(0))
            free, total = torch.cuda.mem_get_info(0)
            print("[debug] 显存: 空闲 %.2f GB / 总 %.2f GB"
                  % (free / 1e9, total / 1e9))
        else:
            print("[debug] CPU 核数: %d | torch 线程数: %d"
                  % (os.cpu_count(), torch.get_num_threads()))

    def _ensure_weights(self):
        """显式预下载 ZH 权重（命中缓存则跳过），打印来源/大小/耗时。失败不影响主流程。"""
        try:
            from huggingface_hub import hf_hub_download, try_to_load_from_cache
            try:
                from melo.download_utils import LANG_TO_HF_REPO_ID
                repo = LANG_TO_HF_REPO_ID.get("ZH")
            except Exception:
                repo = None
            if not repo:
                print("[debug] 未能获取 ZH 权重仓库映射，跳过预下载（仍由 TTS 内部处理）")
                return
            for fname in ("config.json", "checkpoint.pth"):
                cached = try_to_load_from_cache(repo, fname)
                if cached is not None:
                    print("[debug] 权重命中缓存: %s <- %s" % (cached, repo))
                    continue
                print("[debug] 权重未缓存，将从 HF 下载: repo=%s file=%s" % (repo, fname))
                t0 = time.perf_counter()
                path = hf_hub_download(repo_id=repo, filename=fname)
                dt = time.perf_counter() - t0
                print("[debug] 下载完成: %s 字节，%.2fs -> %s"
                      % (os.path.getsize(path), dt, path))
        except Exception as e:  # 预检失败不阻塞主流程
            print("[debug] 权重预检失败(不影响主流程): %s" % e)

    # ---------------- 合成与分句 ----------------
    def _split(self, text):
        """按中文标点与换行切句；过短片段并入下一句，避免碎块。"""
        parts = re.split(r"(?<=[。！？；…\n])", text.strip())
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if sentences and len(sentences[-1]) < 6:
                sentences[-1] += p
            else:
                sentences.append(p)
        return sentences

    def _synth(self, text):
        """整句合成，返回 44.1kHz float32 numpy 数组（[-1,1]）。quiet=True 抑制内部切分打印。"""
        return self._model.tts_to_file(
            text, self._spk, output_path=None, speed=self._speed, quiet=True)

    # ---------------- 流式主接口 ----------------
    def speak(self, text, save_chunks_dir=None):
        """流式播放 text 到扬声器。返回逐句时序记录（profile 开时才有数据）。

        save_chunks_dir: 若非 None，每句音频另存为 <dir>/句<NN>.wav（供 HTML 试听）。
        """
        sentences = self._split(text)
        n = len(sentences)
        if n == 0:
            return []
        if save_chunks_dir:
            os.makedirs(save_chunks_dir, exist_ok=True)

        import sounddevice as sd
        self._stop_flag.clear()
        t_start = time.perf_counter() if self._profile else None
        q = queue.Queue(maxsize=8)
        timing = []  # 每句一条记录
        t_lock = threading.Lock()

        def producer():
            try:
                for i, sent in enumerate(sentences):
                    if self._stop_flag.is_set():
                        break
                    t_s0 = time.perf_counter() if self._profile else None
                    if self._debug:
                        print("[debug] [合成 句%d/%d] %s" % (i + 1, n, sent[:20]))
                    audio = self._synth(sent)
                    t_s1 = time.perf_counter() if self._profile else None
                    if save_chunks_dir:
                        save_wav_np(audio, os.path.join(save_chunks_dir, "句%02d.wav" % (i + 1)), self._sr)
                    if self._profile:
                        with t_lock:
                            timing.append({
                                "idx": i, "text": sent,
                                "synth_start": t_s0, "synth_end": t_s1,
                                "play_start": None, "play_end": None,
                                "audio_dur": len(audio) / self._sr,
                            })
                    q.put((i, sent, audio))
            except Exception as e:
                # 合成异常也放哨兵：消费线程收到 None 即退出，避免死锁
                if self._debug:
                    import traceback
                    print("[debug] 合成线程异常（跳过后续）: %s" % e)
                    traceback.print_exc()
            finally:
                q.put(None)

        def consumer():
            stream = sd.OutputStream(samplerate=self._sr, channels=1, dtype="float32")
            stream.start()
            try:
                while True:
                    item = q.get()
                    if item is None:
                        break
                    i, sent, audio = item
                    if self._stop_flag.is_set():
                        continue
                    t_p0 = time.perf_counter() if self._profile else None
                    if self._debug:
                        print("[debug] [正在播放 句%d/%d] %s" % (i + 1, n, sent[:20]))
                    stream.write(audio.astype(np.float32))
                    t_p1 = time.perf_counter() if self._profile else None
                    if self._profile:
                        with t_lock:
                            for rec in timing:
                                if rec["idx"] == i:
                                    rec["play_start"] = t_p0
                                    rec["play_end"] = t_p1
                                    break
                        if i == 0:
                            self.last_ttfa = t_p0 - t_start
            finally:
                stream.stop()
                stream.close()

        prod = threading.Thread(target=producer, daemon=True)
        cons = threading.Thread(target=consumer, daemon=True)
        prod.start()
        cons.start()
        prod.join()
        cons.join()

        if self._profile:
            self.last_timing = timing
            for rec in timing:
                rec["ttfa"] = rec["play_start"] - t_start
                rec["synth_dur"] = rec["synth_end"] - rec["synth_start"]
                rec["wait"] = rec["play_start"] - rec["synth_end"]
            for idx in range(1, len(timing)):
                prev = timing[idx - 1]
                cur = timing[idx]
                cur["interval"] = cur["play_start"] - (prev["play_start"] + prev["audio_dur"])
            timing[0]["interval"] = None
        return timing

    def stop(self):
        """中断：合成线程在下句停止，已入队音频播完即止。"""
        self._stop_flag.set()

    # ---------------- 非流式落盘 ----------------
    def speak_to_file(self, text, wav_path):
        """整段非流式合成并写 WAV。"""
        os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
        save_wav_np(self._synth(text), wav_path, self._sr)
        return wav_path
