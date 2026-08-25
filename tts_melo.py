# -*- coding: utf-8 -*-
"""MeloTTS 离线实时中文 TTS 核心模块（v2：常驻引擎）。

架构：句子级分块 + 生产者-消费者流式（"模拟"流式，因 MeloTTS/VITS 无原生流式接口）。

v2 常驻引擎（相对 v1 的关键变化）：
- **单例**：任意时刻最多一个 RealtimeTTS 实例（`_instance` 类属性持有）。
  重复 `RealtimeTTS(...)` 拿到同一个实例，绝不重复加载模型 / 开声卡流。
  传入 device 变更 → 自动销毁旧实例、重建新设备；mode / speed 变更 → 原地切换零重载。
- **常驻线程**：合成线程 + 播放线程在构造时启动、`close()` 才结束，全程只创建一次。
  文本经 `submit()` 入队即返回（非阻塞）；`speak()` = submit + wait（阻塞）。
- **两种模式**（`mode`，运行期可切）：
    - `queue`   新文本排到当前文本之后，播完再说（默认）；
    - `bargein` 新文本打断当前播放与排队（`interrupt()` 等价手动触发一次）。
- **`interrupt()` / `stop()`**：打断当前正在说的 + 清空排队文本。
- **`close()`**：销毁——打断剩余任务、关停常驻线程、关闭声卡流、清空单例槽位。
  支持 `with RealtimeTTS(...) as tts:`；`__del__` 与进程退出（atexit）兜底。
- 插桩仍全走开关：profile（时序分析）/ debug（详细日志），生产关闭时热路径零计时埋点。
- **逐句响度归一化**（`normalize`，默认关闭）：MeloTTS 逐句独立合成、不做响度均衡——
  句间活动段 RMS 差实测 ~8dB，句内起伏更大（20ms 帧 p90-p10 差 17~20dB、句首/句尾明显变轻）。
  两种档位：
    - `normalize="rms"`：逐句**静态**活动段 RMS 对齐目标 -24 dBFS（治句间音量不齐）；
    - `normalize="agc"`：静态对齐 + **句内动态压缩**（20ms 帧包络逐帧增益，治句首轻/句尾轻/中间响）
      + **短停压缩**（句内深度静音缝超 0.25s 的截短到 0.25s，治 VITS 烘焙进波形的逗号死寂收音）。
  在 `_synth` 边界生效——播放、save_wav、save_chunks_dir、speak_to_file 同时受益。

典型用法（实时场景）：
    tts = RealtimeTTS(mode="bargein")   # 常驻引擎，只创建一次
    tts.submit("第一句。")                # 非阻塞：入队立即返回
    tts.submit("更重要的话。")            # bargein 模式自动打断上一句
    tts.interrupt()                      # 或手动打断
    tts.close()                          # 用完了手动销毁
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


class Job:
    """一次文本任务（submit 的返回值）。wait() 阻塞到该任务播完（或被打断取消）。"""

    __slots__ = ("job_id", "gen", "sentences", "n", "save_dir", "save_wav_path",
                 "t_start", "timing", "_event", "canceled")

    def __init__(self, job_id, gen, sentences, save_dir, save_wav_path, t_start):
        self.job_id = job_id
        self.gen = gen
        self.sentences = sentences
        self.n = len(sentences)
        self.save_dir = save_dir
        self.save_wav_path = save_wav_path
        self.t_start = t_start
        self.timing = []           # 逐句记录（profile 开时才有内容）
        self.canceled = False
        self._event = threading.Event()

    def wait(self):
        """阻塞到本任务播完或被取消，返回逐句时序记录（与 speak() 一致）。"""
        self._event.wait()
        return self.timing

    def mark_done(self, canceled=False):
        self.canceled = canceled
        self._event.set()


# ---------------------------------------------------------------------------
# 主类（单例 + 常驻引擎）
# ---------------------------------------------------------------------------
class RealtimeTTS:
    _instance = None            # 单例槽位
    _init_lock = threading.Lock()
    # 逐句响度归一化（normalize="rms"）常量：
    # 目标 = 活动段(非静音) RMS -24 dBFS（≈0.0631）；峰值上限 0.95，超则整体下压防爆音。
    _NORMALIZE_TARGET = 10 ** (-24.0 / 20.0)
    _NORMALIZE_PEAK_CEIL = 0.95

    # 单例：任何时刻最多一个活实例。device 变更 → 销毁旧实例重建；
    # 其余配置（mode/speed）变更 → 原地切换。
    def __new__(cls, device="auto", speed=None, mode=None, normalize=None, profile=False, debug=False):
        with cls._init_lock:
            inst = cls._instance
            if inst is not None:
                want = _resolve_device(device)
                if want != inst._device:
                    inst._shutdown_engine()   # device 变更：关旧、重建
                    cls._instance = None
                else:
                    return inst
            obj = super().__new__(cls)
            cls._instance = obj
            return obj

    def __init__(self, device="auto", speed=None, mode=None, normalize=None, profile=False, debug=False):
        # 已是常驻实例：只做运行期可变的配置（mode/speed/normalize），None 表示"不改"。
        if getattr(self, "_inited", False):
            if mode is not None:
                self.mode = mode
            if speed is not None:
                self.speed = float(speed)
            if normalize is not None:
                self.normalize = normalize
            return

        self._device = _resolve_device(device)
        self._speed = 1.0 if speed is None else float(speed)
        self.mode = "queue" if mode is None else mode
        self.normalize = normalize          # None=不处理；"rms"=逐句 RMS 响度均衡
        self._profile = profile
        self._debug = debug
        self._closed = False
        self._shutdown = False
        self._gen = 0
        self._jobs = queue.Queue()                 # 文本级任务队列（无界，可积压）
        self._audio_q = queue.Queue(maxsize=8)     # 音频块队列（有界，天然背压）
        self._synth_lock = threading.Lock()        # 模型非线程安全，串行化 _synth
        self._submit_lock = threading.Lock()       # submit/interrupt 临界区
        self._timing_lock = threading.Lock()
        self._job_counter = 0
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
        with self._synth_lock:
            self._synth("你好。")
        t_warm = time.perf_counter() - t0

        if self._debug:
            print("[debug] 模型加载 %.2fs（含预热 %.2fs） device=%s sr=%d mode=%s"
                  % (t_load, t_warm, self._device, self._sr, self._mode))
            print("[debug] HF 权重缓存目录: %s" % os.environ["HF_HOME"])

        self._start_workers()
        self._inited = True

    # ---------------- 配置属性 ----------------
    @property
    def device(self):
        return self._device

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, v):
        self._speed = float(v)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        if v not in ("queue", "bargein"):
            raise ValueError("mode 只支持 'queue'/'bargein'，收到: %r" % v)
        self._mode = v

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, v):
        if v is not None and v not in ("rms", "agc"):
            raise ValueError("normalize 只支持 None/'rms'/'agc'，收到: %r" % v)
        self._normalize = v

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

    # ---------------- 常驻线程 ----------------
    def _start_workers(self):
        self._synth_th = threading.Thread(target=self._worker_synth, daemon=True)
        self._play_th = threading.Thread(target=self._worker_play, daemon=True)
        self._synth_th.start()
        self._play_th.start()

    def _worker_synth(self):
        while True:
            job = self._jobs.get()
            if job is None or self._shutdown:
                break
            self._run_job(job)

    def _run_job(self, job):
        q = self._audio_q
        aborted = True   # for 正常跑完（未 break）才为 False
        parts = []       # save_wav 时暂存各句 audio（零额外推理，复用已合成数据）
        try:
            for i, sent in enumerate(job.sentences):
                if self._shutdown or job.gen != self._gen:
                    break                      # 关停或被抢占：放弃后续句子
                t_s0 = time.perf_counter() if self._profile else None
                if self._debug:
                    print("[debug] [合成 句%d/%d] %s" % (i + 1, job.n, sent[:20]))
                with self._synth_lock:
                    audio = self._synth(sent)
                t_s1 = time.perf_counter() if self._profile else None
                if job.save_dir:
                    save_wav_np(audio, os.path.join(job.save_dir, "句%02d.wav" % (i + 1)), self._sr)
                if job.save_wav_path:
                    parts.append(audio)
                rec = None
                if self._profile:
                    rec = {"idx": i, "text": sent,
                           "synth_start": t_s0, "synth_end": t_s1,
                           "play_start": None, "play_end": None,
                           "audio_dur": len(audio) / self._sr}
                    with self._timing_lock:
                        job.timing.append(rec)
                # 元组带上 rec 引用：播放线程直接回填播放时间，免按 idx 查找
                q.put((job.gen, job, i, sent, audio, rec))
            else:
                aborted = False
        finally:
            # save_wav：把已合成的句子拼成一个整段 WAV（被打断则存半截已合成的部分）
            if job.save_wav_path and parts:
                save_wav_np(np.concatenate(parts), job.save_wav_path, self._sr)
            # 无论完成、被抢占还是关停，都发 DONE 让播放线程收尾该 job（wait() 不悬挂）。
            # 若中途放弃（关停/抢占），立即就地标记 canceled：播放线程可能已退出或
            # 来不及处理该 DONE，不就地标记则其 wait() 会永久悬挂。
            q.put((job.gen, job, "DONE"))
            if aborted:
                job.mark_done(canceled=True)

    def _worker_play(self):
        import sounddevice as sd
        # blocksize=1024（~23ms/块）让打断时残响更短；个别设备不支持时退回默认
        try:
            stream = sd.OutputStream(samplerate=self._sr, channels=1,
                                     dtype="float32", blocksize=1024)
        except Exception:
            stream = sd.OutputStream(samplerate=self._sr, channels=1, dtype="float32")
        stream.start()
        try:
            while True:
                item = self._audio_q.get()
                if item is None or self._shutdown:
                    break
                gen, job = item[0], item[1]
                if len(item) == 3 and item[2] == "DONE":
                    canceled = (gen != self._gen)
                    self._finalize_job(job, canceled)
                    job.mark_done(canceled)
                    continue
                idx, sent, audio, rec = item[2], item[3], item[4], item[5]
                if gen != self._gen:
                    continue                  # 被抢占的旧块：直接丢弃
                t_p0 = time.perf_counter() if self._profile else None
                if self._debug:
                    print("[debug] [正在播放 句%d/%d] %s" % (idx + 1, job.n, sent[:20]))
                self._play_audio(stream, audio, gen)
                t_p1 = time.perf_counter() if self._profile else None
                if self._profile and rec is not None:
                    rec["play_start"] = t_p0
                    rec["play_end"] = t_p1
        finally:
            # 关停退出前清空队列里残留的块/DONE 并就地标记其 Job，避免对应 wait() 悬挂
            # （正常完成但 DONE 尚未被本线程处理的 job，也在这里兜底标记 canceled）
            while True:
                try:
                    item = self._audio_q.get_nowait()
                except queue.Empty:
                    break
                if len(item) >= 2 and isinstance(item[1], Job):
                    item[1].mark_done(canceled=True)
            stream.stop()
            stream.close()

    def _play_audio(self, stream, audio, gen):
        """按 ~50ms 小块写声卡，块间检查代际标记——抢占时能快速切走，残响 ~50-100ms。"""
        sr = self._sr
        block = max(int(sr * 0.05), 1)
        a = audio.astype(np.float32)
        for off in range(0, len(a), block):
            if self._shutdown or gen != self._gen:
                break
            stream.write(a[off:off + block])

    def _finalize_job(self, job, canceled=False):
        """由播放线程在收到 DONE 时调用：补齐 ttfa/synth_dur/wait/interval 派生字段。"""
        timing = job.timing
        if self._profile and timing and not canceled:
            t_start = job.t_start
            for rec in timing:
                rec["ttfa"] = rec["play_start"] - t_start
                rec["synth_dur"] = rec["synth_end"] - rec["synth_start"]
                rec["wait"] = rec["play_start"] - rec["synth_end"]
            for idx in range(1, len(timing)):
                prev, cur = timing[idx - 1], timing[idx]
                cur["interval"] = cur["play_start"] - (prev["play_start"] + prev["audio_dur"])
            timing[0]["interval"] = None
            self.last_timing = timing
            self.last_ttfa = timing[0]["play_start"] - t_start
        job.timing = timing

    # ---------------- 对外接口 ----------------
    def submit(self, text, save_chunks_dir=None, save_wav=None):
        """非阻塞入队一段文本，立即返回 Job（实时场景用这个）。

        mode="bargein" 时，新文本会打断当前正在说的与排队的旧文本。
        save_chunks_dir: 播放的同时把每句音频各存一份 WAV（零额外推理）。
        save_wav: 播放的同时把全部句子拼成一个整段 WAV 落盘（零额外推理，
                  复用流式合成已算好的逐句 audio）。与 save_chunks_dir 相互独立。
        """
        self._check_alive()
        with self._submit_lock:
            if self._mode == "bargein":
                self._do_interrupt()
            gen = self._gen
            sentences = self._split(text)
            t_start = time.perf_counter() if self._profile else None
            if save_chunks_dir:
                os.makedirs(save_chunks_dir, exist_ok=True)
            if save_wav:
                os.makedirs(os.path.dirname(os.path.abspath(save_wav)), exist_ok=True)
            self._job_counter += 1
            job = Job(self._job_counter, gen, sentences, save_chunks_dir,
                      save_wav, t_start)
            if not sentences:
                job.mark_done()               # 空文本：立即完成，wait() 直接返回
                return job
            self._jobs.put(job)
        return job

    def speak(self, text, save_chunks_dir=None, save_wav=None):
        """阻塞播完一段文本并返回逐句时序记录（= submit + wait，bench 兼容）。"""
        return self.submit(text, save_chunks_dir, save_wav).wait()

    def interrupt(self):
        """手动打断：停止当前正在说的 + 清空排队的旧文本（两种模式都可用）。"""
        self._check_alive()
        with self._submit_lock:
            self._do_interrupt()

    def stop(self):
        """与 interrupt() 同义（保留旧接口名）。"""
        self.interrupt()

    def _do_interrupt(self):
        self._gen += 1                        # 代际标记 +1：旧任务全部作废
        while True:
            try:
                self._jobs.get_nowait().mark_done(canceled=True)
            except queue.Empty:
                break
        # 清空音频队列：其元素（块/DONE）都属于被抢占的任务，但 DONE 可能已被丢弃，
        # 必须就地标记这些 job 为 canceled，否则它们的 wait() 会永远悬挂。
        while True:
            try:
                item = self._audio_q.get_nowait()
            except queue.Empty:
                break
            job = item[1]
            if isinstance(job, Job):
                job.mark_done(canceled=True)

    def speak_to_file(self, text, wav_path):
        """整段非流式合成并写 WAV（不占常驻播放线程，加锁与流式合成串行）。"""
        os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
        with self._synth_lock:
            audio = self._synth(text)
        save_wav_np(audio, wav_path, self._sr)
        return wav_path

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
        """整句合成，返回 44.1kHz float32 numpy 数组（[-1,1]）。调用方须持有 _synth_lock。

        归一化开关在此生效：播放、save_wav、save_chunks_dir、speak_to_file 全部走这里。
        """
        a = self._model.tts_to_file(
            text, self._spk, output_path=None, speed=self._speed, quiet=True)
        if self._normalize:
            a = self._normalize_audio(a)
        return a

    def _normalize_audio(self, audio):
        """归一化分发：normalize="rms"=逐句静态响度对齐（治句间）；"agc"=静态对齐 +
        句内动态压缩（治句内起伏）+ 短停压缩（治句内超长静音缝）。"""
        a = self._static_align(audio)
        if self._normalize == "agc":
            a = self._pause_cap(a)
            a = self._agc(a)
        return a

    def _static_align(self, audio):
        """逐句静态响度对齐：按"活动段(非静音) RMS"缩放整句，对齐目标响度。

        不用整句 RMS 的原因：VITS 句首句尾常带静音，静音占比句间差异大（实测
        37%~64%），会稀释整句 RMS——静音多的句子被过度放大、少的被压小，语音段
        实际响度仍不齐。只按有语音的帧算 RMS 对齐，才接近人耳感知的语音响度。
        峰值超上限则整体下压防爆音；全静音句跳过。
        """
        a = np.asarray(audio, dtype=np.float32)
        rms_act = self._active_rms(a)
        if rms_act <= 1e-6:
            return audio                        # 无活动帧：跳过，避免除零/放大噪声
        out = a * (self._NORMALIZE_TARGET / rms_act)
        peak = float(np.abs(out).max())
        if peak > self._NORMALIZE_PEAK_CEIL:
            out *= self._NORMALIZE_PEAK_CEIL / peak
        return out

    def _pause_cap(self, audio, max_pause=0.25, rel_thr=1e-3):
        """短停压缩（并进 agc）：把句内超过 max_pause 的"深度静音缝"截短到 max_pause。

        背景：VITS 把标点停顿直接烘焙进单个波形，实测逗号处可生成 ~0.6s 的绝对数字
        静音（-inf dB，远超自然短停顿 ~0.1s），听感是"每逗号之间突然消音没了声音"。
        本方法检测帧 RMS 低于 rel_thr×peak（≈-60dB 相对峰值，只认真静音、不碰正常
        呼吸/短停）的连续段，超过上限的从中间抽掉多余静音（两侧各留 max_pause/2 帧，
        保留原渐入渐出边），静音接静音无咔哒；句首/句尾静音不动，保持句子自然边界。
        """
        a = np.asarray(audio, dtype=np.float32)
        win = max(int(self._sr * 0.02), 1)
        n = len(a) // win
        if n == 0:
            return audio
        frames = a[:n * win].reshape(n, win)
        rms_f = np.sqrt(np.mean(frames ** 2, axis=1))
        peak = float(np.abs(a).max())
        if peak <= 1e-12:
            return audio
        deep = rms_f < rel_thr * peak                # 深度静音帧（真静音，非呼吸）
        max_frames = max(int(round(max_pause / 0.02)), 1)

        # 收集连续深度静音段
        runs = []
        start = None
        for i, d in enumerate(deep):
            if d and start is None:
                start = i
            elif not d and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, n - 1))
        # 排除句首/句尾静音段（保持句子自然边界，不压缩）
        runs = [r for r in runs if r[0] != 0 and r[1] != n - 1]

        # 超长段：保留两侧 max_pause/2 帧，中间多余静音整段切除
        cuts = []
        for s0, s1 in runs:
            L = s1 - s0 + 1
            if L <= max_frames:
                continue
            head = max_frames // 2
            tail = max_frames - head
            cuts.append(((s0 + head) * win, (s1 - tail + 1) * win))
        if not cuts:
            return audio
        out = a
        for cs, ce in sorted(cuts, reverse=True):    # 从后往前切，避免索引位移
            out = np.concatenate([out[:cs], out[ce:]])
        return out

    def _agc(self, audio):
        """句内动态压缩（AGC）：按 20ms 帧 RMS 包络做逐帧增益，把句内幅度起伏压平到
        目标响度附近。帧间增益线性插值避免抽吸；噪声门防静音/呼吸被过度放大；
        增益限幅（最多压 10dB / 抬 16dB）防过度压缩；峰值超上限整体下压。
        """
        a = np.asarray(audio, dtype=np.float32)
        win = max(int(self._sr * 0.02), 1)
        n = len(a) // win
        if n == 0:
            return audio
        frames = a[:n * win].reshape(n, win)
        rms_f = np.sqrt(np.mean(frames ** 2, axis=1))
        gate = max(float(rms_f.max()) * 1e-3, 1e-6)   # 噪声门限（相对本句峰值 RMS）
        env = np.maximum(rms_f, gate)
        gain_db = 20.0 * np.log10(self._NORMALIZE_TARGET / env)
        gain_db = np.clip(gain_db, -10.0, 16.0)   # 最多压 10dB / 抬 16dB
        gain = 10.0 ** (gain_db / 20.0)
        centers = win // 2 + np.arange(n) * win
        out = a[:n * win] * np.interp(np.arange(n * win), centers, gain)
        if n * win < len(a):
            out = np.concatenate([out, a[n * win:] * float(gain[-1])])
        peak = float(np.abs(out).max())
        if peak > self._NORMALIZE_PEAK_CEIL:
            out *= self._NORMALIZE_PEAK_CEIL / peak
        return out

    def _active_rms(self, audio, win_s=0.02, rel_thr=0.02):
        """按 20ms 帧统计非静音帧的 RMS（能量均值），静音帧不参与。

        关键：活动阈值取**相对句子峰值**的比例（thr = rel_thr × peak），
        缩放前后峰值等比变化 → 活动帧集合对增益不变，归一化后活动 RMS 精确等于目标
        （若用固定阈值，放大/缩小后贴边帧会跨入/跨出活动集，测得值偏离目标）。
        """
        a = np.asarray(audio, dtype=np.float32)
        win = max(int(self._sr * win_s), 1)
        n = len(a) // win
        if n == 0:
            return 0.0
        frames = a[:n * win].reshape(n, win)
        rms_f = np.sqrt(np.mean(frames ** 2, axis=1))
        peak = float(np.abs(a).max())
        act = rms_f[rms_f > rel_thr * peak]
        if len(act) == 0:
            return 0.0
        return float(np.sqrt(np.mean(act ** 2)))

    # ---------------- 生命周期 ----------------
    def _shutdown_engine(self):
        self.close()

    def close(self):
        """销毁：打断剩余任务、关停常驻线程、关闭声卡流、清空单例槽位。幂等。"""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            with self._submit_lock:
                self._do_interrupt()
            self._shutdown = True
            try:
                self._jobs.put(None)
            except Exception:
                pass
            try:
                self._audio_q.put(None)
            except Exception:
                pass
            for th in (getattr(self, "_synth_th", None), getattr(self, "_play_th", None)):
                if th is not None:
                    th.join(timeout=10)
        finally:
            with type(self)._init_lock:
                if type(self)._instance is self:
                    type(self)._instance = None

    def _check_alive(self):
        if getattr(self, "_closed", False):
            raise RuntimeError("RealtimeTTS 已 close()，无法再使用（请重新创建）")

    # ---------------- 上下文管理器 / 兜底 ----------------
    def __enter__(self):
        self._check_alive()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _atexit_close():
    inst = RealtimeTTS._instance
    if inst is not None:
        try:
            inst.close()
        except Exception:
            pass


import atexit  # noqa: E402
atexit.register(_atexit_close)
