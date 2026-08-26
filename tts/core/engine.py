# -*- coding: utf-8 -*-
"""离线实时中文 TTS 核心模块（v2：常驻引擎，多后端）。

架构：句子级分块 + 生产者-消费者流式（流式能力由后端提供：melo=整句一块、
cosy=原生逐块）。本模块零后端 import，后端经 tts.core.backend.get_backend 惰性加载。

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
  在后端 `synth`/`synth_stream` 边界生效——播放、save_wav、save_chunks_dir、speak_to_file 同时受益。

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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .audio import save_wav_np  # noqa: E402
from .backend import get_backend  # noqa: E402
from .jobs import Job  # noqa: E402


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device 只支持 auto/cpu/cuda，收到: %r" % device)
    return device


# ---------------------------------------------------------------------------
# 主类（单例 + 常驻引擎）
# ---------------------------------------------------------------------------
class RealtimeTTS:
    _instance = None            # 单例槽位
    _init_lock = threading.RLock()   # 可重入：__new__ 持锁调 close()，close() 内再取同锁不自锁

    # 单例：任何时刻最多一个活实例。device/backend 变更 → 销毁旧实例重建；
    # 其余配置（mode/speed/normalize）变更 → 原地切换。
    def __new__(cls, device="auto", speed=None, mode=None, normalize=None,
                backend="melo", voice=None, profile=False, debug=False,
                max_speech_ratio=None, stream=None):
        with cls._init_lock:
            inst = cls._instance
            if inst is not None:
                want = _resolve_device(device)
                want_bk = (backend or "melo").lower()
                want_voice = voice if want_bk == "cosy" else None   # voice 仅 cosy 后端有含义
                want_ratio = max_speech_ratio if want_bk == "cosy" else None
                want_stream = bool(stream) if want_bk == "cosy" else None
                if (want != inst._device or want_bk != inst._backend_name
                        or want_voice != getattr(inst, "_voice", None)
                        or want_ratio != getattr(inst, "_max_speech_ratio", None)
                        or want_stream != getattr(inst, "_stream", None)):
                    inst._shutdown_engine()   # device/backend/voice/长度上限/流式开关变更：关旧、重建
                    cls._instance = None
                else:
                    return inst
            obj = super().__new__(cls)
            cls._instance = obj
            return obj

    def __init__(self, device="auto", speed=None, mode=None, normalize=None,
                 backend="melo", voice=None, profile=False, debug=False,
                 max_speech_ratio=None, stream=None):
        # 已是常驻实例：只做运行期可变的配置（mode/speed/normalize），None 表示"不改"。
        if getattr(self, "_inited", False):
            if mode is not None:
                self.mode = mode
            if speed is not None:
                self.speed = float(speed)
            if normalize is not None:
                self.normalize = normalize
            return

        try:
            self._device = _resolve_device(device)
            self._backend_name = (backend or "melo").lower()
            self._voice = voice if self._backend_name == "cosy" else None
            # cosy 后端 LLM 生成长度有随机性（transformers 修复后已基本解决，见
            # docs/README-cosyvoice2.md §8），max_speech_ratio 把单句生成 token 上限
            # 从默认 20×text 收紧（None=模型默认），作为可选安全阀。
            self._max_speech_ratio = max_speech_ratio if self._backend_name == "cosy" else None
            # cosy 播放方式：False=整句一次合成后播放（推荐，句内无缝；本机 fp32 RTF>1，
            # 原生流式会块间饿死停顿 + 拼接缝）；True=原生 token 级流式（追求首包延迟）。
            self._stream = bool(stream) if self._backend_name == "cosy" else None
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
            self._synth_lock = threading.Lock()        # 后端模型非线程安全，串行化合成
            self._submit_lock = threading.Lock()       # submit/interrupt 临界区
            self._timing_lock = threading.Lock()
            self._job_counter = 0
            self.last_ttfa = None
            self.last_timing = []

            if self._debug:
                self._env_banner()

            t0 = time.perf_counter()
            backend_cfg = {}
            if self._backend_name == "cosy":
                backend_cfg["voice"] = self._voice
                if self._max_speech_ratio is not None:
                    backend_cfg["max_speech_ratio"] = self._max_speech_ratio
                if self._stream is not None:
                    backend_cfg["stream"] = self._stream
            self._backend = get_backend(self._backend_name, device=self._device,
                                        debug=self._debug, **backend_cfg)
            self._backend.load()
            self._sr = int(self._backend.sr)
            t_load = time.perf_counter() - t0

            # 预热：cudnn / 首次推理缓存，不计入验收
            t0 = time.perf_counter()
            with self._synth_lock:
                self._backend.synth("你好。", speed=self._speed, normalize=self._normalize)
            t_warm = time.perf_counter() - t0

            if self._debug:
                print("[debug] 后端=%s 模型加载 %.2fs（含预热 %.2fs） device=%s sr=%d mode=%s"
                      % (self._backend_name, t_load, t_warm, self._device, self._sr, self._mode))
                print("[debug] HF 权重缓存目录: %s" % os.environ.get("HF_HOME", "（未设置）"))

            self._start_workers()
            self._inited = True
        except Exception:
            # 初始化失败：回滚单例槽位，避免后续复用半成品实例（如 backend 依赖未装）
            bk = getattr(self, "_backend", None)
            if bk is not None:
                try:
                    bk.close()
                except Exception:
                    pass
            with type(self)._init_lock:
                if type(self)._instance is self:
                    type(self)._instance = None
            raise

    # ---------------- 配置属性 ----------------
    @property
    def device(self):
        return self._device

    @property
    def backend(self):
        """当前后端名（构建期定死，运行期不可换；换后端需重新 RealtimeTTS(...)）。"""
        return self._backend_name

    @property
    def voice(self):
        """cosy 音色（构建期定死）：'default'（内置参考音频）或 'clone:<wav>:<文本>'。melo 恒为 None。"""
        return self._voice

    @property
    def max_speech_ratio(self):
        """cosy 生成上限收紧（构建期定死）：None=模型默认 20×；收紧后时长可控但可能截语尾。melo 恒为 None。"""
        return self._max_speech_ratio

    @property
    def stream(self):
        """cosy 播放方式（构建期定死）：False=整句一次合成后播放（推荐）；True=原生 token 级流式。melo 恒为 None。"""
        return self._stream

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
        parts = []       # save_wav 时暂存各句 chunk（零额外推理，复用已合成数据）
        try:
            for i, sent in enumerate(job.sentences):
                if self._shutdown or job.gen != self._gen:
                    break                      # 关停或被抢占：放弃后续句子
                t_s0 = time.perf_counter() if self._profile else None
                if self._debug:
                    print("[debug] [合成 句%d/%d] %s" % (i + 1, job.n, sent[:20]))
                rec = None
                if self._profile:
                    rec = {"idx": i, "text": sent,
                           "synth_start": t_s0, "synth_end": None,
                           "play_start": None, "play_end": None,
                           "audio_dur": 0.0, "_chunk0": None}
                    with self._timing_lock:
                        job.timing.append(rec)
                sent_parts = []                # save_chunks_dir：句内各 chunk 累加后落盘
                with self._synth_lock:
                    for chunk in self._backend.synth_stream(
                            sent, speed=self._speed, normalize=self._normalize):
                        if self._shutdown or job.gen != self._gen:
                            break              # 关停/抢占：停吐后续 chunk
                        if rec is not None:
                            rec["audio_dur"] += len(chunk) / self._sr
                            if rec["_chunk0"] is None:
                                rec["_chunk0"] = time.perf_counter()   # 首块产出时刻
                        if job.save_wav_path:
                            parts.append(chunk)
                        if job.save_dir:
                            sent_parts.append(chunk)
                        # 元组带上 rec 引用：播放线程直接回填播放时间，免按 idx 查找
                        q.put((job.gen, job, i, sent, chunk, rec))
                if job.save_dir and sent_parts:
                    save_wav_np(np.concatenate(sent_parts),
                                os.path.join(job.save_dir, "句%02d.wav" % (i + 1)), self._sr)
                if rec is not None:
                    rec["synth_end"] = time.perf_counter()   # 末块产出时刻
            else:
                aborted = False
        finally:
            # save_wav：把已合成的 chunk 拼成一个整段 WAV（被打断则存半截已合成的部分）
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
        last_played_idx = None   # debug 打印按句去重（流式下一句多个 chunk）
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
                    last_played_idx = None
                    continue
                idx, sent, audio, rec = item[2], item[3], item[4], item[5]
                if gen != self._gen:
                    continue                  # 被抢占的旧块：直接丢弃
                t_p0 = time.perf_counter() if self._profile else None
                if self._debug and idx != last_played_idx:
                    print("[debug] [正在播放 句%d/%d] %s" % (idx + 1, job.n, sent[:20]))
                    last_played_idx = idx
                self._play_audio(stream, audio, gen)
                t_p1 = time.perf_counter() if self._profile else None
                if self._profile and rec is not None:
                    if rec["play_start"] is None:
                        rec["play_start"] = t_p0   # 首块填播放起点（ttfa/wait 口径）
                    rec["play_end"] = t_p1         # 末块填播放终点（逐块刷新取最后一次）
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
                c0 = rec.get("_chunk0")
                rec["ttfa"] = rec["play_start"] - t_start
                rec["synth_dur"] = rec["synth_end"] - rec["synth_start"]
                # 流式下首块产出时刻早于整句合成结束，wait 取首块口径（melo 两者一致）
                rec["wait"] = rec["play_start"] - (c0 if c0 is not None else rec["synth_end"])
                rec.pop("_chunk0", None)   # 内部字段：算完即删，对外 schema 干净
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
            audio = self._backend.synth(text, speed=self._speed, normalize=self._normalize)
        save_wav_np(audio, wav_path, self._sr)
        return wav_path

    # ---------------- 合成与分句 ----------------
    def _split(self, text):
        """按中文标点与换行切句；过短片段并入下一句，避免碎块。

        先剔除零宽/不可见字符（U+200B 零宽空格、U+200C 零宽非连接符、
        U+200D 零宽连接符、U+2060 词连接符、U+FEFF BOM），否则句尾这类
        字符会被切成一个"空句"（实测 cosy 前端会给它补句号并合成 ~0.6s
        无效音频）。"""
        text = re.sub(r"[​‌‍⁠﻿]", "", text or "")
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
            bk = getattr(self, "_backend", None)
            if bk is not None:
                try:
                    bk.close()
                except Exception:
                    pass
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
