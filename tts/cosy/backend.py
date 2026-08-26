# -*- coding: utf-8 -*-
"""CosyVoice2 后端（原生流式 + 3s 零样本音色克隆）。

选择性安装的另一半：依赖本仓库 third_party/CosyVoice + Matcha-TTS 子模块，
安装步骤见 docs/README-cosyvoice2.md。权重经 HF 官方仓库
FunAudioLLM/CosyVoice2-0.5B 落到 .cache/hf（snapshot_download 复用缓存）。

定位：melotts 只能中文女声；cosy 提供"默认女声 + 任意 3s 参考音频克隆"，
但句首等待实测 ~5.6-6.2s（=整句合成耗时，约 1.2×音频时长），达不到 melo 的
<1s 硬指标——实时主力仍是 melo，cosy 是"音质 + 克隆"可选项。

播放方式（默认非流式，2026-08-26 改）：本机 2070S fp32 下 RTF≈1.1-1.3
（合成比实时播放还慢），cosy 原生流式（token 级逐块 yield）播出来必然
"字间戛然而止 + 块拼接缝"——播放饿死等下一块、块边界是独立 flow+hifigan
解码的声学接缝。因此默认 `stream=False`：每句一次 LLM→flow→hifigan 解码、
单块播放，句内无缝；代价是句首等待 = 整句合成耗时。追求首包延迟时可
`stream=True` 显式恢复原生流式（本机仍会卡顿，不推荐）。

已知限制（0.5B LLM 生成长度，实测详见 docs/README-cosyvoice2.md §8）：
接入初期测得的"EOS 不可靠/短句必冲上限"其实是坏 transformers（4.57.6）
的产物，pin 4.51.3 后生成有界、时长与文本长度成比例（约 5~7 token/字）。
保留正常的采样随机性（±1.5×）。`max_speech_ratio` 降级为可选安全阀。
"""
import hashlib
import os
import sys

# 本文件位于 tts/cosy/backend.py → 项目根向上取 3 级
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- transformers pin：须最先、在任何 import transformers 之前生效 ----
# CosyVoice2 官方 issue #1546：transformers 高于 4.51.3 就出问题（4.53+ 重写了
# Qwen2Model.forward 的 attention-mask / hidden-state 输出逻辑，Qwen2LM 调用路径
# 不再兼容），表现就是 LLM 产出的 speech token 全错 → "一句输入，多次输出、全是
# 杂音、没完没了"。仓库 requirements.txt 也 pin transformers==4.51.3。
# 对策：自带一份 vendored transformers 4.51.3 + tokenizers 0.21.1
# （.cache/pinned_transformers，94M，setup_cosy_pinned.py 落盘，离线可用），
# 在本模块 import 时注入 sys.path 替换 main env 高版本；melo 等仍用 main env 版本。
_PINNED_TRANSFORMERS = os.path.join(_PROJECT_DIR, ".cache", "pinned_transformers")


def _pin_transformers():
    """把 vendored transformers 4.51.3 + tokenizers 0.21.1 注入 sys.path。

    必须在进程里第一次 `import transformers` 之前调用（本模块顶层即调用）。
    若进程已 import 过高版本（cosy 与 melo 混用），这里明确报错而不是默默产杂音。"""
    if os.path.isdir(_PINNED_TRANSFORMERS) and _PINNED_TRANSFORMERS not in sys.path:
        sys.path.insert(0, _PINNED_TRANSFORMERS)
    try:
        import transformers as _t
    except ImportError as e:
        raise RuntimeError(
            "CosyVoice2 需要 transformers==4.51.3（>=4.52 产出杂音，官方 issue #1546）。"
            "当前进程无可用 transformers（%s）。请先运行 setup_cosy_pinned.py，"
            "或按 docs/README-cosyvoice2.md 安装。" % e) from e
    if _t.__version__ != "4.51.3":
        raise RuntimeError(
            "CosyVoice2 需要 transformers==4.51.3，当前进程加载到 %s——cosy 与 melo "
            "不能在同一进程混用（后者会先 import 高版本 transformers）。请用独立进程跑 cosy。"
            % _t.__version__)


_pin_transformers()

import numpy as np  # noqa: E402

from ..core.backend import BackendNotInstalledError, TTSBackend  # noqa: E402

# 权重缓存/镜像：与 tts/melo/backend.py 同套 env 前置
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# CosyVoice 仓库与其 Matcha-TTS 子模块（PYTHONPATH 注入，须在 import cosyvoice 之前）
_THIRD_PARTY = os.path.join(_PROJECT_DIR, "third_party", "CosyVoice")
_MATCHA = os.path.join(_THIRD_PARTY, "third_party", "Matcha-TTS")

# 内置默认音色：官方 demo 女声参考音频 + 对应转写文本（零样本克隆的 prompt）
_DEFAULT_PROMPT_WAV = os.path.join(_PROJECT_DIR, "assets", "cosy_default_female.wav")
_DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"


def _force_cpu_onnx():
    """campplus/speech_tokenizer 是小模型，CPU 推理足够快（ms 级），且 CPU 版
    onnxruntime 无法提供 CUDAExecutionProvider——而 frontend 会按
    torch.cuda.is_available() 请求它。这里把 onnxruntime.InferenceSession 的
    provider 强制回 CPU，绕开可用性问题，也省去 GPU 会话初始化延迟。
    本进程内只有 cosyvoice 用 onnxruntime，全局替换安全。"""
    import onnxruntime as _ort
    _orig = _ort.InferenceSession

    def _cpu_session(*a, **k):
        k["providers"] = ["CPUExecutionProvider"]
        return _orig(*a, **k)

    _ort.InferenceSession = _cpu_session


def _ensure_wetext_local():
    """wetext.Normalizer 每次构造都会调 modelscope snapshot_download（联网查
    metadata，撞限流 403 会整个降级为"无文本前端"，数字/单位不做中文归一化）。
    若本地已有 FST 缓存，直接把它当 repo_dir 注入，零联网。首次无缓存则维持原逻辑下载。"""
    import wetext.wetext as _w
    local = None
    root = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "models")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not name.startswith("pengzhendong--wetext"):
                continue
            snap = os.path.join(root, name, "snapshots")
            if not os.path.isdir(snap):
                continue
            for rev in sorted(os.listdir(snap)):
                d = os.path.join(snap, rev)
                if os.path.isfile(os.path.join(d, "zh", "tn", "tagger.fst")):
                    local = d
                    break
            if local:
                break
    if not local:
        return   # 首次：无缓存，保持 snapshot_download 联网下载
    _orig = _w.snapshot_download
    def _local_snap(repo_id=None, *a, **k):
        if repo_id == "pengzhendong/wetext":
            return local
        return _orig(repo_id, *a, **k)
    _w.snapshot_download = _local_snap
    if os.environ.get("COSY_DEBUG_WETEXT"):
        print("[debug] wetext 使用本地缓存: %s" % local)


def _silence_tqdm():
    """仓库在推理链里硬编码 tqdm(...)（流式会刷进度条），替换为透传。"""
    try:
        from cosyvoice.cli import cosyvoice as _cc
        _cc.tqdm = lambda *a, **k: (a[0] if a else k.get("iterable", []))
    except Exception:
        pass


def _wrap_max_ratio(llm, ratio):
    """把 llm.inference 的 max_token_text_ratio 从默认 20 收紧到 ratio。

    CosyVoice2-0.5B 的 LLM EOS 不可靠（概率常 <2%、rank 常在 top-25 外），
    短句几乎必冲到 20×text token 上限。收紧上限是唯一确定性的时长闸门；
    代价是"模型本要更晚结束"的样本会被截尾。None 时不动（保留模型行为）。
    """
    if not ratio:
        return
    _orig = llm.inference

    def _capped(**k):
        k["max_token_text_ratio"] = float(ratio)
        return _orig(**k)

    llm.inference = _capped


class CosyBackend(TTSBackend):
    name = "cosy"
    sr = 24000

    def __init__(self, device="auto", voice="default", text_frontend=True, debug=False,
                 max_speech_ratio=None, stream=False):
        self._device = device
        self._voice = voice
        self._text_frontend = bool(text_frontend)
        self._debug = debug
        self._max_speech_ratio = max_speech_ratio
        self._stream = bool(stream)   # False=整句一次合成（推荐，句内无缝）；True=原生 token 流式
        self._model = None
        self._prompt_wav = None
        self._prompt_text = None
        self._spk_id = ""            # 非空 → 走缓存 speaker（默认音色）；空 → 每句重新提 prompt

    def load(self):
        # PYTHONPATH 注入须在 import cosyvoice 之前
        for p in (_THIRD_PARTY, _MATCHA):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(REPO_ID)
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
        except ImportError as e:
            raise BackendNotInstalledError(
                "CosyVoice 依赖缺失：请按 docs/README-cosyvoice2.md 安装后重试（%s）" % e) from e
        _force_cpu_onnx()
        _silence_tqdm()
        _ensure_wetext_local()   # 须在 CosyVoice2()（构造 frontend→wetext）之前
        # load_jit/load_trt/load_vllm/fp16 全关：纯 torch fp32 推理，RTX 2070S 8G 可跑
        self._model = CosyVoice2(model_dir, load_jit=False, load_trt=False,
                                 load_vllm=False, fp16=False)
        self.sr = int(self._model.sample_rate)
        # EOS 不可靠 → 收紧生成上限（须在合成前 wrap llm.inference）
        _wrap_max_ratio(self._model.model.llm, self._max_speech_ratio)
        self._setup_voice()

    # ---------------- 音色 ----------------
    def _setup_voice(self):
        # voice=None（引擎默认）视同 "default"；后端 __init__ 默认值即 "default"
        v = self._voice or "default"
        if v == "default":
            if not os.path.exists(_DEFAULT_PROMPT_WAV):
                raise RuntimeError("内置参考音频缺失: %s（git 拉全或重新 clone 后重试）"
                                   % _DEFAULT_PROMPT_WAV)
            # 缓存 speaker embedding：默认音色每句免重提 prompt
            self._spk_id = "default_female" if self._model.add_zero_shot_spk(
                _DEFAULT_PROMPT_TEXT, _DEFAULT_PROMPT_WAV, "default_female") else ""
            self._prompt_text, self._prompt_wav = _DEFAULT_PROMPT_TEXT, _DEFAULT_PROMPT_WAV
        elif v.startswith("clone:"):
            # 注意 Windows 盘符冒号：先只切第一个冒号（"clone:" 前缀），再对剩余做 rpartition 分 wav/文本。
            _, rest = v.split(":", 1)
            wav, _, text = rest.rpartition(":")
            if not os.path.exists(wav) or not text.strip():
                raise ValueError('voice="clone:<参考wav>:<转写文本>"，参考音频须存在且文本非空，收到: %r' % v)
            # clone 也走 add_zero_shot_spk 缓存（同 default）：参考音频的
            # speech_feat/speech_token/说话人嵌入只编码一次，之后每句复用——
            # 实测 clone 每句比 default 多 ~0.5s 正是这段重复编码（RTF 高 ~0.1-0.15）。
            # 转写文本先 normalize 再缓存，与旧实时路径的 prompt token 保持一致。
            norm = self._model.frontend.text_normalize(
                text.strip(), split=False, text_frontend=self._text_frontend)
            self._spk_id = "clone_" + hashlib.md5(
                (wav + "\0" + norm).encode("utf-8")).hexdigest()[:12]
            self._model.add_zero_shot_spk(norm, wav, self._spk_id)
            self._prompt_wav, self._prompt_text = wav, text.strip()
        else:
            raise ValueError('voice 只支持 "default" 或 "clone:<wav>:<文本>"，收到: %r' % v)
        if self._debug:
            print("[debug] cosy voice=%s spk_id=%r prompt=%s"
                  % (self._voice, self._spk_id, self._prompt_text))

    # ---------------- 合成 ----------------
    def _gen(self, text, *, speed=1.0):
        m = self._model
        if not self._spk_id:
            raise RuntimeError("cosy 音色未初始化：_setup_voice 未给出 spk_id")
        # default 与 clone 统一走缓存 speaker 路径：参考音频编码已在 load 时做一次，
        # 之后每句只 prefill 目标文本 + 解码，不再重复编码参考音频。
        gen = m.inference_zero_shot(text, "", "",
                                    zero_shot_spk_id=self._spk_id, stream=self._stream,
                                    speed=speed, text_frontend=self._text_frontend)
        for j in gen:
            yield j["tts_speech"].detach().cpu().numpy().reshape(-1).astype(np.float32)

    def synth_stream(self, text, *, speed=1.0, normalize=None):
        """默认整句一次合成、单块产出（stream=False）→ 无块拼接缝、无饿死停顿，
        normalize 在整句聚合后应用（与 melo 同语义）。见类 docstring。
        stream=True 时退化为原生 token 级流式（块级 normalize，本机 RTF>1 会卡顿）。"""
        if not self._stream:
            a = self.synth(text, speed=speed, normalize=normalize)
            if a.size:
                yield a
            return
        from ..core.audio import normalize_audio
        for chunk in self._gen(text, speed=speed):
            if normalize:
                chunk = normalize_audio(chunk, self.sr, normalize)
            yield chunk

    def synth(self, text, *, speed=1.0, normalize=None):
        """整句一次合成（默认非流式 = 单次解码；stream=True 时聚合流式块）。
        normalize 在整句聚合后应用（同 melo 语义）。"""
        from ..core.audio import normalize_audio
        parts = list(self._gen(text, speed=speed))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        a = np.concatenate(parts)
        if normalize:
            a = normalize_audio(a, self.sr, normalize)
        return a

    def close(self):
        self._model = None   # 释放引用，torch 显存由 GC 回收
        self._spk_id = ""
