# -*- coding: utf-8 -*-
"""VITS 多音色后端（参照 Alife 项目的 multi-speaker VITS）。

模型：GitHub release `VITS.zip`（约 412MB，含推理代码 models.py/text/commons/utils
+ 权重 G_953000.pth + config.json + speakers_list.txt 音色表），经 preload_vits.py
下载解压到 `.cache/vits/VITS/`，运行期零网络。本后端无 HF/modelscope 依赖，
故不设 HF 环境变量（与 melo/cosy 不同）。

多音色：非自回归 multi-speaker VITS，换音色 = 换 `sid`（同一模型即时切换，
零额外推理成本）。voice 参数：None→默认 551（真央）；int→speaker_id；
str→经 speakers_list.txt 反查名字（找不到报错）。

依赖：torch + numpy + jieba/pypinyin/cn2an 等文本前端包（voice-asr/voice-tts
环境已具备，缺什么装什么）。bundled models.py 仅 `remove_weight_norm` 弃用警告
（torch 2.x 兼容），无其他重依赖。
"""
import os
import sys

import numpy as np  # noqa: E402

# 本文件位于 tts/vits/backend.py → 项目根向上取 3 级
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VITS_DIR = os.path.join(_PROJECT_DIR, ".cache", "vits", "VITS")
_MODEL_PTH = os.path.join(_VITS_DIR, "model", "G_953000.pth")
_CONFIG_JSON = os.path.join(_VITS_DIR, "model", "config.json")
_SPEAKERS_LIST = os.path.join(_VITS_DIR, "speakers_list.txt")

# Alife 默认说话人（VitsSpeechModelConfig.cs SpeakerId=551；speakers_list.txt 实测 = 派蒙）
_DEFAULT_SPEAKER = 551

# 与 Alife 内联代码一致的推理参数（VitsSpeechModelConfig.cs）
_DEFAULT_NOISE_SCALE = 0.6
_DEFAULT_NOISE_SCALE_W = 0.668
_DEFAULT_LENGTH_SCALE = 1.2

from ..core.audio import normalize_audio  # noqa: E402
from ..core.backend import BackendNotInstalledError, TTSBackend  # noqa: E402


class VitsBackend(TTSBackend):
    name = "vits"
    sr = 22050                       # load() 时从 config.json 的 sampling_rate 覆盖

    def __init__(self, device="auto", voice=None, noise_scale=_DEFAULT_NOISE_SCALE,
                 noise_scale_w=_DEFAULT_NOISE_SCALE_W, length_scale=_DEFAULT_LENGTH_SCALE,
                 debug=False):
        self._device = device
        self._voice = voice          # None / int(speaker_id) / str(名字)，load() 里解析
        self._noise_scale = float(noise_scale)
        self._noise_scale_w = float(noise_scale_w)
        self._length_scale = float(length_scale)
        self._debug = debug
        self._model = None           # net_g
        self._hps = None
        self._torch_device = "cpu"
        self._sid = None             # 解析后的 speaker_id
        self._speakers = []          # [(id, name)]，供报错与调试展示

    def load(self):
        if not os.path.isfile(_MODEL_PTH):
            raise BackendNotInstalledError(
                "VITS 模型缺失：%s\n请先运行 `python preload_vits.py` 下载解压到 .cache/vits/。"
                % _MODEL_PTH)
        import logging
        import torch

        # VITS 自带 utils.py 首行 basicConfig(level=DEBUG) 会把全局日志级别拉成 DEBUG，
        # 导致 numba/jieba 的 DEBUG 日志刷屏——在 import utils 前先把 root 压回 WARNING
        # （basicConfig 不覆盖已设的 logger level），再单独压 jieba（它内部自设 DEBUG）。
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("jieba").setLevel(logging.WARNING)

        # 推理代码随模型发布（非 pip 包），把模型目录挂到 sys.path 后按顶层模块导入。
        # 与 Alife 内联 pythonCode 的 init() 一致；本后端进程内常驻，insert 即可。
        if _VITS_DIR not in sys.path:
            sys.path.insert(0, _VITS_DIR)
        try:
            from models import SynthesizerTrn       # 依赖 commons/utils（models.py 内引用）
            from text import text_to_sequence
            import commons
            import utils
        except ImportError as e:
            raise BackendNotInstalledError(
                "VITS 推理代码导入失败（%s）。请重跑 `python preload_vits.py`，"
                "或检查 .cache/vits/VITS/ 完整性。" % e) from e

        self._text_to_sequence = text_to_sequence
        self._commons = commons

        self._torch_device = "cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu"
        hps = utils.get_hparams_from_file(_CONFIG_JSON)
        self._hps = hps
        self.sr = int(hps.data.sampling_rate)

        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        self._model = net_g.eval().to(self._torch_device)
        utils.load_checkpoint(_MODEL_PTH, net_g, None)

        self._speakers = self._load_speakers()
        self._sid = self._resolve_speaker(hps.data.n_speakers)
        if self._debug:
            print("[debug] vits: device=%s sr=%d n_speakers=%d add_blank=%s text_cleaners=%s"
                  % (self._torch_device, self.sr, hps.data.n_speakers, hps.data.add_blank,
                     hps.data.text_cleaners))
            print("[debug] vits: voice=%r -> speaker_id=%d（音色共 %d 个）"
                  % (self._voice, self._sid, len(self._speakers)))

    # ---------------- 音色表 ----------------
    def _load_speakers(self):
        """speakers_list.txt：每行 `id:名字`。返回 [(id, name)]，文件缺失返回 []。"""
        if not os.path.isfile(_SPEAKERS_LIST):
            return []
        out = []
        with open(_SPEAKERS_LIST, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    out.append((int(parts[0].strip()), parts[1].strip()))
        return out

    def _resolve_speaker(self, n_speakers):
        v = self._voice
        if v is None:
            # 默认 551（真央）；表中没有就退回第一个可用 id
            if any(sid == _DEFAULT_SPEAKER for sid, _ in self._speakers):
                return _DEFAULT_SPEAKER
            return self._speakers[0][0] if self._speakers else 0
        if isinstance(v, int):
            sid = v
            if not (0 <= sid < n_speakers):
                raise ValueError("speaker_id %d 超出范围 [0, %d)" % (sid, n_speakers))
            return sid
        name = str(v).strip()
        # str 纯数字（如 "103"）：按 speaker_id 处理，与 int 等价
        if name.isdigit():
            sid = int(name)
            if not (0 <= sid < n_speakers):
                raise ValueError("speaker_id %d 超出范围 [0, %d)" % (sid, n_speakers))
            return sid
        # str 名字：反查（先精确、再忽略大小写）
        for sid, nm in self._speakers:
            if nm == name:
                return sid
        low = name.lower()
        for sid, nm in self._speakers:
            if nm.lower() == low:
                return sid
        sample = "、".join(nm for _, nm in self._speakers[:5])
        raise ValueError("音色 %r 不存在。可用（前几个）：%s%s"
                         % (name, sample,
                            "（共 %d 个）" % len(self._speakers) if self._speakers else ""))

    # ---------------- 合成 ----------------
    def _gen(self, text, speed):
        """复刻 Alife vits()：去空格 → [ZH] 包裹 → text_to_sequence →
        add_blank 则 intersperse → infer。返回 float32（[-1,1]）。"""
        import torch
        from torch import LongTensor, no_grad

        hps = self._hps
        t = (text or "").replace("\n", " ").replace("\r", "").replace(" ", "")
        if not t:
            t = "。"
        t = "[ZH]%s[ZH]" % t
        text_norm, _ = self._text_to_sequence(t, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = self._commons.intersperse(text_norm, 0)
        x = LongTensor(text_norm).unsqueeze(0).to(self._torch_device)
        x_len = LongTensor([len(text_norm)]).to(self._torch_device)
        sid = LongTensor([self._sid]).to(self._torch_device)
        # length_scale 让引擎 speed 生效：speed>1 更快 → 更小的 length_scale
        ls = self._length_scale / speed
        with no_grad():
            audio = self._model.infer(
                x, x_len, sid=sid,
                noise_scale=self._noise_scale,
                noise_scale_w=self._noise_scale_w,
                length_scale=ls)[0][0, 0].data.cpu().float().numpy()
        return audio

    def synth(self, text, *, speed=1.0, normalize=None):
        """整句一次合成，返回 float32 numpy（[-1,1]）。调用方须持有引擎的 _synth_lock。"""
        a = self._gen(text, speed)
        if normalize:
            a = normalize_audio(a, self.sr, normalize)
        return a

    def synth_stream(self, text, *, speed=1.0, normalize=None):
        """VITS 无原生流式接口：整句一块，行为与 melo 一致。"""
        yield self.synth(text, speed=speed, normalize=normalize)

    def close(self):
        """释放模型引用（torch 显存由 GC 回收）；引擎 close() 时调用。"""
        self._model = None
        self._hps = None
        self._text_to_sequence = None
        self._commons = None
