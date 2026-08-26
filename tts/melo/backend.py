# -*- coding: utf-8 -*-
"""MeloTTS 后端（tts/core 默认后端）。

依赖：`pip install melotts`，权重自动经 HF 缓存落地 `.cache/hf`。
melotts 与 melo.api 的 import 全部延迟到 load()——缺失时抛
BackendNotInstalledError，不会在 import 期炸掉 tts.core。
"""
import os
import time

import numpy as np  # noqa: E402

# 本文件位于 tts/melo/backend.py → 项目根向上取 3 级
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 必须在 import melo / torch 之前设置，确保权重缓存落在项目内
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))
# huggingface.co 直连被墙时走镜像（仅首次下载用，运行期零网络）；可用环境变量覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# NLTK 语料（g2p_en 的 cmudict）落到项目内，导入期零联网
os.environ.setdefault("NLTK_DATA", os.path.join(_PROJECT_DIR, ".cache", "nltk_data"))

from ..core.audio import normalize_audio  # noqa: E402
from ..core.backend import BackendNotInstalledError  # noqa: E402


class MeloBackend:
    name = "melo"
    sr = 44100

    def __init__(self, device="auto", debug=False):
        self._device = device
        self._debug = debug
        self._model = None
        self._spk = None

    def load(self):
        if self._debug:
            self._ensure_weights()
        try:
            from melo.api import TTS
        except ImportError as e:
            raise BackendNotInstalledError(
                "MeloTTS 依赖缺失：pip install melotts 后重试（%s）" % e) from e
        self._model = TTS(language="ZH", device=self._device)
        self.sr = int(getattr(self._model.hps.data, "sampling_rate", 44100))
        self._spk = self._model.hps.data.spk2id["ZH"]

    def synth(self, text, *, speed=1.0, normalize=None):
        """整句合成，返回 float32 numpy（[-1,1]）。调用方须持有引擎的 _synth_lock。"""
        a = self._model.tts_to_file(text, self._spk, output_path=None,
                                    speed=speed, quiet=True)
        if normalize:
            a = normalize_audio(a, self.sr, normalize)
        return a

    def synth_stream(self, text, *, speed=1.0, normalize=None):
        """MeloTTS/VITS 无原生流式接口：整句一块，行为与引擎旧版一致。"""
        yield self.synth(text, speed=speed, normalize=normalize)

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

    def close(self):
        """释放模型引用（torch 显存由 GC 回收）；引擎 close() 时调用。"""
        self._model = None
        self._spk = None
