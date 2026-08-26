# -*- coding: utf-8 -*-
"""voice0 TTS 引擎包（后端无关核心 + 各 TTS 后端）。

入口：
    from tts import RealtimeTTS          # 常驻引擎（backend 参数可选，默认 melo）
    from tts import get_backend         # 惰性加载后端（构造实例，不 load）

melo/cosy 后端仅由 core 惰性加载，顶层 import 不触发任何后端——"只装 melotts /
只装 cosyvoice2 / 两者都装"都能正常使用。缺失后端报 BackendNotInstalledError。
"""
from .core.engine import RealtimeTTS  # noqa: F401
from .core.backend import BackendNotInstalledError, get_backend  # noqa: F401

__all__ = ["RealtimeTTS", "get_backend", "BackendNotInstalledError"]
