# -*- coding: utf-8 -*-
"""后端抽象：TTSBackend 协议 + 惰性加载。

选择性安装的核心：`tts/core` 只在需要时 import 具体后端模块（melo/cosy），
后端模块顶层不 import 任何推理依赖（numpy 等轻量依赖除外），
模型与框架级 import 全部延迟到 `load()` 内——缺失依赖时抛
`BackendNotInstalledError`（带安装提示），而不是在 import 期炸掉整个引擎。
"""
import importlib

# 后端模块与类名约定：tts/<name>/backend.py 的 <Name>Backend
_BACKEND_MODULES = {
    "melo": ("tts.melo.backend", "MeloBackend"),
    "cosy": ("tts.cosy.backend", "CosyBackend"),
}


class BackendNotInstalledError(RuntimeError):
    """对应后端依赖未安装 / 加载失败。携带安装提示。"""


class TTSBackend:
    """引擎消费的统一接口。引擎只认这几个能力，其余差异全在后端内部。"""

    name = ""
    sr = 44100                     # 采样率：引擎全用 self._sr 派生，后端定死即可

    def load(self):
        """惰性 import 模型并初始化。失败抛 BackendNotInstalledError。"""
        raise NotImplementedError

    def synth_stream(self, text, *, speed=1.0, normalize=None):
        """整句流式合成，逐块产出 float32 numpy（块时长由后端定，~1s 量级）。

        normalize 在句级/块级聚合处应用（与 melo 同语义：整句一次对齐）。
        """
        raise NotImplementedError

    def synth(self, text, *, speed=1.0, normalize=None):
        """整句一次合成，返回完整数组（melo 原生；cosy 内部聚合 stream）。"""
        raise NotImplementedError

    def close(self):
        """释放模型/会话。引擎 close() 时调用。"""
        raise NotImplementedError


def get_backend(name="melo", device="auto", **cfg):
    """按名字惰性加载后端类并构造实例（不 load——模型在 load() 才真正建）。

    - 未知名字 → ValueError（代码笔误，不是运行时问题）
    - 依赖缺失 / 加载失败 → BackendNotInstalledError（附安装提示）
    """
    n = (name or "melo").lower()
    if n not in _BACKEND_MODULES:
        raise ValueError("未知后端: %r（可选: %s）"
                         % (name, "/".join(sorted(_BACKEND_MODULES))))
    mod_name, cls_name = _BACKEND_MODULES[n]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:          # 后端模块顶层错误：统一转成带提示的异常
        raise BackendNotInstalledError(
            "后端 %r 模块加载失败（%s）。请按对应 README 安装依赖后重试。"
            % (name, e)) from e
    cls = getattr(mod, cls_name)
    return cls(device=device, **cfg)
