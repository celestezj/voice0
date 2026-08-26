# -*- coding: utf-8 -*-
"""CosyVoice2 的 transformers pin 落盘脚本。

CosyVoice2 需要 transformers==4.51.3（官方 issue #1546：>4.51.3 产出杂音）。
本脚本把 transformers 4.51.3 + tokenizers 0.21.1 以 --no-deps 装到
`.cache/pinned_transformers`（约 94M，离线可用），供 `tts/cosy/backend.py`
在 import 时注入 sys.path。不会改动 main env 里其它包（torch / huggingface_hub
等仍用 main env 版本），melo 后端完全不受影响。

用法：`python setup_cosy_pinned.py`（建议用 voice-tts 环境解释器跑）
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PIN = os.path.join(_ROOT, ".cache", "pinned_transformers")


def main():
    os.makedirs(os.path.join(_ROOT, ".cache"), exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--target", _PIN, "--no-deps",
           "transformers==4.51.3", "tokenizers==0.21.1"]
    print(">> " + " ".join(cmd))
    subprocess.check_call(cmd)

    # 验证注入可用
    if _PIN not in sys.path:
        sys.path.insert(0, _PIN)
    import transformers  # noqa: E402
    assert transformers.__version__ == "4.51.3", transformers.__version__
    print("OK: pinned transformers %s ready at %s" % (transformers.__version__, _PIN))


if __name__ == "__main__":
    main()
