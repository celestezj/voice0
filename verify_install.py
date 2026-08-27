# -*- coding: utf-8 -*-
"""安装/复现后的端到端验证：melo 后端合成一句中文并落盘 WAV。

用法（voice-tts 环境）：
    python verify_install.py

做了什么：
    - 加载 melo 后端（device=auto：有 GPU 走 cuda，无 GPU 走 cpu）
    - 合成一句中文 → audio/setup_smoke.wav
    - 断言 WAV 非空

说明：
    - speak_to_file 走后端合成 + 落盘，不依赖声卡/播放设备；
      RealtimeTTS 常驻播放线程在无声卡机器上会自行退出，不影响本验证。
    - 验证的是与 speak_example.py 完全相同的引擎路径（RealtimeTTS 单例
      加载 melo 后端 + 合成），通过即代表 speak_example.py 可跑。
    - 随时可重跑，用于确认本机环境没有退化。
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from tts import RealtimeTTS  # noqa: E402
import torch  # noqa: E402


def main():
    # CUDA 可用性实检（装的是 GPU 版 torch 却返回 False = 驱动/安装不匹配，要提示）
    cuda_ok = torch.cuda.is_available()
    print("[verify] torch.cuda.is_available() = %s" % cuda_ok)
    if cuda_ok:
        print("[verify] 使用 GPU 合成")
    else:
        print("[verify] 使用 CPU 合成（melo CPU 可用；若你本应能用 GPU，"
              "请检查驱动版本或所装 torch 是否为 CUDA 版）")
    print("[verify] 加载 melo 后端并合成一句中文（首次加载模型约需 10-40 秒）...")
    tts = RealtimeTTS(backend="melo")          # device=auto：有 GPU→cuda，无→cpu
    out = os.path.join("audio", "setup_smoke.wav")
    try:
        tts.speak_to_file("你好，欢迎使用 voice0 离线语音合成系统。安装验证通过。", out)
    finally:
        tts.close()
    size = os.path.getsize(out)
    assert size > 0, "合成的 WAV 为空文件！"
    print("[verify] OK: 已合成 -> %s (%.1f KB)" % (out, size / 1024.0))
    print("[verify] 下一步：python speak_example.py（交互式播报 demo）")


if __name__ == "__main__":
    main()
