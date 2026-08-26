# -*- coding: utf-8 -*-
"""MeloTTS 交互式演示：while + input() 循环，逐段输入实时播报。

用法：
    python speak_example.py

流程：
  1. 启动时选择播放模式（queue / bargein）与运行设备（auto / cpu / cuda）；
  2. 之后循环输入文本（一句话或一段话均可），回车即阻塞播报；
  3. 每段播完在控制台打印简易时序甘特图（复用 bench_melo.console_gantt）；
  4. 输入 exit 退出，并 close() 单例 TTS 对象（Ctrl+C / EOF 同样兜底关闭）。

说明：
  - 甘特图依赖 profile 插桩（逐句 4 戳计时 + TTFA/合成/等待派生字段），
    故本 demo 以 profile=True 构造 RealtimeTTS；生产场景按需关闭该开关。
"""
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

from bench.console_gantt import console_gantt  # noqa: E402  复用验收脚本的甘特图渲染
from tts.core.engine import RealtimeTTS       # noqa: E402


def _prompt_choice(title, options, default):
    """通用选择器：options = [(序号, 返回值, 显示描述)]，回车取 default。"""
    print("\n==== %s ====" % title)
    for key, value, desc in options:
        print("  %d) %s" % (key, desc))
    while True:
        s = input("请输入序号 [1-%d]，回车=默认(%s): " % (len(options), default)).strip()
        if not s:
            return default
        if s.isdigit() and 1 <= int(s) <= len(options):
            return options[int(s) - 1][1]
        print("无效输入，请重试。")


def main():
    backend = _prompt_choice("后端引擎", [
        (1, "melo", "melo （默认）MeloTTS：中文女声、TTFA<1s、实时主力"),
        (2, "cosy", "cosy  CosyVoice2：音质更佳 + 3s 音色克隆（首块 ~1-2s，需装依赖）"),
    ], "melo")
    voice = None
    if backend == "cosy":
        v = _prompt_choice("cosy 音色", [
            (1, "default", "default （推荐）内置参考女声"),
            (2, "clone", "clone:<参考wav>:<转写文本> 用自己的 3s 录音克隆音色"),
        ], "default")
        if v == "clone":
            wav = input("参考音频 WAV 路径 > ").strip()
            txt = input("该音频的转写文本 > ").strip()
            voice = "clone:%s:%s" % (wav, txt)
        else:
            voice = v
    max_speech_ratio = None
    if backend == "cosy":
        s = input("生成长度上限 max_speech_ratio [回车=默认(None)] > ").strip()
        if s:
            try:
                max_speech_ratio = float(s)
            except ValueError:
                print("无效数字，用默认 None（cosy 0.5B 的 LLM EOS 不可靠，短句会拖长；"
                      "收紧到 8 左右可换取可控时长，代价是可能截语尾）")
    mode = _prompt_choice("播放模式", [
        (1, "queue", "queue  （默认）新文本排队，播完再说"),
        (2, "bargein", "bargein        新文本打断当前播放"),
    ], "queue")
    device = _prompt_choice("运行设备", [
        (1, "auto", "auto （默认）有 GPU 用 GPU，否则 CPU"),
        (2, "cpu", "cpu"),
        (3, "cuda", "cuda"),
    ], "auto")
    normalize = _prompt_choice("响度归一化", [
        (1, None, "不归一化 （默认）原样播放，音量起伏由模型决定"),
        (2, "rms", "rms 逐句静态响度对齐（句间更一致）"),
        (3, "agc", "agc 句内动态压缩 + 短停压缩 + 句间对齐（推荐）"),
    ], None)

    print("\n正在加载 %s（模型加载 + 预热，melo 首次约 10-30s，cosy 更久）..." % backend.upper())
    tts = RealtimeTTS(device=device, backend=backend, voice=voice,
                      mode=mode, normalize=normalize, profile=True,
                      max_speech_ratio=max_speech_ratio)
    print("就绪！当前 backend=%s voice=%s mode=%s device=%s normalize=%s max_speech_ratio=%s"
          % (tts.backend, tts.voice, tts.mode, tts.device, tts.normalize, tts.max_speech_ratio))
    print("输入一段文本回车即播报；输入 exit 退出。\n")

    seq = 0
    try:
        while True:
            try:
                text = input("输入文本 (exit 退出)> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n（收到中断，退出）")
                break
            if text.lower() == "exit":
                break
            if not text:
                print("（空输入，忽略）")
                continue

            seq += 1
            timing = tts.speak(text)   # 阻塞播完本段，返回逐句时序记录
            if not timing:
                print("（该输入无有效分句，跳过甘特图）")
                continue
            # 甘特图只画合成/播放条，句号与文本不显示；这里先行打印分句一览便于对照
            print("\n---- 输入%d 分句一览 ----" % seq)
            for r in timing:
                print("  句%d：%s" % (r["idx"] + 1, r["text"]))
            console_gantt([{"label": "输入%d" % seq, "timing": timing}])
    finally:
        print("\n关闭 TTS ...")
        tts.close()
        print("已关闭，再见。")


if __name__ == "__main__":
    main()
