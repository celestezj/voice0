# -*- coding: utf-8 -*-
"""TTS 交互式演示（melo / cosy）：while + input() 循环，逐段输入实时播报。

用法：
    python speak_example.py

流程：
  1. 启动时选择后端、播放模式（queue / bargein）与运行设备等；
  2. 之后循环输入文本，回车即 `submit()` 非阻塞播报——提示立即返回，可马上输入下一条；
  3. 每段播完由**后台线程立即打印**简易时序甘特图（不等下一次输入；复用 bench.console_gantt）；
  4. 输入 exit 退出，并 close() 单例 TTS 对象（Ctrl+C / EOF 同样兜底关闭）。

说明：
  - bargein 模式下，新输入会自动打断当前播放；被打断的半段也会如实画进甘特图
    （rec 的 play_start/ttfa 可能为 None，console_gantt 已做防御）。
  - 甘特图打印线程为 daemon：退出时在途任务被 close() 取消，其补打被 _EXITING 抑制。
  - 甘特图依赖 profile 插桩（逐句 4 戳计时 + TTFA/合成/等待派生字段），
    故本 demo 以 profile=True 构造 RealtimeTTS；生产场景按需关闭该开关。
"""
import os
import sys
import threading

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


_EXITING = False   # 退出中：抑制后台线程补打的甘特图，避免盖过"已关闭"提示


def _print_input_gantt(seq, job):
    """打印一次输入的甘特图（后台线程调用）。阻塞到本段播完/被打断再画。
    被打断/空分句时给兜底提示。"""
    job.wait()
    if _EXITING:
        return
    timing = job.timing
    if not timing:
        why = "（可能已被打断）" if job.canceled else ""
        print("\n（输入%d 无有效分句时序%s，跳过甘特图）" % (seq, why))
        return
    # 甘特图只画合成/播放条，句号与文本不显示；这里先行打印分句一览便于对照
    print("\n---- 输入%d 分句一览 ----" % seq)
    for r in timing:
        print("  句%d：%s" % (r["idx"] + 1, r["text"]))
    console_gantt([{"label": "输入%d" % seq, "timing": timing}])


def main():
    backend = _prompt_choice("后端引擎", [
        (1, "melo", "melo （默认）MeloTTS：中文女声、TTFA<1s、实时主力"),
        (2, "cosy", "cosy  CosyVoice2：音质更佳 + 3s 音色克隆（整句合成后播放、句内无缝，需装依赖）"),
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
    stream = None
    if backend == "cosy":
        s = input("生成长度上限 max_speech_ratio [回车=默认(None)] > ").strip()
        if s:
            try:
                max_speech_ratio = float(s)
            except ValueError:
                print("无效数字，用默认 None（cosy 0.5B 的 LLM EOS 不可靠，短句会拖长；"
                      "收紧到 8 左右可换取可控时长，代价是可能截语尾）")
        stream = _prompt_choice("播放方式", [
            (1, False, "整句合成后播放 （默认/推荐）每句一次解码、句内无缝；句首等待=整句合成耗时"),
            (2, True, "原生 token 级流式  首块更快，但本机 2070S fp32 RTF>1，块间会饿死停顿 + 拼接缝"),
        ], False)
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
                      max_speech_ratio=max_speech_ratio, stream=stream)
    print("就绪！当前 backend=%s voice=%s mode=%s device=%s normalize=%s max_speech_ratio=%s stream=%s"
          % (tts.backend, tts.voice, tts.mode, tts.device, tts.normalize, tts.max_speech_ratio, tts.stream))
    print("输入一段文本回车即播报（submit 非阻塞：提示立即返回，可马上输入下一条）；"
          "每段播完立即打印时序图（后台线程）；输入 exit 退出。\n")
    if mode == "bargein":
        print("提示：bargein 模式下，下一条输入会自动打断当前播放（打断的半段也会画进甘特图）。\n")

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
            job = tts.submit(text)   # 非阻塞：入队即返回，bargein 下自动打断当前播放
            # 后台线程等本段播完即打印甘特图（不等下一次输入；daemon，退出不阻塞）
            threading.Thread(target=_print_input_gantt, args=(seq, job), daemon=True).start()
    finally:
        _EXITING = True   # 抑制在途任务的补打线程
        print("\n关闭 TTS ...")
        tts.close()
        print("已关闭，再见。")


if __name__ == "__main__":
    main()
