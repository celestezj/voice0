# -*- coding: utf-8 -*-
"""CosyVoice2 验收脚本：默认音色 + 3s 克隆各跑一轮流式合成。

用法：
    python bench_cosy.py --device cuda --profile          # 默认音色 + 克隆各一轮（推荐）
    python bench_cosy.py --voice default                  # 只跑默认音色（不加 --profile 则裸跑）

产出（全部在本项目路径下）：
    audio/2_cosy_{device}_{case}.wav / .mp3     合成音频（试听/对比克隆相似度）
    audio/bench_report_cosy_{device}.txt       逐句原始指标
    .cache/bench_cosy_{device}.json            结构化指标
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

# 本文件位于 bench/bench_cosy.py → 项目根向上取 1 级
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)

from bench.console_gantt import console_gantt  # noqa: E402

OUT_AUDIO = os.path.join(_PROJECT_DIR, "audio")
OUT_CACHE = os.path.join(_PROJECT_DIR, ".cache")

# 克隆参考音频：默认用内置参考女声跑通"克隆代码路径"；真正换音色时用
# --clone-wav / --clone-text 指定自己的 3s 录音与转写。
CLONE_WAV = os.path.join(_PROJECT_DIR, "assets", "cosy_default_female.wav")
CLONE_TEXT = "希望你以后能够做的比我还好呦。"

CASES = [
    ("default", "默认音色", "default"),
    ("clone",   "3s克隆",   "clone:%s:%s" % (CLONE_WAV, CLONE_TEXT)),
]

TEXT = ("这是一款将 WiFi 无线信号转化为实时空间感知能力的工具。"
        "通过分析人体活动引起的信道状态信息变化。无需摄像头或穿戴设备。"
        "即可实时还原人体姿态。")


def run(device, voice_filter, profile, debug, clone_wav, clone_text, max_ratio):
    import torch
    from tts.core.engine import RealtimeTTS
    results = []
    for key, label, voice in CASES:
        if voice_filter not in ("all", key):
            continue
        if key == "clone":
            if not os.path.exists(clone_wav) or not clone_text.strip():
                print("[bench] 跳过克隆用例（缺参考音频或文本为空）")
                continue
            voice = "clone:%s:%s" % (clone_wav, clone_text.strip())
        t0 = datetime.datetime.now()
        tts = RealtimeTTS(device=device, backend="cosy", voice=voice,
                          profile=profile, debug=debug,
                          max_speech_ratio=max_ratio)
        t_load = (datetime.datetime.now() - t0).total_seconds()
        torch.cuda.reset_peak_memory_stats()
        timing = tts.speak(TEXT)
        wav = os.path.join(OUT_AUDIO, "2_cosy_%s_%s.wav" % (device, key))
        tts.speak_to_file(TEXT, wav)
        mp3 = wav[:-4] + ".mp3"
        _to_mp3(wav, mp3)
        results.append({
            "key": key, "label": label, "voice": voice, "text": TEXT,
            "timing": timing, "t_load": t_load, "wav": os.path.relpath(wav, _PROJECT_DIR),
            "max_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "max_ratio": max_ratio,
        })
        if profile and timing:
            print("[bench] %s/%s: 首句TTFA=%.3fs 句数=%d 加载%.1fs 峰值显存=%.2fGB"
                  % (device, label, timing[0]["ttfa"], len(timing), t_load,
                     results[-1]["max_vram_gb"]))
    return results


def _to_mp3(wav_path, mp3_path):
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                 "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path], check=True)
            return True
        except Exception as e:
            print("[bench] ffmpeg 转 MP3 失败(忽略): %s" % e)
    return False


def write_report_txt(device, results):
    os.makedirs(OUT_AUDIO, exist_ok=True)
    path = os.path.join(OUT_AUDIO, "bench_report_cosy_%s.txt" % device)
    lines = ["CosyVoice2 bench report: device=%s（时间轴按整段跨度归一化，音频≫合成时合成会压成 0 格）"
             % device, ""]
    for res in results:
        ratio = res.get("max_ratio")
        lines.append("[%s] %s（voice=%s, 加载%.1fs, 峰值显存 %.2fGB%s）"
                     % (res["label"], res["text"][:30], res["voice"], res["t_load"], res["max_vram_gb"],
                        "，max_speech_ratio=%s" % ratio if ratio else ""))
        lines.append("%-4s %-14s %10s %10s %10s %10s %10s"
                     % ("句", "文本预览", "合成ms", "等待ms", "TTFA_s", "间隔ms", "音频s"))
        for r in res["timing"]:
            gap = "%d" % round(r["interval"] * 1000) if r["interval"] is not None else "-"
            lines.append("%-4s %-14s %10.0f %10.0f %10.3f %10s %10.2f"
                         % (r["idx"] + 1, r["text"][:14], r["synth_dur"] * 1000,
                            r["wait"] * 1000, r["ttfa"], gap, r["audio_dur"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[bench] 原始指标已写: %s" % path)


def dump_json(device, results):
    os.makedirs(OUT_CACHE, exist_ok=True)
    payload = {"meta": {"device": device, "engine": "cosyvoice2"},
               "results": [{
                   "key": r["key"], "label": r["label"], "voice": r["voice"],
                   "t_load": r["t_load"], "max_vram_gb": r["max_vram_gb"],
                   "timing": r["timing"],
               } for r in results]}
    with open(os.path.join(OUT_CACHE, "bench_cosy_%s.json" % device), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description="CosyVoice2 验收脚本")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--voice", choices=["default", "clone", "all"], default="all")
    ap.add_argument("--profile", action="store_true", help="时序分析：逐句4戳计时+甘特图")
    ap.add_argument("--debug", action="store_true", help="详细日志：环境/设备/加载")
    ap.add_argument("--clone-wav", default=CLONE_WAV, help="克隆参考音频（默认内置）")
    ap.add_argument("--clone-text", default=CLONE_TEXT, help="克隆参考音频转写文本")
    ap.add_argument("--max-ratio", type=float, default=None,
                    help="收紧 LLM 生成上限（默认 None=模型 20×；如 8 → 句长可控但可能截语尾）")
    args = ap.parse_args()

    os.makedirs(OUT_AUDIO, exist_ok=True)
    os.makedirs(OUT_CACHE, exist_ok=True)
    print("\n########## CosyVoice2 bench: device=%s voice=%s max_ratio=%s ##########"
          % (args.device, args.voice, args.max_ratio))
    results = run(args.device, args.voice, args.profile, args.debug,
                  args.clone_wav, args.clone_text, args.max_ratio)
    if not results:
        sys.exit("[bench] 无有效用例（voice 过滤或克隆源缺失）")
    write_report_txt(args.device, results)
    dump_json(args.device, results)
    if args.profile:
        console_gantt(results)


if __name__ == "__main__":
    main()
