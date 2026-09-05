# -*- coding: utf-8 -*-
"""VITS 多音色验收脚本：默认音色 + 2 个额外音色各跑一轮（CASES 与 bench_melo 一致，便于对比）。

用法：
    python bench_vits.py --device cuda --profile          # 3 个音色全跑（推荐）
    python bench_vits.py --device cuda --voice default    # 只跑默认音色 551=派蒙

产出（全部在本项目路径下）：
    audio/2_vits_{device}_{voice}_{case}.wav / .mp3   合成音频（试听音色差异）
    audio/bench_report_vits_{device}.txt              逐句原始指标
    .cache/bench_vits_{device}.json                   结构化指标
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

# 本文件位于 bench/bench_vits.py → 项目根向上取 1 级
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)

from bench.console_gantt import console_gantt  # noqa: E402

OUT_AUDIO = os.path.join(_PROJECT_DIR, "audio")
OUT_CACHE = os.path.join(_PROJECT_DIR, ".cache")

# 用例与 bench_melo.py 完全一致（含句末标点 → 多句流式），保证两后端对比公平
CASE_LONG = ("这是一款将 WiFi 无线信号转化为实时空间感知能力的工具。"
             "通过分析人体活动引起的信道状态信息变化，无需摄像头或穿戴设备。"
             "即可实时还原人体姿态，并检测心率和呼吸。")
CASES = [
    ("short", "短", "今天天气真不错。我们一起去公园散步吧。"),
    ("mid",   "中", "本系统无需摄像头。通过分析信道状态信息。即可实时还原人体姿态。"),
    ("long",  "长", CASE_LONG),
]

# 音色对比：默认 551（派蒙）+ 2 个额外音色（特别周 0 / 黄金船 6，来自 speakers_list.txt）
VOICES = [
    ("default", "默认551·派蒙", None),
    ("sp0",     "特别周(0)",    0),
    ("sp6",     "黄金船(6)",    6),
]


def run(device, voice_filter, profile, debug):
    from tts.core.engine import RealtimeTTS  # noqa: E402
    import torch
    results = []
    for vkey, vlabel, voice in VOICES:
        if voice_filter not in ("all", vkey):
            continue
        t0 = datetime.datetime.now()
        tts = RealtimeTTS(device=device, backend="vits", voice=voice,
                          profile=profile, debug=debug)
        t_load = (datetime.datetime.now() - t0).total_seconds()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for key, label, text in CASES:
            timing = tts.speak(text)   # 阻塞播完（同 bench_melo）
            wav = os.path.join(OUT_AUDIO, "2_vits_%s_%s_%s.wav" % (device, vkey, key))
            tts.speak_to_file(text, wav)
            mp3 = wav[:-4] + ".mp3"
            _to_mp3(wav, mp3)
            results.append({
                "voice_key": vkey, "voice_label": vlabel, "voice": voice,
                "case_key": key, "case_label": label,
                "label": "%s|%s" % (vlabel, label), "text": text,
                "timing": timing, "t_load": t_load,
                "wav": os.path.relpath(wav, _PROJECT_DIR),
                "max_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
                if device == "cuda" else None,
            })
            if profile and timing:
                print("[bench] %s/%s/%s: 首句TTFA=%.3fs 句数=%d"
                      % (device, vlabel, label, timing[0]["ttfa"], len(timing)))
        print("[bench] %s/%s: 加载%.1fs 峰值显存=%sGB"
              % (device, vlabel, t_load,
                 "%.2f" % results[-1]["max_vram_gb"] if results[-1]["max_vram_gb"] else "-"))
        tts.close()
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
    path = os.path.join(OUT_AUDIO, "bench_report_vits_%s.txt" % device)
    lines = ["VITS multi-speaker bench report: device=%s" % device, ""]
    for res in results:
        lines.append("[%s|%s] %s（加载%.1fs, 峰值显存 %sGB）"
                     % (res["voice_label"], res["case_label"], res["text"][:30],
                        res["t_load"], res["max_vram_gb"] if res["max_vram_gb"] is not None else "-"))
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
    payload = {"meta": {"device": device, "engine": "vits"},
               "results": results}
    with open(os.path.join(OUT_CACHE, "bench_vits_%s.json" % device), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description="VITS 多音色验收脚本")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--voice", choices=["default", "sp0", "sp6", "all"], default="all")
    ap.add_argument("--profile", action="store_true", help="时序分析：逐句4戳计时+甘特图")
    ap.add_argument("--debug", action="store_true", help="详细日志：环境/设备/加载")
    args = ap.parse_args()

    os.makedirs(OUT_AUDIO, exist_ok=True)
    os.makedirs(OUT_CACHE, exist_ok=True)
    print("\n########## VITS bench: device=%s voice=%s ##########" % (args.device, args.voice))
    results = run(args.device, args.voice, args.profile, args.debug)
    if not results:
        sys.exit("[bench] 无有效用例（voice 过滤为空）")
    write_report_txt(args.device, results)
    dump_json(args.device, results)
    if args.profile:
        console_gantt(results)


if __name__ == "__main__":
    main()
