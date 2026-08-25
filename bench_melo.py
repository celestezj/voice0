# -*- coding: utf-8 -*-
"""MeloTTS 验收脚本：CPU/GPU 各跑一轮流式 TTS，产出时序指标 + 双版本时序图 + README 自动维护。

用法：
    python bench_melo.py --device all --profile --debug   # 全量验收（默认）
    python bench_melo.py --device cpu --profile           # 单设备
    python bench_melo.py --device cuda                    # 裸管线，无插桩

产出（全部在本项目路径下）：
    audio/2_melo_{device}_{case}.wav / .mp3        整句音频（与 SAPI 并排试听）
    audio/chunks/{device}/{case}/句NN.wav          逐句音频（HTML 悬停行内试听）
    reports/bench_timing_{device}.html             交互式时序图（悬停句块看明细）
    audio/bench_report_{device}.txt                逐句原始指标
    .cache/bench_{device}.json                     结构化指标（供 README 重生成）
    README.md                                      验收结果区间 + 目录树自动刷新
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

OUT_AUDIO = os.path.join(_PROJECT_DIR, "audio")
OUT_REPORTS = os.path.join(_PROJECT_DIR, "reports")
OUT_CACHE = os.path.join(_PROJECT_DIR, ".cache")
README_PATH = os.path.join(_PROJECT_DIR, "README.md")

# 用例文本均含句末标点：_split 会切成多句，时序图才能展示句间"合成/播放"流式重叠。
CASE_LONG = ("这是一款将 WiFi 无线信号转化为实时空间感知能力的工具。"
             "通过分析人体活动引起的信道状态信息变化，无需摄像头或穿戴设备。"
             "即可实时还原人体姿态，并检测心率和呼吸。")
CASES = [
    ("short", "短", "今天天气真不错。我们一起去公园散步吧。"),
    ("mid",   "中", "本系统无需摄像头。通过分析信道状态信息。即可实时还原人体姿态。"),
    ("long",  "长", CASE_LONG),
]


# ---------------------------------------------------------------------------
# 单设备一轮
# ---------------------------------------------------------------------------
def run_device(device, profile, debug):
    # 惰性导入：--device all 时父进程只做子进程编排，不加载重型依赖
    from tts_melo import RealtimeTTS  # noqa: E402
    tts = RealtimeTTS(device=device, profile=profile, debug=debug)
    results = []
    if device == "cuda" and profile:
        import torch
        torch.cuda.reset_peak_memory_stats()
    for key, label, text in CASES:
        chunks_dir = os.path.join(OUT_AUDIO, "chunks", device, key) if profile else None
        timing = tts.speak(text, save_chunks_dir=chunks_dir)
        wav = os.path.join(OUT_AUDIO, "2_melo_%s_%s.wav" % (device, key))
        tts.speak_to_file(text, wav)
        mp3 = wav[:-4] + ".mp3"
        _to_mp3(wav, mp3)
        results.append({
            "key": key, "label": label, "text": text, "timing": timing,
            "wav": os.path.relpath(wav, _PROJECT_DIR),
        })
        if profile:
            print("[bench] %s/%s: 首句TTFA=%.3fs 句数=%d"
                  % (device, label, timing[0]["ttfa"], len(timing)))
    meta = _device_meta(device, profile)
    if profile:
        write_report_txt(device, results, meta)
        generate_html(device, results, os.path.join(OUT_REPORTS, "bench_timing_%s.html" % device))
        _dump_bench_json(device, results, meta)
    return results


def _device_meta(device, profile):
    meta = {"device": device, "gpu_name": None, "max_vram_gb": None,
            "torch": None, "cuda": None}
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.is_available()
        if device == "cuda" and torch.cuda.is_available() and profile:
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["max_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return meta


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


# ---------------------------------------------------------------------------
# 控制台简易甘特图
# ---------------------------------------------------------------------------
def console_gantt(results, width=58):
    print("\n==== 控制台时序甘特图 ====")
    for res in results:
        timing = res["timing"]
        if not timing:
            continue
        t0 = timing[0]["synth_start"]
        t_end = max(max(r["synth_end"], r["play_start"] + r["audio_dur"]) for r in timing)
        span = t_end - t0

        def x(t):
            return int((t - t0) / span * (width - 1)) if span > 0 else 0

        print("\n[%s] 跨度=%.2fs" % (res["label"], span))
        for r in timing:
            i = r["idx"]
            s0, s1 = x(r["synth_start"]), x(r["synth_end"])
            p0, p1 = x(r["play_start"]), x(r["play_start"] + r["audio_dur"])
            row_s = [" "] * width
            for k in range(s0, min(s1, width - 1)):
                row_s[k] = "="
            row_p = [" "] * width
            for k in range(p0, min(p1, width - 1)):
                row_p[k] = "-"
            tag = "  <- 句%d TTFA=%.3fs" % (i + 1, r["ttfa"]) if i == 0 else ""
            print("  句%d合成 %s%s" % (i + 1, "".join(row_s), tag))
            print("  句%d播放 %s" % (i + 1, "".join(row_p)))


# ---------------------------------------------------------------------------
# 原始指标 txt
# ---------------------------------------------------------------------------
def write_report_txt(device, results, meta):
    os.makedirs(OUT_AUDIO, exist_ok=True)
    path = os.path.join(OUT_AUDIO, "bench_report_%s.txt" % device)
    lines = ["MeloTTS bench report: device=%s" % device,
             "torch=%s cuda=%s gpu=%s" % (meta.get("torch"), meta.get("cuda"), meta.get("gpu_name")),
             "max_vram_gb=%s" % meta.get("max_vram_gb"), ""]
    for res in results:
        lines.append("[%s] %s" % (res["label"], res["text"]))
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


# ---------------------------------------------------------------------------
# 交互式 HTML 时序图（自包含、无 CDN、悬停看明细）
# ---------------------------------------------------------------------------
def generate_html(device, results, out_path):
    os.makedirs(OUT_REPORTS, exist_ok=True)
    cases = []
    for res in results:
        timing = res["timing"]
        if not timing:
            continue
        t0 = timing[0]["synth_start"]
        t_end = max(r["play_start"] + r["audio_dur"] for r in timing)
        rows = []
        for r in timing:
            rows.append({
                "label": "句%d" % (r["idx"] + 1),
                "text": r["text"],
                "synth_start": r["synth_start"] - t0,
                "synth_end": r["synth_end"] - t0,
                "play_start": r["play_start"] - t0,
                "audio_dur": r["audio_dur"],
                "synth_dur": r["synth_dur"],
                "wait": r["wait"],
                "ttfa": r["ttfa"],
                "interval": r["interval"],
                "audio_src": "../audio/chunks/%s/%s/句%02d.wav" % (device, res["key"], r["idx"] + 1),
            })
        cases.append({"name": res["label"], "span": t_end - t0, "rows": rows})

    data_json = json.dumps(cases, ensure_ascii=False)
    css = """
    body { font-family: "Microsoft YaHei", sans-serif; margin: 24px; background:#f7f7f7; color:#222; }
    h1 { font-size: 20px; }
    .case { background:#fff; border:1px solid #ddd; border-radius:8px; padding:14px; margin:16px 0; }
    .case h2 { margin:0 0 8px; font-size:16px; }
    .axis { position:relative; height:18px; margin:4px 0 2px 64px; font-size:11px; color:#888; }
    .axis span { position:absolute; transform:translateX(-50%); }
    .row { position:relative; display:flex; align-items:center; margin:3px 0; padding:3px 0; border-radius:4px; cursor:default; }
    .row:hover { background:#eef4ff; }
    .lbl { width:64px; font-size:12px; color:#555; flex:none; text-align:right; padding-right:8px; }
    .track { position:relative; flex:1; height:22px; }
    .b { position:absolute; height:22px; border-radius:3px; opacity:.85; }
    .synth { background:#4f8ef7; }
    .play { background:#31b46e; }
    audio { height:22px; margin-left:8px; flex:none; }
    #tip { position:fixed; display:none; background:#1e293b; color:#fff; padding:8px 10px; border-radius:6px;
           font-size:12px; line-height:1.6; max-width:360px; z-index:99; pointer-events:none; }
    """
    js = r"""
    const DATA = __DATA__;
    const chart = document.getElementById('chart');
    const tip = document.getElementById('tip');
    function pct(t, span){ return (t / span * 100); }
    function ms(v){ return Math.round(v * 1000); }
    function showTip(row, e, r){
        const itv = (r.interval === null || r.interval === undefined) ? '-' : (ms(r.interval) + ' ms');
        tip.innerHTML =
            '<b>' + r.label + '</b><br>' +
            '完整文本：' + r.text + '<br>' +
            '合成耗时：' + ms(r.synth_dur) + ' ms<br>' +
            '入队→播放等待：' + ms(r.wait) + ' ms<br>' +
            '该句 TTFA：' + r.ttfa.toFixed(3) + ' s<br>' +
            '与上句间隔：' + itv + '<br>' +
            '音频时长：' + r.audio_dur.toFixed(2) + ' s<br>' +
            '开始播放(相对 0s)：' + r.play_start.toFixed(3) + ' s';
        tip.style.display = 'block';
        moveTip(e);
    }
    function moveTip(e){
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY + 14) + 'px';
    }
    DATA.forEach(cs => {
        const sec = document.createElement('div'); sec.className = 'case';
        sec.innerHTML = '<h2>' + cs.name + '（span=' + cs.span.toFixed(2) + 's）</h2>';
        const axis = document.createElement('div'); axis.className = 'axis';
        for (let i = 0; i <= 5; i++){
            const l = document.createElement('span');
            l.style.left = (100 * i / 5) + '%';
            l.textContent = (cs.span * i / 5).toFixed(2) + 's';
            axis.appendChild(l);
        }
        sec.appendChild(axis);
        cs.rows.forEach(r => {
            const row = document.createElement('div'); row.className = 'row';
            const track = document.createElement('div'); track.className = 'track';
            const synth = document.createElement('div'); synth.className = 'b synth';
            synth.style.left = pct(r.synth_start, cs.span) + '%';
            synth.style.width = Math.max(pct(r.synth_end - r.synth_start, cs.span), 0.4) + '%';
            const play = document.createElement('div'); play.className = 'b play';
            play.style.left = pct(r.play_start, cs.span) + '%';
            play.style.width = Math.max(pct(r.audio_dur, cs.span), 0.4) + '%';
            track.appendChild(synth); track.appendChild(play);
            const lbl = document.createElement('span'); lbl.className = 'lbl'; lbl.textContent = r.label;
            const au = document.createElement('audio'); au.controls = true; au.preload = 'none'; au.src = r.audio_src;
            row.appendChild(lbl); row.appendChild(track); row.appendChild(au);
            row.addEventListener('mouseenter', e => showTip(row, e, r));
            row.addEventListener('mousemove', moveTip);
            row.addEventListener('mouseleave', () => tip.style.display = 'none');
            sec.appendChild(row);
        });
        chart.appendChild(sec);
    });
    """
    js = js.replace("__DATA__", data_json)
    html = ("<!DOCTYPE html>\n<html lang=\"zh\">\n<head><meta charset=\"utf-8\">"
            "<title>MeloTTS 时序图 - %s</title>\n<style>%s</style></head>\n"
            "<body><h1>MeloTTS 流式时序图 · device=%s（蓝=合成，绿=播放，悬停句块看明细，行尾可试听）</h1>"
            "<div id=\"chart\"></div><div id=\"tip\"></div>\n<script>%s</script>\n</body></html>"
            % (device, css, device, js))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("[bench] 交互式时序图已写: %s" % out_path)


# ---------------------------------------------------------------------------
# README 自动维护（验收结果 + 目录树）
# ---------------------------------------------------------------------------
def _dump_bench_json(device, results, meta):
    os.makedirs(OUT_CACHE, exist_ok=True)
    payload = {"meta": meta,
               "results": [{
                   "key": r["key"], "label": r["label"], "text": r["text"],
                   "timing": r["timing"],
               } for r in results]}
    with open(os.path.join(OUT_CACHE, "bench_%s.json" % device), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def _load_bench(device):
    p = os.path.join(OUT_CACHE, "bench_%s.json" % device)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _markdown_for_device(payload):
    d = payload["meta"]["device"]
    lines = ["### %s（device=%s）" % (d.upper(), d)]
    if payload["meta"].get("gpu_name"):
        lines.append("显卡：%s ｜ 峰值显存：%s GB" % (payload["meta"]["gpu_name"],
                                                     payload["meta"].get("max_vram_gb")))
    lines.append("")
    lines.append("| 用例 | 句数 | 首句TTFA(s) | 平均合成(ms/句) | 最大等待(ms) | 总跨度(s) |")
    lines.append("|---|---|---|---|---|---|")
    for r in payload["results"]:
        t = r["timing"]
        avg = sum(x["synth_dur"] for x in t) / len(t) * 1000 if t else 0
        mxw = max((x["wait"] for x in t), default=0) * 1000
        t0 = t[0]["synth_start"]
        span = max(x["play_start"] + x["audio_dur"] for x in t) - t0
        lines.append("| %s | %d | %.3f | %.0f | %.0f | %.2f |"
                     % (r["label"], len(t), t[0]["ttfa"], avg, mxw, span))
    lines.append("")
    lines.append("- 交互式时序图（悬停看明细）: `reports/bench_timing_%s.html`" % d)
    lines.append("- 逐句原始指标: `audio/bench_report_%s.txt`" % d)
    lines.append("")
    return "\n".join(lines)


def _comparison_markdown():
    cpu = _load_bench("cpu")
    cuda = _load_bench("cuda")
    if not cpu or not cuda:
        have = "CPU" if cpu else "GPU"
        return "（当前仅有 %s 数据，需再跑另一设备后自动生成完整对比）" % have
    c0 = cpu["results"][0]["timing"][0]
    g0 = cuda["results"][0]["timing"][0]
    ratio = c0["ttfa"] / g0["ttfa"] if g0["ttfa"] else float("inf")
    cpu_ok = c0["ttfa"] < 1.0
    gpu_ok = g0["ttfa"] < 1.0
    lines = [
        "### CPU / GPU 对比结论（bench 自动生成）",
        "",
        "| 指标 | CPU | GPU |",
        "|---|---|---|",
        "| 首句 TTFA | %.3fs | %.3fs |" % (c0["ttfa"], g0["ttfa"]),
        "| TTFA 达标(<1s) | %s | %s |" % ("✅" if cpu_ok else "❌", "✅" if gpu_ok else "❌"),
        "| GPU 相对 CPU 提速 | — | %.1f× |" % ratio,
        "| GPU 峰值显存 | — | %s GB |" % cuda["meta"].get("max_vram_gb"),
        "",
        "**结论**：GPU 首句 TTFA=%.3fs%s；CPU 首句 TTFA=%.3fs%s。%s"
        % (g0["ttfa"], " 达标<1s" if gpu_ok else " 未达标",
           c0["ttfa"], " 达标<1s" if cpu_ok else " 未达标",
           "流式重叠已验证：播放段与下一句合成段重叠。" if _has_overlap(cuda) and _has_overlap(cpu)
           else "流式重叠待核对（请查看时序图）。"),
    ]
    return "\n".join(lines)


def _has_overlap(payload):
    for r in payload["results"]:
        t = r["timing"]
        for i in range(len(t) - 1):
            if t[i]["play_start"] < t[i + 1]["synth_end"]:
                return True
    return False


def update_readme():
    os.makedirs(OUT_CACHE, exist_ok=True)
    if not os.path.exists(README_PATH):
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write("# voice0\n\n")
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    parts = []
    for d in ("cpu", "cuda"):
        p = _load_bench(d)
        if p:
            parts.append(_markdown_for_device(p))
    parts.append(_comparison_markdown())
    bench_block = ("\n".join(parts) + "\n\n> 生成时间：%s（每次 bench 运行自动刷新）"
                   % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    content = _replace_marker(content, "<!-- bench:start -->", "<!-- bench:end -->", bench_block)

    tree_block = "```\n" + _dir_tree(_PROJECT_DIR) + "\n```"
    content = _replace_marker(content, "<!-- tree:start -->", "<!-- tree:end -->", tree_block)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("[bench] README.md 已更新（验收结果 + 目录树）")


def _replace_marker(content, start_marker, end_marker, new_block):
    """把 start..end 标记区间替换为 new_block；标记缺失则追加。

    注意 end 标记可能位于 EOF 且无尾随换行（find("\\n", e) 会返回 -1），
    此时 content[e+1:] 会把整个文档再拼一遍造成双份——必须显式处理。
    """
    s = content.find(start_marker)
    if s == -1:
        sep = "\n---\n" if content.strip() else ""
        return content + sep + start_marker + "\n" + new_block + "\n" + end_marker + "\n"
    e = content.find(end_marker, s + len(start_marker))
    if e == -1:
        sep = "\n---\n" if content.strip() else ""
        return content + sep + start_marker + "\n" + new_block + "\n" + end_marker + "\n"
    nl = content.find("\n", e)
    tail_start = (nl + 1) if nl != -1 else len(content)
    return content[:s] + start_marker + "\n" + new_block + "\n" + end_marker + "\n" + content[tail_start:]


def _dir_tree(root):
    skip = {".git", ".cache", "__pycache__", ".claude"}
    lines = []

    def walk(path, prefix):
        entries = sorted(os.listdir(path))
        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and e not in skip]
        files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and e not in skip]
        for i, d in enumerate(dirs):
            last = (i == len(dirs) - 1 and not files)
            lines.append(prefix + ("└── " if last else "├── ") + d + "/")
            walk(os.path.join(path, d), prefix + ("    " if last else "│   "))
        for i, f in enumerate(files):
            last = (i == len(files) - 1)
            lines.append(prefix + ("└── " if last else "├── ") + f)

    lines.append(os.path.basename(root) + "/")
    walk(root, "")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MeloTTS 验收脚本")
    ap.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    ap.add_argument("--profile", action="store_true", help="时序分析：逐句4戳计时+甘特图+HTML+README")
    ap.add_argument("--debug", action="store_true", help="详细日志：环境/设备/权重下载")
    ap.add_argument("--no-readme", action="store_true", help="不自动更新 README")
    args = ap.parse_args()

    # --device all：按设备拆子进程。MeloTTS 的 BERT 是模块级单例且首次按设备加载，
    # CPU/GPU 同进程复用会设备不匹配；隔离进程同时天然隔离 import 与模型状态。
    if args.device == "all":
        for dev in ("cpu", "cuda"):
            cmd = [sys.executable, os.path.abspath(__file__), "--device", dev]
            if args.profile:
                cmd.append("--profile")
            if args.debug:
                cmd.append("--debug")
            cmd.append("--no-readme")
            print("\n########## 子进程: %s ##########" % " ".join(cmd))
            rc = subprocess.run(cmd, cwd=_PROJECT_DIR).returncode
            if rc != 0:
                print("[bench] %s 轮次失败，退出码 %d" % (dev, rc))
                sys.exit(rc)
        if args.profile and not args.no_readme:
            update_readme()
        print("\n完成。裸管线运行: python bench_melo.py --device cuda")
        return

    os.makedirs(OUT_AUDIO, exist_ok=True)
    os.makedirs(OUT_REPORTS, exist_ok=True)
    os.makedirs(OUT_CACHE, exist_ok=True)

    dev = args.device
    print("\n########## 开始 %s 轮次 ##########" % dev)
    results = run_device(dev, args.profile, args.debug)
    if args.profile:
        console_gantt(results)
    if args.profile and not args.no_readme:
        update_readme()
    print("\n完成。裸管线运行: python bench_melo.py --device cuda")


if __name__ == "__main__":
    main()
