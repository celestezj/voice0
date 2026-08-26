# -*- coding: utf-8 -*-
"""控制台简易时序甘特图（跨后端/跨脚本共享）。

输入：results = [{"label": "...", "timing": [逐句 rec, ...]}]。
rec 字段：idx/text/synth_start/synth_end/play_start/audio_dur/ttfa。
时间轴按整段跨度归一化；音频长 ≫ 合成时长时，短合成会被压缩成 0 格。
"""
WIDTH_DEFAULT = 58


def console_gantt(results, width=WIDTH_DEFAULT):
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
            # 时间轴按整段跨度归一化：音频长 ≫ 合成时长时，短合成会被压缩成 0 格
            #（整行空白，看不出合成了）。时长非零却塌缩到 0 格时，强制画至少 1 格。
            # if s1 <= s0 and r["synth_end"] > r["synth_start"]:
            #     row_s[min(s0, width - 1)] = "="
            row_p = [" "] * width
            for k in range(p0, min(p1, width - 1)):
                row_p[k] = "-"
            # if p1 <= p0 and r["audio_dur"] > 0:
            #     row_p[min(p0, width - 1)] = "-"
            tag = "  <- 句%d TTFA=%.3fs" % (i + 1, r["ttfa"]) if i == 0 else ""
            print("  句%d合成 %s%s" % (i + 1, "".join(row_s), tag))
            print("  句%d播放 %s" % (i + 1, "".join(row_p)))
