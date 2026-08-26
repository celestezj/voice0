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

        def _end(r):
            """句结束时刻：合成末端 与（若播过）播放末端 取大。"""
            e = r["synth_end"]
            if r.get("play_start") is not None:
                e = max(e, r["play_start"] + r["audio_dur"])
            return e

        t_end = max(_end(r) for r in timing)
        span = t_end - t0

        def x(t):
            return int((t - t0) / span * (width - 1)) if span > 0 else 0

        print("\n[%s] 跨度=%.2fs" % (res["label"], span))
        for r in timing:
            i = r["idx"]
            s0, s1 = x(r["synth_start"]), x(r["synth_end"])
            row_s = [" "] * width
            for k in range(s0, min(s1, width - 1)):
                row_s[k] = "="
            # 时间轴按整段跨度归一化：音频长 ≫ 合成时长时，短合成会被压缩成 0 格
            #（整行空白，看不出合成了）。时长非零却塌缩到 0 格时，强制画至少 1 格。
            # if s1 <= s0 and r["synth_end"] > r["synth_start"]:
            #     row_s[min(s0, width - 1)] = "="
            row_p = [" "] * width
            # 被打断/未播过的任务 play_start 可能为 None：跳过播放条（不崩）
            if r.get("play_start") is not None:
                p0, p1 = x(r["play_start"]), x(r["play_start"] + r["audio_dur"])
                for k in range(p0, min(p1, width - 1)):
                    row_p[k] = "-"
            # if p1 <= p0 and r["audio_dur"] > 0:
            #     row_p[min(p0, width - 1)] = "-"
            # 打断的任务未派生 ttfa（为 None）：只对已派生的句1 打标签
            tag = ("  <- 句%d TTFA=%.3fs" % (i + 1, r["ttfa"])) \
                if (i == 0 and r.get("ttfa") is not None) else ""
            print("  句%d合成 %s%s" % (i + 1, "".join(row_s), tag))
            print("  句%d播放 %s" % (i + 1, "".join(row_p)))
