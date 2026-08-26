# -*- coding: utf-8 -*-
"""后端无关 TTS 引擎核心包。

模块划分：
    engine.py   RealtimeTTS 单例 + 常驻双线程 + modes/lifecycle（自 tts_melo.py 抽取）
    jobs.py     Job（submit 返回值，wait/mark_done/canceled/timing）
    audio.py    save_wav_np + 逐句响度归一化（static_align/pause_cap/agc/active_rms）
    backend.py  阶段B起：TTSBackend 协议 + get_backend(name) 惰性加载
"""
