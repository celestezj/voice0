# -*- coding: utf-8 -*-
"""Windows SAPI (pyttsx3) TTS case.

产出：audio/1_sapi_huihui.wav（SAPI 原生输出）+ audio/1_sapi_huihui.mp3（ffmpeg 压缩）。
MP3 体积约为 WAV 的 1/5，适合试听与嵌入。
"""
import os
import shutil
import subprocess
import time
import pyttsx3

TEXT = "这是一款将 WiFi 无线信号转化为实时空间感知能力的工具，通过分析人体活动引起的信道状态信息变化，无需摄像头或穿戴设备，即可实时还原人体姿态，并检测心率和呼吸。"
# 本文件位于 sapi/synth_sapi.py → 输出到项目根 audio/（向上取 1 级）
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
os.makedirs(OUT_DIR, exist_ok=True)

wav_path = os.path.join(OUT_DIR, "1_sapi_huihui.wav")
mp3_path = os.path.join(OUT_DIR, "1_sapi_huihui.mp3")

engine = pyttsx3.init()
engine.setProperty("voice", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0")
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

t_start = time.perf_counter()

t0 = time.perf_counter()
engine.save_to_file(TEXT, wav_path)
engine.runAndWait()
t_tts = time.perf_counter() - t0
print("SAPI WAV saved:", wav_path, os.path.getsize(wav_path), "bytes")
print("TTS 合成耗时: %.3f 秒" % t_tts)

# WAV -> MP3（体积小很多，约 1/5）
if shutil.which("ffmpeg"):
    t0 = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
         "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
        check=True,
    )
    t_mp3 = time.perf_counter() - t0
    print("MP3 saved:", mp3_path, os.path.getsize(mp3_path), "bytes")
    print("ffmpeg 转换耗时: %.3f 秒" % t_mp3)
else:
    print("警告: 未找到 ffmpeg，仅生成 WAV")

print("总耗时: %.3f 秒" % (time.perf_counter() - t_start))
