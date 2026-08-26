# -*- coding: utf-8 -*-
"""音频小工具 + 逐句响度归一化（后端无关，全部以 sr 为参数，不持有引擎状态）。

归一化分三档（与引擎 normalize 开关对应）：
  None             原样
  "rms"            static_align：逐句静态活动段 RMS 对齐（治句间音量不齐）
  "agc"            static_align + pause_cap + agc（治句内起伏 + 逗号死寂收音）

设计要点：
- 活动段阈值取**相对句子峰值**的比例（thr = rel_thr × peak），缩放前后峰值等比变化
  → 活动帧集合对增益不变，归一化后活动 RMS 精确等于目标（固定阈值会因缩放跨边界）。
- 各函数可被任一后端复用（melo 整句 / cosy 逐块或整句聚合），不依赖模型。
"""
import numpy as np
import wave

# 逐句响度归一化常量：
# 目标 = 活动段(非静音) RMS -24 dBFS（≈0.0631）；峰值上限 0.95，超则整体下压防爆音。
NORMALIZE_TARGET = 10 ** (-24.0 / 20.0)
NORMALIZE_PEAK_CEIL = 0.95


def save_wav_np(audio, path, samplerate):
    """把 float32 音频数组写成 16bit PCM WAV（无需 soundfile/scipy 依赖）。"""
    a = np.asarray(audio)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if np.abs(a).max() > 1.0:
        a = a / 32767.0
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(samplerate))
        w.writeframes(pcm.tobytes())


def normalize_audio(audio, sr, mode):
    """归一化分发：mode="rms"=逐句静态响度对齐（治句间）；"agc"=静态对齐 +
    句内动态压缩（治句内起伏）+ 短停压缩（治句内超长静音缝）。"""
    a = static_align(audio, sr)
    if mode == "agc":
        a = pause_cap(a, sr)
        a = agc(a, sr)
    return a


def static_align(audio, sr, target=NORMALIZE_TARGET, peak_ceil=NORMALIZE_PEAK_CEIL):
    """逐句静态响度对齐：按"活动段(非静音) RMS"缩放整句，对齐目标响度。

    不用整句 RMS 的原因：VITS 句首句尾常带静音，静音占比句间差异大（实测
    37%~64%），会稀释整句 RMS——静音多的句子被过度放大、少的被压小，语音段
    实际响度仍不齐。只按有语音的帧算 RMS 对齐，才接近人耳感知的语音响度。
    峰值超上限则整体下压防爆音；全静音句跳过。
    """
    a = np.asarray(audio, dtype=np.float32)
    rms_act = active_rms(a, sr)
    if rms_act <= 1e-6:
        return audio                        # 无活动帧：跳过，避免除零/放大噪声
    out = a * (target / rms_act)
    peak = float(np.abs(out).max())
    if peak > peak_ceil:
        out *= peak_ceil / peak
    return out


def pause_cap(audio, sr, max_pause=0.25, rel_thr=1e-3):
    """短停压缩（并进 agc）：把句内超过 max_pause 的"深度静音缝"截短到 max_pause。

    背景：VITS 把标点停顿直接烘焙进单个波形，实测逗号处可生成 ~0.6s 的绝对数字
    静音（-inf dB，远超自然短停顿 ~0.1s），听感是"每逗号之间突然消音没了声音"。
    本方法检测帧 RMS 低于 rel_thr×peak（≈-60dB 相对峰值，只认真静音、不碰正常
    呼吸/短停）的连续段，超过上限的从中间抽掉多余静音（两侧各留 max_pause/2 帧，
    保留原渐入渐出边），静音接静音无咔哒；句首/句尾静音不动，保持句子自然边界。
    """
    a = np.asarray(audio, dtype=np.float32)
    win = max(int(sr * 0.02), 1)
    n = len(a) // win
    if n == 0:
        return audio
    frames = a[:n * win].reshape(n, win)
    rms_f = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(np.abs(a).max())
    if peak <= 1e-12:
        return audio
    deep = rms_f < rel_thr * peak                # 深度静音帧（真静音，非呼吸）
    max_frames = max(int(round(max_pause / 0.02)), 1)

    # 收集连续深度静音段
    runs = []
    start = None
    for i, d in enumerate(deep):
        if d and start is None:
            start = i
        elif not d and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    # 排除句首/句尾静音段（保持句子自然边界，不压缩）
    runs = [r for r in runs if r[0] != 0 and r[1] != n - 1]

    # 超长段：保留两侧 max_pause/2 帧，中间多余静音整段切除
    cuts = []
    for s0, s1 in runs:
        L = s1 - s0 + 1
        if L <= max_frames:
            continue
        head = max_frames // 2
        tail = max_frames - head
        cuts.append(((s0 + head) * win, (s1 - tail + 1) * win))
    if not cuts:
        return audio
    out = a
    for cs, ce in sorted(cuts, reverse=True):    # 从后往前切，避免索引位移
        out = np.concatenate([out[:cs], out[ce:]])
    return out


def agc(audio, sr, target=NORMALIZE_TARGET, peak_ceil=NORMALIZE_PEAK_CEIL):
    """句内动态压缩（AGC）：按 20ms 帧 RMS 包络做逐帧增益，把句内幅度起伏压平到
    目标响度附近。帧间增益线性插值避免抽吸；噪声门防静音/呼吸被过度放大；
    增益限幅（最多压 10dB / 抬 16dB）防过度压缩；峰值超上限整体下压。
    """
    a = np.asarray(audio, dtype=np.float32)
    win = max(int(sr * 0.02), 1)
    n = len(a) // win
    if n == 0:
        return audio
    frames = a[:n * win].reshape(n, win)
    rms_f = np.sqrt(np.mean(frames ** 2, axis=1))
    gate = max(float(rms_f.max()) * 1e-3, 1e-6)   # 噪声门限（相对本句峰值 RMS）
    env = np.maximum(rms_f, gate)
    gain_db = 20.0 * np.log10(target / env)
    gain_db = np.clip(gain_db, -10.0, 16.0)   # 最多压 10dB / 抬 16dB
    gain = 10.0 ** (gain_db / 20.0)
    centers = win // 2 + np.arange(n) * win
    out = a[:n * win] * np.interp(np.arange(n * win), centers, gain)
    if n * win < len(a):
        out = np.concatenate([out, a[n * win:] * float(gain[-1])])
    peak = float(np.abs(out).max())
    if peak > peak_ceil:
        out *= peak_ceil / peak
    return out


def active_rms(audio, sr, win_s=0.02, rel_thr=0.02):
    """按 20ms 帧统计非静音帧的 RMS（能量均值），静音帧不参与。

    关键：活动阈值取**相对句子峰值**的比例（thr = rel_thr × peak），
    缩放前后峰值等比变化 → 活动帧集合对增益不变，归一化后活动 RMS 精确等于目标
    （若用固定阈值，放大/缩小后贴边帧会跨入/跨出活动集，测得值偏离目标）。
    """
    a = np.asarray(audio, dtype=np.float32)
    win = max(int(sr * win_s), 1)
    n = len(a) // win
    if n == 0:
        return 0.0
    frames = a[:n * win].reshape(n, win)
    rms_f = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(np.abs(a).max())
    act = rms_f[rms_f > rel_thr * peak]
    if len(act) == 0:
        return 0.0
    return float(np.sqrt(np.mean(act ** 2)))
