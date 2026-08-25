# voice0 — 离线实时文本转语音（TTS）

把文本实时转换为自然语音的程序，核心指标：**单句 TTFA（首包耗时）< 1s**、**离线运行**（权重缓存后零网络请求）、**声音自然**（非 SAPI 机器音）。

当前实现：**MeloTTS**（见下方选型结论）。流式架构为"句子级分块 + 边合成边播"。

---

## TTS 选型结论（2026-08 定案）

> 完整结论见 `memory/tts-architecture-decision.md`，两者保持一致。

**目标约束**：离线、流式（TTFA + 边合成边播）、平衡 CPU/GPU 速度与音质。

**硬件**：RTX 2070 8GB（Turing sm_75）、NVIDIA 驱动 591.86 / CUDA 13.1；conda 环境 `voice-tts`（克隆自 python3.10，含 torch 2.11.0+cu126）。

| 方案 | 中文音质 | CPU 实时 | GPU | 流式 | 离线 | 部署复杂度 | 许可 |
|---|---|---|---|---|---|---|---|
| **MeloTTS（已采用）** | 较自然 | ✅ 原生 | 更快 | 按句分块 | ✅ | 低（pip） | MIT |
| CosyVoice2 (0.5B) | 很自然 + 音色克隆 + 方言 | ❌ 慢 | ✅ ~4GB | ✅ 原生双流式 | ✅ | 高 | Apache-2.0 |
| MOSS-TTS-Nano (0.1B) | 待测 | ✅ 原生 | 更快 | ✅ 原生 | ✅ | 中 | — |
| ~~edge-tts / Azure 等云端~~ | 自然 | — | — | — | ❌ 需联网 | — | — |

**结论**：MeloTTS 为首选（CPU 实时、离线、简单、中文较自然）；音质上限为 CosyVoice2（需 GPU 常驻）；2026 新候选 MOSS-TTS-Nano 待实测。排除所有云端方案（用户明确要求离线）。**实施顺序：先 MeloTTS，跑通测量后再试下一个。**

---

## 流式架构（"模拟"流式）

MeloTTS（VITS）无原生流式接口——`tts_to_file()` 整句合成完才返回完整音频数组。实时感靠**句子级分块 + 流水线并行**模拟：

```
  文本 → 分句 → 合成线程 ──(有界队列 maxsize=8, 天然背压)──> 播放线程 → sounddevice 写声卡
```

- 合成线程逐句合成入队；播放线程取块即播，两句间无缝。
- **TTFA** = 首帧写进声卡时刻 − `speak()` 入口时刻。
- 队列限长 = 吞吐 vs 延迟的折中（防延迟膨胀）。

### 插桩开关（生产零成本）

| 开关 | 作用 | 生产建议 |
|---|---|---|
| `profile` | 逐句 4 戳计时、TTFA/间隔、控制台甘特图、HTML 时序图、峰值显存 | 关闭（热路径零 `perf_counter`） |
| `debug` | 环境/设备横幅、权重下载明细、"[正在播放 句i/N]" 标注 | 关闭 |

---

## 常驻引擎 API（v2）

模型加载、声卡流、合成/播放线程都是**一次性**的（单例 + 常驻线程），之后任意时刻来文本都零初始化：

- **单例**：任意时刻最多一个 `RealtimeTTS` 实例（重复 `RealtimeTTS(...)` 返回同一个）。
  `device` 变更自动销毁旧实例、重建新设备；`mode` / `speed` 变更**原地切换零重载**。
- **常驻线程**：合成线程 + 播放线程在构造时启动、`close()` 才结束，全程只创建一次。

| API | 阻塞？ | 说明 |
|---|---|---|
| `speak(text)` | 阻塞 | = submit + wait，播完这段才返回（bench 兼容） |
| `submit(text)` | 非阻塞 | 入队立即返回 `Job`；**实时场景用这个** |
| `job.wait()` | 阻塞 | 等该任务播完（或被打断取消），返回逐句时序 |
| `interrupt()` / `stop()` | — | 打断当前正在说的 + 清空排队文本 |
| `close()` | — | 销毁：关线程/关声卡流/清单例；幂等；`with`、`__del__`、atexit 兜底 |

**两种模式**（`mode`，运行期可切）：
- `queue`（默认）：新文本排到当前文本之后，播完再说；
- `bargein`：新文本打断当前播放与排队（`submit` 时自动触发，等价手动 `interrupt()`）。

```python
tts = RealtimeTTS(mode="bargein")   # 常驻引擎，只创建一次
tts.submit("第一句。")                # 非阻塞，立即返回
tts.submit("更重要的话。")            # bargein 模式自动打断上一句
tts.interrupt()                      # 或手动打断
tts.close()                          # 用完了手动销毁
```

> 注意：`del tts` 只删名字引用，单例类属性与常驻线程仍持有对象引用，**不会触发销毁**——请显式 `close()`。

**播放时顺手存 WAV（零额外推理）**：`submit()` / `speak()` 带 `save_wav="out.wav"` 或 `save_chunks_dir="dir/"`，复用流式合成已算好的逐句 audio 直接落盘——**不会把同一段文本再推理一遍**：

- `save_wav`：把全部句子拼成**一个整段 WAV**；
- `save_chunks_dir`：每句各存一份（`句01.wav`、`句02.wav`…）；
- 两者相互独立，可同时传；任务被打断时只存已合成完的句子。
- `speak_to_file(text, wav)` 是另一条**整段非流式**路径（自带一次推理），适合"只落盘、不播放"的场景——不要用它来补存已 `speak()` 过的文本（音频播完不保留，只能重推）。

---

## 环境与运行

```bash
# 1. 克隆虚拟环境（不污染 yolo-gpu 的 python3.10）
conda create -n voice-tts --clone python3.10
D:/anaconda/envs/voice-tts/python.exe -m pip install melotts sounddevice

# 2. 首次运行预下载全部权重（huggingface.co 在本机被墙，脚本自动走 hf-mirror 镜像；一次性联网）
D:/anaconda/envs/voice-tts/python.exe preload_weights.py

# 3. 验收（CPU/GPU 各一轮，含时序图与 README 自动更新）
D:/anaconda/envs/voice-tts/python.exe bench_melo.py --device all --profile --debug

# 裸管线（无插桩，贴近生产）
D:/anaconda/envs/voice-tts/python.exe bench_melo.py --device cuda
```

产出：`audio/2_melo_{device}_{case}.wav`（试听对比）、`reports/bench_timing_{device}.html`（悬停看明细）、`audio/bench_report_{device}.txt`。MP3 压缩依赖 ffmpeg（本机未装，仅产出 WAV）；装了 ffmpeg 后自动追加 `.mp3`。

---

## 验收指标列含义（bench 报告）

bench 的 `profile` 插桩为**每一句**记录 4 个时间戳（合成开始 t_s0、合成完成 t_s1、开始播放 t_p0、播放完成 t_p1），对应流式管线的两个阶段：

```
speak()入口 t_start
   │
   ▼
合成线程 ──(合成完 audio 数组)──> 队列 ──> 播放线程 ──> 声卡
   │                          │
  t_s0→t_s1                  t_p0→t_p1
```

- **合成ms** = `t_s1 − t_s0`：这一句从开始合成到音频就绪的耗时（纯模型推理，不含排队）。
- **等待ms** = `t_p0 − t_s1`：音频合成完到真正开播之间在队列里的排队时间。合成远快于播放（GPU 0.15s 合成 vs ~2s 音频），后几句会提前做好、在队列里等前一句播完，所以等待列随句号递增。
- **TTFA_s** = `t_p0 − t_start`：从 `speak()` 入口到该句开始播放的累计时间。**只有句1 的 TTFA 是验收指标**（<1s 即达标）；句2/3 的 TTFA 包含前面所有句子的合成+播放+等待，是累计值，不是新指标。
- **间隔ms** = `当前句 t_p0 − (上一句 t_p0 + 上一句音频时长)`：上一句播完时刻与当前句开播时刻之差。≈0 或小幅正数为无缝接上；**负数为当前句在上一句音频没播完时就已开始**（靠声卡缓冲衔接），表示句间零停顿、无缝。

### audio 数组是什么（合成产物）

`_synth()` 返回的 `audio` 是一段**原始波形**——1 维 numpy 数组（`float32`），每个元素 = 声音在该瞬间的振幅，范围约 **[-1, 1]**：

```python
array([-0.0012, -0.0008, 0.0031, ..., 0.0005], dtype=float32)
 ↑第0个采样点   ↑第1个   ↑第2个    ...    ↑最后1个
```

- **长度 = 采样点数**。采样率 44.1kHz（`_sr=44100`，每秒 44100 个点），所以 `音频时长 = len(audio) / 44100`——即 bench 表「音频s」列的来源（句1 的 1.79s ≈ 79000 个点）。
- **单声道**：一条振幅曲线随时间变化。

**数字 → 声音**：播放 = 按 44100 个/秒把数组送给声卡，声卡 DAC 把数字转成电压驱动喇叭；写 WAV = 把浮点值 ×32767 转成 16bit PCM（`save_wav_np`）。

**为什么叫「数组」而不落盘**：`tts_to_file(output_path=None)` 走内存路径直接返回数组——合成线程拿到立即入队、播放线程立即送声卡，不用等整个 WAV 写完再播，这是流式不卡顿的前提。

### 实测走一遍（GPU 短句）

| 句 | 合成ms | 等待ms | TTFA_s | 间隔ms | 说明 |
|---|---|---|---|---|---|
| 1 | 184 | 3 | 0.189 | — | 合成完几乎秒播，0.189s 开播 = 验收 TTFA ✅ |
| 2 | 150 | 1624 | 1.960 | -20 | 句1 播放期间后台合成，做完在队列等 1624ms；句1 播完瞬间接上，间隔 -20ms（无缝） |

> **读表口诀**：句1 的 TTFA 看延迟指标，等待列看流水线饱和度，间隔列看句间是否无缝，合成列看纯推理速度。GPU 上合成快导致队列堆积（等待大）但句间无缝（间隔≈0/负），正是"边合成边播"设计生效的证据。

---

## 验收结果

<!-- bench:start -->
### CPU（device=cpu）

| 用例 | 句数 | 首句TTFA(s) | 平均合成(ms/句) | 最大等待(ms) | 总跨度(s) |
|---|---|---|---|---|---|
| 短 | 2 | 0.927 | 970 | 567 | 4.45 |
| 中 | 3 | 0.999 | 1125 | 1571 | 7.17 |
| 长 | 3 | 2.594 | 2637 | 5269 | 17.48 |

- 交互式时序图（悬停看明细）: `reports/bench_timing_cpu.html`
- 逐句原始指标: `audio/bench_report_cpu.txt`

### CUDA（device=cuda）
显卡：NVIDIA GeForce RTX 2070 SUPER ｜ 峰值显存：1.088 GB

| 用例 | 句数 | 首句TTFA(s) | 平均合成(ms/句) | 最大等待(ms) | 总跨度(s) |
|---|---|---|---|---|---|
| 短 | 2 | 0.203 | 201 | 1527 | 3.90 |
| 中 | 3 | 0.162 | 156 | 3772 | 6.50 |
| 长 | 3 | 0.361 | 333 | 10050 | 15.27 |

- 交互式时序图（悬停看明细）: `reports/bench_timing_cuda.html`
- 逐句原始指标: `audio/bench_report_cuda.txt`

### CPU / GPU 对比结论（bench 自动生成）

| 指标 | CPU | GPU |
|---|---|---|
| 首句 TTFA | 0.927s | 0.203s |
| TTFA 达标(<1s) | ✅ | ✅ |
| GPU 相对 CPU 提速 | — | 4.6× |
| GPU 峰值显存 | — | 1.088 GB |

**结论**：GPU 首句 TTFA=0.203s 达标<1s；CPU 首句 TTFA=0.927s 达标<1s。流式重叠已验证：播放段与下一句合成段重叠。

> 生成时间：2026-08-25 17:08:57（每次 bench 运行自动刷新）
<!-- bench:end -->

---

## 目录树

<!-- tree:start -->
```
voice0/
├── audio/
│   ├── chunks/
│   │   ├── cpu/
│   │   │   ├── long/
│   │   │   │   ├── 句01.wav
│   │   │   │   ├── 句02.wav
│   │   │   │   └── 句03.wav
│   │   │   ├── mid/
│   │   │   │   ├── 句01.wav
│   │   │   │   ├── 句02.wav
│   │   │   │   └── 句03.wav
│   │   │   └── short/
│   │   │       ├── 句01.wav
│   │   │       └── 句02.wav
│   │   └── cuda/
│   │       ├── long/
│   │       │   ├── 句01.wav
│   │       │   ├── 句02.wav
│   │       │   └── 句03.wav
│   │       ├── mid/
│   │       │   ├── 句01.wav
│   │       │   ├── 句02.wav
│   │       │   └── 句03.wav
│   │       └── short/
│   │           ├── 句01.wav
│   │           └── 句02.wav
│   ├── 2_melo_cpu_long.wav
│   ├── 2_melo_cpu_mid.wav
│   ├── 2_melo_cpu_short.wav
│   ├── 2_melo_cuda_long.wav
│   ├── 2_melo_cuda_mid.wav
│   ├── 2_melo_cuda_short.wav
│   ├── bench_report_cpu.txt
│   ├── bench_report_cuda.txt
│   ├── smoke_cpu.wav
│   └── smoke_cuda.wav
├── reports/
│   ├── bench_timing_cpu.html
│   └── bench_timing_cuda.html
├── .gitignore
├── README.md
├── _bench_verify_log.txt
├── _test_shutdown.py
├── _test_shutdown_log.txt
├── bench_melo.py
├── preload_weights.py
├── synth_sapi.py
└── tts_melo.py
```
<!-- tree:end -->
