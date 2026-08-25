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

## 验收结果

<!-- bench:start -->
### CPU（device=cpu）

| 用例 | 句数 | 首句TTFA(s) | 平均合成(ms/句) | 最大等待(ms) | 总跨度(s) |
|---|---|---|---|---|---|
| 短 | 2 | 0.877 | 943 | 530 | 4.36 |
| 中 | 3 | 1.063 | 1130 | 1791 | 7.39 |
| 长 | 3 | 2.664 | 2644 | 5446 | 17.65 |

- 交互式时序图（悬停看明细）: `reports/bench_timing_cpu.html`
- 逐句原始指标: `audio/bench_report_cpu.txt`

### CUDA（device=cuda）
显卡：NVIDIA GeForce RTX 2070 SUPER ｜ 峰值显存：1.088 GB

| 用例 | 句数 | 首句TTFA(s) | 平均合成(ms/句) | 最大等待(ms) | 总跨度(s) |
|---|---|---|---|---|---|
| 短 | 2 | 0.189 | 167 | 1624 | 3.91 |
| 中 | 3 | 0.161 | 151 | 4002 | 6.71 |
| 长 | 3 | 0.356 | 330 | 9917 | 15.22 |

- 交互式时序图（悬停看明细）: `reports/bench_timing_cuda.html`
- 逐句原始指标: `audio/bench_report_cuda.txt`

### CPU / GPU 对比结论（bench 自动生成）

| 指标 | CPU | GPU |
|---|---|---|
| 首句 TTFA | 0.877s | 0.189s |
| TTFA 达标(<1s) | ✅ | ✅ |
| GPU 相对 CPU 提速 | — | 4.7× |
| GPU 峰值显存 | — | 1.088 GB |

**结论**：GPU 首句 TTFA=0.189s 达标<1s；CPU 首句 TTFA=0.877s 达标<1s。流式重叠已验证：播放段与下一句合成段重叠。

> 生成时间：2026-08-25 13:19:52（每次 bench 运行自动刷新）
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
├── bench_melo.py
├── preload_weights.py
├── synth_sapi.py
└── tts_melo.py
```
<!-- tree:end -->
