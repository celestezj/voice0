# TTS 实时方案选型（2026-08 定案）

> 本文件是项目的**架构决策记录（ADR）**：为什么选 MeloTTS、硬件与环境的来龙去脉、以及必须遵守的验收约定。
> 实现细节见 README（常驻引擎 v2 设计文档、环境与复现）。

## 目标与核心架构原则

- **目标**：实时文本→音频，单句 TTFA <1s，**离线运行**（最终方案不联网），声音自然（非 SAPI 机器音）。
- **核心架构原则**：TTFA（首包耗时）+ 流式播放——按句分块、边合成边播，不等整段。这是实现方案的硬性基础。

## 硬件与开发环境

- GPU：**RTX 2070 SUPER 8GB**（Turing sm_75），NVIDIA 驱动 591.86 / CUDA 13.1，总显存 8.59GB。
- conda 环境 `voice-tts`（D:\anaconda\envs\voice-tts，**克隆自 python3.10**，含 torch 2.11.0+cu126）——本项目专用；克隆是为了不污染 yolo-gpu 正在用的 python3.10。
- **新设备从零复现以 README「环境与复现（新设备一键复现）」为准**（含实测锁定版本表、conda 步骤、常见坑）。
- 本机网络：huggingface.co / raw.githubusercontent.com 直连被墙或极慢；**hf-mirror.com 与 ghfast.top 代理可用**（仅首次下载权重用）。
- 环境修正（仅克隆大环境路径会遇到）：卸载克隆自带的 jax/jaxlib/ml_dtypes（与 numpy 2.2.6 不兼容，会崩 transformers）；setuptools 固定 80.9.0（81+ 移除 pkg_resources，jieba 需要）。

## 选型结论（2026 版开源 TTS 横向对比，来源见文末）

- **MeloTTS**：首选。CPU 原生实时、MIT、模型 ~60–100MB、离线、中文"较自然"，无音色克隆、表现力有限。GPU 更快。
- **CosyVoice2 (0.5B)**：音质上限。原生流式 + 3s 音色克隆 + 方言，中文第一梯队；代价是 CPU 慢、需 GPU（~4GB VRAM）、依赖链重（LLM + flow matching + vocoder + CUDA）。
- **MOSS-TTS-Nano (0.1B, 2026-04)**：新候选。纯 CPU 原生流式、支持中文 + 音色克隆、有去 PyTorch 的 ONNX 版（~2× 快）；太新、实战验证少，待实测。
- **排除**：edge-tts / Azure / 火山等**云端**（需求明确要求离线）；纯自回归音频大模型（GPT-4o audio 类，端到端普遍 >1s）。

**实施顺序**：先实现 MeloTTS，跑通并测量后，再尝试下一个（MOSS-TTS-Nano / CosyVoice2）。

## 插桩/验收约定（本项目强约定）

- 所有生产无关插桩走开关：`profile`（时序分析）/ `debug`（详细日志）；两者都关 = 裸管线，热路径零 `perf_counter` 埋点（每个计时调用点必须用 `if self._profile:` 守卫）。
- CPU 与 GPU 各跑一轮验收，各出一张时序图（控制台 ASCII 简易版 + `reports/bench_timing_{device}.html` 交互版，悬停句块显示完整文本与耗时明细）。
- bench 结果自动回写 README.md 的验收结果区间（含 CPU/GPU 对比结论）与目录树区间；所有文件都在本项目路径下（HF 权重缓存经 `HF_HOME` 重定向到 `.cache/hf`）。
- 运行时零网络请求（仅首次下载权重可联网）。

## 验收实测（2026-08-25，首轮，CPU/GPU 各一轮）

- **GPU（RTX 2070 SUPER）**：短句首句 TTFA=0.189s、中 0.161s、长 0.356s，全部 <1s ✅；峰值显存 1.088GB。
- **CPU**：短 0.877s ✅、中 1.063s、长 2.664s（首句合成 ~0.9-2.7s，BERT 前向为瓶颈）。
- GPU 相对 CPU 首句提速 **4.7×**；流式重叠已验证（播放段与下一句合成段重叠，句间无缝）。
- **最新数值以 README「验收结果」区为准**（每次 bench 运行自动刷新）。
- 裸管线验证：`bench_melo.py --device cpu`（无开关）零 perf_counter 埋点、零 debug 输出。

## 实现要点（供实现与排障参考）

- `tts_melo.py` 入口设 `HF_HOME`/`HF_ENDPOINT`/`NLTK_DATA` 重定向到项目内 `.cache/`。
- `preload_weights.py` 一次性预下载：6 个 tokenizer + 670MB BERT + 198MB ZH 权重 + NLTK cmudict/tagger/punkt_tab。
- bench `--device all` 按设备拆子进程（MeloTTS BERT 为模块级单例，CPU/GPU 同进程会设备不匹配）。
- mp3 需 ffmpeg（本机未装，仅产出 WAV）。
- 常驻引擎 v2 的完整设计（单例、常驻线程、queue/bargein 模式、`_gen` 抢占、生命周期、timing schema、线程安全坑）见 **README「常驻引擎 v2」节**。

## 决策依据与执行约定

- **为什么选离线流式方案**：需求确认最终方案必须离线、以流式播放为架构基础、追求"CPU 也快 + GPU 更快 + 效果好"的平衡。
- **执行约定**：后续 TTS 相关实现与选型以本结论为依据；评测必须 CPU 与 GPU 双测 TTFA / RTF 并试听，结论实时回写 README；选型结论同时记录在 README（对外）与本文件（依据来源）。

## 来源

- https://blog.csdn.net/w776341482/article/details/161896379 （2026 开源 TTS 横向对比）
- https://www.cnblogs.com/sensorsen/p/21367537 （六款主流语音模型实测）
- https://ai.atomgit.com/OpenMOSS/MOSS-TTS-Nano-100M
- https://wener.tech/notes/ai/model/tts/awesome
