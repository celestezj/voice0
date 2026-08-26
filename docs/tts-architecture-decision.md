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

- **MeloTTS**：首选（**已实现，实时主力**）。CPU 原生实时、MIT、模型 ~60–100MB、离线、中文"较自然"，无音色克隆、表现力有限。GPU 更快。单句 TTFA <1s 硬指标达标。
- **CosyVoice2 (0.5B)**：音质上限（**已实现，第二后端，可选项**）。原生流式能力 + 3s 音色克隆 + 方言，中文第一梯队；本机 RTX 2070S fp32 下 RTF≈1.2 撑不起实时流式（块间饿死），**播放改整句非流式（`stream=False`，句内无缝）**；代价是 CPU 慢、需 GPU（实测 ~2.9GB VRAM）、依赖链重（LLM + flow matching + vocoder + CUDA）、句首等待 ~5.6–6.2s（=整句合成耗时）达不到 <1s、**0.5B LLM 生成长度有随机性**（transformers 修复后已基本解决，见下节 ⚠️）。
- **MOSS-TTS-Nano (0.1B, 2026-04)**：新候选。纯 CPU 原生流式、支持中文 + 音色克隆、有去 PyTorch 的 ONNX 版（~2× 快）；太新、实战验证少，待实测。
- **排除**：edge-tts / Azure / 火山等**云端**（需求明确要求离线）；纯自回归音频大模型（GPT-4o audio 类，端到端普遍 >1s）。

**实施顺序**：MeloTTS 已实现并跑通测量；CosyVoice2 已实现为第二后端（详见下节 + `docs/README-cosyvoice2.md`）。下一步候选 MOSS-TTS-Nano 待实测。

## CosyVoice2 第二后端（2026-08-26 完成，阶段2 目录重构 + 原生流式后端）

- **目录已重构**：平铺 → `tts/core/`（engine/backend/jobs/audio）+ `tts/melo/` + `tts/cosy/`。后端抽象：`TTSBackend` 协议 + `get_backend(name)` 惰性 import（缺依赖抛 `BackendNotInstalledError`，支持选择性安装——melo-only / cosy-only / both）。
- **`RealtimeTTS` 新增** `backend="melo"|"cosy"`、`voice`（cosy 专属）、`max_speech_ratio`（cosy 专属，见下）。
- **实现**（`tts/cosy/backend.py`）：权重经 HF 镜像落 `.cache/hf`；`CosyVoice2(load_jit/trt/vllm/fp16 全关)` fp32；`voice="default"`（内置 `assets/cosy_default_female.wav` + `add_zero_shot_spk` 缓存）｜`"clone:<wav>:<文本>"`（3s 零样本克隆）；`text_frontend=True`（wetext，无需 pynini）。
- **三个 third_party patch**（vendored，均因 0.5B 精简依赖链）：① Matcha `pylogger.py` `rank_zero_only` 本地化（免 lightning）；② Matcha `utils/__init__.py` 只保留 pylogger import（免 hydra/lightning）；③ `file_utils.py` `load_wav` 改用 soundfile（torchaudio 2.x 强制 torchcodec 未装）。backend 内另有 `_force_cpu_onnx()`（campplus/speech_tokenizer 强制 CPU provider）、`_ensure_wetext_local()`（wetext FST 缓存零联网，防 modelscope 403）、`_silence_tqdm()`。
- **实测（RTX 2070S，非流式播放 stream=False，2026-08-26）**：加载 61.5s/52.7s；句1 TTFA 5.584s（默认）/ 6.244s（克隆，=整句合成耗时）；峰值显存 2.88GB/2.98GB；4 句源文本经 **Whisper-medium 转写内容正确** ✅；句间无缝（interval≈0），总跨度 17.8s/19.2s。达不到 melo <1s 硬指标，如实记录——实时主力仍是 melo。⚠️ 接入初期的旧数字（TTFA≈1.5/1.8s、克隆显存 4.57GB）与修复后 token 级流式（TTFA 2.5/2.9s）均作废——前者是坏管线吐杂音，后者在 2070S 上块间饿死、不可听；音质/克隆效果以修复后的试听为准。
- **⚠️ transformers 版本坑（2026-08-26 修复，本项目最重要的一次排障）**：main env 的 transformers 4.57.6 与 CosyVoice2 不兼容（官方 [issue #1546](https://github.com/FunAudioLLM/CosyVoice/issues/1546)：>4.51.3 即出问题；4.53+ 重写 `Qwen2Model.forward`）→ LLM 产出全错 speech token → **用户实测"一句输入，多次输出、全是杂音、没完没了"**（Whisper 听写为单字重复如"我哭哭哭哭"）。诊断链路：频谱/基频证明是"人声性"噪声而非白噪 → flow/hifigan 在参考音频自身 token/mel 上能还原参考 ✓ → 定位在 LLM 生成的 token → 版本对比锁定 transformers。修复：vendored `transformers 4.51.3 + tokenizers 0.21.1` 于 `.cache/pinned_transformers`（`setup_cosy_pinned.py` 落盘，94M），`tts/cosy/backend.py` import 时注入；melo 用 main env 4.57.6 不受影响。修复后 Whisper(base/small/medium) 听写**内容正确**、生成**有界**。**cosy 与 melo 不能同进程混用**（明确报错不产杂音）。
- **生成长度（修复后已基本解决）**：接入初期测得的"EOS 不可靠/时长随机/短句必冲上限"（当时 EOS 概率 <2%、rank 200–1500）其实是坏 transformers 的产物。pin 4.51.3 后实测生成有界、时长与文本长度成比例——8 字 → 51~78 token（~1.3-1.9s）、44 字 → 232~264（~5.8-6.6s）、76 字 → 413（~10.3s），都远低于 20×上限；仅保留正常采样随机性（±1.5×）。`max_speech_ratio` 降级为**可选安全阀**。详见 `docs/README-cosyvoice2.md` §8。
- **句间停顿为何无解（2026-08-26 fp16 探针）**：非流式修复后句内无缝，但句间停顿 ≈ 合成(N+1) − 播放(N) ≈ 音频差（RTF≈1 下串行合成固有代价，软件改不掉）。**fp16 已实测排除**：`CosyVoice2(fp16=True)` RTF=1.215，反而慢于 fp32（≈0.999-1.0）——0.5B LLM 在独立线程、不在 autocast 作用域内，flow/hifigan 太小在 Turing 上 fp16 开销>收益（Whisper-medium 复核内容仍正确，是被速度否的）；flow 已 `n_timesteps=10` 无余量；vLLM（0.5B 太小）/ 双实例流水线（~5.8GB 显存 + 引擎并发改造）收益不值。**结论：接受 cosy 句间停顿，实时主力仍是 melo；cosy 适合整段/逐句预合成。**

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
- 裸管线验证：`bench/bench_melo.py --device cpu`（无开关）零 perf_counter 埋点、零 debug 输出。

## 实现要点（供实现与排障参考）

- melo 后端 `tts/melo/backend.py` 入口设 `HF_HOME`/`HF_ENDPOINT`/`NLTK_DATA` 重定向到项目内 `.cache/`。
- `preload_weights.py` 一次性预下载 melo 权重：6 个 tokenizer + 670MB BERT + 198MB ZH 权重 + NLTK cmudict/tagger/punkt_tab；`preload_cosy.py` 预下载 cosy 权重（HF + wetext FST）。
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
