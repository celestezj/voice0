# CosyVoice2-0.5B 后端（voice0 第二引擎）

> 定位：与 MeloTTS **互补**的第二后端。melotts 只有一种中文女声、无音色克隆；cosy 提供 **中文第一梯队音质 + 任意 3s 参考音频零样本克隆**。代价是句首等待 ~5.6-6.2s（=整句合成耗时，达不到 melo <1s 硬指标）、需 GPU 常驻、且 **0.5B LLM 生成长度有随机性**（见 §8）。实时主力仍是 melo，cosy 是"音质 + 克隆"可选项。
>
> 后端实现：`tts/cosy/backend.py`（`CosyBackend`，选择性安装，缺依赖时报 `BackendNotInstalledError` 带安装提示）。

> ⚠️ **transformers 版本修复（2026-08-26，务必先读）**：main env 的 transformers **4.57.6 与 CosyVoice2 不兼容**。官方 [issue #1546](https://github.com/FunAudioLLM/CosyVoice/issues/1546)：transformers 高于 **4.51.3** 就出问题（4.53+ 重写了 `Qwen2Model.forward` 的 attention-mask / hidden-state 输出逻辑，Qwen2LM 调用路径不再兼容）——LLM 产出**全错的 speech token**，表现就是**"输入一句 → 多次输出、全是杂音、没完没了"**（Whisper 听写为单字重复如"我哭哭哭哭"，非人声）。**对策**：本仓库自带 vendored `transformers==4.51.3 + tokenizers==0.21.1`（`.cache/pinned_transformers`，约 94M，`setup_cosy_pinned.py` 一键落盘），`tts/cosy/backend.py` 在 import 时自动注入 sys.path；melo 等仍用 main env 版本，互不影响。修复后 Whisper(base/small/medium) 听写**内容正确**（"今天天气真不错。"→"今天天氣真不錯!"；长句逐字还原，仅公司名/地名的同音字由 Whisper 自选）。**cosy 与 melo 不能在同一进程混用**（transformers 版本冲突，backend 会明确报错，不会默默产杂音）。

---

## 1. 与 MeloTTS 的定位差异

| 维度 | MeloTTS | CosyVoice2-0.5B |
|---|---|---|
| 音质 | 较自然 | 更自然（LLM + flow matching + HiFi-GAN） |
| 音色 | 固定中文女声 | **默认女声 + 任意 3s 参考音频克隆** |
| 采样率 | 44.1kHz | 24kHz |
| 流式 | 句子级分块（模拟） | **原生 token 级流式**（逐块 ~1s） |
| 句1 TTFA（实测，非流式） | ~0.2s（GPU）/ ~0.9s（CPU） | **~5.6s（默认）/ ~6.2s（克隆）** |
| 设备 | CPU 原生实时 | CPU 极慢，**需 GPU**（~3-4.6GB 显存） |
| 生成长度 | 稳定 | **随机不可控**（见下） |
| 依赖 | pip 轻量 | 第三方仓库 + 重依赖链 |
| 许可 | MIT | Apache-2.0（可商用） |
| 适用 | 实时对话主力 | 整段生成 / 换音色 |

**一句话**：实时 <1s 用 melo；要音质/换音色/录旁白用 cosy（整段生成 + 试听，必要时重生成）。

---

## 2. 安装（新设备，可整段复制）

> 前置：已按根 README 建好 `voice-tts` 环境（torch 2.11.0+cu126 / python 3.10）。cosy 与 melo 相互独立，**不装也不影响 melo 使用**。

```bash
# 0. 项目根目录
cd voice0

# 1. clone 官方仓库（--recursive 拉 Matcha-TTS 子模块）+ 依赖
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
#    （github 被墙时前缀 https://ghfast.top/）

# 2. 推理期不需要训练框架。绕开 hydra/lightning（见第 7 节 patches），
#    只补真正缺的推理依赖：
pip install conformer==0.3.2 diffusers==0.29.0 pyarrow pyworld soundfile
#    cosyvoice.dataset.processor import pyarrow；flow 的 decoder 用 conformer；
#    diffusers 是 flow 里 scheduler 依赖；soundfile 替代 torchaudio 读 wav。

# 3. 预下载权重（HF 官方镜像 → .cache/hf；另拉 wetext 文本前端 FST）
python preload_cosy.py

# 3b. 落盘 cosy 专属 transformers 4.51.3（与 main env 的 4.57.6 不兼容，官方 issue #1546）
#     约 94M 到 .cache/pinned_transformers；backend 启动时自动注入，melo 不受影响
python setup_cosy_pinned.py

# 4. 验收（默认音色 + 3s 克隆各一轮，自动产出报告/时序图）
python bench/bench_cosy.py --device cuda --profile
```

### 复现环境的实测版本（voice-tts 环境）

| 组件 | 版本 | 说明 |
|---|---|---|
| torch / torchaudio | 2.11.0+cu126 | 与根 README 一致 |
| conformer | 0.3.2 | flow decoder 依赖 |
| diffusers | 0.29.0 | flow scheduler 依赖 |
| pyarrow / pyworld | 最新 | dataset.processor / 音高特征 |
| soundfile | 最新 | 替代 torchaudio IO（patch ③） |
| onnxruntime | GPU 版已装 | campplus / speech_tokenizer 前端推理 |
| **transformers（cosy 专用，vendored）** | **4.51.3** | `.cache/pinned_transformers`（`setup_cosy_pinned.py` 落盘，**不是** main env 的 4.57.6；4.57.6 会产出杂音，issue #1546） |
| **tokenizers（cosy 专用，vendored）** | **0.21.1** | 随 transformers pin 一并落盘（4.51.3 要求 `>=0.21,<0.22`） |

---

## 3. 权重与缓存

- **模型权重**：`FunAudioLLM/CosyVoice2-0.5B`（约 2.4GB：`llm.pt` 1.88G + `flow.pt` 0.43G + `hift.pt` 0.08G + onnx + `CosyVoice-BlankEN` Qwen2 tokenizer）。
- **下载**：走 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HOME=.cache/hf`（与 melo 同套 env 前置），`preload_cosy.py` 一次性拉齐，之后运行期零网络。
- **wetext 文本前端 FST**（中文数字/单位归一化，`text_frontend=True` 必需）：`preload_cosy.py` 经 modelscope 拉 `pengzhendong/wetext`。backend 已 patch——本地缓存存在时**零联网**注入（见第 7 节），撞限流 403 不再降级。
- **`third_party/CosyVoice`** 已 gitignore，不进版本库；重装设备后 clone + preload 即可重建。

---

## 4. 音色配置（`voice`）

| `voice` | 含义 |
|---|---|
| `"default"` | 内置默认女声：`assets/cosy_default_female.wav` + 内置转写文本 `"希望你以后能够做的比我还好呦。"`，经 `add_zero_shot_spk` 缓存 speaker id，每句免重提 prompt |
| `"clone:<参考wav>:<转写文本>"` | 任意 3s 参考音频零样本克隆。`<转写文本>` 是该音频的实际内容，必须准确 |
| 其它 | `ValueError` |

注意：
- Windows 盘符冒号会破坏解析——实现用 `split(":", 1)` 切 `clone:` 前缀、再对剩余 `rpartition` 分 wav/文本，`E:` 盘路径可用。
- 参考音频建议 16kHz mono、3-10s。内置默认音色即是从官方 demo 女声提取的 16k 参考。
- 换音色 = 换 `voice` 参数 → 引擎按 `(device, backend, voice)` 判定重建。

```python
from tts import RealtimeTTS
tts = RealtimeTTS(device="cuda", backend="cosy",
                  voice="clone:<参考wav的绝对路径>:我的声音转写文本")
tts.speak_to_file("用我的声音说这句话。", "audio/mine.wav")
tts.close()
```

---

## 5. API 与用法

引擎接口与 melo 完全一致（`tts/core/engine.py`），只多 `backend` / `voice` / `max_speech_ratio` / `stream` 参数：

| 用法 | 代码 |
|---|---|
| 默认播放（整句合成后播放，推荐） | `RealtimeTTS(device="cuda", backend="cosy", voice="default")`（`stream=False` 默认） |
| 3s 克隆 | `RealtimeTTS(device="cuda", backend="cosy", voice="clone:<wav>:<文本>")` |
| 收紧生成上限 | `RealtimeTTS(..., max_speech_ratio=8)`（默认 None=模型行为） |
| 恢复原生 token 级流式 | `RealtimeTTS(..., stream=True)`（首包更快，但本机 RTF>1 会块间停顿，不推荐） |
| 整段生成（cosy 主用途） | `tts.speak_to_file(text, "out.wav")`（非流式，内部聚合） |
| 播放 | `tts.speak(text)` / `tts.submit(text)`（每句合成完一次性播放，句内无缝） |
| 命令行 demo | `python speak_example.py`（启动菜单选 cosy + 音色 + 播放方式） |

**为什么默认非流式（2026-08-26 改）**：本机 2070S fp32 下 cosy RTF≈1.1-1.3，**合成比实时播放还慢**——token 级流式逐块 yield 播放时，上一块播完、下一块还没合成出来（块间饿死），听起来就是"字间戛然而止/明显停顿"；且每个 yield 是独立 flow→hifigan 解码，块边界有声学接缝（再叠加块级 normalize 的响度跳变）。改 `stream=False`（`inference_zero_shot` 非流式）后：每句一次 LLM→flow→hifigan 解码、单块播放，**句内零拼接缝**；`speed` 参数也终于生效（流式下被忽略）。代价是句首等待 = 整句合成耗时（约 1.2× 音频时长），句间停顿即下一句的整句合成时间。**实时 <1s 仍由 melo 承担**；cosy 适合"整段生成 + 试听 + 必要时重生成"。

`text_frontend` 后端默认 `True`（走 wetext 中文归一化：数字/单位/标点规范化）。改 `False` 可更贴官方 demo 音色，但短句/数字文本会更不稳，一般不推荐。

### 5.5 代码使用案例（完整可跑）

> cosy 与 melo **不能同进程混用**（后端会在 import 时明确报错）。下面示例只碰 cosy 路径；
> 首次构造会加载模型 ~60s（二次进程 OS 缓存已热会快不少），属正常现象。

```python
# -*- coding: utf-8 -*-
"""CosyVoice2 后端完整使用案例。先 conda activate voice-tts，再：
   python examples/cosy_demo.py   # 或直接整段贴进脚本
"""
from tts import RealtimeTTS

# 1) 默认音色 → 整段生成文件（cosy 主用途：非流式一次解码、句内无缝）
with RealtimeTTS(device="cuda", backend="cosy", voice="default") as tts:
    tts.speak_to_file("今天天气真不错，适合出门走走。", "audio/cosy_default.wav")

# 2) 3s 音色克隆 → voice = "clone:<参考wav>:<该音频的转写文本>"
with RealtimeTTS(device="cuda", backend="cosy",
                 voice="clone:<参考wav的绝对路径>:我的声音转写文本") as tts:
    tts.speak_to_file("用我的声音说这句话。", "audio/cosy_mine.wav")

# 3) 实时播放：submit 非阻塞入队 / speak 阻塞播完（每句整句合成后播放，句内无缝）
tts = RealtimeTTS(device="cuda", backend="cosy", voice="default", mode="queue")
tts.submit("第一句话。")          # 入队即返回
tts.submit("第二句话。")          # 排在第一句之后
tts.speak("阻塞播完这句再返回。") # 同队列，会把前面排队的先播完
tts.close()                       # 常驻单例，必须显式 close（del 不保证生效）

# 4) 进阶：收紧 LLM 生成上限 + 播放同时留档
tts = RealtimeTTS(device="cuda", backend="cosy", voice="default",
                  max_speech_ratio=8, normalize="rms")
tts.speak("留档这段。", save_wav="audio/full.wav", save_chunks_dir="audio/chunks/")
tts.close()

# 5) 流式 vs 非流式（stream 参数）
with RealtimeTTS(device="cuda", backend="cosy", voice="default") as tts:          # 非流式（默认 stream=False）
    tts.speak("非流式：每句一次解码、单块播放，句内零拼接缝；首包要等整句合成完。")

with RealtimeTTS(device="cuda", backend="cosy", voice="default", stream=True) as tts:  # 原生 token 级流式
    tts.speak("流式：逐 token 边合成边播，首包更快；但本机 2070S fp32 RTF>1 会块间停顿，不推荐。")

# 说明：stream 不是运行期开关（改它走"销毁重建"，见 §5 API 表）；stream=True 时 speed 参数被忽略。
```

---

## 6. 验收实测（RTX 2070 SUPER 8GB，2026-08-26）

```
CosyVoice2 bench: device=cuda（bench/bench_cosy.py --device cuda，非流式播放 stream=False）
文本：这是一款将 WiFi 无线信号转化为实时空间感知能力的工具。通过分析人体活动引起的信道状态信息变化。无需摄像头或穿戴设备。即可实时还原人体姿态。
[默认音色] （加载61.5s, 峰值显存 2.88GB）
  句1 TTFA=5.584s   音频 5.76s
  句2 TTFA=11.095s  音频 3.88s
  句3 TTFA=14.975s  音频 2.80s
  句4 TTFA=17.795s  音频 3.84s
[3s克隆]   （加载52.7s, 峰值显存 2.98GB）
  句1 TTFA=6.244s   音频 5.76s
  句2 TTFA=11.755s  音频 3.88s
  句3 TTFA=15.635s  音频 2.80s
  句4 TTFA=19.247s  音频 3.84s
```

- **模型加载 61.5s（默认）/ 52.7s（克隆）**（fp32 纯 torch，jit/trt/vllm 全关；二次进程 OS 缓存已热，会快不少）。
- **句1 TTFA 5.584s（默认）/ 6.244s（克隆）= 整句合成耗时**（非流式 `stream=False`：每句一次解码后播放）。达不到 melo <1s，如实记录。此值为修复后经**完整引擎链**（`RealtimeTTS(backend="cosy")` → `speak()`）实测，内容为真实完整语音。
- **峰值显存 2.88GB（默认）/ 2.98GB（克隆）**——2070S 8GB 可跑。
- **Whisper-medium 转写验收（内容正确）**：两个 WAV 均完整还原全部 4 句源文本（"型号/信号"等个别字是 Whisper 的同音自选，非模型漏字）；生成有界、每句音频时长与文本长度成比例。
- **句间无缝**：interval ≈ 0/负（句与句首尾相接，无饿死停顿）；总跨度 17.8s/19.2s，比修复前 token 级流式（21.6s/23.0s）还快——因为没有了块间卡顿。每句音频时长有正常采样随机性（±1.5×）。
- ⚠️ 本节为**修复后非流式**数据。接入初期的旧数字（TTFA 1.5/1.8s、克隆显存 4.57GB、音频 0.56s/16.00s）全部作废——那是坏 transformers（4.57.6）在吐杂音、冲生成上限时的产物（见文首 ⚠️）；修复后 token 级流式的数字（TTFA 2.5/2.9s、总跨度 21.6/23.0s）也已被本表的非流式替代（流式在 2070S 上块间饿死、不可听）。音质/克隆效果以修复后的试听为准。

### 句间停顿为何无解（2026-08-26 fp16 探针）

非流式修复后**句内无缝**，但句间仍有停顿：句 N+1 的音频必须等它**整句合成完**才开始播，
`句间停顿 ≈ 合成(N+1) − 播放(N) ≈ 音频(N+1) − 音频(N)`（RTF≈1 时）。这是**串行合成的固有代价**，
软件层面改不掉——除非换更快的推理后端或做并行流水线。

已实测排除的加速路线：

- **fp16 推理（排除）**：`CosyVoice2(fp16=True)` 实测 **RTF=1.215**（21.04s 音频合成耗时 25.56s），
  反而**慢于 fp32（RTF≈0.999-1.0）**。原因：0.5B 的 LLM 跑在独立线程、不在
  `torch.cuda.amp.autocast` 作用域内（吃不到 fp16 加速）；flow/hifigan 又很小，Turing 卡上
  fp16 的前向开销 > 收益。Whisper-medium 复核 fp16 输出内容仍正确——是**速度更慢**被否，不是质量问题。
- **flow 步数（无余量）**：已是 `n_timesteps=10`，再降音质崩。
- **vLLM（收益存疑）**：0.5B LLM 太小，vLLM 的加速收益有限，投入不值。
- **双实例流水线（昂贵）**：两个 cosy 模型实例交替合成，可把句间停顿藏进下一句合成；
  但显存 ~5.8GB（≈2.88×2）+ 引擎并发改造，投入产出比差。

**结论：接受 cosy 的句间停顿。** 实时主力仍是 melo（<1s TTFA 硬指标）；cosy 是"音质 + 克隆"路径，
适合整段离线合成 / 逐句预合成的场景，实时逐句对话不推荐。

### RTF>1 的瓶颈分析：算力，不是显存（2026-08-26）

一句话：**不是 8GB 显存太小，是中端卡（Turing sm_75）在 fp32 串行解码下的算力/延迟不够。**

- **显存排除**：实测峰值 2.88GB（默认）/ 2.98GB（克隆），8GB 卡只用 ~37%，远未到瓶颈。
  显存不够的表现是 OOM / 疯狂换页，不是"慢但正确、数字稳定"。
- **瓶颈① 边际速率 ~0.82 RTF = LLM 串行解码**：Qwen2-0.5B 是唯一不可并行的环节，每句串行解
  ~100-400 个 speech token（~40 token/秒音频），fp32 下 2070S 约 40-50 tok/s。解码是**小 GEMM
  串行**、以启动延迟为主，GPU 实际利用率 <1%（峰值 fp32 ~9 TFLOPS 远用不满）——是延迟/串行依赖
  瓶颈，不是吞吐瓶颈。
- **瓶颈② 每句固定开销 ~0.86s**（prompt prefill ~102 token + flow/hifigan 基础 + 文本前端）：
  `RTF ≈ 0.82 + 0.86/音频时长`——2-4s 短句必然 >1（RTF 1.1-1.3）；要 RTF<1 需句子 >~4.8s。

**Turing 卡的加速路全堵死**：fp16 tensor core 只有 fp32 的 2x 且小 GEMM 喂不饱（实测 fp16 反而
更慢，见上）；bf16 在这代无 tensor core 支持（vLLM 的 bf16 路径同样走不通）。软件开关无解。

> ⚠️ **换卡建议**：**cosy 请用算力更强的显卡**（更大 fp32 吞吐，Ampere+ 带 bf16 tensor core）。
> 在 2070S 这类中端卡上，cosy 整句合成 RTF≈1、句首等待 ~5-6s，**逐句实时对话会明显卡顿**；
> 实时 <1s 的交互场景请用 melo。cosy 定位离线整段预合成 / 试听，该场景下 RTF≈1 可接受。

---

## 7. 对第三方仓库的补丁（three third-party patches）

> 全部在 vendored 的 `third_party/CosyVoice/` 内（gitignore 不入库），换版本/重 clone 后需重打。**这是让 0.5B 在精简依赖下可推理的关键。**

| # | 文件 | 改了什么 | 为什么 |
|---|---|---|---|
| ① | `third_party/Matcha-TTS/matcha/utils/pylogger.py` | `from lightning.pytorch.utilities import rank_zero_only` → 本地 `def rank_zero_only(fn): return fn` | 免装 lightning/torchmetrics/torchvision |
| ② | `third_party/Matcha-TTS/matcha/utils/__init__.py` | 只保留 `from matcha.utils.pylogger import get_pylogger` | 原文件急切 import instantiators（→hydra/lightning）；推理只需 pylogger |
| ③ | `cosyvoice/utils/file_utils.py` `load_wav` | `torchaudio.load(wav, backend='soundfile')` → `soundfile.read` + 纯 torch Resample | torchaudio 2.x 无视 backend 参数强制走 torchcodec（未装）；只读 wav 用 soundfile 更轻 |

另有三处 backend 内注入（`tts/cosy/backend.py`，非文件补丁）：

- **`_force_cpu_onnx()`**：campplus/speech_tokenizer 是小模型，CPU 推理 ms 级——frontend 会按 `torch.cuda.is_available()` 请求 CUDAExecutionProvider，CPU 版 onnxruntime 没有该 provider 会崩。全局把 `ort.InferenceSession` providers 强制回 CPU。
- **`_ensure_wetext_local()`**：`wetext.Normalizer` 每次构造都调 modelscope `snapshot_download` 查 metadata，撞限流 403 会整个降级为"无文本前端"（数字不归一）。若 `~/.cache/modelscope/models/pengzhendong--wetext` 已有 FST 缓存，直接注入本地路径，零联网。
- **`_silence_tqdm()`**：仓库在推理链硬编码 `tqdm(...)`，流式会刷进度条，替换为透传。

---

## 8. 已知限制：0.5B LLM 的生成长度（修复后已基本解决）

**一句话：接入初期测得的"EOS 不可靠、时长随机、短句必冲上限"其实是坏 transformers（4.57.6）的产物；pin 到 4.51.3 后实测生成有界、时长与文本长度成比例。**

### 修复前后对比（2026-08-26，同一探针：direct llm.inference 数 token）

| 文本 | 字符 | 20×上限 | 修复前 4.57.6 | 修复后 4.51.3（3 次复现） |
|---|---|---|---|---|
| 今天天气真不错。 | 8 | 160 | 几乎必冲满（→8.8s，没完没了） | 51 / 52 / 78 token（~1.3-1.9s） |
| 星宇股份…通报。（用户句） | 44 | 880 | 多次 4s 块、持续生成（杂音） | 235 / 232 / 264 token（~5.8-6.6s） |
| 76 字长句 | 76 | 1520 | — | 413 token（~10.3s） |

- **生成有界**：所有实测都远低于 20× 上限，EOS 正常触发。
- **时长与文本长度成比例**：约 5~7 token/字（长句 5.4、短句因前缀开销略高）；音频 ≈ token×25ms。
- **存在正常的采样随机性**：同一句多次生成 token 数有波动（短句 51↔78，约 ±1.5×），是 top-k 采样的固有现象，不是"没完没了"。
- 社区对 CosyVoice2-0.5B EOS 偶发不稳的报告仍存在（模型已知特性），但本项目在默认配置下实测已不构成问题。

### 为什么修复前那么夸张

坏 transformers（4.57.6）破坏的正是 LLM 的采样/终止路径——EOS token 的 logits rank 掉到 top-25 之外、几乎不可采样，于是每句都冲到 `max_len = 20×文本token` 并持续吐"全能量填充"，表现就是**一句输入、多次输出、全是杂音、没完没了**。这是集成环境问题，不是模型固有行为。

### 还需要 `max_speech_ratio` 吗？

可作为**可选安全阀**保留（backend 内 `_wrap_max_ratio` 把 `max_token_text_ratio` 从 20 收紧，如 8 → 11 字 ≈3.5s）。修复后默认（None）已能正常终止，一般不必收紧；若个别句子偶发偏长、或希望时长更紧贴文本，再收紧到 8~10。收紧代价不变：可能在模型"自然结束点"之前截尾。

---

## 9. bench

```bash
python bench/bench_cosy.py --device cuda --profile          # 默认音色 + 克隆各一轮
python bench/bench_cosy.py --voice default                  # 只跑默认音色
python bench/bench_cosy.py --max-ratio 8                    # 收紧生成上限再测
python bench/bench_cosy.py --clone-wav X.wav --clone-text X # 用自己的 3s 参考
```

产出：`audio/2_cosy_{device}_{case}.wav`（+mp3，需 ffmpeg）、`audio/bench_report_cosy_{device}.txt`（逐句指标）、`.cache/bench_cosy_{device}.json`（结构化）、控制台甘特图。

> 克隆用例默认用内置参考女声跑通"克隆代码路径"；真正换音色时用 `--clone-wav`/`--clone-text` 指定自己的录音。

---

## 10. Windows 坑速查

1. **Python 必须是 `voice-tts` 环境**（先 `conda activate voice-tts`，再统一用 `python`），base 无 torch。
2. **torchaudio 2.x 不能用 `backend='soundfile'`**：会强制 torchcodec → patch ③。
3. **onnxruntime provider**：CPU 版没有 CUDA provider，frontend 按 `torch.cuda.is_available()` 请求会崩 → `_force_cpu_onnx()`。
4. **wetext modelscope 限流 403**：第二次加载会降级无文本前端 → `_ensure_wetext_local()`。
5. **盘符冒号**：`voice="clone:E:/...:文本"` 的解析已按 Windows 处理。
6. **`third_party/` 不入库**：换设备重 clone + `preload_cosy.py`，三个 patch 重打。
7. **transformers 必须 pin 4.51.3**：main env 的 4.57.6 会让 cosy 产出杂音（issue #1546）。换环境/重装后先跑 `python setup_cosy_pinned.py`；**cosy 与 melo 不能同一进程混用**（会明确报错，不产杂音）。
