# CLAUDE.md — voice0 项目指南

离线实时中文 TTS 系统。核心指标：**单句 TTFA（首包耗时）<1s**、**离线运行**（权重缓存后零网络请求）、**声音自然**。

两个后端（**选择性安装**，互不影响）：
- **melo**（默认，实时主力）：MeloTTS 中文女声，TTFA<1s 达标。
- **cosy**（可选项，音质 + 3s 音色克隆）：CosyVoice2-0.5B，句首等待 ~5.6-6.2s（整句合成后播放），需 GPU。
- `sapi/` 是非神经对照基线（SAPI 机器音，非主用）。

## 快速上手（安装 → 使用）

详细安装步骤**以 README 为准**，CLAUDE.md 只给地图：

1. **环境**：见 `README.md`「从零复现步骤（新设备推荐，可整段复制）」——从零建 `voice-tts` conda 环境（Python 3.10 + torch 2.11.0+cu126），版本锁定表在「已锁定版本」。cosy 的安装另见 `docs/README-cosyvoice2.md` §2。
2. **权重预下载**（仅首次联网，之后运行期零网络）：
   - melo：`python preload_weights.py`
   - cosy（若用）：`python preload_cosy.py` + `python setup_cosy_pinned.py`（落盘 cosy 专属 transformers 4.51.3）
3. **跑起来**：`python speak_example.py` —— 交互 demo（选后端/音色/播放模式，输入文本即播报）。
4. **代码里用**：
   ```python
   from tts import RealtimeTTS
   tts = RealtimeTTS(backend="melo")                 # 实时：submit() 非阻塞 / speak() 阻塞
   tts.submit("你好。")                               # 或 speak("...") 播完返回
   tts.speak_to_file("你好。", "audio/out.wav")      # 只落盘不播放
   tts.close()                                       # 常驻单例，必须显式关
   ```

## 环境硬性约束（新 Claude Code 接手必须先知道）

- **Python 必须用 `voice-tts` conda 环境**：先 `conda activate voice-tts` 再跑 `python`；base Python 3.12 无 torch。新设备按 README 重建同名环境。
- **跑带中文输出的命令加 `PYTHONIOENCODING=utf-8`**：Windows 默认 GBK 会直接崩。
- 权重/缓存重定向到项目内 `.cache/`（`HF_HOME`/`HF_ENDPOINT` 在模块里已设好）；首次下载走 `HF_ENDPOINT=https://hf-mirror.com` 镜像（huggingface.co 直连被墙）。
- git 仓库根即本目录（`third_party/` 是 gitignored 的上游 clone，别在里面提交）。

## 关键坑（非显而易见的，先看再动）

- **cosy 与 melo 不能同进程混用**：cosy 需要 vendored `transformers==4.51.3`（`.cache/pinned_transformers`）；进程里若先 import 了高版本会**明确报错**（这是设计，不是 bug）。跑 cosy 用独立进程。**不要在构造 cosy 引擎前 `import transformers`。**
- **模型非线程安全**：所有合成调用必须持 `_synth_lock`（引擎已处理，写新后端/新调用路径时别绕过）。
- **`RealtimeTTS` 是单例 + 常驻线程**：必须显式 `close()` 才销毁；`del tts` 不保证生效。`device/backend/voice/stream` 变更 = 销毁重建；`mode/speed/normalize` 原地切换。
- **验收纪律（血泪教训）**：时序/显存数字达标 ≠ 输出正确。新后端或改动后必须 **Whisper 转写或试听**，不能只看 TTFA/RTF。

## 代码结构

```
tts/
├── core/      后端无关骨架
│   ├── engine.py   RealtimeTTS（单例/双线程/mode/lifecycle）——引擎设计权威文档在 README「常驻引擎 v2」
│   ├── jobs.py     Job（wait/mark_done/done/canceled）
│   ├── backend.py  TTSBackend（ABC：load/synth_stream/synth/close）+ get_backend(name) 惰性加载
│   └── audio.py    保存 WAV + normalize
├── melo/        MeloBackend（继承 TTSBackend）
└── cosy/        CosyBackend（继承 TTSBackend；transformers pin / onnx / wetext 前置全在模块顶层）
bench/           bench_melo / bench_cosy / console_gantt（bench 自动回写 README 的验收区间 + 目录树）
docs/            tts-architecture-decision.md（选型与环境决策）+ README-cosyvoice2.md（cosy 完整文档）
sapi/            synth_sapi.py（非神经基线）
assets/          cosy 默认音色参考音频
third_party/     克隆的上游仓库（CosyVoice + Matcha-TTS，gitignored）
```

## 权威文档（动手前先读对应章节）

- `README.md`「常驻引擎 v2」（§1-§8）= 引擎完整设计：单例机制、线程数据流、queue/bargein 模式、`_gen` 抢占、生命周期、API 参考、使用案例、线程安全坑。**改引擎代码前必读。**
- `docs/README-cosyvoice2.md` = cosy 完整文档：安装/音色配置/API/验收实测（含 fp16 已排除结论、句间停顿为固有代价）/已知限制。
- `docs/tts-architecture-decision.md` = 选型结论与硬件环境（新设备复现以 README 为准）。
- `docs/ai-project-methodology.md` = 本项目沉淀的 **AI 项目全流程方法论**（立案→探索→实施→测试→迭代→文档→复现，含验收纪律/探针文化/反模式清单），可复用到其他 AI 项目。

## 协作习惯（本项目）

- 与用户用**中文**交流。
- 里程碑式改动后用户会说「commit吧」——按其节奏提交，commit message 用中文。
- 每次验收必须转写/试听验证内容正确性，不要只报数字。
