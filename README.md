# voice0 — 离线实时文本转语音（TTS）

把文本实时转换为自然语音的程序，核心指标：**单句 TTFA（首包耗时）< 1s**、**离线运行**（权重缓存后零网络请求）、**声音自然**（非 SAPI 机器音）。

当前实现：**MeloTTS**（实时主力，TTFA<1s 达标）+ **CosyVoice2-0.5B**（可选项：音质 + 3s 音色克隆，见 `docs/README-cosyvoice2.md`）。两者经后端抽象独立选择安装，melo 为句子级分块流式，cosy 默认整句合成播放（句内无缝，可切原生 token 级流式）。

---

## TTS 选型结论（2026-08 定案）

> 完整选型依据（目标/硬件/候选对比/验收约定/来源）见 `docs/tts-architecture-decision.md`。

**目标约束**：离线、流式（TTFA + 边合成边播）、平衡 CPU/GPU 速度与音质。

**硬件**：RTX 2070 8GB（Turing sm_75）、NVIDIA 驱动 591.86 / CUDA 13.1；conda 环境 `voice-tts`（克隆自 python3.10，含 torch 2.11.0+cu126）。

| 方案 | 中文音质 | CPU 实时 | GPU | 流式 | 离线 | 部署复杂度 | 许可 |
|---|---|---|---|---|---|---|---|
| **MeloTTS（已采用）** | 较自然 | ✅ 原生 | 更快 | 按句分块 | ✅ | 低（pip） | MIT |
| CosyVoice2 (0.5B) | 很自然 + 音色克隆 + 方言 | ❌ 慢 | ✅ ~4GB | ✅ 原生双流式 | ✅ | 高 | Apache-2.0 |
| MOSS-TTS-Nano (0.1B) | 待测 | ✅ 原生 | 更快 | ✅ 原生 | ✅ | 中 | — |
| ~~edge-tts / Azure 等云端~~ | 自然 | — | — | — | ❌ 需联网 | — | — |

**结论**：MeloTTS 为首选（CPU 实时、离线、简单、中文较自然）；音质上限为 CosyVoice2（需 GPU 常驻）。排除所有云端方案（用户明确要求离线）。**实施顺序：先 MeloTTS，跑通测量后再试下一个。**

> **现状**：MeloTTS 已落地（验收达标）；**CosyVoice2-0.5B 已作为第二后端接入**（2026-08-26，可选项、选择性安装）——提供 melo 没有的"音质上限 + 任意 3s 参考音频克隆"，代价是句首等待 ~5.6-6.2s（整句合成后播放）、需 GPU、且 0.5B 的 LLM 生成长度有随机性（详见下节与 `docs/README-cosyvoice2.md`）。实时 <1s 硬指标仍由 melo 承担。⚠️ 接入初期曾因 main env 的 transformers 4.57.6 与 CosyVoice2 不兼容（官方 [issue #1546](https://github.com/FunAudioLLM/CosyVoice/issues/1546)）产出杂音，已用 vendored transformers 4.51.3 修复（见下节）。

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

## 常驻引擎 v2（单例 · 常驻线程 · 完整生命周期）

> 实现：`tts/core/engine.py` 的 `RealtimeTTS` / `jobs.py` 的 `Job`。本文档是**完整设计说明**，供后续接手的 AI 快速上手——代码、语义、边界、坑全部对齐当前实现。引擎为后端无关骨架：`backend` 参数切换 melo/cosy，差异全在后端模块（`tts/melo/backend.py` / `tts/cosy/backend.py`）。

**背景**：实时场景下文本到达时机不可预测。v1 每次调用都"建对象 → 加载模型 → 建线程 → 用完释放"，反复初始化开销大。v2 用三根支柱解决：**单例**（模型只加载一次）、**常驻线程**（合成/播放线程只创建一次）、**代际标记 `_gen`**（新文本随时可抢占或排队）。

```
submit(text) → Job(gen, sentences…) ─> _jobs 队列(无界)
     │                                      │ 合成线程（持 _synth_lock 逐句 _synth）
     │                                      ▼
     │                        _audio_q 队列(有界 maxsize=8, 天然背压)
     │                                      │ 播放线程（~50ms 小块写声卡, 块间查 _gen）
     │                                      ▼
     │                                DONE → _finalize_job → job.mark_done
     │                                          │
     └────── job.wait() 阻塞于 threading.Event ◄┘ 事件 set → 唤醒 → 返回逐句时序
```

### 1. 单例机制

- 类属性 `_instance` + 类锁 `_init_lock` 持有唯一活实例（`tts/core/engine.py`）。
- **`__new__`**：若已有实例——传入 `device` 解析后与当前实例的 `_device` 不同 → 自动 `close()` 旧实例、清空槽位、重建新设备；否则**直接返回同一实例**，绝不二次加载模型 / 开声卡流。
- **`__init__`**：`_inited` 为真（已是常驻实例）→ 只做运行期可变配置（`mode` / `speed`，`None` 表示"不改"），立即返回；否则做完整初始化。
- 效果：重复 `RealtimeTTS(...)` 永远同一个对象；`mode`/`speed` 变更**原地切换零重载**；`device` 变更**销毁重建**（旧引用随之失效）。
- **注意**：`profile` / `debug` 只在首次构造时生效，二次构造不会改变它们（只认 mode/speed/normalize）。

### 2. 常驻线程架构

- `_start_workers()`：合成线程 `_worker_synth` + 播放线程 `_worker_play`，均 `daemon=True`，**构造末尾启动、`close()` 才结束**。
- **两条队列**：
  - `_jobs`（`queue.Queue`，**无界**）：任务级。`submit` 入队，合成线程取。
  - `_audio_q`（`queue.Queue(maxsize=8)`，**有界**）：音频块级。有界 = 天然背压（合成远快于播放时，合成线程阻塞在 `put`，防延迟膨胀）。
- **合成线程**逐句处理（`_run_job`）：
  1. 每句开头检查 `self._shutdown or job.gen != self._gen` → 命中则放弃后续句子（关停/被抢占）；
  2. `with self._synth_lock: audio = self._synth(sent)` —— **模型非线程安全，所有合成调用必须持 `_synth_lock`**（流式、预热、`speak_to_file` 共用）；
  3. 可选落盘（见 §6）；`profile` 开时构造逐句 rec 并 `_timing_lock` 下 append 进 `job.timing`；
  4. `q.put((gen, job, i, sent, audio, rec))` —— 元组带上 `rec` 引用，播放线程直接回填播放时间戳；
  5. 循环走完（未 break）→ `aborted=False`；**finally** 里：发 `(gen, job, "DONE")`；若 `aborted` → 就地 `job.mark_done(canceled=True)`（兜底，防 `wait()` 悬挂）。
- **播放线程**（`_worker_play`）：开 `sounddevice.OutputStream(sr, 1, float32, blocksize=1024)`（个别设备不支持时退回默认块）；循环取 `_audio_q`：
  - 取到 `None` 或 `_shutdown` → 退出；
  - 取到 `DONE`（元组第 3 元素为 `"DONE"`）→ `canceled = (gen != self._gen)`；`_finalize_job(job, canceled)` 补派生字段；`job.mark_done(canceled)`；continue；
  - 取到音频块 → `gen != self._gen` 直接**丢弃**（被抢占的旧块）；否则 `_play_audio()` 写声卡并回填 `play_start`/`play_end`；
  - **finally**：关停退出前清空 `_audio_q` 残留块/DONE 并就地 `mark_done` 其 Job（兜底防悬挂）；`stop`+`close` 声卡。
- **`_play_audio`**：音频按 **~50ms 一小块**（`max(int(sr*0.05),1)`）逐块写声卡，**块间检查** `self._shutdown or gen != self._gen` → break。这是抢占粒度，残响 ~50-100ms。

### 3. 两种模式（`mode`，运行期可切）

- `mode` 属性只接受 `"queue"` / `"bargein"`，否则 `ValueError`；改属性立即生效。
- **`queue`（默认）**：新文本排到当前文本之后，按序播完再说。
- **`bargein`**：`submit()` 在 `_submit_lock` 内先 `_do_interrupt()` 再入队——新文本**自动打断**当前播放与清空旧排队（等价手动 `interrupt()` + 入队）。
- 切换示例：`tts.mode = "bargein"`。

### 4. 代际标记 `_gen` 与抢占机制

- `_gen` 整型计数（初始 0），每次打断 +1；`Job` 创建时快照当前值（`job.gen`）。
- **抢占判定**：`job.gen != self._gen` → 该任务的所有句/块一律作废。
- **播放抢占粒度 = 50ms 块**（块间检查，立即闭嘴，残响 ~50ms）；**推理抢占粒度 = 句子边界**（正在推理的那句**合完不播**、丢弃，之后句子不再合成；且持有 `_synth_lock` 期间，bargein 新任务的合成要等这句跑完，最多等一句推理时间）。
- **`_do_interrupt()`**：`_gen += 1` → 清空 `_jobs` 逐个 `mark_done(canceled=True)` → 清空 `_audio_q` 逐个 `mark_done(canceled=True)`。**关键坑（实测踩过）**：被清掉的音频块其 DONE 已随队列丢失，必须就地标记其 Job，否则该 `wait()` 永久悬挂。

### 5. 完整生命周期

- **构造**：模块 import 期重定向 `HF_HOME`/`HF_ENDPOINT`/`NLTK_DATA` 到项目内 `.cache/`（权重缓存不落系统盘）→ 加载 `TTS(language="ZH", device=...)` → 读 `_sr`(44.1k)/`_spk` → `_synth_lock` 下预热 `"你好。"`（cudnn 首次推理缓存，不计入验收）→ `_start_workers()` → `_inited=True`。
- **使用**：`submit`（非阻塞）/ `speak`（阻塞）/ `speak_to_file`（独立非流式）/ `interrupt` / `stop`。
- **销毁** `close()`：幂等（`_closed` 置位即返回）→ `_submit_lock` 内 `_do_interrupt()` → `_shutdown=True` → 两条队列各 `put(None)`（唤醒 worker）→ `join(timeout=10)` → `_init_lock` 下清 `_instance` 槽位。
  - 支持 `with RealtimeTTS(...) as tts:`（`__enter__` 检查存活、`__exit__` 自动 close）；
  - `__del__` 与模块级 `atexit.register(_atexit_close)` 兜底（进程退出自动 close）；
  - **`del tts` 无效**：只删名字引用，单例类属性 + 常驻线程仍持有对象 → 销毁必须显式 `close()`；
  - close 后再用任何公开方法 → `_check_alive()` 抛 `RuntimeError("RealtimeTTS 已 close()…")`；
  - close 后可重建：下次 `RealtimeTTS(...)` 走 `_instance is None` 全新初始化。
- **`wait()` 永不悬挂**：由 4 条 `mark_done` 路径共同保证——①正常播完（播放线程收 DONE）；②任务还在 `_jobs` 队列（`_do_interrupt` 清队列时）；③正在合成中被打断（`_run_job` finally 的 aborted 兜底）；④音频块已在 `_audio_q`（`_do_interrupt` 清音频队列时）。`close()` 在途销毁也被这些路径覆盖。
- **Job 对象**：`wait()` 阻塞于内部 `threading.Event`，事件 set 即返回 `job.timing`；`mark_done(canceled)` = 置 `canceled` 标志 + set 事件。**被取消的任务 `timing` 可能是部分或空的**（只含已合成完的句子，且派生字段不计算）。

### 6. API 参考

| 方法 / 属性 | 阻塞？ | 说明 |
|---|---|---|
| `RealtimeTTS(device="auto", speed=None, mode=None, normalize=None, backend="melo", voice=None, profile=False, debug=False, max_speech_ratio=None, stream=None)` | 构造 | 单例；device/backend/voice/max_speech_ratio/stream 变更销毁重建，mode/speed/normalize 原地切换，profile/debug 仅首次生效。`backend`：melo（默认）/ cosy；`voice` 仅 cosy：`"default"` 或 `"clone:<wav>:<文本>"`；`max_speech_ratio` 仅 cosy：收紧 LLM 生成上限（None=模型默认）；`stream` 仅 cosy：`False`=整句合成后播放（默认，句内无缝）/ `True`=原生 token 级流式（本机 2070S 会卡顿，不推荐） |
| `submit(text, save_chunks_dir=None, save_wav=None)` | 否 | 入队立即返回 `Job`；bargein 模式自动打断；**实时场景用这个** |
| `speak(text, save_chunks_dir=None, save_wav=None)` | 是 | = `submit().wait()`，播完返回逐句时序（bench 兼容） |
| `job.wait()` | 是 | 阻塞到本任务播完/取消，返回 `job.timing` |
| `interrupt()` / `stop()` | 否 | 打断当前正在说的 + 清空排队文本（`stop` 是 v1 别名） |
| `speak_to_file(text, wav_path)` | 是 | 整段**非流式**落盘 WAV（独立一次推理），"只落盘不播放"用 |
| `close()` | 否 | 销毁（线程/声卡/单例槽位）；幂等；`with`/`__del__`/atexit 兜底 |
| `mode` / `speed` / `normalize` | — | 属性，运行期可读可写（`mode`/`normalize` 校验取值） |
| `device` | — | 只读 |
| `last_ttfa` / `last_timing` | — | 最近一次 `profile` 任务（未取消）的指标快照 |

**`timing` 逐句记录 schema**（每句一条 dict；**`bench/bench_melo.py` 强依赖，改动必须同步 bench**）：

- 合成线程写入：`idx`、`text`、`synth_start`、`synth_end`、`audio_dur`（`play_start`/`play_end` 初值 `None`）；
- 播放线程回填：`play_start`、`play_end`；
- DONE 收尾（**未取消**时）派生：`ttfa`（`play_start − job.t_start`）、`synth_dur`、`wait`（`play_start − synth_end`）、`interval`（句间；句1 为 `None`）；
- **坑**：`profile=False`（默认）时 `job.timing` 为空列表（`speak()` 返回 `[]`）——要拿逐句时序必须构造时 `profile=True`。

**WAV 落盘（播放时顺手存，零额外推理）**：
- `save_wav="out.wav"`：全部句子 `np.concatenate` 拼成**一个整段 WAV**（复用流式已合成的逐句 audio，不重复推理；被打断存半截）；
- `save_chunks_dir="dir/"`：每句各存一份（`句01.wav`、`句02.wav`…）；
- 两者**相互独立**、可同时传；
- `speak_to_file()` 是另一条独立非流式路径——**不要用它补存已 `speak()` 过的文本**（音频播完不保留在内存，只能重推）。

**逐句响度归一化（`normalize` 开关）**：
- **为什么需要**：MeloTTS 逐句独立合成、不做响度均衡。实测两类不齐：
  - **句间**：活动段（非静音）RMS 差 ~8dB，短句/唱词类句子明显偏小；
  - **句内**：20ms 帧语音 RMS 的 p90−p10 差达 **17~20dB**、最大起伏 ~26dB，且模式稳定——**句首 ~100ms 从约 -40dB 爬升、句尾 200~300ms 回落到 -40dB**（开头轻 / 结尾轻 / 中间响，VITS 幅度包络特性）；
  - **句内死寂**：VITS 把标点停顿直接烘焙进单个波形，逗号处实测可生成 **0.58~0.62s 的绝对数字静音**（-inf dB，远超自然短停顿 ~0.1s），听感是"每逗号之间突然消音没了声音"。
- `normalize=None`（默认）：原样播放，不做任何增益；
- `normalize="rms"`：**静态**逐句响度对齐——按 20ms 帧剔除非静音帧算活动段 RMS（活动阈值取**相对句子峰值的比例**，缩放不变，对齐精确），整句缩放对齐目标 **-24 dBFS**，峰值超 **0.95** 整体下压（全静音句跳过）。治**句间**音量不齐，句内起伏/死寂不变；
- `normalize="agc"`：在 rms 静态对齐基础上再加**句内动态压缩**（20ms 帧 RMS 包络逐帧增益压平到目标响度，帧间线性插值防抽吸、噪声门防静音/呼吸被抬、增益限幅 -10~+16dB，治**句首轻/句尾轻/中间响**）+ **短停压缩**（检测相对峰值 -60dB 以下的深度静音缝，超过 0.25s 的从中间截短到 0.25s、两侧各留一半保留渐入渐出边，静音接静音无咔哒；句首/句尾静音不动，治**逗号死寂收音**）。全句音量最一致、节奏更紧凑（推荐）；
- 均在 `_synth` 边界生效——播放、`save_wav`、`save_chunks_dir`、`speak_to_file` 同时受益；运行期可切：`tts.normalize = "rms"` / `"agc"` / `None`（原地生效）。

### 7. 使用案例

**最简阻塞**（bench 同款）：
```python
tts = RealtimeTTS(device="cuda", mode="queue", profile=True)
timing = tts.speak("你好，世界。")   # 阻塞，播完返回逐句时序
print(timing[0]["ttfa"])             # 首句 TTFA（秒）
tts.close()
```

**实时非阻塞**（核心用法）：
```python
tts = RealtimeTTS(mode="queue")
tts.submit("第一句。")                # 入队即返回，不阻塞
tts.submit("第二句。")                # 排在第一句之后
# ... 继续处理其他业务 ...
tts.close()
```

**bargein 自动打断**：
```python
tts = RealtimeTTS(mode="bargein")
tts.submit("这句话可能还没说完")
tts.submit("打断它！")                # 自动打断上一句，优先播这句
tts.close()
```

**手动打断 + 等待结果**：
```python
tts = RealtimeTTS(mode="queue", profile=True)
job = tts.submit("一段很长的文本……")
time.sleep(0.3)
tts.interrupt()                      # 手动打断；job 被标记 canceled
timing = job.wait()                  # 立即返回（不悬挂），timing 可能不全
print(job.canceled)                  # True
tts.close()
```

**运行期切换模式 / 语速**：
```python
tts = RealtimeTTS(mode="queue")
tts.submit("先排队。")
tts.mode = "bargein"                 # 原地切换，立即生效
tts.speed = 1.2                      # 语速实时生效
tts.submit("现在打断。")
tts.close()
```

**播放时落盘**（零额外推理）：
```python
tts = RealtimeTTS()
tts.speak("要留档的话。", save_wav="audio/full.wav", save_chunks_dir="audio/chunks/")
tts.speak_to_file("只落盘不播放。", "audio/raw.wav")   # 独立非流式，一次推理
tts.close()
```

**逐句响度归一化**（音量一致）：
```python
tts = RealtimeTTS(normalize="agc")     # 或运行期 tts.normalize = "agc"
tts.speak("短句会小声。")              # agc：句间+句内音量都压平、死寂停顿压缩（推荐）
tts.normalize = "rms"                  # 只做句间静态对齐，句内起伏保留
tts.normalize = None                   # 随时切回原样
tts.close()
```

**上下文管理器**（自动销毁）：
```python
with RealtimeTTS(mode="bargein") as tts:
    tts.submit("离开作用域自动 close()。")
```

**device 变更自动重建**：
```python
tts = RealtimeTTS(device="cuda")
tts.speak("先用 GPU。")
RealtimeTTS(device="cpu")            # 旧实例被自动 close()，重建为 cpu 实例
tts.speak("再调用会抛 RuntimeError")  # 旧引用已失效
```

**空文本**：
```python
job = tts.submit("   ")              # 无句子，立即完成
assert job.wait() == []              # wait() 直接返回空列表
```

### 8. 线程安全与易踩的坑（给后续接手者）

1. **模型非线程安全**：所有 `_synth` 必须持 `_synth_lock`（流式合成、预热、`speak_to_file` 共用同一把锁，天然串行）。
2. **`_submit_lock` 不可重入**：`submit`/`interrupt`/`close` 用它串行化；`_do_interrupt` 是内部原语，调用方须已 `_check_alive()` + 持锁（`close` 有意跳过 alive 检查以保持幂等）——别在 `_do_interrupt` 里再拿锁。
3. **别改回一次写整段声卡**：`_play_audio` 的 50ms 小块 + 块间查 `_gen` 是抢占的物理基础，改大残响变长。
4. **`_audio_q` 有界 maxsize=8**：改大 → 延迟膨胀；改小 → 背压提前、合成可能被卡。
5. **改队列结构时必须同传 Job 引用与 DONE 标记**，且中断/关停清空队列时**务必就地 `mark_done`**——否则 `wait()` 永久悬挂（本项目实测踩坑）。
6. **`profile=False` 时拿不到时序**（timing 空）；bench 依赖的 timing schema 改动要同步 `bench/bench_melo.py`。
7. **单例跨进程不共享**：每个进程各一份；bench 按 device 拆子进程，正是因 MeloTTS BERT 为模块级单例，CPU/GPU 同进程会设备不匹配。
8. **销毁只能 `close()`**：`del tts` 只删名字引用，单例槽位 + 常驻线程仍持引用，对象不会被回收。

---

## 多方案扩展与目录演进（阶段 2 已完成）

> **2026-08-26 已执行阶段 2**：引入 CosyVoice2 第二流式后端时一次到位，平铺结构重构为 `tts/core`（后端无关骨架）+ 各后端子目录。本节的"已验证思考"是当初拆分的依据，**现已是现实**——目录树见文末。

### 现状

- **后端抽象落地**：`tts/core/backend.py` 定义 `TTSBackend` 协议（`name`/`sr`/`load`/`synth_stream`/`synth`/`close`）+ `get_backend(name)` 惰性 import。`tts/core` 只依赖 numpy+sounddevice，**零后端 import**；melo/cosy 模块全惰性加载，缺失依赖时抛 `BackendNotInstalledError` 带安装提示——这是**选择性安装**（melotts-only / cosyvoice2-only / both）的根基。
- **melo**：`tts/melo/backend.py`，`synth_stream` = `iter([synth(text)])`（整句一次，行为与重构前一致）。
- **cosy**：`tts/cosy/backend.py`，`synth_stream` 默认 `inference_zero_shot(stream=False)` **整句一次解码、单块产出**（本机 fp32 RTF>1，原生流式会块间饿死；`stream=True` 可切回 token 级逐块）。见 `docs/README-cosyvoice2.md`。
- `synth_sapi.py`（`sapi/synth_sapi.py`）仍是**非神经对照基线**，与神经链路零代码耦合。
- bench 按后端拆：`bench/bench_melo.py` / `bench/bench_cosy.py`，共享 `bench/console_gantt.py`。

### 已验证：v2 引擎的可复用边界

v2 引擎（`RealtimeTTS`）本质是**后端无关骨架 + 一处后端专属**：

| 后端无关（换方案可原样复用） | 后端专属（须替换） |
|---|---|
| `Job`（wait / mark_done / canceled / timing） | **`synth_stream(text) -> 逐块生成器`**（melo=整句一块；cosy=默认整句一块，`stream=True` 切原生逐块） |
| 常驻合成/播放双线程、`_jobs` + `_audio_q`（背压） | timing 口径微调（见下） |
| `_gen` 代际抢占、queue/bargein 模式、interrupt/close 生命周期 | |
| timing schema、save_wav / save_chunks_dir、profile/debug | |

### 原生流式后端（MOSS-TTS-Nano / CosyVoice2）的适配点

> 注（2026-08-26）：CosyVoice2 **保留**原生 token 级流式能力（`stream=True`），但本机 2070S fp32 下 RTF≈1.2（合成比播放慢），流式播放会块间饿死 + 拼接缝，故默认 `stream=False` 整句合成播放（见 cosy 节 ⚠️）。此节适配点对"想用流式的后端"仍成立。

原生流式 = 模型**边生成边吐音频块**，不必等整句合成完。适配集中在合成边界一层：

```python
# Melo（现状）：整句一次性
audio = self._synth(sent)
q.put((gen, job, i, sent, audio, rec))          # 整句入队

# 原生流式（改后）：逐块吐出
for chunk in self._synth_stream(sent):          # 生成器，边推理边产出
    q.put((gen, job, i, sent, chunk, rec))      # 逐块入队，播放线程原样消费
```

- **播放线程零改动**（本就能消费任意长度块）；抢占粒度从"句子边界"→"块边界"，**更细、残响更短**；
- timing 口径微调：`ttfa` 变为"首块到播放时刻"；`wait` / `interval` 按句聚合口径需重算；
- **换后端 ≈ 重写合成边界一层，不是从零重建**（骨架复用约 80%）。

### 目录演进（已执行）

```
voice0/
├── tts/
│   ├── core/          # 后端无关骨架：engine/jobs/audio/backend（自 tts_melo.py 抽出，git mv 保历史）
│   ├── melo/          # Melo 后端（整句 synth）
│   └── cosy/          # CosyVoice2 后端（默认整句合成播放，可切原生流式）
├── sapi/              # 非神经对照基线（synth_sapi.py）
├── bench/             # bench_melo / bench_cosy / console_gantt
├── docs/              # tts-architecture-decision.md + README-cosyvoice2.md
└── README.md
```

> 演进触发条件（2026-08-26 满足）＝ **确实引入第二个流式后端**。sapi 挪动时 `OUTPUT_DIR` 已改为项目根向上取一级（`sapi/synth_sapi.py`）。

> 目标：**别人只凭本文档，在一台新设备上从零得到可用的 voice0 系统**。所有版本号均为 `voice-tts` 环境实测（2026-08-25 抓取），不是估算。

### 硬件与系统要求

| 项 | 要求 | 说明 |
|---|---|---|
| 系统 | Windows / Linux 均可（本仓库在 Windows 11 实测） | — |
| Python | **3.10**（conda 推荐） | 3.11+ 未经本仓库验证 |
| GPU | NVIDIA + **CUDA 12.6 兼容驱动**（推荐） | 实测 RTX 2070 SUPER 8GB，**峰值显存 1.088GB**，首句 TTFA ~0.2s |
| CPU-only | 可用，`device="cpu"` | 首句 TTFA ~1s，达标但明显慢于 GPU |
| 磁盘 | ~5GB | torch（~2.5GB）+ 权重缓存 `.cache/hf`（~0.9GB）+ NLTK 语料 |
| 网络 | **首次**下载权重需联网 | huggingface.co 被墙时脚本自动走 `hf-mirror.com` 镜像；**之后运行期零网络** |
| ffmpeg | 可选 | 仅 MP3 转码用；不装只产 WAV |

### 已锁定版本（复现基准，实测值）

| 组件 | 版本 | 说明 |
|---|---|---|
| python | 3.10.16 | conda |
| torch | 2.11.0+cu126 | cu126 = CUDA 12.6 版，PyPI 专属 index（见下） |
| torchaudio | 2.11.0+cu126 | 与 torch 配套 |
| melotts | **0.1.2**（源码 editable 装，见步骤 3） | PyPI 版为 0.1.1；实测用源码 0.1.2 |
| sounddevice | 0.5.6 | 声卡流（**必须显式安装**，非 melotts 依赖） |
| numpy | 2.2.6 | 勿降级 |
| transformers | 4.57.6 | — |
| huggingface_hub | 0.36.2 | — |
| jieba | 0.42.1 | 中文分词（ZH 必需） |
| nltk | 3.10.3 | 英文音素前端语料 |
| scipy / numba / librosa / pypinyin | 1.15.3 / 0.67.0 / 0.11.0 / 0.55.0 | melotts 推理链路依赖 |
| setuptools | **80.9.0（固定）** | 81+ 移除 `pkg_resources`，**jieba 会崩**（见坑 ①） |

### 从零复现步骤（新设备推荐，可整段复制）

```bash
# 0. 前提：已安装 conda（miniconda 即可）、git

# 1. 建环境（Python 3.10）
conda create -n voice-tts python=3.10 -y
conda activate voice-tts

# 2. torch（CUDA 12.6 版；无 GPU 就装 CPU 版：pip install torch==2.11.0 torchaudio==2.11.0）
pip install torch==2.11.0+cu126 torchaudio==2.11.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# 3. melotts（方式A=精确对齐实测 0.1.2 源码；方式B=PyPI 0.1.1 更省事）
#    方式A（推荐，与实测完全一致；github 被墙时加前缀 https://ghfast.top/）
git clone https://github.com/myshell-ai/MeloTTS.git .cache/MeloTTS
pip install -e .cache/MeloTTS
#    方式B：pip install melotts          # PyPI 0.1.1，功能等价
pip install sounddevice==0.5.6           # 声卡流，必装

# 4. 锁关键依赖版本（melotts 会自带 jieba/g2p-en 等，这里显式对齐实测）
pip install numpy==2.2.6 transformers==4.57.6 huggingface_hub==0.36.2 \
            jieba==0.42.1 nltk==3.10.3 scipy==1.15.3 numba==0.67.0 \
            librosa==0.11.0 pypinyin==0.55.0
pip install setuptools==80.9.0           # jieba 依赖 pkg_resources，必须 <81

# 5. 一次性预下载全部权重（6 个 tokenizer + 670MB BERT + ZH VITS + NLTK 语料）
#    自动重建 .cache/hf 与 .cache/nltk_data（本项目内，不入库）
python preload_weights.py

# 6. 验收：CPU/GPU 各一轮，自动产出时序报告并更新本 README 的验收区间
python bench/bench_melo.py --device all --profile --debug

# 裸管线（无插桩，贴近生产）：
python bench/bench_melo.py --device cuda
```

> **melo 装完即可用**。若要 CosyVoice2 第二后端（音质 + 3s 克隆），按
> `docs/README-cosyvoice2.md` 单独安装（`third_party/` clone + 依赖 + `preload_cosy.py`
> + `setup_cosy_pinned.py` 落盘 cosy 专属 transformers 4.51.3），
> 两引擎相互独立、可只装其一，`bench/bench_cosy.py --device cuda` 验收。

**预期结果对照**（应达到的 TTFA，单位秒）：

| 设备 | 短句 | 中句 | 长句 | 判定 |
|---|---|---|---|---|
| GPU (cuda) | ~0.20 | ~0.16 | ~0.36 | 全部 <1s ✅ |
| CPU | ~0.93 | ~1.00 | ~2.6 | 短/中 <1s ✅，长句 CPU 略超属正常（合成是瓶颈） |

### 原有克隆路径（仅原机器场景：目标机已有含 torch 的 python3.10 大环境）

```bash
conda create -n voice-tts --clone python3.10   # 复制含 torch 2.11.0+cu126 的环境
D:/anaconda/envs/voice-tts/python.exe -m pip install melotts sounddevice
D:/anaconda/envs/voice-tts/python.exe preload_weights.py
D:/anaconda/envs/voice-tts/python.exe bench/bench_melo.py --device all --profile --debug
```

> 从大环境克隆才会遇到坑 ②（jax 冲突）；从零 `conda create python=3.10` 不会。

### 常见坑（复现时）

1. **setuptools ≥ 81 会让 jieba 崩**（`pkg_resources` 被移除）→ 必须 `pip install setuptools==80.9.0`。
2. **克隆大环境带的 jax/jaxlib/ml_dtypes 与 numpy 2.2.6 不兼容**，会在 import transformers 时崩 → 卸载 `pip uninstall -y jax jaxlib ml_dtypes`（仅克隆路径需处理）。
3. **huggingface.co / raw.githubusercontent 在本机被墙/极慢** → 脚本已内置 `HF_ENDPOINT=hf-mirror.com` 与 `ghfast.top` 代理，首次下载自动走镜像。
4. **权重缓存必须落在项目内**：`tts/melo/backend.py` / `tts/cosy/backend.py` 在 import 期就把 `HF_HOME`/`NLTK_DATA` 重定向到 `.cache/`，不要手动改。
5. **换设备 `.cache/` 不会随 git 过来**（已 gitignore）——重装环境后跑一次 `preload_weights.py`（melo）/ `preload_cosy.py`（cosy）即可自动重建，无需手动拷贝。
6. **`import melo` 用的是 editable 源码**（`.cache/MeloTTS`）：删掉该目录会导致 import 失败，需要重跑步骤 3 方式A。

### 产出说明

`audio/2_melo_{device}_{case}.wav`（试听对比）、`reports/bench_timing_{device}.html`（悬停看明细）、`audio/bench_report_{device}.txt`。MP3 压缩依赖 ffmpeg（本机未装，仅产出 WAV）；装了 ffmpeg 后自动追加 `.mp3`。

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
| 短 | 2 | 0.159 | 152 | 1782 | 4.13 |
| 中 | 3 | 0.143 | 152 | 3788 | 6.45 |
| 长 | 3 | 0.326 | 304 | 10057 | 15.29 |

- 交互式时序图（悬停看明细）: `reports/bench_timing_cuda.html`
- 逐句原始指标: `audio/bench_report_cuda.txt`

### CPU / GPU 对比结论（bench 自动生成）

| 指标 | CPU | GPU |
|---|---|---|
| 首句 TTFA | 0.927s | 0.159s |
| TTFA 达标(<1s) | ✅ | ✅ |
| GPU 相对 CPU 提速 | — | 5.8× |
| GPU 峰值显存 | — | 1.088 GB |

**结论**：GPU 首句 TTFA=0.159s 达标<1s；CPU 首句 TTFA=0.927s 达标<1s。流式重叠已验证：播放段与下一句合成段重叠。

> 生成时间：2026-08-26 09:20:44（每次 bench 运行自动刷新）
<!-- bench:end -->

---

## CosyVoice2（第二后端，可选项）

> 完整安装/依赖/权重/音色配置/已知限制见 **[`docs/README-cosyvoice2.md`](docs/README-cosyvoice2.md)**。本节只放结论。

**定位**：melo 只有一种中文女声；cosy 提供 **音质上限 + 任意 3s 参考音频克隆**（0.5B 无 SFT，"默认音色"从内置参考音频零样本克隆实现）。代价：句首等待 ~5.6s（=整句合成耗时，达不到 melo <1s 硬指标）、需 GPU 常驻、且 **0.5B LLM 生成长度有随机性**（见下）。

> ⚠️ **修复记录（2026-08-26，务必先读）**：接入初期的"音质/克隆验收"是在**坏管线上测的**——main env 的 transformers 4.57.6 与 CosyVoice2 不兼容（官方 [issue #1546](https://github.com/FunAudioLLM/CosyVoice/issues/1546)：>4.51.3 就出问题；4.53+ 重写了 `Qwen2Model.forward`），LLM 产出全错的 speech token，表现为**"输入一句 → 多次输出、全是杂音、没完没了"**（Whisper 听写是单字重复如"我哭哭哭哭"，不是人声）。已修复：本仓库自带 vendored `transformers==4.51.3 + tokenizers==0.21.1`（`.cache/pinned_transformers`，`setup_cosy_pinned.py` 一键落盘，94M），`tts/cosy/backend.py` import 时自动注入；melo 仍用 main env 的 4.57.6，互不影响。修复后 Whisper(base/small/medium) 听写**内容正确**（短句全对；长句逐字还原，仅公司名/地名的同音字由 Whisper 自选），且**生成有界**（不再没完没了）。**cosy 与 melo 不能在同一进程混用**（transformers 版本冲突，会明确报错而非产杂音）。

**用法**：
```python
tts = RealtimeTTS(device="cuda", backend="cosy", voice="default")        # 默认音色（内置参考）
tts = RealtimeTTS(device="cuda", backend="cosy",
                  voice="clone:E:/my_voice.wav:这是我的转写文本")          # 3s 克隆
tts = RealtimeTTS(device="cuda", backend="cosy", max_speech_ratio=8)     # 收紧生成上限
tts = RealtimeTTS(device="cuda", backend="cosy", stream=True)            # 恢复原生 token 级流式（首包快但本机会卡顿）
tts.speak_to_file("要生成的文本。", "audio/cosy_out.wav")                 # 文件生成（cosy 主用途）
```

> **播放方式**：默认 `stream=False`（整句合成后一次性播放，**句内无缝**）。接入初期 cosy 走原生 token 级流式（`stream=True`），但本机 2070S fp32 下 RTF≈1.1-1.3（合成比实时播放慢），逐块播放必然"字间戛然而止 + 块拼接缝"——即你实测听到的不连续。改非流式后每句一次解码、单块播放，消除拼接缝与饿死停顿；代价是句首等待 = 整句合成耗时（~1.2×音频时长），`speed` 也同时生效（流式下被忽略）。实时 <1s 主力仍是 melo。

**验收实测**（RTX 2070 SUPER 8GB，`bench_cosy.py --device cuda`）：

| 用例 | 模型加载 | 句1 TTFA | 峰值显存 | 备注 |
|---|---|---|---|---|
| 默认音色 | 61.5s | **~5.6s** | 2.88 GB | 整句合成后播放（stream=False，句内无缝）；Whisper 转写 4 句内容正确 ✅ |
| 3s 克隆 | 52.7s | **~6.2s** | 2.98 GB | 整句合成后播放；克隆路径跑通 + Whisper 转写正确（音色质量以试听为准，见上 ⚠️） |

**已知限制（修复后已基本解决）**：接入初期测得的"生成时长随机 / 短句必冲上限 / 没完没了"其实是坏 transformers（4.57.6）的产物（见上 ⚠️），pin 到 4.51.3 后实测**生成有界、时长与文本长度成比例**（约 5~7 token/字，音频 ≈ token×25ms），仅保留正常的采样随机性（同句 token 数 ±1.5×）。`max_speech_ratio` 降级为**可选安全阀**（如 8 → 11 字 ≈3.5s），一般不必收紧。**实时 <1s 主力仍是 melo**，cosy 适合"整段生成 + 试听 + 必要时重生成"。

---

## 目录树

<!-- tree:start -->
```
voice0/
├── assets/
│   └── cosy_default_female.wav
├── audio/  # 生成的音频/bench 报告（chunks/、bench_report_*.txt、smoke_*.wav 等；gitignored）
├── bench/
│   ├── __init__.py
│   ├── bench_cosy.py
│   ├── bench_melo.py
│   └── console_gantt.py
├── docs/
│   ├── README-cosyvoice2.md
│   └── tts-architecture-decision.md
├── reports/
│   ├── bench_timing_cpu.html
│   └── bench_timing_cuda.html
├── sapi/
│   └── synth_sapi.py
├── third_party/  # 克隆的上游仓库（CosyVoice + Matcha-TTS；gitignored）
├── tmp/  # 诊断/临时 scratch（diag_*.py、探针与产物；gitignored）
├── tts/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   ├── backend.py
│   │   ├── engine.py
│   │   └── jobs.py
│   ├── cosy/
│   │   ├── __init__.py
│   │   └── backend.py
│   ├── melo/
│   │   ├── __init__.py
│   │   └── backend.py
│   └── __init__.py
├── .gitignore
├── README.md
├── preload_cosy.py
├── preload_weights.py
├── setup_cosy_pinned.py
└── speak_example.py
```
<!-- tree:end -->
