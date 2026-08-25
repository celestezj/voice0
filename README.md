# voice0 — 离线实时文本转语音（TTS）

把文本实时转换为自然语音的程序，核心指标：**单句 TTFA（首包耗时）< 1s**、**离线运行**（权重缓存后零网络请求）、**声音自然**（非 SAPI 机器音）。

当前实现：**MeloTTS**（见下方选型结论）。流式架构为"句子级分块 + 边合成边播"。

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

## 常驻引擎 v2（单例 · 常驻线程 · 完整生命周期）

> 实现：`tts_melo.py` 的 `RealtimeTTS` / `Job`。本文档是**完整设计说明**，供后续接手的 AI 快速上手——代码、语义、边界、坑全部对齐当前实现。

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

- 类属性 `_instance` + 类锁 `_init_lock` 持有唯一活实例（`tts_melo.py`）。
- **`__new__`**：若已有实例——传入 `device` 解析后与当前实例的 `_device` 不同 → 自动 `close()` 旧实例、清空槽位、重建新设备；否则**直接返回同一实例**，绝不二次加载模型 / 开声卡流。
- **`__init__`**：`_inited` 为真（已是常驻实例）→ 只做运行期可变配置（`mode` / `speed`，`None` 表示"不改"），立即返回；否则做完整初始化。
- 效果：重复 `RealtimeTTS(...)` 永远同一个对象；`mode`/`speed` 变更**原地切换零重载**；`device` 变更**销毁重建**（旧引用随之失效）。
- **注意**：`profile` / `debug` 只在首次构造时生效，二次构造不会改变它们（只认 mode/speed）。

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
| `RealtimeTTS(device="auto", speed=None, mode=None, profile=False, debug=False)` | 构造 | 单例；device 变更销毁重建，mode/speed 原地切换，profile/debug 仅首次生效 |
| `submit(text, save_chunks_dir=None, save_wav=None)` | 否 | 入队立即返回 `Job`；bargein 模式自动打断；**实时场景用这个** |
| `speak(text, save_chunks_dir=None, save_wav=None)` | 是 | = `submit().wait()`，播完返回逐句时序（bench 兼容） |
| `job.wait()` | 是 | 阻塞到本任务播完/取消，返回 `job.timing` |
| `interrupt()` / `stop()` | 否 | 打断当前正在说的 + 清空排队文本（`stop` 是 v1 别名） |
| `speak_to_file(text, wav_path)` | 是 | 整段**非流式**落盘 WAV（独立一次推理），"只落盘不播放"用 |
| `close()` | 否 | 销毁（线程/声卡/单例槽位）；幂等；`with`/`__del__`/atexit 兜底 |
| `mode` / `speed` | — | 属性，运行期可读可写（`mode` 校验取值） |
| `device` | — | 只读 |
| `last_ttfa` / `last_timing` | — | 最近一次 `profile` 任务（未取消）的指标快照 |

**`timing` 逐句记录 schema**（每句一条 dict；**bench_melo.py 强依赖，改动必须同步 bench**）：

- 合成线程写入：`idx`、`text`、`synth_start`、`synth_end`、`audio_dur`（`play_start`/`play_end` 初值 `None`）；
- 播放线程回填：`play_start`、`play_end`；
- DONE 收尾（**未取消**时）派生：`ttfa`（`play_start − job.t_start`）、`synth_dur`、`wait`（`play_start − synth_end`）、`interval`（句间；句1 为 `None`）；
- **坑**：`profile=False`（默认）时 `job.timing` 为空列表（`speak()` 返回 `[]`）——要拿逐句时序必须构造时 `profile=True`。

**WAV 落盘（播放时顺手存，零额外推理）**：
- `save_wav="out.wav"`：全部句子 `np.concatenate` 拼成**一个整段 WAV**（复用流式已合成的逐句 audio，不重复推理；被打断存半截）；
- `save_chunks_dir="dir/"`：每句各存一份（`句01.wav`、`句02.wav`…）；
- 两者**相互独立**、可同时传；
- `speak_to_file()` 是另一条独立非流式路径——**不要用它补存已 `speak()` 过的文本**（音频播完不保留在内存，只能重推）。

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
6. **`profile=False` 时拿不到时序**（timing 空）；bench 依赖的 timing schema 改动要同步 `bench_melo.py`。
7. **单例跨进程不共享**：每个进程各一份；bench 按 device 拆子进程，正是因 MeloTTS BERT 为模块级单例，CPU/GPU 同进程会设备不匹配。
8. **销毁只能 `close()`**：`del tts` 只删名字引用，单例槽位 + 常驻线程仍持引用，对象不会被回收。

---

## 多方案扩展与目录演进（未来路线）

> 当前**保持平铺结构**（4 个顶层 `.py`，规模小）。本节记录"未来接入新 TTS 方案"的**已验证思考**，后续接手者直接按此演进，不必重新论证。

### 现状：为什么 sapi 与 melo 平铺共存

- `synth_sapi.py` 是**非神经对照基线**（Windows SAPI / pyttsx3，一次性落盘），与神经链路**零代码耦合**——唯一联系是 `audio/1_sapi_huihui.wav` 并排试听约定 + 目录树。
- melo 栈三件套（`tts_melo.py` / `bench_melo.py` / `preload_weights.py`）自洽：bench 只 `import tts_melo`，preload 独立。
- 规模小 → 暂不拆目录（避免 YAGNI 与"为改结构而改"的回归风险）。

### 已验证：v2 引擎的可复用边界

v2 引擎（`RealtimeTTS`）本质是**后端无关骨架 + 一处后端专属**：

| 后端无关（换方案可原样复用） | 后端专属（须替换） |
|---|---|
| `Job`（wait / mark_done / canceled / timing） | **`_synth(text) -> 整句数组`**（Melo 无原生流式，一次性合成整句） |
| 常驻合成/播放双线程、`_jobs` + `_audio_q`（背压） | timing 口径微调（见下） |
| `_gen` 代际抢占、queue/bargein 模式、interrupt/close 生命周期 | |
| timing schema、save_wav / save_chunks_dir、profile/debug | |

### 原生流式后端（MOSS-TTS-Nano / CosyVoice2）的适配点

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

### 目录演进（分两阶段，避免过早重构）

- **阶段 1（现状，不拆）**：melo 平铺；sapi 暂留根目录。若将来要挪 sapi，唯一代码改动是 `synth_sapi.py` 的 `OUTPUT_DIR`——现用 `__file__` 相对定位，挪进子目录会写错路径（变成 `sapi/audio/`），须改为项目根向上取一级。
- **阶段 2（引入第二个流式后端并验证跑通后，一次到位）**：抽后端无关骨架到 `tts/core/`，各后端子目录：

```
voice0/
├── tts/
│   ├── core/          # 后端无关骨架：Job/workers/gen/modes/lifecycle（自 tts_melo.py 抽出）
│   ├── melo/          # Melo 后端：_synth + preload（薄壳）
│   ├── moss/          # （未来）MOSS-TTS-Nano：_synth_stream
│   └── cosy/          # （未来）CosyVoice2：_synth_stream
├── sapi/              # 非神经对照基线
├── bench/             # 各后端验收脚本
├── docs/
└── README.md
```

> 阶段 2 的触发条件 = **确实引入第二个流式后端**。在此之前不拆 `tts/core`，避免为"未来可能"动正在工作的引擎。

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
python bench_melo.py --device all --profile --debug

# 裸管线（无插桩，贴近生产）：
python bench_melo.py --device cuda
```

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
D:/anaconda/envs/voice-tts/python.exe bench_melo.py --device all --profile --debug
```

> 从大环境克隆才会遇到坑 ②（jax 冲突）；从零 `conda create python=3.10` 不会。

### 常见坑（复现时）

1. **setuptools ≥ 81 会让 jieba 崩**（`pkg_resources` 被移除）→ 必须 `pip install setuptools==80.9.0`。
2. **克隆大环境带的 jax/jaxlib/ml_dtypes 与 numpy 2.2.6 不兼容**，会在 import transformers 时崩 → 卸载 `pip uninstall -y jax jaxlib ml_dtypes`（仅克隆路径需处理）。
3. **huggingface.co / raw.githubusercontent 在本机被墙/极慢** → 脚本已内置 `HF_ENDPOINT=hf-mirror.com` 与 `ghfast.top` 代理，首次下载自动走镜像。
4. **权重缓存必须落在项目内**：`tts_melo.py` / `preload_weights.py` 在 import 期就把 `HF_HOME`/`NLTK_DATA` 重定向到 `.cache/`，不要手动改。
5. **换设备 `.cache/` 不会随 git 过来**（已 gitignore）——重装环境后跑一次 `preload_weights.py` 即可自动重建，无需手动拷贝。
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
├── docs/
│   └── tts-architecture-decision.md
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
