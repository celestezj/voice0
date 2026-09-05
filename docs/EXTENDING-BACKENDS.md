# 扩展新后端指南（melo / cosy 之外）

> 面向「将来接入第三个 TTS 后端」（如 GPT-SoVITS、edge-tts 等）的
> **实操步骤手册**。回答三个问题：后端长什么样、引擎怎么调它、新后端要改哪几处。
>
> 关联文档（动手前按需精读）：
> - `README.md`「常驻引擎 v2」（§1-§8）= 引擎完整设计，**引擎语义以它为准**。
> - `docs/README-cosyvoice2.md` = cosy 接入的最全参考实现（含上游补丁）。
> - `docs/tts-architecture-decision.md` = 选型背景（为什么是 melo + cosy）；**「MOSS-TTS-Nano 探针」节 = 一个已实测排除的候选**（RTF 1.25 非实时、fp16/nq=8 无效、流式饿死），写新后端前先看，省得重复踩。

---

## 1. 先搞懂：后端在引擎里的位置

```
RealtimeTTS（tts/core/engine.py，后端无关骨架）
   └─ get_backend("name", device=..., **cfg)   # tts/core/backend.py 注册表 + 惰性 import
        └─ <Name>Backend（tts/<name>/backend.py，后端专属）
             ├─ load()          # 惰性建模型，失败抛 BackendNotInstalledError
             ├─ synth_stream()  # 逐块生成器（引擎 submit/speak 走这里）
             ├─ synth()         # 整句一次合成（speak_to_file 走这里）
             └─ close()         # 释放模型
```

**一句话：换后端 ≈ 重写"合成边界"这一层，骨架复用约 80%**（README「已验证：v2 引擎的可复用边界」）。
`Job` / 常驻双线程 / `_gen` 抢占 / queue-bargein / 落盘 / 归一化，全部后端无关，白拿。

**两层流式先分清**（避免接口理解错位，详见 README「流式架构」节）：
- **引擎层（句子级）**：任何后端都流式——引擎按句分块、逐句合成、逐句播放、句间无缝。后端不用管。
- **后端层（句内块级）**：可选能力。整句一块（melo）或逐 token 块（cosy `stream=True`）。决定 `synth_stream` 怎么 yield。

---

## 2. 接口契约（TTSBackend ABC）

`tts/core/backend.py` 的 `TTSBackend` 是**编译期契约**（`@abstractmethod` 未实现完无法实例化）：

```python
class TTSBackend(ABC):
    name = ""          # 类属性：注册表 key，须与 _BACKEND_MODULES 的 key 一致
    sr = 44100         # 类属性：采样率。引擎所有时长派生用 self._sr，后端定死即可

    @abstractmethod
    def load(self): ...            # 惰性 import 模型并初始化；缺依赖抛 BackendNotInstalledError
    @abstractmethod
    def synth_stream(self, text, *, speed=1.0, normalize=None): ...   # 生成器，逐块 yield
    @abstractmethod
    def synth(self, text, *, speed=1.0, normalize=None): ...          # 整句一次，返回完整数组
    @abstractmethod
    def close(self): ...           # 释放模型引用（torch 显存交给 GC）
```

引擎实际怎么调（全在 `tts/core/engine.py`）：
- 构造期：`get_backend(...)` → `.load()` → 预热一次 `synth("你好。")`。
- 运行期：`submit()/speak()` → 合成线程对**每句**调 `synth_stream()`（持 `_synth_lock`）；`speak_to_file()` → 直接调 `synth()`（持锁）。
- 销毁：`close()` → 后端 `.close()`。

**块数据约定**：`synth_stream` 每块必须是 **float32 numpy 一维数组、幅度 [-1,1]**（播放线程与
`save_wav_np` 都按这个假设消费）。`synth` 返回同样的整句数组。

---

## 3. 步骤总览（checklist）

1. **建模块**：`tts/<name>/backend.py`，定义 `<Name>Backend(TTSBackend)`。
2. **注册**：`tts/core/backend.py` 的 `_BACKEND_MODULES` 加一行 `"name": ("tts.name.backend", "NameBackend")`。
3. **实现 `load()`**：惰性 import；依赖缺失抛 `BackendNotInstalledError`（带安装提示）。
4. **实现合成**：按「第 5 节」选非流式或原生流式模板。
5. **环境前置**（可选）：按「第 6 节」在模块顶层 setdefault `HF_HOME`/`HF_ENDPOINT` 等。
6. **装依赖**：跑 `pip install ...`，或写进 `setup_env.py`（加一个 deps 列表 / 步骤）。
7. **权重预载**：写/改 `preload_*.py`，保证离线零网络；并在 `setup_env.py` 幂等检测里登记。
8. **跑通 + 验收**：`speak_example.py` 或新写 `bench/bench_<name>.py`，**Whisper 转写/试听**（验收纪律，见 §9）。
9. **文档**：`docs/README-<name>.md` + 在本文档 / README 登记。

**先跑通第 1-4 步（核心）再回头补 5-9**。

---

## 4. 最小骨架（可直接抄）

```python
# tts/<name>/backend.py
# -*- coding: utf-8 -*-
import os
import numpy as np

# 本文件位于 tts/<name>/backend.py → 项目根向上取 3 级（melo/cosy 同款，勿硬编码路径）
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 权重缓存重定向到项目内（必须在 import 模型库之前）
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 首次下载走镜像

from ..core.audio import normalize_audio            # 可选：复用响度归一化
from ..core.backend import BackendNotInstalledError, TTSBackend


class FooBackend(TTSBackend):
    name = "foo"
    sr = 24000            # 你的模型实际采样率

    def __init__(self, device="auto", debug=False):
        self._device = device
        self._debug = debug
        self._model = None

    def load(self):
        try:
            from foo import FooTTS          # 所有重量 import 都放 load() 里
        except ImportError as e:
            raise BackendNotInstalledError(
                "FooTTS 依赖缺失：pip install foo-tts 后重试（%s）" % e) from e
        self._model = FooTTS(device=self._device)

    def synth(self, text, *, speed=1.0, normalize=None):
        a = self._model.tts_to_file(text, speed=speed, output_path=None)   # float32 [-1,1]
        if normalize:
            a = normalize_audio(a, self.sr, normalize)
        return a

    def synth_stream(self, text, *, speed=1.0, normalize=None):
        yield self.synth(text, speed=speed, normalize=normalize)           # 非流式：整句一块

    def close(self):
        self._model = None
```

注册（`tts/core/backend.py`）：

```python
_BACKEND_MODULES = {
    "melo": ("tts.melo.backend", "MeloBackend"),
    "cosy": ("tts.cosy.backend", "CosyBackend"),
    "foo":  ("tts.foo.backend",  "FooBackend"),      # ← 加这一行
}
```

之后引擎一行不改成：`RealtimeTTS(backend="foo")`。**运行期换后端 = 销毁重建**（`__new__` 检测
`backend` 变更 → `close()` 旧实例重建；`tts.backend` 是只读属性，不能原地换）。

---

## 5. 合成实现：两类模板

### A. 非流式（melo 型）—— 无块级流式能力

```python
def synth_stream(self, text, *, speed=1.0, normalize=None):
    """整句一次合成完，产出一整块。行为与 melo 一致。"""
    yield self.synth(text, speed=speed, normalize=normalize)
```

- **TTFA 口径**：首块即整句，TTFA ≈ 首句合成耗时。目标 <1s 就得选够快的模型。
- 块时长 = 整句，播放线程按 ~50ms 子块写声卡，无感。

### B. 原生流式（cosy `stream=True` 型）—— 边生成边吐块

```python
def synth_stream(self, text, *, speed=1.0, normalize=None):
    for chunk in self._gen(text, speed=speed):       # 模型逐 token/逐块 yield
        if normalize:
            chunk = normalize_audio(chunk, self.sr, normalize)   # 块级归一化
        yield chunk

def synth(self, text, *, speed=1.0, normalize=None):
    a = np.concatenate(list(self._gen(text, speed=speed)))       # 聚合整句
    if normalize:
        a = normalize_audio(a, self.sr, normalize)               # 句级归一化
    return a
```

- **TTFA 口径**：首块到播放时刻；`wait`/`interval` 按句聚合口径引擎会重算（见 README §4）。
- **RTF>1 的坑（血泪教训）**：原生流式要求合成**快于**实时（RTF<1）。若目标机器 RTF≈1.1-1.3，
  逐块播必然**块间饿死停顿 + 块拼接缝**（每块是独立解码的声学接缝）。对策（cosy 现状）：
  **默认整句一次解码、单块播放、句内无缝**，把原生流式藏在 `stream=True` 开关后。

### 关键决策：要不要开原生流式？

| 条件 | 结论 |
|---|---|
| 目标卡 RTF<1（合成快于实时） | 可原生流式，首包延迟低 |
| RTF≈1 或 >1 | **默认非流式**（整句单块），流式放开关后 |
| 完全无 GPU（纯 CPU） | 只考虑 melo 这类 CPU 实时模型 |

---

## 6. 环境前置与上游补丁（melo/cosy 同套模式）

**路径重定向**（必须在 `import` 任何模型库之前）——`tts/core/backend.py` 已设过的不重复设：

```python
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))     # 权重缓存落项目内
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")                    # huggingface.co 被墙
os.environ.setdefault("NLTK_DATA", os.path.join(_PROJECT_DIR, ".cache", "nltk_data"))  # 需要 g2p 语料时
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(_PROJECT_DIR, ".cache", "modelscope"))  # 走 modelscope 通道的资源（如 wetext FST）
```
> **铁律：所有下载资源必须重定向进项目 `.cache/`**，绝不落主目录（`~/.cache/...`）。cosy 曾漏了
> modelscope 的 wetext FST（落 `~/.cache/modelscope`），已修。新后端凡有下载，先在模块顶层列全
> env 前置并核对落点。

**transformers pin（仅 cosy 需要）**：cosy 因上游 issue #1546 必须 `transformers==4.51.3`，
用 `setup_cosy_pinned.py` 落盘 vendored 副本 + 模块顶层注入 `sys.path`。**新后端若对 transformers
无特殊要求，直接用 main env 版本即可，别学 cosy 的 pin**。若你的模型也需要特定版本，照 cosy
的 `_pin_transformers()` 模式（注入 + 版本校验 + 明确报错，而不是默默产杂音）。

**上游仓库补丁模式**（cosy 示范）：
- 上游 clone 进 `third_party/`，模块顶层把 `third_party/CosyVoice` 和子模块注入 `sys.path`（须在 `import cosyvoice` 之前）。
- 在**合成边界一层封死上游坑**，不 fork 上游：
  - onnxruntime 强制 CPU provider（`_force_cpu_onnx`）；
  - 硬编码 `tqdm` 替换为透传（`_silence_tqdm`）；
  - 联网 metadata 检查替换为本地缓存（`_ensure_wetext_local`）；
  - 生成上限收紧（`_wrap_max_ratio` 包 `llm.inference`）。
- 每处补丁都是**"何时、为什么、不改会怎样"三要素**写清在注释里，方便后人验证。

---

## 7. 引擎对接点：注意「cosy 专属硬编码」

引擎把「后端专属配置参数」透传的方式目前**写死为 cosy**，新后端要用这些参数就得动引擎两处：

**① `__new__` 的变更判定**（`tts/core/engine.py`，约 line 74-80）：
```python
want_voice = voice if want_bk == "cosy" else None
want_ratio = max_speech_ratio if want_bk == "cosy" else None
want_stream = bool(stream) if want_bk == "cosy" else None
```
**② `__init__` 的 backend_cfg 构造**（约 line 133-140）：
```python
backend_cfg = {}
if self._backend_name == "cosy":
    backend_cfg["voice"] = self._voice
    if self._max_speech_ratio is not None:
        backend_cfg["max_speech_ratio"] = self._max_speech_ratio
    if self._stream is not None:
        backend_cfg["stream"] = self._stream
self._backend = get_backend(self._backend_name, device=self._device,
                            debug=self._debug, **backend_cfg)
```

**通用透传（任何后端都拿得到）**：`device`、`debug`。后端 `__init__` 签名至少接受 `(device="auto", debug=False)`。

**要动引擎的情况**：
- 新后端有 `voice` / `stream` / 其他专属参数 → 把上面两处的 `== "cosy"` 扩展成你后端名
  （或重构为「后端声明的 cfg 白名单」，见下）。
- 只想先跑通 → 后端参数全走默认，`__init__` 只收 `device`/`debug`，一行引擎都不用改。

> 建议（将来有第 3 个后端时再动手）：给 `TTSBackend` 加一个类属性如
> `engine_params = ("voice", "stream", "max_speech_ratio")`，引擎按 `self._backend_class.engine_params`
> 透传，替换 `if name == "cosy"` 的硬编码。现在只有 2 个后端，不值得先动。

**`mode` / `speed` / `normalize`** 是引擎级参数，与后端无关，后端只要实现 `speed=`/`normalize=` 关键字即可（ABC 已按 keyword-only 声明）。

---

## 8. 权重预下载（离线约束）

项目硬指标：**权重缓存后运行期零网络请求**。melo/cosy 各自有 `preload_*.py`（项目根）：
下载 → `.cache/hf` → 首次联网、之后命中缓存。

新后端二选一：
- 复用现有 preload 脚本（加一个分支）；或
- 新写 `preload_<name>.py`（照 `preload_weights.py` 的 `hf_hub_download`/`snapshot_download` 模式）。

同时 `setup_env.py` 幂等检测要登记新 slug，否则重复跑会重新下载：
```python
def foo_weights_ok():
    return _snapshot_has("someuser--FooTTS", "checkpoint.pth")   # HF repo id "someuser/FooTTS" 的 `/` → `--`
```
（`_snapshot_has` 已兼容 huggingface_hub 新旧两种缓存布局，不用改。）

依赖安装：把新后端 deps 加进 `setup_env.py`（如 `FOO_DEPS = [...]` + 一个新 step），或手动
`pip install`。**若新后端要特殊 transformers，绝不能让 setup_env 覆盖 cosy 的 pin 环境**（melo/cosy
共存时注意 pip 解析——先满足已锁版本，别让新依赖把 torch/transformers 换掉）。

---

## 9. bench 与验收纪律

- **bench**：`bench/bench_<name>.py` 照 `bench_melo.py` / `bench_cosy.py` 抄（`--device`/`--profile`），
  时序可视化用 `bench/console_gantt.py`；实测区间自动回写 README 验收表。
- **验收纪律（血泪教训）**：时序/显存达标 ≠ 输出正确。新后端或改动后必须
  **Whisper 转写或试听**内容正确性，不能只看 TTFA/RTF。
- **文档**：`docs/README-<name>.md` 至少覆盖：安装、API、已知限制、验收实测、踩过的坑。

---

## 10. 排障优先读哪

| 症状 | 先查 |
|---|---|
| 引擎行为（mode/interrupt/流式/落盘/抢占） | README「常驻引擎 v2」（§1-§8） |
| 后端不工作 / 依赖 / 上游补丁 | 本文档 §4-§6 + 对应 `docs/README-<name>.md` |
| cosy 集成细节（pin/onnx/wetext/RTF） | `docs/README-cosyvoice2.md`（最全参考实现） |
| 为什么选 melo+cosy | `docs/tts-architecture-decision.md` |
| 接入后的开发流程 | `docs/ai-project-methodology.md` |
