# VITS（第三后端，多音色）

> 参照 Alife 项目（github.com/BDFFZI/Alife）的 multi-speaker VITS 接入 voice0。
> 与前两后端的定位差异：**melo**=中文标准女声实时主力；**cosy**=音质+3s 克隆；
> **vits**=**804 个动漫角色音色**（赛马娘 umamusume / 派蒙 / 原神角色等），
> 中文文本按**日式罗马音**发音（`zh_ja_mixture_cleaners` 把中文转成 romaji 标注）。

## 安装（一次下载，长期缓存）

```bash
python preload_vits.py
```

- 从 GitHub release 下载 `VITS.zip`（约 412MB）→ 解压到 `.cache/vits/VITS/`。
- 幂等：`G_953000.pth` 已存在即跳过（重跑安全）。
- 下载源：`https://ghfast.top/...`（代理，本机 ~1.4MB/s）优先，GitHub 直连兜底
  （直连极慢 ~150KB/s，被墙时换源）。
- 权重全在项目 `.cache/vits/`，**运行期零网络**（无 HF/modelscope 依赖）。

依赖：torch + jieba/pypinyin/cn2an/Unidecode（voice-asr / voice-tts 环境已具备）。
推理代码随模型发布（`models.py/text/commons/utils/monotonic_align`），无需额外 pip 包。
`monotonic_align` 走 numba JIT（首次加载 ~4.6s 含编译，之后 ~1.4s）。

## 音色配置

`voice` 参数（仅 vits/cosy 后端有含义）：
- `None`（默认）→ 551 = 派蒙
- `int` → speaker_id（0~803）
- `str` → 经 `speakers_list.txt` 反查名字（如 `"特别周"` / `"黄金船"`），找不到报错并列出前几个

换音色 = 换 `sid`，同一模型即时切换、零额外推理成本；引擎侧 `voice` 变更 = 销毁重建
（与 cosy 同语义）。音色表见 `.cache/vits/VITS/speakers_list.txt`（`id:名字`，804 条）。

## API 用法

```python
from tts import RealtimeTTS
tts = RealtimeTTS(device="cuda", backend="vits")                    # 默认 551=派蒙
tts = RealtimeTTS(device="cuda", backend="vits", voice=0)           # 特别周
tts = RealtimeTTS(device="cuda", backend="vits", voice="黄金船")    # 按名字反查
tts = RealtimeTTS(device="cuda", backend="vits",
                  voice=0, noise_scale=0.6, noise_scale_w=0.668, length_scale=1.2)
tts.submit("你好。")                      # 非阻塞实时（同 melo）
tts.speak_to_file("你好。", "audio/vits_out.wav")
tts.close()
```

交互试玩：`python speak_example.py` → 选 `3) vits` → 选默认或挑音色。

## 验收实测（RTX 2070 SUPER 8GB，`bench_vits.py --device cuda --profile`）

与 melo（`bench_melo.py`）同 3 用例、同文本对比，TTFA 均 **< 0.2s，vits 全面快于 melo**：

| 用例 | vits TTFA | melo TTFA | vits 峰值显存 | melo 峰值显存 |
|---|---|---|---|---|
| 短（2 句） | 0.09–0.13s | 0.17s | 0.56 GB | ~1.1 GB |
| 中（3 句） | 0.10–0.12s | 0.15s | 0.56 GB | ~1.1 GB |
| 长（3 句） | 0.14–0.16s | 0.37s | 0.56 GB | ~1.1 GB |

- 3 音色（默认551派蒙 / 特别周0 / 黄金船6）TTFA 与显存几乎一致 —— 换音色零额外成本。
- 模型加载：首次 4.6s（含 numba JIT 编译），二次 1.3–1.5s。
- Whisper/funasr 转写：3 音色×多句内容**全部正确**（长句逐字还原，个别同音字为 ASR 自选）；
  音色确实不同（峰值/听感差异，试听 `audio/2_vits_cuda_*.wav`）。

## 已知限制

- **中文按日式罗马音发音**（`zh_ja_mixture_cleaners`）：这是该日系音色模型的固有前端——
  中文会带上"日式动漫腔"，与 melo 的标准普通话不同，属设计如此（Alife 同款行为）。
- **裸文本不合成**：`zh_ja_mixture_cleaners` 只认 `[ZH]...[/ZH]` 块，无包裹的文本产出 0 音素。
  后端内部已自动 `[ZH]%s[ZH]` 包裹，外部使用者无需关心。
- **非流式**：VITS 无 token 级流式接口，`synth_stream` = 整句一块（同 melo）。实时 <1s 达标
  靠的是整句单次前向极快（~0.1s），不是流式。
- 推理参数（`noise_scale`/`noise_scale_w`/`length_scale`）默认 Alife 原值；`speed` 经
  `length_scale/speed` 生效（speed>1 更快）。
