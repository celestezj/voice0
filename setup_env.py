# -*- coding: utf-8 -*-
"""voice0 一键安装脚本（离线实时中文 TTS）。

给全新用户：已装 conda、没有任何虚拟环境、从未用过神经网络模型。
- 默认必装 melo 后端（CPU/GPU 都能跑，实时主力）；
- 可选项 cosyvoice2（音质 + 3s 音色克隆）需 NVIDIA GPU，按提示安装；
- 自动检测显卡与驱动：解析 nvidia-smi 的 CUDA 版本（驱动支持的最大 CUDA），
  驱动支持 CUDA≥12.6 → 装 cu126 版 torch；否则（驱动太旧/无 N 卡）→ 装 CPU 版
  torch（melo 照常可用，cosy 自动跳过）。安装前会把选定命令完整打印出来；
- 幂等：重复执行自动跳过已完成步骤（例：第一次没装 cosy，再跑本脚本
  会直接走到 cosy 询问）；
- 网络步骤：pip/git/conda/tqdm 自带进度条会直接透传显示；长时间无输出
  时脚本每 20s 提示"仍在进行"，避免误以为卡死。

用法（任意 python 即可，conda base 也行）：
    python setup_env.py              # 交互式
    python setup_env.py --cosy       # 自动同意装 cosy（需 GPU）
    python setup_env.py --skip-cosy  # 自动跳过 cosy

说明：不硬编码任何绝对路径，conda 位置由 `conda info --base` 动态获取。
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ENV_NAME = "voice-tts"
PY_VER = "3.10"
TORCH_VER = "2.11.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache")
THIRD = os.path.join(ROOT, "third_party")

# torch：CUDA 12.6 版（README 锁定）；CPU 版走普通 PyPI
CUDA_TORCH = ["torch==%s+cu126" % TORCH_VER, "torchaudio==%s+cu126" % TORCH_VER,
              "--index-url", "https://download.pytorch.org/whl/cu126"]
CPU_TORCH = ["torch==%s" % TORCH_VER, "torchaudio==%s" % TORCH_VER]

MELO_REPO = "https://github.com/myshell-ai/MeloTTS.git"
COSY_REPO = "https://github.com/FunAudioLLM/CosyVoice.git"

# 锁关键依赖版本（README「已锁定版本」表；setuptools 必须 <81，否则 jieba 崩 pkg_resources）
MELO_DEPS = ["sounddevice==0.5.6", "numpy==2.2.6", "transformers==4.57.6",
             "huggingface_hub==0.36.2", "jieba==0.42.1", "nltk==3.10.3",
             "scipy==1.15.3", "numba==0.67.0", "librosa==0.11.0",
             "pypinyin==0.55.0"]
# cosy 推理依赖 = README-cosyvoice2 §2 精简行 + 后端/预下载脚本实际 import 的
# modelscope(→preload_cosy.py)、onnxruntime(→campplus/speech_tokenizer)、
# wetext(→文本前端)。win32 用 CPU onnxruntime 即可（后端 _force_cpu_onnx 强制 CPU provider）。
COSY_DEPS = ["conformer==0.3.2", "diffusers==0.29.0", "pyarrow", "pyworld",
             "soundfile", "modelscope==1.20.0", "onnxruntime==1.18.0",
             "wetext==0.0.4"]

_STEPS = []


# ---------------------------------------------------------------------------
# 子进程执行：中文输出强制 UTF-8（Windows GBK 会崩）+ HF 镜像兜底
# ---------------------------------------------------------------------------
def _env():
    e = dict(os.environ)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return e


def run(args, capture=False):
    """跑命令。capture=False 时输出直接透传控制台（pip/git/tqdm 进度条可见）。"""
    cmd = args if isinstance(args, (list, tuple)) else args.split()
    if capture:
        return subprocess.run(cmd, cwd=ROOT, env=_env(), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    return subprocess.run(cmd, cwd=ROOT, env=_env())


def run_live(args, label, heartbeat=20.0):
    """流式跑网络/耗时命令；期间每 heartbeat 秒打一条"仍在进行"提示，
    避免用户误以为卡死（下载类步骤 pip/git/tqdm 自带进度条会正常滚动）。"""
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(heartbeat):
            sys.stderr.write("\n  ⏳ %s：仍在进行中（已用时 %d 秒），请耐心等待…\n"
                             % (label, int(time.time() - t0)))
            sys.stderr.flush()

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        return run(args, capture=False)
    finally:
        stop.set()
        th.join(timeout=1)


def die(msg):
    print("\n[错误] %s" % msg)
    sys.exit(1)


def run_ok_else(p, msg):
    if p.returncode != 0:
        die(msg)


# ---------------------------------------------------------------------------
# 环境检测（conda 动态定位，不硬编码路径）
# ---------------------------------------------------------------------------
_BASE = None


def conda_base():
    global _BASE
    if _BASE:
        return _BASE
    p = run(["conda", "info", "--base"], capture=True)
    if p.returncode == 0 and p.stdout.strip():
        _BASE = p.stdout.strip()
        return _BASE
    # Windows 下 conda 可能是 .bat，需经 shell 执行
    p2 = subprocess.run("conda info --base", shell=True, capture_output=True,
                        text=True, env=_env())
    if p2.returncode == 0 and p2.stdout.strip():
        _BASE = p2.stdout.strip()
        return _BASE
    return None


def env_exists():
    base = conda_base()
    return bool(base) and os.path.isdir(os.path.join(base, "envs", ENV_NAME))


def env_python():
    """voice-tts 环境的 python 绝对路径；环境尚未创建时返回 None。"""
    base = conda_base()
    if not base:
        return None
    ed = os.path.join(base, "envs", ENV_NAME)
    cand = os.path.join(ed, "python.exe") if os.name == "nt" else os.path.join(ed, "bin", "python")
    return cand if os.path.isfile(cand) else None


def env_py_cmd(args):
    """在 voice-tts 环境里跑 python 的命令列表（args 不含 python 本身）。"""
    py = env_python()
    if py:
        return [py] + args
    return ["conda", "run", "-n", ENV_NAME, "--no-capture-output", "python"] + args


def env_pip_cmd(args):
    return env_py_cmd(["-m", "pip"] + args)


def env_py(args, capture=False):
    return run(env_py_cmd(args), capture=capture)


def env_py_ok(args):
    return env_py(args, capture=True).returncode == 0


_NV = None


def nvidia_info():
    """检测并缓存 NVIDIA 驱动信息，返回 (driver_version, cuda_version)；无 nvidia-smi 时 (None, None)。
    - driver：`nvidia-smi --query-gpu=driver_version` CSV（不受系统语言影响）
    - cuda：解析 `nvidia-smi` 头部的 CUDA Version（容忍中英文）
    nvidia-smi 的 "CUDA Version" = 驱动支持的最大 CUDA 版本（PyTorch 的 +cuXXX 运行时
    只要驱动兼容即可，不需要系统装 CUDA Toolkit）。"""
    global _NV
    if _NV is not None:
        return _NV
    driver = cuda = None
    p = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture=True)
    if p.returncode == 0 and p.stdout.strip():
        driver = p.stdout.strip().splitlines()[0].strip()
    h = run(["nvidia-smi"], capture=True).stdout
    m = re.search(r"CUDA\s*(?:Version|版本)[\s:：]*(\d+\.\d+)", h)
    if m:
        cuda = float(m.group(1))
    _NV = (driver, cuda)
    return _NV


def nvidia_cuda_ok(driver, cuda):
    """驱动能否跑 cu126（本项目 torch 2.11.0+cu126 需驱动支持 CUDA ≥ 12.6）。
    优先用解析到的 CUDA 版本；解析不到时按驱动号对最低要求
    （CUDA 12.6 对应驱动：Windows ≥561.33，Linux ≥560.28）。"""
    if cuda is not None:
        return cuda >= 12.6
    if driver:
        try:
            return float(driver.split()[0]) >= (561.33 if os.name == "nt" else 560.28)
        except ValueError:
            return False
    return False


def gpu_usable():
    """本项目能否用 NVIDIA GPU = 有 nvidia-smi 且驱动支持 CUDA ≥ 12.6。"""
    driver, cuda = nvidia_info()
    return driver is not None and nvidia_cuda_ok(driver, cuda)


def _nv_desc(driver, cuda):
    if driver is None:
        return "未检测到（无 nvidia-smi）"
    cuda_s = ("%.1f" % cuda) if cuda else "未知"
    return "驱动 %s（支持 CUDA %s）" % (driver, cuda_s)


def ghfast(url):
    return url.replace("https://github.com/", "https://ghfast.top/https://github.com/")


def git_clone(url, dst, recursive=False):
    """克隆并返回是否成功；直连失败自动换 ghfast.top 镜像重试一次。"""
    args = ["git", "clone"] + (["--recursive"] if recursive else []) + [url, dst]
    if run(args).returncode == 0:
        return True
    print("  直连失败，改用 ghfast.top 镜像重试...")
    args = ["git", "clone"] + (["--recursive"] if recursive else []) + [ghfast(url), dst]
    return run(args).returncode == 0


# ---------------------------------------------------------------------------
# 幂等检测：每步先查"已完成"再决定跳不跳
# ---------------------------------------------------------------------------
def torch_state():
    """返回 (是否已装对版, 版本字符串)。已装对版 = 版本前缀对 + flavor 与驱动能力匹配。"""
    p = env_py(["-c", "import torch;print(torch.__version__)"], capture=True)
    if p.returncode != 0:
        return False, None
    v = p.stdout.strip()
    ok = v.startswith(TORCH_VER) and ("+cu126" in v) == gpu_usable()
    return ok, v


def melo_installed():
    return env_py_ok(["-c", "import melo"])


def deps_pinned():
    """关键依赖是否已锁定（sounddevice/numpy/transformers/setuptools 为哨兵）。"""
    code = ("import sounddevice,numpy,transformers,setuptools,sys;"
            "ok=(sounddevice.__version__=='0.5.6' and numpy.__version__=='2.2.6'"
            " and transformers.__version__=='4.57.6' and setuptools.__version__=='80.9.0');"
            "sys.exit(0 if ok else 1)")
    return env_py(["-c", code], capture=True).returncode == 0


def _snapshot_has(slug, filename):
    """权重是否已下载。兼容 huggingface_hub 两种缓存布局：
    - 新版（0.30+ 默认）：.cache/hf/hub/models--<slug>/snapshots/*/<file>
    - 旧版：              .cache/hf/models--<slug>/snapshots/*/<file>
    """
    for base in (os.path.join(CACHE, "hf", "hub"), os.path.join(CACHE, "hf")):
        d = os.path.join(base, "models--%s" % slug, "snapshots")
        if os.path.isdir(d) and any(
                os.path.isfile(os.path.join(d, rev, filename)) for rev in os.listdir(d)):
            return True
    return False


def melo_weights_ok():
    return _snapshot_has("myshell-ai--MeloTTS-Chinese", "checkpoint.pth")


def cosy_weights_ok():
    return _snapshot_has("FunAudioLLM--CosyVoice2-0.5B", "llm.pt")


def cosy_installed():
    """cosy 是否已装：仓库克隆 + transformers pin 都在即算装过（运行期由 speak_example 注入路径）。"""
    return (os.path.isdir(os.path.join(THIRD, "CosyVoice"))
            and os.path.isdir(os.path.join(CACHE, "pinned_transformers")))


# ---------------------------------------------------------------------------
# 安装步骤（状态机）
# ---------------------------------------------------------------------------
def step(name):
    def deco(fn):
        _STEPS.append((name, fn))
        return fn
    return deco


def ask_yesno(prompt):
    while True:
        try:
            ans = input(prompt + " [y/N] ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("  请输入 y 或 n。")


@step("检查环境前提（conda / git / 显卡 / 驱动）")
def s1():
    if not conda_base():
        die("未找到 conda。请先安装 Miniconda/Anaconda，再从 Anaconda Prompt 运行本脚本。")
    if not shutil.which("git"):
        die("未找到 git。请先安装 Git for Windows（默认安装选项即可）。")
    print("  conda: %s" % conda_base())
    print("  git:   %s" % shutil.which("git"))
    driver, cuda = nvidia_info()
    print("  NVIDIA: %s" % _nv_desc(driver, cuda))
    print("  本项目 torch %s 的 CUDA 构建范围：cu126（锁定、已验证，需驱动支持 CUDA≥12.6）" % TORCH_VER)
    if driver is None:
        print("          → 未检测到 nvidia-smi：装 CPU 版 torch（melo 可跑，cosy 不可用）")
    elif nvidia_cuda_ok(driver, cuda):
        print("          → 你的驱动支持 CUDA≥12.6：cu126 可用（GPU 与 cosy 都可用）")
    else:
        print("          → 你的驱动不支持 CUDA 12.6：装 CPU 版 torch；想用 GPU 请升级驱动")


@step("创建 voice-tts 环境 (python %s)" % PY_VER)
def s2():
    if env_exists():
        print("  环境已存在，跳过。")
        return
    run_live(["conda", "create", "-n", ENV_NAME, "python=%s" % PY_VER, "-y"],
             "创建 conda 环境")


@step("安装 torch %s（按显卡驱动选 CUDA/CPU 版）" % TORCH_VER)
def s3():
    ok, v = torch_state()
    if ok:
        print("  已安装 %s，跳过。" % v)
        return
    if gpu_usable():
        print("  驱动支持 CUDA≥12.6 → 安装 cu126 版（本项目锁定、已验证）。")
        print("  安装命令：")
        print("    python -m pip install " + " ".join(CUDA_TORCH))
        run_ok_else(run_live(env_pip_cmd(["install"] + CUDA_TORCH), "安装 CUDA 版 torch"),
                    "torch 安装失败（请确认能访问 download.pytorch.org）。")
    else:
        print("  无 NVIDIA 或驱动不支持 CUDA 12.6 → 安装 CPU 版（melo 可跑，稍慢）。")
        print("  安装命令：")
        print("    python -m pip install " + " ".join(CPU_TORCH))
        run_ok_else(run_live(env_pip_cmd(["install"] + CPU_TORCH), "安装 CPU 版 torch"),
                    "torch 安装失败（请确认网络）。")


@step("安装 melo 后端（默认必装）+ 锁定依赖")
def s4():
    if not melo_installed():
        dst = os.path.join(CACHE, "MeloTTS")
        if os.path.isdir(dst):
            print("  检测到不完整/不可用的 .cache/MeloTTS，删除后重克隆...")
            shutil.rmtree(dst)
        if git_clone(MELO_REPO, dst):
            run_ok_else(run_live(env_pip_cmd(["install", "-e", dst]), "安装 melo（源码 editable）"),
                        "melo 安装失败。")
        else:
            print("  github 直连与 ghfast.top 镜像都失败，退回 PyPI 版 melotts（0.1.1）...")
            run_ok_else(run_live(env_pip_cmd(["install", "melotts"]), "安装 melotts"),
                        "melotts 安装失败（网络不通？）。")
    else:
        print("  已安装 melo，跳过。")
    # 锁定关键依赖版本（README 版本锁定表；setuptools<81 否则 jieba 崩）
    if deps_pinned():
        print("  关键依赖版本已锁定，跳过。")
    else:
        run_ok_else(run_live(env_pip_cmd(["install"] + MELO_DEPS), "锁定依赖版本"),
                    "依赖锁定失败。")
        run_ok_else(run_live(env_pip_cmd(["install", "setuptools==80.9.0"]), "锁定 setuptools"),
                    "setuptools 锁定失败。")


@step("预下载 melo 权重（.cache/hf，首次联网）")
def s5():
    if melo_weights_ok():
        print("  权重已存在，跳过。")
        return
    run_ok_else(run_live(env_py_cmd(["preload_weights.py"]), "下载 melo 权重（首次约 700MB+）"),
                "melo 权重下载失败（检查网络）。")


@step("cosyvoice2（可选，需 NVIDIA GPU + CUDA 12.6 驱动）")
def s6():
    driver, cuda = nvidia_info()
    if driver is None:
        print("  未检测到 NVIDIA 显卡/驱动：cosy 需 GPU（CPU 极慢，实测不可用），跳过。")
        return
    if not nvidia_cuda_ok(driver, cuda):
        print("  驱动不支持 CUDA 12.6（当前 %s）：cosy 需 GPU + cu126 驱动，跳过。"
              % (("CUDA %.1f" % cuda) if cuda else driver))
        return
    if cosy_installed():
        print("  已安装 cosy，跳过。")
        return
    if "--cosy" in sys.argv:
        want = True
    elif "--skip-cosy" in sys.argv:
        want = False
    else:
        want = ask_yesno("是否额外安装 cosyvoice2（音质更好 + 3s 音色克隆，需 ~3-4.6GB 显存）？")
    if not want:
        print("  跳过 cosy。以后想装：重新运行本脚本，会直接走到这一步。")
        return
    dst = os.path.join(THIRD, "CosyVoice")
    if os.path.isdir(dst):
        print("  检测到不完整的 third_party/CosyVoice，删除后重克隆...")
        shutil.rmtree(dst)
    if not git_clone(COSY_REPO, dst, recursive=True):
        die("cosy 克隆失败（含 Matcha-TTS 子模块）。请检查网络后重试，或先跳过 cosy（melo 不受影响）。")
    run_ok_else(run_live(env_pip_cmd(["install"] + COSY_DEPS), "安装 cosy 推理依赖"),
                "cosy 依赖安装失败。")
    if not cosy_weights_ok():
        run_ok_else(run_live(env_py_cmd(["preload_cosy.py"]), "下载 cosy 权重（~2.4GB）"),
                    "cosy 权重下载失败（检查网络）。")
    if not os.path.isdir(os.path.join(CACHE, "pinned_transformers")):
        run_ok_else(run_live(env_py_cmd(["setup_cosy_pinned.py"]),
                             "落盘 cosy 专属 transformers 4.51.3（~94MB）"),
                    "transformers pin 落盘失败。")
    print("  cosy 安装完成。")


@step("端到端验证（melo 合成一句落盘 WAV）")
def s7():
    p = run_live(env_py_cmd(["verify_install.py"]), "端到端验证（首次加载模型约需 10-40 秒）")
    if p.returncode != 0:
        die("验证失败：melo 合成未通过。请查看上方输出，必要时重跑本脚本。")
    print("  验证通过。")


BANNER = r"""
============================================================
  voice0 一键安装 —— 离线实时中文 TTS（melo 必装 / cosy 可选）
  - 自动检测显卡，安装匹配的 torch
  - 重复执行自动跳过已完成步骤（想补装 cosy 直接再跑一次）
  - pip/git 自带进度条会实时显示；长时间无输出时脚本会提示"仍在进行"
============================================================
"""


def main():
    print(BANNER)
    for i, (name, fn) in enumerate(_STEPS, 1):
        print("\n[%d/%d] %s" % (i, len(_STEPS), name))
        print("-" * 72)
        fn()
    print("\n" + "=" * 72)
    print("安装完成！接下来：")
    print("  conda activate voice-tts")
    print("  python speak_example.py       # 交互式播报 demo")
    print("  python verify_install.py      # 随时重新验证")
    print("（cosy 本次若未装，重跑本脚本即可补装）")


if __name__ == "__main__":
    main()
