# -*- coding: utf-8 -*-
"""CosyVoice2-0.5B 权重预下载（HF 官方仓库 → 项目内 .cache/hf）。

用法：
    python preload_cosy.py

只做下载、不加载模型。下载完即可被 tts/cosy/backend.py 命中缓存复用
（backend 内部用 snapshot_download 同样路径，零二次下载）。
"""
import os

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 与 tts/melo/backend.py 同一套 env 前置：缓存落在项目内、走镜像
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"


def main():
    from huggingface_hub import snapshot_download
    print("[preload_cosy] 下载仓库: %s" % REPO_ID)
    print("[preload_cosy] HF_HOME=%s" % os.environ["HF_HOME"])
    path = snapshot_download(repo_id=REPO_ID)
    print("[preload_cosy] 完成 -> %s" % path)

    # wetext 文本前端（中文数字/单位归一化）的 FST 资源，走 modelscope 默认缓存；
    # backend 已 patch 缓存存在时零联网，这里确保首次就备好。
    print("[preload_cosy] 下载 wetext 文本前端 FST ...")
    from modelscope import snapshot_download as ms_snapshot
    wpath = ms_snapshot("pengzhendong/wetext")
    print("[preload_cosy] wetext 完成 -> %s" % wpath)


if __name__ == "__main__":
    main()
