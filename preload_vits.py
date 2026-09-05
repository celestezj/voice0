# -*- coding: utf-8 -*-
"""VITS 多音色模型权重预下载（GitHub release → 项目内 .cache/vits）。

用法：
    python preload_vits.py

模型来源：Alife 项目（github.com/BDFFZI/Alife）发布的多说话人 VITS 包
（VITS.zip，约 412MB，含推理代码 models.py/text/commons/utils +
权重 G_953000.pth + config.json + speakers_list.txt 音色表）。
下载后解压到 .cache/vits/VITS/，tts/vits/backend.py 从这里加载，运行期零网络。
"""
import os
import shutil
import sys
import time
import urllib.request
import zipfile

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_VITS_CACHE = os.path.join(_PROJECT_DIR, ".cache", "vits")
_VITS_DIR = os.path.join(_VITS_CACHE, "VITS")
_ZIP_PATH = os.path.join(_VITS_CACHE, "VITS.zip")

# 模型落点哨兵：G_953000.pth 存在即视为已就绪（幂等）
_SENTINEL = os.path.join(_VITS_DIR, "model", "G_953000.pth")

# 与 CLAUDE.md 记录一致：github.com 直连极慢/被墙，ghfast.top 代理可用
# （本机实测：直连 ~150KB/s，ghfast ~4MB/s）。ghfast 优先，直连兜底。
VITS_URL = "https://github.com/BDFFZI/Alife/releases/download/VITS/VITS.zip"
VITS_URL_GHFAST = "https://ghfast.top/" + VITS_URL
VITS_URLS = [VITS_URL_GHFAST, VITS_URL]


def _fmt(n):
    if n >= 1 << 30:
        return "%.2f GB" % (n / (1 << 30))
    if n >= 1 << 20:
        return "%.1f MB" % (n / (1 << 20))
    return "%.1f KB" % (n / (1 << 10))


def _download(url, dst):
    """流式下载到 dst，带进度。失败抛异常（由调用方换 URL 重试）。"""
    t0 = time.perf_counter()
    print("[preload_vits] 下载 %s" % url)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if now - last > 2:  # 每 2s 刷一次进度
                    pct = "%.1f%%" % (done / total * 100) if total else "%s" % _fmt(done)
                    print("  %s / %s (%s)  %.1f MB/s"
                          % (_fmt(done), _fmt(total) if total else "?", pct,
                             done / (1 << 20) / (now - t0)), flush=True)
                    last = now
    shutil.move(tmp, dst)
    print("[preload_vits] 下载完成 %s，总耗时 %.1fs" % (_fmt(os.path.getsize(dst)), time.perf_counter() - t0))


def _extract():
    """解压 VITS.zip 到 .cache/vits/VITS/（兼容 zip 有无顶层 VITS/ 目录两种布局）。"""
    print("[preload_vits] 解压 %s -> %s" % (_ZIP_PATH, _VITS_DIR))
    stage = os.path.join(_VITS_CACHE, "_stage")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    with zipfile.ZipFile(_ZIP_PATH) as z:
        z.extractall(stage)
    # 顶层 VITS/ 目录有无都归一到 .cache/vits/VITS/
    src = os.path.join(stage, "VITS") if os.path.isdir(os.path.join(stage, "VITS")) else stage
    if os.path.isdir(_VITS_DIR):
        shutil.rmtree(_VITS_DIR)
    shutil.move(src, _VITS_DIR)
    shutil.rmtree(stage, ignore_errors=True)


def main():
    if os.path.isfile(_SENTINEL):
        print("[preload_vits] 已就绪，跳过：%s" % _SENTINEL)
        return
    os.makedirs(_VITS_CACHE, exist_ok=True)
    if not os.path.isfile(_ZIP_PATH):
        for url in VITS_URLS:
            try:
                _download(url, _ZIP_PATH)
                break
            except Exception as e:
                print("[preload_vits] %s 失败（%s），换下一个源 ..." % (url, e))
        else:
            sys.exit("所有下载源均失败，请检查网络后重试")
    _extract()
    if not os.path.isfile(_SENTINEL):
        sys.exit("解压后未找到 %s，VITS.zip 内容可能不符预期" % _SENTINEL)
    print("[preload_vits] 完成 -> %s（可用音色见 %s）"
          % (_VITS_DIR, os.path.join(_VITS_DIR, "speakers_list.txt")))


if __name__ == "__main__":
    main()
