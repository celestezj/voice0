# -*- coding: utf-8 -*-
"""一次性权重预下载脚本（首次联网，之后运行期零网络）。

MeloTTS 中文（ZH→ZH_MIX_EN）链路实际需要的全部模型：
  1) 六个语种前端 tokenizer（模块 import 期就会联网加载，必须先缓存好）
     - tohoku-nlp/bert-base-japanese-v3          (japanese)
     - bert-base-uncased                         (english)
     - bert-base-multilingual-uncased            (chinese_mix，仅 tokenizer 部分)
     - kykim/bert-kor-base                       (korean)
     - dbmdz/bert-base-french-europeana-cased    (french)
     - dccuchile/bert-base-spanish-wwm-uncased   (spanish)
  2) bert-base-multilingual-uncased 完整模型（~670MB，ZH 推理时跑 BERT 特征用）
  3) myshell-ai/MeloTTS-Chinese 的 config.json + checkpoint.pth（VITS 合成模型）

huggingface.co 直连在本机被墙，走 hf-mirror.com 镜像（可用环境变量 HF_ENDPOINT 覆盖）。
所有文件落在项目内 .cache/hf（与 tts_melo.py 的 HF_HOME 重定向一致）。
"""
import os
import time
import urllib.request

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NLTK_DATA", os.path.join(_PROJECT_DIR, ".cache", "nltk_data"))

# g2p_en 在 import/推理期自动下载 NLTK 语料（raw.githubusercontent 在本机极慢/易断），用代理预下载。
# 所需资源：cmudict（词典）、averaged_perceptron_tagger_eng（英文词性标注）、punkt_tab（分句分词）。
NLTK_PACKAGES = [
    "corpora/cmudict",
    "taggers/averaged_perceptron_tagger_eng",  # NLTK 3.10 pos_tag 实际使用
    "taggers/averaged_perceptron_tagger",      # g2p_en 用旧名做存在性检查，缺它会联网重试
    "tokenizers/punkt_tab",
]
NLTK_ZIP_URL = ("https://ghfast.top/"
                "https://raw.githubusercontent.com/nltk/nltk_data/"
                "gh-pages/packages/%s.zip")

from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForMaskedLM,
    AutoTokenizer,
)


def _fmt(n):
    if n >= 1 << 30:
        return "%.2f GB" % (n / (1 << 30))
    if n >= 1 << 20:
        return "%.1f MB" % (n / (1 << 20))
    return "%.1f KB" % (n / (1 << 10))


def dl_tokenizer(model_id, who):
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    dt = time.perf_counter() - t0
    cached = tok.save_pretrained(os.environ["HF_HOME"]) if False else tok
    print("[preload] tokenizer %-42s <- %s  (%.1fs)" % (who, model_id, dt))
    return tok


def ensure_nltk_packages():
    """NLTK 语料：缺失时从代理下载 zip 并解压到项目内 NLTK_DATA。"""
    import zipfile
    nltk_root = os.environ["NLTK_DATA"]
    os.makedirs(nltk_root, exist_ok=True)
    for pkg in NLTK_PACKAGES:
        zip_path = os.path.join(nltk_root, pkg + ".zip")
        if os.path.exists(zip_path):
            print("[preload] NLTK %s 命中本地" % pkg)
            continue
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        t0 = time.perf_counter()
        print("[preload] 下载 NLTK %s ..." % pkg)
        with urllib.request.urlopen(NLTK_ZIP_URL % pkg, timeout=120) as r, \
                open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.dirname(zip_path))
        print("[preload] NLTK %s 就绪 %d 字节，%.1fs"
              % (pkg, os.path.getsize(zip_path), time.perf_counter() - t0))


def main():
    print("[preload] HF_ENDPOINT=%s" % os.environ["HF_ENDPOINT"])
    print("[preload] HF_HOME=%s" % os.environ["HF_HOME"])
    print("[preload] NLTK_DATA=%s" % os.environ["NLTK_DATA"])
    t_all = time.perf_counter()

    ensure_nltk_packages()

    # 1) 六个 tokenizer（导入期必需）
    for who, mid in [
        ("japanese", "tohoku-nlp/bert-base-japanese-v3"),
        ("english", "bert-base-uncased"),
        ("chinese_mix", "bert-base-multilingual-uncased"),
        ("korean", "kykim/bert-kor-base"),
        ("french", "dbmdz/bert-base-french-europeana-cased"),
        ("spanish", "dccuchile/bert-base-spanish-wwm-uncased"),
    ]:
        dl_tokenizer(mid, who)

    # 2) 完整 BERT 模型（ZH 推理特征提取）
    t0 = time.perf_counter()
    print("[preload] 下载完整模型 bert-base-multilingual-uncased ...")
    AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-uncased")
    print("[preload] 完整模型就绪 %.1fs" % (time.perf_counter() - t0))

    # 3) ZH VITS 合成权重
    for fname in ("config.json", "checkpoint.pth"):
        t0 = time.perf_counter()
        p = hf_hub_download(repo_id="myshell-ai/MeloTTS-Chinese", filename=fname)
        print("[preload] %-14s %8s <- myshell-ai/MeloTTS-Chinese  (%.1fs)"
              % (fname, _fmt(os.path.getsize(p)), time.perf_counter() - t0))

    print("[preload] 全部完成，总耗时 %.1fs" % (time.perf_counter() - t_all))


if __name__ == "__main__":
    main()
