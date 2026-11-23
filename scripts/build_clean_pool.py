"""Step 3a: build a pool of clean Swedish texts to corrupt later.

Run from the repo root:
    pip install datasets
    python scripts/build_clean_pool.py

Sources (all allow redistribution, so the final dataset can go on the HF Hub):
    talbanken  UD Swedish-Talbanken treebank, mixed genres          CC BY-SA 4.0
    tatoeba    short everyday sentences, questions, commands        CC BY 2.0 FR
    wikipedia  Swedish Wikipedia, factual sentences + paragraphs    CC BY-SA 4.0

Writes data/own/clean_pool.jsonl ({"text", "source", "n_sentences"}) and data/own/clean_pool_stats.json.
"""

import argparse
import bz2
import io
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 0

TALBANKEN_URL = ("https://raw.githubusercontent.com/UniversalDependencies/"
                 "UD_Swedish-Talbanken/master/sv_talbanken-ud-{split}.conllu")
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/swe/swe_sentences.tsv.bz2"
WIKI_DATASET, WIKI_CONFIG = "wikimedia/wikipedia", "20231101.sv"

# A sentence ends with . ! or ? and the next one starts with a capital letter.
# Requiring the capital avoids splitting inside abbreviations like "t.ex. att" or "bl.a. för".
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ0-9\"”])")
BAD_CHARS = re.compile(r"[()\[\]{}|=<>#*_~@/\\^]|https?:|www\.")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def normalise(text):
    """Key for de-duplication and leakage checks: lowercase letters and digits only."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^0-9a-zåäöéü]+", " ", text).strip()


def is_clean(text, min_words=3, max_words=40):
    """Keep ordinary prose sentences; drop headings, lists, markup, numbers-heavy and foreign text."""
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not text[0].isupper() or text[-1] not in ".!?":
        return False
    if BAD_CHARS.search(text) or "  " in text:
        return False
    chars = text.replace(" ", "")
    if sum(c.isalpha() for c in chars) / len(chars) < 0.75 or sum(c.isdigit() for c in text) > 4:
        return False
    if sum(w.isupper() and len(w) > 1 for w in words) > 2:  # acronym soup
        return False
    # Very rough Swedish check: at least one common Swedish function word for longer texts.
    common = {"och", "att", "är", "det", "som", "en", "på", "i", "av", "för", "med", "till", "den",
              "har", "de", "inte", "jag", "du", "vi", "han", "hon", "var", "ett", "om", "kan", "så",
              "vid", "under", "efter", "från", "sig", "sin", "sitt", "när", "men", "eller", "också",
              "finns", "blev", "hade", "ska", "man", "nu", "här", "där", "mycket", "alla", "nära"}
    return len(words) < 5 or any(w.lower().strip(".,!?") in common for w in words)


def talbanken():
    for split in ["train", "dev", "test"]:
        raw = fetch(TALBANKEN_URL.format(split=split)).decode("utf-8")
        for line in raw.splitlines():
            if line.startswith("# text = "):
                yield line[len("# text = "):].strip()


def tatoeba():
    raw = bz2.decompress(fetch(TATOEBA_URL)).decode("utf-8")
    for line in io.StringIO(raw):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3 and parts[1] == "swe":
            yield parts[2].strip()


def wikipedia(max_articles, paragraph_share, rng):
    """Yield single sentences and, for some paragraphs, 2–3 consecutive sentences as one text."""
    from datasets import load_dataset
    articles = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True)
    for i, article in enumerate(articles):
        if i >= max_articles:
            break
        for para in article["text"].split("\n"):
            sents = [s.strip() for s in SENT_SPLIT.split(para.strip()) if s.strip()]
            if len(sents) >= 2 and rng.random() < paragraph_share:
                k = rng.choice([2, 3])
                start = rng.randrange(0, max(1, len(sents) - k + 1))
                chunk = sents[start:start + k]
                if all(is_clean(s) for s in chunk):
                    yield " ".join(chunk)
                    continue
            yield from sents


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", default="talbanken,tatoeba,wikipedia")
    ap.add_argument("--max-per-source", type=int, default=40_000)
    ap.add_argument("--wiki-articles", type=int, default=15_000)
    ap.add_argument("--paragraph-share", type=float, default=0.25,
                    help="share of Wikipedia paragraphs kept as 2–3 sentence texts")
    ap.add_argument("--out", default=ROOT / "data" / "own" / "clean_pool.jsonl", type=Path)
    args = ap.parse_args()
    rng = random.Random(SEED)

    # Never let toy texts (or near-copies) into our training data.
    toy = set()
    for name in ["train.jsonl", "test.jsonl"]:
        for line in open(ROOT / "data" / name, encoding="utf-8"):
            toy.add(normalise(json.loads(line)["text"]))

    readers = {
        "talbanken": talbanken,
        "tatoeba": tatoeba,
        "wikipedia": lambda: wikipedia(args.wiki_articles, args.paragraph_share, rng),
    }
    seen, rows, stats = set(), [], {}
    for source in args.sources.split(","):
        counts = Counter()
        kept = []
        try:
            for text in readers[source]():
                counts["read"] += 1
                text = unicodedata.normalize("NFC", " ".join(text.split()))
                key = normalise(text)
                n_sent = len(SENT_SPLIT.split(text))
                ok = is_clean(text, max_words=60 if n_sent > 1 else 40)
                if not ok:
                    counts["filtered"] += 1
                elif key in toy:
                    counts["toy_overlap"] += 1
                elif key in seen:
                    counts["duplicate"] += 1
                else:
                    seen.add(key)
                    kept.append({"text": text, "source": source, "n_sentences": n_sent})
        except Exception as e:  # one failing download should not stop the others
            print(f"[{source}] failed: {type(e).__name__}: {e}")
            counts["error"] = 1
        rng.shuffle(kept)
        kept = kept[:args.max_per_source]
        counts["kept"] = len(kept)
        counts["multi_sentence"] = sum(r["n_sentences"] > 1 for r in kept)
        stats[source] = dict(counts)
        rows += kept
        print(f"[{source}] {dict(counts)}")

    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lengths = sorted(len(r["text"].split()) for r in rows)
    stats["total"] = {"texts": len(rows),
                      "median_words": lengths[len(lengths) // 2] if rows else 0,
                      "seed": SEED, "args": {k: str(v) for k, v in vars(args).items()}}
    (args.out.parent / "clean_pool_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} texts to {args.out}")
    print("Random examples:")
    for r in rng.sample(rows, min(10, len(rows))):
        print(f"  [{r['source']}] {r['text']}")


if __name__ == "__main__":
    main()
