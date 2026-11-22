"""Analyse the starter toy data: labels, scores, error types, error counts, text length and pair edits.

Run from the repo root:
    python scripts/analyze_data.py

Writes a Markdown report to results/data_analysis.md and prints it.
Uses only the standard library; token counts are added if the ModernBERT tokenizer is available.
"""

import argparse
import difflib
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Order matters: the first matching category wins for each error part.
CATEGORIES = [
    ("compound split", ["compound"]),
    ("lexical / calque / unclear", ["calque", "anglicism", "means", "unclear", "word salad", "eftersom"]),
    ("adverb / negation placement", ["negation", "adverb placement", "adverb order"]),
    ("word order (V2, questions)", ["v2", "verb-second", "word order"]),
    ("definiteness", ["definite", "missing 'den'"]),
    ("gender", ["gender"]),
    ("agreement", ["agreement", "neuter"]),
    ("preposition", ["preposition", "superfluous 'för'", "superfluous 'av'"]),
    ("pronoun (sitt/deras, de/dem)", ["pronoun", "reflexive", "de/dem", "subject form", "object form", "possessive", "varandra"]),
    ("plural form", ["plural"]),
    ("verb form / tense", ["verb", "tense", "infinitive", "auxiliary", "passive", "brukade", "imperative"]),
    ("missing / extra word", ["missing", "superfluous", "conjunction"]),
    ("adjective / adverb form", ["comparative", "superlative", "adverb form", "adverbial form"]),
]


def categorise(part):
    p = part.lower()
    for name, keys in CATEGORIES:
        if any(k in p for k in keys):
            return name
    return "other"


def error_parts(error):
    return [p.strip() for p in (error or "").split(";") if p.strip()]


def n_words(text):
    return len(text.split())


def n_sentences(text):
    return max(1, len(re.findall(r"[.!?]+(\s|$)", text)))


def word_diff(good, poor):
    """Return (number of changed words, short 'before → after' description)."""
    a, b = good.split(), poor.split()
    changes, changed = [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            changed += max(i2 - i1, j2 - j1)
            changes.append(f"'{' '.join(a[i1:i2])}' → '{' '.join(b[j1:j2])}'")
    return changed, "; ".join(changes)


def summary(values):
    if not values:
        return "–"
    return f"{st.mean(values):.2f} (min {min(values)}, median {st.median(values)}, max {max(values)})"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=ROOT / "data", type=Path)
    ap.add_argument("--out", default=ROOT / "results" / "data_analysis.md", type=Path)
    ap.add_argument("--tokenizer", default="AI-Sweden-Models/ModernBERT-large",
                    help="HF tokenizer for token counts; use 'none' to skip")
    args = ap.parse_args()

    train = load_jsonl(args.data_dir / "train.jsonl")
    test_text = {r["id"]: r["text"] for r in load_jsonl(args.data_dir / "test.jsonl")}
    test = [{**r, "text": test_text[r["id"]]} for r in load_jsonl(args.data_dir / "test_labels.jsonl")]
    everything = train + test
    out = ["# Data analysis: starter toy data", ""]

    # 1. Labels and scores
    out += ["## 1. Labels and scores", ""]
    rows = []
    for name, data in [("train", train), ("test", test), ("all", everything)]:
        c = Counter(r["label"] for r in data)
        rows.append([name, len(data), c["good"], c["poor"]])
    out += [table(["split", "examples", "good", "poor"], rows), ""]
    for label in ["good", "poor"]:
        dist = Counter(r["score"] for r in everything if r["label"] == label)
        out.append(f"- **{label}** scores: " + ", ".join(f"{s} ×{n}" for s, n in sorted(dist.items())))
    out.append("")

    # 2. Error types (train only: test has no error field)
    poor = [r for r in train if r["label"] == "poor"]
    cat_count, cat_scores, cat_examples, other = Counter(), defaultdict(list), {}, []
    for r in poor:
        cats = {categorise(p) for p in error_parts(r["error"])}
        for c in cats:
            cat_count[c] += 1
            if len(error_parts(r["error"])) == 1:
                cat_scores[c].append(r["score"])
        for p in error_parts(r["error"]):
            if categorise(p) == "other":
                other.append(p)
    out += ["## 2. Error types (train poor examples)", "",
            "An example with several error types counts once per type. "
            "'Mean score' uses single-error examples only, so it shows how severe each type is judged.", ""]
    rows = [[c, n, f"{n / len(poor):.0%}",
             f"{st.mean(cat_scores[c]):.2f}" if cat_scores[c] else "–"]
            for c, n in cat_count.most_common()]
    out += [table(["error type", "examples", "share of poor", "mean score (1 error)"], rows), ""]
    if other:
        out += ["Uncategorised error descriptions (check by hand):", ""]
        out += [f"- {p}" for p in sorted(set(other))] + [""]

    # 3. Number of errors -> score
    by_n = defaultdict(list)
    for r in poor:
        by_n[len(error_parts(r["error"]))].append(r["score"])
    out += ["## 3. Number of errors → score", "",
            "Counted by splitting the `error` field on ';' (approximate).", ""]
    rows = [[n, len(s), f"{st.mean(s):.2f}", min(s), max(s)] for n, s in sorted(by_n.items())]
    out += [table(["errors", "examples", "mean score", "min", "max"], rows), ""]

    # 4. Text length
    out += ["## 4. Text length", ""]
    rows = []
    for label in ["good", "poor"]:
        data = [r for r in everything if r["label"] == label]
        rows.append([label, summary([n_words(r["text"]) for r in data]),
                     summary([n_sentences(r["text"]) for r in data])])
    out += [table(["label", "words", "sentences"], rows), ""]
    bins = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 10_000)]
    rows = []
    for lo, hi in bins:
        data = [r for r in everything if lo <= n_words(r["text"]) < hi]
        if data:
            share = sum(r["label"] == "poor" for r in data) / len(data)
            rows.append([f"{lo}–{hi - 1 if hi < 10_000 else '+'}", len(data), f"{share:.0%}"])
    out += [table(["words", "examples", "share poor"], rows), "",
            "If one length range is mostly poor or mostly good, a model can cheat on length; check this.", ""]

    if args.tokenizer.lower() != "none":
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            lens = [len(tok(r["text"])["input_ids"]) for r in everything]
            over = sum(n > 128 for n in lens)
            note = ("so their ends are cut off by the baseline" if over
                    else "so the baseline's max_length=128 cuts nothing off")
            out += [f"Tokens ({args.tokenizer}): {summary(lens)}. "
                    f"**{over} of {len(lens)} texts are longer than 128 tokens**, {note}.", ""]
        except Exception as e:  # tokenizer not installed or not downloadable
            out += [f"(Token counts skipped: {type(e).__name__})", ""]

    # 5. Good/poor pairs
    groups = defaultdict(list)
    for r in train:
        groups[r["group"]].append(r)
    pairs = [g for g in groups.values() if len(g) == 2 and {r["label"] for r in g} == {"good", "poor"}]
    singles = Counter(g[0]["label"] for g in groups.values() if len(g) == 1)
    out += ["## 5. Good/poor pairs (train)", "",
            f"- {len(pairs)} groups are a good text plus its corrupted copy.",
            f"- {sum(singles.values())} groups have one text only "
            f"({singles['good']} good, {singles['poor']} poor).", ""]
    changed, examples = [], {}
    for g in pairs:
        good = next(r for r in g if r["label"] == "good")
        bad = next(r for r in g if r["label"] == "poor")
        n, desc = word_diff(good["text"], bad["text"])
        changed.append(n)
        for c in {categorise(p) for p in error_parts(bad["error"])}:
            if len(error_parts(bad["error"])) == 1 and c not in examples and len(desc) < 90:
                examples[c] = desc
    out += [f"Words changed between a good text and its corrupted copy: {summary(changed)}.", "",
            "Small numbers mean the errors are subtle: one or two words in an otherwise fine text.", ""]
    if examples:
        out += ["One example edit per error type:", ""]
        out += [table(["error type", "good → poor"], sorted(examples.items())), ""]

    report = "\n".join(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
