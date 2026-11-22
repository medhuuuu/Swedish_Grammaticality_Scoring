"""Evaluate a predict.py on the toy data with confidence intervals, error-type breakdown and a run log.

Run from the repo root:
    python scripts/evaluate_dev.py --name baseline --split test
    python scripts/evaluate_dev.py --name kbbert_synth_v1 --split all --notes "KB-BERT, D1 synthetic"

--split test  the 120 test examples. Use this for anything trained on data/train.jsonl (like the baseline).
--split all   all 598 toy examples. Use this for models trained only on your own data.

Every run appends one row to results/runs.csv and writes its scores and report to results/runs/<name>/.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_data import categorise, error_parts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_dev(data_dir, split):
    test_text = {r["id"]: r["text"] for r in load_jsonl(data_dir / "test.jsonl")}
    test = [{**r, "text": test_text[r["id"]], "error": None, "group": None}
            for r in load_jsonl(data_dir / "test_labels.jsonl")]
    if split == "test":
        return test
    return load_jsonl(data_dir / "train.jsonl") + test


def metrics(gold, pred, good):
    return spearmanr(gold, pred).correlation, roc_auc_score(good, pred)


def bootstrap_ci(gold, pred, good, n=1000, seed=SEED):
    """95% confidence intervals for Spearman and AUC by resampling examples."""
    rng = np.random.default_rng(seed)
    gold, pred, good = map(np.asarray, (gold, pred, good))
    sp, auc = [], []
    for _ in range(n):
        i = rng.integers(0, len(gold), len(gold))
        if good[i].all() or not good[i].any():
            continue
        s, a = metrics(gold[i], pred[i], good[i])
        sp.append(s)
        auc.append(a)
    return np.percentile(sp, [2.5, 97.5]), np.percentile(auc, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="short run name, e.g. baseline or kbbert_synth_v1")
    ap.add_argument("--split", choices=["test", "all"], default="all")
    ap.add_argument("--repo", default=ROOT, type=Path, help="folder with the predict.py to evaluate")
    ap.add_argument("--data-dir", default=ROOT / "data", type=Path)
    ap.add_argument("--notes", default="", help="free text saved in runs.csv")
    ap.add_argument("--predict-args", default="", help="extra arguments for predict.py, e.g. '--model x'")
    args = ap.parse_args()

    run_dir = ROOT / "results" / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = load_dev(args.data_dir, args.split)
    inp, out = run_dir / "input.jsonl", run_dir / "scores.jsonl"
    with open(inp, "w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")

    # Run predict.py exactly like the official evaluate.py: timed, including model loading.
    cmd = [sys.executable, "predict.py", "--input", str(inp.resolve()), "--output", str(out.resolve())]
    cmd += args.predict_args.split()
    start = time.perf_counter()
    subprocess.run(cmd, cwd=args.repo, check=True)
    seconds = time.perf_counter() - start

    scores = {r["id"]: r["score"] for r in load_jsonl(out)}
    assert scores.keys() == {r["id"] for r in dev}, "predict.py must write exactly one score per input id"
    assert all(0 <= s <= 1 for s in scores.values()), "every score must be between 0 and 1"
    for r in dev:
        r["pred"] = scores[r["id"]]

    gold = [r["score"] for r in dev]
    pred = [r["pred"] for r in dev]
    good = [r["label"] == "good" for r in dev]
    spearman, auc = metrics(gold, pred, good)
    sp_ci, auc_ci = bootstrap_ci(gold, pred, good)

    # Pair accuracy: how often the good text scores higher than its own corrupted copy.
    groups = defaultdict(dict)
    for r in dev:
        if r["group"] is not None:
            groups[r["group"]][r["label"]] = r["pred"]
    pairs = [g for g in groups.values() if len(g) == 2]
    pair_acc = np.mean([g["good"] > g["poor"] for g in pairs]) if pairs else float("nan")

    # Error-type AUC: poor texts of one type against all good texts (threshold-free).
    good_preds = [r["pred"] for r in dev if r["label"] == "good"]
    by_type = defaultdict(list)
    for r in dev:
        if r["label"] == "poor" and r["error"]:
            for c in {categorise(p) for p in error_parts(r["error"])}:
                by_type[c].append(r["pred"])
    type_rows = []
    for c, preds in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        y = [1] * len(good_preds) + [0] * len(preds)
        type_rows.append((c, len(preds), roc_auc_score(y, good_preds + preds), np.mean(preds)))

    n = len(dev)
    pair_str = "–" if np.isnan(pair_acc) else f"{pair_acc:.3f}"
    lines = [
        f"# Run: {args.name}", "",
        f"Split: {args.split} ({n} examples). {args.notes}", "",
        "| Spearman (95% CI) | ROC-AUC (95% CI) | Pair acc | Time (s) | Samples/min |",
        "|---|---|---|---|---|",
        f"| {spearman:.3f} ({sp_ci[0]:.3f}–{sp_ci[1]:.3f}) | {auc:.3f} ({auc_ci[0]:.3f}–{auc_ci[1]:.3f}) "
        f"| {pair_str} | {seconds:.1f} | {n / seconds * 60:.0f} |", "",
        "- **Pair acc:** share of good/poor pairs where the good text gets the higher score "
        f"({len(pairs)} pairs; only available with --split all).",
        f"- Mean predicted score: good {np.mean(good_preds):.2f}, "
        f"poor {np.mean([r['pred'] for r in dev if r['label'] == 'poor']):.2f}.", "",
    ]
    if type_rows:
        lines += ["## By error type", "",
                  "AUC of poor texts with this error type against all good texts. "
                  "0.5 means the model cannot see this error at all.", "",
                  "| error type | examples | AUC | mean predicted score |", "|---|---|---|---|"]
        lines += [f"| {c} | {k} | {a:.3f} | {m:.2f} |" for c, k, a, m in type_rows]
    report = "\n".join(lines)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)

    # One row per run in results/runs.csv, so all experiments end up in one table.
    csv_path = ROOT / "results" / "runs.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "split", "n", "spearman", "spearman_lo", "spearman_hi",
                        "auc", "auc_lo", "auc_hi", "pair_acc", "seconds", "samples_per_min", "notes"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.name, args.split, n,
                    f"{spearman:.4f}", f"{sp_ci[0]:.4f}", f"{sp_ci[1]:.4f}",
                    f"{auc:.4f}", f"{auc_ci[0]:.4f}", f"{auc_ci[1]:.4f}",
                    f"{pair_acc:.4f}", f"{seconds:.1f}", f"{n / seconds * 60:.0f}", args.notes])
    print(f"\nSaved {run_dir / 'report.md'} and a row in {csv_path}")


if __name__ == "__main__":
    main()
