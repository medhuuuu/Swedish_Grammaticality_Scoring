# Experiment log

## 2026-11-22: Step 0, baseline

- Baseline reproduced (ModernBERT-large mean-pool + SVR, starter code unchanged)
- Laptop: Intel Core i7-1165G7 @ 2.80GHz, 8 GB RAM, CPU only
  Spearman 0.528 | AUC 0.783 | 29.8 s | 242 samples/min
- Kaggle Tesla T4: Spearman 0.529 | AUC 0.784 | 12.3 s | 584 samples/min
  (≤0.001 diff vs CPU: float non-determinism)
- Takeaway: on 120 short texts, model loading dominates run time
  → smaller model = faster speed score

## 2026-11-22: Step 1, data analysis

Full report: results/data_analysis.md (script: scripts/analyze_data.py)

- Data: 598 examples (478 train + 120 test), 50/50 good/poor
- Pairs: 159 good + corrupted-copy pairs; 160 single texts (80 good, 80 poor)
- Score scale: good ≈ 0.95; poor with 1 error ≈ 0.39, 2 ≈ 0.23, 3 ≈ 0.09
  → synthetic data scores: good 0.95, 1/2/3 errors → 0.4/0.2/0.1
- Top error types: verb form/tense (21%), agreement (17%), V2 word order (15%),
  definiteness (13%), gender (13%); then pronoun, preposition, negation placement,
  compound split (6%)
- Lexical/calque errors judged most severe (≈0.21 with 1 error) but rare (3%)
- Errors are subtle: median 1 word changed between good and poor copy
  → mean pooling dilutes the signal; fine-tuning should help
- No length bias: good and poor have same length; ~50% poor in every length bin
- Texts short: median 8 words / 14 tokens, max 50 tokens; 0/598 > 128 tokens
  → no truncation on toy data; use max_length 256–512 in final model
    in case hidden test has longer texts