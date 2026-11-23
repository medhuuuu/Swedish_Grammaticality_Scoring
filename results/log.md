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

## 2026-11-23: Step 2, evaluation setup

- scripts/evaluate_dev.py: 95% bootstrap CI, pair accuracy, error-type AUC, logs to results/runs.csv
- Baseline (--split test, 120 ex): Spearman 0.528 (0.375–0.661) | AUC 0.783 (0.698–0.862) | 40.0 s | 180/min
- Takeaways:
  - CI ±0.14 on 120 examples → use --split all (598) for own-data models
  - Predictions compressed (good 0.75 vs poor 0.56; gold 0.95 vs ~0.35)
  - Laptop timing varies 30–40 s between runs → report median of 3 runs
- Baseline error analysis (test split, results/runs/baseline/scores.jsonl):
  - 22/60 poor texts scored > 0.7 → misses single-word morphology errors
    (förstå, mitt nya cykel, grannar, inlämnat, ägar, mer korta)
  - Some correct texts scored < 0.5 (e.g. "Algoritmen sorterar listan…" 0.49)
    → hypothesis: embeddings capture topic more than grammar
  - Catches texts with many errors (0.0–0.2), e.g. "Efter skolan barnen brukar lekar i park…"
  → need fine-tuning + synthetic data rich in subtle 1-word errors

## 2026-11-23: Step 3a, clean text pool

- scripts/build_clean_pool.py → data/own/clean_pool.jsonl (84,723 texts, seed 0)
- Sources: Talbanken 4,723 | Tatoeba 40,000 | Wikipedia 40,000 (first 15k articles)
  (read: 6,038 | 56,870 | 1,012,606)
- Filters: 3–40 words, capital start + .!? end, no markup/URLs, few digits, Swedish function word
- Leakage check removed 13 texts that matched toy data (11 Tatoeba, 2 Wikipedia)
- Multi-sentence texts ≈ 2.3% (toy data ≈ 3%)
- Manual check (30 random): ~28 clean; 1 verbless Wikipedia list fragment, 1 old heading-style
  → add "must contain a verb" filter in 3b (spaCy POS)
- Limitations: Talbanken is 1970s–80s Swedish; some Tatoeba sentences written by learners;
  Tatoeba reuses a few names (Tom, Mary)
