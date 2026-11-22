# Experiment log

## 2026-11-22
- Baseline reproduced (ModernBERT-large mean-pool + SVR)
- Hardware: Intel Core i7-1165G7 @ 2.80GHz, 8 GB RAM, CPU only
- Spearman 0.528 | AUC 0.783 | 29.8 s | 242 samples/min

- Kaggle Tesla T4: Spearman 0.529 | AUC 0.784 | 12.3 s | 584 samples/min
  (≤0.001 diff vs CPU: float non-determinism; load time dominates)