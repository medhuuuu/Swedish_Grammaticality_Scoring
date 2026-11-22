# Experiment log

## 2026-11-22
- Baseline reproduced (ModernBERT-large mean-pool + SVR)
- Hardware: Intel Core i7-1165G7 @ 2.80GHz, 8 GB RAM, CPU only
- Spearman 0.528 | AUC 0.783 | 29.8 s | 242 samples/min
- Note: load time dominates on 120 examples → smaller model = faster