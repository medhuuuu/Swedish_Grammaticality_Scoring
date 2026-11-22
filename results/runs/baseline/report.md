# Run: baseline

Split: test (120 examples). ModernBERT-large mean-pool + SVR, laptop CPU

| Spearman (95% CI) | ROC-AUC (95% CI) | Pair acc | Time (s) | Samples/min |
|---|---|---|---|---|
| 0.528 (0.375–0.661) | 0.783 (0.698–0.862) | – | 40.0 | 180 |

- **Pair acc:** share of good/poor pairs where the good text gets the higher score (0 pairs; only available with --split all).
- Mean predicted score: good 0.75, poor 0.56.
