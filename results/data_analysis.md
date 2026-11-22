# Data analysis: starter toy data

## 1. Labels and scores

| split | examples | good | poor |
|---|---|---|---|
| train | 478 | 239 | 239 |
| test | 120 | 60 | 60 |
| all | 598 | 299 | 299 |

- **good** scores: 0.8 ×1, 0.85 ×2, 0.9 ×10, 0.95 ×254, 1.0 ×32
- **poor** scores: 0.0 ×1, 0.05 ×6, 0.1 ×5, 0.15 ×8, 0.2 ×27, 0.25 ×18, 0.3 ×11, 0.35 ×67, 0.4 ×93, 0.45 ×58, 0.5 ×3, 0.55 ×2

## 2. Error types (train poor examples)

An example with several error types counts once per type. 'Mean score' uses single-error examples only, so it shows how severe each type is judged.

| error type | examples | share of poor | mean score (1 error) |
|---|---|---|---|
| verb form / tense | 49 | 21% | 0.36 |
| agreement | 41 | 17% | 0.42 |
| word order (V2, questions) | 35 | 15% | 0.35 |
| definiteness | 32 | 13% | 0.39 |
| gender | 30 | 13% | 0.40 |
| pronoun (sitt/deras, de/dem) | 26 | 11% | 0.44 |
| preposition | 21 | 9% | 0.44 |
| adverb / negation placement | 19 | 8% | 0.37 |
| compound split | 15 | 6% | 0.45 |
| plural form | 11 | 5% | 0.39 |
| missing / extra word | 8 | 3% | 0.34 |
| lexical / calque / unclear | 8 | 3% | 0.21 |
| adjective / adverb form | 2 | 1% | 0.45 |

## 3. Number of errors → score

Counted by splitting the `error` field on ';' (approximate).

| errors | examples | mean score | min | max |
|---|---|---|---|---|
| 1 | 184 | 0.39 | 0.0 | 0.55 |
| 2 | 47 | 0.23 | 0.1 | 0.4 |
| 3 | 8 | 0.09 | 0.05 | 0.15 |

## 4. Text length

| label | words | sentences |
|---|---|---|
| good | 10.19 (min 3, median 8, max 30) | 1.03 (min 1, median 1, max 3) |
| poor | 9.97 (min 2, median 8, max 31) | 1.03 (min 1, median 1, max 3) |

| words | examples | share poor |
|---|---|---|
| 0–9 | 352 | 52% |
| 10–19 | 191 | 46% |
| 20–39 | 55 | 55% |

If one length range is mostly poor or mostly good, a model can cheat on length; check this.

Tokens (AI-Sweden-Models/ModernBERT-large): 16.84 (min 5, median 14.0, max 50). **0 of 598 texts are longer than 128 tokens**, so the baseline's max_length=128 cuts nothing off.

## 5. Good/poor pairs (train)

- 159 groups are a good text plus its corrupted copy.
- 160 groups have one text only (80 good, 80 poor).

Words changed between a good text and its corrupted copy: 1.59 (min 1, median 1, max 5).

Small numbers mean the errors are subtle: one or two words in an otherwise fine text.

One example edit per error type:

| error type | good → poor |
|---|---|
| adverb / negation placement | '' → 'inte'; 'inte' → '' |
| agreement | 'god' → 'gott' |
| compound split | 'vinterjacka.' → 'vinter jacka.' |
| definiteness | 'sjön' → 'sjö' |
| gender | 'Ett' → 'En' |
| missing / extra word | 'Det snöar' → 'Snöar' |
| plural form | 'katter' → 'katts' |
| preposition | '' → 'av' |
| pronoun (sitt/deras, de/dem) | 'sitt' → 'deras' |
| verb form / tense | 'reste' → 'reser' |
| word order (V2, questions) | 'kom' → ''; '' → 'kom' |
