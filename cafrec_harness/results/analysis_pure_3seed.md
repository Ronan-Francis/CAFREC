# Pure (mode:full, seq_len 20) — 3-seed significance + diversity

Seeds: 2020 / 2021 / 403092. n = 22,912 users. Same Modal A10G environment.
Source: top-k batch rank dumps (significance) and top-k item dumps (diversity).

## 1. Accuracy — 3-seed means

| Condition | HR@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| SASRec | 0.0787 | 0.0396 | 0.0279 |
| CAFREC no_profiler | 0.0799 | 0.0406 | 0.0289 |
| CAFREC bge-profiler | 0.0791 | 0.0405 | 0.0289 |

## 2. Paired significance (Wilcoxon signed-rank p; bootstrap 95% CI of mean Δ)

- **bge-profiler vs SASRec:** NOT significant on any metric or seed (all p ≥ 0.17).
- **no_profiler vs SASRec:** NDCG trends positive at seeds 2020/2021 (Δ≈+0.0015, p≈0.06)
  but is null at seed 403092 (p=0.90). Not significant at α=0.05.
- **bge-profiler vs no_profiler:** no consistent difference; the one flagged result
  (seed 2020 HR, p=0.027 in favour of no_profiler) reverses sign at seed 403092 → noise.

Conclusion: on Pure, CAFREC (either variant) is **statistically tied with SASRec** on rank
quality, and the frozen LLM profile adds **no significant lift over the z_long=0 ablation**.
The only robust profile effect remains profiler ≫ learnable stand-in (established 2026-08-14,
Wilcoxon p<1e-6): the frozen offline profile beats a learned embedding of equal capacity,
but does not beat a strong short-term encoder on this tier.

## 3. Diversity (ILD multihot tags; Coverage vs 7,210-item catalogue) — 3-seed means

| Condition | ILD@10 | Coverage@10 |
|---|---|---|
| SASRec | 0.7930 | 0.1889 |
| CAFREC no_profiler | 0.7862 | 0.2336 |
| CAFREC bge-profiler | 0.7920 | 0.2294 |

- **ILD** essentially equal (~0.79) across models: intra-list category spread is unchanged.
- **Coverage**: both CAFREC variants recommend a **notably broader slice of the catalogue**
  (~23% vs SASRec's ~19%; ~+4-5 pp, ~+22% relative). CAFREC spreads recommendations across
  more distinct items than SASRec despite tied accuracy — a genuine diversity advantage and a
  first-class output for the stratified/policy sections (RON-45/53).

Category matrix: 7,583 rows × 46 tags, multihot (R2d strategy = multihot).
