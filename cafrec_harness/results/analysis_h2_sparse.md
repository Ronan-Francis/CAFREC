# H2 — does the LLM profiler help SPARSE-history users? (Pure, 3-seed)

Date: 2026-08-25. Direct test of H2 (offline LLM temporal profiling most benefits
short-history users). Cleanest on/off contrast: **CAFREC_bge** (frozen bge-large
profile) vs **CAFREC_noprof** (z_long forced to 0, identical architecture). Cohort
= total interaction count per user in the atomic file, split **sparse (<20)** vs
**dense (>=20)** (matches the T1.1 threshold; MAX_ITEM_LIST_LENGTH is also 20).
Per-user NDCG@10/HIT@10 reconstructed from dumped ranks (== RecBole aggregate),
3-seed means; paired per-user Wilcoxon of (bge − noprof) within each cohort.
Script: `h2_sparse_strata.py`.

History length: n=22,912, min 5 / q25 12 / median 22 / q75 38 / max 237.
Cohort sizes: **Sparse 10,345**, **Dense 12,567**.

## 1. Per-cohort accuracy (3-seed means)

| Model | Overall NDCG | Sparse NDCG | Dense NDCG | Sparse HR | Dense HR |
|---|---|---|---|---|---|
| SASRec | 0.0396 | 0.0525 | 0.0290 | 0.1033 | 0.0585 |
| CAFREC_noprof | 0.0406 | **0.0543** | 0.0294 | **0.1058** | 0.0585 |
| CAFREC_bge | 0.0405 | 0.0526 | **0.0305** | 0.1022 | **0.0601** |

(Sparse users are *easier* in absolute terms — a short, recent, coherent history
is highly predictable — so all models score higher on them than on dense users.)

## 2. Profiler on/off (bge − noprof), paired per-user Wilcoxon by cohort

| Cohort | seed | NDCG Δ | p | HIT Δ | p |
|---|---|---|---|---|---|
| Sparse | 2020 | −0.00299 | 2.6e-3 | −0.00483 | 1.6e-2 |
| Sparse | 2021 | −0.00238 | 8.3e-3 | −0.00309 | 0.13 |
| Sparse | 403092 | −0.00002 | 0.81 | −0.00280 | 0.17 |
| Dense | 2020 | +0.00055 | 0.38 | −0.00088 | 0.54 |
| Dense | 2021 | +0.00151 | 2.8e-2 | +0.00223 | 0.13 |
| Dense | 403092 | +0.00148 | 1.9e-2 | +0.00342 | 1.5e-2 |

## 3. Reading (RQ2 / H2) — the hypothesis is REVERSED

- **H2 predicted the profiler helps sparse-history users most. The data shows the
  opposite.** On sparse users the frozen profile is a net *negative* (NDCG Δ
  −0.003/−0.002, significant at seeds 2020/2021; null at 403092); the
  profiler-*off* variant posts the best sparse accuracy of all three models
  (NDCG 0.0543 > SASRec 0.0525 > bge 0.0526). On dense users the profiler is the
  net *positive* it was meant to be everywhere (NDCG Δ +0.0015, significant at
  seeds 2021/403092).
- **Mechanism.** The profile is an LLM summary of the user's own history. For a
  thin history there is little to summarise, so z_long is noisy/uninformative and
  the context gate — which still admits some of it — displaces the short-term
  signal that alone already predicts these easy, recent-coherent sparse users.
  For a rich history the LLM has real material, so z_long carries genuine
  long-term taste that complements z_short. **The profiler needs history to be
  useful; it does not manufacture signal where history is thin.**
- **Why the aggregate looked like a tie.** The overall bge≈noprof≈SASRec tie
  (analysis_pure_3seed) is a *cancellation*: a sparse-cohort loss and a
  dense-cohort gain of similar size net to ~0. The cohort split is what makes the
  real behaviour visible.
- **Thesis framing.** Report as a primary, honest RQ2/H2 finding: on this dense,
  small-catalogue tier the LLM temporal profiler is a **history-dependent** signal
  — beneficial for well-observed (dense) users, mildly harmful for the
  sparse-history users H2 targeted. This motivates a clear future-work direction:
  gate the profile on history sufficiency (down-weight z_long when the history is
  too thin to profile), which the current context gate does not explicitly do.
