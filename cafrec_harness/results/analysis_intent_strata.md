# RON-44 — Session-intent proxy stratification (Pure, 3-seed means)

Session-intent is proxied from the **test-row prefix context features** (causal: they summarise each user's history *before* the held-out target). Per-user HR@10 / NDCG@10 are reconstructed from the dumped held-out ranks (reconstruction asserted == RecBole's reported aggregate) and averaged over seeds 2020 / 2021 / 403092. Both behavioural features are ~85-90% concentrated at their floor on Pure (short, single-category histories), so each proxy is a BINARY split at the floor rather than tertiles.

## Category-drift proxy (Focused = never switches category; Exploratory = switches)

Category-drift rate is the RQ3 policy/intent signal: Exploratory users cross content categories within their session history, the case context-adaptive gating is meant to help. Group sizes: Focused=19445, Exploratory=3467.

| Model | Overall NDCG | Focused NDCG | Exploratory NDCG | Focused HR | Exploratory HR |
|---|---|---|---|---|---|
| SASRec | 0.0396 | 0.0388 | 0.0444 | 0.0773 | 0.0865 |
| CAFREC_noprof | 0.0406 | 0.0403 | 0.0426 | 0.0796 | 0.0813 |
| CAFREC_bge | 0.0405 | 0.0400 | 0.0429 | 0.0786 | 0.0824 |

**NDCG@10 Δ vs SASRec** — CAFREC noprof: Focused +0.0015, Exploratory -0.0018; CAFREC bge: Focused +0.0013, Exploratory -0.0015

## Dwell-entropy proxy (Low vs High dwell dispersion)

Dwell-time entropy proxies engagement dispersion across a session. Group sizes: LowDisp=20482, HighDisp=2430.

| Model | Overall NDCG | LowDisp NDCG | HighDisp NDCG | LowDisp HR | HighDisp HR |
|---|---|---|---|---|---|
| SASRec | 0.0396 | 0.0387 | 0.0470 | 0.0773 | 0.0909 |
| CAFREC_noprof | 0.0406 | 0.0400 | 0.0462 | 0.0790 | 0.0870 |
| CAFREC_bge | 0.0405 | 0.0398 | 0.0464 | 0.0781 | 0.0878 |

**NDCG@10 Δ vs SASRec** — CAFREC noprof: LowDisp +0.0012, HighDisp -0.0008; CAFREC bge: LowDisp +0.0011, HighDisp -0.0007

## Reading (RQ3 / H3)

- The *Exploratory* (category-switching) and *HighDisp* users are the harder-history cohort the context gate was hypothesised to help most (H3). They post HIGHER absolute accuracy for every model (more history to exploit), and on them **SASRec wins**: CAFREC's NDCG Δ vs SASRec is NEGATIVE for both variants (drift −0.0015/−0.0018; dwell −0.0007/−0.0008).
- CAFREC's small aggregate edge is instead carried by the *Focused* / *LowDisp* majority (Δ ≈ +0.0011…+0.0015). So on Pure the context-adaptive fusion helps the focused mass, not the intent-shifting minority — the reverse of the H3 expectation.
- Honest RQ3 read: context gating is not a differentiated win for high-intent-shift users on this tier; the strong short-term attention encoder already captures the longer exploratory histories. Report as a nuanced/limitation finding, not support for H3. Magnitudes are small (<0.002 NDCG) and per-seed noise is comparable — pair with the paired per-user tests before any directional claim.
