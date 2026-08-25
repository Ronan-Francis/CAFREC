# RON-40 — R^4 classic-four vs R^6 full x_ctx (CAFREC bge, Pure)

Do the two AUXILIARY recency features (inter_session_gap_log_z, is_first_session) help beyond the four core RQ3 features (session_len, dwell_entropy, category_drift, policy_flag)? R^6 = full six-feature gate (already reported); R^4 = gate re-fit on the four core features only (gate MLP 128 fewer params). Same bge-large frozen profile, seq_len 20, mode:full. Paired per-user Wilcoxon + 95% bootstrap CI of the R^6 - R^4 mean difference.

| Seed | Metric | R^6 | R^4 | mean Δ(R6−R4) | 95% CI | Wilcoxon p | n |
|---|---|---|---|---|---|---|---|
| 2020 | NDCG@10 | 0.0400 | 0.0401 | -0.00005 | [-0.00111,+0.00107] | 0.889 | 22912 |
| 2020 | HIT@10 | 0.0786 | 0.0787 | -0.00004 | [-0.00240,+0.00245] | 0.971 | 22912 |
| 2021 | NDCG@10 | 0.0406 | 0.0403 | +0.00028 | [-0.00073,+0.00137] | 0.608 | 22912 |
| 2021 | HIT@10 | 0.0793 | 0.0785 | +0.00079 | [-0.00148,+0.00314] | 0.508 | 22912 |
| 403092 | NDCG@10 | 0.0408 | 0.0409 | -0.00008 | [-0.00113,+0.00097] | 0.949 | 22912 |
| 403092 | HIT@10 | 0.0795 | 0.0792 | +0.00026 | [-0.00192,+0.00266] | 0.824 | 22912 |

**Reading.** Across all three seeds the R^6−R^4 difference is tiny and its 95% CI straddles zero on both metrics (Wilcoxon p well above 0.05). The two auxiliary recency features add **no significant lift** over the four core features — the classic-four x_ctx carries essentially all of CAFREC's gate signal on Pure. This supports the M1 thesis framing (four CORE RQ3 features + two AUXILIARY features whose marginal value is tested here and found negligible).
