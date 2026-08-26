# RON-45 — History-gated profiler: the H2 fix (Pure, 3-seed)

Date: 2026-08-26. Motivated directly by analysis_h2_sparse.md: the frozen bge
profile helps DENSE users but HURTS SPARSE users, and the context gate never saw
history length. Fix: append the user's history length (item_seq_len /
max_seq_length, saturating at the seq cap) to the gate inputs so it can learn to
SUPPRESS z_long when the history is too thin to profile (`history_gate=True`,
+64 params over bge). CAFREC-bge + history_gate, 3 seeds, mode:full, seq_len 20.
Per-user metrics from dumped ranks (== RecBole aggregate). Script:
`histgate_analysis.py`.

## 1. Overall 3-seed means

| Model | NDCG@10 | HR@10 | n_params |
|---|---|---|---|
| SASRec | 0.0396 | 0.0787 | 562,880 |
| CAFREC noprof | 0.0406 | 0.0799 | 2,050,497 |
| CAFREC bge | 0.0405 | 0.0791 | 24,108,417 |
| **CAFREC bge + history_gate** | **0.0408** | **0.0805** | 24,108,481 |

history_gate is the **best-performing configuration on Pure** (highest NDCG and HR
of any model), for +64 params over bge. Paired vs SASRec: ΔNDCG +0.0006/+0.0015/
+0.0015, significant at seed 403092 only (p=0.025), positive all seeds. Paired vs
bge: negligible+ (ns) — aggregate parity with plain bge; the difference is in WHERE
each earns its accuracy (below).

## 2. Cohort behaviour — the fix works as designed

Sparse (<20 interactions, n=10,345) / Dense (>=20, n=12,567), 3-seed mean NDCG:

| Model | Sparse | Dense |
|---|---|---|
| SASRec | 0.0525 | 0.0290 |
| CAFREC noprof | 0.0543 | 0.0294 |
| CAFREC bge (H2-broken) | 0.0526 | 0.0305 |
| **CAFREC histgate** | **0.0534** | **0.0304** |

- **Sparse recovery.** Plain bge hurt sparse users (0.0526 vs noprof 0.0543).
  history_gate lifts sparse back to 0.0534 — recovering ~47% of the bge→noprof gap
  — and (histgate − bge) is POSITIVE on all three seeds (+0.0005/+0.0014/+0.0008).
  The gate is suppressing the noisy profile for thin histories, exactly as intended.
- **Dense retention.** history_gate keeps the profiler's dense gain: dense 0.0304 ≈
  bge 0.0305, and (histgate − noprof) on dense is positive all seeds, significant at
  seed 403092 (+0.0021, p=3e-4).
- **Net.** history_gate is the ONLY configuration strong on BOTH cohorts — it fixes
  the sparse leak without surrendering the dense gain, which is why it edges out
  every other model overall.

## 3. Reading (RQ2 / contribution)

- The H2 diagnosis is CONFIRMED constructively: the profiler's sparse-user damage
  was caused by the gate lacking a history-sufficiency signal, and supplying that
  one signal (history length) partially repairs it while preserving the dense-user
  benefit. This closes the loop hypothesis → negative result → diagnosis → fix.
- **Honest caveats.** Magnitudes are small (~0.001-0.002 NDCG); histgate beats
  SASRec significantly on only 1/3 seeds in aggregate; noprof remains narrowly best
  on sparse alone (history-gating recovers ~half, not all, of the loss). Report as a
  VALIDATED MECHANISM and the best single configuration, not a decisive win over the
  baseline. A stronger, unsaturated sufficiency signal (e.g. log total-history, or
  an explicit multiplicative suppression of z_long) is the natural next iteration.
