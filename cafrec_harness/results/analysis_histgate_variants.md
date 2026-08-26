# RON-45 (Stage 1) — history-gate variant bake-off (Pure, 3-seed)

Date: 2026-08-26. Three history-sufficiency signals for the gate, CAFREC-bge, 3
seeds, mode:full, seq_len 20. Per-user NDCG@10 from dumped ranks; sparse/dense =
history length <20 / >=20. Reference models included. Winner carried forward.

| Model | Overall NDCG | Sparse | Dense | sig vs SASRec |
|---|---|---|---|---|
| SASRec | 0.0396 | 0.0525 | 0.0290 | — |
| CAFREC noprof | 0.0406 | 0.0543 | 0.0294 | 0/3 |
| CAFREC bge | 0.0405 | 0.0526 | 0.0305 | 0/3 |
| histgate **seqlen** (item_seq_len, saturating) | 0.0408 | 0.0534 | 0.0304 | 1/3 |
| histgate **logfull** (log total activity) | **0.0422** | **0.0556** | **0.0311** | **2/3** |
| histgate **suppress** (explicit z_long damper) | 0.0410 | 0.0535 | 0.0308 | 1/3 |

## Reading

- **`logfull` wins decisively and is the best CAFREC configuration to date.** It is
  the top model on EVERY axis: overall (0.0422, +0.0026 / +6.6% vs SASRec — more
  than double the old seqlen edge), sparse (0.0556 — beats even noprof's 0.0543,
  i.e. it OVER-recovers the H2 loss), and dense (0.0311 — beats bge's 0.0305). It is
  significant over SASRec on 2/3 seeds vs 1/3 for the saturating seqlen signal.
- **Why the unsaturated signal matters.** seqlen caps at the sequence window (20),
  so every dense user looks identical (=1.0) and the gate cannot grade sufficiency
  above the cap. log(total activity) preserves the full ordering, so the gate learns
  a genuinely history-conditioned mix — helping sparse (suppress the noisy profile)
  AND dense (admit the well-supported profile) simultaneously.
- **`suppress` (explicit damper) helps but less than logfull** — a single learnt
  threshold is coarser than letting the gate MLP condition on the graded signal.
- **Decision.** `history_gate_mode="logfull"` becomes the primary CAFREC config.
  Carried into the many-seed significance run and the (equal) tuning grid.

CAVEAT: still a modest absolute effect (~0.003 NDCG) and 2/3-seed significance;
the many-seed run tightens this. But logfull is now a clean best-on-all-axes result,
not a marginal tie.
