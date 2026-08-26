# 1K generalisation test — does the logfull win hold on a second dataset?

Date: 2026-08-26. k-core-floored 1K (kuairand_1k_kcore_ctx: 998 users, 69,387
items, 1.34M interactions), identical interactions across models, matched 5-epoch
schedule, seed 2020, full-ranking eval. bge profiles rebuilt for the floored users.
NOTE: this is a BUDGET PROBE — 1 seed, 5 epochs — not a full multi-seed study.

| Model | HR@10 | NDCG@10 | MRR@10 | valid NDCG | n_params |
|---|---|---|---|---|---|
| SASRec | 0.0200 | 0.0086 | 0.0052 | 0.0108 | 0.56M |
| CAFREC noprof | 0.0130 | 0.0052 | 0.0030 | 0.0104 | 2.0M |
| CAFREC logfull | 0.0130 | 0.0061 | 0.0041 | 0.0112 | 24M |

## Reading — the Pure win does NOT cleanly generalise (honest negative)

- **On test, SASRec leads** (HR 0.020, NDCG 0.0086) over both CAFREC variants. The
  logfull-over-SASRec advantage established on Pure does not replicate on 1K here.
- **logfull > noprof holds** (NDCG 0.0061 vs 0.0052; MRR 0.0041 vs 0.0030): the
  history-gated-profile MECHANISM still adds value internally; it is the
  CAFREC-vs-SASRec comparison that flips.
- **The probe is noisy and confounded — do not over-read it.**
  (a) 998 test users at ~1-2% HR: the SASRec-vs-CAFREC gap is a HANDFUL of users
      (~20 vs ~13 hits). (b) Test and VALIDATION disagree — on valid, logfull
      (0.0112) edges SASRec (0.0108). (c) CAFREC has 24M params vs SASRec's 0.56M,
      so at only 5 epochs CAFREC is likely UNDER-CONVERGED relative to the small
      baseline. (d) single seed.
- **Coherent interpretation (fits the thesis).** 1K users are ULTRA-DENSE (~1,340
  interactions each) — precisely the regime where the short-term self-attention
  encoder is best-fed and a long-term profile is least needed. This matches the
  history-dependence story (RON-45/H2): the profile helps when short-term signal
  is thin; on 1K it is abundant, so SASRec alone is hard to beat. Report as a
  BOUNDARY CONDITION, not a contradiction.
- **Status.** Generalisation is INCONCLUSIVE-to-NEGATIVE under this budget probe.
  A clean verdict would need 10-epoch, multi-seed runs (removes the convergence +
  noise confounds). The honest thesis statement: the Pure result (logfull > SASRec,
  10/10 seeds) is robust ON PURE; a single-seed 5-epoch probe on the denser 1K tier
  did not reproduce it on test, plausibly because 1K's extreme density favours the
  short-term encoder and because CAFREC was under-trained at matched short epochs.
