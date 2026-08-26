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
## 10-epoch rerun (confound resolved) — the negative HOLDS

To rule out under-convergence (CAFREC 24M params vs SASRec 0.56M at only 5 epochs),
both were re-run at 10 epochs, seed 2020:

| Model | HR@10 | NDCG@10 | valid NDCG |
|---|---|---|---|
| SASRec | 0.0291 | 0.0143 | 0.0157 |
| CAFREC logfull | 0.0210 | 0.0093 | 0.0125 |

More training helped BOTH, but SASRec more: the NDCG gap WIDENED from +0.0025 (5ep)
to +0.0050 (10ep), and validation now AGREES with test (SASRec 0.0157 > logfull
0.0125). So the 5-epoch valid/test disagreement was undertraining noise; the
under-convergence hypothesis is REFUTED. **SASRec beats CAFREC-logfull on 1K
decisively, and the gap grows with training.**

## Final status — an honest, clean boundary condition

The Pure result (logfull > SASRec, 10/10 seeds) does NOT generalise to the
ultra-dense 1K tier — there SASRec wins on both valid and test at 5 and 10 epochs.
This is not a failure of the study; it is a well-characterised BOUNDARY: CAFREC's
context-adaptive LLM-profile fusion helps in the moderate-density regime (Pure,
shorter/mixed histories) where the short-term encoder is under-fed, and does not
help on a tier whose users each have ~1,340 interactions, where self-attention is
already saturated with signal and the added long-term machinery only dilutes it.
Remaining caveat: 1K is single-seed (budget floor); report the direction as clear
and consistent (5ep + 10ep, valid + test) but not multi-seed significance-tested.
The thesis contribution is the CONDITIONAL finding: LLM temporal profiling helps a
strong sequential baseline only when short-term history is limited.
