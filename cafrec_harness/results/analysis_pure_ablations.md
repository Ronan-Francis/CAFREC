# Pure T3.1 ablations + profiler capacity ladder (3-seed, paired significance)

Date: 2026-08-25. Seeds 2020 / 2021 / 403092, n = 22,912 users, Modal A10G,
mode:full, seq_len 20. Each condition is CAFREC differing in ONE component;
compared per-user against the reported CAFREC bge-profiler (ablation none,
frozen bge-large d1024) via paired Wilcoxon + 95% bootstrap CI of the mean
per-user difference (bge − condition). Reconstructed per-user NDCG/HIT asserted
== RecBole aggregate. Script: `pure_ablations_sig.py`.

## 1. 3-seed means (NDCG@10 / HIT@10)

| Condition | NDCG@10 | HIT@10 | vs bge (NDCG) | Component isolated |
|---|---|---|---|---|
| CAFREC **bge** (ref) | 0.0405 | 0.0791 | — | full model |
| CAFREC **prof_7b** | 0.0400 | 0.0786 | −0.0005 | profiler LLM: 7B vs bge-large |
| CAFREC **static_gate** | 0.0386 | 0.0764 | −0.0019 | context-adaptive vs constant gate |
| CAFREC **concat** | 0.0351 | 0.0722 | −0.0054 | gated vs concat fusion |
| CAFREC **standin** | 0.0350 | 0.0714 | −0.0055 | frozen profile vs learnable z_long |

(standin / prof_7b means over the two new seeds 2021+403092; the 08-14 seed-2020
runs agree — standin≈0.0354, prof_7b≈0.0403 — but carry no rank dump.)

## 2. Paired per-user Wilcoxon (bge − condition), by seed

**static_gate — context-adaptive gate > constant gate (RQ3 mechanism).**
Positive on all three seeds; significant on two:
- seed 2020: NDCG +0.0007 (p=0.14, ns); HIT +0.0005 (p=0.65, ns)
- seed 2021: NDCG +0.0030 (p=3.7e-4); HIT +0.0049 (p=9.4e-4)
- seed 403092: NDCG +0.0021 (p=3.3e-4); HIT +0.0027 (p=0.025)

**concat — gated fusion ≫ concatenation.** Significant on every seed × metric,
mean ΔNDCG +0.004…+0.006 (p from 1.8e-8 down to 4.6e-20).

**standin — frozen profiler ≫ learnable stand-in (RQ2).** Both seeds, both
metrics: mean ΔNDCG ≈ +0.006, ΔHIT ≈ +0.009 (p 2.9e-9 … 5.6e-11).

**prof_7b — no gain over bge-large.** NS on 3/4 seed×metric cells; the one
significant cell (seed 403092 NDCG, p=0.014) favours **bge**. Scaling the offline
profiler LLM from bge-large to 7B does not improve rank quality.

## 3. Reading (RQ2 / RQ3)

- **RQ3 (architecture works as designed).** Both structural ablations degrade the
  model and the degradation is significant: removing context-adaptivity
  (static_gate) costs ~0.002 NDCG (sig 2/3 seeds), and replacing gated fusion with
  concatenation (concat) costs ~0.005 NDCG (sig all seeds). So CAFREC's
  context-adaptive gating AND its element-wise gated fusion each carry real,
  measurable weight — the design is doing work, not decoration.
- **Reconciles with RON-44.** The static_gate result shows the adaptive gate helps
  in AGGREGATE; RON-44 shows that help is NOT concentrated on the exploratory /
  high-dwell minority H3 named. Honest combined story: the gate is a net positive
  vs a constant gate, but its benefit is spread across the focused majority rather
  than targeted at intent-shifters.
- **RQ2 (profiler value + capacity).** The frozen profile is essential (standin
  collapses to ~0.035, the same floor as concat), confirming the load-bearing
  profiler≫stand-in claim across seeds. But profiler CAPACITY saturates at
  bge-large: the 7B profile is statistically tied with bge. bge-large is the
  reported operating point.
- **Caveat.** All gaps here are vs CAFREC-bge, an internal reference; recall bge
  itself is statistically TIED with SASRec on aggregate accuracy (analysis_pure_3seed).
  These ablations establish that CAFREC's components matter *relative to each
  other*, not that CAFREC beats the short-term baseline on this tier.
