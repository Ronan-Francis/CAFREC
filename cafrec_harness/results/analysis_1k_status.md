# kuairand_1k tier — collection status + degenerate-result finding

Date: 2026-08-20. Tier role: SECONDARY / scale-generalisation upside (D6 made
kuairand_pure the PRIMARY reported tier).

## What was collected

The 08-15 background 1K batch (tags `base1k_s403092` / `bge1k_s403092`) was
launched over-time on Modal. Checking the `cafrec-results` volume on 2026-08-20,
**only one 1K job produced a JSON**:

| file | model | seed | test HR@10 | valid HR@10 |
|---|---|---|---|---|
| `HGRU4Rec_kuairand_1k_topk1k_s403092_...162902.json` | HGRU4Rec | 403092 | **0.0000** | 0.0010 |

The SASRec / HGN / CAFREC 1K legs from that batch left no JSON on the volume and
no app remains in Modal's history — they did not complete.

## The one landed result is degenerate (not a metric bug)

- Split: 1,001 users, **1,819,490 items**, 4,426,840 train interactions.
- test HR@10 = NDCG@10 = MRR@10 = **0.0** across all 1,000 test users; the
  independent `ranks_check` recomputed from dumped ranks agrees (0.0), so this is
  a real model outcome, not an extraction error. valid HR@10 is 0.001 (≈1/1000).

**Root cause (verified 2026-08-20 by inspecting `kuairand_1k.inter` directly).**
The dominant cause is a **data-preprocessing gap, not model difficulty and not a
code bug** (the same harness gives sensible numbers on Pure). The 1K `.inter` was
built with **no item-frequency floor (no k-core filtering)**:

  - 1,819,489 unique items across 1,000 users / 4,429,840 rows -> **2.43 mean
    appearances per item**; **65.6% of items appear exactly once**.
  - **33.5% of the held-out TEST targets appear only once in the whole file** ->
    they are **never in training**, so NO model can rank them (their item
    embedding is untrained/random). A further ~12% are seen only twice.

So HR@10 is capped near zero *by construction*: a third of the test set is
unrankable regardless of model, and the remaining trainable targets (median 3
appearances) have barely-learned embeddings competing against 1.8M items under
full ranking. The BPR + one-negative contract compounds this on the RNN/HGN
baselines, but it is a secondary factor, not the primary one.

Note: the Cycle-8 record already flagged that the heavy tiers "need the heavy-tier
item floor (MIN_ITEM_INTER=10, k-core)" — that floor was planned for 27K but never
applied to the 1K build. This is precisely why supervisor decision **D6** moved the
primary tier to Pure (7,211 items, each seen many times -> trainable).

**The fix (if a real 1K tier is wanted):** rebuild kuairand_1k with a k-core /
MIN_ITEM_INTER item floor so every retained item (and every test target) is seen
enough times to be trainable, then re-run. This is a data-build change, not a
model change.

## Recommendation (cost-aware)

- **Do not** treat the 1K zero as a comparative result, and do not fold it into
  `thesis_table.csv` (Pure) as if it were one.
- Completing a *meaningful* 1K tier is not just a re-run: the BPR baselines need a
  protocol change (many more sampled negatives, or a sampled-metric eval such as
  the retired uni100) to be non-degenerate at 1.8M items, and each 1K run is
  expensive on A10G (one SASRec CE epoch >25 min; ~$5–10 for the four-model set,
  per the 08-14 record). SASRec/CAFREC use full-softmax CE and *may* be less
  degenerate than the BPR models, but still at real GPU cost and OOM risk.
- Cheapest informative next step, **only if the 1K upside is wanted**: a single
  SASRec-1K CE diagnostic (~$1–2) to see whether full-softmax escapes the zero
  floor. If it does not, the 1K tier is not viable under `mode:full` without an
  eval-protocol change, and Pure stands as the sole reported tier — consistent
  with D6.

## Diagnostic RESULT (2026-08-25) — CE does NOT escape the floor

Ran the SASRec-1K full-softmax CE diagnostic (2 epochs, seed 403092, mode:full,
detached on Modal; ~1h wall, ~$1). Result (independent `ranks_check` agrees):

| model | loss | test HR@10 | NDCG@10 | valid HR@10 |
|---|---|---|---|---|
| SASRec | CE | **0.0020** | 0.0008 | 0.0000 |
| HGRU4Rec (prior) | BPR | 0.0000 | 0.0000 | 0.0010 |

SASRec-CE reaches HR@10 = 0.002 (2 of 1,000 test users) — marginally above the
BPR zero, but ~40× below Pure's ~0.078 and effectively still on the floor. So
**full-softmax CE does not rescue the 1K tier**: the cause is structural (33.5% of
test targets never appear in training → unrankable by any model/loss), not the
BPR + one-negative contract. CONCLUSION: 1K is non-viable under `mode:full`
without a k-core / item-floor rebuild; the loss function is not the lever. **Pure
stands as the sole reported tier (D6 confirmed).** The 1K result is NOT folded
into `thesis_table.csv`; it is a methodology/limitations point only. A genuine 1K
tier would require rebuilding `kuairand_1k.inter` with MIN_ITEM_INTER≥10 (k-core)
and re-running — a data-build change deferred to future work.
