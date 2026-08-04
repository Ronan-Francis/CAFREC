------------------------------------------------------------
T1.1 Dataset Preparation — Decision Record
Date   : 2026-06-11
Block  : 1 — Foundations
Author : R. Francis
------------------------------------------------------------

DATASET SELECTED
  KuaiRand-Pure  (schema inspection + session engineering)
  Zenodo DOI  : 10.1145/3511808.3557624
  MD5 (tar.gz): 0820331067a3784d9691136f772b35a7
  NOTE: switch to KuaiRand-1K or 27K for full model training.

SCHEMA CONFIRMED
  All 19 interaction-log columns present; zero null values.
  time_ms  (int64, ms) present  -> enables 30-min gap segmentation.
  is_rand  (int8)      present  -> search_to_rec structural replacement.
  tab      (int8, 0-14) present -> scenario / surface signal.
  play_time_ms + duration_ms    -> dwell_ratio computable.

SESSION SEGMENTATION
  Algorithm : 30-min inactivity gap on time_ms per user
  Results   : [see Cell 5 printed output]

COHORT COUNTS  (standard-recommendation rows, is_rand == 0)
  Sparse (< 20) : [see Cell 6 printed output]
  Dense  (>= 20): [see Cell 6 printed output]

FIELD MAPPING — FINAL DECISIONS
  +--------------------------------+------------------------------------------+
  | CAFREC feature                 | KuaiRand source                          |
  +--------------------------------+------------------------------------------+
  | session_length                 | len(session group) via time_ms 30-min gap|
  | dwell_time_entropy             | entropy(play_time_ms / duration_ms)      |
  |                                |   per session (10-bin histogram, base-2) |
  | category_drift_rate            | tag_id from video_features_basic         |
  |                                |   change-count / (session_len - 1)       |
  | search_to_rec_flag             | policy_transition_flag:                  |
  |   (KuaiSAR replacement)        |   is_rand value change within session    |
  |                                |   (standard <-> random policy boundary)  |
  +--------------------------------+------------------------------------------+

RATIONALE FOR is_rand -> search_to_rec REPLACEMENT
  KuaiSAR: flag = 1 when user crossed from search UI into rec feed.
  KuaiRand: no search UI. Structural equivalent = is_rand boundary:
  a standard->random (or reverse) transition within a session marks
  a surface discontinuity with the same gating-relevant property —
  the user receives content outside their organic preference trajectory.
  Sensitivity test planned: T3.1 ablation without policy_transition_flag.

OUTPUTS
  data/logs_sessioned.parquet    (full log + derived columns)
  data/session_features.parquet  (four per-session context features)
  data/cohort_labels.parquet     (per-user cohort assignment)

OPEN ISSUES / RISKS
  R2a  tag_id cardinality should be verified on KuaiRand-1K/27K.
  R2b  KuaiRand-Pure item pool (7,551) is much smaller than KuaiSAR
       (2M items); category diversity may differ — rerun on 1K/27K.
  R2c  Absence of search interface means RQ3's search-to-rec sub-
       question is addressed indirectly; document in methodology chapter.
------------------------------------------------------------

------------------------------------------------------------
Cycle 1 Completion Record
Date   : 2026-06-12
Block  : 1 — Foundations
Author : R. Francis
------------------------------------------------------------

CYCLE 1 ISSUE STATUS
  RON-5  KuaiRand download and schema audit         -> DONE
  RON-6  Session segmentation and cohort labelling   -> DONE
  RON-7  KuaiRand context feature mapping            -> DONE (see open items below)
  RON-8  Dissertation — methodology section (data)   -> DEFERRED to future cycle (tracked)

RON-5 COMPLETION NOTES
  All 19 interaction-log columns confirmed present, zero nulls.
  MD5 checksum verified against Zenodo record.
  Three parquet outputs written and confirmed:
    data/logs_sessioned.parquet
    data/session_features.parquet
    data/cohort_labels.parquet

RON-6 COMPLETION NOTES
  30-min inactivity gap segmentation implemented and executed on
  combined log (log_standard_4_08_to_4_21, log_standard_4_22_to_5_08,
  log_random_4_22_to_5_08).
  Cohort split on standard-recommendation rows (is_rand == 0) only;
  random-policy rows excluded from cohort assignment counts to match
  KuaiSAR's organic-interaction definition.
  Cohort counts written to data/cohort_labels.parquet.

RON-7 COMPLETION NOTES
  Field mapping confirmed and locked (see FIELD MAPPING table above).

  TAG CARDINALITY AUDIT (KuaiRand-Pure)
    Category field : 'tag' (video_features_basic_pure.csv)
    Unique categories : 110
    Null tag count    : 96
    Top categories by video count:
      tag 39 -> 815 videos
      tag 3  -> 651 videos
      tag 9  -> 418 videos
      tag 6  -> 378 videos
      tag 20 -> 344 videos
    Assessment: cardinality of 110 is appropriate for category_drift_rate.
    Not so high that drift becomes noise; not so concentrated that the
    feature is uninformative. Feature is valid on Pure; rerun cardinality
    audit when switching to KuaiRand-1K/27K (risk R2a).

  MULTI-TAG HANDLING — OPEN DECISION (deferred to T2.1)
    Some videos carry compound tag values (e.g. '39,68', '39,43').
    Decision on encoding strategy deferred to T2.1:
      Option A: primary tag only (first value before comma)
      Option B: explode multi-tag rows into separate category events
      Option C: treat full string as a single category label
    This decision must be locked before category_drift_rate computation
    in T2.1. Add as first agenda item for T2.1 kick-off.
    Logged as risk R2d below.

  is_rand REPLACEMENT — DISSERTATION JUSTIFICATION NOTE
    The structural-equivalence rationale (is_rand boundary = surface
    discontinuity breaking the user's organic preference trajectory,
    analogous to KuaiSAR's search->rec transition) is final and logged
    above. This argument must appear verbatim in the methodology chapter
    (RON-8) to preempt examiner scrutiny. Do not rely on the PROJECT_LOG
    alone as the record — reproduce the full rationale in the dissertation.
    Sensitivity validation via T3.1 ablation remains the planned empirical
    backstop.

UPDATED OPEN ISSUES / RISKS
  R2a  tag_id cardinality should be re-verified on KuaiRand-1K/27K.
  R2b  KuaiRand-Pure item pool (7,551) << KuaiSAR (2M items); category
       diversity metrics may differ — rerun on 1K/27K before T3.2.
  R2c  Absence of search interface means RQ3's search-to-rec sub-question
       is addressed indirectly; document explicitly in methodology chapter.
  R2d  Multi-tag video encoding strategy for category_drift_rate not yet
       decided. Must be locked at T2.1 kick-off. Three options documented
       above. Whichever is chosen, report it as a methodological decision
       in the dissertation with sensitivity rationale.
------------------------------------------------------------

------------------------------------------------------------
Cycle 7 Working Record
Date   : 2026-07-23
Block  : 1 Foundations / 2 Core Build
Author : R. Francis
------------------------------------------------------------

SCOPE DECISIONS (this cycle)
  * Modal cloud-GPU runs: approved. LLM inference: NOT run this cycle.
  * Reasoning-model intent feature (RON-20/21/22): DEFERRED pending supervisor
    approval. No Claude/GPT calls built or run.
  * LLM profiler inference (RON-24): full run deferred; only the embedding
    FORMAT + loader contract (RON-25) built and tested against a synthetic
    cache.
  * Dissertation writing tasks (RON-12/33/34/35/58/59): skipped this cycle.
  * Target scales: features/profiler -> 27k (eventual); model runs -> 1k,
    Pure as validation.

DONE
  RON-16/17/18  Context features implemented at the HARNESS level as a tested,
                leakage-free (causal / prefix-only) module:
                  cafrec/features/context.py       (session_len, dwell_entropy,
                                                     cat_drift; 30-min sessions)
                  cafrec/features/build_context_inter.py  -> emits
                    data/recbole/kuairand_pure_ctx/kuairand_pure_ctx.inter
                    (same 651,099 rows / 22,912 users as the base .inter, plus
                     three :float context columns)
                  tests/test_features.py           (16 tests incl. causality)
                Row-alignment with the base atomic file confirmed.
  RON-26/28/29  Gating MLP + element-wise fusion verified end-to-end: the three
                context fields survive RecBole augmentation into the batch, the
                gating MLP receives non-zero gradients (features drive the gate,
                not the fallback).
  RON-30        CAFREC forward+backward on a 10% Pure slice (2,292 users):
                training loss decreases monotonically over 5 epochs
                8.42 -> 7.89 -> 7.55 -> 7.49 -> 7.43.
  RON-25        LLM profile-embedding format + loader (cafrec/features/profiles.py):
                shape/dtype/NaN validation, sparse-user (zero-row) handling,
                zero-fill of missing tail, and an end-to-end load into CAFREC
                (frozen, row-aligned). tests/test_profiles.py (9 tests).
  RON-14        Ranking metrics (HR/NDCG/MRR) + Wilcoxon + paired bootstrap were
                already implemented; test coverage confirmed green.
  RON-15        ILD + Coverage + category matrix already implemented; tests green.
                (Diversity BASELINES still need a trained-model top-k dump.)
  Baselines     Reference numbers locked from the existing Modal runs (uni100,
                seed 2020), Pure test split:
                  SASRec  HR@10 0.638  NDCG@10 0.375  MRR 0.295
                  HGN     HR@10 0.528  NDCG@10 0.292  MRR 0.220
                  CAFREC  HR@10 0.637  NDCG@10 0.372  MRR 0.291  (context_fields
                          empty -> fallback; ~= SASRec, as expected pre-features)
                  HGN(1k) HR@10 0.399  NDCG@10 0.262  MRR 0.219

OPEN DECISIONS (need sign-off before they can be closed)
  D1  x_ctx COMPOSITION IS INCONSISTENT ACROSS ARTIFACTS.
        - submitted plan / registry default : 5  (4 classic + reasoning intent)
        - T1.1 notebook context_features.parquet : 6
              [prefix_session_len_log_z, prefix_dwell_entropy_z,
               prefix_category_drift_z, inter_session_gap_log_z,
               prefix_policy_flag, is_first_session]
        - harness context.py (this cycle)   : 3  (session_len, dwell_entropy,
                                                   cat_drift)
      The T1.1 notebook is the richer, standardised, dissertation-facing
      pipeline; the harness module is the simpler path that already runs inside
      RecBole today. DECIDE the canonical x_ctx set, then align both. Note the
      notebook computes features over the FULL behavioural sequence (incl.
      non-click / random rows as context), whereas the .inter that CAFREC trains
      on is organic clicks only -> a join to attach notebook features to the
      .inter must reconcile that row-set difference.
  D2  EVALUATION PROTOCOL: base.yaml still uses eval_args mode: uni100 (99
      negatives). RON-60 specifies FLAG 3 = TO / LS:valid_and_test / mode:FULL,
      and the uni100 sampler ticket RON-13 was CANCELLED as related to RON-60.
      Switching to full ranking invalidates the uni100 reference numbers above
      and requires re-running all baselines. NOT flipped unilaterally.
  D3  BASELINE MODEL: RON-10 names HGRU4Rec/HRNN, which is not a RecBole
      built-in; only HGN is registered (documented deviation in the dataset
      builder notebook). Decide: register a RecBole RNN stand-in (e.g. GRU4Rec)
      or drop HGRU4Rec.

BLOCKED
  * Modal runs (RON-10 SASRec-1k, any full-ranking re-runs): the Modal workspace
    has EXCEEDED ITS SPEND LIMIT ("Resource exhausted"). No cloud runs can
    execute until the budget is raised/reset. Existing results predate the limit.

RON-60 STATUS (In Progress)
  Split-integrity guarantees TESTED in the harness:
    - last-interaction holdout per user      -> tests/test_leave_one_out.py
    - no prefix leakage in context features   -> tests/test_features.py
      (held-out target uses prefix only; mirrors the notebook's cell-21 assertion)
  "no is_rand==1 test targets" holds by construction: the .inter is organic-only
  (is_rand==1 dropped at build; the notebook splits LOO over organic rows only).
  REMAINING: the mode:full switch (D2) and the SASRec full-ranking sanity check
  (blocked on Modal budget).
------------------------------------------------------------

------------------------------------------------------------
Cycle 8 Working Record
Date   : 2026-08-04
Block  : 2 — Core Build
Author : R. Francis
------------------------------------------------------------

CYCLE 7 OPEN DECISIONS D1/D2/D3 — RESOLVED AND IMPLEMENTED

D1  x_ctx COMPOSITION -> SIX features, full-log context, organic targets.
    DECISION: standardise x_ctx on the notebook's six-feature set and align the
    harness to it. Features are computed over the FULL behavioural log (organic
    is_rand==0 AND random-policy is_rand==1 impressions kept as context); only
    ORGANIC CLICKS are emitted as training/eval targets. This keeps
    prefix_policy_flag (the is_rand -> search_to_rec structural replacement,
    RQ3) non-degenerate while the model trains/evaluates on organic clicks only.
    Had we computed features over organic-only rows, prefix_policy_flag would be
    a constant 0 (no is_rand transitions among organic rows) — the reason this
    decision was surfaced before implementing.

    x_ctx (order): prefix_session_len_log_z, prefix_dwell_entropy_z,
                   prefix_category_drift_z, inter_session_gap_log_z,
                   prefix_policy_flag, is_first_session   (4 z-scored + 2 binary)

    IMPLEMENTED (harness now at parity with T1.1 notebook cells 28/30):
      cafrec/features/context.py         six prefix-causal features over the full
                                         log; requires is_rand; standardize_context
                                         fits mean/std on TRAIN rows only.
      cafrec/features/build_context_inter.py  full-log compute -> organic-click
                                         emit; 9-column .inter + ctx_scaler_params.json.
      cafrec/registry.py (CAFREC)        n_context_features=6, context_fields=<6>,
                                         context_load_col (promoted to load_col by
                                         the runner ONLY for a *_ctx dataset).
      cafrec/runner.py                   _ctx-only load_col promotion so baselines
                                         and the ml-100k fallback are unaffected.

    VERIFIED (KuaiRand-Pure, local):
      kuairand_pure_ctx.inter  interactions = 651,099  (EXACT match to the base
      kuairand_pure.inter row-set; users = 22,912). Features derived from
      2,622,668 full-impression context rows. policy_flag_rate = 0.0098 (non-zero
      -> feature is live, not degenerate). Scaler fit_rows = 605,275
      = 651,099 - 2*22,912 (leave-two-out per user). A RecBole load of CAFREC on
      kuairand_pure_ctx confirmed all six context fields arrive in the training
      batch (the gate uses the real x_ctx branch, not the session-length fallback).

D2  EVALUATION -> FULL RANKING; per-dataset workflow (NOT a cross-tier split).
    configs/base.yaml eval_args.mode: uni100 -> full. The "1K test / 27K train /
    Pure validation" intent is realised as a PER-DATASET workflow — each tier is
    a self-contained dataset with its own leave-one-out train/valid/test split;
    there is NO single cross-tier RecBole run (the tiers have disjoint user/item
    id spaces, so a model trained on one tier cannot be evaluated on another).
    Roles: kuairand_pure = fast dev/validation, kuairand_1k = PRIMARY reported
    results, kuairand_27k = heavy-scale training. Documented in base.yaml.

    CONSEQUENCE: mode:full INVALIDATES every uni100 baseline number from Cycle 7
    (SASRec/HGN/CAFREC HR/NDCG/MRR). All baselines must be re-run under full
    ranking before they are comparable. Full-ranking mechanics verified for all
    four models via the ml-100k smoke suite; KuaiRand full-ranking re-runs at
    scale remain BLOCKED on the Modal spend limit. 27K full-softmax CE over the
    full catalogue needs the heavy-tier item floor (MIN_ITEM_INTER=10, k-core).

D3  RNN BASELINE -> custom HGRU4Rec (Quadrana et al. 2017).
    Confirmed HGRU4Rec/HRNN is NOT a RecBole built-in (installed sequential
    models: gru4rec, hgn, hrm, sasrec, ...; hrm is a different model). Implemented
    cafrec/models/hgru4rec.py as a custom SequentialRecommender (session-GRU over
    the recent-item window + user-GRU; a learnable per-user state seeds the
    session-GRU). Registered "HGRU4Rec" (BPR contract). DOCUMENTED DEVIATION:
    RecBole's LOO loader hands one flattened item window per target with no
    in-window session delimiters, so true cross-session hidden-state propagation
    is approximated by the learnable per-user state; a per-interaction session
    index would enable the faithful variant (out of scope this cycle).

TESTS (all green: 56 passed, incl. full-ranking smoke for SASRec/HGN/HGRU4Rec/CAFREC)
  tests/test_features.py   reworked to the six-feature semantics; leakage/
                           prefix-causality guarantees preserved; adds
                           policy_flag / inter_session_gap / is_first_session /
                           train-only-scaler coverage.
  tests/test_hgru4rec.py   new: BPR loss decreases on a fixed batch; CE backprops;
                           full_sort/predict shapes.
  tests/test_smoke.py      HGRU4Rec added to the end-to-end parametrize.

REMAINING / STILL BLOCKED
  * Re-run all baselines + CAFREC under mode:full on kuairand_1k (primary) and
    kuairand_pure (validation); 27K when Modal budget is restored.
  * Build kuairand_1k_ctx / kuairand_27k_ctx (build_context_inter --tier
    medium|heavy); the 27K feature pass needs an out-of-core implementation.
  * Faithful HGRU4Rec cross-session variant pending a session-index list field.
------------------------------------------------------------

------------------------------------------------------------
Cycle 8 Addendum — Linear backlog + thesis alignment
Date   : 2026-08-04 (same day, later)
Author : R. Francis
------------------------------------------------------------

LINEAR BACKLOG RECONCILED TO THE BUILD
  Closed as Done (implementation + passing tests): RON-16/17/18 (context
    features), RON-26 (gating MLP), RON-29 (element-wise fusion), RON-30
    (end-to-end CAFREC), RON-25 (profile format/loader), RON-15 (ILD/Coverage).
  Duplicates: RON-28 -> RON-26; RON-19 -> RON-7.
  Rescoped to the six-feature build:
    RON-26/35/59  x_ctx now stated as R^6 (was R^5 "four classic + reasoning
                  intent"); Figure 4 = 6 inputs.
    RON-40        ablation retitled -> classic-four x_ctx (R^4), dropping the two
                  session-recency features (inter_session_gap_log, is_first_session).
                  Runnable via config override (context_fields=<4>, n=4); no new
                  model code.
    RON-44        rescoped -> session-intent PROXY from category-drift / dwell-
                  entropy strata (the reasoning-LLM label is not built).
    RON-31        "HRNN" -> "HGRU4Rec"; RON-32 dropped the reasoning-intent tuning.
  Moved out of Cycle 8: RON-20/21/22/33 (reasoning model), RON-24 (profiler
    inference) -> Cycle 9.

THESIS REVIEW (Thesis.pdf) — SCOPE DECISIONS
  D4  REASONING-LLM SESSION-INTENT FEATURE -> DROPPED.
      The thesis architecture (Sec 1.2), RQ3, H1-H4, and the contributions all
      specify FOUR session-context features (session length, dwell-time entropy,
      category drift rate, policy-transition flag) + the LLM temporal profiler +
      the short-term encoder. The GPT-4o/Claude session-intent label appears ONLY
      in the embedded Linear backlog table, never in the thesis's RQs/architecture
      -> it was never a thesis commitment. Cancel/shelve RON-20/21/22/33.
  D5  THE CORE LLM CLAIM IS THE TEMPORAL PROFILER, NOT THE REASONING FEATURE.
      RQ2 / H2 ("offline LLM temporal profiling helps sparse-history users") is a
      load-bearing question. In the current build z_long is a LEARNABLE STAND-IN
      (llm_profile_path=None). RQ2/H2 CANNOT be answered with the stand-in, so the
      real critical-path LLM task is RON-24 (run the profiler over all users and
      load frozen embeddings), not any reasoning-intent work.

THESIS <-> BUILD MISMATCHES TO FIX
  M1  FEATURE COUNT. Thesis/RQ3 = FOUR context features; the build emits SIX
      (added inter_session_gap_log + is_first_session). Resolution: keep the four
      as the RQ3 signals; frame the two extras as AUXILIARY features whose value
      is tested by the RON-40 ablation. Update RON-59 + the thesis text to say so.
  M2  EVALUATION PROTOCOL. Thesis Sec 2.1 still reads "leave-one-out with 99
      sampled negatives" (uni100) — this contradicts the mode:full switch (D2).
      Update Sec 2.1 AND RON-36 to full ranking, with the sampled-negative-bias
      rationale.
  M3  TERMINOLOGY. Thesis "HRNN" == build "HGRU4Rec" (same hierarchical-RNN
      family). Unify to one term across thesis + Linear.

UPDATED TO-DO (priority order)
  1. RON-24  Run the real LLM temporal profiler over all users; replace the
             learnable z_long stand-in with frozen embeddings. REQUIRED for RQ2/H2.
  2. Thesis edits: Sec 2.1 -> full ranking (M2); four-core + two-auxiliary feature
     framing (M1); HRNN/HGRU4Rec terminology (M3).
  3. Re-run baselines + CAFREC under mode:full on kuairand_1k (primary) +
     kuairand_pure (validation) [RON-10/11]; 27K when Modal restored. BLOCKED: Modal.
  4. Grid search RON-31/32 (shared + CAFREC-specific). BLOCKED: Modal.
  5. SASRec full-ranking sanity check vs published KuaiRand numbers (RON-60).
     BLOCKED: Modal.
  6. Build kuairand_1k_ctx / kuairand_27k_ctx (--tier medium|heavy); 27K needs an
     out-of-core feature pass.
  7. Re-validate diversity/metric baselines (RON-14/15) after the full-ranking
     baseline re-runs (their old uni100-based validation is void).
  8. OPTIONAL: faithful HGRU4Rec cross-session variant (needs a session-index
     field); cancel/shelve reasoning-model tickets RON-20/21/22/33 per D4.
------------------------------------------------------------
