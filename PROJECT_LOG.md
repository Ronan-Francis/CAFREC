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

------------------------------------------------------------
Cycle 9 Working Record — Primary reporting tier: 1K -> Pure
Date   : 2026-08-12
Block  : 2 — Core Build
Author : R. Francis
------------------------------------------------------------

SUPERVISOR DECISION (Mike, 2026-08-12)
  No Modal credits available; no departmental GPU resources available. Suggested
  free-GPU alternatives: Beam ($30/mo free GPU credit, similar to Modal) and
  Kaggle (free GPU, more setup). Explicitly approved moving the PRIMARY reporting
  tier from kuairand_1k to kuairand_pure if compute remains a constraint, on the
  grounds that a COMPLETE four-model result set on Pure is worth more than an
  incomplete evaluation blocked on compute. Change + reason to be documented in
  the write-up (this record + thesis edit M2).

DECISION (D6): PRIMARY REPORTED TIER -> kuairand_pure.
  This SUPERSEDES the Cycle 8 D2 tier-role assignment (which named kuairand_1k as
  PRIMARY). New roles:
    kuairand_pure -> PRIMARY reported results (all four models, mode:full)
    kuairand_1k   -> SECONDARY / upside, only if free compute is secured
    kuairand_27k  -> heavy-scale, deferred to paid/credit compute

  WHY THIS ALSO UN-BLOCKS THE WORK (not just a scope cut):
    The Modal spend-limit block was specific to 1K/27K, which carry the full
    ~million-item video catalogue. Under mode:full the CE loss materialises a
    (batch x n_items) logits tensor that OOMs anything below a 24 GB card (see
    modal_run.py:72 rationale; A10G required). Pure's catalogue is only 7,551
    items, so full-ranking CE fits comfortably in local CPU/RAM. Switching the
    primary tier to Pure therefore removes the GPU dependency entirely — the
    complete four-model result set can be produced locally with zero cloud spend.

COMPUTE-OPTIONS SURVEY (for the 1K/27K upside; recorded for the write-up)
  Kaggle    free, 30 hr/wk, P100/T4 16 GB, 12 hr/session. Handles Pure trivially;
            1K workable at train_batch_size=512 (workaround already in modal_run.py);
            27K RAM-tight. First choice for a free 1K run.
  Modal     $30/mo Starter free tier RESETS MONTHLY — the "spend limit exceeded"
            block may already be lifted at the start of a new month; check before
            porting elsewhere. Existing modal_run.py already wired.
  Beam      $30/mo free credit, A10G 24 GB (~$1.05/hr); serverless like Modal so
            modal_run.py ports with modest rework. 24 GB covers 27K. Best full-
            harness remote target if Modal stays blocked.
  Colab     free tier unreliable/not guaranteed; Pro $11.99/mo for stable 16 GB.
  Vast.ai / RunPod  pay-as-you-go A100 ~$0.67-1.49/hr; a few dollars runs the
            entire matrix including 27K. Cheapest paid path.

ACTION TAKEN
  Launched LOCAL CPU full-ranking runs (configs/base.yaml: mode:full, epochs 10,
  early-stop NDCG@10, stopping_step 3) for all four models:
    SASRec, HGN, HGRU4Rec  -> kuairand_pure
    CAFREC                 -> kuairand_pure_ctx  (six causal context features)
  Results land in cafrec_harness/results/<model>_<dataset>_seed<seed>_<ts>.json
  and append to results/summary.csv. These are the NEW mode:full numbers that
  replace the void Cycle 7 uni100 baselines.

THESIS EDIT (extends M2)
  The tier-role text (Sec 2.1 / results framing) must state kuairand_pure as the
  primary evaluation tier, with the compute-constraint rationale above and a note
  that 1K/27K remain as scale-generalisation checks pending GPU access. Pair this
  with the already-flagged M2 uni100 -> full-ranking edit.

REMAINING
  * Collect the four Pure result JSONs; rebuild the results table + diversity
    metrics (RON-14/15) on the trained top-k dumps.
  * 1K as upside: attempt on Kaggle (or Modal if the monthly free tier reset).
  * 27K: defer to paid/credit compute.

UPDATE (2026-08-12, later): CPU RUN TOO SLOW -> SEQUENCE-LENGTH SPEED-TUNE
  The first local run under the full base.yaml config (MAX_ITEM_LIST_LENGTH=50)
  was impractically slow on CPU (no local CUDA GPU): after ~2h10m it had not
  completed even epoch 0 of SASRec (model 1 of 4), while pegging ~4 cores. Full
  matrix projected at >1 day. Terminated and speed-tuned.

  CONFIG CHANGE (methodological, to report in the write-up):
    MAX_ITEM_LIST_LENGTH 50 -> 20 in configs/base.yaml. Cost of the sequential
    models' self-attention scales with seq_len^2, so ~6x less compute per step.
    Applied UNIFORMLY to all four models (SASRec, HGN, HGRU4Rec, CAFREC) so the
    cross-model comparison stays fair. Rationale: KuaiRand sessions are short, so
    a 20-interaction causal history window is an adequate cap for next-item
    prediction; the truncation is a compute-driven choice, not a modelling
    advantage for any one model. epochs unchanged (10, early-stop NDCG@10 pat 3).
    NOTE for thesis: state the 20-item cap and this rationale where the sequence
    encoder / training setup is described.
  ENV: OMP/MKL threads set to 8 (all logical cores) for the relaunch.

RESULTS — kuairand_pure, mode:full, seed 2020, seq_len=20 (2026-08-12, ~2h total)
  Split: 22,913 users; 582,363 train / 22,912 valid / 22,912 test; 7,211 items.
  Full ranking (true item vs all items). HR@10 == Recall@10 (LOO, 1 relevant/user).
    Model      HR@10    NDCG@10   MRR@10    params
    SASRec     0.0795   0.0397    0.0279    562,880   (CE)   <- best
    CAFREC     0.0730   0.0364    0.0255    2,050,497 (CE)
    HGRU4Rec   0.0449   0.0232    0.0166    1,981,248 (BPR)
    HGN        0.0290   0.0131    0.0084    2,399,168 (BPR)
  Result JSONs: results/{SASRec_...174721, HGN_...175100, HGRU4Rec_...181006,
    CAFREC_kuairand_pure_ctx_...184720}.json (+ summary.csv).

  READING:
  * Numbers are ~10x below the void Cycle 7 uni100 figures (HR@10 ~0.64) purely
    from the uni100 -> full-ranking protocol switch (D2). Do NOT cross-compare.
  * CAFREC (0.0730) is marginally BELOW SASRec (0.0795), i.e. NOT yet beating the
    strongest baseline. Consistent with build state: z_long is still the learnable
    STAND-IN (llm_profile_path=None; RON-24 profiler not loaded -> RQ2/H2 mechanism
    inert, per D5) and NO hyperparameter tuning yet (RON-31/32 pending). This is the
    expected pre-profiler / pre-tuning parity point, not a refutation of H1-H4.
  * Robustness: prior seq_len=50 SASRec (results ...160222) HR@10 0.0785 vs
    seq_len=20 0.0795 -> near-identical; the seq_len=20 cap is defensible.

  NEXT: (a) RON-24 real profiler embeddings -> re-run CAFREC to actually test
  RQ2/H2; (b) grid search RON-31/32; (c) diversity/coverage (ILD/Coverage) on
  top-k dumps; (d) 1K upside on free GPU; (e) build the thesis results table.
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — RON-24 LLM PROFILER BUILT + RQ2/H2 TESTED ON PURE
Date   : 2026-08-14
Author : R. Francis
------------------------------------------------------------

COMPUTE PIVOT
  Modal bill PAID 2026-08-14 -> cloud GPU unblocked. Kaggle dropped. 1K attempted
  on Modal but is EXPENSIVE: 1.82M-item catalogue, 4.43M interactions over 1000
  dense users (~4,430 each) -> full-softmax CE caps batch ~1024 (7.5GB) and one
  SASRec epoch ran >25 min on A10G (CPU-dataloader-bound). Est. ~$5-10 for the
  four-model 1K set. DECISION (supervisor-cost tradeoff): test the profiler's
  RQ2/H2 payoff on PURE (7.2K items, the approved primary tier) for <$2 first;
  1K held as paid upside.

RON-24 PROFILER PIPELINE (built + validated)
  cafrec/features/build_profiles.py  causal, leakage-safe per-user temporal
    profile (TRAINING rows only, i.e. all but each user's last two LOO holdouts),
    row-aligned to RecBole's user remap, swappable offline backend:
      HashingBackend  deterministic (crc32) bag-of-tokens; deps-free no-LLM control.
      HFBackend       sentence-transformers embedder (the offline "LLM profiler").
  modal_profiles.py  A10G function; renders+remaps FIRST then loads the model, so
    a data bug fails before any multi-GB download; HF weights cached in hf-cache vol.
  modal_run.py  now accepts --llm-profile-path/--profile-dim/--ablation for CAFREC.
  Caches (bge-large-en-v1.5, d=1024) on cafrec-data volume:
    profiles/kuairand_1k_ctx.profiles.hf.d1024.pt    (1000/1000 users, 118s)
    profiles/kuairand_pure_ctx.profiles.hf.d1024.pt  (22912/22913 users, 118s)

RESULTS — CAFREC on kuairand_pure_ctx, mode:full, seed 2020, seq_len 20
    Condition                         HR@10    NDCG@10   MRR@10
    SASRec (strongest baseline)       0.0795   0.0397    0.0279
    CAFREC no_profiler ablation (0)   0.0791   0.0399    0.0281
    CAFREC real bge-large profiler    0.0786   0.0400    0.0285   <- best NDCG/MRR
    CAFREC learnable stand-in         0.0725   0.0354    0.0244

  FINDINGS:
  * CLEAR: real LLM profile >> learnable stand-in (+8% HR, +13% NDCG, +17% MRR).
    The frozen offline profile carries signal a learned embedding of equal
    capacity does not -> supports the RQ2/H2 premise; reframes the earlier
    "CAFREC below SASRec" (stand-in) as a representation gap, not an arch failure.
  * MARGINAL: profiler posts best-in-table NDCG@10 (0.0400) + MRR@10 (0.0285),
    edging SASRec + its own ablation on rank quality, but HR@10 (0.0786) is a hair
    below SASRec (0.0795)/ablation (0.0791). Top-3 gaps <1.5% -> NOT established
    without the paired Wilcoxon/bootstrap (RON-14). Run significance before any
    "CAFREC wins" claim.
  * Modal stand-in (0.0725) reproduces the earlier local-CPU CAFREC (0.0730) ->
    Modal-vs-CPU consistent; comparison is clean.

  Windows gotchas logged to memory: PYTHONUTF8=1 for `modal run` (✓ charmap
  crash); MSYS_NO_PATHCONV=1 so Git Bash does not mangle a /data/... CLI arg.

NEXT: (1) paired significance tests (Wilcoxon/bootstrap) profiler vs SASRec vs
  ablation on Pure; (2) OPTIONAL scale the profiler model up (7B embedder, "as big
  as possible") to see if the margin widens; (3) 1K as paid upside; (4) diversity
  metrics; (5) thesis results table + RQ2/H2 write-up.
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — 7B PROFILER + PAIRED SIGNIFICANCE TESTS (Pure)
Date   : 2026-08-14
Author : R. Francis
------------------------------------------------------------

NEW INFRA
  cafrec/eval/full_rank.py  per-user held-out ranks under FULL ranking, mirroring
    RecBole's _full_sort_batch_eval masking (pad+history -> -inf); keyed by
    ORIGINAL user id so models on different datasets (SASRec on pure, CAFREC on
    pure_ctx) pair correctly. runner return_ranks=; modal_run --dump-ranks/--tag.
    VALIDATED: aggregate-from-ranks == RecBole reported metrics for all 5 conds.
  build_profiles HFBackend now supports fp16 (dtype=) so a 7B embedder fits A10G.
  Profile caches on cafrec-data: pure_ctx d1024 (bge-large) + d4096 (e5-mistral-7b).

RESULTS — CAFREC on kuairand_pure_ctx, Modal A10G, mode:full, seed 2020, seq_len 20
  (all 5 rows this batch are same-environment -> directly comparable)
    Condition                       HR@10    NDCG@10   MRR@10
    CAFREC 7B profiler (e5-mistral) 0.0805   0.0403    0.0282
    CAFREC no_profiler ablation     0.0791   0.0399    0.0281
    CAFREC bge-large profiler       0.0786   0.0400    0.0285
    SASRec                          0.0776   0.0389    0.0274
    CAFREC learnable stand-in       0.0725   0.0354    0.0244

  PAIRED TESTS (Wilcoxon signed-rank + 95% paired bootstrap CI, n=22,912 users):
    profiler (7B & bge) vs STAND-IN : SIGNIFICANT on HR/NDCG/MRR, p<1e-6, CIs
      exclude 0 (e.g. 7B-standin NDCG Δ=+0.0048 [+0.0031,+0.0064]). => the real
      offline LLM profile is a materially better long-term representation than a
      learnable embedding of equal capacity. Core RQ2/H2 support.
    profiler vs NO_PROFILER ablation: NOT significant (NDCG p=0.44/0.76; HR ns).
    profiler vs SASRec              : NOT significant (7B NDCG p=0.11, HR p=0.057).
    7B vs bge-large                 : NOT significant (all p>=0.10). "As big as
      possible" gave NO significant lift over the 1024-d model -> bge-large suffices.

  HONEST READ: the profile clears the noise only against the stand-in. Against the
  strong short-term encoder (SASRec / the z_long=0 ablation) CAFREC-with-profiler
  is statistically TIED on Pure — it no longer HURTS (as the stand-in did) but adds
  no significant gain. Note SASRec here (0.0776) vs the earlier local-CPU SASRec
  (0.0795): ~0.002 HR of train nondeterminism across hardware, comparable to the
  between-condition gaps -> single-seed aggregate diffs are unreliable; the paired
  per-user tests are the trustworthy comparison. Multi-seed runs would tighten this.

NEXT: HGN/HGRU4Rec re-run on Modal for a same-env thesis table; then diversity
  metrics; 1K paid upside; multi-seed for a publishable significance claim.
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — SEED-403092 MULTI-SEED BATCH + RESULT CONSOLIDATION + 1K LAUNCH
Date   : 2026-08-15
Author : R. Francis
------------------------------------------------------------

COMPUTE
  Modal (paid tier, unblocked since 2026-08-14). All runs A10G, mode:full,
  seq_len 20, epochs 10 (early-stop NDCG@10, patience 3). Ranks dumped for every
  condition (--dump-ranks) so the paired per-user tests can be recomputed offline.

WHAT WAS RUN (seed 403092 = project/student seed; second seed alongside 2020)
  PURE head-to-head (kuairand_pure / kuairand_pure_ctx), completed this session:
    SASRec              tag seed403092    (~3 min wall)
    CAFREC no_profiler  tag noprof_s403092
    CAFREC bge-profiler tag bge_s403092   (profiles/kuairand_pure_ctx...d1024.pt)
  1K tier (kuairand_1k / kuairand_1k_ctx), LAUNCHED as background/over-time
  compute (results pending, folded in on completion):
    SASRec, HGN, HGRU4Rec  tag base1k_s403092
    CAFREC bge-profiler     tag bge1k_s403092 (profiles/kuairand_1k_ctx...d1024.pt)
  Note: kuairand_1k_ctx.inter was already rendered + uploaded (Cycle 8 listed it as
  "remaining"; volume check 2026-08-15 confirms it present), so the 1K CAFREC leg
  is NOT blocked.

CONSOLIDATION (data hygiene)
  The 2026-08-14 same-env Modal batch (SASRec, CAFREC noprof/standin/prof_bge/
  prof_7b, HGN base, HGRU4Rec base) previously lived only on the cafrec-results
  volume and in this log's prose. Pulled to local results/ and consolidated with
  all prior JSONs into results/thesis_table.csv (condition tag derived per file),
  16 rows. This is now the single on-disk source for the results chapter.

RESULTS — PURE, same-env Modal, mode:full, seq_len 20, two seeds (2020 / 403092)
    Condition            HR@10  (2020 / 403092 -> mean)   NDCG@10 -> mean   MRR@10 -> mean
    SASRec               0.0776 / 0.0793 -> 0.0785        0.0395           0.0279
    CAFREC no_profiler   0.0791 / 0.0807 -> 0.0799        0.0406           0.0287
    CAFREC bge-profiler  0.0786 / 0.0795 -> 0.0790        0.0404           0.0289
  (learnable stand-in remains 0.0725 HR at seed 2020; not re-run at 403092.)

  READING:
  * On the 2-seed MEAN, both CAFREC variants now edge SASRec on all three metrics.
    The single-seed-2020 picture (SASRec HR 0.0776 ABOVE CAFREC) was seed noise:
    per-condition HR moved 0.0776->0.0793 (SASRec) and 0.0791->0.0807 (noprof)
    between seeds, a ~0.0016 swing comparable to the between-condition gaps. This
    re-frames the earlier "CAFREC below SASRec" line as single-seed noise, not an
    architecture deficit.
  * bge-profiler vs no_profiler stays TIED (noprof slightly higher HR/NDCG, bge
    slightly higher MRR; direction flips across seeds). No evidence on Pure that the
    frozen profile beats the z_long=0 ablation, consistent with the 08-14 paired
    tests (profiler vs no_profiler not significant).
  * The one ROBUST effect remains profiler >> stand-in (0.0725): the load-bearing
    RQ2/H2 evidence.

  CAVEAT: 2-seed means are NOT significance. The trustworthy comparison is the
  per-user paired Wilcoxon/bootstrap recomputed PER SEED from the dumped ranks
  (all six seed-403092 + seed-2020 rank files are on cafrec-results). A 3rd/4th
  seed would tighten the aggregate further.

DIVERSITY (ILD / Coverage) — STILL BLOCKED, WRONG DUMP TYPE
  RON-15 code is implemented + tested (cafrec/eval/diversity.py: coverage,
  intra_list_diversity, topk_from_scores; category_matrix.py). BUT ILD/Coverage
  need per-user TOP-K RECOMMENDED ITEM-ID LISTS ([n_users, k]); the current
  --dump-ranks path (full_rank.py) emits only each user's RANK OF THE HELD-OUT
  TARGET (the paired unit for significance), not the top-k lists. So diversity
  cannot be computed from any existing dump. FIX: extend full_sort_ranks to also
  capture topk_from_scores(masked_scores, k) keyed by user id, then a cheap Pure
  re-run (or fold into the next seed batch) produces the top-k dumps; ILD needs the
  RecBole-remap-aligned category_matrix and Coverage needs n_items (Pure = 7,211).

ARTIFACTS
  results/thesis_table.csv                  16-row consolidated table (this session)
  results/*_seed2020_2026081[24]*.json      08-12 local + 08-14 Modal same-env
  results/*_seed403092_20260815*.json       this session's Pure seed-403092 runs
  cafrec-results volume                     rank dumps for all conditions; 1K JSONs
                                            land here as those jobs finish

NEXT
  1. Recompute paired Wilcoxon/bootstrap PER SEED (2020 + 403092) from the rank
     dumps -> turn the 2-seed means into a defensible significance statement.
  2. Extend full_rank.py with a top-k dump; produce ILD/Coverage baselines (RON-15).
  3. Collect the 1K JSONs; add the 1K tier to thesis_table.csv as a scale check.
  4. Optional 3rd/4th seed on Pure for a publishable multi-seed claim.
  5. Draft the results chapter table + RQ2/H2 write-up off thesis_table.csv.
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — 1K COLLECTION + OPTIONAL TASKS (RON-40 / RON-44 / multi-seed)
Date   : 2026-08-20
Author : R. Francis
------------------------------------------------------------

1K TIER COLLECTION -> ONLY ONE JOB LANDED, AND IT IS DEGENERATE
  The 08-15 background 1K batch left exactly ONE JSON on cafrec-results:
  HGRU4Rec_kuairand_1k_topk1k_s403092 -> test HR@10 = NDCG = MRR = 0.0 (valid
  HR@10 0.001; independent ranks_check agrees). Root cause VERIFIED 2026-08-20 by
  inspecting kuairand_1k.inter: a DATA-PREPROCESSING gap, not a code bug (same
  harness is fine on Pure) and not mystical scale. The 1K .inter has NO item floor
  -> 1,819,489 items over 1,000 users (2.43 mean appearances/item; 65.6% of items
  appear once), and **33.5% of held-out test targets appear only once in the whole
  file -> never in training -> unrankable by ANY model**. HR is thus capped near
  zero by construction. The BPR + one-negative contract is a secondary compounding
  factor. The Cycle-8 MIN_ITEM_INTER=10/k-core floor was planned for 27K but never
  applied to the 1K build. FIX = rebuild kuairand_1k with a k-core item floor, then
  re-run (data-build change, not model). The SASRec/HGN/CAFREC legs produced no
  JSON (did not complete). Documented in results/analysis_1k_status.md. DECISION: do NOT fold the 1K zero into the
  Pure thesis_table; the degenerate BPR result confirms D6's rationale for making
  Pure the primary tier. A single cheap SASRec-1K full-softmax CE diagnostic was
  launched to test whether CE escapes the zero floor (result appended below when
  the run finishes).

RON-40  R^4 CLASSIC-FOUR x_ctx ABLATION -> AUXILIARY FEATURES ADD NOTHING.
  Added a --context-fields override to modal_run.py (comma-separated subset of the
  six ctx columns; n_context_features derived from the count; no model change --
  CAFREC._build_context already selects columns by name). Ran CAFREC bge-large
  profiler with the four CORE features only (dropping inter_session_gap_log_z and
  is_first_session), 3 seeds, mode:full, seq_len 20. Gate MLP is 128 params
  smaller (2 features x 64 units), confirming the narrower input.
    R^4 (classic-four) 3-seed mean: HR@10 0.0788, NDCG@10 0.0404, MRR 0.0289
    R^6 (full)         3-seed mean: HR@10 0.0790, NDCG@10 0.0405, MRR 0.0289
  Paired per-user Wilcoxon + bootstrap (R^6 - R^4), every seed x metric: mean Δ
  tiny, 95% CI straddles 0, p in [0.51, 0.97]. The two auxiliary recency features
  give NO significant lift over the four core RQ3 features -> supports the M1
  framing (four CORE + two AUXILIARY, marginal value tested and negligible).
  results/analysis_ron40_r4_ablation.md (+ ron40_r4_ablation.py).

RON-44  SESSION-INTENT PROXY STRATIFICATION -> NUANCED / AGAINST H3.
  Proxy from the test-row prefix context features (causal). Both behavioural
  features are ~85-90% concentrated at their floor on Pure (short/single-category
  histories), so each proxy is a BINARY split: category-drift Focused (85%) vs
  Exploratory (15%); dwell-entropy Low (89%) vs High (11%). Per-user HR/NDCG
  reconstructed from dumped ranks (asserted == RecBole aggregate), 3-seed means.
  FINDING: the Exploratory / high-dwell minority -- the cohort H3 says context
  gating should help MOST -- is where SASRec WINS (CAFREC NDCG Δ vs SASRec is
  negative there: drift -0.0015/-0.0018, dwell -0.0007/-0.0008). CAFREC's small
  edge is carried by the Focused/Low majority instead. Report as a nuanced RQ3/H3
  limitation, not support. Magnitudes <0.002 NDCG; pair with paired tests before
  any directional claim. results/analysis_intent_strata.md (+ ron44_intent_strata.py).

MULTI-SEED BASELINES (HGN, HGRU4Rec) on Pure, seeds 2021 + 403092 (had only 2020),
  with rank+topk dumps -> 3-seed symmetry and diversity availability for the two
  weak baselines. Non-degenerate and consistent with seed 2020:
    HGN      HR@10 ~0.019-0.029 across seeds (weak, variable)
    HGRU4Rec HR@10 ~0.042-0.049 across seeds
  All 7 new runs (3 RON-40 + 4 baselines) appended to thesis_table.csv (now 32 rows).

REASONING-LLM SESSION-INTENT TICKETS -> FORMALLY SHELVED (per D4).
  RON-20/21/22/33 (GPT-4o/Claude reasoning session-intent label) are CANCELLED:
  they never appear in the thesis RQs/architecture/H1-H4 (only in the embedded
  Linear backlog), and RON-44 above delivers the intended session-intent analysis
  via a behavioural PROXY instead. No reasoning-model inference will be built.

STILL OPEN (thesis-writing, no compute)
  * Thesis edits M1 (four core + two auxiliary; cite RON-40), M2 (uni100 -> full
    ranking; Pure primary tier per D6), M3 (HRNN -> HGRU4Rec).
  * Results chapter table + RQ2/H2 + RQ3 write-up off thesis_table.csv and the
    analysis_*.md files (3-seed significance, diversity, RON-40, RON-44).
  * 27k tier: deferred to paid/credit compute (heavy).
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — T3.1 STRUCTURAL ABLATIONS + CAPACITY LADDER (Pure)
Date   : 2026-08-25
Author : R. Francis
------------------------------------------------------------

Launched 10 parallel Pure legs on Modal (run_parallel_pure.py, train.spawn):
static_gate x3 seeds, concat x3, prof_7b x{2021,403092}, standin x{2021,403092};
default 10-epoch schedule (comparable to reported bge), dump-ranks + dump-topk.
Appended to thesis_table.csv (now 42 rows). Paired per-user Wilcoxon + bootstrap
vs reported CAFREC-bge in results/pure_ablations_sig.py + analysis_pure_ablations.md.

FINDINGS (bge - condition, per-user, 3 seeds unless noted):
  T3.1 static_gate  -> context-adaptive gate BEATS a constant gate. +NDCG all 3
        seeds, SIGNIFICANT on 2/3 (s2021 p=3.7e-4, s403092 p=3.3e-4; s2020 ns).
        ~+0.002 NDCG. Positive RQ3 support; reconciles with RON-44 (helps the
        focused majority in aggregate, not the exploratory minority specifically).
  T3.1 concat       -> gated fusion >> concatenation. SIGNIFICANT every seedxmetric
        (p 1.8e-8 .. 4.6e-20), ~+0.005 NDCG. Validates the element-wise gated
        fusion design (RON-29/M-fusion).
  standin (RQ2)     -> frozen profiler >> learnable z_long, now robust across the
        added seeds (p 2.9e-9 .. 5.6e-11), ~+0.006 NDCG / +0.009 HIT.
  prof_7b           -> NO gain over bge-large (NS 3/4 cells; one marginally favours
        bge). Profiler capacity SATURATES at bge-large -> reported operating point.
  CAVEAT: all gaps are vs CAFREC-bge (internal ref); bge itself is TIED with SASRec
  on aggregate accuracy. Ablations show the components matter relative to each
  other, not that CAFREC beats the short-term baseline on this tier.

These give the thesis a clean T3.1 ablation table: both architectural claims
(context-adaptive gating; gated fusion) are individually validated with paired
significance, and the profiler capacity ladder (standin << bge ~= 7B) is complete.
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — H2 SPARSE/DENSE COHORT TEST -> HYPOTHESIS REVERSED
Date   : 2026-08-25
Author : R. Francis
------------------------------------------------------------

Direct test of H2 (LLM profiler helps SHORT-history users most): CAFREC_bge vs
CAFREC_noprof (profile on/off, identical arch), users split sparse(<20)/dense(>=20)
interactions (Sparse 10,345 / Dense 12,567; median history 22). Per-user metrics
from dumped ranks, 3-seed, paired Wilcoxon within cohort. h2_sparse_strata.py +
analysis_h2_sparse.md (free, no compute).

FINDING — H2 IS REVERSED, with significance:
  * Sparse users: profiler is a NET NEGATIVE (NDCG bge-noprof -0.003/-0.002,
    sig s2020/s2021; null s403092). noprof posts the BEST sparse NDCG (0.0543 >
    SASRec 0.0525 > bge 0.0526).
  * Dense users: profiler is the intended NET POSITIVE (NDCG +0.0015, sig
    s2021/s403092).
  * Mechanism: an LLM profile built from a THIN history is noisy and displaces the
    already-good short-term signal; a profile from a RICH history carries real
    long-term taste. The profiler needs history to help -- opposite of H2.
  * Explains the aggregate tie: sparse loss + dense gain cancel to ~0. The cohort
    split is what makes the real behaviour visible.
  * Future work motivated: gate the profile on history sufficiency (down-weight
    z_long when history too thin), which the current context gate does not do.

This is a PRIMARY RQ2/H2 result for the write-up (honest hypothesis refutation),
stronger and cleaner than the RON-44 intent stratification.
------------------------------------------------------------

------------------------------------------------------------
Cycle 9 (cont.) — 1K CE DIAGNOSTIC + SEQ-LENGTH SENSITIVITY
Date   : 2026-08-25
Author : R. Francis
------------------------------------------------------------

1K DIAGNOSTIC (detached SASRec-1K full-softmax CE, 2 epochs, s403092, ~1h, ~$1):
  test HR@10 = 0.0020 (ranks_check agrees), valid 0.0. Marginally above the
  HGRU4Rec BPR 0.0 but ~40x below Pure -> CE does NOT escape the floor. Cause is
  STRUCTURAL (33.5% of test targets never in training), not the BPR/one-negative
  contract. 1K non-viable under mode:full without a k-core rebuild; loss is not the
  lever. Pure remains the sole reported tier (D6 confirmed). NOT folded into
  thesis_table. analysis_1k_status.md updated.

SEQ-LENGTH SENSITIVITY (SASRec/bge/noprof x seq_len{10,20,50} x 3 seeds; new
--max-seq-len flag; run_seqlen_sweep.py; +18 rows -> thesis_table 60 rows):
  KEY FINDING: the SASRec tie is SPECIFIC to seq_len 20. CAFREC-noprof beats
  SASRec with paired significance on ALL 3 seeds at BOTH seq_len 10 (p 1e-3/7.6e-3/
  3e-2) and seq_len 50 (p 4.4e-2/9.1e-4/3.8e-2), dNDCG ~+0.002 (~+5% rel); at
  seq_len 20 it is the known ns tie. Short window: the gate compensates for a
  starved encoder (SASRec drops, CAFREC holds). Long window: noprof posts the
  project-best Pure accuracy (NDCG 0.0414, HR 0.0819) and beats SASRec; bge trails
  noprof (profiler adds noise, per H2). The edge is the CONTEXT GATING (bge<=noprof
  everywhere), not the LLM profiler. analysis_pure_seqlen.md + pure_seqlen_sig.py.
  -> First clean CAFREC>SASRec evidence with significance; report with the caveat
  that seq_len 20 (tuned default) stays a tie and magnitudes are small.
------------------------------------------------------------
