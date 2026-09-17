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

------------------------------------------------------------
Cycle 9 (cont.) — DIVERSITY ACROSS ALL CONDITIONS (offline, no GPU)
Date   : 2026-08-26
Author : R. Francis
------------------------------------------------------------

ILD + Coverage for the ablation + seq-length conditions, computed offline from the
dumped top-k lists (original video_id tokens) joined to the local video-tag matrix
(7,583 x 46). NO Modal needed -- runner already emits topk as original tokens.
Method validated: reproduces the reported pure_3seed diversity exactly.
pure_diversity.py + analysis_pure_diversity.md.

FINDINGS:
  * ILD flat (~0.78-0.81) everywhere -- not discriminative (as before).
  * CAFREC coverage advantage is ROBUST to seq length: bge/noprof ~0.23-0.24 vs
    SASRec ~0.185-0.189 at seq 10/20/50 (~+22% rel, stable).
  * Coverage advantage is ARCHITECTURAL: concat collapses it to 0.186 (~=SASRec
    0.189), static_gate reduces it to 0.201; full gated models keep ~0.23. So the
    context-adaptive gated fusion drives the catalogue spread, NOT the LLM profile
    (bge~=noprof; prof7b~=bge). Ties the diversity win to the same mechanism the
    accuracy ablations validate.

Empirical phase now closed: accuracy (3-seed sig), ablations (T3.1 + capacity),
H2 cohort, seq-length sensitivity, diversity, 1K diagnostic all complete on Pure.
Remaining work is the write-up (gated on the external thesis draft).
------------------------------------------------------------

------------------------------------------------------------
Cycle 10 — RON-45 HISTORY-GATED PROFILER (the H2 fix)
Date   : 2026-08-26
Author : R. Francis
------------------------------------------------------------

Built the fix the H2 diagnosis pointed to: the context gate never saw history
LENGTH, so it could not suppress the profile for thin histories. Added
history_gate flag (cafrec.py + registry + modal_run --history-gate; +64 params):
appends item_seq_len/max_seq_length to the gate inputs. Validated locally on
ml-100k (both paths). Ran CAFREC-bge + history_gate x3 seeds on Pure (ranks+topk).
histgate_analysis.py + analysis_histgate.md. thesis_table -> 63 rows.

RESULT — the fix works as designed (modest but real):
  * BEST overall config on Pure: NDCG 0.0408 / HR 0.0805 (> SASRec 0.0396/0.0787,
    > bge 0.0405, > noprof 0.0406). Only +64 params over bge.
  * Sparse recovery: bge hurt sparse (0.0526 vs noprof 0.0543); histgate lifts to
    0.0534 (~47% of the gap recovered), positive vs bge on ALL 3 seeds.
  * Dense retention: histgate 0.0304 ~= bge 0.0305 (keeps profiler gain), > noprof
    (sig s403092 p=3e-4). The ONLY config strong on BOTH cohorts.
  * CAVEATS: small magnitudes; histgate>SASRec sig only 1/3 seeds; noprof still
    narrowly best on sparse alone. Report as VALIDATED MECHANISM + best single
    config, not a decisive win. Next iteration: unsaturated sufficiency signal
    (log total-history) or explicit multiplicative z_long suppression.

Completes the research arc: hypothesis -> rigorous test -> negative result ->
diagnosis (H2) -> targeted fix (RON-45) -> fix behaves as predicted, best model.
------------------------------------------------------------

------------------------------------------------------------
Cycle 10 (cont.) — 10-SEED SIGNIFICANCE: logfull > SASRec
Date   : 2026-08-26
Author : R. Francis
------------------------------------------------------------

Stage 4: ran SASRec/noprof/bge/logfull at 7 new seeds (-> 10 total). thesis_table
-> 97 rows. manyseed_sig.py + analysis_manyseed.md.

RESULT — the clean win:
  * 10-seed mean NDCG: SASRec 0.0397, noprof 0.0413, bge 0.0412, logfull 0.0422.
  * logfull vs SASRec: meanD +0.00254 (+6.4%), WINS 10/10 seeds, t-test p=5e-5,
    Wilcoxon p=2e-3 (n=10 floor); per-user sig 8/10 seeds. Decisive, robust.
  * CORRECTION: the earlier 3-seed "SASRec tie" was under-powered. At 10 seeds
    bge AND noprof also significantly beat SASRec (10/10, 9/10; p<0.01). Honest
    update: CAFREC context fusion gives a small-but-significant lift; logfull
    (history-gated profile) enlarges it. Magnitudes small; Pure-only. Report the
    seed-count correction plainly (more power, same direction).
  Est. spend to date ~$5-6 of ~$12.
------------------------------------------------------------

------------------------------------------------------------
Cycle 10 (cont.) — 1K GENERALISATION PROBE (honest negative)
Date   : 2026-08-26
Author : R. Francis
------------------------------------------------------------

Built kuairand_1k_kcore_ctx (build_1k_kcore_ctx.py) + rebuilt bge profiles for
the 998 floored users. Ran SASRec/noprof/logfull, 5 epochs, seed 2020, matched
interactions. run_1k_generalization.py + analysis_1k_generalization.md.

RESULT — the Pure win does NOT cleanly generalise:
  SASRec HR 0.0200 NDCG 0.0086 > logfull 0.0130/0.0061 > noprof 0.0130/0.0052.
  logfull > noprof holds (mechanism carries); CAFREC-vs-SASRec flips.
  Confounds: 1 seed, 5 epochs (CAFREC 24M params vs SASRec 0.56M -> under-
  converged), 998 test users (gap = handful of users), valid DISAGREES (logfull
  0.0112 > SASRec 0.0108). Coherent read: 1K users ultra-dense (~1340 inter each)
  -> short-term encoder best-fed, profile least needed (fits history-dependence).
  INCONCLUSIVE-to-NEGATIVE probe; clean verdict needs 10-epoch multi-seed.
  Est. budget ~$1-2 left.
------------------------------------------------------------

------------------------------------------------------------
Cycle 10 (cont.) — 1K GEN 10-EPOCH: negative HOLDS (confound resolved)
Date   : 2026-08-26
------------------------------------------------------------
Re-ran logfull vs SASRec at 10 epochs (kcore-1k, seed 2020) to rule out CAFREC
under-convergence. SASRec HR 0.0291 NDCG 0.0143 (valid 0.0157) > logfull HR 0.0210
NDCG 0.0093 (valid 0.0125). Gap WIDENED vs 5ep (+0.0025 -> +0.0050); valid now
AGREES with test. Under-convergence REFUTED -> the 1K negative is real. Honest
boundary condition: LLM-profile fusion helps only when short-term history is
limited (Pure); on ultra-dense 1K (~1340 inter/user) the short-term encoder
dominates. 1K single-seed (budget floor). EMPIRICAL PHASE DONE.
------------------------------------------------------------

------------------------------------------------------------
Cycle 11 — RON-31 GRID SEARCH: the untuned-baseline hole, closed
Date   : 2026-09-10
Author : R. Francis
------------------------------------------------------------

Reversed the Cycle 8 "grid search abandoned" position. The recorded reason
(budget exhausted at the Cycle 7-8 boundary) was STALE: the Modal bill was paid
2026-08-14 and ~$5-6 was subsequently spent on 10-seed + 1K work. The honest
framing is that budget was spent on POWER and GENERALISATION instead of tuning
-- a prioritisation, not a blocker. With the headline margin at only +0.00254
NDCG, an untuned SASRec was the softest target on the board.

RAN: full factorial, 2 levels, seed 403092, mode:full. SASRec 64/64 configs,
CAFREC-logfull 52/64. run_gridsearch.py -> gridsearch_ron31.csv. Then each
model's VALIDATION-best config re-run over the same 10 seeds as the defaults
study (run_tuned_confirm.py -> tuned_confirm.csv). analysis_gridsearch_ron31.md.

VALIDATION: config [000] == current defaults for each model, and both reproduce
  the published numbers (SASRec 0.0400 vs 0.0396-7; logfull 0.0422 vs 0.0422).
  Environment has not drifted.

RESULT -- THE CLAIM SURVIVES, THE MARGIN SHRINKS:
  * Tuned 10-seed: SASRec 0.04509 (sd 0.00076), logfull 0.04680 (sd 0.00077).
  * logfull - SASRec: +0.00171 (+3.79%), WINS 10/10, Wilcoxon p=1.95e-3
    (n=10 floor), t-test p=4.24e-4. Direction, consistency, significance intact
    after an EQUAL 64-config search for each -> not a tuning artefact.
  * BUT the margin falls from +0.00254 (+6.4%) to +0.00171 (+3.79%), -33% rel.
    Tuning lifted SASRec MORE (+0.00539) than CAFREC (+0.00460). Part of the old
    margin was a well-suited default meeting a poorly-suited one. Report +3.79%.
  * ONE PARAMETER DOMINATES: dropout 0.5 -> 0.2, in every top config for BOTH
    models. RecBole's 0.5 default costs ~12% NDCG on Pure. Every accuracy number
    currently in the results chapter was produced at 0.5 and is ~12% low.

COST -- OVERRUN, HALTED (record this honestly):
  Sweep estimated at $9.68-19.36 (GPU-only at ~$1.10/hr). ACTUAL burn hit ~$40
  before it was stopped at 116/176 runs. Cause: modal_run.train reserves cpu=8.0
  and memory=65536 MiB -- sized for the 27k million-item catalogue -- and Modal
  bills reserved CPU+MEMORY ON TOP of GPU. Real billed container time was ~7
  min/run, not the ~3 min training figure. ~4x under-estimate.
  FIX: added modal_run.train_pure (cpu=4.0, memory=16384) for Pure-tier work;
  `train` untouched so the 1k/27k paths keep the reservations they need. The
  18-leg confirmation ran on train_pure for ~$3-5.
  HGRU4Rec + HGN were NOT tuned (sweep halted). They stay at defaults and must
  be labelled as such -- an asymmetry to state, not hide. Neither is competitive,
  so H1 is unaffected.

THESIS EDITS REQUIRED:
  * 04_methodology.tex sec:hparam is now FALSE ("could not be executed"). Rewrite
    around what was run, the halt, and its reason.
  * Same paragraph cites risk R4; the compute risk is R1. R4 is "baselines fail
    to reproduce" -- which this sweep actually retires. Fix the cross-reference.
  * Headline -> +3.79%; report defaults as the untuned comparison.
  * RON-52: drop "no hyperparameter tuning"; add the partial-sweep asymmetry.
  * RON-31 -> Done (partial, documented). RON-32 stays Cancelled: its intent is
    covered by the profiler capacity ladder + histgate variant bake-off.
------------------------------------------------------------

------------------------------------------------------------
Cycle 11 (cont.) — RON-61 FUSION FORM: null result (honest)
Date   : 2026-09-10
------------------------------------------------------------

Ran the pre-scaffolded RON-61 fusion-form family (never previously executed;
fusion_form.csv was header-only). Added a HYBRID mode this session:
lambda_u = sigmoid(f(x_ctx) + b_u) -- context head + free per-user offset --
alongside the existing convex / additive / per_user modes. 3 arms x 10 seeds =
30 legs on train_pure, 0 failures, ~$2. run_fusion_form.py +
analysis_fusion_form.md.

    z = z_short + g * (z_long - (1 - lambda) * z_short)
    lambda 0 -> convex (incumbent)   lambda 1 -> additive

RESULT -- NO significant gain over the incumbent:
  convex 0.04680 | additive 0.04642 | per_user 0.04652 | hybrid 0.04718
  * hybrid vs convex: +0.00038, 7/10, p=0.235 -> NOT SIGNIFICANT. Nominally the
    best config to date but statistically TIED. Do NOT adopt it: +65 params and
    a more complex fusion equation cannot be defended on a null.
  * additive vs convex: -0.00038 -> the zero-sum property of convex fusion was
    hypothesised to be a handicap. REFUTED: removing it COSTS accuracy. The
    subtraction acts as a useful constraint.
  * per_user vs convex: -0.00028, 4/10 -> free personalisation of the FORM fails.
  * hybrid vs additive: +0.00076, 8/10, p=0.0195 -> conditioning helps WITHIN
    the additive family; it just does not beat plain convex.
  * hybrid vs SASRec tuned: +0.00209 (+4.64%), 10/10, p=1.95e-3 -- but convex
    already gives +3.79% at 10/10, so no form change is needed for the headline.

MECHANISM (seed 2020 lambda export):
  * per_user lambda NEVER LEFT INIT: 0.497 +- 0.024 over 22,913 users
    (sigmoid(0)=0.5). No features -> no gradient signal -> null by construction.
  * hybrid context head DID learn: bias -0.919, weights inter_session_gap -0.895,
    is_first_session +0.822, policy_flag -0.767, session_len_log +0.717.
  * logfull_history weight ~= -0.057 (near zero). DIRECT NEGATIVE ANSWER to the
    motivating hypothesis: the sparse/dense axis behind the H2 failure does NOT
    determine fusion FORM. H2 is a MAGNITUDE problem (logfull already fixes it),
    not a FORM problem.
  * Caveat: for hybrid the persisted per-user lambda is the BIAS component only;
    realised lambda also carries the per-request context term.
  * Asymmetry worth noting: the two AUXILIARY recency features RON-40 found
    contribute nothing to the GATE dominate the fusion-form head.

VERDICT: incumbent convex fusion stands. Report RON-61 in discussion/future work
as a tested-and-rejected design alternative with a mechanistic explanation.
EMPIRICAL PHASE CLOSED (again). Remaining work is WRITING: 06_design,
07_evaluation, 08_conclusion, 01_abstract are still ATU template stubs, and the
sec:hparam + R4->R1 fixes from the RON-31 entry are outstanding.
------------------------------------------------------------

------------------------------------------------------------
Cycle 10 Working Record — GATE CONTROLS (np_vector_gate / np_shuffled_ctx)
Date   : 2026-09-16
Block  : 3 — Evaluation
Author : R. Francis
------------------------------------------------------------

LOG GAP (flagged, not filled by this entry)
  This log jumps 2026-08-15 -> 2026-09-16. Undocumented work sitting in
  results/modal/: the 08-26 multi-seed batch (noprof_ms, seq10/seq50), the
  09-10 grid search (gs_CAFREC_logfull_*), the tuned configs (tuned_*) and the
  fusion-form seeds (ff_additive / ff_per_user / ff_hybrid). Those still need
  their own retrospective entry; the eval chapter currently cites them from a
  commented-out TODO(tuning) block.

WHAT RAN
  Two new CAFREC-NP gate controls (implemented in cafrec/models/cafrec.py,
  commit ae63ec4), each paired against no_profiler at seeds 42 / 77 / 123:
    np_vector_gate    g := sigmoid(theta), theta in R^H  -- gate keeps its
                      parameters but loses its context INPUT entirely.
    np_shuffled_ctx   x_ctx rows permuted within each batch, train AND eval
                      -- gate is fed another user's context.

  Executed twice:
    (1) 2026-09-15 22:07 -> 2026-09-16 02:18, local CPU via local_run.py
        --queue overnight. Seven runs SERIAL, ~2140s each, 4h11m total.
    (2) 2026-09-16 06:40, Modal A10G via modal_run.py::rerun. Same seven
        conditions, spawned in PARALLEL on train_pure. Total wall 169s.

  Run (2) exists because run (1) is not comparable to anything. See below.

DEFECT FOUND — local and Modal were never hyperparameter-matched
  modal_run.py sizes batches with:  if dataset != "kuairand_pure".
  The context dataset is named kuairand_pure_ctx, so it FAILS that test and
  inherits the big-catalogue path: train_batch_size 512, eval_batch_size 256.
  local_run.py has no such branch and takes base.yaml's 2048.
  Consequence: every Pure-tier _ctx run on Modal has trained at 512 while the
  local runs trained at 2048. The result JSONs record model/dataset/seed/metrics
  and NO hyperparameters, so this is invisible on disk -- which is also the
  origin of the "ff_* configs inferred as tuned from logs (train_batch_size=512)
  -- VERIFY" note in 07_evaluation.tex. Those configs may never have been tuned;
  512 may simply be this branch.
  ACTION TAKEN: run (2) deliberately does NOT override batch size, so the new
  rank dumps sit on the identical code path as noprof_ms / ff_* / tuned and pair
  user-for-user with them.

REPRODUCIBILITY CHECK (GPU vs GPU, seed 42, paired over 22,912 users)
  gpu_noprof vs noprof_ms:  dNDCG = -0.00059,  p = 0.275  -> NOT significant.
  The harness reproduces itself on the same code path. An earlier reading of a
  CPU-vs-Modal gap as a HARDWARE effect was wrong; it was the batch-size branch.

RESULTS — PURE _ctx, Modal A10G, mode:full, seq_len 20, seeds 42/77/123
    Condition              HR@10             NDCG@10           MRR@10
    no_profiler (ctrl)     0.0830 +- 0.0006  0.0421 +- 0.0004  0.0298 +- 0.0005
    np_vector_gate         0.0796 +- 0.0024  0.0405 +- 0.0011  0.0288 +- 0.0006
    np_shuffled_ctx        0.0815 +- 0.0012  0.0416 +- 0.0012  0.0297 +- 0.0012

  Paired per-user Wilcoxon (two-sided) vs no_profiler, per seed, n = 22,912:
    np_vector_gate   s42  dNDCG -0.00093  dHR -0.00087  p = 0.157
                     s77  dNDCG -0.00251  dHR -0.00646  p = 0.000191
                     s123 dNDCG -0.00117  dHR -0.00288  p = 0.0315
                     -> lower in 3/3 seeds, significant in 2/3
    np_shuffled_ctx  s42  dNDCG -0.00019  dHR -0.00039  p = 0.754
                     s77  dNDCG -0.00169  dHR -0.00362  p = 0.000697
                     s123 dNDCG +0.00058  dHR -0.00061  p = 0.168
                     -> lower in 2/3 seeds, significant in 1/3

  READING:
  * THE HEADLINE IS np_shuffled_ctx. Permuting x_ctx destroys the gate's input
    signal completely, and costs 1.8% HR@10 / 1.2% NDCG@10 -- indistinguishable
    from the control in two of three seeds, with a POSITIVE NDCG delta at seed
    123. A gate whose input can be randomised for ~1.8% is not demonstrably
    reading context.
  * Reshaping the gate costs MORE than removing its information content
    (vector_gate -4.1% HR vs shuffled_ctx -1.8%). That ordering is the signature
    of a gate earning its keep through PARAMETERS rather than through context.
  * This does not formally refute H3, which was tested against static_gate and
    concat -- different comparisons. But shuffled_ctx is the more direct probe of
    the same mechanism and it does not deliver an effect of the size H3's
    p = 6.3e-6 implies. Reconciling the two is now the open question for RQ3.
  * np_vector_gate is a clean supporting result for the scalar gate design:
    lower on every metric in 3/3 seeds.

  CORRECTIONS to the single-seed reading taken from run (1) on the morning of
  09-16, before the control seeds existed:
  * "CPU does not reproduce Modal" -- withdrawn. Batch-size branch, not hardware.
    Do NOT add a hardware paragraph to Threats to Validity.
  * "vector_gate is a low-variance effect (sd 0.00035)" -- the CPU path's seed sd
    was 0.00035; on GPU it is 0.0024, ~7x larger. Direction holds, tightness does
    not.
  * "shuffled_ctx costs 3.3% HR" -- it costs 1.8%, and is mostly non-significant.
    The concern is STRONGER than first reported, not weaker.
  * Qualitative ordering DID replicate across both paths:
    control > shuffled_ctx > vector_gate.

  CAVEAT: three seeds, one dataset tier, one split. Seed 42 is non-significant
  for BOTH controls, so per-seed replication counts (not a pooled p) are the
  honest summary. The control is noprof_ms (08-26 batch), not a same-batch
  control -- acceptable because it is the same condition on the same code path,
  and the seed-42 reproducibility check above supports that.

COST / THROUGHPUT NOTE
  Seven serial CPU runs = 4h11m. The same seven, parallel on A10G = 169s wall.
  modal_run.py::main still dispatches to train (8 CPU / 64 GiB) rather than
  train_pure (4 CPU / 16 GiB) despite the RON-31 cost fix, so every Pure-tier
  run launched through main since 09-10 has paid for oversized reservations.
  ::rerun uses train_pure and caps each job at 45 min via with_options().

ARTIFACTS
  results/modal/CAFREC_kuairand_pure_ctx_gpu_noprof_seed42_*.json
  results/modal/CAFREC_kuairand_pure_ctx_gpu_vector_gate_seed{42,77,123}_*.json
  results/modal/CAFREC_kuairand_pure_ctx_gpu_shuffled_ctx_seed{42,77,123}_*.json
      -- all with per-user rank + top-k dumps; pair against the existing corpus.
  results/local/*_np_{vector_gate,shuffled_ctx}_seed{42,77,123}.json
      -- run (1), CPU, batch 2048. Internally consistent; NOT comparable to
         results/modal/. Retained only as the CPU/GPU offset measurement.
  cafrec_harness/compare_gpu_rerun.py   reproduces every number in RESULTS
                                        above (pairs rank dumps by user id)
  modal_run.py::rerun + RERUN_JOBS      the parallel GPU re-run entry point
  local_run.py "controls" queue + QUEUE NOTES

NEXT
  1. Resolve H3 vs np_shuffled_ctx. Either the gate is not context-dependent, or
     shuffled_ctx is a weaker manipulation than it appears (check whether the
     permutation is re-drawn per epoch, and whether the gate saturates). This
     gates the RQ3 write-up.
  2. Persist the RESOLVED config (train_batch_size, lr, hidden_size, epochs) into
     every result JSON. One runner change; permanently settles the ff_* VERIFY.
  3. Fix the dataset != "kuairand_pure" test to match the _ctx variants, or key
     the batch sizing off catalogue size rather than dataset name.
  4. Re-check whether the ff_* / tuned configs were genuinely tuned, once (2) lands.
  5. Point modal_run.py::main at train_pure for Pure-tier work.
  6. Retrospective log entry for the 08-26 and 09-10 batches (see LOG GAP).
------------------------------------------------------------

------------------------------------------------------------
Cycle 11 (cont.) — MATCHED-BATCH DIVERSITY: the coverage advantage was the batch size
Date   : 2026-09-16
Block  : 3 — Evaluation
Author : R. Francis
------------------------------------------------------------

TRIGGER: abstract TODO(compute) "add ILD result to the coverage sentence".
The coverage sentence (CAFREC ~+22% catalogue coverage vs SASRec) rested on
analysis_pure_diversity.md, whose SASRec arm trained at batch 2048 and CAFREC
arms at 512 -- the same dataset-name batch-size branch that invalidated the H1
accuracy headline (Cycle 10 GATE CONTROLS entry). No same-batch multi-seed
top-k dump existed for either arm (the 08-26 ms corpus has ranks only).

RAN (Modal A10G, train_pure, NOT local CPU): 2 models {SASRec, CAFREC-NP} x
2 train_batch_size {512, 2048} x 7 seeds (42/77/123/256/512/1024/2048) = 28
legs, 0 failures, dump_ranks + dump_topk, eval_batch_size pinned to 4096 and
train_batch_size passed explicitly. 1 metering leg (509 s) then 27 in parallel
(776 s wall). ~$5-9 of the $20.22 ceiling (client wall summed 4.6 h incl.
queueing; check dashboard). run_diversity_matched.py -> diversity_matched.csv;
results/diversity_matched.py -> analysis_diversity_matched.md.
REPRODUCIBILITY: SASRec@2048 and NP@512 reproduce the 08-26 ms and 09-16
gpu_noprof numbers seed-for-seed (SASRec s42 HR .0818/NDCG .0411; NP s42
.0820/.0419), so these pair with the existing corpus.

RESULT (7-seed means):
                    ILD@10   Cov@10   HR@10   NDCG@10
  SASRec    @512    0.7914   0.2277   0.0810  0.0411
  CAFREC-NP @512    0.7894   0.2268   0.0811  0.0413
  SASRec    @2048   0.7892   0.1944   0.0786  0.0397
  CAFREC-NP @2048   0.7937   0.1941   0.0788  0.0399
  * MATCHED batch, NP - SASRec: coverage -0.4% (3/7, p=1.0) at 512 and -0.1%
    (4/7, p=0.69) at 2048. ILD -0.2% (p=0.94) / +0.6% (6/7, p=0.11). HR/NDCG
    +0.1..0.7%, 3-4/7, p>0.4. EVERYTHING TIED.
  * WITHIN model, 2048 -> 512: coverage +14.5% for BOTH models (t p=0.015),
    HR +2.9% / NDCG +3.3% (Wilcoxon p=0.016, 7/7).
  * The old confounded pairing (NP@512 vs SASRec@2048) reproduces exactly:
    Cov +16.7% 7/7 p=0.016, HR +3.1% 7/7, NDCG +4.1% 7/7.
  => The coverage advantage was ENTIRELY the batch size. ILD is tied (~0.79).

THESIS CONSEQUENCES:
  * Abstract coverage sentence: REWRITE. Replacement text in
    analysis_diversity_matched.md ("Sentence for the abstract").
  * analysis_pure_diversity.md's ablation rows (concat .186, static_gate .201,
    noprof .234, all @512) remain valid INTERNAL comparisons, but the reference
    changes: SASRec@512 is ~.228, so gated fusion does not ADD coverage over
    SASRec; concat/static_gate LOSE coverage that both SASRec and the gated
    model retain. Rewrite that paragraph wherever cited (07_evaluation).
  * Unaffected: H2 (profile null), H3 fusion ablations, cohort analysis --
    all same-batch internal comparisons, as the 09-16 banner already states.
  * Batch 512 is the better operating point for BOTH models on Pure; if one
    defaults table is reported, put both arms at 512 and say so.
  * Cycle 11 tuned confirm (+3.79%) ALSO ran SASRec on kuairand_pure and
    CAFREC on _ctx through the same branch, and batch size was not a grid
    factor -> the tuned headline needs the same matched-batch treatment
    before it is cited.

ARTIFACTS
  results/modal/{SASRec_kuairand_pure,CAFREC_kuairand_pure_ctx}_div_b{512,2048}_seed*_20260916-*.json
  cafrec_harness/run_diversity_matched.py, results/diversity_matched.{py,csv}
  results/analysis_diversity_matched.md
------------------------------------------------------------

------------------------------------------------------------
Cycle 11 (cont.) — MATCHED-BATCH RE-TEST OF EVERY CLAIM: tuned win survives, defaults win is batch-512-only
Date   : 2026-09-16
Block  : 3 — Evaluation
Author : R. Francis
------------------------------------------------------------

CODE FIRST (free): runner.py now persists the RESOLVED config (train/eval
batch, lr, wd, dropout, layers, ablation, history_gate_mode, profile path...)
in every result JSON, so the 512-vs-2048 confound can never be invisible on
disk again. modal_run.py's batch branch is keyed on the "kuairand_pure" name
PREFIX (was `!= "kuairand_pure"`, which _ctx failed). rerun / run_tuned_confirm
/ run_fusion_form / run_gridsearch now pass their HISTORICAL batch sizes
explicitly so re-running them reproduces what they did.

RAN (Modal A10G, train_pure, NOT local CPU): run_matched_queue.py, 44 legs,
0 failures (3 canary, then 41 in parallel, 2399 s wall). ~$0.30/leg -> ~$13;
confirm on dashboard. With the morning's 28 (run_diversity_matched.py) that
is one same-day corpus, batch explicit everywhere:
  A  SASRec @512 seeds 2020/2021/403092 (+topk)           -> SASRec@512 x10
  B  SASRec TUNED @512 x10                                 -> matched tuned pair
  C  logfull TUNED @2048 x10                               -> reverse arm
  D  logfull @512 x7 (+topk), logfull @2048 x10 (+topk)    -> diversity + 2x2
  E  np_shuffled_ctx @512 seeds 256/512/1024/2048 (+topk)  -> 7-seed control
results/matched_queue.py -> analysis_matched_queue.md (full tables + abstract
sentences).

RESULTS (NDCG@10, across-seed paired, n=10 unless noted):
  DEFAULTS logfull - SASRec:
    @512   +4.05%  9/10  Wilcoxon p=0.0039   per-user: higher 6/10, lower 0/10
    @2048  +0.55%  7/10  p=0.63 (NULL)       per-user: higher 2/10, lower 1/10
    confounded (512 vs 2048): +6.68% 10/10 p=0.002  <- the Cycle 10 headline
    batch 2048-512: SASRec -2.5% (p=0.004), logfull -5.7% (p=0.002)
    => SURVIVES AT 512 ONLY. Half the published margin was batch size. The
       profiler is more batch-sensitive than SASRec.
    decomposition (n=7): gate alone (NP - SASRec) +0.5..0.7%, p>0.6 at both
    batches = FREE BUT NULL; profiler on top (logfull - NP) +2.6% 6/7 p=0.028
    at 512, +0.4% p=0.58 at 2048. The 512 win is the PROFILE, not the gate.
  TUNED logfull_t - SASRec_t (RON-31 configs, aggregate only):
    @512   +2.90%  9/10  p=0.0076
    @2048  +2.04%  9/10  p=0.0039
    confounded: +3.79% 10/10 p=0.002  <- the Cycle 11 headline
    batch 2048-512: SASRec_t -0.9% (n.s.), logfull_t -1.7% (p=0.049)
    => SURVIVES AT BOTH BATCH SIZES. Margin +2.0..2.9%, not +3.79%. Tuning
       (dropout 0.2) removes most batch sensitivity. THIS IS THE CLAIM TO LEAD
       WITH: equal search, 10 seeds, matched batch, significant both ways.
  DIVERSITY logfull vs SASRec (n=7, matched):
    ILD@10  +1.5% 6/7 p=0.047 (@512), +1.1% 7/7 p=0.016 (@2048)
    Cov@10  -3.8% 1/7 p=0.11  (@512), -2.9% 1/7 p=0.031 (@2048)
    => small ILD-for-coverage trade. ILD, previously "non-discriminative", is
       now the diversity metric that separates the profiler from SASRec, in
       the MORE-diverse direction. Neither "+22% coverage" nor "equal coverage"
       is right for logfull (the morning's "equal" holds for NP only).
  GATE CONTROL np_shuffled_ctx vs NP @512 (n=7):
    -1.5% 2/7 p=0.16; per-user lower 3/7, higher 0/7; diversity identical.
    => 7 seeds do not change the 3-seed verdict: destroying the gate input
       costs ~1.5%, directionally consistent, not significant across seeds.
       RQ3: the gate is cheap, harmless, reads context weakly, and is not
       where the accuracy comes from.

THESIS CONSEQUENCES (supersede the morning's list where they overlap):
  * Abstract/H1: lead with the TUNED matched-batch result (+2.0..2.9%, 9/10,
    p<0.01 at both batch sizes). Report defaults as +4.0% at 512 with the
    batch dependence stated. Drop every 10/10 and every +6.4/+6.7/+3.79%.
  * Attribute the gain to the history-gated PROFILE; state the context gate's
    isolated effect as null (RQ3 honest negative, shuffled_ctx supports it).
  * Coverage sentence -> ILD-for-coverage trade sentence (text in
    analysis_matched_queue.md). analysis_pure_diversity.md's SASRec-vs-CAFREC
    coverage paragraph is withdrawn; its within-CAFREC ablation rows stand.
  * The 09-16 abstract banner's "gate effect +0.2..0.7%, not significant" is
    confirmed here (NP - SASRec), but that banner framed the thesis around
    the GATE; the surviving result is the PROFILE. Reframe accordingly.
  * Operating point: batch 512 for every model in every reported table.
  * H2 (profile null) was tested on ctx-vs-ctx (same batch) and is unaffected
    as a cohort result, but note logfull - NP is now significant at 512.

ARTIFACTS
  cafrec_harness/run_matched_queue.py, results/matched_queue.{py,csv}
  results/analysis_matched_queue.md
  results/modal/*_div_b{512,2048}_*, *_tuned_SASRec_b512_*,
  *_tuned_logfull_b2048_*, *_gpu_shuffled_ctx_seed{256,512,1024,2048}_*
  cafrec/runner.py (config persisted), modal_run.py (prefix fix),
  run_tuned_confirm.py / run_fusion_form.py / run_gridsearch.py (explicit batch)
------------------------------------------------------------

------------------------------------------------------------
Cycle 12 — CONTENT-BEARING PROFILES, ILD COMPUTED, REGISTERED-WORDING PAPER PASS
Date   : 2026-09-17 (18:10-23:00 local; Stage 1 queue continues overnight)
Author : R. Francis
Machine: labBL6O0K (lab VM, 4-core Xeon 8272CL, no CUDA), conda env `recbole`
------------------------------------------------------------

Two jobs this cycle: (A) compute intra-list diversity, unmeasured since the plan
registered it; (B) build content-bearing long-term profiles and test them against
a CAFREC-NP reference trained on the same device. Modal is at its $100 limit until
2026-10-01, so every run here is CPU. Per author decision (2026-09-17), the 7-seed
extension waits for that reset rather than spending ~19 h of CPU.

DATA AUDIT  audit_content_data.py -> results/content_data_audit_2026-09-17.{txt,json}
  Two new files arrived: data/kuairand_video_categories.csv (3.69 GB) and
  kuairand_video_captions.csv (3.22 GB). Both key on `final_video_id` over
  32,038,725 videos (the 27K id space); KuaiRand-Pure ids are the same ids.
  Coverage of the 7,210 items in kuairand_pure.inter: 99.83% have a first-level
  category (38 categories), 85.46% a second-level (154), 98.90% a non-empty
  caption. Captions are Chinese (99.9% contain CJK; median 36 chars).

  ALIGNMENT (matching ids do not prove matching videos):
    * caption-file `duration` == video_features_basic_pure.video_duration for
      100.00% of the 6,918 videos with both; 0.00% under an id+1 shift; the raw
      Pure logs' duration_ms agrees 100% too.
    * co-consumption lift, consecutive same-user clicks <=30 min, distinct items
      (201,597 pairs): first level 1.89 vs permuted null 0.985 +- 0.018 (z~51);
      second level 2.33 vs 0.965 +- 0.026; id+1 shift 0.99 == null.
    * KuaiRand `tag` (which DOES exist on this machine, contrary to the brief)
      agrees with the first-level category for 77.87% of videos vs 7.06% permuted,
      and gives a weaker lift (1.58), so the categories file is the better source.
  VERDICT: aligned. 33 Pure caption rows have caption text split across the
  caption/show_cover_text/duration fields in the file itself; the builder rejoins
  the non-numeric text fields in file order.

JOB A — ILD  analyze_ild.py -> results/ild_2026-09-17.{txt,json},
             results/ild_all_runs_2026-09-17.csv (174 Pure runs with top-k dumps)
  Primary: one-hot first-level category. Sensitivity: second level and multi-hot
  `tag`. Vectorised ILD verified equal to cafrec.eval.diversity.intra_list_diversity
  (0.809466 on SASRec bs512 seed 42). At first level <=0.03% of recommended slots
  hold a video with no category, so the zero-row convention is immaterial.

  ILD@10 (mean +- sd over seeds)        Coverage@10      HR@10
    SASRec @512   (7)  0.8207 +- 0.0097   0.228          0.0810
    CAFREC-NP     (3)  0.8092 +- 0.0035   0.234          0.0799
    CAFREC        (3)  0.8132 +- 0.0076   0.229          0.0791
    Static gate   (3)  0.8175 +- 0.0026   0.201          0.0764
    Concatenation (3)  0.8161 +- 0.0085   0.186          0.0722
    CAFREC-H      (3)  0.8254 +- 0.0011   0.201          0.0829
    HGN (CE)      (7)  0.8193 +- 0.0024   0.422          0.0717
    HGRU4Rec (CE) (7)  0.8504 +- 0.0039   0.163          0.0668
    HGN (BPR)     (7)  0.8087 +- 0.0719   0.011          0.0235
    HGRU4Rec(BPR) (7)  0.8776 +- 0.0131   0.057          0.0554
    MostPop       (1)  0.9581             0.004          0.0357

  H4 AS REGISTERED (higher ILD *and* coverage vs static fusion, largest gains on
  high-drift sessions; 3 ablation seeds, equal-size samples of 3,521 per stratum):
    * Coverage: context gate > static gate and > concatenation, overall and in
      both drift strata (1,654 items vs 1,447 and 1,340).
    * ILD overall: context gate BELOW both (-0.53% vs static, -0.36% vs concat;
      same sign in only 1-2 of 3 seeds).
    * ILD high drift: ABOVE both, 3/3 seeds (+0.82% vs static, CI [+0.0046,
      +0.0090]; +1.52% vs concat, CI [+0.0096, +0.0152]); below both on low drift.
    * high-minus-low margin +0.0118 and +0.0172, both CIs exclude zero. Second
      level and `tag` reproduce the pattern.
    VERDICT: H4 PARTLY SUPPORTED. Coverage as predicted; ILD only in the stratum
    H4 named. Caveat: seed sd (0.001-0.011) is the size of the model differences.
  NOTE: ILD and coverage rank models differently. MostPop recommends 28 distinct
  videos and has the highest ILD measured, because a few popular videos still span
  categories. Report both; neither substitutes for the other.

JOB B — CONTENT PROFILES  build_content_profiles.py -> data/profiles/*.pt + .json
  Beyond-window = after dropping each user's valid/test rows, also drop the 20
  rows before them (the rows SASRec attends over at test time); <=22 rows -> zero.
    cat_beyond  d=38    L1 category histogram, beyond-window   11,123 users (48.5%)
    cap_beyond  d=1024  L2-normalised mean caption embedding   11,118 users (48.5%)
    cap_all     d=1024  same over ALL training rows            22,912 users (100%)
  Encoder BAAI/bge-large-zh-v1.5 (Chinese captions), 7,131 captions embedded once
  in 536 s on CPU. By cohort, the beyond-window profiles are non-zero for 88.5% of
  dense users (the 11.5% gap is the 1,444 dense users with 20-22 rows) and for NO
  sparse or Q1/Q2 user, which is why cap_all exists: it is the only one that can
  test H2 in the lowest quartile.

  VALIDATION (all passed before training):
    * RecBole internal user id == order of first appearance in the .inter file for
      22,912/22,912 users.
    * 320 users recomputed from the raw file in plain Python: 0 mismatches; all 451
      users with exactly 22 rows get a zero cat_beyond vector.
    * RecBole stores timestamps as float32 (~65 s resolution at KuaiRand epochs), so
      near-simultaneous rows tie and fall back to file order. The file is already
      time-ordered within each user, and RecBole's order matches the stable sort for
      every user (0 differences in valid/test item, training sequence, beyond set),
      so no profile can contain a held-out row.
    * Smoke run (1 epoch, cap_beyond, seed 2020): profile loads at d=1024 and the
      saved checkpoint's user_profile.weight is bit-identical to the cache, i.e. it
      stayed frozen; JSON records batch 512 and top-10 lists for all 22,912 users.

STAGE 1 RUNS  local_run.py --queue content_profiles -> results/local/ (CPU, batch 512)
  12 jobs = {no_profiler, cat_beyond, cap_beyond, cap_all} x seeds {2020, 2021,
  403092}. ~41 min/run. 4/12 done at 22:25; the rest finish ~04:00 (detached
  process, survives session end).

  PRELIMINARY, SEED 2020 ONLY -- do not quote:
    np512        HR@10 0.0806  NDCG@10 0.0406
    catbeyond512       0.0782          0.0397
    capbeyond512       0.0792          0.0407
    capall512          0.0613          0.0291   <- large drop, stopped early
  Local CAFREC-NP (0.0806) vs the Modal ablation-seed reference (0.0799) is the
  expected CPU/GPU gap; that is why these runs pair only with each other.

PAPER  JournalPaper/ (untracked; pre-edit copy at JournalPaper_pre-edit_2026-09-17/)
  MiKTeX installed on this machine (winget, user scope). Build: 19 pages, no
  undefined references except sec:res:content (the pending profile section), no
  overfull boxes. Baseline before edits was 16 pages.
  * H1-H4 now stated and tested AS REGISTERED in Project_Plan_2. This changes two
    outcomes: H3 -> partly supported (wins every stratum vs concatenation with the
    predicted drift ordering; vs the static gate only in aggregate after Holm) and
    H4 -> partly supported. H4 no longer "fails because accuracy did not improve";
    the registered wording has no accuracy condition.
  * New: Deviations From the Registered Plan (protocol, seeds, tuning, hypothesis
    wording, profile, category data, unexecuted analyses); session-position table
    for H1; H3 per-stratum table; ILD column + MostPop row in tab:res:coverage;
    H4 coverage/ILD-by-drift table; content-profile description in Section III;
    three ablation rows marked with the post-hoc dagger.
  * Whole-paper style pass: 96 American spellings -> British/Irish, we/our removed
    except in two paragraphs pending Stage 1, em-dashes cut to the one-per-section
    rule in every finished section.

COST
  No Modal spend (limit until 2026-10-01). ~5.5 h CPU so far tonight, ~6 h more
  queued overnight. Disk: 15 GB free after MiKTeX + the bge encoder download.

ARTIFACTS
  cafrec_harness/audit_content_data.py, analyze_ild.py, analyze_content_profiles.py,
  build_content_profiles.py, local_run.py (llm_profile_path/profile_dim + queues
  content_profiles, content_smoke)
  results/content_data_audit_2026-09-17.{txt,json}, results/ild_2026-09-17.{txt,json},
  results/ild_all_runs_2026-09-17.csv, results/local/*512_seed*.json (4 of 12 so far)
  data/profiles/kuairand_pure_ctx.profiles.{cat_beyond.d38,cap_beyond.d1024,
  cap_all.d1024}.pt + sidecar .json; data/content_pure/ (Pure subsets, caption
  embeddings, per-user row counts)

NEXT — PLAN TO THE 2026-10-19 DEADLINE
  Week of 09-18 (no new compute; ~1-2 h)
    1. Stage 1 finishes ~04:00. Run analyze_content_profiles.py: dense cohort first
       (Holm over 3 profiles x HR/NDCG), then Q1 NDCG (cap_all vs CAFREC-NP, and the
       cross-device stand-in comparison, indicative only), all users, session
       position, coverage and ILD.
    2. Paper Phase 2: new Section V profile-results subsection (resolves the
       sec:res:content reference), H2 row and profile paragraphs in the discussion,
       introduction outcome sentence and contribution bullet, abstract (<=250 words)
       and conclusion. Rebuild; report page count.
    3. AUTHOR ITEMS (blocking, small): provenance/citation for the two category and
       caption CSVs (04_experimental_setup.tex TODO(cite)); replace or drop
       wang2023continual; verify the CA-GGNN characterisation; harness repository
       URL; decide on supervisor co-authorship.
  Week of 09-22
    4. Optional: fold results/gate_export_20260917.txt into Section V-D, or state
       why it is not reported (the paper currently says attribution is not reported).
    5. Full proofread against the writing rules; check every number against a file.
  2026-10-01 (Modal credits reset)
    6. Upload the three profile caches to the cafrec-data volume; run Stage 2:
       4 conditions x 7 headline seeds = 28 GPU runs (~$15-20 at ~$0.60/job).
    7. Same batch: SASRec @512 reruns (local_run.py --queue sasrec512, 9 runs) to
       remove the last batch-size mismatch (tab:res:seqlen's SASRec column at 2048)
       and add its coverage/ILD row. Label the device in the caption.
  Week of 10-05
    8. Fold Stage 2 into the profile tables; move the content-profile claims from
       3 seeds to 7; update H2/H4 rows, abstract and conclusion; rerun analyze_ild
       for the new runs; rebuild.
  Week of 10-12
    9. Final proof, page count, figure and table placement, reference check.
   10. Buffer. Submission 2026-10-19.
  NOT PLANNED unless the author asks: no_policy_flag ablation, tuned comparison,
  fresh temporal split for CAFREC-H, KuaiRand-1K k-core replication, fusion-form
  block. Each is a paper TODO with its reason recorded.
------------------------------------------------------------
