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
