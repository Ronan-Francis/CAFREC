"""Context-feature computation for CAFREC's gating module (RON-16/17/18/19).

The gating MLP conditions on a per-request context vector x_ctx. The canonical
set is SIX prefix-causal, full-log features (see `context.py`):

    prefix_session_len_log_z   position in the current 30-min session
    prefix_dwell_entropy_z     Shannon entropy of the dwell-ratio histogram
    prefix_category_drift_z    rate of consecutive item-category changes
    inter_session_gap_log_z    log ms since the previous session (recency)
    prefix_policy_flag         an is_rand transition occurred in the prefix
    is_first_session           user's first session (cold-start flag)

Every feature is computed CAUSALLY over the session prefix STRICTLY BEFORE i, so
it aligns with the item sequence the model conditions on and leaks nothing about
the held-out target (RON-60). Features see the full log (organic + random rows);
only organic clicks become training/eval targets downstream (D1).
"""
from cafrec.features.context import (
    BINARY,
    CONTINUOUS,
    CTX_FIELDS,
    compute_context_features,
    dwell_ratio,
    segment_sessions,
    standardize_context,
)

__all__ = [
    "BINARY",
    "CONTINUOUS",
    "CTX_FIELDS",
    "compute_context_features",
    "dwell_ratio",
    "segment_sessions",
    "standardize_context",
]
