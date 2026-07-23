"""Context-feature computation for CAFREC's gating module (RON-16/17/18).

The gating MLP conditions on a per-request context vector x_ctx. These are the
"classic four" session-context features derived from KuaiRand:

    session_length       RON-16   number of items seen so far this session
    dwell_time_entropy   RON-17   Shannon entropy of the dwell-ratio histogram
    category_drift_rate  RON-18   rate of consecutive item-category changes
    request_source       (RON-19 -> folded into the builder as is_rand policy)

Every feature is computed CAUSALLY: the value attached to interaction i uses
only the session history STRICTLY BEFORE i, so it aligns with the item sequence
the model conditions on and introduces no leakage of the held-out target
(RON-60 split-integrity requirement).
"""
from cafrec.features.context import (
    CTX_FIELDS,
    compute_context_features,
    dwell_ratio,
    segment_sessions,
)

__all__ = [
    "CTX_FIELDS",
    "compute_context_features",
    "dwell_ratio",
    "segment_sessions",
]
