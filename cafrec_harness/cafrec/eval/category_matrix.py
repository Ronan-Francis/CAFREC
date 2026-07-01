"""Item-category matrix for intra-list diversity (feeds RON-15).

`cafrec.eval.diversity.intra_list_diversity` needs a `[n_items, d]` category
matrix indexed by item id. This module builds that matrix from KuaiRand's
`tag` field (`video_features_basic_pure.csv`).

KuaiRand tags are messy: some videos carry a single tag ("39"), some carry a
compound comma-joined value ("39,68"), and some are null (96 videos on
KuaiRand-Pure). PROJECT_LOG risk **R2d** flags that the multi-tag encoding
strategy must be locked before diversity is computed. This module implements
all three logged options behind one `strategy` argument so the decision is
explicit and reproducible rather than buried:

    strategy="multihot"  (default, recommended for ILD)
        A video with tags "39,68" contributes 1 to BOTH the tag-39 and tag-68
        columns. ILD then reflects a video's full category membership, which is
        the honest notion of "how different are these two items". This is the
        recommended choice; note it in the methodology chapter with the R2d
        sensitivity rationale.

    strategy="primary"
        Keep only the first tag before the comma ("39,68" -> 39). Simplest,
        one-hot, but discards genuine multi-category signal.

    strategy="label"
        Treat the whole string as one atomic category ("39,68" is its own
        category, distinct from "39"). Inflates cardinality; rarely what you
        want for diversity.

Null tags yield an all-zero row; `intra_list_diversity` already unit-normalises
with a 1e-12 floor, so zero rows contribute maximal (1.0) dissimilarity to
every other item rather than crashing.
"""
from __future__ import annotations

import numpy as np

VALID_STRATEGIES = ("multihot", "primary", "label")


def _parse_tags(raw, strategy):
    """Return the list of category tokens for one raw tag value."""
    if raw is None:
        return []
    # treat NaN / empty as null
    if isinstance(raw, float) and np.isnan(raw):
        return []
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return []
    if strategy == "label":
        return [s]
    if strategy == "primary":
        return [s.split(",")[0].strip()]
    # multihot
    return [tok.strip() for tok in s.split(",") if tok.strip() != ""]


def build_category_matrix(item_ids, tag_values, strategy="multihot", n_items=None):
    """Build an item-id-indexed multi-hot category matrix.

    Parameters
    ----------
    item_ids : array-like[int]
        Item ids, one per row of `tag_values`.
    tag_values : array-like
        Raw tag entries aligned with `item_ids` (single, compound or null).
    strategy : {"multihot", "primary", "label"}
        Multi-tag encoding (see module docstring); resolves R2d.
    n_items : int, optional
        Number of rows in the output. Defaults to max(item_id) + 1 so that the
        matrix can be indexed directly by item id (`cat[item_id]`), matching
        the diversity module's usage. Pass explicitly if the catalogue has
        higher-id items than appear in `item_ids`.

    Returns
    -------
    matrix : np.ndarray  shape (n_items, n_tags), dtype float32
        `matrix[item_id]` is the item's category vector. Items with no tag row
        or a null tag are all-zero.
    tag_to_col : dict  tag_token -> column index
        The category vocabulary, in first-seen order.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {VALID_STRATEGIES}, got '{strategy}'"
        )
    item_ids = np.asarray(item_ids)
    if item_ids.shape[0] != len(tag_values):
        raise ValueError("item_ids and tag_values must be the same length")

    if n_items is None:
        n_items = int(item_ids.max()) + 1 if item_ids.size else 0

    # First pass: build the tag vocabulary in first-seen order (stable columns).
    tag_to_col = {}
    parsed = []
    for raw in tag_values:
        toks = _parse_tags(raw, strategy)
        parsed.append(toks)
        for tok in toks:
            if tok not in tag_to_col:
                tag_to_col[tok] = len(tag_to_col)

    matrix = np.zeros((n_items, len(tag_to_col)), dtype=np.float32)
    for item_id, toks in zip(item_ids, parsed):
        iid = int(item_id)
        if iid < 0 or iid >= n_items:
            raise IndexError(
                f"item id {iid} outside [0, {n_items}); pass n_items explicitly."
            )
        for tok in toks:
            matrix[iid, tag_to_col[tok]] = 1.0
    return matrix, tag_to_col


def category_matrix_from_dataframe(df, item_col="video_id", tag_col="tag",
                                   strategy="multihot", n_items=None):
    """Convenience wrapper for a pandas DataFrame of video features.

    Deduplicates on `item_col` (KuaiRand lists one row per video) before
    building the matrix.
    """
    sub = df[[item_col, tag_col]].drop_duplicates(subset=item_col)
    return build_category_matrix(
        sub[item_col].to_numpy(),
        sub[tag_col].to_numpy(),
        strategy=strategy,
        n_items=n_items,
    )
