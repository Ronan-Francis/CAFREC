import numpy as np


def coverage(topk_lists, n_items):
    """topk_lists: [n_users, k] recommended item ids.
    Proportion of the catalogue that appears in any user's top-k."""
    recommended = set()
    for row in topk_lists:
        recommended.update(int(i) for i in row)
    return len(recommended) / float(n_items)


def intra_list_diversity(topk_lists, category_matrix):
    """ILD = mean over users of mean pairwise (1 - cosine) across the top-k
    items' category vectors. category_matrix: [n_items, d] (KuaiRand: one-hot
    over first-level category, or multi-hot over tags)."""
    cat = np.asarray(category_matrix, float)
    norms = np.linalg.norm(cat, axis=1, keepdims=True)
    cat_n = cat / np.clip(norms, 1e-12, None)        # unit vectors for cosine
    ilds = []
    for row in topk_lists:
        idx = np.asarray(row, int)
        v = cat_n[idx]                                # [k, d]
        sim = v @ v.T
        iu = np.triu_indices(len(idx), k=1)           # unique pairs only
        dissim = 1.0 - sim[iu]
        ilds.append(float(dissim.mean()) if dissim.size else 0.0)
    return float(np.mean(ilds))


def topk_from_scores(scores, k=10):
    """scores: [n_users, n_items] full-catalogue scores (mask seen items to
    -inf first). Returns [n_users, k] item ids. Order within top-k doesn't
    matter for ILD/Coverage."""
    scores = np.asarray(scores)
    return np.argpartition(-scores, k, axis=1)[:, :k]