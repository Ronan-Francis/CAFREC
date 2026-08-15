"""Per-user held-out ranks under FULL ranking, for paired significance tests.

RecBole's evaluator reports only aggregate means. To run the paired Wilcoxon /
bootstrap (`cafrec.eval.significance`) between two models we need each user's
rank of their held-out test item — the paired unit. This mirrors RecBole 1.2.1's
`Trainer._full_sort_batch_eval` masking exactly (pad item and already-seen items
set to -inf), so the aggregate of these ranks reproduces the reported
HR/NDCG/MRR (asserted by the caller).
"""
from __future__ import annotations

import numpy as np
import torch


def full_sort_ranks(model, eval_data, device, tot_item_num, uid_field,
                    k=10, collect_topk=False):
    """Return (internal_user_ids, ranks) as aligned np.int64 arrays. `ranks` is
    the 1-indexed rank of each held-out user's target = 1 + #items scoring
    strictly higher than the target after masking pad(0) + history items
    (RecBole semantics). User ids let callers PAIR across models/datasets whose
    RecBole remap may differ, rather than trusting eval-order alignment.

    With `collect_topk=True` also return a [n_users, k] array of the top-k
    RECOMMENDED internal item ids (highest-scoring after the same pad+history
    masking) -> (uids, ranks, topk). ILD/Coverage need these per-user lists,
    which the rank-only path does not carry. Selection order within the k does
    not matter for diversity metrics."""
    model.eval()
    ranks_out, uid_out, topk_out = [], [], []
    with torch.no_grad():
        for batched_data in eval_data:
            interaction, history_index, positive_u, positive_i = batched_data
            scores = model.full_sort_predict(interaction.to(device))
            scores = scores.view(-1, tot_item_num)
            scores[:, 0] = -np.inf                     # padding item
            if history_index is not None:              # already-seen items
                if isinstance(history_index, (tuple, list)):
                    history_index = tuple(h.to(scores.device) for h in history_index)
                else:
                    history_index = history_index.to(scores.device)
                scores[history_index] = -np.inf
            positive_u = positive_u.to(scores.device)
            positive_i = positive_i.to(scores.device)
            tgt = scores[positive_u, positive_i]                    # [n_users]
            rows = scores[positive_u]                              # [n_users, I]
            rank = 1 + (rows > tgt.unsqueeze(1)).sum(dim=1)         # strict-greater
            uids = interaction[uid_field].to(scores.device)[positive_u]
            ranks_out.append(rank.cpu().numpy())
            uid_out.append(uids.cpu().numpy())
            if collect_topk:
                topk_idx = torch.topk(rows, k, dim=1).indices       # [n_users, k]
                topk_out.append(topk_idx.cpu().numpy())
    uids = np.concatenate(uid_out).astype(np.int64)
    ranks = np.concatenate(ranks_out).astype(np.int64)
    if collect_topk:
        return uids, ranks, np.concatenate(topk_out).astype(np.int64)
    return uids, ranks


def metrics_at_k(ranks, k=10):
    """Aggregate HR@k / NDCG@k / MRR@k from a rank array (all cut at k, matching
    RecBole's reported '@k' metrics). Used to verify ranks vs reported means."""
    ranks = np.asarray(ranks, dtype=float)
    ink = ranks <= k
    return {
        f"hit@{k}": float(ink.mean()),
        f"ndcg@{k}": float(np.where(ink, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
        f"mrr@{k}": float(np.where(ink, 1.0 / ranks, 0.0).mean()),
    }
