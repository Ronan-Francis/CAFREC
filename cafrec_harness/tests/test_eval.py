import numpy as np
import torch
from cafrec.eval.metrics import per_user_metrics, ranks_from_scores
from cafrec.eval.full_rank import full_sort_ranks

def test_metrics_hand_computed():
    pu = per_user_metrics(np.array([1, 11, 2]), k=10)
    assert pu["HR@10"].tolist() == [1.0, 0.0, 1.0]           # rank 11 misses @10
    assert abs(pu["NDCG@10"][2] - 1/np.log2(3)) < 1e-9
    assert abs(pu["MRR"][1] - 1/11) < 1e-9

def test_ranks_from_scores():
    s = np.array([[0.9, 0.5, 0.95], [0.2, 0.1, 0.05]])        # pos = col 0
    assert ranks_from_scores(s, target_col=0).tolist() == [2, 1]


class _FakeInter:
    """Minimal stand-in for a RecBole Interaction batch."""
    def __init__(self, uids):
        self._uids = torch.tensor(uids)
    def to(self, _device):
        return self
    def __getitem__(self, _field):
        return self._uids


class _FakeModel:
    """full_sort_predict just replays a fixed score matrix."""
    def __init__(self, scores):
        self._scores = scores
    def eval(self):
        pass
    def full_sort_predict(self, _interaction):
        return self._scores.clone()          # clone: full_sort_ranks masks in place


def test_full_sort_ranks_topk_selection():
    # 2 users, 5 items (item 0 = pad, masked to -inf inside).
    scores = torch.tensor([[0.0, 3.0, 1.0, 2.0, 0.5],   # user0 order: 1,3,2,4
                           [0.0, 0.1, 0.2, 5.0, 0.3]])   # user1 order: 3,4,2,1
    batch = (_FakeInter([10, 20]), None,
             torch.tensor([0, 1]),          # positive_u (row per user)
             torch.tensor([1, 3]))          # positive_i (targets: item1, item3)
    uids, ranks, topk = full_sort_ranks(
        _FakeModel(scores), [batch], "cpu", tot_item_num=5,
        uid_field="uid", k=2, collect_topk=True)
    assert uids.tolist() == [10, 20]
    assert ranks.tolist() == [1, 1]                       # both targets top-scored
    assert set(topk[0].tolist()) == {1, 3}                # excludes pad(0)
    assert set(topk[1].tolist()) == {3, 4}
    assert topk.shape == (2, 2)


def test_full_sort_ranks_without_topk_is_two_tuple():
    scores = torch.tensor([[0.0, 3.0, 1.0]])
    batch = (_FakeInter([7]), None, torch.tensor([0]), torch.tensor([1]))
    out = full_sort_ranks(_FakeModel(scores), [batch], "cpu",
                          tot_item_num=3, uid_field="uid")
    assert len(out) == 2                                  # backward-compatible