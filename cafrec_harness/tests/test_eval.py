import numpy as np
from cafrec.eval.metrics import per_user_metrics, ranks_from_scores

def test_metrics_hand_computed():
    pu = per_user_metrics(np.array([1, 11, 2]), k=10)
    assert pu["HR@10"].tolist() == [1.0, 0.0, 1.0]           # rank 11 misses @10
    assert abs(pu["NDCG@10"][2] - 1/np.log2(3)) < 1e-9
    assert abs(pu["MRR"][1] - 1/11) < 1e-9

def test_ranks_from_scores():
    s = np.array([[0.9, 0.5, 0.95], [0.2, 0.1, 0.05]])        # pos = col 0
    assert ranks_from_scores(s, target_col=0).tolist() == [2, 1]