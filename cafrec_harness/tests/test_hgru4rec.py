"""Unit tests for the custom HGRU4Rec model (RON-10 RNN baseline).

These exercise the model in isolation (no RecBole data pipeline) via a tiny
synthetic batch, asserting the RON-30-style sanity guarantees:
  * forward + backward run and the loss DECREASES on a fixed batch,
  * the scoring heads return the right shapes.
End-to-end integration on ml-100k is covered by tests/test_smoke.py.
"""
import torch

from cafrec.models.hgru4rec import HGRU4Rec

N_ITEMS = 20
N_USERS = 5
MAX_LEN = 10


class _StubConfig(dict):
    """Minimal RecBole-style config: bracket access to the fields the model and
    SequentialRecommender base read at construction time."""


class _StubDataset:
    def __init__(self, n_users, n_items):
        self.user_num = n_users
        self.item_num = n_items
        self._n_items = n_items

    def num(self, field):        # SequentialRecommender reads dataset.num(ITEM_ID)
        return self._n_items


def _config(loss_type="BPR"):
    return _StubConfig(
        USER_ID_FIELD="user_id", ITEM_ID_FIELD="item_id",
        LIST_SUFFIX="_list", ITEM_LIST_LENGTH_FIELD="item_length",
        MAX_ITEM_LIST_LENGTH=MAX_LEN, NEG_PREFIX="neg_", device="cpu",
        hidden_size=16, num_layers=1, dropout_prob=0.0, loss_type=loss_type,
    )


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    b = 8
    return {
        "item_id_list": torch.randint(1, N_ITEMS, (b, MAX_LEN), generator=g),
        "item_length": torch.randint(1, MAX_LEN + 1, (b,), generator=g),
        "user_id": torch.randint(0, N_USERS, (b,), generator=g),
        "item_id": torch.randint(1, N_ITEMS, (b,), generator=g),
        "neg_item_id": torch.randint(1, N_ITEMS, (b,), generator=g),
    }


def _build(loss_type="BPR"):
    torch.manual_seed(0)
    return HGRU4Rec(_config(loss_type), _StubDataset(N_USERS, N_ITEMS))


def test_bpr_loss_decreases_on_fixed_batch():
    model = _build("BPR")
    batch = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = model.calculate_loss(batch).item()
    for _ in range(60):
        opt.zero_grad()
        loss = model.calculate_loss(batch)
        loss.backward()
        opt.step()
    last = model.calculate_loss(batch).item()
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_ce_loss_runs_and_backprops():
    model = _build("CE")
    loss = model.calculate_loss(_batch())
    loss.backward()
    # the item embedding must receive a gradient (the model actually trains)
    assert model.item_embedding.weight.grad is not None
    assert torch.isfinite(loss)


def test_full_sort_and_predict_shapes():
    model = _build("BPR")
    batch = _batch()
    scores = model.full_sort_predict(batch)
    assert scores.shape == (batch["user_id"].shape[0], N_ITEMS)
    point = model.predict(batch)
    assert point.shape == (batch["user_id"].shape[0],)
