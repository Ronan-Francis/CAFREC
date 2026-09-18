"""Batch-size defaults key off catalogue tier, and results record resolved config.

Guards the H1 batch-size confound (PROJECT_LOG, Cycle 10 cont.): SASRec on
kuairand_pure trained at 2048 while CAFREC on kuairand_pure_ctx trained at 512,
because batch sizing tested the dataset name.
"""
import json

import pytest

from cafrec.results import _flatten
from cafrec.runner import RESOLVED_CONFIG_KEYS, _resolved_config
from cafrec.tiers import batch_size_overrides, catalogue_tier


def test_context_variant_shares_its_base_tier_batch_sizes():
    # the exact pair behind the confound
    assert batch_size_overrides("kuairand_pure_ctx") == batch_size_overrides("kuairand_pure")


@pytest.mark.parametrize("dataset", ["kuairand_pure", "kuairand_pure_ctx"])
def test_small_catalogue_uses_base_yaml(dataset):
    assert catalogue_tier(dataset) == "small"
    assert batch_size_overrides(dataset) == {}


@pytest.mark.parametrize("dataset", ["kuairand_1k", "kuairand_1k_kcore_ctx",
                                     "kuairand_27k", "kuairand_27k_ctx"])
def test_large_catalogue_shrinks_batches(dataset):
    assert catalogue_tier(dataset) == "large"
    assert batch_size_overrides(dataset) == {"train_batch_size": 512,
                                             "eval_batch_size": 256}


@pytest.mark.parametrize("dataset", ["ml-100k", "kuairand_purex", "kuairand"])
def test_unknown_dataset_raises_instead_of_guessing(dataset):
    with pytest.raises(ValueError):
        catalogue_tier(dataset)


class _FakeConfig:
    """Mimics recbole Config.__getitem__: missing keys resolve to None."""

    def __init__(self, d):
        self.d = d

    def __getitem__(self, k):
        return self.d.get(k)


def test_resolved_config_is_complete_and_json_safe():
    cfg = _FakeConfig({"train_batch_size": 512, "learning_rate": 1e-3,
                       "eval_args": {"mode": "full"}, "ablation": object()})
    out = _resolved_config(cfg)
    assert set(out) == set(RESOLVED_CONFIG_KEYS)
    assert out["train_batch_size"] == 512
    assert out["hidden_size"] is None
    assert isinstance(out["ablation"], str)  # non-serialisable -> str
    json.dumps(out)


def test_summary_row_carries_batch_size():
    row = _flatten({"model": "SASRec", "test": {"hit@10": 0.1},
                    "config": {"train_batch_size": 2048, "eval_args": {"mode": "full"}}})
    assert row["cfg_train_batch_size"] == 2048
    assert "cfg_eval_args" not in row
