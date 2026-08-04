"""Smoke tests — this is what makes it a *testing* harness, not a run script.

Each runs a single epoch on ml-100k (fast on CPU) and asserts the pipeline
returns real metrics. Add your model's key to the parametrize list once it's
registered, and it's covered automatically.
"""
import pytest

from cafrec.runner import run_experiment


@pytest.mark.parametrize("model_key", ["SASRec", "HGN", "HGRU4Rec", "CAFREC"])
def test_run_returns_metrics(model_key):
    metrics = run_experiment(
        model_key, dataset="ml-100k", config_overrides={"epochs": 1}
    )
    assert metrics["test"], f"{model_key} returned no test metrics"
    assert any(v > 0 for v in metrics["test"].values()), \
        f"{model_key} test metrics are all zero"
    # the per-model contract should be respected
    assert metrics["loss_type"] in {"CE", "BPR"}
