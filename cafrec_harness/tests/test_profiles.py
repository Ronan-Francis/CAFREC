"""Tests for the LLM profile-embedding format + loader (RON-25).

These lock the frozen-profile contract CAFREC's long-term path depends on, and
prove the cache flows from disk into the model — before any real profiler
(RON-24) exists, using a deterministic synthetic cache.
"""
import numpy as np
import pytest
import torch

from cafrec.features.profiles import (
    build_placeholder_profiles,
    load_profiles,
    save_profiles,
    sparse_user_rows,
    validate_profiles,
)

N_USERS, DIM = 12, 8


def test_validate_accepts_well_formed():
    p = build_placeholder_profiles(N_USERS, DIM)
    out = validate_profiles(p, N_USERS, DIM)
    assert out.shape == (N_USERS, DIM) and out.dtype == torch.float32


def test_validate_rejects_wrong_shape():
    p = torch.zeros(N_USERS, DIM + 1)
    with pytest.raises(ValueError, match="!="):
        validate_profiles(p, N_USERS, DIM)


def test_validate_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        validate_profiles(torch.zeros(N_USERS), N_USERS, DIM)


def test_validate_rejects_nan_inf():
    p = build_placeholder_profiles(N_USERS, DIM).clone()
    p[3, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_profiles(p, N_USERS, DIM)


def test_sparse_rows_detected():
    p = build_placeholder_profiles(N_USERS, DIM, sparse_ids=[4, 7])
    sparse = set(sparse_user_rows(p).tolist())
    assert {0, 4, 7} <= sparse            # 0 is the padding user, plus the two we zeroed


def test_save_load_roundtrip(tmp_path):
    p = build_placeholder_profiles(N_USERS, DIM)
    path = tmp_path / "profiles.pt"
    save_profiles(path, p, N_USERS, DIM)
    loaded = load_profiles(path, N_USERS, DIM)
    assert torch.equal(loaded, p)


def test_load_zero_fills_missing_tail(tmp_path):
    # cache built for 10 users, model now has 12 -> tail zero-filled as sparse
    p = build_placeholder_profiles(10, DIM)
    path = tmp_path / "p.pt"
    save_profiles(path, p)
    loaded = load_profiles(path, N_USERS, DIM, fill_missing=True)
    assert loaded.shape == (N_USERS, DIM)
    assert torch.count_nonzero(loaded[10:]) == 0


def test_load_rejects_extra_rows(tmp_path):
    p = build_placeholder_profiles(N_USERS + 3, DIM)
    path = tmp_path / "p.pt"
    save_profiles(path, p)
    with pytest.raises(ValueError, match="cannot align"):
        load_profiles(path, N_USERS, DIM)


def test_end_to_end_into_cafrec(tmp_path):
    """The cache loads into CAFREC, is frozen, and drives the long-term path."""
    from cafrec.runner import run_experiment  # noqa: F401  (ensures torch.load patch)
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from cafrec.registry import get_spec

    spec = get_spec("CAFREC")
    dim = spec.config["profile_dim"]

    # Build the dataset first so we know n_users, then a matching profile cache.
    cfg = {**spec.config, **spec.contract, "epochs": 1}
    config = Config(model=spec.model, dataset="ml-100k",
                    config_file_list=["configs/base.yaml"], config_dict=cfg)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    n_users = train_data.dataset.user_num

    profiles = build_placeholder_profiles(n_users, dim, sparse_ids=[1])
    path = tmp_path / "profiles.pt"
    save_profiles(path, profiles, n_users, dim)

    # Rebuild the model WITH the profile path set.
    cfg2 = {**spec.config, **spec.contract, "epochs": 1, "llm_profile_path": str(path)}
    config2 = Config(model=spec.model, dataset="ml-100k",
                     config_file_list=["configs/base.yaml"], config_dict=cfg2)
    model = spec.model(config2, train_data.dataset)

    # loaded, frozen, and identical to what we saved
    assert model.user_profile.weight.requires_grad is False
    assert torch.allclose(model.user_profile.weight.data, profiles)
    # long-term path returns the projected profile for a batch of users
    users = torch.arange(n_users)
    z_long = model._long_term(users)
    assert z_long.shape == (n_users, model.hidden_size)
