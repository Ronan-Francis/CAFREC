"""LLM temporal-profile embedding cache: format + validation (RON-25).

The CAFREC long-term path loads a frozen per-user profile tensor of shape
[n_users, profile_dim] (see CAFREC._maybe_load_profiles). RON-24 will produce
those vectors by running an LLM profiler over every user's history; RON-25 (this
module) locks down the *contract* that cache must satisfy and unit-tests the data
flow from disk into the model — independently of, and before, any profiler run.

What this guarantees
--------------------
  * shape is exactly (n_users, profile_dim), row i == user id i (RecBole remaps
    user ids to a contiguous 0..n_users-1 range, so the cache is row-aligned to
    that remapping — build it from `dataset.field2id_token['user_id']`),
  * dtype float32, no NaN/Inf,
  * null / sparse users (no history to profile) are representable as an all-zero
    row rather than a missing row — the model still gets a valid, if
    uninformative, z_long for them,
  * the tensor loads with torch.load and drops straight into CAFREC.

`build_placeholder_profiles` produces a deterministic synthetic cache so the
whole pipeline (and the gating fusion) can be exercised before the real RON-24
profiler exists.
"""
from __future__ import annotations

import numpy as np
import torch

PROFILE_DTYPE = torch.float32


def validate_profiles(profiles, n_users, profile_dim):
    """Assert `profiles` meets the CAFREC frozen-profile contract.

    Raises ValueError with a specific message on the first violation; returns
    the tensor (as float32) unchanged on success.
    """
    if not torch.is_tensor(profiles):
        raise ValueError(f"profiles must be a torch.Tensor, got {type(profiles)!r}")
    if profiles.dim() != 2:
        raise ValueError(f"profiles must be 2-D [n_users, profile_dim], got {tuple(profiles.shape)}")
    if tuple(profiles.shape) != (n_users, profile_dim):
        raise ValueError(
            f"profile tensor {tuple(profiles.shape)} != expected ({n_users}, {profile_dim})"
        )
    p = profiles.to(PROFILE_DTYPE)
    if not torch.isfinite(p).all():
        n_bad = int((~torch.isfinite(p)).sum())
        raise ValueError(f"profiles contain {n_bad} non-finite value(s) (NaN/Inf)")
    return p


def sparse_user_rows(profiles, atol=0.0):
    """Indices of all-zero rows — the null/sparse users with no profile signal."""
    p = profiles.to(PROFILE_DTYPE)
    return torch.nonzero((p.abs() <= atol).all(dim=1), as_tuple=False).flatten()


def save_profiles(path, profiles, n_users=None, profile_dim=None):
    """Validate (if dims given) and persist a profile cache with torch.save."""
    if n_users is not None and profile_dim is not None:
        profiles = validate_profiles(profiles, n_users, profile_dim)
    else:
        profiles = profiles.to(PROFILE_DTYPE)
    torch.save(profiles, path)
    return path


def load_profiles(path, n_users, profile_dim, fill_missing=True):
    """Load and validate a profile cache for a model with `n_users`/`profile_dim`.

    If `fill_missing` and the cache has FEWER rows than n_users (users appended
    since it was built), the missing tail is zero-filled (treated as sparse
    users) rather than raising — a convenience for incremental profiling. Extra
    rows are always an error (ambiguous alignment).
    """
    profiles = torch.load(path, weights_only=False)
    if not torch.is_tensor(profiles):
        profiles = torch.as_tensor(profiles)
    profiles = profiles.to(PROFILE_DTYPE)

    if profiles.dim() != 2 or profiles.shape[1] != profile_dim:
        raise ValueError(
            f"cached profiles {tuple(profiles.shape)} incompatible with profile_dim={profile_dim}"
        )
    if profiles.shape[0] > n_users:
        raise ValueError(
            f"cache has {profiles.shape[0]} rows > n_users={n_users}; cannot align"
        )
    if profiles.shape[0] < n_users:
        if not fill_missing:
            raise ValueError(
                f"cache has {profiles.shape[0]} rows < n_users={n_users} and fill_missing=False"
            )
        pad = torch.zeros(n_users - profiles.shape[0], profile_dim, dtype=PROFILE_DTYPE)
        profiles = torch.cat([profiles, pad], dim=0)

    return validate_profiles(profiles, n_users, profile_dim)


def build_placeholder_profiles(n_users, profile_dim, seed=403092, sparse_ids=None):
    """Deterministic synthetic profile cache for pre-RON-24 pipeline testing.

    `sparse_ids` (iterable of user indices) are written as all-zero rows so the
    null/sparse-user path can be exercised. Row 0 is the RecBole padding user and
    is always zeroed.
    """
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_users, profile_dim)).astype(np.float32)
    arr[0] = 0.0  # RecBole padding user id 0
    if sparse_ids is not None:
        for i in sparse_ids:
            arr[int(i)] = 0.0
    return torch.from_numpy(arr)
