"""Build the frozen LLM temporal-profile cache CAFREC loads (RON-24).

`cafrec.features.profiles` locked the *contract* (shape `[n_users, profile_dim]`,
float32, row i == RecBole internal user id i, row 0 = padding, sparse users =
all-zero) and shipped a synthetic placeholder. THIS module produces the real
cache: for every user it renders a causal, leakage-safe natural-language
*temporal profile* from that user's TRAINING history, embeds it with an offline
model, and writes a contract-valid tensor.

Design
------
* **Row alignment.** We load the *same* RecBole dataset CAFREC trains on (same
  base config + the `_ctx` load_col promotion) purely to inherit its user/item
  id remap, then place each user's vector at its internal id. Guarantees the
  cache lines up with `CAFREC.user_profile`.
* **Leakage safety.** The profile for a user summarises only that user's
  TRAINING interactions — every row except the last two (RecBole's
  LS:valid_and_test leave-one-out holdout, time-ordered). The held-out
  valid/test targets never inform the frozen z_long. Mirrors the causal
  guarantee in `cafrec.features.context` / RON-60.
* **Offline, swappable backend.** `EmbeddingBackend.encode(list[str]) -> [n, d]`.
  `HashingBackend` is dependency-free and deterministic (a real structured
  no-LLM control, and the local plumbing proof). `HFBackend` wraps a
  sentence-transformers / HF model — the "as large as compute allows" offline
  profiler — and is the one run on Modal GPU. `profile_dim` in the CAFREC config
  must equal the backend's output dim.

CLI
---
    # local plumbing / no-LLM control (no extra deps):
    python -m cafrec.features.build_profiles --dataset kuairand_1k_ctx \
        --backend hashing --dim 256

    # real offline model (needs transformers+torch GPU; run on Modal):
    python -m cafrec.features.build_profiles --dataset kuairand_1k_ctx \
        --backend hf --model BAAI/bge-large-en-v1.5
"""
from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cafrec.features.context import CTX_FIELDS
from cafrec.features.profiles import save_profiles, validate_profiles

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"
PROFILE_ROOT = DATA_DIR / "profiles"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


# --------------------------------------------------------------------------- #
# RecBole id remap (so the cache is row-aligned to what CAFREC trains on)
# --------------------------------------------------------------------------- #
def load_user_item_remap(dataset, base_config=DEFAULT_BASE_CONFIG, data_path=None):
    """Return (uid_token2id, iid_token2id, n_users) from the SAME RecBole
    dataset CAFREC builds. Tokens are the original string ids; internal id 0 is
    RecBole's padding token in both maps. `data_path` overrides RecBole's atomic-
    file root (e.g. Modal's `/data/recbole`)."""
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.utils import init_seed

    from cafrec.registry import get_spec

    spec = get_spec("CAFREC")
    config_dict = {}
    config_dict.update(spec.config)
    config_dict.update(spec.contract)
    if data_path is not None:
        config_dict["data_path"] = str(data_path)
    # Mirror runner._ctx load_col promotion so the user/item filtering (k-core,
    # column set) matches CAFREC's training dataset exactly.
    context_load_col = config_dict.pop("context_load_col", None)
    if context_load_col is not None and str(dataset).endswith("_ctx"):
        config_dict.setdefault("load_col", context_load_col)

    config = Config(model=spec.model, dataset=dataset,
                    config_file_list=[str(base_config)], config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    rb = create_dataset(config)

    uid = rb.field2id_token[rb.uid_field]
    iid = rb.field2id_token[rb.iid_field]
    uid_token2id = {str(t): i for i, t in enumerate(uid)}
    iid_token2id = {str(t): i for i, t in enumerate(iid)}
    return uid_token2id, iid_token2id, len(uid)


# --------------------------------------------------------------------------- #
# Per-user causal profile text
# --------------------------------------------------------------------------- #
def _bucket(z):
    """Qualitative label for a z-scored mean (profile is embedded as text, so we
    verbalise magnitudes rather than feed raw floats)."""
    if z <= -0.75:
        return "low"
    if z >= 0.75:
        return "high"
    return "moderate"


def render_profiles(inter_path, uid_token2id):
    """Render one temporal-profile string per user from the `_ctx` atomic file,
    using TRAINING rows only (all but each user's last two, time-ordered).

    Returns (user_ids, texts): parallel lists over internal user ids that have
    at least one training row. Sparse/absent users are simply omitted here and
    zero-filled by the caller.
    """
    df = pd.read_csv(inter_path, sep="\t")
    # atomic-file headers carry ":token"/":float" suffixes — strip to bare names.
    df.columns = [c.split(":")[0] for c in df.columns]
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # leave-one-out train mask: drop each user's last two interactions.
    rank_desc = df.groupby("user_id").cumcount(ascending=False)
    train = df[rank_desc >= 2]

    user_ids, texts = [], []
    for uid_tok, g in train.groupby("user_id", sort=False):
        internal = uid_token2id.get(str(uid_tok))
        if internal is None:          # user filtered out of the RecBole dataset
            continue
        n = len(g)
        means = {f: float(g[f].mean()) for f in CTX_FIELDS if f in g.columns}
        session_depth = _bucket(means.get("prefix_session_len_log_z", 0.0))
        dwell = _bucket(means.get("prefix_dwell_entropy_z", 0.0))
        drift = _bucket(means.get("prefix_category_drift_z", 0.0))
        gap = _bucket(means.get("inter_session_gap_log_z", 0.0))
        policy_rate = float(g.get("prefix_policy_flag", pd.Series([0.0])).mean())
        cold = float(g.get("is_first_session", pd.Series([0.0])).mean())
        history = "sparse" if n < 10 else ("moderate" if n < 40 else "rich")

        texts.append(
            f"User temporal profile. History length: {history} ({n} logged "
            f"interactions). Typical session depth: {session_depth}. "
            f"Dwell-time entropy: {dwell} (viewing-attention variability). "
            f"Category drift: {drift} (tendency to switch content categories "
            f"within a session). Return-gap between sessions: {gap}. "
            f"Policy/surface-shift exposure: {policy_rate:.0%} of context. "
            f"Cold-start signal: {cold:.0%} first-session activity."
        )
        user_ids.append(internal)
    return user_ids, texts


# --------------------------------------------------------------------------- #
# Embedding backends (offline)
# --------------------------------------------------------------------------- #
class HashingBackend:
    """Deterministic, dependency-free hashed bag-of-tokens embedding.

    A real structured no-LLM control condition AND the local plumbing proof:
    token-hash into `dim` buckets, L2-normalised. No network, no model download.
    """

    def __init__(self, dim=256):
        self.dim = dim

    @staticmethod
    def _stable_hash(tok):
        # built-in hash() is salted per process (PYTHONHASHSEED) -> not
        # reproducible; crc32 of the utf-8 bytes is stable across runs.
        return zlib.crc32(tok.encode("utf-8"))

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = self._stable_hash(tok) % self.dim
                out[i, h] += 1.0
            nrm = np.linalg.norm(out[i])
            if nrm > 0:
                out[i] /= nrm
        return out


class HFBackend:
    """Offline HF / sentence-transformers embedder — the "as large as compute
    allows" profiler, run on Modal GPU. Imported lazily so this module still
    imports in the base `recbole` env (which has no transformers)."""

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device=None,
                 batch_size=64, dtype=None):
        from sentence_transformers import SentenceTransformer

        # 7B-class embedders would OOM a 24GB card in fp32 (~28GB weights); load
        # them in half precision. `dtype` is a torch dtype name e.g. "float16".
        model_kwargs = None
        if dtype:
            model_kwargs = {"torch_dtype": getattr(torch, dtype)}
        self.model = SentenceTransformer(model_name, device=device,
                                         model_kwargs=model_kwargs)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts):
        emb = self.model.encode(texts, batch_size=self.batch_size,
                                normalize_embeddings=True,
                                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def make_backend(kind, dim=256, model=None):
    if kind == "hashing":
        return HashingBackend(dim=dim)
    if kind == "hf":
        return HFBackend(model_name=model or "BAAI/bge-large-en-v1.5")
    raise ValueError(f"unknown backend {kind!r}")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(dataset, backend_factory, base_config=DEFAULT_BASE_CONFIG,
          inter_root=None, out_dir=None, data_path=None, write=True):
    """Render every user's causal profile, embed, and write a contract-valid
    cache. `backend_factory` is a zero-arg callable returning the embedding
    backend — it is called only AFTER the RecBole remap + text rendering
    succeed, so a heavy model (HFBackend) is never downloaded if an upstream
    step fails. `inter_root`/`out_dir`/`data_path` override the atomic-file
    root, output dir, and RecBole data_path (for Modal's `/data` volume)."""
    t0 = time.perf_counter()
    inter_root = Path(inter_root) if inter_root else DATA_DIR / "recbole"
    out_dir = Path(out_dir) if out_dir else PROFILE_ROOT
    inter_path = inter_root / dataset / f"{dataset}.inter"
    if not inter_path.exists():
        raise FileNotFoundError(inter_path)

    # Cheap steps first (remap + render) — fail here before any model download.
    uid_token2id, _iid_token2id, n_users = load_user_item_remap(
        dataset, base_config, data_path=data_path)
    user_ids, texts = render_profiles(inter_path, uid_token2id)

    backend = backend_factory()   # constructs/loads the model only now
    vecs = backend.encode(texts) if texts else np.zeros((0, backend.dim), np.float32)
    profile_dim = backend.dim
    cache = torch.zeros(n_users, profile_dim, dtype=torch.float32)
    if len(user_ids):
        cache[torch.tensor(user_ids, dtype=torch.long)] = torch.from_numpy(vecs)
    cache[0] = 0.0  # RecBole padding user

    validate_profiles(cache, n_users, profile_dim)
    n_sparse = int((cache.abs().sum(dim=1) == 0).sum())
    tag = type(backend).__name__.replace("Backend", "").lower()

    stats = {
        "dataset": dataset, "backend": type(backend).__name__,
        "n_users": n_users, "profile_dim": profile_dim,
        "profiled_users": len(user_ids), "sparse_users": n_sparse,
        "sec": round(time.perf_counter() - t0, 1),
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{dataset}.profiles.{tag}.d{profile_dim}.pt"
        save_profiles(out, cache, n_users, profile_dim)
        stats["path"] = str(out)
        stats["size_mb"] = round(out.stat().st_size / 1e6, 2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="kuairand_1k_ctx",
                    help="the _ctx dataset CAFREC trains on")
    ap.add_argument("--backend", choices=["hashing", "hf"], default="hashing")
    ap.add_argument("--dim", type=int, default=256, help="hashing backend dim")
    ap.add_argument("--model", default=None, help="hf backend model name")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    s = build(args.dataset,
              lambda: make_backend(args.backend, dim=args.dim, model=args.model),
              write=not args.no_write)
    print("\n--- profile build ---")
    for k, v in s.items():
        print(f"  {k:16s}: {v}")


if __name__ == "__main__":
    main()
