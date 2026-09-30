"""Rank-truncated copies of the LLM profile cache, for the rank sweep (2026-09-21).

Why
---
H2 came out null on 2026-09-21: the LLM profile does not beat the trainable
stand-in for lowest-quartile users (-0.4%, Holm p=0.80), and it LOSES to the
templated profile (-10.4%) and to having no long-term branch at all (-12.6%).
Measuring the caches explained why, and inverted the obvious hypothesis:

    cache            mean pairwise cosine    effective rank (of 1024)
    LLM (plain)            +0.890                    94.8
    Template (bge)         +0.991                     6.5
    Hashing                +0.937                    35.4

The templated profile is very nearly a CONSTANT vector, and it is the one that
helps sparse users. So the +11.1% that a frozen profile buys over the stand-in in
the lowest quartile may not be personalisation at all — it may be conditioning,
with the per-user variation in a richer profile acting as noise for users who
have too little history to place them.

This module tests that directly by varying ONLY the rank of the profile cache,
holding the text, the encoder, the dimension and the row norms fixed:

    X_k = mean + (top-k principal components of the centred cache), rows
          renormalised to unit L2

k = 0 is every user receiving the corpus mean — a profile with zero per-user
information, i.e. the constant control. The untruncated cache (effective rank
94.8) is already evaluated as the `llm_plain` arm, so the sweep plus that run
gives the accuracy-versus-rank curve.

`--random-k` additionally writes a control that keeps the mean, the rank and each
user's principal-component SCORES but replaces the principal DIRECTIONS with a
random orthonormal basis. It separates "per-user variation of this magnitude
hurts" from "the LLM's particular semantic directions hurt" — the same role the
shuffled-context and hashing controls play elsewhere in the paper.

Runs on CPU against the volume, so the 94 MB caches are never downloaded or
uploaded. Diagnostics for every cache are returned and written to a sidecar.

    modal run modal_rank_profiles.py
    modal run modal_rank_profiles.py --ranks 0,2,6,16,32 --random-k 32
"""
import json
import time

import modal

app = modal.App("cafrec-rank-profiles")

image = (
    modal.Image.debian_slim(python_version="3.10")
    # pandas is not used here, but importing cafrec.features.profiles pulls in
    # cafrec.features.__init__ -> context.py, which imports it.
    .pip_install("torch==2.8.0", "numpy<2", "pandas")
    .add_local_python_source("cafrec")
)

data_vol = modal.Volume.from_name("cafrec-data", create_if_missing=True)

SRC = "/data/profiles/kuairand_pure_ctx.profiles.llm_plain.d1024.pt"
OUT_TMPL = "/data/profiles/kuairand_pure_ctx.profiles.{name}.d{d}.pt"
DIAG_SEED = 0          # same sample seed as the original geometry measurement
SAMPLE = 3000


@app.function(image=image, cpu=4.0, memory=32768, timeout=60 * 60,
              volumes={"/data": data_vol})
def truncate(ranks: str = "0,2,6,16,32", random_k: int = 32,
             seed: int = 20260921) -> dict:
    import hashlib
    import os

    import numpy as np
    import torch

    from cafrec.features.profiles import save_profiles, validate_profiles

    t0 = time.time()
    data_vol.reload()
    if not os.path.exists(SRC):
        raise FileNotFoundError(f"{SRC} missing — run modal_h2_embed.py --variant plain first")

    X = torch.load(SRC, weights_only=False).float()
    n_users, d = X.shape
    body = X[1:]                                  # row 0 is RecBole padding
    mean = body.mean(dim=0, keepdim=True)
    Xc = body - mean

    # PCA by eigendecomposition of the 1024x1024 covariance: exact, and far
    # cheaper than an SVD of the 22,912 x 1,024 matrix.
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)          # ascending
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    scores = Xc @ evecs                            # [n-1, d] principal scores

    def diagnostics(M):
        """Mean pairwise cosine and effective rank, measured exactly as the
        original cache geometry was, so the numbers are comparable."""
        rng = np.random.default_rng(DIAG_SEED)
        idx = rng.choice(len(M), min(SAMPLE, len(M)), replace=False)
        S = M[idx]
        S = S / S.norm(dim=1, keepdim=True).clamp(min=1e-12)
        C = (S @ S.T).numpy()
        iu = np.triu_indices(len(S), 1)
        sims = C[iu]
        Mc = M - M.mean(0)
        c = (Mc.T @ Mc / max(len(Mc) - 1, 1)).numpy()
        ev = np.clip(np.linalg.eigvalsh(c), 0, None)
        tot = ev.sum()
        if tot <= 1e-20:                           # k=0: every row identical
            eff = 1.0
        else:
            p = ev / tot
            eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        norms = M.norm(dim=1)
        return dict(mean_pairwise_cos=float(sims.mean()),
                    p95_pairwise_cos=float(np.quantile(sims, 0.95)),
                    effective_rank=round(eff, 2),
                    row_norm_min=float(norms.min()), row_norm_max=float(norms.max()))

    def write(name, body_mat):
        cache = torch.zeros(n_users, d, dtype=torch.float32)
        # unit L2 rows, matching the source cache (bge output is normalised), so
        # the sweep varies rank and not scale
        body_mat = body_mat / body_mat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cache[1:] = body_mat
        cache[0] = 0.0
        validate_profiles(cache, n_users, d)
        path = OUT_TMPL.format(name=name, d=d)
        save_profiles(path, cache, n_users, d)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        diag = diagnostics(cache[1:])
        meta = dict(name=name, path=path, source=SRC, dataset="kuairand_pure_ctx",
                    n_users=n_users, profile_dim=d, sha256=h.hexdigest(),
                    built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    script="modal_rank_profiles.py", rows_unit_normalised=True,
                    **diag)
        with open(path.replace(".pt", ".json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  {name:<12} eff.rank {diag['effective_rank']:>7.2f}  "
              f"cos {diag['mean_pairwise_cos']:+.3f}", flush=True)
        return meta

    out = {"source_diagnostics": diagnostics(body), "caches": {}}
    for k in [int(x) for x in ranks.split(",") if x.strip() != ""]:
        if k == 0:
            recon = mean.expand(len(body), d).clone()
        else:
            recon = mean + scores[:, :k] @ evecs[:, :k].T
        out["caches"][f"llm_rank{k}"] = write(f"llm_rank{k}", recon)

    if random_k:
        # same mean, same rank, same per-user scores; random DIRECTIONS
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(d, random_k, generator=g))
        recon = mean + scores[:, :random_k] @ Q.T
        out["caches"][f"llm_rand{random_k}"] = write(f"llm_rand{random_k}", recon)
        out["random_control"] = dict(
            k=random_k, seed=seed,
            note="mean and per-user PC scores preserved; principal directions "
                 "replaced by a random orthonormal basis")

    data_vol.commit()
    out["variance_explained"] = {
        f"k={k}": round(float(evals[:k].sum() / evals.sum()), 4)
        for k in (1, 2, 6, 16, 32, 64, 128)
    }
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.local_entrypoint()
def main(ranks: str = "0,2,6,16,32", random_k: int = 32, seed: int = 20260921):
    rep = truncate.remote(ranks=ranks, random_k=random_k, seed=seed)
    print("\n--- rank-truncated profile caches (Modal) ---")
    print(json.dumps(rep, indent=2))
    print("\nNext:")
    print("  modal run modal_run.py::h2 --conditions "
          + ",".join(list(rep["caches"])) + ",hashing")
