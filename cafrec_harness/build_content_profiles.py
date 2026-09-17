"""Content-bearing long-term profiles for CAFREC on KuaiRand-Pure (2026-09-17).

    cat_beyond  L1-normalised histogram over first-level categories of the
                user's beyond-window rows                       d = n categories
    cap_beyond  L2-normalised mean caption embedding over beyond-window rows
                                                                d = encoder dim
    cap_all     as cap_beyond, over ALL training rows (window not excluded)

Causality / alignment (same rules as cafrec.features.build_profiles.render_profiles):
  * kuairand_pure_ctx.inter is stable-sorted by (user_id, timestamp); a row's
    rank from the end r = 0 is test, r = 1 is validation. Training rows: r >= 2.
  * "Beyond the window": r >= 22, i.e. the valid/test rows AND the 20 rows
    immediately before them are dropped. Users with <= 22 rows get a zero vector.
  * Row i of each tensor is RecBole internal user id i (load_user_item_remap);
    row 0 is padding. Tensors are float32, validated, frozen by the model.

Inputs are the Pure-item subsets written by audit_content_data.py
(data/content_pure/{categories,captions}_pure.csv). Captions are embedded once
per video and cached in data/content_pure/.

    PYTHONUTF8=1 <env python> build_content_profiles.py      (from cafrec_harness/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cafrec.features.build_profiles import load_user_item_remap
from cafrec.features.profiles import save_profiles, validate_profiles

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATASET = "kuairand_pure_ctx"
INTER = DATA / "recbole" / DATASET / f"{DATASET}.inter"
SUBSET = DATA / "content_pure"
CAT_CSV = SUBSET / "categories_pure.csv"
CAP_CSV = SUBSET / "captions_pure.csv"
PROFILE_DIR = DATA / "profiles"
ID = "final_video_id"
UNKNOWN_ID = -124.0
N_HOLDOUT = 2          # valid + test
WINDOW = 20            # MAX_ITEM_LIST_LENGTH rows SASRec sees at test time
DEFAULT_ENCODER = "BAAI/bge-large-zh-v1.5"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(uid_token2id):
    df = pd.read_csv(INTER, sep="\t", usecols=[0, 1, 2])
    df.columns = [c.split(":")[0] for c in df.columns]
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["r"] = df.groupby("user_id").cumcount(ascending=False)
    df["uid"] = df["user_id"].astype(str).map(uid_token2id)
    if df["uid"].isna().any():
        raise ValueError(f"{int(df['uid'].isna().sum())} rows have users missing from the RecBole remap")
    df["uid"] = df["uid"].astype(np.int64)
    return df


def repaired_captions():
    """Caption text per Pure video. For the rows whose `duration` field holds
    text (the source line splits the caption across the caption /
    show_cover_text / duration slots) the caption is the non-numeric text
    fields joined in file order."""
    cap = pd.read_csv(CAP_CSV, dtype={"caption": str, "show_cover_text": str, "duration": str})
    dur_num = pd.to_numeric(cap["duration"], errors="coerce")
    spill = cap["duration"].notna() & dur_num.isna()

    def is_text(v):
        return isinstance(v, str) and v.strip() != "" and pd.isna(pd.to_numeric(v, errors="coerce"))

    text = cap["caption"].fillna("").str.strip()
    for i in np.flatnonzero(spill.to_numpy()):
        row = cap.iloc[i]
        parts = [row["caption"], row["show_cover_text"], row["duration"]]
        text.iloc[i] = " ".join(p.strip() for p in parts if is_text(p))
    return dict(zip(cap[ID].astype(np.int64), text)), int(spill.sum())


def caption_embeddings(item_ids, texts, encoder, batch_size):
    """Embed each non-empty caption once; cached per encoder."""
    slug = encoder.replace("/", "__")
    cache = SUBSET / f"caption_emb.{slug}.npz"
    ids = np.asarray([i for i in item_ids if texts.get(i, "") != ""], np.int64)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if np.array_equal(z["ids"], ids):
            print(f"  reusing {cache.name}", flush=True)
            return ids, z["emb"], dict(cache=str(cache), reused=True, revision=str(z["revision"]))
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(encoder, device="cuda" if torch.cuda.is_available() else "cpu")
    revision = str(model[0].auto_model.config._name_or_path)
    emb = model.encode([texts[i] for i in ids], batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.savez(cache, ids=ids, emb=emb, revision=revision)
    info = dict(cache=str(cache), reused=False, revision=revision,
                max_seq_length=int(model.max_seq_length), device=str(model.device),
                encode_seconds=round(time.time() - t0, 1))
    print(f"  embedded {len(ids):,} captions in {info['encode_seconds']}s on {info['device']}", flush=True)
    return ids, emb, info


def write_profile(name, cache, n_users, meta):
    d = int(cache.shape[1])
    cache[0] = 0.0
    validate_profiles(cache, n_users, d)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{DATASET}.profiles.{name}.d{d}.pt"
    save_profiles(path, cache, n_users, d)
    nonzero = (cache.abs().sum(dim=1) > 0)
    meta = dict(name=name, path=str(path), dataset=DATASET, n_users=n_users, profile_dim=d,
                nonzero_users=int(nonzero.sum()), nonzero_share_excl_padding=float(nonzero[1:].float().mean()),
                sha256=sha256(path), built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                script="build_content_profiles.py", **meta)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    print(f"  {path.name}: nonzero {meta['nonzero_users']:,}/{n_users - 1:,} users "
          f"({100 * meta['nonzero_share_excl_padding']:.1f}%)", flush=True)
    return nonzero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    uid_token2id, iid_token2id, n_users = load_user_item_remap(DATASET)
    df = load_rows(uid_token2id)
    train = df[df["r"] >= N_HOLDOUT]
    beyond = df[df["r"] >= N_HOLDOUT + WINDOW]
    rows_def = dict(sort="stable (user_id, timestamp)", holdout_rows_dropped=N_HOLDOUT,
                    window_rows_dropped=WINDOW, inter=str(INTER), inter_sha256=sha256(INTER))
    print(f"rows {len(df):,}; training rows {len(train):,}; beyond-window rows {len(beyond):,}; "
          f"users with beyond rows {beyond['uid'].nunique():,}", flush=True)

    # per-user row counts, for cohort breakdowns of profile coverage
    per_user = df.groupby(["user_id", "uid"]).agg(n_rows=("r", "size")).reset_index()
    per_user["n_train"] = (per_user["n_rows"] - N_HOLDOUT).clip(lower=0)
    per_user["n_beyond"] = (per_user["n_rows"] - N_HOLDOUT - WINDOW).clip(lower=0)

    # ------------------------------------------------------------- cat_beyond
    cat = pd.read_csv(CAT_CSV)
    known = cat["first_level_category_id"].notna() & (cat["first_level_category_id"] != UNKNOWN_ID)
    cats = sorted(int(c) for c in cat.loc[known, "first_level_category_id"].astype(int).unique())
    col = {c: j for j, c in enumerate(cats)}
    first = cat.loc[known].drop_duplicates("first_level_category_id")
    names = dict(zip(first["first_level_category_id"].astype(int), first["first_level_category_name"]))
    item_col = dict(zip(cat.loc[known, ID].astype(np.int64), cat.loc[known, "first_level_category_id"].astype(int).map(col)))
    b_col = beyond["item_id"].map(item_col)
    ok = b_col.notna().to_numpy()
    hist = torch.zeros(n_users, len(cats), dtype=torch.float32)
    hist.index_put_((torch.as_tensor(beyond["uid"].to_numpy()[ok]), torch.as_tensor(b_col.to_numpy()[ok].astype(np.int64))),
                    torch.ones(int(ok.sum())), accumulate=True)
    tot = hist.sum(dim=1, keepdim=True)
    hist = torch.where(tot > 0, hist / tot.clamp(min=1.0), hist)
    nz_cat = write_profile("cat_beyond", hist, n_users, dict(
        kind="L1-normalised first-level category histogram over beyond-window rows",
        rows=rows_def, categories_file=str(CAT_CSV),
        columns=[dict(col=j, first_level_category_id=c, name=names[c]) for j, c in enumerate(cats)],
        beyond_rows_used=int(ok.sum()), beyond_rows_without_category=int((~ok).sum())))

    # ------------------------------------------------------------- captions
    texts, n_spill = repaired_captions()
    items = sorted(int(t) for t in iid_token2id if t != "[PAD]")
    ids, emb, enc_info = caption_embeddings(items, texts, args.encoder, args.batch_size)
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    emb_t = torch.from_numpy(emb)

    def mean_caption_profile(rows):
        idx = rows["item_id"].map(row_of)
        ok = idx.notna().to_numpy()
        uid = torch.as_tensor(rows["uid"].to_numpy()[ok])
        eidx = torch.as_tensor(idx.to_numpy()[ok].astype(np.int64))
        acc = torch.zeros(n_users, emb.shape[1], dtype=torch.float32)
        for s in range(0, len(uid), 100_000):
            acc.index_add_(0, uid[s:s + 100_000], emb_t[eidx[s:s + 100_000]])
        norm = acc.norm(dim=1, keepdim=True)
        acc = torch.where(norm > 0, acc / norm.clamp(min=1e-12), acc)
        return acc, int(ok.sum()), int((~ok).sum())

    cap_meta = dict(encoder=args.encoder, encoder_info=enc_info, captions_file=str(CAP_CSV),
                    caption_text="raw `caption` field; the rows whose duration field holds text use "
                                 "the non-numeric text fields joined in file order",
                    caption_rows_repaired=n_spill, videos_with_caption_embedding=int(len(ids)),
                    videos_in_catalogue=len(items))
    prof, used, skipped = mean_caption_profile(beyond)
    nz_capb = write_profile("cap_beyond", prof, n_users, dict(
        kind="L2-normalised mean caption embedding over beyond-window rows", rows=rows_def,
        rows_used=used, rows_skipped_no_caption=skipped, **cap_meta))
    prof, used, skipped = mean_caption_profile(train)
    rows_all = dict(rows_def, window_rows_dropped=0)
    nz_capa = write_profile("cap_all", prof, n_users, dict(
        kind="L2-normalised mean caption embedding over all training rows", rows=rows_all,
        rows_used=used, rows_skipped_no_caption=skipped, **cap_meta))

    uid_arr = torch.as_tensor(per_user["uid"].to_numpy())
    per_user["nz_cat_beyond"] = nz_cat[uid_arr].numpy()
    per_user["nz_cap_beyond"] = nz_capb[uid_arr].numpy()
    per_user["nz_cap_all"] = nz_capa[uid_arr].numpy()
    out = SUBSET / "profile_user_rows.csv"
    per_user.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
