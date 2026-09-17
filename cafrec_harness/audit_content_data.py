"""Step 1 audit: do kuairand_video_categories.csv / kuairand_video_captions.csv
describe the same videos as KuaiRand-Pure?

Checks
  1. inventory of both files (columns, dtypes, rows, distinct ids, samples,
     category levels, caption language / null rate / length distribution);
  2. id coverage of the item ids in kuairand_pure.inter;
  3. co-consumption lift: P(same category) for consecutive same-user clicks
     <= 30 min apart, vs sum_c share_c^2, vs a permuted-category null;
  4. per-video attributes shared with KuaiRand-Pure itself: caption-file
     `duration` vs video_features_basic_pure.csv `video_duration` and vs the raw
     log `duration_ms`; first-level category id vs KuaiRand `tag`. Each has an
     id+1 shift control.

Also writes the Pure-item subset of both files to data/content_pure/ so later
steps need not rescan the multi-GB CSVs.

    PYTHONUTF8=1 <env python> audit_content_data.py      (from cafrec_harness/)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CAT_CSV = DATA / "kuairand_video_categories.csv"
CAP_CSV = DATA / "kuairand_video_captions.csv"
INTER = DATA / "recbole" / "kuairand_pure" / "kuairand_pure.inter"
INTER_CTX = DATA / "recbole" / "kuairand_pure_ctx" / "kuairand_pure_ctx.inter"
PURE = DATA / "KuaiRand-Pure" / "data"
BASIC = PURE / "video_features_basic_pure.csv"
LOGS = [PURE / "log_standard_4_08_to_4_21_pure.csv",
        PURE / "log_standard_4_22_to_5_08_pure.csv",
        PURE / "log_random_4_22_to_5_08_pure.csv"]
SUBSET_DIR = DATA / "content_pure"
DATE = time.strftime("%Y-%m-%d")
_DRY = "_dryrun" if os.environ.get("AUDIT_MAX_ROWS") else ""
OUT_TXT = HERE / "results" / f"content_data_audit_{DATE}{_DRY}.txt"
OUT_JSON = HERE / "results" / f"content_data_audit_{DATE}{_DRY}.json"

ID = "final_video_id"
UNKNOWN_ID = -124.0          # the files' "no category at this level" code
GAP_MS = 30 * 60 * 1000
N_PERM = 200
CHUNK = 1_000_000
MAX_ROWS = int(os.environ.get("AUDIT_MAX_ROWS", "0"))   # >0: dry run on the first rows only
LEN_CLIP = 5000
SEED = 403092

_lines = []
summary = {}


def say(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))
    OUT_TXT.write_text("\n".join(_lines) + "\n", encoding="utf-8")


def pct(x):
    return f"{100.0 * x:.2f}%"


# --------------------------------------------------------------------------- #
# interactions
# --------------------------------------------------------------------------- #
def load_inter(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# chunked scans of the big files
# --------------------------------------------------------------------------- #
def scan_categories(item_set):
    t0 = time.time()
    n_rows, n_bad_id = 0, 0
    ids, keep = [], []
    head, dtypes = None, None
    unk_full = {lvl: 0 for lvl in ("first", "second", "third", "fourth")}
    for chunk in pd.read_csv(CAT_CSV, chunksize=CHUNK, dtype={ID: str}):
        if head is None:
            head, dtypes = chunk.head(5).copy(), None
        num = pd.to_numeric(chunk[ID], errors="coerce")
        n_bad_id += int(num.isna().sum())
        chunk[ID] = num
        chunk = chunk[num.notna()]
        if dtypes is None:
            dtypes = chunk.dtypes
        n_rows += len(chunk)
        vid = chunk[ID].to_numpy(np.int64)
        ids.append(vid)
        for lvl in unk_full:
            col = chunk[f"{lvl}_level_category_id"]
            unk_full[lvl] += int((col.isna() | (col == UNKNOWN_ID)).sum())
        keep.append(chunk[np.isin(vid, item_set)])
        print(f"  categories: {n_rows:,} rows scanned ({time.time() - t0:.0f}s)", flush=True)
        if MAX_ROWS and n_rows >= MAX_ROWS:
            break
    ids = np.concatenate(ids)
    return dict(n_rows=n_rows, n_bad_id=n_bad_id, ids=ids, head=head, dtypes=dtypes,
                unk_full=unk_full, subset=pd.concat(keep, ignore_index=True),
                sec=time.time() - t0)


def _cjk_mask(s):
    return s.str.contains(r"[一-鿿]", regex=True, na=False)


def scan_captions(item_set):
    t0 = time.time()
    n_rows, n_bad_id = 0, 0
    ids, keep = [], []
    head, dtypes = None, None
    cap_null = cap_empty = cap_cjk = cover_nonempty = dur_null = dur_nonnum = 0
    dur_examples = []
    len_hist = np.zeros(LEN_CLIP + 1, np.int64)
    len_max = 0
    for chunk in pd.read_csv(CAP_CSV, chunksize=CHUNK, dtype={ID: str, "caption": str,
                                                                "show_cover_text": str}):
        if head is None:
            head = chunk.head(5).copy()
        num = pd.to_numeric(chunk[ID], errors="coerce")
        n_bad_id += int(num.isna().sum())
        chunk[ID] = num
        chunk = chunk[num.notna()]
        if dtypes is None:
            dtypes = chunk.dtypes
        n_rows += len(chunk)
        vid = chunk[ID].to_numpy(np.int64)
        ids.append(vid)
        cap = chunk["caption"]
        cap_null += int(cap.isna().sum())
        stripped = cap.fillna("").str.strip()
        nonempty = stripped != ""
        cap_empty += int((cap.notna() & ~nonempty).sum())
        cap_cjk += int(_cjk_mask(stripped[nonempty]).sum())
        lens = stripped[nonempty].str.len().to_numpy(np.int64)
        if lens.size:
            len_max = max(len_max, int(lens.max()))
            len_hist += np.bincount(np.clip(lens, 0, LEN_CLIP), minlength=LEN_CLIP + 1)
        cover_nonempty += int((chunk["show_cover_text"].fillna("").str.strip() != "").sum())
        dur_null += int(chunk["duration"].isna().sum())
        dur_num = pd.to_numeric(chunk["duration"], errors="coerce")
        bad = chunk["duration"].notna() & dur_num.isna()
        dur_nonnum += int(bad.sum())
        if bad.any() and len(dur_examples) < 5:
            dur_examples.extend(chunk.loc[bad, "duration"].astype(str).str[:40].head(5 - len(dur_examples)).tolist())
        keep.append(chunk[np.isin(vid, item_set)])
        print(f"  captions: {n_rows:,} rows scanned ({time.time() - t0:.0f}s)", flush=True)
        if MAX_ROWS and n_rows >= MAX_ROWS:
            break
    ids = np.concatenate(ids)
    return dict(n_rows=n_rows, n_bad_id=n_bad_id, ids=ids, head=head, dtypes=dtypes,
                cap_null=cap_null, cap_empty=cap_empty, cap_cjk=cap_cjk,
                cover_nonempty=cover_nonempty, dur_null=dur_null, dur_nonnum=dur_nonnum,
                dur_examples=dur_examples, len_hist=len_hist,
                len_max=len_max, subset=pd.concat(keep, ignore_index=True),
                sec=time.time() - t0)


def hist_quantiles(hist, qs):
    cdf = np.cumsum(hist) / hist.sum()
    return {q: int(np.searchsorted(cdf, q)) for q in qs}


def caption_stats(cap_series):
    s = cap_series.fillna("").astype(str).str.strip()
    nonempty = s != ""
    lens = s[nonempty].str.len()
    out = dict(n=len(s), null=int(cap_series.isna().sum()),
               empty=int((cap_series.notna() & ~nonempty).sum()),
               nonempty=int(nonempty.sum()),
               cjk=int(_cjk_mask(s[nonempty]).sum()))
    if len(lens):
        out["len_q"] = {str(q): float(lens.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
        out["len_mean"] = float(lens.mean())
        out["len_max"] = int(lens.max())
        text = "".join(s[nonempty].tolist())
        cjk_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
        latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        out["cjk_char_share"] = cjk_chars / max(len(text), 1)
        out["latin_char_share"] = latin_chars / max(len(text), 1)
    return out


# --------------------------------------------------------------------------- #
# lift test
# --------------------------------------------------------------------------- #
def consecutive_pairs(inter):
    df = inter.sort_values(["user_id", "timestamp"], kind="mergesort")
    u = df["user_id"].to_numpy()
    it = df["item_id"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy(np.float64)
    gap = ts[1:] - ts[:-1]
    ok = (u[1:] == u[:-1]) & (gap >= 0) & (gap <= GAP_MS)
    return it[:-1][ok], it[1:][ok], it


def encode_map(ids, labels, size):
    """item id -> dense category code; -1 = no category."""
    codes, uniques = pd.factorize(pd.Series(labels), sort=True)
    arr = np.full(size, -1, np.int64)
    arr[np.asarray(ids, np.int64)] = codes          # factorize gives -1 for NaN
    return arr, len(uniques)


def lift(cat_of, a, b, clicks, exclude_same_item):
    ca, cb = cat_of[a], cat_of[b]
    m = (ca >= 0) & (cb >= 0)
    if exclude_same_item:
        m &= a != b
    k = int(max(cat_of.max(), 0)) + 1
    obs = float((ca[m] == cb[m]).mean())
    cc = cat_of[clicks]
    cc = cc[cc >= 0]
    share = np.bincount(cc, minlength=k) / len(cc)
    exp = float((share ** 2).sum())
    pa = np.bincount(ca[m], minlength=k) / m.sum()
    pb = np.bincount(cb[m], minlength=k) / m.sum()
    exp_pair = float((pa * pb).sum())
    return dict(pairs=int(m.sum()), observed=obs, expected=exp, lift=obs / exp,
                expected_pair_marginals=exp_pair, lift_pair_marginals=obs / exp_pair)


def lift_with_null(name, cat_of, a, b, clicks, item_pool, rng):
    res = {}
    for excl in (False, True):
        key = "excl_same_item" if excl else "incl_same_item"
        real = lift(cat_of, a, b, clicks, excl)
        has = item_pool[cat_of[item_pool] >= 0]
        null = []
        for _ in range(N_PERM):
            perm = cat_of.copy()
            perm[has] = rng.permutation(cat_of[has])
            null.append(lift(perm, a, b, clicks, excl)["lift"])
        null = np.asarray(null)
        real["null_lift_mean"] = float(null.mean())
        real["null_lift_sd"] = float(null.std(ddof=1))
        real["null_lift_p99"] = float(np.quantile(null, 0.99))
        real["null_lift_max"] = float(null.max())
        real["z_vs_null"] = float((real["lift"] - null.mean()) / max(null.std(ddof=1), 1e-12))
        res[key] = real
        say(f"  {name:28s} {key:15s} pairs={real['pairs']:,}  P(same)={real['observed']:.4f}  "
            f"E={real['expected']:.4f}  lift={real['lift']:.2f}  "
            f"(pair-marginal E={real['expected_pair_marginals']:.4f}, lift={real['lift_pair_marginals']:.2f})  "
            f"null lift {real['null_lift_mean']:.3f} +/- {real['null_lift_sd']:.3f} "
            f"[p99 {real['null_lift_p99']:.3f}, max {real['null_lift_max']:.3f}]  z={real['z_vs_null']:.0f}")
    return res


# --------------------------------------------------------------------------- #
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    say(f"CONTENT DATA AUDIT  {time.strftime('%Y-%m-%d %H:%M:%S')}  (script: audit_content_data.py)")
    say(f"categories file : {CAT_CSV}  ({CAT_CSV.stat().st_size / 1e9:.2f} GB)")
    say(f"captions file   : {CAP_CSV}  ({CAP_CSV.stat().st_size / 1e9:.2f} GB)")
    say(f"interactions    : {INTER}")
    say("")

    inter = load_inter(INTER)
    items = np.unique(inter["item_id"].to_numpy(np.int64))
    inter_ctx = load_inter(INTER_CTX)
    items_ctx = np.unique(inter_ctx["item_id"].to_numpy(np.int64))
    df_sorted = inter.sort_values(["user_id", "timestamp"], kind="mergesort")
    rank_desc = df_sorted.groupby("user_id").cumcount(ascending=False)
    train_items = np.unique(df_sorted.loc[rank_desc >= 2, "item_id"].to_numpy(np.int64))
    say("== 0. Interaction file")
    say(f"kuairand_pure.inter     : {len(inter):,} rows, {inter['user_id'].nunique():,} users, "
        f"{len(items):,} distinct item_id (min {items.min()}, max {items.max()})")
    say(f"kuairand_pure_ctx.inter : {len(inter_ctx):,} rows, {len(items_ctx):,} distinct item_id; "
        f"same item set as kuairand_pure: {np.array_equal(items, items_ctx)}")
    say(f"items in TRAINING rows (all but each user's last 2): {len(train_items):,}")
    summary["inter"] = dict(rows=len(inter), users=int(inter["user_id"].nunique()),
                            items=len(items), train_items=len(train_items))
    say("")

    # ---------------------------------------------------------------- 1. inventory
    say("== 1a. Categories file inventory")
    cs = scan_categories(items)
    ids_u = np.unique(cs["ids"])
    say(f"rows {cs['n_rows']:,}; non-numeric id rows dropped {cs['n_bad_id']:,}; "
        f"distinct {ID} {len(ids_u):,} (min {ids_u.min()}, max {ids_u.max()}); "
        f"duplicate-id rows {cs['n_rows'] - len(ids_u):,}; scan {cs['sec']:.0f}s")
    say("columns / dtypes:")
    for c, t in cs["dtypes"].items():
        say(f"   {c:32s} {t}")
    say("first 5 rows:")
    say(cs["head"].to_string())
    for lvl, n in cs["unk_full"].items():
        say(f"   {lvl}-level UNKNOWN (-124) or null, whole file: {pct(n / cs['n_rows'])}")
    summary["categories_file"] = dict(rows=cs["n_rows"], distinct_ids=len(ids_u),
                                      id_min=int(ids_u.min()), id_max=int(ids_u.max()),
                                      unknown_rate_by_level={k: v / cs["n_rows"] for k, v in cs["unk_full"].items()})
    cat = cs["subset"].drop_duplicates(subset=ID, keep="first").copy()
    cat[ID] = cat[ID].astype(np.int64)
    say(f"rows matching a Pure item id: {len(cs['subset']):,} (distinct {len(cat):,})")
    say("category levels on the Pure item subset:")
    lvl_stats = {}
    for lvl in ("first", "second", "third", "fourth"):
        cid = cat[f"{lvl}_level_category_id"]
        known = cid.notna() & (cid != UNKNOWN_ID)
        n_cat = int(cid[known].nunique())
        probs = cat.loc[known, f"{lvl}_level_category_prob"]
        lvl_stats[lvl] = dict(n_categories=n_cat, known_share=float(known.mean()),
                              prob_median=float(probs.median()) if len(probs) else None)
        say(f"   {lvl:6s}: {n_cat:4d} distinct categories; known for {pct(known.mean())} of Pure videos; "
            f"median prob {lvl_stats[lvl]['prob_median']}")
    fl = cat[cat["first_level_category_id"].notna() & (cat["first_level_category_id"] != UNKNOWN_ID)]
    top = (fl.groupby(["first_level_category_id", "first_level_category_name"]).size()
             .sort_values(ascending=False))
    say("first-level categories on Pure videos (id, name, n videos):")
    for (cid, name), n in top.items():
        say(f"   {int(cid):5d}  {name}  {n}")
    summary["categories_pure_levels"] = lvl_stats
    say("")

    say("== 1b. Captions file inventory")
    ps = scan_captions(items)
    ids_c = np.unique(ps["ids"])
    say(f"rows {ps['n_rows']:,}; non-numeric id rows dropped {ps['n_bad_id']:,}; "
        f"distinct {ID} {len(ids_c):,} (min {ids_c.min()}, max {ids_c.max()}); "
        f"duplicate-id rows {ps['n_rows'] - len(ids_c):,}; scan {ps['sec']:.0f}s")
    say("columns / dtypes:")
    for c, t in ps["dtypes"].items():
        say(f"   {c:32s} {t}")
    say("first 5 rows:")
    say(ps["head"].to_string())
    n = ps["n_rows"]
    nonempty = n - ps["cap_null"] - ps["cap_empty"]
    q = hist_quantiles(ps["len_hist"], (0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    say(f"whole file: caption null {pct(ps['cap_null'] / n)}, empty/whitespace {pct(ps['cap_empty'] / n)}; "
        f"non-empty captions containing CJK ideographs {pct(ps['cap_cjk'] / max(nonempty, 1))}; "
        f"show_cover_text non-empty {pct(ps['cover_nonempty'] / n)}; duration null {pct(ps['dur_null'] / n)}, "
        f"non-numeric {ps['dur_nonnum']:,} rows (examples: {ps['dur_examples']})")
    say(f"whole file caption length (chars, non-empty): quantiles {q}; max {ps['len_max']}")
    summary["captions_file"] = dict(rows=n, distinct_ids=len(ids_c), id_min=int(ids_c.min()),
                                    id_max=int(ids_c.max()), null_rate=ps["cap_null"] / n,
                                    empty_rate=ps["cap_empty"] / n,
                                    cjk_share_nonempty=ps["cap_cjk"] / max(nonempty, 1),
                                    len_quantiles=q, len_max=ps["len_max"])
    cap = ps["subset"].drop_duplicates(subset=ID, keep="first").copy()
    cap[ID] = cap[ID].astype(np.int64)
    st = caption_stats(cap["caption"])
    say(f"Pure item subset: {len(ps['subset']):,} rows (distinct {len(cap):,}); caption null {st['null']}, "
        f"empty {st['empty']}, non-empty {st['nonempty']}; non-empty with CJK {pct(st['cjk'] / max(st['nonempty'], 1))}; "
        f"CJK char share {st.get('cjk_char_share', 0):.3f}, Latin-letter char share {st.get('latin_char_share', 0):.3f}")
    say(f"Pure subset caption length (chars): quantiles {st.get('len_q')}; mean {st.get('len_mean', 0):.1f}; max {st.get('len_max')}")
    summary["captions_pure"] = st
    say("")

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    cat.to_csv(SUBSET_DIR / "categories_pure.csv", index=False, encoding="utf-8")
    cap.to_csv(SUBSET_DIR / "captions_pure.csv", index=False, encoding="utf-8")
    say(f"wrote Pure subsets to {SUBSET_DIR}")
    say("")

    # ---------------------------------------------------------------- 2. coverage
    say("== 2. ID coverage of kuairand_pure.inter item ids")
    known_first = set(fl[ID].tolist())
    any_cat = set(cat[ID].tolist())
    nonempty_cap = set(cap.loc[cap["caption"].fillna("").str.strip() != "", ID].tolist())
    cov = {}
    for label, pool in (("all items in .inter", items), ("items in training rows", train_items)):
        pool_set = pool.tolist()
        c = dict(n=len(pool),
                 has_category_row=float(np.mean([i in any_cat for i in pool_set])),
                 has_known_first_level=float(np.mean([i in known_first for i in pool_set])),
                 has_nonempty_caption=float(np.mean([i in nonempty_cap for i in pool_set])))
        cov[label] = c
        say(f"{label:24s} n={c['n']:,}: category row {pct(c['has_category_row'])}; known first-level "
            f"{pct(c['has_known_first_level'])}; non-empty caption {pct(c['has_nonempty_caption'])}")
    clicks = inter["item_id"].to_numpy(np.int64)
    say(f"click-weighted: known first-level {pct(np.isin(clicks, list(known_first)).mean())}; "
        f"non-empty caption {pct(np.isin(clicks, list(nonempty_cap)).mean())}")
    summary["coverage"] = cov
    say("")

    # ---------------------------------------------------------------- 3. lift
    say(f"== 3. Co-consumption lift (consecutive same-user clicks <= 30 min apart; {N_PERM} permutations)")
    a, b, _ = consecutive_pairs(inter)
    say(f"consecutive pairs within 30 min: {len(a):,} (same item twice: {int((a == b).sum()):,})")
    size = int(max(items.max(), cat[ID].max(), 0)) + 2
    lifts = {}
    first_of, k1 = encode_map(fl[ID], fl["first_level_category_id"].astype(int), size)
    lifts["first_level"] = lift_with_null("first-level category", first_of, a, b, clicks, items, rng)
    sl = cat[cat["second_level_category_id"].notna() & (cat["second_level_category_id"] != UNKNOWN_ID)]
    second_of, _ = encode_map(sl[ID], sl["second_level_category_id"].astype(int), size)
    lifts["second_level"] = lift_with_null("second-level category", second_of, a, b, clicks, items, rng)

    basic = pd.read_csv(BASIC)
    tag_primary = basic["tag"].astype(str).str.split(",").str[0].str.strip()
    tag_ok = basic["tag"].notna() & (tag_primary != "") & (tag_primary.str.lower() != "nan")
    tag_of, _ = encode_map(basic.loc[tag_ok, "video_id"], tag_primary[tag_ok], size)
    lifts["kuairand_tag_reference"] = lift_with_null("KuaiRand tag (reference)", tag_of, a, b, clicks, items, rng)
    # misalignment control: categories shifted by one id
    shifted = np.full(size, -1, np.int64)
    shifted[1:] = first_of[:-1]
    lifts["first_level_id_shift_control"] = lift_with_null("first-level, id+1 shift", shifted, a, b, clicks, items, rng)
    summary["lift"] = lifts
    say("")

    # ---------------------------------------------------------------- 4. attributes
    say("== 4. Per-video attributes shared with KuaiRand-Pure")
    dur_num = pd.to_numeric(cap["duration"], errors="coerce")
    spill = cap["duration"].notna() & dur_num.isna()
    say(f"Pure rows whose `duration` field holds non-numeric text (caption text spilled across columns, "
        f"so the parsed caption may be truncated): {int(spill.sum())} of {len(cap):,}")
    for _, row in cap[spill].head(3).iterrows():
        say(f"   id {row[ID]}: caption={str(row['caption'])[:60]!r} | cover={str(row['show_cover_text'])[:40]!r} "
            f"| duration={str(row['duration'])[:40]!r}")
    summary["captions_pure"]["duration_field_spill_rows"] = int(spill.sum())
    cap["duration"] = dur_num
    m = cap[[ID, "duration"]].merge(basic[["video_id", "video_duration", "tag"]],
                                    left_on=ID, right_on="video_id", how="inner")
    both = m["duration"].notna() & m["video_duration"].notna()
    d = m[both]
    exact = float((d["duration"] == d["video_duration"]).mean())
    within1s = float(((d["duration"] - d["video_duration"]).abs() <= 1000).mean())
    r = float(np.corrcoef(d["duration"], d["video_duration"])[0, 1])
    say(f"captions.duration vs video_features_basic_pure.video_duration: {len(d):,} videos with both; "
        f"exact match {pct(exact)}; |diff|<=1s {pct(within1s)}; Pearson r {r:.4f}")
    bas_shift = basic[["video_id", "video_duration"]].copy()
    bas_shift["video_id"] = bas_shift["video_id"] - 1
    ms = cap[[ID, "duration"]].merge(bas_shift, left_on=ID, right_on="video_id", how="inner").dropna()
    exact_shift = float((ms["duration"] == ms["video_duration"]).mean())
    say(f"   id+1 shift control: exact match {pct(exact_shift)} over {len(ms):,} videos")

    logs = pd.concat([pd.read_csv(p, usecols=["video_id", "duration_ms"]) for p in LOGS], ignore_index=True)
    log_dur = logs.groupby("video_id")["duration_ms"].agg(lambda s: s.mode().iloc[0]).rename("log_duration")
    ml = cap[[ID, "duration"]].merge(log_dur, left_on=ID, right_index=True, how="inner").dropna()
    exact_log = float((ml["duration"] == ml["log_duration"]).mean())
    within1s_log = float(((ml["duration"] - ml["log_duration"]).abs() <= 1000).mean())
    say(f"captions.duration vs raw-log duration_ms (per-video mode over the 3 Pure logs): {len(ml):,} videos; "
        f"exact {pct(exact_log)}; |diff|<=1s {pct(within1s_log)}")

    mt = fl[[ID, "first_level_category_id"]].merge(basic[["video_id", "tag"]], left_on=ID,
                                                   right_on="video_id", how="inner")
    mt = mt[mt["tag"].notna()]
    toks = mt["tag"].astype(str).str.split(",")
    first_int = mt["first_level_category_id"].astype(int).astype(str)
    agree = np.array([f in [t.strip() for t in tt] for f, tt in zip(first_int, toks)])
    agree_primary = float((first_int.to_numpy() == toks.str[0].str.strip().to_numpy()).mean())
    null_agree = []
    for _ in range(N_PERM):
        pf = rng.permutation(first_int.to_numpy())
        null_agree.append(np.mean([f in [t.strip() for t in tt] for f, tt in zip(pf, toks)]))
    say(f"first_level_category_id in KuaiRand tag tokens: {len(mt):,} videos; agreement {pct(agree.mean())} "
        f"(equals primary tag {pct(agree_primary)}); permuted null {pct(np.mean(null_agree))} "
        f"+/- {100 * np.std(null_agree, ddof=1):.2f} pp")
    summary["attributes"] = dict(duration_vs_basic=dict(n=len(d), exact=exact, within_1s=within1s, pearson_r=r,
                                                       exact_id_shift_control=exact_shift),
                                 duration_vs_log=dict(n=len(ml), exact=exact_log, within_1s=within1s_log),
                                 first_level_vs_tag=dict(n=len(mt), agree_any_token=float(agree.mean()),
                                                         agree_primary=agree_primary,
                                                         null_mean=float(np.mean(null_agree))))
    say("")
    say(f"total audit time {time.time() - t_start:.0f}s")
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    say(f"wrote {OUT_TXT.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
