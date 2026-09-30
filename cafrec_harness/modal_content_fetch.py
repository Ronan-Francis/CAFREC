"""Fetch the KuaiRand supplementary content files and write the Pure-item
subsets, entirely on Modal (2026-09-21).

Why this exists
---------------
`audit_content_data.py` consumed two multi-GB CSVs that were downloaded onto a
VM that is no longer reachable, and neither the CSVs nor the
`data/content_pure/` subsets they produced are on this machine or on the
`cafrec-data` volume. The files themselves are public: the paper already cites
Zenodo record 18159199 (04_experimental_setup.tex), which holds both.

Downloading 6.9 GB over a 9 Mbps domestic link takes ~1h45m; downloading them
*inside* Modal takes minutes, and only the ~2 MB of Pure-item subsets ever has
to come back. So this module runs the download AND the subsetting remotely and
leaves `content_pure/{categories,captions}_pure.csv` on the volume.

What it reproduces
------------------
Steps 1a/1b of `audit_content_data.py`: chunked scan keyed on `final_video_id`,
keep the rows whose id appears in `<dataset>.inter`, `drop_duplicates(keep=
"first")`, write UTF-8 CSV. Same filter, same dedup, same column set, so the
subsets are byte-comparable with the 2026-09-17 ones. The audit's step-4
alignment checks are NOT repeated: they need KuaiRand-Pure's raw logs, they
passed on 2026-09-17, and they are already reported in the paper.

Instead this prints the three coverage figures the audit reported, which is
enough to confirm the Zenodo copy is the same data as the VM copy:
    first-level category 99.83%   second-level 85.46%   non-empty caption 98.90%

    modal run modal_content_fetch.py                      # pure, purge raw after
    modal run modal_content_fetch.py --keep-raw           # leave the CSVs on /data
    modal run modal_content_fetch.py --datasets kuairand_pure,kuairand_1k_kcore

Then pull the subsets down (~2 MB):
    python -m modal volume get cafrec-data content_pure ../data/content_pure
"""
import json
import time

import modal

app = modal.App("cafrec-content-fetch")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "numpy<2", "requests")
)

data_vol = modal.Volume.from_name("cafrec-data", create_if_missing=True)

# Zenodo 10.5281/zenodo.18159199, "Kuairand Supplementary Files: Video Captions
# and Categories" (published 2026-01-06). MD5s are the record's own checksums;
# the download is rejected if they do not match, so a truncated or substituted
# file can never reach the subsetting step.
RECORD = "18159199"
FILES = {
    "kuairand_video_categories.csv": "8a4c772d3d8f8cf65fa1e1eb896f4177",
    "kuairand_video_captions.csv": "9fd4e7587ffd0c8dc289d013858fb229",
}
URL = "https://zenodo.org/records/{rec}/files/{name}?download=1"

ID = "final_video_id"
UNKNOWN_ID = -124.0
CHUNK = 1_000_000
RAW_DIR = "/data/content_raw"


def _subset_dir(dataset):
    """kuairand_pure -> content_pure (what build_content_profiles.py reads)."""
    return "/data/content_pure" if dataset == "kuairand_pure" else f"/data/content_{dataset}"


@app.function(
    image=image,
    # The download is I/O bound and the pandas scan is single-threaded, so the
    # RON-31 lesson applies: reserved CPU+memory is billed alongside, and a big
    # reservation here would buy nothing. 2 cores / 16 GiB keeps this at pennies.
    cpu=2.0,
    memory=16384,
    timeout=3 * 60 * 60,
    volumes={"/data": data_vol},
)
def fetch_and_subset(datasets: str = "kuairand_pure", keep_raw: bool = False,
                     force_download: bool = False) -> dict:
    import hashlib
    import os
    import shutil

    import numpy as np
    import pandas as pd
    import requests

    t_start = time.time()
    names = [d.strip() for d in datasets.split(",") if d.strip()]
    os.makedirs(RAW_DIR, exist_ok=True)
    report = {"record": RECORD, "datasets": names, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------------------------------------------------------- download
    def md5_of(path):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()

    dl = {}
    for name, want in FILES.items():
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path) and not force_download:
            got = md5_of(path)
            if got == want:
                print(f"  {name}: already on the volume, md5 ok", flush=True)
                dl[name] = dict(bytes=os.path.getsize(path), md5=got, cached=True)
                continue
            print(f"  {name}: md5 {got} != {want}, re-downloading", flush=True)

        last_err = None
        for attempt in range(1, 4):
            t0 = time.time()
            h = hashlib.md5()
            n = 0
            try:
                with requests.get(URL.format(rec=RECORD, name=name), stream=True,
                                  timeout=(30, 300)) as r:
                    r.raise_for_status()
                    with open(path, "wb") as fh:
                        for block in r.iter_content(chunk_size=1 << 22):
                            fh.write(block)
                            h.update(block)
                            n += len(block)
                got = h.hexdigest()
                if got != want:
                    raise ValueError(f"md5 {got} != expected {want}")
                sec = time.time() - t0
                print(f"  {name}: {n / 1e9:.2f} GB in {sec:.0f}s "
                      f"({n / 1e6 / max(sec, 1):.0f} MB/s), md5 ok", flush=True)
                dl[name] = dict(bytes=n, md5=got, seconds=round(sec, 1), cached=False)
                break
            except Exception as e:                      # noqa: BLE001 - retried below
                last_err = e
                print(f"  {name}: attempt {attempt} failed: {e}", flush=True)
                if os.path.exists(path):
                    os.remove(path)
        else:
            raise RuntimeError(f"{name}: download failed after 3 attempts: {last_err}")
    data_vol.commit()
    report["download"] = dl

    cat_path = os.path.join(RAW_DIR, "kuairand_video_categories.csv")
    cap_path = os.path.join(RAW_DIR, "kuairand_video_captions.csv")

    # ---------------------------------------------------------------- item ids
    # One scan of each 3+ GB file serves every requested dataset: keep the union
    # of their item ids, then split per dataset afterwards.
    item_sets = {}
    for ds in names:
        inter = f"/data/recbole/{ds}/{ds}.inter"
        if not os.path.exists(inter):
            raise FileNotFoundError(f"{inter} is not on the cafrec-data volume")
        df = pd.read_csv(inter, sep="\t", usecols=[1])
        item_sets[ds] = np.unique(df.iloc[:, 0].to_numpy(np.int64))
        print(f"  {ds}: {len(item_sets[ds]):,} distinct item ids", flush=True)
    union = np.unique(np.concatenate(list(item_sets.values())))

    # ---------------------------------------------------------------- scans
    def scan(path, dtypes, label):
        t0 = time.time()
        n_rows = n_bad = 0
        keep = []
        for chunk in pd.read_csv(path, chunksize=CHUNK, dtype=dtypes):
            num = pd.to_numeric(chunk[ID], errors="coerce")
            n_bad += int(num.isna().sum())
            chunk = chunk.assign(**{ID: num})[num.notna()]
            n_rows += len(chunk)
            vid = chunk[ID].to_numpy(np.int64)
            keep.append(chunk[np.isin(vid, union)])
            if n_rows % (10 * CHUNK) < CHUNK:
                print(f"  {label}: {n_rows:,} rows ({time.time() - t0:.0f}s)", flush=True)
        out = pd.concat(keep, ignore_index=True)
        print(f"  {label}: {n_rows:,} rows scanned in {time.time() - t0:.0f}s; "
              f"{len(out):,} match a catalogue id", flush=True)
        return out, dict(rows=n_rows, bad_id_rows=n_bad, matched_rows=len(out),
                         seconds=round(time.time() - t0, 1))

    cat_all, cat_stats = scan(cat_path, {ID: str}, "categories")
    cap_all, cap_stats = scan(cap_path, {ID: str, "caption": str, "show_cover_text": str},
                              "captions")
    report["categories_file"] = cat_stats
    report["captions_file"] = cap_stats

    # ---------------------------------------------------------------- write
    per_dataset = {}
    for ds in names:
        items = item_sets[ds]
        out_dir = _subset_dir(ds)
        os.makedirs(out_dir, exist_ok=True)

        cat = cat_all[np.isin(cat_all[ID].to_numpy(np.int64), items)]
        cat = cat.drop_duplicates(subset=ID, keep="first").copy()
        cat[ID] = cat[ID].astype(np.int64)
        cap = cap_all[np.isin(cap_all[ID].to_numpy(np.int64), items)]
        cap = cap.drop_duplicates(subset=ID, keep="first").copy()
        cap[ID] = cap[ID].astype(np.int64)

        cat.to_csv(os.path.join(out_dir, "categories_pure.csv"), index=False, encoding="utf-8")
        cap.to_csv(os.path.join(out_dir, "captions_pure.csv"), index=False, encoding="utf-8")

        # the three figures the 2026-09-17 audit reported, as an identity check
        cov = {}
        for lvl in ("first", "second"):
            cid = cat[f"{lvl}_level_category_id"]
            known = set(cat.loc[cid.notna() & (cid != UNKNOWN_ID), ID].tolist())
            cov[f"{lvl}_level_category"] = float(np.mean([int(i) in known for i in items]))
        nonempty = set(cap.loc[cap["caption"].fillna("").str.strip() != "", ID].tolist())
        cov["nonempty_caption"] = float(np.mean([int(i) in nonempty for i in items]))
        per_dataset[ds] = dict(items=len(items), categories_rows=len(cat),
                               captions_rows=len(cap), coverage=cov, dir=out_dir)
        print(f"  {ds}: wrote {out_dir}; first-level {100 * cov['first_level_category']:.2f}%, "
              f"second-level {100 * cov['second_level_category']:.2f}%, "
              f"caption {100 * cov['nonempty_caption']:.2f}%", flush=True)
    report["subsets"] = per_dataset

    # ---------------------------------------------------------------- provenance
    prov = dict(
        source="Zenodo record 10.5281/zenodo.18159199 — Kuairand Supplementary "
               "Files: Video Captions and Categories",
        urls={n: URL.format(rec=RECORD, name=n) for n in FILES},
        md5=FILES, fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        script="modal_content_fetch.py", report=report)
    for ds in names:
        with open(os.path.join(_subset_dir(ds), "PROVENANCE.json"), "w") as fh:
            json.dump(prov, fh, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- cleanup
    if keep_raw:
        report["raw"] = "kept on /data/content_raw"
    else:
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        report["raw"] = "purged (re-fetch with `modal run modal_content_fetch.py`)"
    data_vol.commit()
    report["total_seconds"] = round(time.time() - t_start, 1)
    return report


@app.local_entrypoint()
def main(datasets: str = "kuairand_pure", keep_raw: bool = False,
         force_download: bool = False):
    rep = fetch_and_subset.remote(datasets=datasets, keep_raw=keep_raw,
                                  force_download=force_download)
    print("\n--- content fetch (Modal) ---")
    print(json.dumps(rep, indent=2))
    print("\nExpected from the 2026-09-17 audit, for kuairand_pure:")
    print("  first-level category 99.83%   second-level 85.46%   non-empty caption 98.90%")
    print("\nNext:")
    print("  python -m modal volume get cafrec-data content_pure ../data/content_pure")
