"""T9 (TODO.md): batch-size manifest, one row per reported table row. Fills E2.

Writes cafrec_harness/results/MANIFEST.csv (author's choice Q12-A, 2026-09-30) with the columns
TODO.md specifies, plus `batch_size_source`, which says whether the batch size was read from
the run's resolved `config` block or established from the versioned launch code:

  result_file, paper_table, paper_row, commit, launch_script, batch_size, seeds, device, date,
  batch_size_source

The row -> run-family mapping follows the analysis scripts that produced each table
(analyze_plan_hypotheses.py, analyze_followups.py, analyze_paper_512.py, analyze_ild.py,
analyze_content_profiles.py, analyze_tuned_gpu.py) and, for Table VIII, t9_check_profile_rows.py.
`commit` is the commit that added each result file; `date` is the run date from the file name
(or the JSON timestamp for results/local_gpu and results/local).

Launch-code rule for runs without a config block: modal_run.py's `train` set
train_batch_size=512 for every dataset other than "kuairand_pure" when no batch size was
passed (branch at modal_run.py line 72 from commit 5f8f956, 2026-07-23, until b478590,
2026-09-16 07:08). Every *_ctx run below without a config block ran in that window.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO = os.path.dirname(HARNESS)
OUT = os.path.join(HARNESS, "results", "MANIFEST.csv")

HEAD = [42, 77, 123, 256, 512, 1024, 2048]
ABL = [2020, 2021, 403092]
A10G, GPU2060, CPU = "Modal A10G", "NVIDIA RTX 2060 SUPER (home PC)", "CPU (labBL6O0K, Xeon 8272CL)"
BRANCH = "launch code: modal_run.py L72 branch (5f8f956..b478590) sets 512 for *_ctx"

# family: (dir, [patterns tried in order], seeds, launch_script, device, inferred batch, inferred source)
F = {
    "sasrec512": ("modal", ["SASRec_kuairand_pure_bs512_sasrec_seed{s}_*.json"], HEAD,
                  "modal_run.py::bsweep", A10G, 512, "launch code: bsweep passes train_batch_size=512"),
    "np_ms": ("modal", ["CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json"], HEAD,
              "run_manyseed.py (modal_run.train)", A10G, 512, BRANCH),
    "bge_ms": ("modal", ["CAFREC_kuairand_pure_ctx_bge_ms_s{s}_seed{s}_*.json"], HEAD,
               "run_manyseed.py (modal_run.train)", A10G, 512, BRANCH),
    "sasrec_tuned": ("local_gpu", ["SASRec_kuairand_pure_tuned16_sasrec_gs*_seed{s}.json"], HEAD,
                     "local_run.py --queue gpu_tuned (gpu_chain.py)", GPU2060, None, None),
    "np_tuned": ("local_gpu", ["CAFREC_kuairand_pure_ctx_tuned16_np_gs*_seed{s}.json"], HEAD,
                 "local_run.py --queue gpu_tuned (gpu_chain.py)", GPU2060, None, None),
    "hgn_ce": ("modal", ["HGN_kuairand_pure_bs512_ce_hgn_seed{s}_*.json"], HEAD,
               "modal_run.py::basesweep --loss ce", A10G, 512, "launch code: _fan_out passes 512"),
    "ugru_ce": ("modal", ["HGRU4Rec_kuairand_pure_bs512_ce_hgru4rec_seed{s}_*.json"], HEAD,
                "modal_run.py::basesweep --loss ce", A10G, 512, "launch code: _fan_out passes 512"),
    "hgn_bpr": ("modal", ["HGN_kuairand_pure_bs512_hgn_seed{s}_*.json"], HEAD,
                "modal_run.py::basesweep", A10G, 512, "launch code: _fan_out passes 512"),
    "ugru_bpr": ("modal", ["HGRU4Rec_kuairand_pure_bs512_hgru4rec_seed{s}_*.json"], HEAD,
                 "modal_run.py::basesweep", A10G, 512, "launch code: _fan_out passes 512"),
    "bge_topk": ("modal", ["CAFREC_kuairand_pure_ctx_bge_topk_s{s}_seed{s}_*.json"], ABL,
                 "modal_run.py::main CLI, --dump-ranks --dump-topk (command not logged)", A10G, 512, BRANCH),
    "np_topk": ("modal", ["CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_seed{s}_*.json"], ABL,
                "modal_run.py::main CLI, --dump-ranks --dump-topk (command not logged)", A10G, 512, BRANCH),
    "static": ("modal", ["CAFREC_kuairand_pure_ctx_static_gate_s{s}_seed{s}_*.json"], ABL,
               "run_parallel_pure.py (modal_run.train)", A10G, 512, BRANCH),
    "concat": ("modal", ["CAFREC_kuairand_pure_ctx_concat_s{s}_seed{s}_*.json"], ABL,
               "run_parallel_pure.py (modal_run.train)", A10G, 512, BRANCH),
    "vector": ("modal", ["CAFREC_kuairand_pure_ctx_vector_gate_s{s}_seed{s}_*.json"], ABL,
               "modal_run.py::gatectl", A10G, 512, "launch code: _fan_out passes 512"),
    "shuffled3": ("modal", ["CAFREC_kuairand_pure_ctx_shuffled_ctx_s{s}_seed{s}_*.json"], ABL,
                  "modal_run.py::gatectl", A10G, 512, "launch code: _fan_out passes 512"),
    "prof7b": ("modal", ["CAFREC_kuairand_pure_ctx_prof7b_s{s}_seed{s}_*.json",
                         "CAFREC_kuairand_pure_ctx_prof_7b_seed{s}_*.json"], ABL,
               "run_parallel_pure.py (s2021, s403092); modal_run.py::main CLI (s2020, 2026-08-14, command not logged)",
               A10G, 512, BRANCH),
    "hash": ("modal", ["CAFREC_kuairand_pure_ctx_hash_s{s}_seed{s}_*.json"], ABL,
             "modal_run.py::hashsweep", A10G, 512, "launch code: _fan_out passes 512"),
    "standin": ("modal", ["CAFREC_kuairand_pure_ctx_standin_s{s}_seed{s}_*.json",
                          "CAFREC_kuairand_pure_ctx_standin_seed{s}_*.json"], ABL,
                "run_parallel_pure.py (s2021, s403092); modal_run.py::main CLI (s2020, 2026-08-14, command not logged)",
                A10G, 512, BRANCH),
    "np512": ("local", ["CAFREC_kuairand_pure_ctx_np512_seed{s}.json"], ABL,
              "local_run.py --queue content_profiles", CPU, None, None),
    "catbeyond": ("local", ["CAFREC_kuairand_pure_ctx_catbeyond512_seed{s}.json"], ABL,
                  "local_run.py --queue content_profiles", CPU, None, None),
    "capbeyond": ("local", ["CAFREC_kuairand_pure_ctx_capbeyond512_seed{s}.json"], ABL,
                  "local_run.py --queue content_profiles", CPU, None, None),
    "capall": ("local", ["CAFREC_kuairand_pure_ctx_capall512_seed{s}.json"], ABL,
               "local_run.py --queue content_profiles", CPU, None, None),
    # Table XIII MostPop row since 2026-09-30: true-count popularity, no training, no batch size
    # (replaces RecBole Pop, results/local/Pop_kuairand_pure_pop_seed2020.json, which counted
    # batch appearances; AUDIT.md, "MostPop row replaced").
    "pop_true": ("todo_runs/MostPop", ["mostpop_results.json"], ["n/a"],
                 "results/todo_runs/MostPop/mostpop_true_count.py", CPU, None, None),
}

T = {
    "IV": "Table IV (tab:res:main)", "V": "Table V (tab:res:tests)", "VI": "Table VI (tab:res:seeds)",
    "VII": "Table VII (tab:res:tuneddiff)", "VIII": "Table VIII (tab:res:profile)",
    "IX": "Table IX (tab:res:cohort_effect)", "X": "Table X (tab:res:content)",
    "XI": "Table XI (tab:res:fusion)", "XII": "Table XII (tab:res:h3strata)",
    "XIII": "Table XIII (tab:res:coverage)", "XIV": "Table XIV (tab:res:h4)",
}

ROWS = [
    ("IV", "SASRec (CE)", ["sasrec512"]), ("IV", "CAFREC-NP (CE)", ["np_ms"]),
    ("IV", "CAFREC (CE)", ["bge_ms"]), ("IV", "SASRec, tuned", ["sasrec_tuned"]),
    ("IV", "CAFREC-NP, tuned", ["np_tuned"]), ("IV", "HGN (CE)", ["hgn_ce"]),
    ("IV", "U-GRU (CE)", ["ugru_ce"]), ("IV", "HGN (BPR)", ["hgn_bpr"]), ("IV", "U-GRU (BPR)", ["ugru_bpr"]),
]
ROWS += [("V", f"{c} vs SASRec, {m}", [fam, "sasrec512"])
         for c, fam in (("CAFREC-NP", "np_ms"), ("CAFREC", "bge_ms")) for m in ("HR@10", "NDCG@10", "MRR@10")]
ROWS += [("VI", f"Seed {s}", [("sasrec512", [s]), ("np_ms", [s])]) for s in HEAD]
ROWS += [("VI", "Mean", ["sasrec512", "np_ms"]), ("VI", "SD", ["sasrec512", "np_ms"])]
ROWS += [("VII", f"{c}, {m}", ["np_tuned", "sasrec_tuned"])
         for c in ("All users", "Opener", "Within") for m in ("HR@10", "NDCG@10")]
ROWS += [("VIII", f"{c}, {m}", fams)
         for c, fams in (("CAFREC vs CAFREC-NP (7 seeds)", ["bge_ms", "np_ms"]),
                         ("e5-mistral-7B vs bge-large profile (3 seeds)", ["prof7b", "bge_topk"]),
                         ("Hashing vs bge-large profile (3 seeds)", ["hash", "bge_topk"]),
                         ("Frozen bge profile vs trainable stand-in (3 seeds)", ["bge_topk", "standin"]))
         for m in ("HR@10", "NDCG@10")]
ROWS += [("IX", f"{c}, {m}", fams)
         for c, fams in (("CAFREC vs CAFREC-NP", ["bge_ms", "np_ms"]),
                         ("CAFREC-NP vs SASRec", ["np_ms", "sasrec512"]))
         for m in ("HR@10", "NDCG@10")]
ROWS += [("X", f"Dense, {p}, {m}", [fam, "np512"])
         for p, fam in (("cat_beyond", "catbeyond"), ("cap_beyond", "capbeyond"), ("cap_all", "capall"))
         for m in ("HR@10", "NDCG@10")]
ROWS += [("X", f"{c}, {p}, NDCG@10", [fam, "np512"])
         for c in ("All", "Q1")
         for p, fam in (("cat_beyond", "catbeyond"), ("cap_beyond", "capbeyond"), ("cap_all", "capall"))]
ROWS += [("XI", "CAFREC (context gate)", ["bge_topk"]), ("XI", "Static scalar gate", ["static", "bge_topk"]),
         ("XI", "Concatenation", ["concat", "bge_topk"]), ("XI", "CAFREC-NP (reference)", ["np_topk"]),
         ("XI", "Context-free vector gate", ["vector", "np_topk"]),
         ("XI", "Shuffled context", ["shuffled3", "np_topk"])]
ROWS += [("XII", s, ["bge_topk", "static", "concat"])
         for s in ("All users", "Q1 (<=12)", "Q2 (13-22)", "Q3 (23-38)", "Q4 (>38)", "High drift", "Low drift")]
ROWS += [("XIII", lab, [fam]) for lab, fam in (
    ("SASRec", "sasrec512"), ("CAFREC-NP", "np_topk"), ("CAFREC", "bge_topk"), ("Static scalar gate", "static"),
    ("Concatenation", "concat"), ("HGN (CE)", "hgn_ce"), ("U-GRU (CE)", "ugru_ce"), ("HGN (BPR)", "hgn_bpr"),
    ("U-GRU (BPR)", "ugru_bpr"), ("MostPop", "pop_true"))]
ROWS += [("XIV", lab, [fam]) for lab, fam in (
    ("Context gate (CAFREC)", "bge_topk"), ("Static scalar gate", "static"), ("Concatenation", "concat"),
    ("CAFREC-NP", "np_topk"))]


def git(*a):
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


_cache = {}


def run_info(fam, seed):
    key = (fam, seed)
    if key in _cache:
        return _cache[key]
    d, pats, _, launch, device, inf_bs, inf_src = F[fam]
    path = None
    for pat in pats:
        hits = sorted(glob.glob(os.path.join(HARNESS, "results", d, pat.format(s=seed))))
        if hits:
            path = hits[-1]
            break
    if path is None:
        raise FileNotFoundError(f"{fam} seed {seed}")
    j = json.load(open(path))
    rel = os.path.relpath(path, REPO).replace("\\", "/")
    if fam == "pop_true":                       # a query output, not a training run
        info = {"file": rel, "commit": git("log", "--diff-filter=A", "--format=%h", "-1", "--", rel)
                or "untracked", "launch": launch, "device": device, "seed": seed, "bs": "n/a",
                "bs_src": "no training (popularity counted over training rows)",
                "date": j["provenance"]["date"][:10]}
        _cache[key] = info
        return info
    cfg = j.get("config") or {}
    bs = cfg.get("train_batch_size")
    stamp = j.get("timestamp", "")
    info = {
        "file": rel,
        "commit": git("log", "--diff-filter=A", "--format=%h", "-1", "--", rel) or "untracked",
        "launch": launch, "device": device, "seed": seed,
        "bs": bs if bs is not None else inf_bs,
        "bs_src": "config block" if bs is not None else inf_src,
        "date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" if stamp else "",
    }
    assert info["bs"] is not None, (fam, seed)
    _cache[key] = info
    return info


def uniq(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def main():
    rows = []
    for tab, label, fams in ROWS:
        infos = []
        for fam in fams:
            fam, seeds = fam if isinstance(fam, tuple) else (fam, F[fam][2])
            infos += [run_info(fam, s) for s in seeds]
        dates = sorted(i["date"] for i in infos)
        rows.append({
            "result_file": ";".join(i["file"] for i in infos),
            "paper_table": T[tab], "paper_row": label,
            "commit": ";".join(uniq(i["commit"] for i in infos)),
            "launch_script": ";".join(uniq(i["launch"] for i in infos)),
            "batch_size": ";".join(uniq(str(i["bs"]) for i in infos)),
            "seeds": ";".join(str(s) for s in uniq(i["seed"] for i in infos)),
            "device": ";".join(uniq(i["device"] for i in infos)),
            "date": dates[0] if dates[0] == dates[-1] else f"{dates[0]}..{dates[-1]}",
            "batch_size_source": ";".join(uniq(i["bs_src"] for i in infos)),
        })
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    files = sorted({i["file"] for i in _cache.values()})
    summary = {"rows": len(rows), "distinct_result_files": len(files),
               "batch_sizes": sorted({str(i["bs"]) for i in _cache.values()}),
               "from_config": sum(i["bs_src"] == "config block" for i in _cache.values()),
               "from_launch_code": sum(i["bs_src"] != "config block" for i in _cache.values()),
               "untracked_files": [f for f in files if _cache and any(
                   i["file"] == f and i["commit"] == "untracked" for i in _cache.values())],
               "head": git("rev-parse", "HEAD")}
    with open(os.path.join(HERE, "t9_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
