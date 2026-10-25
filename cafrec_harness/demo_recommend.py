"""Per-user inference demo — what the models actually recommend, and what the gate does.

The analysis scripts report aggregates. This one opens up a handful of individual
users so the pipeline is visible end to end: the causal context the gate reads, the
held-out target, where each model ranked it, and the top-10 each model served.

Two sections, each labelled with the run set it comes from, because the dumps that
carry top-k lists differ between the headline and ablation seed sets:

  H1  SASRec / CAFREC-NP / CAFREC over the 7 headline seeds at batch 512
      (results/modal, the same files analyze_plan_hypotheses.py and
      audit_paper_claims.py read). SASRec's dumps carry top-k, CAFREC's carry ranks.
  H3  the fusion comparison (CAFREC against a static gate and against plain
      concatenation) and the context controls (CAFREC-NP against a shuffled context
      and against a context-free vector gate) over the 3 ablation seeds. The two
      groups keep the paper's pairing: the controls zero z_long, so they are paired
      against CAFREC-NP rather than against CAFREC. All carry top-k, so the
      recommendation lists are comparable within a group.

The three users shown by default are picked by a stated rule, not by hand: the user
CAFREC-NP helps most, the user it hurts most, and a user where the two models tie.
That spread is the point — per-user ranks move a great deal while the aggregate does
not, which is what H1's null result looks like from the inside.

    .venv/Scripts/python demo_recommend.py
    .venv/Scripts/python demo_recommend.py --users 10491 2839
    .venv/Scripts/python demo_recommend.py --live-gate

Run from cafrec_harness/. CPU only. Reads the stored dumps; `--live-gate` additionally
loads a local checkpoint and re-scores it, which takes about a minute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_plan_hypotheses as aph  # noqa: E402  (needs HERE on the path first)

H1_MODELS = ["SASRec", "CAFREC-NP", "CAFREC"]
# Two ablation groups, each with its own reference model first. Keeping them apart
# matters: the controls force z_long := 0, so they belong against CAFREC-NP.
FUSION_MODELS = ["CAFREC [abl]", "Static gate", "Concatenation"]
CTRL_MODELS = ["CAFREC-NP [abl]", "Shuffled ctx", "Vector gate"]

# Local checkpoint used by --live-gate: the CAFREC-NP run whose gate stayed alive.
# The batch-512 local checkpoints trained their gate to ~0 (with z_long zeroed, g = 0
# recovers z_short exactly, so collapsing is the cheapest thing the gate can do), and
# a gate pinned at zero makes every intervention a no-op. results/gate_export_*.txt
# has the same statistics for all seven local checkpoints.
LIVE_CKPT = os.path.join(HERE, "saved", "CAFREC-Sep-15-2026_22-08-38.pth")

K = 10


# --------------------------------------------------------------------------- #
#  stored dumps
# --------------------------------------------------------------------------- #
def rank_matrix(name, seeds, index):
    """(ranks [n_seeds, n_users], {seed: {user_token: [top-k item tokens]}})."""
    ranks = np.full((len(seeds), len(index)), np.nan)
    topk = {}
    for j, s in enumerate(seeds):
        d = aph._load_json(name, s)
        for u, r in zip(d["test_user_ids"], d["test_ranks"]):
            i = index.get(str(u))
            if i is not None and r is not None:
                ranks[j, i] = r
        if "test_topk_items" in d:
            topk[s] = dict(zip((str(u) for u in d["test_topk_user_ids"]),
                               d["test_topk_items"]))
    return ranks, topk


def ndcg(ranks):
    return np.where(ranks <= K, 1.0 / np.log2(ranks + 1.0), 0.0)


def metrics(ranks):
    return {f"hit@{K}": float((ranks <= K).mean()), f"ndcg@{K}": float(ndcg(ranks).mean())}


# --------------------------------------------------------------------------- #
#  per-user context, straight from the feature-augmented .inter
# --------------------------------------------------------------------------- #
def user_detail(users):
    """Raw (de-z-scored) context of each user's held-out row, plus their history.

    The held-out row is the last interaction in time order, so its context columns
    are exactly the x_ctx the gate saw at test time. The scaler is inverted with the
    train-fit parameters the feature builder stored.
    """
    path = os.path.join(aph.DATA, "kuairand_pure_ctx", "kuairand_pure_ctx.inter")
    ctx = pd.read_csv(path, sep="\t", dtype={"user_id:token": str})
    ctx.columns = [c.split(":")[0] for c in ctx.columns]
    ctx = ctx.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)

    uid = ctx["user_id"].to_numpy()
    item = ctx["item_id"].to_numpy()
    end = np.flatnonzero(np.r_[uid[1:] != uid[:-1], True])      # last row of each user
    start = np.r_[0, end[:-1] + 1]
    row_of = {uid[e]: (s, e) for s, e in zip(start, end)}

    sc = json.load(open(aph.SCALER))
    mu = dict(zip(sc["features"], sc["mean"]))
    sd = dict(zip(sc["features"], sc["std"]))

    out = {}
    for u in users:
        if u not in row_of:
            continue
        s, e = row_of[u]
        r = ctx.iloc[e]
        raw = {k: float(r[k + "_z"]) * sd[k] + mu[k] for k in sc["features"]}
        out[u] = {
            "n_rows": int(e - s + 1),
            "target": str(item[e]),
            "history": [str(x) for x in item[max(s, e - 5):e]],
            "session_len": int(round(np.expm1(raw["prefix_session_len_log"]))),
            "dwell_entropy": raw["prefix_dwell_entropy"],
            "category_drift": raw["prefix_category_drift"],
            "gap_minutes": float(np.expm1(raw["inter_session_gap_log"]) / 60000.0),
            "policy_flag": int(r["prefix_policy_flag"]),
            "first_session": int(r["is_first_session"]),
            "z": [float(r[k + "_z"]) for k in sc["features"]]
                 + [float(r["prefix_policy_flag"]), float(r["is_first_session"])],
        }
    return out


# --------------------------------------------------------------------------- #
#  user selection
# --------------------------------------------------------------------------- #
def pick_users(index, h1_ranks, n):
    """Stated rule, so the choice is reproducible and not hand-picked.

    Score each user by CAFREC-NP's mean NDCG@10 advantage over SASRec across the 7
    headline seeds, then take the best, the worst, and a tie, in that order.
    """
    tokens = np.array(list(index.keys()))
    order = np.array([index[t] for t in tokens])
    tokens = tokens[np.argsort(order)]

    d = ndcg(h1_ranks["CAFREC-NP"]).mean(0) - ndcg(h1_ranks["SASRec"]).mean(0)
    hit_np = (h1_ranks["CAFREC-NP"] <= K).sum(0)
    hit_sr = (h1_ranks["SASRec"] <= K).sum(0)

    n_seeds = h1_ranks["SASRec"].shape[0]
    chosen = []
    best = np.argsort(-d)
    chosen.append((tokens[best[0]], "CAFREC-NP gains most over SASRec here"))
    worst = np.argsort(d)
    chosen.append((tokens[worst[0]], "SASRec gains most over CAFREC-NP here"))
    # A tie at rank 1 on every seed is usually a repeat view of the same item, which
    # is trivially easy and makes a dull example; require the target to sit below the
    # top slot so the tie says something about the models.
    med = np.median(h1_ranks["SASRec"], axis=0)
    tie = np.flatnonzero((d == 0) & (hit_np == n_seeds) & (hit_sr == n_seeds) & (med >= 2))
    if len(tie):
        chosen.append((tokens[tie[0]], "both models hit on every seed, same NDCG"))

    # pad with further large-advantage users if more were asked for
    for i in best:
        if len(chosen) >= n:
            break
        t = tokens[i]
        if t not in [c[0] for c in chosen]:
            chosen.append((t, "next largest CAFREC-NP advantage"))
    return chosen[:n]


# --------------------------------------------------------------------------- #
#  live gate (optional)
# --------------------------------------------------------------------------- #
def live_gate(tokens, verbose=True):
    """Load a trained checkpoint and re-score the test set under four gate conditions.

    The gate is fed its real context, a context permuted across users, the test-set
    mean gate (a constant), and g := 0. The weights never change, so any movement in
    the ranking is the gate's doing.

    The loading, gate export and scoring helpers are analyze_gate.py's, so this is the
    same code path the thesis used - only the reporting is per user rather than
    aggregate. `analyze_gate.py <checkpoint>` runs the full version with paired tests.

    The checkpoint is the no_profiler ablation (CAFREC-NP), where z_long is zeroed, so
    the fusion reduces to z = (1 - g) * z_short: a context-conditioned rescaling of the
    session vector. That is the variant the paper reports.
    """
    import warnings

    import torch

    import analyze_gate as ag

    torch.set_num_threads(4)
    t0 = time.time()
    if verbose:
        print(f"  loading {os.path.basename(LIVE_CKPT)} and rebuilding its test split "
              f"(about a minute) ...", flush=True)
    # RecBole 1.2.1 fills missing feature values with pandas' chained-assignment idiom,
    # which emits two screenfuls of FutureWarning on every create_dataset. Nothing here
    # depends on it. Silenced for this call only, so the analysis scripts still show it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        config, dataset, test_data, model = ag._load(LIVE_CKPT)
    fields = list(config["context_fields"])
    if verbose:
        print(f"  ready in {time.time() - t0:.0f}s  (seed {config['seed']}, batch "
              f"{config['train_batch_size']}, ablation {config['ablation']})", flush=True)
        print("  exporting the gate for every test user ...", flush=True)

    g, x, cuids = ag._collect(model, test_data)
    # prefix length 0 <=> the smallest z-scored prefix_session_len, i.e. the held-out
    # interaction opens a new session (same rule as analyze_gate.py)
    pz = x[:, fields.index("prefix_session_len_log_z")]
    opener = np.isclose(pz, pz.min())
    stats = {
        "all": ag._gate_stats(g, x, fields),
        "opener": ag._gate_stats(g[opener], x[opener], fields),
        "continuing": ag._gate_stats(g[~opener], x[~opener], fields),
    }

    gbar = torch.tensor(g.mean(0))
    gen = torch.Generator().manual_seed(0)

    def real(self, context, ref):
        return self.gating_mlp(context)

    def shuffled(self, context, ref):
        return self.gating_mlp(context[torch.randperm(context.size(0), generator=gen)])

    def constant(self, context, ref):
        return gbar.expand(ref.size(0), -1)

    def zero(self, context, ref):
        return torch.zeros_like(ref)

    conds = (("real context", real), ("shuffled context", shuffled),
             ("constant gate", constant), ("g := 0 (z_short only)", zero))
    ranks, aggregate = {}, {}
    for name, fn in conds:
        if verbose:
            print(f"  re-scoring under: {name} ...", flush=True)
        uids, r = ag._score(model, test_data, dataset, fn)
        ranks[name] = r
        aggregate[name] = metrics(r)

    # ranks come back sorted by internal user id; the gate export is in eval order
    uid_field = dataset.uid_field
    pos = {int(u): i for i, u in enumerate(uids)}
    gpos = {int(u): i for i, u in enumerate(cuids)}
    per_user = {}
    for t in tokens:
        try:
            internal = int(dataset.token2id(uid_field, t))
        except ValueError:
            continue                                # user filtered out of this build
        i, gi = pos.get(internal), gpos.get(internal)
        if i is None or gi is None:
            continue
        gu = g[gi]
        per_user[t] = {
            "g_mean": float(gu.mean()), "g_min": float(gu.min()),
            "g_max": float(gu.max()), "g_dims": int(gu.size),
            "opener": bool(opener[gi]),
            "ranks": {n: int(ranks[n][i]) for n, _ in conds},
        }
    return per_user, stats, aggregate, config, fields


# --------------------------------------------------------------------------- #
#  printing
# --------------------------------------------------------------------------- #
def fmt_ranks(ranks_row):
    return "  ".join(f"{int(r):>5d}" for r in ranks_row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--users", nargs="+", help="KuaiRand user tokens to show")
    ap.add_argument("-n", type=int, default=3, help="how many users to pick (default 3)")
    ap.add_argument("--live-gate", action="store_true",
                    help="also load a local checkpoint and re-score under gate edits")
    args = ap.parse_args(argv)

    t0 = time.time()
    probe = aph._load_json("SASRec", aph.HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    index = {u: i for i, u in enumerate(users)}

    print("CAFREC - per-user inference demo")
    print(f"KuaiRand-Pure, leave-one-out, FULL ranking against all "
          f"{probe['n_items_catalog']:,} items; {len(users):,} test users")
    print(f"Reading the stored run dumps in {os.path.relpath(aph.RES, HERE)} - the same "
          f"files the analysis\nand the paper claim audit read.\n")

    h1_ranks, h1_topk = {}, {}
    for m in H1_MODELS:
        h1_ranks[m], h1_topk[m] = rank_matrix(m, aph.HEADLINE, index)
    gate_ranks, gate_topk = {}, {}
    for m in FUSION_MODELS + CTRL_MODELS:
        gate_ranks[m], gate_topk[m] = rank_matrix(m, aph.ABLATION, index)

    if args.users:
        chosen = [(u, "requested") for u in args.users if u in index]
        missing = [u for u in args.users if u not in index]
        for u in missing:
            print(f"  (user {u} is not in the test split - skipped)")
    else:
        chosen = pick_users(index, h1_ranks, args.n)
        print("User selection rule: rank every test user by CAFREC-NP's mean NDCG@10")
        print("advantage over SASRec across the 7 headline seeds, then take the best,")
        print("the worst, and a tie. No user is hand-picked.\n")

    detail = user_detail([u for u, _ in chosen])

    for token, why in chosen:
        i = index[token]
        d = detail.get(token)
        print("=" * 78)
        print(f"User {token}   ({why})")
        if d:
            pos = ("opens a new session" if d["session_len"] == 0
                   else f"{d['session_len']} earlier interactions in this session")
            print(f"  activity        {d['n_rows']} interactions in the log")
            print(f"  session         held-out item {pos}")
            print(f"  context x_ctx   session_len {d['session_len']}   "
                  f"dwell_entropy {d['dwell_entropy']:.2f}   "
                  f"category_drift {d['category_drift']:.2f}")
            print(f"                  gap {d['gap_minutes']:.0f} min   "
                  f"policy_flag {d['policy_flag']}   "
                  f"first_session {d['first_session']}")
            print(f"  session so far  {' '.join(d['history']) or '(none)'}")
            print(f"  held-out target item {d['target']}")
        print()

        print(f"  H1  rank of the target, 7 headline seeds at batch 512 "
              f"(seeds {', '.join(map(str, aph.HEADLINE))})")
        for m in H1_MODELS:
            row = h1_ranks[m][:, i]
            hits = int((row <= K).sum())
            print(f"    {m:<12s} {fmt_ranks(row)}   mean {row.mean():>7.0f}   "
                  f"hit@10 {hits}/{len(row)}")
        for m in H1_MODELS:
            if aph.HEADLINE[0] in h1_topk[m]:
                lst = h1_topk[m][aph.HEADLINE[0]].get(token)
                if lst:
                    print(f"    {m} top-10 (seed {aph.HEADLINE[0]}): {' '.join(lst)}")
        print()

        for label, group in (("H3  fusion form", FUSION_MODELS),
                             ("H3  context controls (z_long := 0)", CTRL_MODELS)):
            print(f"  {label}, 3 ablation seeds "
                  f"(seeds {', '.join(map(str, aph.ABLATION))})")
            for m in group:
                row = gate_ranks[m][:, i]
                hits = int((row <= K).sum())
                print(f"    {m:<16s} {fmt_ranks(row)}   mean {row.mean():>7.0f}   "
                      f"hit@10 {hits}/{len(row)}")
            ref = group[0]
            base = gate_topk[ref].get(aph.ABLATION[0], {}).get(token)
            if base:
                print(f"    {ref:<16s} top-10 (seed {aph.ABLATION[0]}): {' '.join(base)}")
                for m in group[1:]:
                    lst = gate_topk[m].get(aph.ABLATION[0], {}).get(token)
                    if lst:
                        shared = len(set(base) & set(lst))
                        print(f"    {m:<16s} top-10: {' '.join(lst)}   "
                              f"({shared}/10 shared)")
            print()

    print("=" * 78)
    print(f"Aggregate over all {len(users):,} test users - what the paper reports\n")
    print(f"  {'model':<14s} {'hit@10':>9s} {'ndcg@10':>9s}     7 headline seeds, batch 512")
    for m in H1_MODELS:
        r = h1_ranks[m]
        print(f"  {m:<14s} {(r <= K).mean():>9.4f} {ndcg(r).mean():>9.4f}")
    print()
    print(f"  {'model':<16s} {'hit@10':>9s} {'ndcg@10':>9s}     3 ablation seeds")
    for group in (FUSION_MODELS, CTRL_MODELS):
        for m in group:
            r = gate_ranks[m]
            print(f"  {m:<16s} {(r <= K).mean():>9.4f} {ndcg(r).mean():>9.4f}"
                  f"{'     <- reference for the rows below' if m is group[0] else ''}")
    print()

    # how far apart the two models are per user, measured rather than asserted
    a, b = h1_ranks["CAFREC-NP"].mean(0), h1_ranks["SASRec"].mean(0)
    gap = np.abs(a - b)
    agg = ndcg(h1_ranks["CAFREC-NP"]).mean() - ndcg(h1_ranks["SASRec"]).mean()
    print(f"  CAFREC-NP vs SASRec, per user: the mean rank differs for "
          f"{(gap > 0.5).mean():.0%} of users,")
    print(f"  by a median of {np.median(gap[gap > 0.5]):.0f} places and more than 100 "
          f"places for {(gap > 100).mean():.0%}.")
    print(f"  The aggregate NDCG@10 difference is {agg:+.4f}. Movement at the user "
          f"level does not")
    print("  add up to movement in the mean - that is what H1's null result looks like.")
    print("  analyze_plan_hypotheses.py runs the paired tests and the Holm correction.")

    if args.live_gate:
        print()
        print("=" * 78)
        print("Live gate - one trained checkpoint, four gate conditions, same weights\n")
        res, stats, agg, config, fields = live_gate([u for u, _ in chosen])
        print(f"\n  checkpoint      {os.path.basename(LIVE_CKPT)}")
        print(f"  ablation        {config['ablation']} - z_long is zeroed, so the fusion")
        print( "                  reduces to z = (1 - g) * z_short")
        print(f"  note            a separate run from the dumps above "
              f"(seed {config['seed']}, batch")
        print( "                  2048), so these ranks are not comparable to them\n")

        print("  What the gate does with the context it is given")
        for name in ("all", "opener", "continuing"):
            s = stats[name]
            print(f"    {name:<11s} mean g {s['mean_g']:.3f}   "
                  f"per-dim means {s['dim_mean_min']:.3f}-{s['dim_mean_max']:.3f}   "
                  f"across-user sd {s['across_user_sd_mean']:.4f}")
        r2 = stats["all"]["r2_by_feature"]
        print(f"    linear R^2 of g on all {len(fields)} context features: "
              f"{stats['all']['r2_all_features_linear']:.3f}")
        for f, v in sorted(r2.items(), key=lambda kv: -kv[1])[:3]:
            print(f"      {f:<28s} {v:.3f}")
        print()

        print("  Aggregate over all test users, by gate condition")
        base = agg["real context"]
        for name in agg:
            m = agg[name]
            d = m["ndcg@10"] - base["ndcg@10"]
            tail = "" if name == "real context" else \
                f"   dNDCG {d:+.5f} ({100 * d / base['ndcg@10']:+.2f}%)"
            print(f"    {name:<24s} hit@10 {m['hit@10']:.4f}  "
                  f"ndcg@10 {m['ndcg@10']:.4f}{tail}")
        print()

        for token, _ in chosen:
            r = res.get(token)
            if not r:
                print(f"  User {token}: not in this build's test split\n")
                continue
            where = "session opener" if r["opener"] else "continuing session"
            print(f"  User {token}  ({where})")
            print(f"    gate g over {r['g_dims']} dimensions: "
                  f"mean {r['g_mean']:.3f}  min {r['g_min']:.3f}  max {r['g_max']:.3f}")
            moved = len(set(r["ranks"].values())) > 1
            for name, rank in r["ranks"].items():
                print(f"    {name:<24s} target rank {rank:>6d}")
            if not moved:
                print("    (this user's rank is unchanged by the gate edits; only a")
                print("     minority of users move - see the aggregate above)")
            print()
        print("  analyze_gate.py runs this over every local checkpoint with paired")
        print("  Wilcoxon tests and writes results/gate_export_<date>.{txt,json}.")

    print(f"\nwall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
