"""Per-user generation prompts for the H2 test, in four variants (2026-09-21).

H2 as registered (Project_Plan_2 / 01_introduction.tex): *LLM-generated temporal
user profiles will achieve higher NDCG@10 than interaction-derived embeddings for
users in the lowest quartile of interaction counts.* No profile evaluated in the
paper is LLM-generated, so H2 is currently reported as untested. This builds the
prompts that let it be tested.

The variants exist because of a limitation of the plain prompt. For a
lowest-quartile user the model receives five category labels and six captions,
and the category histogram is already what `cat_beyond` encodes — so the plain
profile can only re-represent information the harness already has. The variants
are the two documented ways out, and running all four separates them:

  plain    behaviour block + own category mix + own recent captions, asked for a
           summary. THE REGISTERED ARM. Tests re-representation only.
  kar      same inputs, but asked to INFER latent preferences and item attributes
           using the model's own world knowledge rather than summarise. Adds
           information that is in the model's weights, not in the data, so it is
           still "an LLM-generated temporal user profile" and stays inside H2.
           After Xi et al., "Towards Open-World Recommendation with Knowledge
           Augmentation from Large Language Models", RecSys 2024 (arXiv
           2306.10933): reasoning knowledge on preferences + factual knowledge on
           items, elicited by factorisation prompting.
  collab   plain + a block of collaborative evidence: items watched by this
           user's nearest long-history users that the user has not watched, scored
           sum(similarity)/sqrt(popularity). Adds information that is in the data
           but not in this user's own rows. After Wu, Chang et al., "CoRAL:
           Collaborative Retrieval-Augmented Large Language Models Improve
           Long-tail Recommendation", KDD 2024 (arXiv 2403.06447), which likewise
           retrieves user-item interactions, minus its RL retrieval policy.
           An item-item co-occurrence version was tried first and FAILED — it
           returned popularity, not collaboration; see `UserBasedRetrieval` for the
           measurements, and `--collab-retrieval item` to reproduce them.
  support  plain + the category mixes of the k nearest LONG-HISTORY users, as
           in-context examples. Same information class as `collab`, cheaper in
           tokens, and it reveals no item identities.

`collab` and `support` are EXPLORATORY, not the registered arm: a profile built
partly from other users' behaviour is no longer purely this user's temporal
profile, and calling it the registered test would be hypothesis drift.

Leakage
-------
Every variant is built from training rows only (`r >= 2`), and the retrieval
variants additionally exclude each user's OWN held-out valid/test items from
their retrieved set — otherwise a co-occurrence list could name the very item the
user is scored on and the profile could describe it.

Design notes shared by all variants
-----------------------------------
* **Row selection: r >= 2, i.e. every training row.** NOT the r >= 22
  "beyond-window" rule of `build_content_profiles.py`. That rule drops users with
  22 rows or fewer, and the H2 cohort is the lowest activity quartile — 6,240
  users with at most 12 organic clicks. Under the beyond-window rule every single
  one of them would get a zero vector, exactly as `cap_beyond` did, and the test
  would be vacuous. r >= 2 drops RecBole's leave-one-out valid+test rows, so the
  profile is still causal.
* **Behaviour block mirrors the template profile.** Same six context fields, same
  `_bucket` thresholds, same history bucketing as
  `cafrec.features.build_profiles.render_profiles`, imported rather than copied.
* Profiles are written in English and embedded with the same English encoder as
  the template profile, so the comparison is text-vs-text, not encoder-vs-encoder.
  The first 64 generations (2026-09-21) kept Chinese category names in otherwise
  English prose, which would have put Chinese tokens through `bge-large-en-v1.5`
  and reintroduced exactly that confound, so every instruction now carries `GLOSS`:
  English names, Chinese in brackets on first mention.
* `--recent` is the cost lever: prefill dominates generation, and lowering it
  costs the H2 cohort nothing (those users have <= 10 rows, all of them listed).
  All variants must share one setting to stay comparable.

    .venv/Scripts/python build_h2_prompts.py --variant plain
    .venv/Scripts/python build_h2_prompts.py --variant all --recent 20
    .venv/Scripts/python build_h2_prompts.py --variant collab --sample 2 --dry-run

Writes ../data/h2/prompts.<variant>.jsonl (one {user_id, uid, n_train, prompt}
per line, ordered by internal user id) and ../data/h2/prompts_meta.<variant>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cafrec.features.build_profiles import _bucket, load_user_item_remap
from cafrec.features.context import CTX_FIELDS

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATASET = "kuairand_pure_ctx"
INTER = DATA / "recbole" / DATASET / f"{DATASET}.inter"
SUBSET = DATA / "content_pure"
CAT_CSV = SUBSET / "categories_pure.csv"
CAP_CSV = SUBSET / "captions_pure.csv"
OUT_DIR = DATA / "h2"
ID = "final_video_id"
UNKNOWN_ID = -124.0
N_HOLDOUT = 2          # RecBole LS:valid_and_test leave-one-out
VARIANTS = ("plain", "kar", "collab", "support")

SYSTEM = (
    "You write concise, factual user profiles for a video recommender system. "
    "You are given one user's viewing history statistics and the videos they "
    "watched. Video captions are in Chinese; always write the profile in English."
)

# plain / collab / support ask for a summary; kar asks for inference. Everything
# else about the four prompts is identical, so a difference between kar and plain
# is attributable to the instruction alone.
GLOSS = (
    "Give every category, genre and topic name in ENGLISH, with the original "
    "Chinese in brackets on first mention only. "
)

SUMMARISE = (
    "Write a single paragraph of 60 to 100 words describing this user's durable "
    "content preferences and their temporal viewing pattern. State which topics "
    "and genres they return to, how varied their interests are, and how their "
    "session behaviour and return rhythm look. Be specific about content. "
    + GLOSS +
    "Do not mention video identifiers, counts, percentages, or this instruction, "
    "and do not speculate about demographics. Begin with \"This user\"."
)

INFER = (
    "Do not summarise the list. Using your own knowledge of the people, shows, "
    "sports, games, music and internet culture these captions refer to, infer "
    "what this user is actually interested in. Name the specific subjects, "
    "franchises and treatments the captions point to, say what those choices have "
    "in common, and state which neighbouring content they imply the user would "
    "watch but has not watched yet. Then describe their viewing rhythm. Write a "
    "single paragraph of 60 to 100 words. "
    + GLOSS +
    "Do not mention video identifiers, counts, percentages, or this instruction, "
    "and do not speculate about demographics. Begin with \"This user\"."
)

INSTRUCTIONS = {"plain": SUMMARISE, "kar": INFER,
                "collab": SUMMARISE, "support": SUMMARISE}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_content():
    """first-level category name per item, and caption text per item.

    Caption repair is `build_content_profiles.repaired_captions`' rule: 33 Pure
    rows split the caption across the caption / show_cover_text / duration slots,
    and the caption is the non-numeric text fields joined in file order.
    """
    cat = pd.read_csv(CAT_CSV)
    known = cat["first_level_category_id"].notna() & (cat["first_level_category_id"] != UNKNOWN_ID)
    cat_name = dict(zip(cat.loc[known, ID].astype(np.int64),
                        cat.loc[known, "first_level_category_name"].astype(str)))

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
    cap_text = dict(zip(cap[ID].astype(np.int64), text))
    return cat_name, cap_text, int(spill.sum())


def behaviour_block(g, n):
    """The template profile's sentence set, verbatim in structure."""
    means = {f: float(g[f].mean()) for f in CTX_FIELDS if f in g.columns}
    policy_rate = float(g.get("prefix_policy_flag", pd.Series([0.0])).mean())
    cold = float(g.get("is_first_session", pd.Series([0.0])).mean())
    history = "sparse" if n < 10 else ("moderate" if n < 40 else "rich")
    return (
        f"History length: {history} ({n} logged interactions). "
        f"Typical session depth: {_bucket(means.get('prefix_session_len_log_z', 0.0))}. "
        f"Dwell-time entropy: {_bucket(means.get('prefix_dwell_entropy_z', 0.0))} "
        f"(viewing-attention variability). "
        f"Category drift: {_bucket(means.get('prefix_category_drift_z', 0.0))} "
        f"(tendency to switch content categories within a session). "
        f"Return-gap between sessions: {_bucket(means.get('inter_session_gap_log_z', 0.0))}. "
        f"Policy/surface-shift exposure: {policy_rate:.0%} of context. "
        f"Cold-start signal: {cold:.0%} first-session activity."
    )


def _caption_line(i, cat_name, cap_text, caption_chars):
    cap = (cap_text.get(i) or "").strip().replace("\n", " ")
    if len(cap) > caption_chars:
        cap = cap[:caption_chars].rstrip() + "…"
    cat = cat_name.get(i, "uncategorised")
    return f"- [{cat}] {cap}" if cap else f"- [{cat}] (no caption)"


def content_block(items, cat_name, cap_text, recent, caption_chars, top_categories):
    """Category mix over all training rows + the most recent `recent` videos.

    The mix is computed over the whole training history so nothing is silently
    dropped when a dense user's watch list is truncated for prompt length.
    """
    names = [cat_name[i] for i in items if i in cat_name]
    lines = []
    if names:
        counts = pd.Series(names).value_counts()
        share = counts / len(names)
        top = ", ".join(f"{k} ({v:.0%})" for k, v in share.head(top_categories).items())
        lines.append(f"Category mix over the full history ({len(names)} categorised "
                     f"videos, {counts.size} distinct categories): {top}.")
    watched = [_caption_line(i, cat_name, cap_text, caption_chars) for i in items[-recent:]]
    if watched:
        label = ("Most recent " + str(len(watched)) + " videos watched"
                 if len(items) > recent else "Videos watched")
        lines.append(f"{label} (oldest first, category in brackets, caption in Chinese):")
        lines.extend(watched)
    return "\n".join(lines) if lines else "No content information is available for this user."


# --------------------------------------------------------------------------- #
# collab: item-item co-consumption retrieval (CoRAL without the RL policy)
# --------------------------------------------------------------------------- #
class CoConsumption:
    """Item-item co-consumption over TRAINING rows, across all users.

    `top_for(own_items, exclude)` returns the items most strongly co-watched with
    the items this user watched — with the user's own items and their held-out
    valid/test items removed, so retrieval can never surface the item the user is
    scored on.

    Weighting matters more than it looks. RAW CO-OCCURRENCE COUNTS RETURN
    POPULARITY, NOT AFFINITY: the most co-watched item with anything is whatever
    is most watched overall, so a sparse user's evidence block fills up with the
    global top items and the LLM is handed noise. This is the failure CoRAL
    (arXiv 2403.06447) avoids by learning the retrieval policy; the cheap
    equivalent is to divide out popularity, i.e. cosine item-item similarity
    C_ij / sqrt(n_i n_j), which is the default here. `count` is kept only so the
    difference can be shown.
    """

    def __init__(self, train, weighting="cosine"):
        from scipy.sparse import csr_matrix, diags

        u, _ = pd.factorize(train["user_id"], sort=False)
        self.items = np.unique(train["item_id"].to_numpy(np.int64))
        self.col = {int(it): j for j, it in enumerate(self.items)}
        self.weighting = weighting
        c = train["item_id"].map(self.col).to_numpy(np.int64)
        A = csr_matrix((np.ones(len(c), np.float32), (u, c)),
                       shape=(int(u.max()) + 1, len(self.items)))
        A.data[:] = 1.0                       # a repeated watch counts once
        C = (A.T @ A).tocsr()
        C.setdiag(0.0)
        C.eliminate_zeros()
        self.popularity = np.asarray(A.sum(axis=0)).ravel()
        if weighting == "cosine":
            inv = 1.0 / np.sqrt(np.maximum(self.popularity, 1.0))
            D = diags(inv.astype(np.float32))
            C = (D @ C @ D).tocsr()
        elif weighting != "count":
            raise ValueError(f"unknown weighting {weighting!r}")
        self.C = C
        self.pairs = int(C.nnz)

    def top_for(self, own_items, exclude, k):
        cols = [self.col[int(i)] for i in set(own_items) if int(i) in self.col]
        if not cols:
            return []
        scores = np.asarray(self.C[cols].sum(axis=0)).ravel()
        for i in set(own_items) | set(exclude):
            j = self.col.get(int(i))
            if j is not None:
                scores[j] = 0.0
        if not scores.any():
            return []
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.items[j]), float(scores[j])) for j in top if scores[j] > 0]


class UserBasedRetrieval:
    """Items watched by this user's nearest long-history users, popularity-damped.

    Replaces an item-item co-occurrence version that DID NOT WORK ON THIS DATA.
    Measured on 400 users, item-item retrieval filled 4,000 slots with only 417
    distinct items, its top item appeared for 60% of users, and two random users
    shared 23.5% of their top-10 — it was returning popularity, not collaborative
    evidence, because 651k interactions over 7,164 items leaves long-tail
    co-occurrence counts at noise level. Cosine normalisation did not rescue it.

    This is also the formulation CoRAL (arXiv 2403.06447) actually uses: it
    retrieves user-item INTERACTIONS and has the model reason over shared and
    distinct preferences among users. Neighbours come from the same dense pool as
    the support set, candidate items exclude the target's own and held-out items,
    and each candidate is scored

        score_j = sum_n sim_n * 1[neighbour n watched j] / sqrt(popularity_j)

    so an item has to be watched by several *similar* users, not by everyone, to
    surface. Re-run the popularity check on whatever it returns with
    `build_h2_prompts.py --variant collab --diagnose-retrieval`, so the same
    failure cannot pass unnoticed twice.
    """

    def __init__(self, train, neighbours: "CategoryNeighbours", n_users=20):
        self.nb = neighbours
        self.n_users = n_users
        rows = train["user_id"].astype(str).map(neighbours.row_of)
        g = train.assign(_row=rows).dropna(subset=["_row"]).groupby("_row")["item_id"]
        self.items_of = {int(r): set(v.to_numpy(np.int64).tolist()) for r, v in g}
        pop = train.groupby("item_id")["user_id"].nunique()
        self.pop = {int(k): int(v) for k, v in pop.items()}

    def top_for(self, user_token, own_items, exclude, k):
        near = self.nb.nearest(user_token, self.n_users)
        if not near:
            return []
        banned = set(int(i) for i in own_items) | set(int(i) for i in exclude)
        score = {}
        for row, sim in near:
            for i in self.items_of.get(row, ()):
                if i not in banned:
                    score[i] = score.get(i, 0.0) + sim
        if not score:
            return []
        ranked = sorted(((s / np.sqrt(max(self.pop.get(i, 1), 1)), i)
                         for i, s in score.items()), reverse=True)
        return [(i, float(s)) for s, i in ranked[:k]]


def collab_block(neighbours, cat_name, cap_text, caption_chars):
    if not neighbours:
        return ("No collaborative evidence is available: no comparable user in the "
                "training data has a usable history.")
    lines = [f"Videos watched by the long-history users whose taste most resembles "
             f"this one, which this user has not watched ({len(neighbours)} "
             f"strongest first, category in brackets, caption in Chinese):"]
    lines += [_caption_line(i, cat_name, cap_text, caption_chars) for i, _ in neighbours]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# support: nearest users by category mix (in-context support set)
# --------------------------------------------------------------------------- #
class CategoryNeighbours:
    """Nearest users by cosine over L1-normalised first-level category histograms.

    Only category SHARES are shown, never item identities, so a support set
    cannot leak any specific item — including the neighbour's held-out one.

    Two constraints make the set informative rather than a mirror. THE NEIGHBOUR
    POOL IS RESTRICTED TO USERS WITH `min_rows` OR MORE TRAINING ROWS: a
    lowest-quartile user's histogram is a handful of 1/n values, thousands of
    sparse users tie on it exactly, and an unrestricted top-k returns five
    arbitrary copies of the target's own three categories — no information, in
    the one cohort the support set exists to help. Matching a sparse user to
    similar DENSE users instead answers "what else do people like this watch".
    Identical mixes are then de-duplicated, so k distinct profiles are shown.
    """

    def __init__(self, train, cat_name, n_categories_shown, min_rows=20):
        cats = sorted(set(cat_name.values()))
        self.cat_names = cats
        cidx = {c: j for j, c in enumerate(cats)}
        item_cat = {i: cidx[c] for i, c in cat_name.items()}
        uu, self.user_ids = pd.factorize(train["user_id"], sort=False)
        col = train["item_id"].map(item_cat)
        ok = col.notna().to_numpy()
        M = np.zeros((len(self.user_ids), len(cats)), np.float32)
        np.add.at(M, (uu[ok], col.to_numpy()[ok].astype(np.int64)), 1.0)
        self.n_rows = np.bincount(uu, minlength=len(self.user_ids))
        tot = M.sum(1, keepdims=True)
        self.M = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
        nrm = np.linalg.norm(self.M, axis=1, keepdims=True)
        self.U = np.divide(self.M, nrm, out=np.zeros_like(self.M), where=nrm > 0)
        self.row_of = {str(u): j for j, u in enumerate(self.user_ids)}
        self.n_shown = n_categories_shown
        self.min_rows = min_rows
        self.pool = np.flatnonzero((self.n_rows >= min_rows) & (nrm.ravel() > 0))
        self.P = self.U[self.pool]

    def nearest(self, user_token, k):
        """The k most similar pool users as (row, similarity), no de-duplication.

        Item-level retrieval needs the raw nearest users: two users with the same
        category mix have different item sets, so de-duplicating on the mix (which
        `top_for` does, to avoid showing five identical support examples) would
        throw away exactly the variety item retrieval depends on.
        """
        j = self.row_of.get(str(user_token))
        if j is None or not self.U[j].any() or not len(self.pool):
            return []
        sims = self.P @ self.U[j]
        cand = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        cand = cand[np.argsort(-sims[cand])]
        return [(int(self.pool[p]), float(sims[p]))
                for p in cand if int(self.pool[p]) != j and sims[p] > 0]

    def top_for(self, user_token, k):
        j = self.row_of.get(str(user_token))
        if j is None or not self.U[j].any() or not len(self.pool):
            return []
        sims = self.P @ self.U[j]
        # over-fetch, then de-duplicate identical mixes and drop the user itself
        cand = np.argpartition(-sims, min(8 * k, len(sims) - 1))[:8 * k]
        cand = cand[np.argsort(-sims[cand])]
        out, seen = [], set()
        for p in cand:
            row = int(self.pool[p])
            if row == j or sims[p] <= 0:
                continue
            text = self.describe(row)
            if text in seen:
                continue
            seen.add(text)
            out.append((row, float(sims[p])))
            if len(out) == k:
                break
        return out

    def describe(self, row):
        share = self.M[row]
        order = np.argsort(-share)[:self.n_shown]
        return ", ".join(f"{self.cat_names[o]} ({share[o]:.0%})"
                         for o in order if share[o] > 0)


def support_block(neighbours, nb):
    if not neighbours:
        return "No comparable users were found in the training data."
    lines = [f"Category mixes of the {len(neighbours)} long-history users whose "
             f"viewing most resembles this one — what people with this taste watch "
             f"once they have watched more (similarity in brackets):"]
    for row, sim in neighbours:
        lines.append(f"- (similarity {sim:.2f}) {nb.describe(row)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(variant, recent, caption_chars, top_categories, n_collab, n_support,
          n_support_categories, n_collab_users=20, support_min_rows=20,
          collab_retrieval="user", write=True):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    t0 = time.perf_counter()
    for p in (INTER, CAT_CSV, CAP_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run `modal run modal_content_fetch.py` then "
                f"`python -m modal volume get cafrec-data content_pure ../data/content_pure`")

    uid_token2id, _iid, n_users = load_user_item_remap(DATASET)
    cat_name, cap_text, n_spill = load_content()

    df = pd.read_csv(INTER, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    rank_desc = df.groupby("user_id").cumcount(ascending=False)
    train = df[rank_desc >= N_HOLDOUT]
    # each user's own held-out items, excluded from anything retrieved for them
    heldout = (df[rank_desc < N_HOLDOUT].groupby("user_id")["item_id"]
               .apply(lambda s: set(int(i) for i in s)).to_dict())

    cooc = nb = None
    retrieval_meta = {}
    if variant == "collab" and collab_retrieval == "item":
        # The version that FAILED, kept runnable so the negative finding stays
        # reproducible rather than becoming an unsourced sentence in the paper.
        cooc = CoConsumption(train, weighting="cosine")
        retrieval_meta = dict(kind="item-item co-consumption, cosine weighting",
                              n_items=len(cooc.items), nonzero_pairs=cooc.pairs,
                              n_retrieved=n_collab, status="superseded: returns popularity",
                              excludes="the user's own items and their held-out valid/test items")
    elif variant == "collab":
        nb = CategoryNeighbours(train, cat_name, n_support_categories,
                                min_rows=support_min_rows)
        cooc = UserBasedRetrieval(train, nb, n_users=n_collab_users)
        retrieval_meta = dict(
            kind="items watched by the nearest long-history users, scored "
                 "sum(similarity) / sqrt(item popularity)",
            neighbours_pooled=n_collab_users, n_retrieved=n_collab,
            neighbour_pool_min_training_rows=support_min_rows,
            neighbour_pool_size=int(len(nb.pool)),
            replaces="item-item co-occurrence, which returned popularity on this "
                     "dataset (417 distinct items over 4,000 slots for 400 users)",
            excludes="the user's own items and their held-out valid/test items")
    elif variant == "support":
        nb = CategoryNeighbours(train, cat_name, n_support_categories,
                                min_rows=support_min_rows)
        retrieval_meta = dict(kind="cosine over L1-normalised first-level category histograms",
                              n_categories=len(nb.cat_names), n_neighbours=n_support,
                              categories_shown=n_support_categories,
                              neighbour_pool_min_training_rows=support_min_rows,
                              neighbour_pool_size=int(len(nb.pool)),
                              deduplicated="identical category mixes are shown once",
                              excludes="item identities are never shown, only category shares")

    instruction = INSTRUCTIONS[variant]
    rows, skipped, n_no_content, n_no_retrieval = [], 0, 0, 0
    for uid_tok, g in train.groupby("user_id", sort=False):
        internal = uid_token2id.get(str(uid_tok))
        if internal is None:                     # filtered out of the RecBole dataset
            skipped += 1
            continue
        items = g["item_id"].to_numpy(np.int64).tolist()
        blocks = [behaviour_block(g, len(g)),
                  content_block(items, cat_name, cap_text, recent, caption_chars,
                                top_categories)]
        if blocks[1].startswith("No content"):
            n_no_content += 1
        if variant == "collab":
            got = (cooc.top_for(items, heldout.get(uid_tok, set()), n_collab)
                   if collab_retrieval == "item" else
                   cooc.top_for(uid_tok, items, heldout.get(uid_tok, set()), n_collab))
            n_no_retrieval += not got
            blocks.append(collab_block(got, cat_name, cap_text, caption_chars))
        elif variant == "support":
            got = nb.top_for(uid_tok, n_support)
            n_no_retrieval += not got
            blocks.append(support_block(got, nb))
        blocks.append(instruction)
        rows.append(dict(user_id=str(uid_tok), uid=int(internal), n_train=int(len(g)),
                         prompt="\n\n".join(blocks)))
    rows.sort(key=lambda r: r["uid"])

    chars = np.array([len(r["prompt"]) for r in rows])
    n_rows = np.array([r["n_train"] for r in rows])
    meta = dict(
        script="build_h2_prompts.py", variant=variant,
        registered=(variant == "plain"),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        dataset=DATASET, n_users_recbole=n_users, n_prompts=len(rows),
        users_not_in_remap=skipped, users_without_content=n_no_content,
        users_without_retrieval=n_no_retrieval,
        holdout_rows_dropped=N_HOLDOUT, window_rows_dropped=0,
        row_rule="r >= 2 (all training rows), as cap_all; NOT the r >= 22 beyond-window rule",
        recent=recent, caption_chars=caption_chars, top_categories=top_categories,
        retrieval=retrieval_meta, collab_retrieval=collab_retrieval,
        caption_rows_repaired=n_spill, system=SYSTEM, instruction=instruction,
        inter=str(INTER), inter_sha256=sha256(INTER),
        categories_csv=str(CAT_CSV), categories_sha256=sha256(CAT_CSV),
        captions_csv=str(CAP_CSV), captions_sha256=sha256(CAP_CSV),
        prompt_chars=dict(mean=float(chars.mean()), median=float(np.median(chars)),
                          p95=float(np.quantile(chars, 0.95)), max=int(chars.max()),
                          total=int(chars.sum())),
        train_rows=dict(mean=float(n_rows.mean()), median=float(np.median(n_rows)),
                        min=int(n_rows.min()), max=int(n_rows.max())),
        seconds=round(time.perf_counter() - t0, 1),
    )
    out = OUT_DIR / f"prompts.{variant}.jsonl"
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (OUT_DIR / f"prompts_meta.{variant}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows, meta, out


def diagnose_retrieval(variant, rows, n_sample=400, seed=0):
    """Is the retrieved block per-user information, or the same popular items for
    everyone? Reports the three numbers that condemned item-item co-occurrence:
    distinct lines over total slots, the most-repeated line's user share, and the
    mean pairwise overlap of two users' retrieved sets. High concentration means
    the block carries little per-user signal, whatever it is nominally retrieving.
    """
    import collections
    import itertools

    if variant not in ("collab", "support"):
        print(f"  {variant}: no retrieval block to diagnose")
        return None
    rng = np.random.default_rng(seed)
    pick = [rows[i] for i in rng.choice(len(rows), min(n_sample, len(rows)), replace=False)]
    blocks = []
    for r in pick:
        parts = r["prompt"].split("\n\n")
        blocks.append([ln for ln in parts[2].split("\n")[1:] if ln.strip()])
    slots = sum(len(b) for b in blocks)
    counts = collections.Counter(ln for b in blocks for ln in b)
    ov = [len(set(a) & set(b)) / max(min(len(a), len(b)), 1)
          for a, b in itertools.islice(itertools.combinations(blocks, 2), 2000)]
    out = dict(variant=variant, users=len(pick), slots=slots,
               distinct_lines=len(counts),
               top_line_user_share=counts.most_common(1)[0][1] / len(pick) if counts else 0.0,
               mean_pairwise_overlap=float(np.mean(ov)) if ov else 0.0)
    print(f"\n=== retrieval diagnostic: {variant} ===")
    print(f"  distinct retrieved lines : {out['distinct_lines']:,} of {slots:,} slots "
          f"across {len(pick)} users")
    print(f"  most-repeated line       : {100 * out['top_line_user_share']:.1f}% of users")
    print(f"  mean pairwise overlap    : {100 * out['mean_pairwise_overlap']:.1f}%")
    print("  (item-item co-occurrence scored 417/4,000, 60.0%, 23.5% — that was "
          "popularity, not collaboration)")
    for ln, c in counts.most_common(3):
        print(f"    {100 * c / len(pick):5.1f}%  {ln[:72]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="plain",
                    help=f"one of {list(VARIANTS)}, or 'all'")
    ap.add_argument("--recent", type=int, default=32,
                    help="most recent training videos listed in the prompt; the "
                         "main generation-cost lever, and irrelevant to the H2 cohort")
    ap.add_argument("--caption-chars", type=int, default=100, help="per-caption truncation")
    ap.add_argument("--top-categories", type=int, default=6)
    ap.add_argument("--n-collab", type=int, default=10,
                    help="co-consumed items retrieved for the collab variant")
    ap.add_argument("--n-support", type=int, default=5,
                    help="neighbour users shown for the support variant")
    ap.add_argument("--n-support-categories", type=int, default=6)
    ap.add_argument("--collab-retrieval", choices=["user", "item"], default="user",
                    help="user: items from the nearest long-history users. "
                         "item: the superseded item-item co-occurrence, kept so "
                         "its popularity failure stays reproducible")
    ap.add_argument("--n-collab-users", type=int, default=20,
                    help="neighbour users pooled to retrieve items from, for collab")
    ap.add_argument("--support-min-rows", type=int, default=20,
                    help="minimum training rows for a user to enter the neighbour pool")
    ap.add_argument("--diagnose-retrieval", action="store_true",
                    help="report whether the retrieved block is per-user or just "
                         "popularity, and exit without writing")
    ap.add_argument("--sample", type=int, default=1, help="prompts to print")
    ap.add_argument("--dry-run", action="store_true", help="print without writing")
    args = ap.parse_args()

    wanted = list(VARIANTS) if args.variant == "all" else [args.variant]
    totals = {}
    for variant in wanted:
        rows, meta, out = build(variant, args.recent, args.caption_chars,
                                args.top_categories, args.n_collab, args.n_support,
                                args.n_support_categories,
                                n_collab_users=args.n_collab_users,
                                support_min_rows=args.support_min_rows,
                                collab_retrieval=args.collab_retrieval,
                                write=not (args.dry_run or args.diagnose_retrieval))
        if args.diagnose_retrieval:
            diagnose_retrieval(variant, rows)
            continue
        n = np.array([r["n_train"] for r in rows])
        c = np.array([len(r["prompt"]) for r in rows])
        q1 = n <= 10          # <= 12 organic clicks minus the 2 held out
        totals[variant] = dict(chars=int(c.sum()), mean=int(c.mean()),
                               q1_mean=int(c[q1].mean()), q1_users=int(q1.sum()),
                               no_retrieval=meta["users_without_retrieval"])
        print(f"\n=== {variant}{' (REGISTERED ARM)' if variant == 'plain' else ' (exploratory)'} ===")
        for k in ("n_prompts", "users_not_in_remap", "users_without_content",
                  "users_without_retrieval", "seconds"):
            print(f"  {k:24s}: {meta[k]}")
        print(f"  prompt_chars            : mean {int(c.mean()):,}  "
              f"median {int(np.median(c)):,}  p95 {int(np.quantile(c, 0.95)):,}  "
              f"total {c.sum() / 1e6:.1f}M")
        print(f"  lowest quartile         : {int(q1.sum()):,} users, "
              f"mean {int(c[q1].mean()):,} chars")
        if meta["retrieval"]:
            print(f"  retrieval               : {meta['retrieval']}")
        for r in rows[:args.sample]:
            print(f"\n--- user {r['user_id']} (uid {r['uid']}, {r['n_train']} training rows) ---")
            print(r["prompt"])

    if len(totals) > 1:
        base = totals["plain"]["chars"]
        print("\n=== relative generation cost (prefill dominates) ===")
        for v, t in totals.items():
            print(f"  {v:<8} {t['chars'] / 1e6:5.1f}M chars  "
                  f"{t['chars'] / base:4.2f}x plain   Q1 mean {t['q1_mean']:,} chars")
    if not args.dry_run:
        print("\nNext, per variant:")
        print("  python -m modal volume put cafrec-data ../data/h2/prompts.<v>.jsonl h2/prompts.<v>.jsonl")
        print("  modal run modal_h2_generate.py --variant <v> --limit 64")


if __name__ == "__main__":
    main()
