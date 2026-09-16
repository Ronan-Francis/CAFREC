"""Catalogue tiers and their batch-size defaults.

Batch sizing used to test `dataset != "kuairand_pure"`, which sent the context
variant `kuairand_pure_ctx` down the large-catalogue branch: CAFREC trained at
512 while SASRec/HGN/HGRU4Rec trained at base.yaml's 2048 on the same data.
That gap is +3.0% of the published +3.7% H1 margin (PROJECT_LOG, Cycle 10
cont.). The rule now keys off the catalogue tier, which every variant of a tier
(_ctx, _kcore, ...) shares, and refuses to guess for a dataset it does not know.
"""

# CE loss and full-ranking eval each materialise a (batch x n_items) tensor.
# Pure's ~7.2K items fit base.yaml's 2048 / 4096 on an A10G. 1K and 27K carry the
# ~million-item video catalogue: 2048 OOMs a 16GB T4 and 4096 eval needs 13.5GB+.
SMALL_CATALOGUE_TIERS = ("kuairand_pure",)
LARGE_CATALOGUE_TIERS = ("kuairand_1k", "kuairand_27k")

LARGE_CATALOGUE_BATCHES = {"train_batch_size": 512, "eval_batch_size": 256}


def _in_tier(dataset, tier):
    return dataset == tier or dataset.startswith(tier + "_")


def catalogue_tier(dataset):
    """'small' or 'large'. Raises for a dataset outside the known tiers."""
    if any(_in_tier(dataset, t) for t in SMALL_CATALOGUE_TIERS):
        return "small"
    if any(_in_tier(dataset, t) for t in LARGE_CATALOGUE_TIERS):
        return "large"
    raise ValueError(
        f"unknown catalogue tier for dataset {dataset!r}; add it to "
        f"SMALL_CATALOGUE_TIERS or LARGE_CATALOGUE_TIERS in cafrec/tiers.py")


def batch_size_overrides(dataset):
    """Default batch-size overrides for `dataset`'s tier.

    Empty for small catalogues, so base.yaml's values apply. Explicit caller
    overrides should be applied after these.
    """
    if catalogue_tier(dataset) == "large":
        return dict(LARGE_CATALOGUE_BATCHES)
    return {}
