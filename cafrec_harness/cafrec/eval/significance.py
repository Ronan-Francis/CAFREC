import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_test(metric_a, metric_b):
    """Paired Wilcoxon signed-rank over per-user values (same users, aligned)."""
    a = np.asarray(metric_a, float)
    b = np.asarray(metric_b, float)
    if np.allclose(a, b):              # scipy errors when every diff is zero
        return 0.0, 1.0
    stat, p = wilcoxon(a, b)           # zero-diff pairs dropped by default
    return float(stat), float(p)


def paired_bootstrap_diff(metric_a, metric_b, n_boot=1000, alpha=0.05, seed=42):
    """95% CI for the per-user mean difference (model A - model B).
    Resamples user indices ONCE per iteration and applies to both -> properly paired."""
    rng = np.random.default_rng(seed)
    a = np.asarray(metric_a, float)
    b = np.asarray(metric_b, float)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float((a - b).mean()), float(lo), float(hi)