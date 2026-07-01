"""Tests for significance testing (RON-14)."""
import numpy as np

from cafrec.eval.significance import wilcoxon_test, paired_bootstrap_diff


def test_wilcoxon_identical_arrays():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    stat, p = wilcoxon_test(a, a)
    assert stat == 0.0 and p == 1.0                 # no difference -> p = 1


def test_wilcoxon_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.random(200)
    better = base + 0.05                            # model A beats B everywhere
    stat, p = wilcoxon_test(better, base)
    assert p < 0.01                                 # clearly significant


def test_bootstrap_ci_brackets_the_mean_diff():
    rng = np.random.default_rng(1)
    a = rng.random(500) + 0.1
    b = rng.random(500)
    mean_diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=500, seed=1)
    assert lo <= mean_diff <= hi
    assert abs(mean_diff - (a - b).mean()) < 1e-12


def test_bootstrap_ci_is_reproducible():
    a = np.linspace(0, 1, 100)
    b = np.linspace(0.1, 0.9, 100)
    r1 = paired_bootstrap_diff(a, b, n_boot=300, seed=42)
    r2 = paired_bootstrap_diff(a, b, n_boot=300, seed=42)
    assert r1 == r2
