"""Tests for diversity metrics (RON-15) and the category matrix builder."""
import numpy as np

from cafrec.eval.diversity import coverage, intra_list_diversity, topk_from_scores
from cafrec.eval.category_matrix import build_category_matrix


def test_coverage_fraction():
    # two users, top-2 lists covering items {0,1,2} out of a catalogue of 5
    topk = np.array([[0, 1], [1, 2]])
    assert coverage(topk, n_items=5) == 3 / 5


def test_ild_hand_computed():
    # item 0 -> [1,0], item 1 -> [0,1], item 2 -> [1,1]
    cat = np.array([[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])
    # single user recommended all three; ILD = mean pairwise (1 - cosine)
    # pairs: (0,1)=1.0, (0,2)=1-1/sqrt2, (1,2)=1-1/sqrt2
    expected = (1.0 + (1 - 1 / np.sqrt(2)) + (1 - 1 / np.sqrt(2))) / 3
    got = intra_list_diversity([[0, 1, 2]], cat)
    assert abs(got - expected) < 1e-9


def test_ild_zero_vector_is_maximally_dissimilar():
    # a null-tag item (all-zero row) should not crash and should read as
    # fully dissimilar (distance 1) from any other item
    cat = np.array([[1.0, 0.0],
                    [0.0, 0.0]])            # item 1 has no tags
    got = intra_list_diversity([[0, 1]], cat)
    assert abs(got - 1.0) < 1e-9


def test_topk_from_scores_selects_highest():
    scores = np.array([[0.1, 0.9, 0.5, 0.2],
                       [0.7, 0.1, 0.8, 0.05]])
    top2 = topk_from_scores(scores, k=2)
    assert set(top2[0].tolist()) == {1, 2}         # highest two for user 0
    assert set(top2[1].tolist()) == {0, 2}         # highest two for user 1


def test_category_matrix_multihot():
    # item 0 -> "39", item 1 -> "39,68", item 2 -> null
    mat, cols = build_category_matrix(
        item_ids=[0, 1, 2],
        tag_values=["39", "39,68", None],
        strategy="multihot",
    )
    assert mat.shape[0] == 3
    # tag 39 is shared; tag 68 only on item 1
    c39, c68 = cols["39"], cols["68"]
    assert mat[0, c39] == 1 and mat[0, c68] == 0
    assert mat[1, c39] == 1 and mat[1, c68] == 1
    assert mat[2].sum() == 0                        # null -> zero row


def test_category_matrix_primary_vs_label():
    prim, pcols = build_category_matrix([0, 1], ["39,68", "39"], strategy="primary")
    # primary keeps only "39" for both -> single shared column
    assert set(pcols) == {"39"}
    assert prim[0, pcols["39"]] == 1 and prim[1, pcols["39"]] == 1

    lab, lcols = build_category_matrix([0, 1], ["39,68", "39"], strategy="label")
    # label treats "39,68" as its own atomic category, distinct from "39"
    assert set(lcols) == {"39,68", "39"}
    assert lab[0, lcols["39,68"]] == 1 and lab[0, lcols["39"]] == 0
