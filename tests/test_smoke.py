"""Tiny CPU-only smoke test for HARP.

Runs without a GPU, a model download, or any data files. It exercises:
  * the package import and public API,
  * the ``harp()`` entry point's empty-input early return,
  * the core HARP-C / HARP-E selection semantics (:func:`harp.greedy_taskaware`),
  * the Stage-1a node-leaf hierarchy on CPU (sklearn fallback, no faiss/GPU).

Run directly::

    python tests/test_smoke.py

or under pytest::

    pytest tests/test_smoke.py
"""
import os
import sys

# Make `import harp` work regardless of the working directory / how this is run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from harp import harp, HarpResult, greedy_taskaware


def test_imports():
    assert callable(harp)
    assert set(HarpResult.__dataclass_fields__) == {"harp_c", "harp_e", "info"}


def test_empty_input_returns_harpresult():
    """The real ``harp()`` entry point short-circuits on empty input (no GPU)."""
    res = harp(None, "dummy-model", [], [], tokenizer=None)
    assert isinstance(res, HarpResult)
    assert res.harp_c.size == 0 and res.harp_e.size == 0
    assert res.info.get("error") == "empty input"


def _best_prefix(gains):
    g = np.asarray(gains, dtype=float)
    return int(np.argmax(np.cumsum(g))) + 1 if g.size else 0


def test_harp_c_vs_harp_e_envelopes():
    """HARP-C (max-positive) drops a redundant leaf; HARP-E (additive) keeps it.

    Two tasks, three leaves. leaf0 and leaf1 both only help task 0 (leaf1 is
    redundant given leaf0); leaf2 covers task 1.
    """
    phi = np.array([[0.3, 0.0],   # leaf 0: helps task 0
                    [0.2, 0.0],   # leaf 1: also helps task 0 (redundant)
                    [0.0, 0.4]])  # leaf 2: helps task 1
    w = np.array([0.5, 0.5])
    b = np.array([0.0, 0.0])
    leaf_groups = [np.array([0]), np.array([1]), np.array([2])]  # 1 point per leaf

    c_order, c_info = greedy_taskaware(
        phi, w, b, leaf_groups, points_cap=10**9, mode="conservative", use_knee_stop=False)
    e_order, e_info = greedy_taskaware(
        phi, w, b, leaf_groups, points_cap=10**9, mode="optimistic", use_knee_stop=False)

    c_cut = _best_prefix(c_info["gains"])
    e_cut = _best_prefix(e_info["gains"])
    assert c_cut == 2, f"HARP-C should drop the redundant leaf (cut=2), got {c_cut}"
    assert e_cut == 3, f"HARP-E should keep the additive leaf (cut=3), got {e_cut}"
    assert 1 not in c_order[:c_cut].tolist(), "HARP-C must exclude the redundant leaf"


def test_hierarchy_partitions_points_on_cpu():
    """Stage-1a hierarchy runs on CPU and partitions every point into a leaf."""
    from harp.selector import build_parent_leaf_hierarchy

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 16)).astype("float32")
    (large_groups, parent_groups, leaf_groups, large_of_parent,
     parent_of_leaf, large_of_leaf, leaf_of_i, leaf_feat) = build_parent_leaf_hierarchy(
        X, rng=rng,
        large_max_group_size=100, parent_max_group_size=50,
        leaf_max_group_size=20, leaf_min_group_size=5,
        ann_use_gpu=False, lora_config={})

    covered = sorted(int(p) for g in leaf_groups for p in np.asarray(g).tolist())
    assert covered == list(range(n)), "leaves must partition all points exactly once"
    assert len(leaf_groups) >= 1 and all(len(g) > 0 for g in leaf_groups)
    assert len(parent_of_leaf) == len(leaf_groups)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nAll HARP CPU smoke tests passed.")
