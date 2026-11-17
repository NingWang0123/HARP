"""CPU oracles for representative selection, legacy RNG behavior, and resume.

Run: python -m unittest discover -s tests -p 'test_representatives.py' -v
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harp import knn_grouping as k
from harp import selector as s


FEATURES = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.],
                     [1., 1.], [-1., 1.]])


class FarthestFirstTests(unittest.TestCase):
    def test_each_pick_maximizes_minimum_distance_on_normalized_geometry(self):
        features = FEATURES * np.array([3., 10., 2., 9., 0.2, 4.])[:, None]
        unit = FEATURES / np.linalg.norm(FEATURES, axis=1, keepdims=True)
        for seed in range(12):
            chosen = k.diverse_sample_farthest_first(features, 5, np.random.default_rng(seed))
            self.assertEqual(chosen[0], np.random.default_rng(seed).integers(len(features)))
            self.assertEqual(len(set(chosen)), 5)
            for t in range(1, len(chosen)):
                remaining = np.array(sorted(set(range(len(features))) - set(chosen[:t])))
                distances = np.sum((unit[remaining, None] - unit[chosen[:t]])**2, axis=2)
                expected = remaining[np.argmax(distances.min(axis=1))]
                self.assertEqual(chosen[t], expected)
            np.testing.assert_array_equal(chosen, k.diverse_sample_farthest_first(
                features, 5, np.random.default_rng(seed)))

    def test_duplicate_zero_embeddings_and_ties_have_unique_input_order_completion(self):
        for features in [np.zeros((6, 2)), np.ones((6, 2))]:
            for seed in [0, 1, 8]:
                first = int(np.random.default_rng(seed).integers(6))
                expected = [first] + [g for g in range(6) if g != first][:4]
                np.testing.assert_array_equal(k.diverse_sample_farthest_first(
                    features, 5, np.random.default_rng(seed)), expected)
        # Equal distances from [1,0] to both vertical vectors: lowest input index wins.
        class FirstZero:
            def integers(self, *args):
                return 0
        np.testing.assert_array_equal(k.diverse_sample_farthest_first(
            FEATURES[:4], 3, FirstZero()), [0, 2, 1])

    def test_empty_nonpositive_and_capped_budgets_and_single_seed(self):
        for budget in [-2, 0, 1, 5, 6, 20]:
            rng = np.random.default_rng(13)
            chosen = k.diverse_sample_farthest_first(FEATURES, budget, rng)
            self.assertEqual(len(chosen), min(6, max(0, budget)))
            self.assertEqual(len(set(chosen)), len(chosen))
            if budget >= 6:
                np.testing.assert_array_equal(chosen, np.arange(6))
                self.assertEqual(rng.random(), np.random.default_rng(13).random())
        self.assertEqual(k.diverse_sample_farthest_first(
            np.empty((0, 2)), 3, np.random.default_rng(0)).size, 0)
        for seed in range(6):
            ff_rng, kpp_rng = np.random.default_rng(seed), np.random.default_rng(seed)
            np.testing.assert_array_equal(
                k.diverse_sample_farthest_first(FEATURES, 1, ff_rng),
                k.diverse_sample_kpp(FEATURES, 1, kpp_rng))
            self.assertEqual(ff_rng.random(), kpp_rng.random())

    def test_finite_large_features_normalize_and_invalid_features_fail(self):
        large = np.array([[1e300, 0.], [0., 1e300], [-1e300, 0.]])
        np.testing.assert_array_equal(k.diverse_sample_farthest_first(
            large, 2, np.random.default_rng(0)), k.diverse_sample_farthest_first(
                large / 1e300, 2, np.random.default_rng(0)))
        for features in [np.array([[np.nan]]), np.array([[np.inf]]),
                         np.ones((2, 2, 2)), np.empty((2, 0))]:
            with self.assertRaises(ValueError):
                k.diverse_sample_farthest_first(features, 1, np.random.default_rng(0))


class RepresentativePipelineTests(unittest.TestCase):
    def test_existing_kpp_outputs_and_rng_match_committed_legacy_golden_cases(self):
        path = Path(__file__).parent / "fixtures" / "representatives_kpp_6aab4bc.json"
        fixture = json.loads(path.read_text())
        self.assertEqual(fixture["source_commit"], "6aab4bc42777e839bede5bf747e0ec73a4ae2ac6")
        for case in fixture["cases"]:
            with self.subTest(seed=case["seed"], reps=case["rep_per_parent"]):
                rng = np.random.default_rng(case["seed"])
                representatives, ids = k.sample_representative_leaf_groups(
                    [np.arange(n) for n in case["sizes"]], np.array(case["features"]),
                    np.array(case["parents"]), rng=rng,
                    rep_per_parent=case["rep_per_parent"], use_diverse=case["use_diverse"],
                    representative_strategy="kmeans_plus_plus")
                self.assertEqual([v.tolist() for v in representatives], case["representatives"])
                np.testing.assert_array_equal(ids, case["ids"])
                np.testing.assert_array_equal(rng.random(4), case["next_random"])

    def test_real_adaptive_regions_and_proportional_allocation_are_shared(self):
        sizes = [1]*8 + [20]*3 + [100]
        features = np.random.default_rng(8).normal(size=(12, 3))
        groups = [np.arange(n) for n in sizes]
        for strategy in k.REPRESENTATIVE_STRATEGIES:
            reps, ids = k.sample_representative_leaf_groups(
                groups, features, np.zeros(12, dtype=int), rep_per_parent=5,
                rng=np.random.default_rng(8), representative_strategy=strategy)
            self.assertEqual(len(ids), 5)
            self.assertEqual([sum(sizes[g] == n for g in reps[0]) for n in [1, 20, 100]],
                             [3, 1, 1])
        # A remaining fractional allocation goes to the largest fractional part.
        with patch.object(k, "_choose_size_regions_1d_kmeans",
                          return_value=np.array([0]*6+[1]*3+[2]*3)):
            _, ids = k.sample_representative_leaf_groups(
                groups, features, np.zeros(12, dtype=int), rep_per_parent=5,
                rng=np.random.default_rng(8))
        self.assertEqual([sum(g < 6 for g in ids), sum(6 <= g < 9 for g in ids),
                          sum(g >= 9 for g in ids)], [3, 1, 1])

    def test_default_routes_to_farthest_first_and_uniform_takes_precedence(self):
        groups = [np.array([g]) for g in range(6)]
        with patch.object(k, "diverse_sample_farthest_first",
                          wraps=k.diverse_sample_farthest_first) as ff, \
             patch.object(k, "diverse_sample_kpp", side_effect=AssertionError("KPP used")):
            reps, ids = k.sample_representative_leaf_groups(
                groups, FEATURES, np.zeros(6, dtype=int), rng=np.random.default_rng(0))
            self.assertEqual(ff.call_count, 1)
            np.testing.assert_array_equal(reps[0], k.diverse_sample_farthest_first(
                FEATURES, 3, np.random.default_rng(0)))
            np.testing.assert_array_equal(ids, sorted(reps[0]))
        states = []
        outputs = []
        with patch.object(k, "diverse_sample_farthest_first", side_effect=AssertionError), \
             patch.object(k, "diverse_sample_kpp", side_effect=AssertionError):
            for strategy in k.REPRESENTATIVE_STRATEGIES:
                rng = np.random.default_rng(9)
                outputs.append(k.sample_representative_leaf_groups(
                    groups, FEATURES, np.zeros(6, dtype=int), rng=rng, use_diverse=False,
                    representative_strategy=strategy)[1])
                states.append(rng.random())
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertEqual(states[0], states[1])

    def test_single_representative_and_full_parent_budgets_preserve_both_strategies(self):
        groups = [np.arange(g+1) for g in range(6)]
        for budget in [1, 6, 20]:
            results, states = [], []
            for strategy in k.REPRESENTATIVE_STRATEGIES:
                rng = np.random.default_rng(10)
                results.append(k.sample_representative_leaf_groups(
                    groups, FEATURES, np.zeros(6, dtype=int), rep_per_parent=budget,
                    rng=rng, representative_strategy=strategy)[1])
                states.append(rng.random())
            np.testing.assert_array_equal(results[0], results[1])
            self.assertEqual(len(results[0]), min(budget, 6))
            self.assertEqual(states[0], states[1])

    def test_gaps_empty_parents_and_degenerate_strata_fill_requested_unique_counts(self):
        groups = [np.array([g]) for g in range(10)]
        parents = np.array([0]*6 + [2]*4)
        for strategy in k.REPRESENTATIVE_STRATEGIES:
            reps, ids = k.sample_representative_leaf_groups(
                groups, np.zeros((10, 2)), parents, rep_per_parent=3,
                rng=np.random.default_rng(7), representative_strategy=strategy)
            self.assertEqual([len(v) for v in reps], [3, 0, 3])
            self.assertEqual(len(set(ids)), 6)
            self.assertTrue(all(parents[g] == p for p, row in enumerate(reps) for g in row))

    def test_all_entry_points_default_to_ff_and_reject_invalid_strategy_early(self):
        for function in [s.harp, s.bootstrap_parent_representative_independent_per_task,
                         k.sample_representative_leaf_groups]:
            self.assertEqual(inspect.signature(function).parameters[
                "representative_strategy"].default, "farthest_first")
        with self.assertRaisesRegex(ValueError, "representative_strategy"):
            s.harp(None, "unused", [], [], tokenizer=None, representative_strategy="unknown")
        with self.assertRaisesRegex(ValueError, "representative_strategy"):
            k.sample_representative_leaf_groups([], np.empty((0, 2)), np.array([]),
                rng=np.random.default_rng(0), use_diverse=False, representative_strategy="unknown")
        with self.assertRaisesRegex(ValueError, "representative_strategy"):
            s.bootstrap_parent_representative_independent_per_task(
                None, leaf_groups=[], leaf_feat=np.empty((0, 2)), parent_of_leaf=np.array([]),
                group_of_i=np.array([]), eval_metric="accuracy", v0_per_task={},
                subj_id_to_task={}, representative_strategy="unknown")


def checkpoint_kwargs():
    return dict(leaf_groups=[np.array([g]) for g in range(6)], leaf_feat=FEATURES,
        parent_of_leaf=np.zeros(6, dtype=int), bootstrap_seed_int=0,
        eval_metric="accuracy", base_model_name="fake", n_train=6, n_eval=1,
        validation_fingerprint="full-validation", rep_per_parent=3,
        interpolation_temperature=1., use_eb_shrinkage=True, prior_var=.01, se_floor=1e-3)


class RepresentativeCheckpointTests(unittest.TestCase):
    def test_strategy_changes_identity_and_pre_strategy_caches_are_rejected(self):
        kwargs = checkpoint_kwargs()
        ff = s._stage1b_ckpt_key(**kwargs)
        kpp = s._stage1b_ckpt_key(**kwargs, representative_strategy="kmeans_plus_plus")
        self.assertNotEqual(ff, kpp)
        # Exact old v2 identity, including metadata and binary hierarchy inputs.
        metadata = dict(protocol="full-validation-cosine-interpolation-v2", seed=0,
            metric="accuracy", model="fake", n_train=6, n_eval=1,
            validation="full-validation", rep_per_parent=3, interpolation_temperature=1.,
            use_eb_shrinkage=True, prior_var=.01, se_floor=1e-3)
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        for value in [*kwargs["leaf_groups"], kwargs["parent_of_leaf"], kwargs["leaf_feat"]]:
            array = np.ascontiguousarray(value)
            digest.update(json.dumps([str(array.dtype), array.shape]).encode("utf-8"))
            digest.update(array.tobytes())
        old = digest.hexdigest()[:16]
        with tempfile.TemporaryDirectory() as directory:
            for level in ["A", "B"]:
                payload = {"leaf_is_measured": np.ones(6, dtype=bool)}
                for key in [ff, kpp]:
                    self.assertNotEqual(old, key)
                    s._save_stage1b_ckpt(directory, level, old, payload)
                    self.assertIsNone(s._try_load_stage1b_ckpt(directory, level, key))
                    s._save_stage1b_ckpt(directory, level, key, payload)
                    self.assertIsNotNone(s._try_load_stage1b_ckpt(directory, level, key))
                self.assertIsNone(s._try_load_stage1b_ckpt(directory, level, ff))

    def test_public_and_standalone_bootstrap_propagate_strategy_and_resume_safely(self):
        measured_ids = []

        class Pool:
            def __init__(self, **kwargs):
                self.eval_arrays = {name: np.ones((1, 1), dtype=np.int64)
                                    for name in s._DATA_ARRAY_KEYS}
                self.eval_arrays["_subject_map"] = {"task": 0}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def parallel_run_independent_singletons_per_task(self, tasks):
                measured_ids.append([task[0] for task in tasks])
                return {task[0]: {0: .5+.01*(task[0]+1)} for task in tasks}

        groups = [np.array([g]) for g in range(6)]
        hierarchy = ([np.arange(6)], [np.arange(6)], groups, np.array([0]),
                     np.zeros(6, dtype=int), np.zeros(6, dtype=int), np.arange(6), FEATURES)
        for strategy in k.REPRESENTATIVE_STRATEGIES:
            result = s.bootstrap_parent_representative_independent_per_task(
                Pool(), leaf_groups=groups, leaf_feat=FEATURES,
                parent_of_leaf=np.zeros(6, dtype=int), group_of_i=np.arange(6),
                eval_metric="accuracy", v0_per_task={"task": .5}, subj_id_to_task={0: "task"},
                rep_per_parent=3, representative_strategy=strategy, rng=np.random.default_rng(0))
            expected = k.sample_representative_leaf_groups(groups, FEATURES, np.zeros(6, dtype=int),
                rng=np.random.default_rng(0), representative_strategy=strategy)[1]
            np.testing.assert_array_equal(measured_ids[-1], expected)
            self.assertEqual(result[-1]["bootstrap"]["representative_strategy"], strategy)
        measured_ids.clear()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"baseline.jsonl"
            path.write_text(json.dumps({"subject": "task", "acc01": .5,
                                        "_meta": {"bucket_name": "task"}})+"\n")
            with patch.object(s, "GPUWorkerPool", Pool), \
                 patch.object(s, "build_parent_leaf_hierarchy", return_value=hierarchy):
                for strategy in k.REPRESENTATIVE_STRATEGIES:
                    before = len(measured_ids)
                    for repeat in range(2):
                        result = s.harp(np.eye(6), "fake",
                            [{"input": f"train{g}"} for g in range(6)],
                            [{"input": "validation", "subject": "task"}], tokenizer=None,
                            representatives_path=str(path), hierarchy_csv_path=None,
                            checkpoint_dir=directory, representative_strategy=strategy,
                            rng=np.random.default_rng(0), TOP_K_POINTS=6, use_knee_stop=False)
                        self.assertEqual(len(measured_ids), before+1)
                        self.assertEqual(result.info["representative_strategy"], strategy)
                        self.assertEqual(result.info["bootstrap"]["representative_strategy"], strategy)
                    expected = k.sample_representative_leaf_groups(
                        groups, FEATURES, np.zeros(6, dtype=int), rng=np.random.default_rng(0),
                        representative_strategy=strategy)[1]
                    np.testing.assert_array_equal(measured_ids[-1], expected)
        self.assertEqual(len(measured_ids), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
