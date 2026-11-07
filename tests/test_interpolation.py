"""Deterministic paper §3.3 regression oracles; CPU only, no model downloads.

Run: python -m unittest discover -s tests -p 'test_interpolation.py' -v
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from harp import selector as s


FEATURES = np.array([[2., 0.], [0., 3.], [5., 0.], [0., 9.], [-2., 0.], [-4., 0.]])
PARENTS = np.array([0, 0, 0, 0, 1, 1])
REPS = [np.array([0, 1]), np.array([4])]
EFFECTS = {0: 0.4, 1: -0.2, 4: 0.2}


class MeasuredPool:
    def parallel_run_independent_singletons_per_task(self, tasks):
        return {task[0]: {0: 0.5 + EFFECTS[task[0]]} for task in tasks}


def bootstrap(*, shrink=True, temperature=1.0, features=None, reps=None, pool=None,
              prior=0.01, floor=1e-3, baselines=None, domains=None):
    features = FEATURES if features is None else features
    reps = REPS if reps is None else reps
    representative_ids = np.array(sorted({int(g) for row in reps for g in row}))
    with patch.object(s, "sample_representative_leaf_groups", return_value=(reps, representative_ids)):
        return s.bootstrap_parent_representative_independent_per_task(
            MeasuredPool() if pool is None else pool,
            leaf_groups=[np.array([g]) for g in range(6)], leaf_feat=features,
            parent_of_leaf=PARENTS, group_of_i=np.arange(6), eval_metric="accuracy",
            v0_per_task={"task": 0.5} if baselines is None else baselines,
            subj_id_to_task={0: "task"} if domains is None else domains,
            rep_per_parent=2, use_eb_shrinkage=shrink, prior_var=prior, se_floor=floor,
            interpolation_temperature=temperature,
        )


class InterpolationTests(unittest.TestCase):
    def test_cosine_weights_and_effective_count_match_closed_form(self):
        weights, n_eff = s._cosine_interpolation_weights(FEATURES[2:4], FEATURES[:2], temperature=1.)
        a = math.e / (math.e + 1.)
        np.testing.assert_allclose(weights, [[a, 1-a], [1-a, a]], atol=1e-14)
        np.testing.assert_allclose(weights.sum(axis=1), 1.)
        np.testing.assert_allclose(n_eff, 1. / (a*a+(1-a)**2))
        self.assertTrue(np.all((weights >= 0) & (weights <= 1)))

    def test_real_bootstrap_uses_leaf_specific_weights_without_shrinkage(self):
        macro, errors, means, task_errors, measured, info = bootstrap(shrink=False)
        a = math.e / (math.e + 1.)
        expected = np.array([0.4, -0.2, a*0.4+(1-a)*(-0.2),
                             (1-a)*0.4+a*(-0.2), 0.2, 0.2])
        np.testing.assert_allclose(macro, expected)
        np.testing.assert_allclose([means[g]["task"] for g in range(6)], expected)
        self.assertGreater(means[2]["task"], means[3]["task"])
        np.testing.assert_array_equal(measured, [True, True, False, False, True, False])
        neff = 1. / (a*a + (1-a)**2)
        self.assertAlmostEqual(task_errors[2]["task"]**2, 0.18 / neff)
        # The similar representative in the other parent must never contribute.
        self.assertAlmostEqual(means[5]["task"], 0.2)
        self.assertEqual(info["leaf_n_eff"][5], 1.)

    def test_real_bootstrap_eb_mean_and_variance_match_paper_oracle(self):
        macro, errors, means, task_errors, measured, info = bootstrap()
        a = math.e / (math.e + 1.)
        neff = 1. / (a*a + (1-a)**2)
        mu0 = (0.4-0.2+0.2)/3  # Pooled measured effects, not mean of parent means.
        obs_var = 0.18/neff
        rho = 0.01/(0.01+obs_var)
        self.assertAlmostEqual(means[2]["task"], rho*(a*0.4+(1-a)*(-0.2))+(1-rho)*mu0)
        self.assertAlmostEqual(means[3]["task"], rho*((1-a)*0.4+a*(-0.2))+(1-rho)*mu0)
        self.assertAlmostEqual(task_errors[2]["task"]**2, rho*obs_var)
        self.assertAlmostEqual(info["global_measured_effect_mean"]["task"], mu0)
        self.assertNotAlmostEqual(mu0, (0.1+0.2)/2)
        # Single-representative parent uses pooled within-parent variance, neff=1.
        single_rho = 0.01/(0.01+0.18)
        self.assertAlmostEqual(means[5]["task"], single_rho*0.2+(1-single_rho)*mu0)
        for gid, observed in EFFECTS.items():
            self.assertAlmostEqual(means[gid]["task"], observed)
        self.assertAlmostEqual(task_errors[0]["task"]**2, 0.18)
        np.testing.assert_allclose(macro, [means[g]["task"] for g in range(6)])

    def test_public_harp_reaches_actual_interpolation_with_configured_temperature(self):
        class Pool(MeasuredPool):
            def __init__(self, **kwargs):
                self.eval_arrays = {name: np.ones((1, 1), dtype=np.int64)
                                    for name in s._DATA_ARRAY_KEYS}
                self.eval_arrays["_subject_map"] = {"task": 0}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        groups = [np.array([g]) for g in range(6)]
        hierarchy = ([np.arange(6)], [np.arange(4), np.array([4, 5])], groups,
                     np.array([0, 0]), PARENTS, np.zeros(6, dtype=int), np.arange(6), FEATURES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"baseline.jsonl"
            path.write_text(json.dumps({"subject": "task", "acc01": 0.5,
                                       "_meta": {"bucket_name": "task"}})+"\n")
            with patch.object(s, "GPUWorkerPool", Pool), \
                 patch.object(s, "build_parent_leaf_hierarchy", return_value=hierarchy), \
                 patch.object(s, "sample_representative_leaf_groups",
                              return_value=(REPS, np.array([0, 1, 4]))):
                result = s.harp(np.eye(6), "fake", [{"input": f"train{g}"} for g in range(6)],
                    [{"input": "validation", "subject": "task"}], tokenizer=None,
                    representatives_path=str(path), hierarchy_csv_path=None,
                    interpolation_temperature=0.25, use_eb_shrinkage=False,
                    TOP_K_POINTS=6, use_knee_stop=False)
        a = math.exp(4.)/(math.exp(4.)+1.)
        self.assertAlmostEqual(result.info["mu_leaf_per_task"]["2"]["task"],
                               a*0.4+(1-a)*(-0.2), places=6)
        self.assertAlmostEqual(result.info["mu_leaf_per_task"]["3"]["task"],
                               (1-a)*0.4+a*(-0.2), places=6)
        self.assertEqual(result.info["bootstrap"]["interpolation_temperature"], .25)
        self.assertEqual(result.info["validation_protocol"], "full_fixed")

    def test_variance_and_shrinkage_are_domain_specific(self):
        class TwoDomainPool:
            def parallel_run_independent_singletons_per_task(self, tasks):
                other = {0: .05, 1: -.05, 4: .1}
                return {task[0]: {0: .5+EFFECTS[task[0]], 1: .5+other[task[0]]}
                        for task in tasks}

        _, _, means, errors, _, info = bootstrap(pool=TwoDomainPool(),
            baselines={"task": .5, "other": .5}, domains={0: "task", 1: "other"})
        a = math.e/(math.e+1.)
        neff = 1/(a*a+(1-a)**2)
        obs_var = .005/neff
        rho = .01/(.01+obs_var)
        expected = rho*(a*.05+(1-a)*(-.05))+(1-rho)*(.05-.05+.1)/3
        self.assertAlmostEqual(means[2]["other"], expected)
        self.assertAlmostEqual(errors[2]["other"]**2, rho*obs_var)
        self.assertAlmostEqual(info["within_variance_per_parent"]["0"]["other"], .005)
        self.assertNotAlmostEqual(errors[2]["other"], errors[2]["task"])

    def test_zero_embedding_uses_uniform_weights_and_single_rep_is_exact(self):
        weights, n_eff = s._cosine_interpolation_weights([[0., 0.]], FEATURES[:2], temperature=0.3)
        np.testing.assert_allclose(weights, [[0.5, 0.5]])
        np.testing.assert_allclose(n_eff, [2.])
        weights, n_eff = s._cosine_interpolation_weights([[0., 0.], [1., 0.]], [[0., 0.]], temperature=1e-300)
        np.testing.assert_allclose(weights, [[1.], [1.]])
        np.testing.assert_allclose(n_eff, [1., 1.])

    def test_temperature_extremes_are_stable_and_match_limits(self):
        for temperature in [1e-300, 1e300]:
            weights, n_eff = s._cosine_interpolation_weights([[1., 0.]], FEATURES[:2], temperature=temperature)
            self.assertTrue(np.isfinite(weights).all() and np.isfinite(n_eff).all())
            self.assertAlmostEqual(weights.sum(), 1.)
            np.testing.assert_allclose(weights, [[1., 0.]] if temperature < 1 else [[0.5, 0.5]])
        # Same zero similarities remain tied even at the smallest temperatures.
        weights, n_eff = s._cosine_interpolation_weights([[0., 0.]], FEATURES[:2], temperature=1e-300)
        np.testing.assert_allclose(weights, [[0.5, 0.5]])
        # Normalization remains finite for extreme but finite vector magnitudes.
        weights, _ = s._cosine_interpolation_weights([[1e300, 0.]], [[1e-300, 0.], [0., 1e300]], temperature=1.)
        np.testing.assert_allclose(weights, [[math.e/(math.e+1), 1/(math.e+1)]])

    def test_invalid_temperature_and_embedding_values_fail(self):
        for value in [0., -1., float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "interpolation_temperature"):
                bootstrap(temperature=value)
        for features in [np.full((6, 2), np.nan), np.ones((5, 2))]:
            with self.assertRaisesRegex(ValueError, "leaf_feat"):
                bootstrap(features=features)

    def test_single_rep_per_parent_has_finite_floor_and_no_between_parent_borrowing(self):
        # With no estimable within variance, floor² is the explicit fallback.
        result = bootstrap(shrink=False, reps=[np.array([0]), np.array([4])])
        means, task_errors = result[2], result[3]
        self.assertAlmostEqual(means[2]["task"], 0.4)
        self.assertAlmostEqual(means[5]["task"], 0.2)
        self.assertAlmostEqual(task_errors[2]["task"], 1e-3)
        self.assertTrue(np.isfinite(result[1]).all())

    def test_missing_representatives_or_partial_domains_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "same-parent representatives"):
            bootstrap(reps=[np.array([0, 1]), np.array([], dtype=int)])

        class FailedPool:
            def parallel_run_independent_singletons_per_task(self, tasks):
                return {0: {0: 0.9}, 1: {0: 0.3}}  # parent 1 measurement failed

        with self.assertRaisesRegex(RuntimeError, "incomplete or failed"):
            bootstrap(pool=FailedPool())

    def test_checkpoint_identity_covers_interpolation_inputs_and_invalidates_broadcast(self):
        kwargs = dict(leaf_groups=[np.array([g]) for g in range(6)], leaf_feat=FEATURES,
                      parent_of_leaf=PARENTS, bootstrap_seed_int=0, eval_metric="accuracy",
                      base_model_name="fake", n_train=6, n_eval=5,
                      validation_fingerprint="full-validation", rep_per_parent=2,
                      interpolation_temperature=1., use_eb_shrinkage=True,
                      prior_var=0.01, se_floor=1e-3)
        current = s._stage1b_ckpt_key(**kwargs)
        old = hashlib.sha1(
            b"full-validation-v1|G=6|sizes=1+1+1+1+1+1|seed=0|metric=accuracy|model=fake|n_train=6|n_eval=5|validation=full-validation"
        ).hexdigest()[:16]
        self.assertNotEqual(current, old)
        for name, value in [("leaf_feat", FEATURES[::-1]), ("parent_of_leaf", PARENTS[::-1]),
                            ("rep_per_parent", 1), ("interpolation_temperature", .2),
                            ("use_eb_shrinkage", False), ("prior_var", .02), ("se_floor", .02),
                            ("leaf_groups", [np.array([g]) for g in range(5, -1, -1)])]:
            updated = dict(kwargs, **{name: value})
            self.assertNotEqual(current, s._stage1b_ckpt_key(**updated), name)
        with tempfile.TemporaryDirectory() as directory:
            s._save_stage1b_ckpt(directory, "A", old, {"leaf_is_measured": np.ones(6)})
            self.assertIsNone(s._try_load_stage1b_ckpt(directory, "A", current))
            s._save_stage1b_ckpt(directory, "A", current, {"leaf_is_measured": np.ones(6)})
            self.assertIsNotNone(s._try_load_stage1b_ckpt(directory, "A", current))


class ActiveDomainTests(unittest.TestCase):
    def filter(self, phi, *, mode="macro", threshold=0.005):
        tasks = [f"d{i}" for i in range(np.asarray(phi).shape[1])]
        return s.filter_active_tasks_and_weights(np.asarray(phi), tasks,
            {t: 0.5 for t in tasks}, {t: i+1 for i, t in enumerate(tasks)}, True,
            epsilon_task=threshold, task_weight_mode=mode)

    def test_harmful_domain_retained_alongside_beneficial_with_normalized_weights(self):
        for mode, expected in [("macro", [0.5, 0.5]), ("micro", [1/3, 2/3])]:
            _, active, weights, _ = self.filter([[0.2, -0.3, 0.0], [0.1, -0.1, 0.0]], mode=mode)
            self.assertEqual(active, ["d0", "d1"])
            np.testing.assert_allclose(weights, expected)
            self.assertAlmostEqual(weights.sum(), 1.)

    def test_negative_only_domains_thresholds_and_none_active_fallback(self):
        _, active, _, _ = self.filter([[-0.2, -0.3, -0.001]])
        self.assertEqual(active, ["d0", "d1"])
        above = np.nextafter(0.005, np.inf)
        _, active, _, _ = self.filter([[0., .005, -.005, above, -above]])
        self.assertEqual(active, ["d3", "d4"])
        for mode, expected in [("macro", [1/3]*3), ("micro", [1/6, 2/6, 3/6])]:
            _, active, weights, _ = self.filter([[0., .005, -.005]], mode=mode)
            self.assertEqual(active, ["d0", "d1", "d2"])
            np.testing.assert_allclose(weights, expected)

    def test_harms_affect_both_envelopes_scores_and_greedy_ranking(self):
        phi, active, weights, baseline = self.filter([[0.4, -0.5], [0.2, 0.]])
        self.assertEqual(active, ["d0", "d1"])
        for mode in ["conservative", "optimistic"]:
            order, info = s.greedy_taskaware(phi, weights, baseline,
                [np.array([0]), np.array([1])], points_cap=2, mode=mode, use_knee_stop=False)
            self.assertEqual(order[0], 1)  # clean gain outranks larger gain with harm
            self.assertAlmostEqual(info["gains"][0], 0.1)
            self.assertLess(info["gains"][1], 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
