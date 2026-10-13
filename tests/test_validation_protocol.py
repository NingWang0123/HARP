"""CPU regression tests for full fixed validation; no models, downloads, or GPUs.

Run: python -m unittest discover -s tests -p 'test_validation_protocol.py' -v
"""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from harp import selector as s
from harp import vllm_batched_inner_eval as v


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 999
    bos_token_id = None
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(self, text, **kwargs):
        token = int(text[4:]) if text.startswith("row-") else ord(text[0])
        return {"input_ids": [token]}

    def decode(self, tokens, **kwargs):
        return f"row-{max(tokens)}"

    def batch_decode(self, tokens, **kwargs):
        return ["B" if int(row[-1]) == 205 else "A" for row in tokens]


def validation_examples():
    return [
        {"input": f"row-{201+i}", "target": "A",
         "subject": "alpha" if i < 2 else "beta"}
        for i in range(5)
    ]


def validation_arrays():
    return s.precompute_dataset_arrays(validation_examples(), TinyTokenizer(),
                                       mode="generate", max_length=4)


def baseline_file(directory, examples=None):
    examples = validation_examples() if examples is None else examples
    path = Path(directory) / "baseline.jsonl"
    path.write_text("".join(json.dumps({"subject": row["subject"], "acc01": 0.5}) + "\n"
                            for row in examples))
    return str(path)


class ValidationProtocolTests(unittest.TestCase):
    def test_loader_scores_all_rows_in_fixed_order_for_every_batch_size(self):
        arrays = validation_arrays()
        for batch_size in [1, 2, 7]:
            loader = s._full_validation_loader(arrays, batch_size)
            batches = list(loader)
            ids = torch.cat([b["_real_idx"] for b in batches]).tolist()
            subjects = torch.cat([b["subjects"] for b in batches]).flatten().tolist()
            self.assertEqual(ids, list(range(5)))
            self.assertEqual(subjects, [0, 0, 1, 1, 1])

    def test_hf_scores_full_domains_without_train_or_test_rows(self):
        class Model:
            seen = []

            def eval(self):
                return self

            def generate(self, input_ids, **kwargs):
                markers = input_ids.max(dim=1).values
                self.seen.extend(markers.tolist())
                return torch.cat([input_ids, markers[:, None]], dim=1)

        arrays = validation_arrays()
        model = Model()
        with patch.dict(s._G, {"tokenizer_gen": TinyTokenizer(),
                               "eval_task_types": arrays["_task_types"],
                               "eval_gold_strings": arrays["_gold_strings"]}, clear=True):
            scores = s._eval_accuracy_per_task(
                model, torch.device("cpu"), s._full_validation_loader(arrays, 2))
        self.assertEqual(model.seen, [201, 202, 203, 204, 205])
        self.assertTrue(set(model.seen).isdisjoint({1001, 1002, 9001}))
        self.assertEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 2/3)

    def test_vllm_decoding_and_scores_use_every_domain_row(self):
        arrays = validation_arrays()
        prompts, subjects, labels, real_ids = v._decode_eval_prompts(
            s._full_validation_loader(arrays, 2), TinyTokenizer())
        self.assertEqual(real_ids, list(range(5)))
        self.assertEqual(prompts, [f"row-{i}" for i in range(201, 206)])
        scores = v._score_predictions_per_task(
            ["A", "A", "A", "A", "B"], subjects, labels, real_ids,
            arrays["_task_types"], arrays["_gold_strings"], "multi_choice")
        self.assertEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 2/3)

    def test_active_vllm_pool_path_receives_full_loader(self):
        class Workers:
            def imap_unordered(self, function, tasks, chunksize):
                return iter((task[0], "fake-adapter") for task in tasks)

            def map(self, function, tasks):
                return [None for _ in tasks]

        captured = []

        def evaluate(adapters, *, eval_dl, tokenizer, **kwargs):
            prompts, subjects, labels, ids = v._decode_eval_prompts(eval_dl, tokenizer)
            captured.extend(ids)
            return {gid: {0: 1.0, 1: 2/3} for gid in adapters}

        with tempfile.TemporaryDirectory() as directory:
            pool = s.GPUWorkerPool.__new__(s.GPUWorkerPool)
            pool.pool = Workers()
            pool._data_tmpdir = directory
            pool.eval_arrays = validation_arrays()
            pool._tokenizer_gen = TinyTokenizer()
            pool.lora_config = {"eval_batch_size": 2}
            pool.num_gpus = 1
            pool.base_model_name = "fake-model"
            with patch.object(v, "eval_all_adapters_via_vllm", side_effect=evaluate), \
                 patch.dict(os.environ, {"HARP_INNER_EVAL": "vllm",
                                         "HARP_VLLM_LOG": str(Path(directory)/"vllm.log")}):
                result = pool.parallel_run_independent_singletons_per_task(
                    [(0, np.array([0]), "accuracy", 0, "{}"),
                     (1, np.array([1]), "accuracy", 0, "{}")])
            self.assertEqual(captured, list(range(5)))
            self.assertEqual(set(result), {0, 1})

    def test_pool_keeps_train_validation_arrays_separate(self):
        tokenizer_module = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda name: TinyTokenizer()))
        train = [{"input": "row-1001", "target": "A", "subject": "training"},
                 {"input": "row-1002", "target": "A", "subject": "training"}]
        with patch.dict(sys.modules, {"transformers": tokenizer_module}):
            pool = s.GPUWorkerPool("fake-model", train, validation_examples(),
                                   lora_config={"max_length": 4}, eval_metric="accuracy",
                                   verbose=False)
        try:
            train_ids = np.load(Path(pool._data_tmpdir)/"train_input_ids.npy").max(axis=1)
            validation_ids = np.load(Path(pool._data_tmpdir)/"eval_input_ids.npy").max(axis=1)
            self.assertEqual(train_ids.tolist(), [1001, 1002])
            self.assertEqual(validation_ids.tolist(), [201, 202, 203, 204, 205])
            self.assertNotIn(9001, validation_ids)
            self.assertFalse(hasattr(pool, "eval_indices"))
        finally:
            pool.__exit__(None, None, None)
            pool.gpu_queue.close()

    def test_removed_subset_parameters_fail_instead_of_silently_filtering(self):
        for name, value in [("eval_indices", np.array([0])),
                            ("eval_indices_path", "old_proxy.npy")]:
            self.assertNotIn(name, inspect.signature(s.harp).parameters)
            with self.assertRaisesRegex(TypeError, name):
                s.harp(None, "fake", [], [], tokenizer=None, **{name: value})
        with self.assertRaisesRegex(TypeError, "eval_indices"):
            s.GPUWorkerPool("fake", [], [], eval_indices=np.array([0]))

    def test_empty_validation_and_shared_holdout_objects_fail_early(self):
        train = [{"input": "row-1001", "target": "A"}]
        with self.assertRaisesRegex(ValueError, "nonempty"):
            s.harp(np.ones((1, 2)), "fake", train, [], tokenizer=None)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            s.harp(np.ones((1, 2)), "fake", train, [train[0]], tokenizer=None)

    def test_baseline_requires_full_matching_domain_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping, means, counts, _ = s.load_task_grouping_from_representatives(
                baseline_file(directory))
            self.assertEqual(s._validate_full_validation_baseline(
                validation_examples(), mapping, means, counts), {"alpha": 2, "beta": 3})
            self.assertEqual(means, {"alpha": 0.5, "beta": 0.5})
            for rows in [validation_examples()[:-1], validation_examples()[::3]]:
                mapping, means, counts, _ = s.load_task_grouping_from_representatives(
                    baseline_file(directory, rows))
                with self.assertRaisesRegex(ValueError, "full validation"):
                    s._validate_full_validation_baseline(validation_examples(), mapping,
                                                         means, counts)

    def test_baseline_missing_domain_and_invalid_score_fail(self):
        with self.assertRaisesRegex(ValueError, "missing validation subject"):
            s._validate_full_validation_baseline(validation_examples(), {"alpha": "alpha"},
                                                 {"alpha": 0.5}, {"alpha": 2})
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            s._validate_full_validation_baseline(validation_examples(),
                {"alpha": "alpha", "beta": "beta"}, {"alpha": 0.5, "beta": 2.0},
                {"alpha": 2, "beta": 3})

    def test_validation_identity_covers_content_order_labels_domains_and_baseline(self):
        arrays = validation_arrays()
        mapping = {0: "alpha", 1: "beta"}
        baseline = {"alpha": 0.5, "beta": 0.5}
        fingerprint = s._validation_fingerprint(arrays, mapping, baseline,
                                                validation_examples())
        changes = []
        content = copy.deepcopy(arrays)
        content["input_ids"][0, -1] += 1
        changes.append((content, mapping, baseline))
        order = copy.deepcopy(arrays)
        order["input_ids"] = order["input_ids"][[1, 0, 2, 3, 4]]
        changes.append((order, mapping, baseline))
        labels = copy.deepcopy(arrays)
        labels["_gold_strings"][0] = ["B"]
        changes.append((labels, mapping, baseline))
        changes.append((arrays, {0: "beta", 1: "alpha"}, baseline))
        changes.append((arrays, mapping, {"alpha": 0.6, "beta": 0.5}))
        for data, domains, scores in changes:
            self.assertNotEqual(fingerprint, s._validation_fingerprint(
                data, domains, scores, validation_examples()))
        raw_rows = validation_examples()
        raw_rows[0]["input"] = "prefix omitted by token truncation " + raw_rows[0]["input"]
        self.assertNotEqual(fingerprint, s._validation_fingerprint(
            arrays, mapping, baseline, raw_rows))

    def test_legacy_subset_checkpoint_rejected_and_current_checkpoint_resumes(self):
        kwargs = dict(leaf_groups=[np.array([0, 1])], bootstrap_seed_int=0,
                      eval_metric="accuracy", base_model_name="fake", n_train=2, n_eval=5)
        legacy = hashlib.sha1(
            b"G=1|sizes=2|seed=0|metric=accuracy|model=fake|n_train=2|n_eval=5"
        ).hexdigest()[:16]
        fingerprint = s._validation_fingerprint(validation_arrays(),
            {0: "alpha", 1: "beta"}, {"alpha": 0.5, "beta": 0.5}, validation_examples())
        current = s._stage1b_ckpt_key(**kwargs, validation_fingerprint=fingerprint)
        self.assertNotEqual(legacy, current)
        with tempfile.TemporaryDirectory() as directory:
            payload = {"leaf_is_measured": np.array([True]), "mu_leaf": np.array([0.2])}
            s._save_stage1b_ckpt(directory, "A", legacy, payload)
            self.assertIsNone(s._try_load_stage1b_ckpt(directory, "A", current))
            s._save_stage1b_ckpt(directory, "A", current, payload)
            self.assertIsNotNone(s._try_load_stage1b_ckpt(directory, "A", current))

    def test_public_harp_uses_full_validation_without_an_index_file(self):
        train = [{"input": f"row-{1001+i}", "target": "A"} for i in range(4)]
        groups = [np.array([0, 1]), np.array([2, 3])]
        hierarchy = ([np.arange(4)], [np.arange(4)], groups, np.array([0]),
                     np.array([0, 0]), np.array([0, 0]), np.array([0, 0, 1, 1]),
                     np.eye(2))
        seen = []

        class Pool:
            def __init__(self, *, eval_examples, **kwargs):
                seen.extend(eval_examples)
                self.eval_arrays = s.precompute_dataset_arrays(
                    eval_examples, TinyTokenizer(), mode="generate", max_length=4)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        bootstrap = (np.array([0.2, 0.2]), np.zeros(2),
                     {0: {"alpha": 0.2, "beta": 0.0},
                      1: {"alpha": 0.0, "beta": 0.2}},
                     {0: {"alpha": 0.0, "beta": 0.0},
                      1: {"alpha": 0.0, "beta": 0.0}},
                     np.ones(2, dtype=bool), {"bootstrap": {}})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(s, "build_parent_leaf_hierarchy", return_value=hierarchy), \
             patch.object(s, "GPUWorkerPool", Pool), \
             patch.object(s, "bootstrap_parent_representative_independent_per_task",
                          return_value=bootstrap), \
             patch.object(np, "load", side_effect=AssertionError("obsolete index-file load")):
            result = s.harp(np.eye(4), "fake", train, validation_examples(),
                            tokenizer=TinyTokenizer(), representatives_path=baseline_file(directory),
                            hierarchy_csv_path=None, TOP_K_POINTS=4, use_knee_stop=False)
        self.assertEqual(seen, validation_examples())
        self.assertEqual(result.info["n_validation"], 5)
        self.assertEqual(result.info["validation_protocol"], "full_fixed")
        self.assertTrue(set(result.harp_c).issubset(range(4)))
        self.assertTrue(set(result.harp_e).issubset(range(4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
