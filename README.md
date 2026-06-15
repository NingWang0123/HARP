# HARP — Hierarchical Active Region Pruning

Efficient, **train-based** data selection for finetuning large language models.

> HARP organizes the training pool into a node–leaf hierarchy, finetunes only a
> few *representative* leaves, infers the unmeasured leaf utilities with
> empirical-Bayes posteriors, and then greedily selects data under two
> complementary first-order envelopes — **HARP-C** and **HARP-E**.

Paper: *HARP: Efficient Data Selection for Finetuning Large Language Models*
([arXiv:2606.07690](https://arxiv.org/abs/2606.07690)).

---

## Method

HARP estimates a per-task **main effect** `phi[g, d]` for every leaf `g` and
evaluation domain `d` (how much finetuning on leaf `g` changes utility on domain
`d`), using a leaf-restricted, first-order utility approximation. Selection then
optimizes one of two envelopes, where `phi+ = max(phi, 0)`, `phi- = max(-phi, 0)`,
and `b_d` is the null (base-model) score for domain `d`:

| Variant | Predicted domain utility `ŷ_d(S)` | Intuition |
| ------- | --------------------------------- | --------- |
| **HARP-C** (conservative) | `clip[0,1]( b_d + max_{g∈S} phi+_g(d) − Σ_{g∈S} phi-_g(d) )` | credits only the **strongest** positive leaf per domain while accumulating all harms → controls redundancy |
| **HARP-E** (expansive) | `clip[0,1]( b_d + Σ_{g∈S} phi_g(d) )` | **sums** positive contributions (and subtracts harms) → recovers multiple complementary regions |

Both envelopes are optimized greedily and cut with the **best-prefix** rule:
rank leaves by greedy marginal gain, then keep the prefix that maximizes the
cumulative gain (`argmax` of the running sum). Both point selections are then
capped at the budget `TOP_K_POINTS`.

### Pipeline

1. **Stage 1a — Hierarchy.** Build a `large → parent → leaf` ANN hierarchy over
   the feature matrix `X_feat` (`build_parent_leaf_hierarchy`).
2. **Stage 1b — Main effects.** Finetune a LoRA adapter on representative leaves
   and evaluate on the proxy set; infer unmeasured leaves with empirical-Bayes
   shrinkage (`bootstrap_parent_representative_independent_per_task`).
3. **Selection.** Filter active tasks, then run the conservative (HARP-C) and
   expansive (HARP-E) greedy selections (`greedy_taskaware`) and apply the
   best-prefix cut.

> This repository implements the **first-order** method from the paper. Main
> effects are aggregated additively; there are no pairwise / second-order
> interaction terms.

---

## Install

```bash
pip install -r requirements.txt
# optional, recommended:
#   pip install faiss-gpu     # fast ANN leaf grouping (falls back to scikit-learn KMeans)
#   pip install vllm          # fast batched inner-eval during Stage 1b
```

Python ≥ 3.10.

---

## Usage

```python
from harp import harp

result = harp(
    X_feat=X_feat,                      # (n, d) features, one row per train example
    base_model_name="meta-llama/Llama-3.1-8B",
    train_examples=train_examples,      # tokenised train pool (list of dicts)
    eval_examples=eval_examples,        # tokenised proxy eval set
    tokenizer=tokenizer,
    eval_indices=eval_indices,          # or eval_indices_path="eval_indices_accuracy.npy"
    representatives_path="representatives.jsonl",
    eval_metric="accuracy",
    TOP_K_POINTS=10_000,                # point budget for the HARP-E selection
    lora_config={"r": 16, "lora_alpha": 32},
)

harp_c_idx = result.harp_c   # np.ndarray[int] — selected train-example indices (HARP-C)
harp_e_idx = result.harp_e   # np.ndarray[int] — selected train-example indices (HARP-E)
info       = result.info     # diagnostics: phi, task filtering, leaf orders, cut indices, ...
```

### Inputs

| Argument | Description |
| -------- | ----------- |
| `X_feat` | `(n, d)` L2-normalisable feature/embedding matrix, one row per train example. |
| `train_examples` / `eval_examples` | Tokenised examples (`input_ids`, `attention_mask`, `labels`, …). |
| `tokenizer` | HF tokenizer for `base_model_name`. |
| `eval_indices` / `eval_indices_path` | Proxy eval-set indices, or a path to a `.npy` of them (default `eval_indices_<metric>.npy`). |
| `representatives_path` | JSONL of per-subject base-model scores (`acc01`); supplies the null scores `b_d` and the task grouping. |
| `v0` | Base-model utility; if `None`, computed from `representatives_path`. |
| `TOP_K_POINTS` | Point budget applied to the HARP-E selection. |
| `checkpoint_dir` | If set, enables Stage-1b resume-on-preempt. |

### Output — `HarpResult`

| Field | Description |
| ----- | ----------- |
| `harp_c` | `np.ndarray[int]` of selected train-example indices under HARP-C (best-prefix cut, capped at `TOP_K_POINTS`). |
| `harp_e` | `np.ndarray[int]` of selected train-example indices under HARP-E (best-prefix cut, capped at `TOP_K_POINTS`). |
| `info` | Diagnostics: per-task main effects `phi`, active-task filtering, full conservative/expansive leaf orders, cut indices, and the bootstrap summary. The per-envelope selections are also under `info["harp_c"]` / `info["harp_e"]`. |

---

## Layout

```
harp/
  selector.py                 # the algorithm: hierarchy, bootstrap, harp(), greedy_taskaware()
  knn_grouping.py             # ANN / KMeans leaf grouping
  data_preprocess.py          # task-aware response scoring
  vllm_inner_eval.py          # optional vLLM inner-eval server
  vllm_batched_inner_eval.py  # optional vLLM batched inner-eval
```

---

## Tests

A tiny CPU-only smoke test (no GPU, no model download, no data files) covers the
import + public API, the `harp()` entry point, the HARP-C / HARP-E selection
semantics, and the Stage-1a hierarchy:

```bash
python tests/test_smoke.py        # or: pytest tests/test_smoke.py
```

## Citation

```bibtex
@article{harp2026,
  title  = {HARP: Efficient Data Selection for Finetuning Large Language Models},
  year   = {2026},
  eprint = {2606.07690},
  archivePrefix = {arXiv}
}
```
