import math
import os
import shutil
import tempfile
import csv
import time
import functools
import traceback
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
import json
import re
import pickle
import hashlib
import numpy as np
import torch
import torch.multiprocessing as mp


# ============================================================================
# Stage-1b checkpointing
# ----------------------------------------------------------------------------
# The two expensive phases of Stage 1b are (a) the per-leaf bootstrap (LoRA
# train + eval over every leaf group) and (b) the pair-union measurements
# (LoRA train + eval per selected pair). Together they dominate Stage 1b
# wall-clock. We checkpoint both so a preempt-then-resume cycle skips them.
#
#   CKPT_A:  output of bootstrap — saves ~60-70% of Stage 1b
#   CKPT_B:  CKPT_A + pair-union measurements — saves ~90%+ of Stage 1b
#
# Cache key is a hash of the inputs that determine those expensive phases. If
# the user changes leaf size, seed, model, or dataset, the key drifts and the
# old cache is ignored.
# ============================================================================

def _stage1b_ckpt_path(checkpoint_dir: str, level: str) -> str:
    return os.path.join(checkpoint_dir, f"stage1b_ckpt_{level}.pkl")


def _stage1b_ckpt_key(*, leaf_groups, bootstrap_seed_int, eval_metric,
                      base_model_name, n_train, n_eval, validation_fingerprint) -> str:
    """Versioned Stage-1b key including the complete fixed validation identity.

    Legacy proxy-subset checkpoints and changed validation/baseline inputs
    cannot reuse measurements from the earlier protocol.
    """
    sizes = "+".join(str(int(len(g))) for g in leaf_groups)
    raw = f"full-validation-v1|G={len(leaf_groups)}|sizes={sizes}|seed={bootstrap_seed_int}" \
          f"|metric={eval_metric}|model={base_model_name}|n_train={n_train}|n_eval={n_eval}" \
          f"|validation={validation_fingerprint}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _save_stage1b_ckpt(checkpoint_dir: Optional[str], level: str, key: str,
                       payload: Dict[str, Any]) -> None:
    if not checkpoint_dir:
        return
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = _stage1b_ckpt_path(checkpoint_dir, level)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "level": level, **payload}, f, protocol=4)
        os.replace(tmp, path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[stage1b-ckpt] saved CKPT-{level} → {path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[stage1b-ckpt] save CKPT-{level} failed: {e}", flush=True)


def _try_load_stage1b_ckpt(checkpoint_dir: Optional[str], level: str,
                           key: str) -> Optional[Dict[str, Any]]:
    """Load CKPT-A/B if it matches `key` AND survived a staleness check.

    Guard: if a prior bootstrap run was killed mid-flight (OOM, scancel),
    it may have saved a CKPT with most leaves at mu=0 / leaf_is_measured=False.
    Loading that cache silently turns a failed run into "successful" garbage
    output — the stage1_only fast path makes this visible (S0 picks from
    the few measured leaves, S0_info["points_total"] looks plausible, but
    every leaf's per-task phi is zero).

    Threshold via HARP_CKPT_MIN_MEASURED_FRAC (default 0.10). Ckpt is
    rejected if measured_leaf_frac falls below it.

    Default 0.10 reflects empirical reality: a healthy bootstrap with
    rep_per_parent=3 covers ~36-42% of total leaves (the rest inherit the
    parent's gain). 50% would reject every legitimate run. The poisoned
    OOM cache from earlier (618699 / 620669) had ~5% of leaves measured,
    so 0.10 catches it cleanly while letting real runs through.
    """
    if not checkpoint_dir:
        return None
    path = _stage1b_ckpt_path(checkpoint_dir, level)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        if ckpt.get("key") != key:
            print(f"[stage1b-ckpt] CKPT-{level} key mismatch (cached={ckpt.get('key')}, "
                  f"current={key}); ignoring", flush=True)
            return None

        # Staleness guard
        leaf_is_measured = ckpt.get("leaf_is_measured")
        if leaf_is_measured is not None:
            try:
                meas = np.asarray(leaf_is_measured, dtype=bool)
                frac = float(np.mean(meas)) if meas.size > 0 else 0.0
            except Exception:
                frac = 0.0
            min_frac = float(os.environ.get("HARP_CKPT_MIN_MEASURED_FRAC", "0.10"))
            if frac < min_frac:
                print(f"[stage1b-ckpt] CKPT-{level} REJECTED — only {frac:.1%} of "
                      f"leaves measured (threshold {min_frac:.0%}). Likely a stale "
                      f"OOM-poisoned cache; re-running bootstrap. To override set "
                      f"HARP_CKPT_MIN_MEASURED_FRAC=0.0", flush=True)
                return None

        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[stage1b-ckpt] loaded CKPT-{level} from {path} ({size_mb:.1f} MB)",
              flush=True)
        return ckpt
    except Exception as e:
        print(f"[stage1b-ckpt] load CKPT-{level} failed: {e}; ignoring", flush=True)
        return None
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import tqdm
from collections import defaultdict
from dataclasses import dataclass

from .knn_grouping import build_groups_from_shapley_ann, sample_representative_leaf_groups

logging.basicConfig(level=logging.INFO)


# ============================================================
# timing helper
# ============================================================
def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            logging.info("⏱ %s took %.3fs", fn.__qualname__, dt)
    return wrapper


# ============================================================
# Token cache arrays
# ============================================================
@timeit
def precompute_dataset_arrays(
    examples: List[Dict[str, str]],
    tokenizer,
    mode: str = "train",
    max_length: int = 256
) -> Dict[str, np.ndarray]:
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    mc_labels_list = []  
    subjects_list = []
    
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    subject_map = defaultdict(int)

    # 1. 关键设置：获取 EOS Token ID
    # 如果没有 EOS，模型不知道什么时候说完，target 也会失效
    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id
    has_bos = bos_token_id is not None
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id # 兜底

    # 设置 Padding 方向
    if mode == "generate": 
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    for ex in examples:
        prompt = ex.get("input", "")
        target = ex.get("target", "")
        subject = str(ex.get("subject", "") or "").strip()
        
        if subject not in subject_map:
            subject_map[subject] = len(subject_map)
        subj_id = subject_map[subject]

        # =======================================================
        # 【核心修改】分别编码，物理拼接，杜绝 Target 消失
        # =======================================================
        
        # A. 编码 Prompt (纯净版)
        prompt_tokens = tokenizer(prompt, truncation=False, padding=False, add_special_tokens=False)
        prompt_ids = prompt_tokens["input_ids"]
        
        if has_bos:
            prompt_ids = [bos_token_id] + prompt_ids
        
        # B. 编码 Target (纯净版)
        if mode == "generate":
            target_ids = [] 
        else:
            # 训练/Loss模式：一定要给 Target 加上 EOS！
            # 哪怕 target 是空字符串，加上 EOS 后长度也是 1，Loss 就不会 NaN
            target_tokens = tokenizer(target, truncation=False, padding=False, add_special_tokens=False)
            target_ids = target_tokens["input_ids"] + [eos_token_id]

        # C. 拼接与截断逻辑
        if mode == "generate":
            # Generate 模式：只留 Prompt
            input_ids = prompt_ids
            # 如果超长，截断左边（保留最新上下文）
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
            labels = [-100] * len(input_ids)
            
        else:
            # Train 模式：Prompt + Target
            # 必须保证 Target 完整，优先截断 Prompt
            max_prompt_len = max_length - len(target_ids)
            
            if max_prompt_len <= 0:
                # 极端情况：Target 太长，Prompt 没地儿了
                target_ids = target_ids[:max_length]
                prompt_ids = []
            elif len(prompt_ids) > max_prompt_len:
                # 截断 Prompt 前面
                prompt_ids = prompt_ids[-max_prompt_len:] # 取后半截
                # 或者 prompt_ids[:max_prompt_len] 取前半截，看你需求
            
            # 【物理拼接】 List + List
            input_ids = prompt_ids + target_ids
            
            # 【制作 Labels】
            # Prompt 部分给 -100
            # Target 部分保留原样
            labels = [-100] * len(prompt_ids) + target_ids

        # D. 执行 Padding (手动 Pad 到 numpy 矩阵)
        pad_len = max_length - len(input_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        if pad_len > 0:
            if tokenizer.padding_side == "left":
                input_ids = [pad_id] * pad_len + input_ids
                labels = [-100] * pad_len + labels
                attention_mask = [0] * pad_len + [1] * (max_length - pad_len)
            else:
                input_ids = input_ids + [pad_id] * pad_len
                labels = labels + [-100] * pad_len
                attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
        else:
            # 刚好填满
            attention_mask = [1] * max_length

        # 存入列表
        ids_np = np.array(input_ids, dtype=np.int64)
        mask_np = np.array(attention_mask, dtype=np.int64)
        labels_np = np.array(labels, dtype=np.int64)
        
        input_ids_list.append(ids_np)
        attention_mask_list.append(mask_np)
        labels_list.append(labels_np)
        subjects_list.append(subj_id)

        # MC Labels (保持不变)
        t = str(target).strip().upper()
        if len(t) > 2:
            mc = 0
        elif t in letter_map:
            mc = letter_map[t]
        elif t.isdigit():
            mc = int(t)
        else:
            mc = 0
        mc_labels_list.append(mc)

    arrays = {
        "input_ids": np.array(input_ids_list, dtype=np.int64),
        "attention_mask": np.array(attention_mask_list, dtype=np.int64),
        "labels": np.array(labels_list, dtype=np.int64),
        "mc_labels": np.array(mc_labels_list, dtype=np.int64).reshape(-1, 1),
        "subjects": np.array(subjects_list, dtype=np.int64).reshape(-1, 1),
    }
    arrays["_subject_map"] = dict(subject_map)  # {subject_name_str: int_id}

    # Task-type metadata (for non-multi-choice scoring like SQuAD F1, GSM8K numeric, MATH boxed)
    task_types = []
    gold_strings = []  # list-of-lists of accepted answer strings (for extractive); else [target]
    for ex in examples:
        tt = str(ex.get("task_type", "multi_choice"))
        task_types.append(tt)
        ans = ex.get("answers")
        if isinstance(ans, list) and ans:
            gold_strings.append([str(a) for a in ans])
        else:
            t = str(ex.get("target", "")).strip()
            gold_strings.append([t] if t else [])
    arrays["_task_types"] = task_types
    arrays["_gold_strings"] = gold_strings
    return arrays
    


class CachedTensorDataset(Dataset):
    def __init__(self, indices: np.ndarray, data_arrays: Dict[str, np.ndarray]):
        self.indices = np.asarray(indices, dtype=int)
        self.data = data_arrays

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self.indices[idx])
        return {
            "input_ids": torch.from_numpy(self.data["input_ids"][real_idx].copy()).long(),
            "attention_mask": torch.from_numpy(self.data["attention_mask"][real_idx].copy()).long(),
            "labels": torch.from_numpy(self.data["labels"][real_idx].copy()).long(),
            "mc_labels": torch.from_numpy(self.data["mc_labels"][real_idx].copy()).long(),
            "subjects": torch.from_numpy(self.data["subjects"][real_idx].copy()).long(),
            "_real_idx": torch.tensor(real_idx, dtype=torch.long),
        }


def _full_validation_loader(data_arrays: Dict[str, np.ndarray], batch_size: int):
    """Score every supplied validation row once, in the same fixed order."""
    n_eval = len(data_arrays["input_ids"])
    if n_eval == 0:
        raise ValueError("HARP requires a nonempty fixed validation set.")
    dataset = CachedTensorDataset(np.arange(n_eval, dtype=int), data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _validate_validation_split(train_examples, eval_examples):
    if not eval_examples:
        raise ValueError("HARP requires a nonempty fixed validation set.")
    train_objects = {id(row) for row in train_examples}
    if any(id(row) in train_objects for row in eval_examples):
        raise ValueError(
            "Training and validation examples must be disjoint; shared example "
            "objects were supplied. Remove held-out validation rows from the training pool."
        )


def _validate_full_validation_baseline(eval_examples, subject_to_task, v0_per_task,
                                       task_counts):
    """Require one baseline score per full validation row, in matching domains."""
    expected = defaultdict(int)
    for row in eval_examples:
        subject = str(row.get("subject", "") or "").strip()
        if subject not in subject_to_task:
            raise ValueError(f"Baseline scores are missing validation subject {subject!r}.")
        expected[subject_to_task[subject]] += 1
    expected = dict(expected)
    if set(v0_per_task) != set(expected) or task_counts != expected:
        raise ValueError(
            "Baseline scores must cover the same full validation domains and row counts: "
            f"expected {expected}, got {task_counts}. Regenerate representatives_path "
            "with one base-model score per validation example; proxy subsets and "
            "one-row-per-domain summaries are not accepted."
        )
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0
           for score in v0_per_task.values()):
        raise ValueError("Full-validation baseline scores must be finite values in [0, 1].")
    return expected


def _validation_fingerprint(eval_arrays, subj_id_to_task, v0_per_task, eval_examples):
    """Identify fixed validation content/order, domains, scoring metadata and baselines."""
    digest = hashlib.sha256()
    for name in _DATA_ARRAY_KEYS:
        array = np.ascontiguousarray(eval_arrays[name])
        digest.update(json.dumps([name, str(array.dtype), array.shape]).encode("utf-8"))
        digest.update(array.tobytes())
    metadata = {
        "validation_rows": eval_examples,
        "task_types": eval_arrays.get("_task_types", []),
        "gold_strings": eval_arrays.get("_gold_strings", []),
        "subject_map": eval_arrays.get("_subject_map", {}),
        "domain_mapping": subj_id_to_task,
        "baseline_scores": v0_per_task,
    }
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


# ============================================================
# math utilities
# ============================================================


def eb_shrinkage_from_obsvar(
    cmean: float,
    obs_var: float,          # variance of observation (mean estimator)
    global_mean: float,
    prior_var: float,        # between-parent variance tau^2
) -> Tuple[float, float]:
    """
    Normal–Normal posterior with plug-in hyperparams:

      y | θ ~ N(θ, obs_var)
      θ     ~ N(global_mean, prior_var)

    Returns:
      post_mean, post_var  (posterior for θ)
    """
    obs_var = max(float(obs_var), 1e-12)
    prior_var = max(float(prior_var), 1e-12)

    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (global_mean / prior_var + float(cmean) / obs_var)
    return float(post_mean), float(post_var)


# ============================================================
# Embedding normalization + diverse sampling
# ============================================================
@timeit
def l2_normalize_cuda_if_available(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, copy=False)
    try:
        import torch.nn.functional as F
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Xt = torch.from_numpy(X).to(device)
        Xt = F.normalize(Xt, p=2, dim=1, eps=eps)
        return Xt.detach().cpu().numpy()
    except Exception:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return X / norms



# ============================================================
# LoRA helpers
# ============================================================
def infer_lora_targets(model) -> List[str]:
    names = set()
    for n, _ in model.named_modules():
        names.add(n.split(".")[-1] if "." in n else n)
    common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    gpt2 = ["c_attn", "c_proj"]
    picked = [m for m in common if m in names]
    if not picked:
        picked = [m for m in gpt2 if m in names]
    if not picked:
        picked = [m for m in ["proj", "out_proj"] if m in names]
    return picked


def _extract_lora_param_tensors(model) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, p in model.named_parameters():
        if ("lora_" in name) or (".lora_A." in name) or (".lora_B." in name):
            sd[name] = p.detach().cpu().clone()
    return sd


def _reset_lora_params_from_snapshot(model, snap: Dict[str, torch.Tensor], device: torch.device) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snap:
                p.copy_(snap[name].to(device, non_blocking=True))



def _eval_accuracy_per_task(model, device, eval_dl) -> Dict[int, float]:
    """
    Per-task evaluation: returns Dict[subject_id_int, score] for each subject/task
    present in the eval set. Macro avg = mean of these.

    Task-type-aware: dispatches scoring to score_response (multi_choice / numeric /
    latex_boxed / extractive) based on per-example task_type carried via _G.
    Scores are continuous in [0, 1] (e.g., F1 for SQuAD); the per-task average is
    treated as accuracy by HARP downstream.
    """
    model.eval()

    tokenizer = _G.get("tokenizer_gen")
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    eval_task_types = _G.get("eval_task_types", [])
    eval_gold_strings = _G.get("eval_gold_strings", [])
    # Predominant task type — used to set max_new_tokens
    if eval_task_types:
        unique_tts = set(eval_task_types)
        primary_tt = next(iter(unique_tts)) if len(unique_tts) == 1 else "mixed"
    else:
        primary_tt = "multi_choice"

    # Match Stage 3 vLLM generation budgets so validation utility is predictive
    # of final accuracy. Mismatched token caps were a real bug — small
    # caps truncated CoT-trained models before they emitted the answer letter.
    if primary_tt in ("numeric", "latex_boxed"):
        max_new_tokens = 512
    elif primary_tt == "extractive":
        max_new_tokens = 64
    else:
        max_new_tokens = 256

    # Lazy import to avoid circular dependency at module load
    from .data_preprocess import score_response

    letters = ["A", "B", "C", "D"]
    subject_score_sum = defaultdict(float)
    subject_total = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dl, desc="Gen Eval PerTask"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels_tensor = batch["mc_labels"]
            subjects_tensor = batch["subjects"]
            real_indices = batch.get("_real_idx")  # added by CachedTensorDataset below

            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            input_len = input_ids.shape[1]
            new_tokens = gen_outputs[:, input_len:]
            decoded_preds = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for i, pred_text in enumerate(decoded_preds):
                subj_id = int(subjects_tensor[i].item())
                subject_total[subj_id] += 1

                # Determine task_type for this example
                if real_indices is not None and eval_task_types:
                    real_idx = int(real_indices[i].item())
                    tt = eval_task_types[real_idx] if real_idx < len(eval_task_types) else "multi_choice"
                else:
                    tt = primary_tt if primary_tt != "mixed" else "multi_choice"

                if tt == "multi_choice":
                    gold_idx = int(labels_tensor[i].item())
                    gold_char = letters[gold_idx] if 0 <= gold_idx < len(letters) else ""
                    s = score_response("multi_choice", pred_text, gold_char)
                else:
                    real_idx = int(real_indices[i].item()) if real_indices is not None else -1
                    if 0 <= real_idx < len(eval_gold_strings):
                        gold_for_ex = eval_gold_strings[real_idx]
                    else:
                        gold_for_ex = []
                    if tt == "extractive":
                        s = score_response("extractive", pred_text, gold_for_ex)
                    else:  # numeric / latex_boxed
                        g = gold_for_ex[0] if gold_for_ex else ""
                        s = score_response(tt, pred_text, g)

                subject_score_sum[subj_id] += float(s)

    per_task_acc = {}
    for subj_id, n in subject_total.items():
        if n > 0:
            per_task_acc[int(subj_id)] = subject_score_sum[subj_id] / n

    macro_avg = float(np.mean(list(per_task_acc.values()))) if per_task_acc else 0.0
    print(f"Eval PerTask: {len(per_task_acc)} tasks, Macro={macro_avg:.2%}", flush=True)
    return per_task_acc


# ============================================================
# Multiprocessing GPU worker pool (perm-run)
# ============================================================
# Per-worker global state. Populated by `_init_worker_process` in each GPU
# worker process; `vllm_batched_inner_eval.py` imports this same object to read
# the per-worker model / LoRA snapshot. This module is the single source of
# truth for `_G`.
_G: Dict[str, Any] = {}


_DATA_ARRAY_KEYS = ("input_ids", "attention_mask", "labels", "mc_labels", "subjects")


@timeit
def _init_worker_process(
    gpu_queue,
    base_model_name: str,
    data_tmpdir: str,
    lora_hparams: Dict[str, Any],
    num_gpus: int,
    verbose: bool,
):
    try:
        gpu_id = gpu_queue.get(timeout=10)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        _G.clear()
        _G["gpu_id"] = gpu_id
        _G["device"] = device
        _G["lora_hparams"] = dict(lora_hparams)
        _G["verbose"] = bool(verbose)
        _G["base_model_name"] = base_model_name  # for vLLM-batched re-init helper

        # Load data arrays via memory-mapped files (zero-copy shared across workers)
        train_data_arrays = {
            k: np.load(os.path.join(data_tmpdir, f"train_{k}.npy"), mmap_mode="r")
            for k in _DATA_ARRAY_KEYS
        }
        eval_data_arrays = {
            k: np.load(os.path.join(data_tmpdir, f"eval_{k}.npy"), mmap_mode="r")
            for k in _DATA_ARRAY_KEYS
        }
        # Load task-type metadata for non-multi-choice scoring (SQuAD F1, etc.)
        for split_name, target_dict in [("train", train_data_arrays), ("eval", eval_data_arrays)]:
            meta_path = os.path.join(data_tmpdir, f"{split_name}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as fh:
                    m = json.load(fh)
                target_dict["_task_types"] = list(m.get("task_types", []))
                target_dict["_gold_strings"] = list(m.get("gold_strings", []))
            else:
                target_dict["_task_types"] = []
                target_dict["_gold_strings"] = []

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        tok = AutoTokenizer.from_pretrained(base_model_name)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _G["tokenizer"] = tok
        
        tok_gen = AutoTokenizer.from_pretrained(base_model_name)
        tok_gen.padding_side = "left"
        if tok_gen.pad_token is None:
            tok_gen.pad_token = tok_gen.eos_token
        _G["tokenizer_gen"] = tok_gen

        torch_dtype = torch.float16
        kwargs = {}
        attn_impl = _G["lora_hparams"].get("attn_implementation", None)
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map={"": f"cuda:{gpu_id}"},
            **kwargs,
        )
        base_model.eval()

        targets = infer_lora_targets(base_model)
        if _G["verbose"]:
            print(f"[Worker {gpu_id}] device={device} LoRA targets={targets}")

        cfg = LoraConfig(
            r=int(_G["lora_hparams"].get("lora_r", 8)),
            lora_alpha=int(_G["lora_hparams"].get("lora_alpha", 16)),
            lora_dropout=float(_G["lora_hparams"].get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets if targets else None,
        )

        peft_model = get_peft_model(base_model, cfg)
        peft_model.train()

        # Gradient checkpointing — required for the bootstrap worker to fit on
        # 48 GB GPUs at BS×ml > 2048 tokens. Without it, autograd saves all
        # forward activations across all 36 Qwen3-4B layers (~38 GiB at
        # BS=4×ml=1024); with it, peak drops to ~17 GiB.
        #
        # Trade-off: ckpt is ~36% slower per step than no-ckpt at the same BS.
        # Default OFF: bench (scripts/diag/bench_lora_4b.py, run 655439) shows
        # noCkpt + lower BS is faster than ckpt + higher BS at every ml on a
        # 4B model. Set HARP_GRAD_CKPT=1 to re-enable for a specific run that
        # needs the memory headroom (e.g. very long sequences or a bigger model).
        if os.environ.get("HARP_GRAD_CKPT", "0") == "1":
            peft_model.enable_input_require_grads()
            peft_model.gradient_checkpointing_enable()
            print("[Worker] gradient_checkpointing ENABLED", flush=True)
        else:
            print("[Worker] gradient_checkpointing DISABLED (default)", flush=True)

        snap = _extract_lora_param_tensors(peft_model)

        _G["base_model"] = base_model
        _G["peft_model"] = peft_model
        _G["lora_snapshot"] = snap

        _G["train_data"] = train_data_arrays

        # Expose task-type metadata for downstream score_response dispatch
        _G["eval_task_types"] = list(eval_data_arrays.get("_task_types", []))
        _G["eval_gold_strings"] = list(eval_data_arrays.get("_gold_strings", []))
        _G["train_task_types"] = list(train_data_arrays.get("_task_types", []))
        _G["train_gold_strings"] = list(train_data_arrays.get("_gold_strings", []))

        eval_bs = int(_G["lora_hparams"].get("eval_batch_size",
                                              os.environ.get("HARP_EVAL_BS", 64)))
        _G["eval_dl"] = _full_validation_loader(eval_data_arrays, eval_bs)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception:
        traceback.print_exc()
        raise


def _train_epochs_on_indices(
    model,
    device: torch.device,
    train_data: Dict[str, np.ndarray],
    idxs: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_steps: int = 0,
    show_progress: bool = False,
    seed: int = 0,
    grad_accum: int = 1,
) -> None:
    """
    Train LoRA on the given indices.

    If max_steps > 0, stop after that many gradient steps regardless of
    epochs.  This normalizes compute across groups of different sizes so
    that Shapley φ is not biased by training duration.
    """
    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0 or int(epochs) <= 0:
        return

    ds = CachedTensorDataset(idxs, train_data)

    g = torch.Generator()
    g.manual_seed(int(seed))

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        generator=g,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
    )

    epoch_iter = range(int(epochs))
    if show_progress:
        epoch_iter = tqdm.tqdm(epoch_iter, desc="Training", leave=False)

    step_count = 0
    micro_count = 0
    grad_accum = max(1, int(grad_accum))
    for _ in epoch_iter:
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            model_input = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
            out = model(**model_input)
            (out.loss / grad_accum).backward()
            micro_count += 1
            if micro_count % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                if max_steps > 0 and step_count >= max_steps:
                    break
        if max_steps > 0 and step_count >= max_steps:
            break
    # Flush any remaining accumulated gradients
    if micro_count % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()



def _worker_run_independent_singleton_per_task(
    task: Tuple[int, np.ndarray, str, int, str]
) -> Tuple[int, Dict[int, float]]:
    """
    Per-task variant: returns (gid, {subj_id: accuracy}) after training on this group.
    task = (gid, idxs, eval_metric, seed_int, v0_per_task_json)
    v0_per_task_json is a JSON string of the per-task null scores (not used here,
    we return raw per-task accuracy, the caller computes the diff).
    """
    gid, idxs, eval_metric, seed_int, _ = task
    try:
        device: torch.device = _G["device"]
        model = _G["peft_model"]
        snap = _G["lora_snapshot"]
        h = _G["lora_hparams"]
        train_data = _G["train_data"]
        eval_dl = _G["eval_dl"]

        # Per-group seed: XOR base seed with gid for differentiation
        seed_int = int(seed_int) ^ int(gid)
        torch.manual_seed(seed_int)
        np.random.seed(seed_int % (2**32 - 1))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed_int)

        lr = float(h.get("lr", 2e-4))
        epochs = int(h.get("epochs_per_player", h.get("num_epochs", 1)))
        batch_size = int(h.get("batch_size", 4))
        grad_accum = int(h.get("grad_accum", 1))
        max_steps = int(h.get("max_steps_per_player", 0))

        idxs = np.asarray(idxs, dtype=int)
        if idxs.size == 0 or epochs <= 0:
            return int(gid), {}

        _reset_lora_params_from_snapshot(model, snap, device)

        _train_epochs_on_indices(
            model, device, train_data, idxs,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_steps=max_steps,
            show_progress=False,
            seed=seed_int,
            grad_accum=grad_accum,
        )

        # Use per-task eval
        per_task_acc = _eval_accuracy_per_task(model, device, eval_dl)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return int(gid), per_task_acc

    except Exception as e:
        gpu_id = _G.get("gpu_id", None)
        print(f"[Worker gpu={gpu_id}] per-task-singleton gid={gid} error: {e}")
        traceback.print_exc()
        return int(gid), {}


class GPUWorkerPool:
    def __init__(
        self,
        base_model_name: str,
        train_examples: List[Dict],
        eval_examples: List[Dict],
        *,
        num_gpus: Optional[int] = None,
        lora_config: Optional[Dict] = None,
        verbose: bool = True,
        eval_metric: str = "loss",
    ):
        _validate_validation_split(train_examples, eval_examples)
        self.base_model_name = base_model_name
        self.verbose = bool(verbose)
        self.lora_config = dict(lora_config or {})
        self.eval_metric = eval_metric
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.num_gpus = max(1, int(num_gpus) if torch.cuda.is_available() else 1)

        # Simple mp.Queue — no need for a heavyweight Manager process
        self.gpu_queue = mp.Queue()
        for i in range(self.num_gpus):
            self.gpu_queue.put(i)

        if self.verbose:
            print("[GPUWorkerPool] Pre-tokenizing data (once)...")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        max_len = int(self.lora_config.get("max_length", 256))

        tok_gen = AutoTokenizer.from_pretrained(base_model_name)
        tok_gen.padding_side = "left"
        if tok_gen.pad_token is None:
            tok_gen.pad_token = tok_gen.eos_token

        train_arrays = precompute_dataset_arrays(train_examples, tok, mode="train", max_length=max_len)
        if self.eval_metric == "accuracy":
            mode = "generate"
        else:
            mode = "train"
        eval_arrays = precompute_dataset_arrays(eval_examples, tok, mode=mode, max_length=max_len)
        self.eval_arrays = eval_arrays

        # Save arrays to temp dir so workers can mmap them (zero-copy shared memory).
        # Honor HARP_TMPDIR (set by run_v10rev_combo.sh to /scratch/<user>/harp_tmp/<job>)
        # so this dir + any harp_adapters_* subdir created off it both land off the
        # node-local /tmp (typically 15 GB on shared nodes; fills fast under retry).
        _harp_tmpdir = os.environ.get("HARP_TMPDIR")
        if _harp_tmpdir and os.path.isdir(_harp_tmpdir):
            self._data_tmpdir = tempfile.mkdtemp(prefix="gpu_pool_data_", dir=_harp_tmpdir)
        else:
            self._data_tmpdir = tempfile.mkdtemp(prefix="gpu_pool_data_")
        for key, arr in train_arrays.items():
            if key.startswith("_") or not isinstance(arr, np.ndarray):
                continue  # skip metadata; saved as JSON below
            np.save(os.path.join(self._data_tmpdir, f"train_{key}.npy"), arr)
        for key, arr in eval_arrays.items():
            if key.startswith("_") or not isinstance(arr, np.ndarray):
                continue
            np.save(os.path.join(self._data_tmpdir, f"eval_{key}.npy"), arr)
        # Persist non-numpy task metadata (task_types + gold_strings) for workers
        for split_name, arrays_dict in [("train", train_arrays), ("eval", eval_arrays)]:
            meta = {
                "task_types": list(arrays_dict.get("_task_types", []) or []),
                "gold_strings": list(arrays_dict.get("_gold_strings", []) or []),
            }
            with open(os.path.join(self._data_tmpdir, f"{split_name}_meta.json"), "w") as fh:
                json.dump(meta, fh)
        if self.verbose:
            print(f"[GPUWorkerPool] Data arrays saved to {self._data_tmpdir}")

        self.base_model_name = base_model_name
        # Stash tokenizer_gen for main-process eval_dl reconstruction (used by
        # the vLLM-batched inner-eval path; harmless otherwise).
        self._tokenizer_gen = tok_gen
        self.pool = None

    def __enter__(self):
        if self.verbose:
            print(f"[GPUWorkerPool] Spawning {self.num_gpus} workers...")

        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.num_gpus,
            initializer=_init_worker_process,
            initargs=(
                self.gpu_queue,
                self.base_model_name,
                self._data_tmpdir,
                self.lora_config,
                self.num_gpus,
                self.verbose,
            ),
            maxtasksperchild=None,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pool:
            if self.verbose:
                print("[GPUWorkerPool] Closing pool...")
            self.pool.close()
            self.pool.join()
        # Clean up temp data files
        if hasattr(self, "_data_tmpdir") and os.path.isdir(self._data_tmpdir):
            shutil.rmtree(self._data_tmpdir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



    @timeit
    def parallel_run_independent_singletons_per_task(
        self,
        tasks: List[Tuple[int, np.ndarray, str, int, str]]
    ) -> Dict[int, Dict[int, float]]:
        """
        Returns dict: gid -> {subj_id: accuracy}

        Two execution paths:
          1. HF interleaved (default): each worker trains AND evals per leaf,
             returns (gid, per_task_acc). Memory-light but eval is HF-slow.
          2. vLLM batched (HARP_INNER_EVAL=vllm): workers train all leaves and
             save adapters; main process spawns vLLM after train phase, batch
             evaluates all adapters. Single-GPU, ~3× faster overall.

        Empty results (crashed tasks) are excluded with a warning.
        """
        print(f"[parallel_run] entered with {len(tasks)} tasks, HARP_INNER_EVAL={os.environ.get('HARP_INNER_EVAL', '')!r}", flush=True)
        if len(tasks) == 0:
            return {}
        if self.pool is None:
            raise RuntimeError("Pool is not initialized. Use 'with GPUWorkerPool(...) as pool:'")

        if os.environ.get("HARP_INNER_EVAL", "").lower() == "vllm":
            print(f"[parallel_run] taking vLLM-batched path", flush=True)
            try:
                return self._parallel_run_independent_singletons_per_task_vllm(tasks)
            except Exception as e:
                print(f"[GPUWorkerPool] vLLM-batched path failed ({e}); falling back to HF interleaved", flush=True)
                traceback.print_exc()
                # Workers had peft_model torn down by _release_worker_gpu before
                # the vLLM step. Re-init it so the HF fallback can call
                # _worker_run_independent_singleton_per_task without KeyError.
                try:
                    from .vllm_batched_inner_eval import _reinit_worker_peft_model
                    print(f"[parallel_run] re-initializing {self.num_gpus} worker(s) before HF fallback", flush=True)
                    list(self.pool.map(_reinit_worker_peft_model, [None] * self.num_gpus))
                except Exception as e2:
                    print(f"[parallel_run] worker re-init before HF fallback failed: {e2}", flush=True)
                    traceback.print_exc()

        print(f"[parallel_run] taking HF interleaved path", flush=True)
        out: Dict[int, Dict[int, float]] = {}
        for gid, per_task in self.pool.imap_unordered(
            _worker_run_independent_singleton_per_task, tasks, chunksize=1
        ):
            if not per_task:
                logging.warning("Task gid=%d returned empty per-task dict (worker error); excluded.", gid)
                continue
            out[int(gid)] = dict(per_task)
        return out

    @timeit
    def _parallel_run_independent_singletons_per_task_vllm(
        self,
        tasks: List[Tuple[int, np.ndarray, str, int, str]]
    ) -> Dict[int, Dict[int, float]]:
        """vLLM-batched path: train all leaves, then batch-eval via vLLM subprocess."""
        print(f"[vllm_path] entering, importing vllm_batched_inner_eval", flush=True)
        from .vllm_batched_inner_eval import (
            train_only_singleton_per_task,
            eval_all_adapters_via_vllm,
            _release_worker_gpu,
        )
        print(f"[vllm_path] import done; creating adapter dir", flush=True)

        # Adapter staging dir under data tmpdir (auto-cleanup with pool teardown)
        adapters_dir = tempfile.mkdtemp(prefix="harp_adapters_", dir=self._data_tmpdir)
        tasks_with_dir = [t + (adapters_dir,) for t in tasks]
        n_tasks = len(tasks_with_dir)
        print(f"[GPUWorkerPool/vllm] Train phase: {n_tasks} leaves → {adapters_dir}", flush=True)

        # 1. TRAIN PHASE — workers train + save adapter, return (gid, path)
        adapter_paths: Dict[int, str] = {}
        for gid, path in self.pool.imap_unordered(
            train_only_singleton_per_task, tasks_with_dir, chunksize=1
        ):
            if path:
                adapter_paths[int(gid)] = path

        if not adapter_paths:
            print("[GPUWorkerPool/vllm] No adapters trained; returning empty")
            shutil.rmtree(adapters_dir, ignore_errors=True)
            return {}

        # 2. RELEASE worker GPUs so vLLM can claim memory
        print(f"[GPUWorkerPool/vllm] Releasing worker GPU memory ({self.num_gpus} workers)...")
        try:
            list(self.pool.map(_release_worker_gpu, [None] * self.num_gpus))
        except Exception as e:
            print(f"[GPUWorkerPool/vllm] Worker release warning: {e}")

        # 3. EVAL PHASE — reconstruct eval_dl in main process, run vLLM batch eval
        eval_bs = int(self.lora_config.get("eval_batch_size",
                                           os.environ.get("HARP_EVAL_BS", 64)))
        eval_dl = _full_validation_loader(self.eval_arrays, eval_bs)

        eval_task_types = list(self.eval_arrays.get("_task_types", []) or [])
        eval_gold_strings = list(self.eval_arrays.get("_gold_strings", []) or [])

        # Persistent log path so we can debug startup failures after the
        # adapters_dir is cleaned. Honors HARP_VLLM_LOG; defaults to
        # slurm_logs/vllm_server_${SLURM_JOB_ID:-PID}.log relative to cwd.
        default_log = os.path.join(
            "slurm_logs",
            f"vllm_server_{os.environ.get('SLURM_JOB_ID', os.getpid())}.log",
        )
        log_path = os.environ.get("HARP_VLLM_LOG", default_log)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        eval_gpu = int(os.environ.get("HARP_VLLM_GPU", "0"))
        max_loras = int(os.environ.get("HARP_VLLM_MAX_LORAS", "64"))

        per_gid_per_task = eval_all_adapters_via_vllm(
            adapter_paths,
            base_model=self.base_model_name,
            eval_dl=eval_dl,
            tokenizer=self._tokenizer_gen,
            eval_task_types=eval_task_types,
            eval_gold_strings=eval_gold_strings,
            gpu_id=eval_gpu,
            log_path=log_path,
            max_loras=max_loras,
        )

        # 4. Adapter cleanup
        try:
            shutil.rmtree(adapters_dir, ignore_errors=True)
        except Exception:
            pass

        # 5. RE-INIT worker peft_models for the next stage (workers freed GPU
        # in step 2; a subsequent parallel_run call would otherwise crash).
        from .vllm_batched_inner_eval import _reinit_worker_peft_model
        print(f"[GPUWorkerPool/vllm] Re-initializing worker peft_models...")
        try:
            list(self.pool.map(_reinit_worker_peft_model, [None] * self.num_gpus))
        except Exception as e:
            print(f"[GPUWorkerPool/vllm] Worker re-init warning: {e}")

        print(f"[GPUWorkerPool/vllm] Eval phase complete: {len(per_gid_per_task)}/{len(adapter_paths)} adapters scored")

        return per_gid_per_task





def estimate_group_independent_phi_per_task_parallel(
    player_ids: np.ndarray,
    player_sets: Dict[int, np.ndarray],
    gpu_pool: GPUWorkerPool,
    eval_metric: str,
    *,
    v0_per_task: Dict[str, float],       # {task_name: null_score}
    subj_id_to_task: Dict[int, str],     # {int_subj_id: task_name}
    seed_int: int = 0,
) -> Dict[int, Dict[str, float]]:
    """
    Per-task independent φ for many groups, parallelized across GPUs.
    
    Returns:
      phi_per_task_map: {gid: {task_name: phi_for_that_task}}
      where phi_for_task = acc_task_with_data - v0_task (null score for that task)
    """
    player_ids = np.asarray(player_ids, dtype=int)
    if player_ids.size == 0:
        return {}

    print(f"[estimate_phi] building {player_ids.size} tasks", flush=True)
    tasks = []
    for pid in player_ids.tolist():
        idxs = np.asarray(player_sets[int(pid)], dtype=int)
        # Last arg is unused by worker but kept for signature compatibility
        tasks.append((int(pid), idxs, str(eval_metric), int(seed_int), ""))

    # Get per-task accuracy for each group
    print(f"[estimate_phi] dispatching {len(tasks)} tasks to gpu_pool", flush=True)
    per_task_acc_map = gpu_pool.parallel_run_independent_singletons_per_task(tasks)

    # Convert raw accuracy to phi (diff from null) per task
    phi_per_task_map: Dict[int, Dict[str, float]] = {}
    for gid, per_task_acc in per_task_acc_map.items():
        gid = int(gid)
        phi_tasks: Dict[str, float] = {}
        
        for subj_id_int, acc in per_task_acc.items():
            subj_id_int = int(subj_id_int)
            task_name = subj_id_to_task.get(subj_id_int, None)
            if task_name is None:
                continue
            null_score = v0_per_task.get(task_name, 0.0)
            phi_tasks[task_name] = float(acc) - float(null_score)
        
        phi_per_task_map[gid] = phi_tasks
    
    return phi_per_task_map




def bootstrap_parent_representative_independent_per_task(
    gpu_pool: "GPUWorkerPool",
    *,
    leaf_groups: List[np.ndarray],
    leaf_feat: np.ndarray,
    parent_of_leaf: np.ndarray,
    group_of_i: np.ndarray,
    eval_metric: str,
    v0_per_task: Dict[str, float],           # {task_name: null_score}
    subj_id_to_task: Dict[int, str],         # {int_subj_id: task_name}
    rep_per_parent: int = 2,
    seed_int: int = 0,
    rng: np.random.Generator = np.random.default_rng(0),
    se_floor: float = 1e-3,
    use_eb_shrinkage: bool = True,
    prior_var: float = 0.01,
) -> Tuple[
    np.ndarray, np.ndarray,   # mu_leaf (macro), se_leaf (macro)
    Dict[int, Dict[str, float]],  # mu_leaf_per_task: {gid: {task: phi}}
    Dict[int, Dict[str, float]],  # se_leaf_per_task: {gid: {task: se}}
    np.ndarray,               # leaf_is_measured
    Dict[str, Any],           # info
]:
    """
    Per-task bootstrap: same as bootstrap_parent_representative_independent but
    computes phi decomposed per task.
    
    Key output: mu_leaf_per_task[gid][task_name] = phi for that task
    The macro phi = mean(mu_leaf_per_task[gid].values()) for each gid
    
    Also returns mu_leaf (macro scalar per leaf) and se_leaf for backward compatibility
    with the interaction stage.
    """
    leaf_feat = np.asarray(leaf_feat, dtype=np.float32)
    parent_of_leaf = np.asarray(parent_of_leaf, dtype=int)
    group_of_i = np.asarray(group_of_i, dtype=int)

    n = int(group_of_i.size)
    G = int(len(leaf_groups))
    task_names = sorted(v0_per_task.keys())
    n_tasks = len(task_names)
    
    if G == 0 or n == 0:
        return (
            np.zeros(G, dtype=float),
            np.full(G, np.inf, dtype=float),
            {}, {},
            np.zeros(G, dtype=bool),
            {"error": "empty groups"},
        )

    P = int(parent_of_leaf.max()) + 1 if parent_of_leaf.size else 0

    # 1) Representative sampling (same as original)
    reps_per_parent, rep_leaf_gids = sample_representative_leaf_groups(
        leaf_groups,
        leaf_feat,
        parent_of_leaf,
        rep_per_parent=int(rep_per_parent),
        rng=rng,
        use_diverse=True,
    )

    # 2) Per-task independent φ for rep leaf groups
    player_sets = {int(gid): np.asarray(leaf_groups[int(gid)], dtype=int)
                   for gid in rep_leaf_gids.tolist()}
    
    phi_per_task_map = estimate_group_independent_phi_per_task_parallel(
        player_ids=rep_leaf_gids,
        player_sets=player_sets,
        gpu_pool=gpu_pool,
        eval_metric=eval_metric,
        v0_per_task=v0_per_task,
        subj_id_to_task=subj_id_to_task,
        seed_int=int(seed_int),
    )

    # 3) Also compute macro phi for each rep (for backward compatibility / SE estimation)
    phi_rep_map_macro: Dict[int, float] = {}
    for gid, task_phis in phi_per_task_map.items():
        if task_phis:
            phi_rep_map_macro[int(gid)] = float(np.mean(list(task_phis.values())))
        else:
            phi_rep_map_macro[int(gid)] = 0.0

    # 4) Aggregate to parent level (macro) for SE estimation, same as original
    mu_parent = np.zeros(P, dtype=float)
    se_parent = np.full(P, np.inf, dtype=float)
    r_parent = np.zeros(P, dtype=int)
    rep_vals_parent: List[np.ndarray] = []

    for p in range(P):
        reps = reps_per_parent[p] if p < len(reps_per_parent) else np.zeros((0,), dtype=int)
        vals = []
        for gid in reps.tolist():
            if int(gid) in phi_rep_map_macro:
                vals.append(float(phi_rep_map_macro[int(gid)]))
        vals = np.asarray(vals, dtype=float)
        rep_vals_parent.append(vals)
        r = int(vals.size)
        r_parent[p] = r

        if r == 0:
            mu_parent[p] = 0.0
            se_parent[p] = np.inf
        elif r == 1:
            mu_parent[p] = float(vals[0])
            se_parent[p] = np.inf
        else:
            mu_parent[p] = float(vals.mean())
            cvar = float(vals.var(ddof=1))
            se_parent[p] = float(max(se_floor, np.sqrt(cvar / r)))

    # EB shrinkage on parent means — BOTH macro and per-task
    # For macro: same as original
    # For per-task: same EB model applied independently per task t
    #   θₚ(t) ~ N(μ_global(t), τ²(t))
    #   μ̂ₚ(t) | θₚ(t) ~ N(θₚ(t), σ²ₚ(t)/rₚ)
    
    # First collect per-task rep values per parent (needed for per-task variance)
    rep_vals_parent_per_task: Dict[int, Dict[str, np.ndarray]] = {}
    for p in range(P):
        reps = reps_per_parent[p] if p < len(reps_per_parent) else np.zeros((0,), dtype=int)
        task_vals: Dict[str, list] = {t: [] for t in task_names}
        for gid in reps.tolist():
            gid = int(gid)
            if gid in phi_per_task_map:
                for t, v in phi_per_task_map[gid].items():
                    if t in task_vals:
                        task_vals[t].append(v)
        rep_vals_parent_per_task[p] = {t: np.asarray(vs, dtype=float) for t, vs in task_vals.items()}

    # Raw per-task parent means (before shrinkage)
    mu_parent_per_task_raw: Dict[int, Dict[str, float]] = {}
    for p in range(P):
        parent_tasks = {}
        for t in task_names:
            vals = rep_vals_parent_per_task[p][t]
            if vals.size > 0:
                parent_tasks[t] = float(vals.mean())
            else:
                parent_tasks[t] = 0.0
        mu_parent_per_task_raw[p] = parent_tasks

    if use_eb_shrinkage and P > 0:
        good = (r_parent > 0)
        
        # --- Macro shrinkage (same as original) ---
        global_mean = float(mu_parent[good].mean()) if np.any(good) else 0.0

        num = 0.0
        den = 0.0
        for p in range(P):
            r = int(r_parent[p])
            if r >= 2:
                vals = rep_vals_parent[p]
                sp2 = float(vals.var(ddof=1))
                if np.isfinite(sp2) and sp2 >= 0.0:
                    num += (r - 1) * sp2
                    den += (r - 1)

        sigma2_within = float(num / den) if den > 0 else 0.0
        sigma2_within = float(max(sigma2_within, se_floor**2))

        for p in range(P):
            r = int(r_parent[p])
            if r <= 0:
                mu_parent[p] = float(global_mean)
                se_parent[p] = float(max(se_floor, np.sqrt(prior_var)))
                continue

            cmean = float(mu_parent[p])
            if r >= 2:
                vals = rep_vals_parent[p]
                sp2 = float(vals.var(ddof=1))
                if (not np.isfinite(sp2)) or sp2 < 0.0:
                    sp2 = sigma2_within
            else:
                sp2 = sigma2_within

            obs_var = float(max(sp2 / max(r, 1), se_floor**2))
            post_mean, post_var = eb_shrinkage_from_obsvar(
                cmean=cmean, obs_var=obs_var,
                global_mean=global_mean, prior_var=float(prior_var),
            )
            mu_parent[p] = float(post_mean)
            se_parent[p] = float(max(se_floor, np.sqrt(max(post_var, 0.0))))

        # --- Per-task shrinkage: same EB model applied per task independently ---
        # For each task t:
        #   global_mean(t) = mean of μ̂ₚ(t) across parents with rₚ > 0
        #   σ²_within(t) = pooled within-parent variance of φ(g,t) values
        #   Then shrink each μ̂ₚ(t) toward global_mean(t)
        
        mu_parent_per_task: Dict[int, Dict[str, float]] = {}

        for t in task_names:
            # global mean for this task
            task_parent_means = []
            for p in range(P):
                if r_parent[p] > 0:
                    task_parent_means.append(mu_parent_per_task_raw[p][t])
            global_mean_t = float(np.mean(task_parent_means)) if task_parent_means else 0.0

            # pooled within-parent variance for this task
            num_t = 0.0
            den_t = 0.0
            for p in range(P):
                r = int(r_parent[p])
                if r >= 2:
                    vals_t = rep_vals_parent_per_task[p][t]
                    if vals_t.size >= 2:
                        sp2_t = float(vals_t.var(ddof=1))
                        if np.isfinite(sp2_t) and sp2_t >= 0.0:
                            num_t += (r - 1) * sp2_t
                            den_t += (r - 1)

            sigma2_within_t = float(num_t / den_t) if den_t > 0 else 0.0
            sigma2_within_t = float(max(sigma2_within_t, se_floor**2))

            # shrink each parent's per-task mean
            for p in range(P):
                if p not in mu_parent_per_task:
                    mu_parent_per_task[p] = {}

                r = int(r_parent[p])
                if r <= 0:
                    mu_parent_per_task[p][t] = float(global_mean_t)
                    continue

                cmean_t = mu_parent_per_task_raw[p][t]

                if r >= 2:
                    vals_t = rep_vals_parent_per_task[p][t]
                    if vals_t.size >= 2:
                        sp2_t = float(vals_t.var(ddof=1))
                        if (not np.isfinite(sp2_t)) or sp2_t < 0.0:
                            sp2_t = sigma2_within_t
                    else:
                        sp2_t = sigma2_within_t
                else:
                    sp2_t = sigma2_within_t

                obs_var_t = float(max(sp2_t / max(r, 1), se_floor**2))
                post_mean_t, _ = eb_shrinkage_from_obsvar(
                    cmean=cmean_t, obs_var=obs_var_t,
                    global_mean=global_mean_t, prior_var=float(prior_var),
                )
                mu_parent_per_task[p][t] = float(post_mean_t)
    else:
        # No shrinkage: use raw per-task parent means
        mu_parent_per_task = dict(mu_parent_per_task_raw)

    # 5) Broadcast to leaf level
    mu_leaf = np.zeros(G, dtype=float)
    se_leaf = np.full(G, np.inf, dtype=float)
    leaf_is_measured = np.zeros(G, dtype=bool)

    mu_leaf_per_task: Dict[int, Dict[str, float]] = {}
    se_leaf_per_task: Dict[int, Dict[str, float]] = {}
    
    for gid in range(G):
        pid = int(parent_of_leaf[gid])
        mu_leaf[gid] = float(mu_parent[pid])
        se_leaf[gid] = float(se_parent[pid])
        mu_leaf_per_task[gid] = dict(mu_parent_per_task.get(pid, {}))
        # SE per task: use parent SE for all tasks (conservative)
        se_leaf_per_task[gid] = {t: float(se_parent[pid]) for t in task_names}

    # (b) override measured rep leaves
    for gid, phi_tasks in phi_per_task_map.items():
        gid = int(gid)
        if 0 <= gid < G:
            leaf_is_measured[gid] = True
            mu_leaf_per_task[gid] = dict(phi_tasks)
            
            # macro from per-task
            if phi_tasks:
                mu_leaf[gid] = float(np.mean(list(phi_tasks.values())))
            
            pid = int(parent_of_leaf[gid])
            se_leaf[gid] = float(max(se_floor, se_parent[pid]))
            se_leaf_per_task[gid] = {t: float(max(se_floor, se_parent[pid])) for t in task_names}

    info: Dict[str, Any] = {
        "bootstrap": {
            "mode": "parent_representative_independent_per_task",
            "G_leaf": int(G),
            "P_parent": int(P),
            "n_tasks": int(n_tasks),
            "task_names": task_names,
            "rep_per_parent": int(rep_per_parent),
            "rep_leaf_total": int(rep_leaf_gids.size),
            "seed_int": int(seed_int),
            "v0_per_task": {str(k): float(v) for k, v in v0_per_task.items()},
            "use_eb_shrinkage": bool(use_eb_shrinkage),
            "prior_var": float(prior_var),
            "se_floor": float(se_floor),
        },
        "rep_leaf_gids": rep_leaf_gids.tolist(),
        "reps_per_parent": [arr.tolist() for arr in reps_per_parent],
        "parent_mu": mu_parent.tolist(),
        "parent_se": se_parent.tolist(),
        "parent_r": r_parent.tolist(),
        "phi_rep_map_macro": {str(k): float(v) for k, v in phi_rep_map_macro.items()},
        "phi_per_task_map_summary": {
            str(gid): {t: round(v, 6) for t, v in tasks.items()}
            for gid, tasks in list(phi_per_task_map.items())[:20]
        },
    }
    info["leaf_is_measured"] = leaf_is_measured.tolist()

    return (
        mu_leaf, se_leaf,
        mu_leaf_per_task, se_leaf_per_task,
        leaf_is_measured,
        info,
    )





########### find optimal k
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _knee_point_from_curve(xs: np.ndarray, ys: np.ndarray) -> int:
    """
    Kneedle-style knee for a decreasing curve y(k) (e.g., inertia).
    Returns index into xs (not the k value).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    # normalize to [0,1]
    x = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-12)
    y = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-12)
    # want maximum distance to line between endpoints
    # line: y = y0 + (y1-y0)*x
    y_line = y[0] + (y[-1] - y[0]) * x
    dist = y_line - y  # for decreasing convex curves
    return int(np.argmax(dist))





def _auto_n_clusters(n_points: int, max_group_size: int) -> int:
    """
    Same logic everywhere: number of anchor buckets so that after chunking,
    groups are ~ max_group_size.
    """
    n_points = int(n_points)
    max_group_size = int(max(1, max_group_size))
    if n_points <= 1:
        return 1
    c = int(np.ceil(n_points / float(max_group_size)))
    return int(max(2, min(c, n_points)))


# only for leaf ones to make fewer small groups
def _auto_n_clusters_bounded(n_points: int, min_group_size: int, max_group_size: int) -> int:
    """
    Choose anchor count A so buckets can plausibly end up within [min, max].

    Feasible range:
      ceil(n/max) <= A <= floor(n/min)
    If infeasible (lo > hi), return lo (you'll need pooling anyway).
    """
    n = int(n_points)
    mn = int(max(1, min_group_size))
    mx = int(max(1, max_group_size))
    if n <= 1:
        return 1

    lo = int(np.ceil(n / float(mx)))
    hi = int(max(1, n // mn))

    if lo > hi:
        return max(2, min(lo, n))
    return max(2, min(hi, lo))


def build_parent_leaf_hierarchy(
    X_feat: np.ndarray,
    *,
    rng: np.random.Generator,
    large_max_group_size: int = 65536,     # max points per LARGE group
    parent_max_group_size: int = 8192,     # max points per PARENT group
    leaf_max_group_size: int = 1024,       # max points per LEAF group (your target)
    leaf_min_group_size: int = 256,        # NEW: min points per LEAF group (your target)
    large_n_clusters: Optional[int] = None,
    ann_use_gpu: bool = True,
    lora_config: Optional[Dict[str, Any]] = None,
) -> Tuple[
    List[np.ndarray],  # large_groups: each is array of parent_ids
    List[np.ndarray],  # parent_groups: each is array of leaf_ids
    List[np.ndarray],  # leaf_groups: each is array of point_ids
    np.ndarray,        # large_of_parent: (P,)
    np.ndarray,        # parent_of_leaf: (G,)
    np.ndarray,        # large_of_leaf:  (G,)
    np.ndarray,        # leaf_of_i:      (n,)
    np.ndarray,        # leaf_feat:      (G,d)
]:
    """
    Large -> Parent -> Leaf hierarchy.

    Key change vs your current version:
      - Enforce leaf sizes approximately within [leaf_min_group_size, leaf_max_group_size]
      - Avoid tiny leaves by pooling "too-small parents" within each LARGE and regrouping them
      - Also uses build_groups_from_shapley_ann(..., min_group_size=leaf_min_group_size, merge_small_buckets=True)
        at leaf-level for bounded splitting inside a parent.

    Notes / Guarantees:
      - If a parent has < leaf_min_group_size points, we DO NOT create tiny leaves from it.
        We pool those points and later form valid leaves from the pool (within the same LARGE).
      - If an entire LARGE has < leaf_min_group_size points, you will still end up with a tiny leaf
        because it's mathematically impossible to reach the min bound without mixing across LARGE.
    """
    lora_config = dict(lora_config or {})
    X_feat = np.asarray(X_feat, dtype=np.float32)
    if X_feat.ndim != 2:
        raise ValueError(f"X_feat must be 2D, got shape {X_feat.shape}")
    n, d = X_feat.shape
    if n == 0:
        return (
            [], [], [],
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0, d), dtype=np.float32),
        )

    # sanity
    large_max_group_size = int(max(1, large_max_group_size))
    parent_max_group_size = int(max(1, parent_max_group_size))
    leaf_max_group_size = int(max(1, leaf_max_group_size))
    leaf_min_group_size = int(max(1, leaf_min_group_size))
    if leaf_min_group_size > leaf_max_group_size:
        raise ValueError(f"leaf_min_group_size ({leaf_min_group_size}) > leaf_max_group_size ({leaf_max_group_size})")

    # --------------------------
    # 1) Build LARGE groups (points)
    # --------------------------
    if large_n_clusters is None or int(large_n_clusters) <= 0:
        large_n_clusters = _auto_n_clusters(n, int(large_max_group_size))

    large_groups_local, _, _ = build_groups_from_shapley_ann(
        X_feat,
        n_group_clusters=int(large_n_clusters),
        max_group_size=int(large_max_group_size),
        rng=rng,
        ann_use_gpu=bool(lora_config.get("ann_use_gpu", ann_use_gpu)),
        # IMPORTANT: do not enforce min at large-level; large-level can be tiny naturally
        min_group_size=1,
        merge_small_buckets=False,
    )
    large_point_groups: List[np.ndarray] = [np.asarray(g, dtype=int) for g in large_groups_local]
    L = len(large_point_groups)

    # Outputs
    leaf_groups: List[np.ndarray] = []
    parent_groups: List[np.ndarray] = []
    large_groups: List[np.ndarray] = []

    parent_of_leaf: List[int] = []
    large_of_parent: List[int] = []
    leaf_of_i = -np.ones(n, dtype=int)

    # --------------------------
    # 2) For each LARGE, build PARENT groups (within it), then LEAF groups (within parents)
    # --------------------------
    for lid in range(L):
        large_mem = np.asarray(large_point_groups[lid], dtype=int)
        if large_mem.size == 0:
            large_groups.append(np.zeros((0,), dtype=int))
            continue

        # PARENT grouping inside this LARGE (local indices)
        X_large = X_feat[large_mem]
        c_parent = _auto_n_clusters(int(large_mem.size), int(parent_max_group_size))

        parent_groups_local, _, _ = build_groups_from_shapley_ann(
            X_large,
            n_group_clusters=int(c_parent),
            max_group_size=int(parent_max_group_size),
            rng=rng,
            ann_use_gpu=bool(lora_config.get("ann_use_gpu", ann_use_gpu)),
            # Do not enforce min at parent level either; parent can be small, we'll pool those
            min_group_size=1,
            merge_small_buckets=False,
        )

        parents_in_large: List[int] = []
        pool_small_points: List[int] = []  # GLOBAL point ids pooled for this LARGE

        # ---- LEAF grouping per parent ----
        for p_local in parent_groups_local:
            p_local = np.asarray(p_local, dtype=int)
            if p_local.size == 0:
                continue

            # parent points are GLOBAL ids
            parent_points = large_mem[p_local]

            # If this parent is too small to make >= leaf_min leaves without mixing,
            # pool it to avoid tiny leaves.
            if int(parent_points.size) < int(leaf_min_group_size):
                pool_small_points.extend(parent_points.tolist())
                continue

            # leaf grouping inside this parent (local indices into parent_points)
            X_parent = X_feat[parent_points]
            # c_leaf = _auto_n_clusters(int(parent_points.size), int(leaf_max_group_size))
            c_leaf = _auto_n_clusters_bounded(
                int(parent_points.size),
                int(leaf_min_group_size),
                int(leaf_max_group_size))
       
            leaf_local, _, _ = build_groups_from_shapley_ann(
                X_parent,
                n_group_clusters=int(c_leaf),
                max_group_size=int(leaf_max_group_size),
                min_group_size=int(leaf_min_group_size),  # ENFORCE min at leaf level
                rng=rng,
                ann_use_gpu=bool(lora_config.get("ann_use_gpu", ann_use_gpu)),
                merge_small_buckets=True,                 # merge tiny anchor buckets before splitting
                refine_steps=1
            )

            # create global parent id
            pid = len(parent_groups)
            parents_in_large.append(pid)
            large_of_parent.append(int(lid))

            leaf_ids_for_parent: List[int] = []
            for g_local in leaf_local:
                g_local = np.asarray(g_local, dtype=int)
                if g_local.size == 0:
                    continue
                leaf_points = parent_points[g_local]  # GLOBAL

                gid = len(leaf_groups)
                leaf_groups.append(np.asarray(leaf_points, dtype=int))
                leaf_ids_for_parent.append(gid)

                parent_of_leaf.append(pid)
                leaf_of_i[leaf_points] = gid

            parent_groups.append(np.asarray(leaf_ids_for_parent, dtype=int))

        # ---- Pool stage: regroup pooled small parents into valid leaves (within same LARGE) ----
        if len(pool_small_points) > 0:
            pool = np.asarray(pool_small_points, dtype=int)
            if pool.size > 0:
                # If pool is still smaller than leaf_min, we cannot satisfy the min bound
                # without mixing across LARGE. Keep it as one leaf under a pooled parent.
                if int(pool.size) < int(leaf_min_group_size):
                    pooled_pid = len(parent_groups)
                    parents_in_large.append(pooled_pid)
                    large_of_parent.append(int(lid))

                    gid = len(leaf_groups)
                    leaf_groups.append(pool.copy())
                    parent_of_leaf.append(pooled_pid)
                    leaf_of_i[pool] = gid
                    parent_groups.append(np.asarray([gid], dtype=int))
                else:
                    X_pool = X_feat[pool]
                    # c_pool = _auto_n_clusters(int(pool.size), int(leaf_max_group_size))
                    c_pool = _auto_n_clusters_bounded(int(pool.size), int(leaf_min_group_size), int(leaf_max_group_size))


                    pooled_leaf_local, _, _ = build_groups_from_shapley_ann(
                        X_pool,
                        n_group_clusters=int(c_pool),
                        max_group_size=int(leaf_max_group_size),
                        min_group_size=int(leaf_min_group_size),
                        rng=rng,
                        ann_use_gpu=bool(lora_config.get("ann_use_gpu", ann_use_gpu)),
                        merge_small_buckets=True,
                        refine_steps=1
                    )

                    pooled_pid = len(parent_groups)
                    parents_in_large.append(pooled_pid)
                    large_of_parent.append(int(lid))

                    pooled_leaf_ids: List[int] = []
                    for g_local in pooled_leaf_local:
                        g_local = np.asarray(g_local, dtype=int)
                        if g_local.size == 0:
                            continue
                        leaf_points = pool[g_local]

                        gid = len(leaf_groups)
                        leaf_groups.append(np.asarray(leaf_points, dtype=int))
                        pooled_leaf_ids.append(gid)

                        parent_of_leaf.append(pooled_pid)
                        leaf_of_i[leaf_points] = gid

                    parent_groups.append(np.asarray(pooled_leaf_ids, dtype=int))

        # this large holds these parent ids
        large_groups.append(np.asarray(parents_in_large, dtype=int))

    # --------------------------
    # finalize arrays
    # --------------------------
    P = len(parent_groups)
    G = len(leaf_groups)

    large_of_parent_arr = np.asarray(large_of_parent, dtype=int) if P > 0 else np.zeros((0,), dtype=int)
    parent_of_leaf_arr = np.asarray(parent_of_leaf, dtype=int) if G > 0 else np.zeros((0,), dtype=int)

    if G > 0 and P > 0:
        large_of_leaf_arr = large_of_parent_arr[parent_of_leaf_arr]
    else:
        large_of_leaf_arr = np.zeros((0,), dtype=int)

    # leaf_feat (normalized mean embedding per leaf)
    leaf_feat = np.zeros((G, d), dtype=np.float32)
    for gid, g in enumerate(leaf_groups):
        if g.size == 0:
            continue
        mu = X_feat[g].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        leaf_feat[gid] = mu.astype(np.float32)

    return (
        large_groups,            # List[np.ndarray] of parent_ids
        parent_groups,           # List[np.ndarray] of leaf_ids
        leaf_groups,             # List[np.ndarray] of point_ids
        large_of_parent_arr,     # (P,)
        parent_of_leaf_arr,      # (G,)
        large_of_leaf_arr,       # (G,)
        leaf_of_i,               # (n,)
        leaf_feat,               # (G,d)
    )


############## get macro score and per-task utilities

def load_task_grouping_from_representatives(
    jsonl_path: str,
    *,
    subject_key: str = "subject",
    score_key: str = "acc01",
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, int], bool]:
    """
    Read full-validation baseline scores to determine domains and null scores.
    
    Returns:
        subject_to_task: mapping from subject_name -> task_group_name
        v0_per_task: Dict[task_group_name, mean_null_score] for each task
        task_counts: Dict[task_group_name, n_validation_rows] for micro weighting
        use_subject: True if grouping by subject, False if by embedding bucket
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    
    if not rows:
        return {}, {}, {}, True
    
    # Determine if we have real subjects
    real_subjects = set()
    for row in rows:
        subj = _get_subject(row, subject_key).strip()
        if subj and subj != "1" and subj != "":
            real_subjects.add(subj)
    
    use_subject = len(real_subjects) > 1  # more than 1 distinct real subject
    
    # Build subject_to_task mapping and per-task null scores
    task_sums = defaultdict(float)
    task_cnts = defaultdict(int)
    subject_to_task = {}
    
    for row in rows:
        subj = _get_subject(row, subject_key).strip()
        meta = row.get("_meta", {})
        
        if use_subject:
            task_name = subj if subj else "unknown"
        else:
            bucket_name = meta.get("bucket_name", "unknown") if isinstance(meta, dict) else "unknown"
            task_name = bucket_name
        
        subject_to_task[subj] = task_name
        
        score = _get_score(row, score_key)
        if score is not None and math.isfinite(score):
            task_sums[task_name] += score
            task_cnts[task_name] += 1
    
    v0_per_task = {}
    task_counts = {}
    for t in task_cnts:
        if task_cnts[t] > 0:
            v0_per_task[t] = task_sums[t] / task_cnts[t]
            task_counts[t] = task_cnts[t]
    
    logging.info(f"[task_grouping] use_subject={use_subject}, n_tasks={len(v0_per_task)}, "
                 f"tasks={list(v0_per_task.keys())[:10]}...")
    
    return subject_to_task, v0_per_task, task_counts, use_subject


def _get_subject(row: Dict[str, Any], subject_key: str) -> str:
    # Prefer _meta.subject if present
    meta = row.get("_meta")
    if isinstance(meta, dict) and "subject" in meta and meta["subject"] is not None:
        return str(meta["subject"])
    val = row.get(subject_key, "")
    return "" if val is None else str(val)


def _to_float_maybe(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s:
        return None

    # "66.67%" -> 0.6667
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return None

    # extract first float-looking token (handles "acc=0.5", etc.)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _get_score(row: Dict[str, Any], score_key: str) -> Optional[float]:
    meta = row.get("_meta")
    if isinstance(meta, dict):
        # 1) requested key in _meta
        if score_key in meta:
            v = _to_float_maybe(meta.get(score_key))
            if v is not None:
                return v

        # 2) fallbacks in _meta
        for k in ("acc01", "score01", "score", "accuracy", "acc"):
            if k in meta:
                v = _to_float_maybe(meta.get(k))
                if v is not None:
                    return v

    # 3) requested key top-level
    if score_key in row:
        v = _to_float_maybe(row.get(score_key))
        if v is not None:
            return v

    # 4) fallbacks top-level
    for k in ("acc01", "score01", "score", "accuracy", "acc"):
        if k in row:
            v = _to_float_maybe(row.get(k))
            if v is not None:
                return v

    return None


def compute_macro_micro(
    jsonl_path: str,
    *,
    subject_key: str = "subject",
    score_key: str = "acc01",
    drop_empty_subject: bool = True,
) -> Tuple[float, float, Dict[str, float], Dict[str, int], Dict[str, int]]:
    sums = defaultdict(float)
    cnts = defaultdict(int)

    total_sum = 0.0
    total_cnt = 0

    stats = {"rows": 0, "valid_scores": 0, "missing_scores": 0, "empty_subject": 0}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            row = json.loads(line)

            subj = _get_subject(row, subject_key).strip()
            if subj == "":
                stats["empty_subject"] += 1
                if drop_empty_subject:
                    # still counts for micro if score exists? We choose to drop entirely.
                    # If you want to keep it, set drop_empty_subject=False.
                    pass

            score = _get_score(row, score_key)
            if score is None or not math.isfinite(score):
                stats["missing_scores"] += 1
                continue

            stats["valid_scores"] += 1

            # micro
            total_sum += score
            total_cnt += 1

            # macro buckets
            if subj == "" and drop_empty_subject:
                continue
            sums[subj] += score
            cnts[subj] += 1

    per_mean = {s: (sums[s] / cnts[s]) for s in cnts.keys() if cnts[s] > 0}
    macro = float("nan") if not per_mean else sum(per_mean.values()) / len(per_mean)
    micro = float("nan") if total_cnt == 0 else total_sum / total_cnt

    return macro, micro, per_mean, cnts, stats


def write_hierarchy_cluster_sizes(
    out_path: str,
    *,
    large_groups: List[np.ndarray],     # large -> parent_ids
    parent_groups: List[np.ndarray],    # parent -> leaf_ids
    leaf_groups: List[np.ndarray],      # leaf -> point_ids
    large_of_parent: np.ndarray,        # (P,)
    parent_of_leaf: np.ndarray,         # (G,)
) -> None:
    """
    Write cluster sizes for Large/Parent/Leaf into a CSV file.

    CSV columns:
      level, large_id, parent_id, leaf_id, size
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # infer sizes
    # - leaf size = number of points
    leaf_sizes = np.array([len(g) for g in leaf_groups], dtype=int)

    # - parent size = sum leaf sizes
    parent_sizes = np.zeros((len(parent_groups),), dtype=int)
    for pid, leaf_ids in enumerate(parent_groups):
        if len(leaf_ids) > 0:
            parent_sizes[pid] = int(leaf_sizes[np.asarray(leaf_ids, dtype=int)].sum())

    # - large size = sum parent sizes
    large_sizes = np.zeros((len(large_groups),), dtype=int)
    for lid, parent_ids in enumerate(large_groups):
        if len(parent_ids) > 0:
            large_sizes[lid] = int(parent_sizes[np.asarray(parent_ids, dtype=int)].sum())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "large_id", "parent_id", "leaf_id", "size"])

        # LARGE rows
        for lid, sz in enumerate(large_sizes.tolist()):
            w.writerow(["large", lid, "", "", sz])

        # PARENT rows
        for pid, sz in enumerate(parent_sizes.tolist()):
            lid = int(large_of_parent[pid]) if len(large_of_parent) > 0 else ""
            w.writerow(["parent", lid, pid, "", sz])

        # LEAF rows
        for gid, sz in enumerate(leaf_sizes.tolist()):
            pid = int(parent_of_leaf[gid]) if len(parent_of_leaf) > 0 else ""
            lid = int(large_of_parent[pid]) if pid != "" and len(large_of_parent) > 0 else ""
            w.writerow(["leaf", lid, pid, gid, sz])



# ============================================================
# Stage 1.5: Task filtering, weights, and phi matrix
# ============================================================

def build_task_phi_matrix(
    mu_leaf_per_task: Dict[int, Dict[str, float]],
    G: int,
) -> Tuple[np.ndarray, List[str]]:
    """Convert mu_leaf_per_task dict to (G, n_tasks) numpy array.
    Returns phi_matrix and sorted task_names list."""
    all_tasks = set()
    for gid in range(G):
        if gid in mu_leaf_per_task:
            all_tasks.update(mu_leaf_per_task[gid].keys())
    task_names = sorted(all_tasks)
    n_tasks = len(task_names)
    task_idx = {t: i for i, t in enumerate(task_names)}

    phi_matrix = np.zeros((G, n_tasks), dtype=np.float64)
    for gid in range(G):
        if gid in mu_leaf_per_task:
            for t, v in mu_leaf_per_task[gid].items():
                phi_matrix[gid, task_idx[t]] = float(v)
    return phi_matrix, task_names


def filter_active_tasks_and_weights(
    phi_matrix: np.ndarray,
    task_names: List[str],
    v0_per_task: Dict[str, float],
    task_counts: Dict[str, int],
    use_subject: bool,
    epsilon_task: float = 0.005,
    task_weight_mode: str = "auto",
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Filter active tasks and compute weights.

    task_weight_mode:
      "macro" - equal weight per task. Use when subjects are meaningful
                (e.g. MMLU with 57 subjects, eval = macro accuracy).
      "micro" - weight proportional to # of eval examples in each task.
                Use when tasks are embedding chunks (eval ≈ micro accuracy).
      "auto"  - if use_subject=True (real subjects), use "macro".
                if use_subject=False (embedding chunks), use "micro".

    Returns:
      phi_active: (G, n_active) filtered phi matrix
      active_tasks: list of active task names
      w_t: (n_active,) normalized task weights
      b_t: (n_active,) null scores for active tasks
    """
    G, n_tasks = phi_matrix.shape
    phi_pos = np.maximum(phi_matrix, 0.0)

    # max gain per task
    max_gain = phi_pos.max(axis=0)  # (n_tasks,)

    # filter active tasks
    active_mask = max_gain > epsilon_task
    active_idx = np.where(active_mask)[0]

    if active_idx.size == 0:
        active_idx = np.arange(n_tasks)

    active_tasks = [task_names[i] for i in active_idx]
    phi_active = phi_matrix[:, active_idx]

    # resolve auto mode
    if task_weight_mode == "auto":
        task_weight_mode = "macro" if use_subject else "micro"

    # compute weights
    if task_weight_mode == "macro":
        # equal weight per task (macro accuracy)
        w_t = np.ones(len(active_tasks), dtype=np.float64) / len(active_tasks)
    elif task_weight_mode == "micro":
        # weight proportional to the full validation row count in each task
        counts = np.array([task_counts.get(t, 1) for t in active_tasks], dtype=np.float64)
        w_sum = counts.sum()
        w_t = counts / w_sum if w_sum > 0 else np.ones(len(active_tasks)) / len(active_tasks)
    else:
        raise ValueError(f"Unknown task_weight_mode: {task_weight_mode}")

    # null scores
    b_t = np.array([v0_per_task.get(t, 0.0) for t in active_tasks], dtype=np.float64)

    print(f"[task_filter] {len(active_tasks)}/{n_tasks} active tasks, "
          f"weight_mode={task_weight_mode}, "
          f"top weights: {sorted(zip(active_tasks, w_t), key=lambda x:-x[1])[:5]}")

    return phi_active, active_tasks, w_t, b_t


# ============================================================
# Stage 1.6 & 1.7: Conservative and Optimistic Greedy
# ============================================================

def _conservative_task_score(
    best_pos: np.ndarray,   # (n_tasks,) current max positive per task
    sum_neg: np.ndarray,    # (n_tasks,) current sum of negatives per task
    b_t: np.ndarray,        # (n_tasks,) null scores
    w_t: np.ndarray,        # (n_tasks,) weights
) -> float:
    """U_con(S) = Σ_t w_t · clip[0,1](b_t + best_pos_t - sum_neg_t)"""
    y_hat = np.clip(b_t + best_pos - sum_neg, 0.0, 1.0)
    return float(np.dot(w_t, y_hat))


def greedy_taskaware(
    phi_matrix: np.ndarray,  # (G, n_tasks)
    w_t: np.ndarray,         # (n_tasks,)
    b_t: np.ndarray,         # (n_tasks,)
    leaf_groups: List[np.ndarray],
    *,
    points_cap: int,
    mode: str = "conservative",  # "conservative" (HARP-C) or "optimistic" (HARP-E)
    use_knee_stop: bool = True,
    knee_min_steps: int = 10,
    knee_patience: int = 5,
    candidate_gids: Optional[np.ndarray] = None,  # if None, use all G
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Task-aware greedy leaf selection under one of the two HARP envelopes.

    mode="conservative"  (HARP-C):  ŷ_t = clip(b_t + max_{g∈S} p(g,t) - Σ n(g,t))
    mode="optimistic"    (HARP-E):  ŷ_t = clip(b_t + Σ p(g,t) - Σ n(g,t))

    where p(g,t)=max(phi[g,t],0) and n(g,t)=max(-phi[g,t],0). HARP-C credits only
    the strongest positive leaf per task (controls redundancy); HARP-E sums all
    positive contributions (recovers complementary regions). Both subtract harms.

    Returns: (selected_leaf_order, info_dict)
    """
    G, n_tasks = phi_matrix.shape
    phi_pos = np.maximum(phi_matrix, 0.0)
    phi_neg = np.maximum(-phi_matrix, 0.0)
    sizes = np.array([len(leaf_groups[g]) for g in range(G)], dtype=int)

    if candidate_gids is None:
        candidate_gids = np.arange(G, dtype=int)
    else:
        candidate_gids = np.asarray(candidate_gids, dtype=int)

    picked = np.zeros(G, dtype=bool)
    S: List[int] = []
    gains: List[float] = []
    pts_total = 0

    # running state
    best_pos = np.zeros(n_tasks, dtype=np.float64)     # max positive per task in S
    sum_neg = np.zeros(n_tasks, dtype=np.float64)       # sum of negatives per task in S
    sum_pos = np.zeros(n_tasks, dtype=np.float64)       # sum of positives (for optimistic)

    current_score = _conservative_task_score(best_pos, sum_neg, b_t, w_t)

    cutoff_idx = None

    while True:
        best_gain = -np.inf
        best_g = -1

        for g in candidate_gids:
            g = int(g)
            if picked[g]:
                continue

            if mode == "conservative":
                new_best_pos = np.maximum(best_pos, phi_pos[g])
                new_sum_neg = sum_neg + phi_neg[g]
                y_new = np.clip(b_t + new_best_pos - new_sum_neg, 0.0, 1.0)
            else:  # optimistic
                new_sum_pos = sum_pos + phi_pos[g]
                new_sum_neg = sum_neg + phi_neg[g]
                y_new = np.clip(b_t + new_sum_pos - new_sum_neg, 0.0, 1.0)

            score_new = float(np.dot(w_t, y_new))
            gain = score_new - current_score

            if gain > best_gain:
                best_gain = gain
                best_g = g

        if best_g < 0 or not np.isfinite(best_gain):
            break

        S.append(best_g)
        gains.append(best_gain)
        picked[best_g] = True
        pts_total += int(sizes[best_g])

        # update running state
        best_pos = np.maximum(best_pos, phi_pos[best_g])
        sum_neg = sum_neg + phi_neg[best_g]
        sum_pos = sum_pos + phi_pos[best_g]

        if mode == "conservative":
            current_score = _conservative_task_score(best_pos, sum_neg, b_t, w_t)
        else:
            y_hat = np.clip(b_t + sum_pos - sum_neg, 0.0, 1.0)
            current_score = float(np.dot(w_t, y_hat))

        # stopping
        if cutoff_idx is None:
            should_stop = False
            if pts_total >= points_cap:
                should_stop = True
            if use_knee_stop and len(S) >= knee_min_steps:
                xs = np.arange(1, len(S) + 1, dtype=np.float64)
                ys = np.asarray(gains, dtype=np.float64)
                knee_idx = _knee_point_from_curve(xs, ys)
                if len(S) >= int(xs[knee_idx]) + knee_patience:
                    should_stop = True
            if should_stop:
                cutoff_idx = len(S)

    if cutoff_idx is None:
        cutoff_idx = len(S)

    S_arr = np.asarray(S, dtype=int)
    info = {
        "mode": mode,
        "k_selected": int(cutoff_idx),
        "k_total": int(S_arr.size),
        "points_total": int(pts_total),
        "gains": np.asarray(gains, dtype=np.float64),
        "cutoff_idx": int(cutoff_idx),
        "final_score": float(current_score),
    }
    return S_arr, info


# ============================================================
# Stage 2: Pair-type scores (R, H, C)
# ============================================================



# ============================================================
# Main entry point: harp()  — first-order HARP-C / HARP-E selection
# ============================================================

@dataclass
class HarpResult:
    """Output of :func:`harp`.

    Attributes:
        harp_c: 1-D int array of selected training-example indices under the
            HARP-C (conservative) envelope, best-prefix cut, capped at the point
            budget ``TOP_K_POINTS``.
        harp_e: 1-D int array of selected training-example indices under the
            HARP-E (expansive) envelope, best-prefix cut, capped at the point
            budget ``TOP_K_POINTS`` (same cap as HARP-C).
        info: diagnostics — per-task main effects (phi), task filtering, the
            full conservative / expansive leaf orders, cut indices, and the
            bootstrap summary.
    """
    harp_c: np.ndarray
    harp_e: np.ndarray
    info: Dict[str, Any]


def harp(
    X_feat: np.ndarray,
    base_model_name: str,
    train_examples: List[Dict],
    eval_examples: List[Dict],
    *,
    tokenizer,
    num_gpus: Optional[int] = None,
    eval_metric: str = "accuracy",
    rng: np.random.Generator = np.random.default_rng(0),
    lora_config: Optional[Dict[str, Any]] = None,

    # --- Stage 1a: node-leaf hierarchy ---
    large_max_group_size: int = 25000,
    parent_max_group_size: int = 8192,
    leaf_max_group_size: int = 1024,
    ann_use_gpu: bool = True,

    # --- Stage 1b: representative bootstrap + empirical-Bayes main effects ---
    rep_per_parent: int = 3,
    bootstrap_seed_int: int = 0,
    v0: Optional[float] = None,
    se_floor: float = 1e-3,
    use_eb_shrinkage: bool = True,
    prior_var: float = 0.01,

    # --- Selection ---
    TOP_K_POINTS: int = 10000,           # point budget for the HARP-C / HARP-E cuts
    use_knee_stop: bool = True,
    task_weight_mode: str = "auto",      # "macro", "micro", or "auto"
    epsilon_task: float = 0.005,         # active-task filtering threshold

    # --- Inputs / artifacts ---
    representatives_path: str = "representatives.jsonl",
    hierarchy_csv_path: Optional[str] = "workspace/hierarchy_cluster_sizes.csv",

    # --- Stage-1b checkpointing (pass a dir to enable resume-on-preempt) ---
    checkpoint_dir: Optional[str] = None,
) -> "HarpResult":
    """Select finetuning data with HARP (Hierarchical Active Region Pruning).

    HARP is a train-based data selector. It (1) organises the training pool into
    a node-leaf hierarchy, (2) finetunes only representative leaves and infers
    the unmeasured ones with empirical-Bayes posteriors to obtain per-task main
    effects ``phi[g, d]``, then (3) greedily selects leaves under one of two
    complementary first-order envelopes:

      HARP-C (conservative)
          y_hat_d(S) = clip[0,1]( b_d + max_{g in S} phi+_g(d) - sum_{g in S} phi-_g(d) )
          Credits only the strongest positive leaf per domain while accumulating
          all estimated harms -> controls redundancy.

      HARP-E (expansive)
          y_hat_d(S) = clip[0,1]( b_d + sum_{g in S} phi_g(d) )
          Sums positive contributions (and subtracts harms) -> recovers multiple
          complementary regions.

    where ``phi+ = max(phi, 0)``, ``phi- = max(-phi, 0)``, and ``b_d`` is the
    null (base-model) score for domain ``d``.

    Both envelopes use the *best-prefix* cut: rank leaves by greedy marginal
    gain, then keep the prefix that maximises cumulative gain (``argmax`` of the
    running sum). Both point selections are then capped at the budget
    ``TOP_K_POINTS``.

    This implements the first-order method from the paper: per-task main effects
    are aggregated additively, with no pairwise / second-order interaction terms.

    Args:
        X_feat: (n, d) L2-normalisable feature matrix, one row per train example.
        base_model_name: HF model id finetuned on each representative leaf.
        train_examples / eval_examples: SFT rows (input, target, subject) for the
            training pool and the complete fixed held-out validation set. Every
            validation row is scored for every measured model; final test data
            must not be supplied for selection.
        tokenizer: HF tokenizer for ``base_model_name``.
        v0: base-model utility; if ``None`` it is read from ``representatives_path``.
        TOP_K_POINTS: point budget applied to both selections.
        representatives_path: JSONL containing one base-model score (``acc01``)
            for every row in the same full validation set, with matching subject
            labels, used for the null scores ``b_d`` and the task grouping.
        hierarchy_csv_path: where to dump cluster sizes (diagnostic; ``None`` to skip).
        checkpoint_dir: enable Stage-1b resume-on-preempt when set.

    Returns:
        HarpResult with ``.harp_c`` and ``.harp_e`` (selected train-example index
        arrays) and ``.info`` (diagnostics).
    """
    lora_config = dict(lora_config or {})

    if X_feat is None or len(train_examples) == 0:
        return HarpResult(
            harp_c=np.zeros(0, dtype=int),
            harp_e=np.zeros(0, dtype=int),
            info={"error": "empty input"},
        )

    _validate_validation_split(train_examples, eval_examples)

    X_feat = l2_normalize_cuda_if_available(X_feat)
    n = int(X_feat.shape[0])
    assert n == len(train_examples), "X_feat rows must match train_examples length."

    rep_path = representatives_path
    if v0 is None:
        v0, _micro, _per_mean, _cnts, _stats = compute_macro_micro(
            rep_path,
            subject_key="subject",
            score_key="acc01",
        )
    # Per-task null scores and task grouping come from the same file.
    subject_to_task, v0_per_task, task_counts, use_subject_grouping = load_task_grouping_from_representatives(
        rep_path,
        subject_key="subject",
        score_key="acc01",
    )
    task_counts = _validate_full_validation_baseline(
        eval_examples, subject_to_task, v0_per_task, task_counts,
    )

    print(f"[per-task] use_subject={use_subject_grouping}, n_tasks={len(v0_per_task)}")
    for t, s in sorted(v0_per_task.items())[:10]:
        print(f"  task={t}: null_score={s:.4f}")

    # Build subj_id_to_task mapping: we need to map the integer subject IDs
    # from precompute_dataset_arrays back to task names.
    # The eval_arrays will have a _subject_map: {subject_name: int_id}
    # We'll build this after spawning the pool (when eval arrays are created).


    # ----------------------------
    # (1) Build parent/leaf hierarchy
    # ----------------------------
    large_groups,parent_groups,leaf_groups,large_of_parent,parent_of_leaf,large_of_leaf,leaf_of_i,leaf_feat = build_parent_leaf_hierarchy(
                                                                                                                            X_feat,
                                                                                                                            rng=rng,
                                                                                                                            large_max_group_size=int(large_max_group_size),
                                                                                                                            parent_max_group_size=int(parent_max_group_size),
                                                                                                                            leaf_max_group_size=int(leaf_max_group_size),
                                                                                                                            leaf_min_group_size=int(leaf_max_group_size/4),
                                                                                                                            # parent_n_clusters=parent_n_clusters,
                                                                                                                            ann_use_gpu=bool(ann_use_gpu),
                                                                                                                            lora_config=lora_config)
    


    if hierarchy_csv_path is not None:
        write_hierarchy_cluster_sizes(
            hierarchy_csv_path,
            large_groups=large_groups,
            parent_groups=parent_groups,
            leaf_groups=leaf_groups,
            large_of_parent=large_of_parent,
            parent_of_leaf=parent_of_leaf,
        )
        print(f"Wrote cluster sizes to {hierarchy_csv_path}")
                                                                                                                        

    P = int(len(parent_groups))
    G = int(len(leaf_groups))

    info: Dict[str, Any] = {
        "mode": "RepBoot-Shapley-v1",
        "n": int(n),
        "P_parent": int(P),
        "G_leaf": int(G),
        "parent_max_group_size": int(parent_max_group_size),
        "leaf_max_group_size": int(leaf_max_group_size),
        "rep_per_parent": int(rep_per_parent),
        "eval_metric": str(eval_metric),
        "validation_protocol": "full_fixed",
        "n_validation": int(len(eval_examples)),
    }

    if G == 0:
        return HarpResult(
            harp_c=np.zeros(0, dtype=int),
            harp_e=np.zeros(0, dtype=int),
            info={**info, "error": "no leaf groups built"},
        )

    # ----------------------------
    # (2) Spawn pool and run bootstrap
    # ----------------------------
    with GPUWorkerPool(
        base_model_name=base_model_name,
        train_examples=train_examples,
        eval_examples=eval_examples,
        num_gpus=num_gpus,
        lora_config=lora_config,
        verbose=bool(lora_config.get("verbose", True)),
        eval_metric=eval_metric,
    ) as gpu_pool:

        # compute v0 once if not provided
        if v0 is None:
            raise RuntimeError("v0 is None: pass v0 from caller (baseline utility) for now.")

        # Build subj_id_to_task mapping from the eval arrays' subject_map
        # The eval arrays have _subject_map: {subject_name_str: int_id}
        eval_subject_map = gpu_pool.eval_arrays.get("_subject_map", {})
        # Invert: int_id -> subject_name_str
        id_to_subject_name = {v: k for k, v in eval_subject_map.items()}
        # Now map: int_id -> task_name (via subject_to_task)
        subj_id_to_task: Dict[int, str] = {}
        for int_id, subj_name in id_to_subject_name.items():
            task = subject_to_task.get(subj_name, None)
            if task is None:
                # Fallback: if use_subject, task = subject name itself
                if use_subject_grouping:
                    task = subj_name if subj_name else "unknown"
                else:
                    task = "unknown"
            subj_id_to_task[int(int_id)] = task
        
        print(f"[subj_id_to_task] mapped {len(subj_id_to_task)} subject IDs to tasks")

        # --- Stage 1: Per-task bootstrap (with optional checkpoint resume) ---
        ckpt_key = _stage1b_ckpt_key(
            leaf_groups=leaf_groups,
            bootstrap_seed_int=int(bootstrap_seed_int),
            eval_metric=str(eval_metric),
            base_model_name=str(base_model_name),
            n_train=int(len(train_examples)),
            n_eval=int(len(eval_examples)),
            validation_fingerprint=_validation_fingerprint(
                gpu_pool.eval_arrays, subj_id_to_task, v0_per_task, eval_examples,
            ),
        )
        # Try CKPT-B first (covers the most ground); fall back to CKPT-A.
        _ckpt_b = _try_load_stage1b_ckpt(checkpoint_dir, "B", ckpt_key)
        _ckpt_a = _ckpt_b or _try_load_stage1b_ckpt(checkpoint_dir, "A", ckpt_key)

        if _ckpt_a is not None:
            mu_leaf = _ckpt_a["mu_leaf"]
            se_leaf = _ckpt_a["se_leaf"]
            mu_leaf_per_task = _ckpt_a["mu_leaf_per_task"]
            se_leaf_per_task = _ckpt_a["se_leaf_per_task"]
            leaf_is_measured = _ckpt_a["leaf_is_measured"]
            boot_info = _ckpt_a["boot_info"]
            print(f"[Stage1] bootstrap SKIPPED via cached CKPT-"
                  f"{'B' if _ckpt_b else 'A'} (saved on prior run)", flush=True)
        else:
            (
                mu_leaf, se_leaf,
                mu_leaf_per_task, se_leaf_per_task,
                leaf_is_measured,
                boot_info
            ) = bootstrap_parent_representative_independent_per_task(
                gpu_pool,
                leaf_groups=leaf_groups,
                leaf_feat=leaf_feat,
                parent_of_leaf=parent_of_leaf,
                group_of_i=leaf_of_i,
                eval_metric=eval_metric,
                v0_per_task=v0_per_task,
                subj_id_to_task=subj_id_to_task,
                rep_per_parent=int(rep_per_parent),
                seed_int=int(bootstrap_seed_int),
                rng=rng,
                se_floor=float(se_floor),
                use_eb_shrinkage=bool(use_eb_shrinkage),
                prior_var=float(prior_var),
            )
            _save_stage1b_ckpt(checkpoint_dir, "A", ckpt_key, {
                "mu_leaf": mu_leaf,
                "se_leaf": se_leaf,
                "mu_leaf_per_task": mu_leaf_per_task,
                "se_leaf_per_task": se_leaf_per_task,
                "leaf_is_measured": leaf_is_measured,
                "boot_info": boot_info,
            })

        info["bootstrap"] = boot_info["bootstrap"]
        info["bootstrap"]["P_parent"] = int(P)
        info["bootstrap"]["G_leaf"] = int(G)

        # keep some useful summaries
        info["bootstrap"]["measured_leaf_frac"] = float(np.mean(leaf_is_measured)) if G > 0 else 0.0

        # Print per-task phi summary for measured leaves
        print(f"\n[Stage1 Per-Task] {len(mu_leaf_per_task)} leaves, {len(v0_per_task)} tasks")
        for gid in sorted(mu_leaf_per_task.keys())[:5]:
            tasks = mu_leaf_per_task[gid]
            macro = np.mean(list(tasks.values())) if tasks else 0.0
            top_tasks = sorted(tasks.items(), key=lambda x: -x[1])[:3]
            print(f"  leaf {gid}: macro_phi={macro:.4f}, top_tasks={top_tasks}")

        # Store per-task info for later use in greedy selection
        info["mu_leaf_per_task"] = {
            str(gid): {t: round(v, 6) for t, v in tasks.items()}
            for gid, tasks in mu_leaf_per_task.items()
        }
        info["v0_per_task"] = {str(k): float(v) for k, v in v0_per_task.items()}

        # ============================================================
        # Stage 1.5: Task filtering, weights, phi matrix
        # ============================================================
        phi_matrix, all_task_names = build_task_phi_matrix(mu_leaf_per_task, G)
        phi_active, active_tasks, w_t, b_t = filter_active_tasks_and_weights(
            phi_matrix, all_task_names, v0_per_task, task_counts, use_subject_grouping,
            epsilon_task=float(epsilon_task),
            task_weight_mode=str(task_weight_mode),
        )
        n_active = len(active_tasks)
        print(f"[Stage1.5] {n_active} active tasks, phi_active shape={phi_active.shape}")

        info["task_filtering"] = {
            "n_all_tasks": len(all_task_names),
            "n_active_tasks": n_active,
            "active_tasks": active_tasks,
            "w_t": w_t.tolist(),
            "b_t": b_t.tolist(),
        }

        # ============================================================
        # HARP-C: conservative-envelope greedy -> S(0)
        # ============================================================
        refine_buffer = float(lora_config.get("refine_buffer", 1.30))
        points_cap_max = int(TOP_K_POINTS * refine_buffer)

        S0_leaf, S0_info = greedy_taskaware(
            phi_active, w_t, b_t, leaf_groups,
            points_cap=points_cap_max,
            mode="conservative",
            use_knee_stop=bool(use_knee_stop),
        )
        S0_cutoff = int(S0_info["cutoff_idx"])
        print(f"[HARP-C] conservative greedy: {S0_cutoff} leaves, "
              f"{S0_info['points_total']} points, score={S0_info['final_score']:.6f}")
        S0_info["leaf_order"] = S0_leaf.tolist()

        # ============================================================
        # HARP-E: expansive-envelope greedy -> S(+)
        # ============================================================
        Splus_leaf, Splus_info = greedy_taskaware(
            phi_active, w_t, b_t, leaf_groups,
            points_cap=points_cap_max,
            mode="optimistic",   # expansive (additive) envelope = HARP-E
            use_knee_stop=bool(use_knee_stop),
        )
        Splus_cutoff = int(Splus_info["cutoff_idx"])
        print(f"[HARP-E] expansive greedy: {Splus_cutoff} leaves, "
              f"{Splus_info['points_total']} points, score={Splus_info['final_score']:.6f}")
        # ------------------------------------------------------------------
        # Best-prefix selection for each envelope.
        # Rank leaves by greedy marginal gain, keep the prefix that maximises
        # cumulative gain (argmax of the running sum), then expand the selected
        # leaves to unique training-example indices.
        # ------------------------------------------------------------------
        def _leaves_to_points(leaf_order: np.ndarray) -> np.ndarray:
            pts: List[int] = []
            seen: set = set()
            for g in leaf_order.tolist():
                for pid in np.asarray(leaf_groups[int(g)], dtype=int).tolist():
                    if pid not in seen:
                        seen.add(pid)
                        pts.append(int(pid))
            return np.asarray(pts, dtype=int)

        def _best_prefix_cut(gains: np.ndarray, n_leaves: int, fallback: int) -> int:
            if gains.size > 0:
                cut = int(np.argmax(np.cumsum(gains))) + 1
            else:
                cut = int(fallback)
            return int(min(cut, int(n_leaves)))

        def _select_points(leaf_order: np.ndarray) -> np.ndarray:
            # Expand leaves to unique point ids, then apply the point budget.
            # On dense pools argmax-cumsum can pick a prefix above the budget;
            # truncate the tail, preserving leaf order. Applied symmetrically to
            # HARP-C and HARP-E.
            pts = _leaves_to_points(leaf_order)
            if int(TOP_K_POINTS) > 0 and pts.size > int(TOP_K_POINTS):
                pts = pts[: int(TOP_K_POINTS)]
            return pts

        # --- HARP-C: conservative envelope, best-prefix cut (capped at TOP_K_POINTS) ---
        s0_gains = np.asarray(S0_info.get("gains", []), dtype=float)
        c_cut = _best_prefix_cut(s0_gains, S0_leaf.size, S0_cutoff)
        harp_c_leaves = S0_leaf[:c_cut]
        harp_c_points = _select_points(harp_c_leaves)
        S0_info["best_prefix_leaf_cutoff_idx"] = int(c_cut)
        S0_info["best_prefix_point_cutoff_idx"] = int(harp_c_points.size)
        S0_info["best_prefix_leaves"] = harp_c_leaves.tolist()
        S0_info["best_prefix_points"] = harp_c_points.tolist()
        info["harp_c"] = S0_info

        # --- HARP-E: expansive envelope, best-prefix cut (capped at TOP_K_POINTS) ---
        Splus_info["leaf_order"] = Splus_leaf.tolist()
        splus_gains = np.asarray(Splus_info.get("gains", []), dtype=float)
        e_cut = _best_prefix_cut(splus_gains, Splus_leaf.size, Splus_cutoff)
        harp_e_leaves = Splus_leaf[:e_cut]
        harp_e_points = _select_points(harp_e_leaves)
        Splus_info["best_prefix_leaf_cutoff_idx"] = int(e_cut)
        Splus_info["best_prefix_point_cutoff_idx"] = int(harp_e_points.size)
        Splus_info["best_prefix_leaves"] = harp_e_leaves.tolist()
        Splus_info["best_prefix_points"] = harp_e_points.tolist()
        info["harp_e"] = Splus_info

        print(
            f"[HARP] HARP-C: {c_cut} leaves / {harp_c_points.size} pts;  "
            f"HARP-E: {e_cut} leaves / {harp_e_points.size} pts.",
            flush=True,
        )
        return HarpResult(harp_c=harp_c_points, harp_e=harp_e_points, info=info)
