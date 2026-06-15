"""Batched HARP Stage 1b inner-eval: train all adapters first, then batch-eval via vLLM.

This is the single-GPU alternative to per-leaf train+eval interleaving. The key
insight: HARP's per-leaf gain estimation is INDEPENDENT (each leaf measured from
the same snapshot), so we can train all leaves first, save their LoRA adapters,
then batch-evaluate all of them via a vLLM subprocess on the same GPU.

Phases per stage:
  1. TRAIN: workers train each leaf, save adapter to disk, return (gid, path)
  2. RELEASE: del peft_model in workers, torch.cuda.empty_cache()
  3. EVAL: spawn vLLM, batch_generate_multi_lora across all adapters
  4. AGGREGATE: parse predictions → per-task accuracies, return {gid: per_task_acc}

Single GPU is sufficient because train and vLLM phases don't coexist in time.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================
# Train-only worker (mirror of _worker_run_independent_singleton_per_task,
# but skips eval — just trains and saves the adapter)
# ============================================================
def train_only_singleton_per_task(
    task: Tuple[int, np.ndarray, str, int, str, str],
) -> Tuple[int, str]:
    """Train LoRA on the leaf's indices, save adapter, return (gid, adapter_path).

    task = (gid, idxs, eval_metric, seed_int, v0_per_task_json, adapters_dir).
    The trailing `adapters_dir` is where this worker saves `leaf_{gid}/`.

    Empty result (path="") signals worker error; caller treats as missing.
    """
    # Imports inside the function so this module is safe to import in
    # non-multiprocessing contexts (e.g. when not running HARP).
    from .selector import (
        _G,
        _reset_lora_params_from_snapshot,
        _train_epochs_on_indices,
    )

    gid, idxs, eval_metric, seed_int, _, adapters_dir = task
    try:
        device: torch.device = _G["device"]
        model = _G["peft_model"]
        snap = _G["lora_snapshot"]
        h = _G["lora_hparams"]
        train_data = _G["train_data"]

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
            return int(gid), ""

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

        # Save the LoRA adapter only (skip embedding-layer copies; ~50MB).
        # ROOT-CAUSE FIX: PEFT's save_pretrained() calls safetensors which uses
        # buffered I/O. Under tight disk, writes can succeed at the page-cache
        # level without flushing, and close() error propagation is unreliable.
        # The result: save returns "success" but the on-disk file is 0-byte
        # or truncated. vLLM later 404s when trying to load it.
        #
        # Fix: after save, fsync each file and load_file() the safetensors
        # back to verify it's structurally valid. If fsync fails or load fails,
        # we know the save is bad and return empty path so the bootstrap
        # excludes the leaf (instead of feeding vLLM a corrupt adapter).
        adapter_path = os.path.join(adapters_dir, f"leaf_{int(gid)}")
        os.makedirs(adapter_path, exist_ok=True)

        # Pre-flight: check free disk space before attempting save. Real LoRA
        # r=16 on Qwen3-4B writes ~125 MB; require 200 MB headroom to avoid
        # the silent-truncation regime in the first place.
        try:
            free_b = shutil.disk_usage(adapter_path).free
        except Exception:
            free_b = -1
        if 0 <= free_b < 200 * 1024 * 1024:
            gpu_id = _G.get("gpu_id", "?")
            print(f"[Worker gpu={gpu_id}] gid={gid} SAVE SKIPPED: "
                  f"only {free_b / 1024 / 1024:.0f}MB free in {adapter_path} "
                  f"(need 200MB for safe save)", flush=True)
            return int(gid), ""

        try:
            model.save_pretrained(adapter_path, save_embedding_layers=False)
        except Exception as e:
            gpu_id = _G.get("gpu_id", "?")
            print(f"[Worker gpu={gpu_id}] gid={gid} save_pretrained raised: {e}",
                  flush=True)
            return int(gid), ""

        # Force disk flush + verify by loading back. fsync converts a buffered
        # write that might be sitting in page cache into a durable error if
        # the write didn't actually land.
        cfg_p = os.path.join(adapter_path, "adapter_config.json")
        wts_p = os.path.join(adapter_path, "adapter_model.safetensors")
        save_ok = True
        fail_reason = ""
        for fp in (cfg_p, wts_p):
            if not os.path.isfile(fp):
                save_ok = False
                fail_reason = f"missing {os.path.basename(fp)}"
                break
            try:
                fd = os.open(fp, os.O_RDONLY)
                try:
                    os.fsync(fd)  # idempotent flush from kernel cache to disk
                finally:
                    os.close(fd)
            except OSError as e:
                save_ok = False
                fail_reason = f"fsync({os.path.basename(fp)}) failed: {e}"
                break

        # Read-back integrity check: load the safetensors with strict mode.
        # If file is truncated, safe_open raises HeaderTooSmall / similar.
        if save_ok:
            try:
                from safetensors import safe_open
                with safe_open(wts_p, framework="pt") as f:
                    keys = list(f.keys())
                    if not keys:
                        save_ok = False
                        fail_reason = "safetensors has 0 tensors"
            except Exception as e:
                save_ok = False
                fail_reason = f"safe_open() failed: {type(e).__name__}: {e}"

        cfg_sz = os.path.getsize(cfg_p) if os.path.isfile(cfg_p) else 0
        wts_sz = os.path.getsize(wts_p) if os.path.isfile(wts_p) else 0
        if not save_ok:
            gpu_id = _G.get("gpu_id", "?")
            print(f"[Worker gpu={gpu_id}] gid={gid} SAVE INVALID: {fail_reason} "
                  f"(config={cfg_sz}B weights={wts_sz}B free={free_b / 1024 / 1024:.0f}MB) "
                  f"dir={adapter_path}", flush=True)
            return int(gid), ""

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return int(gid), adapter_path

    except Exception as e:
        gpu_id = _G.get("gpu_id", None)
        print(f"[Worker gpu={gpu_id}] train-only gid={gid} error: {e}")
        traceback.print_exc()
        return int(gid), ""


# ============================================================
# Batch-eval phase: spawn vLLM, eval all adapters, score per-task
# ============================================================
def _decode_eval_prompts(eval_dl, tokenizer) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Walk the eval DataLoader and detokenize each example's prompt.

    Returns:
        prompts: List[str] — one per eval example, same order as eval_dl iteration
        subjects: List[int] — subject id per example
        mc_labels: List[int] — multi-choice gold index per example
        real_indices: List[int] — original eval-set index per example
    """
    prompts: List[str] = []
    subjects: List[int] = []
    mc_labels: List[int] = []
    real_idxs: List[int] = []

    for batch in eval_dl:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        sub = batch["subjects"]
        mcl = batch["mc_labels"]
        ridx = batch.get("_real_idx")
        bs = ids.shape[0]
        for i in range(bs):
            valid = mask[i].bool()
            tokens = ids[i][valid].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            prompts.append(text)
            subjects.append(int(sub[i].item()))
            mc_labels.append(int(mcl[i].item()))
            real_idxs.append(int(ridx[i].item()) if ridx is not None else -1)
    return prompts, subjects, mc_labels, real_idxs


def _score_predictions_per_task(
    decoded_preds: List[str],
    subjects: List[int],
    mc_labels: List[int],
    real_indices: List[int],
    eval_task_types: List[str],
    eval_gold_strings: List[List[str]],
    primary_tt: str,
) -> Dict[int, float]:
    """Compute per-subject accuracy from predictions; mirrors HF path's scoring."""
    from .data_preprocess import score_response

    letters = ["A", "B", "C", "D"]
    subject_score_sum: Dict[int, float] = defaultdict(float)
    subject_total: Dict[int, int] = defaultdict(int)

    for i, pred_text in enumerate(decoded_preds):
        subj_id = int(subjects[i])
        subject_total[subj_id] += 1
        real_idx = int(real_indices[i])

        if eval_task_types and 0 <= real_idx < len(eval_task_types):
            tt = eval_task_types[real_idx]
        else:
            tt = primary_tt if primary_tt != "mixed" else "multi_choice"

        if tt == "multi_choice":
            gold_idx = int(mc_labels[i])
            gold_char = letters[gold_idx] if 0 <= gold_idx < len(letters) else ""
            s = score_response("multi_choice", pred_text, gold_char)
        else:
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

    per_task_acc: Dict[int, float] = {}
    for subj_id, n in subject_total.items():
        if n > 0:
            per_task_acc[int(subj_id)] = subject_score_sum[subj_id] / n
    return per_task_acc


def _max_new_tokens_for(primary_tt: str) -> int:
    if primary_tt in ("numeric", "latex_boxed"):
        return 512
    if primary_tt == "extractive":
        return 64
    return 256


def eval_all_adapters_via_vllm(
    adapter_paths: Dict[int, str],
    *,
    base_model: str,
    eval_dl,
    tokenizer,
    eval_task_types: List[str],
    eval_gold_strings: List[List[str]],
    gpu_id: int = 0,
    log_path: Optional[str] = None,
    max_loras: int = 64,
) -> Dict[int, Dict[int, float]]:
    """Spawn vLLM, batch-eval every adapter, return {gid: per_task_acc}.

    Caller should have released GPU memory (del peft_model + empty_cache) before
    calling this — vLLM grabs gpu_memory_utilization * total at startup.
    """
    from .vllm_inner_eval import VLLMInnerEvalServer

    # 0. Defensive: filter out adapters whose files vanished between save
    # and now (e.g. cleanup race from a sibling job on the same compute
    # node). Without this, a single missing adapter 404s the whole batch
    # and forces fallback to the slow HF-interleaved path. Skip the bad
    # ones and let the bootstrap inherit parent mu for them.
    valid_paths: Dict[int, str] = {}
    for gid, lora_path in adapter_paths.items():
        cfg_p = os.path.join(lora_path, "adapter_config.json")
        wts_p = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.isfile(cfg_p) and os.path.isfile(wts_p) \
           and os.path.getsize(wts_p) > 1_000_000:
            valid_paths[int(gid)] = lora_path
        else:
            cfg_sz = os.path.getsize(cfg_p) if os.path.isfile(cfg_p) else 0
            wts_sz = os.path.getsize(wts_p) if os.path.isfile(wts_p) else 0
            print(f"[batched_inner_eval] gid={gid} adapter MISSING/EMPTY at "
                  f"eval time (cfg={cfg_sz}B wts={wts_sz}B); skipping. "
                  f"path={lora_path}", flush=True)
    if not valid_paths:
        print("[batched_inner_eval] no valid adapters at eval time; returning empty",
              flush=True)
        return {}
    if len(valid_paths) < len(adapter_paths):
        print(f"[batched_inner_eval] proceeding with {len(valid_paths)}/"
              f"{len(adapter_paths)} surviving adapters", flush=True)
    adapter_paths = valid_paths

    # 1. Decode prompts once (shared across all adapters)
    prompts, subjects, mc_labels, real_idxs = _decode_eval_prompts(eval_dl, tokenizer)
    if not prompts:
        return {}

    primary_tt = (
        next(iter(set(eval_task_types))) if eval_task_types and len(set(eval_task_types)) == 1
        else "mixed"
    ) if eval_task_types else "multi_choice"
    max_new_tokens = _max_new_tokens_for(primary_tt)

    # 2. Spawn vLLM
    print(f"[batched_inner_eval] Spawning vLLM on GPU {gpu_id} for {len(adapter_paths)} adapters")
    server = VLLMInnerEvalServer(
        base_model,
        gpu_id=gpu_id,
        max_loras=max_loras,
        max_lora_rank=16,
        gpu_memory_utilization=float(os.environ.get("HARP_VLLM_MEM_UTIL", "0.55")),
        max_model_len=int(os.environ.get("HARP_VLLM_MAX_MODEL_LEN", "2048")),
        log_path=log_path,
        startup_timeout=int(os.environ.get("HARP_VLLM_STARTUP_TIMEOUT", "900")),
    )
    try:
        # 3. Build batch-eval input
        items = [
            {"gid": int(gid), "lora_path": str(path), "prompts": prompts}
            for gid, path in adapter_paths.items()
            if path  # skip empty (errored) adapters
        ]
        # 4. Batch generate — chunked by max_loras
        decoded_per_gid = server.batch_generate_multi_lora(
            items, max_tokens=max_new_tokens, chunk_size=max_loras - 4
        )
    finally:
        server.shutdown()

    # 5. Score per adapter
    out: Dict[int, Dict[int, float]] = {}
    for gid, decoded in decoded_per_gid.items():
        if len(decoded) != len(prompts):
            print(f"[batched_inner_eval] gid={gid}: prediction count {len(decoded)} != prompt count {len(prompts)}; excluded")
            continue
        per_task = _score_predictions_per_task(
            decoded, subjects, mc_labels, real_idxs,
            eval_task_types, eval_gold_strings, primary_tt,
        )
        out[int(gid)] = per_task
    return out


# ============================================================
# Convenience: full train + batch-eval cycle
# ============================================================
def parallel_train_then_batch_eval(
    gpu_pool,
    tasks: List[Tuple[int, np.ndarray, str, int, str]],
    *,
    base_model: str,
    log_path: Optional[str] = None,
    max_loras: int = 64,
    eval_gpu_id: int = 0,
) -> Dict[int, Dict[int, float]]:
    """Full single-GPU vLLM-batched cycle for one HARP stage.

    1. Dispatch train-only tasks via the worker pool
    2. Release training models (workers free their GPU memory)
    3. Spawn vLLM on `eval_gpu_id`, batch-eval all adapters
    4. Aggregate per-task accuracies

    Returns {gid: {subj_id: accuracy}} — same shape as
    parallel_run_independent_singletons_per_task.
    """
    if not tasks:
        return {}
    if gpu_pool.pool is None:
        raise RuntimeError("GPUWorkerPool not initialized")

    # Adapters dir lives under the data tmpdir; auto-cleanup happens with pool teardown
    adapters_dir = tempfile.mkdtemp(prefix="harp_adapters_", dir=getattr(gpu_pool, "_data_tmpdir", None) or None)

    # Augment tasks with adapters_dir for the train-only worker
    tasks_with_dir = [t + (adapters_dir,) for t in tasks]

    print(f"[batched_inner_eval] Train phase: {len(tasks)} leaves → {adapters_dir}")
    adapter_paths: Dict[int, str] = {}
    for gid, path in gpu_pool.pool.imap_unordered(
        train_only_singleton_per_task, tasks_with_dir, chunksize=1
    ):
        if path:
            adapter_paths[int(gid)] = path
        else:
            print(f"[batched_inner_eval] gid={gid} train returned empty path; excluded")

    if not adapter_paths:
        print("[batched_inner_eval] No adapters trained successfully; skipping eval phase")
        return {}

    # Workers don't auto-release the model. Tell them to free GPU before vLLM.
    print("[batched_inner_eval] Training done; releasing worker GPU memory...")
    try:
        # Run a no-op task that calls torch.cuda.empty_cache() in each worker.
        # Use a sentinel task list that maps to a release function.
        gpu_pool.pool.map(_release_worker_gpu, [None] * gpu_pool.num_gpus)
    except Exception as e:
        print(f"[batched_inner_eval] Worker release warning: {e}")

    # Eval phase
    eval_dl = gpu_pool.eval_arrays  # actually we need the DataLoader, see below
    # Actually the eval DL lives in worker _G, not on gpu_pool. We need to
    # reconstruct it in the main process. Store eval_arrays + eval_indices on
    # gpu_pool init so we can rebuild here.
    raise NotImplementedError(
        "parallel_train_then_batch_eval: eval_dl reconstruction in main process not yet wired. "
        "See follow-up integration step."
    )


def _release_worker_gpu(_unused) -> None:
    """Worker-side GPU cleanup: release peft_model and empty cache.

    Stashes the LoRA snapshot + model name so we can re-init in
    `_reinit_worker_peft_model` after vLLM eval is done.
    """
    print(f"[release_worker] starting", flush=True)
    try:
        from .selector import _G
        # Cache state needed for re-init
        _G["_releasable_state"] = {
            "base_model_name": _G.get("base_model_name"),
            "lora_snapshot": _G.get("lora_snapshot"),
            "lora_hparams": dict(_G.get("lora_hparams", {})),
            "device": _G.get("device"),
        }
        print(f"[release_worker] stashed state: base_model_name={_G.get('base_model_name')!r}, snap={_G.get('lora_snapshot') is not None}", flush=True)
        for k in ("peft_model", "model", "base_model"):
            if k in _G:
                try:
                    del _G[k]
                    print(f"[release_worker] deleted {k}", flush=True)
                except Exception as e:
                    print(f"[release_worker] failed to delete {k}: {e}", flush=True)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[release_worker] post-release GPU free={free_b/1e9:.2f}GB / total={total_b/1e9:.2f}GB", flush=True)
        print(f"[release_worker] DONE", flush=True)
    except Exception as e:
        print(f"[release_worker] EXCEPTION: {e}", flush=True)
        traceback.print_exc()


def _reinit_worker_peft_model(_unused) -> None:
    """Worker-side: rebuild peft_model after vLLM eval released GPU.

    Mirrors the model-loading section of _init_worker_process, but only the
    parts we need (model + LoRA + snapshot restoration). Other _G state
    (data tensors, eval_dl, tokenizer_gen) survives across calls.
    """
    print(f"[reinit_worker] starting", flush=True)
    try:
        from .selector import _G, infer_lora_targets, _reset_lora_params_from_snapshot
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        st = _G.get("_releasable_state", {})
        print(f"[reinit_worker] _releasable_state keys: {list(st.keys())}", flush=True)
        print(f"[reinit_worker] _G keys: {list(_G.keys())}", flush=True)
        base_model_name = st.get("base_model_name") or _G.get("base_model_name")
        snap = st.get("lora_snapshot") or _G.get("lora_snapshot")
        h = st.get("lora_hparams") or _G.get("lora_hparams", {})
        device = st.get("device") or _G.get("device")
        print(f"[reinit_worker] base_model_name={base_model_name!r} snap_keys={list(snap.keys()) if snap else None} device={device}", flush=True)

        if base_model_name is None:
            print(f"[reinit_worker] FATAL: base_model_name is None — cannot re-init", flush=True)
            return

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch_dtype = torch.float16
        attn_impl = h.get("attn_implementation", None)
        kwargs = {}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        # Determine cuda device id from device
        gpu_id = 0
        try:
            gpu_id = device.index or 0
        except Exception:
            pass

        print(f"[reinit_worker] loading {base_model_name} on cuda:{gpu_id}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map={"": f"cuda:{gpu_id}"},
            **kwargs,
        )
        base_model.eval()
        print(f"[reinit_worker] base_model loaded", flush=True)

        targets = infer_lora_targets(base_model)
        cfg = LoraConfig(
            r=int(h.get("lora_r", 8)),
            lora_alpha=int(h.get("lora_alpha", 16)),
            lora_dropout=float(h.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets if targets else None,
        )
        peft_model = get_peft_model(base_model, cfg)
        # Match _init_worker_process — gated by HARP_GRAD_CKPT (default 0 = off).
        if os.environ.get("HARP_GRAD_CKPT", "0") == "1":
            peft_model.enable_input_require_grads()
            peft_model.gradient_checkpointing_enable()
        # Restore LoRA snapshot so subsequent stage starts from the same state
        if snap is not None:
            _reset_lora_params_from_snapshot(peft_model, snap, device)

        _G["base_model"] = base_model
        _G["peft_model"] = peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[reinit_worker] DONE — _G now has base_model + peft_model", flush=True)
    except Exception as e:
        print(f"[reinit_worker] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
