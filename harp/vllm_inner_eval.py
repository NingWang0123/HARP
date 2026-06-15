"""HTTP-based vLLM inner-eval helper for HARP Stage 1b.

Spawns a long-lived vLLM OpenAI-compatible server in the `infer` conda env
(vLLM is not installed in `eb-lora`), pinned to a dedicated GPU. The HARP
training process (running in `eb-lora`) sends per-leaf eval requests via REST.

Key features:
- Dynamic LoRA loading at runtime (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True).
- Each leaf saves its LoRA to a temp dir, registers it with the server, runs
  eval, then unloads to free a slot in the LoRA cache.
- Health-checks on startup; falls back to None on failure (caller should
  detect and use HF generate path).
"""
from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests  # available in eb-lora env
except ImportError:
    requests = None  # caller will detect and skip vLLM path


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class VLLMInnerEvalServer:
    """Wraps a long-lived `vllm serve` subprocess on a dedicated GPU."""

    def __init__(
        self,
        base_model: str,
        gpu_id: int,
        *,
        max_loras: int = 100,
        max_lora_rank: int = 16,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        infer_env: str = "infer",
        port: Optional[int] = None,
        startup_timeout: int = 900,
        log_path: Optional[str] = None,
    ):
        if requests is None:
            raise RuntimeError("`requests` not available in this env; cannot use vLLM HTTP eval")

        self.base_model = base_model
        self.gpu_id = int(gpu_id)
        self.max_loras = int(max_loras)
        self.port = int(port) if port else _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: Optional[subprocess.Popen] = None
        self._registered_loras: Dict[str, str] = {}  # name -> path
        self._next_lora_id = 1

        # Spawn `vllm serve` on a dedicated GPU.
        # CUDA_VISIBLE_DEVICES forces GPU placement.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["VLLM_USE_V1"] = "0"

        # Prefer VLLM_PYTHON (path to the vllm venv's python) so we don't
        # depend on conda being on PATH inside SLURM jobs. The vllm CLI
        # script lives next to it as <venv>/bin/vllm. Fall back to
        # `conda run -n <infer_env>` only when VLLM_PYTHON is unset.
        vllm_python = os.environ.get("VLLM_PYTHON", "").strip()
        vllm_bin: Optional[str] = None
        if vllm_python:
            candidate = os.path.join(os.path.dirname(vllm_python), "vllm")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                vllm_bin = candidate

        vllm_args = [
            "serve", base_model,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--enable-lora",
            "--max-loras", str(self.max_loras),
            "--max-lora-rank", str(max_lora_rank),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--dtype", "float16",
        ]
        if vllm_bin:
            cmd = [vllm_bin] + vllm_args
            print(f"[vllm_inner_eval] launching via VLLM_PYTHON-resolved bin: {vllm_bin}",
                  flush=True)
        else:
            cmd = [
                "conda", "run", "-n", infer_env,
                "--live-stream", "--no-capture-output",
                "vllm",
            ] + vllm_args
            print(f"[vllm_inner_eval] VLLM_PYTHON unset/invalid; falling back to "
                  f"`conda run -n {infer_env}` (may fail if conda not on PATH)",
                  flush=True)

        log_f = open(log_path, "ab") if log_path else subprocess.DEVNULL
        print(f"[vllm_inner_eval] Starting vLLM on GPU {self.gpu_id}, port {self.port}: "
              f"{' '.join(shlex.quote(c) for c in cmd[:4])} ...")
        self.proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_f if log_path else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
            preexec_fn=os.setsid,  # so we can SIGTERM the whole group
        )

        # Health check: poll /health until 200 or timeout
        ok = self._wait_ready(startup_timeout)
        if not ok:
            self.shutdown()
            raise RuntimeError(f"vLLM server failed to come up within {startup_timeout}s "
                               f"(port {self.port}, gpu {self.gpu_id})")
        print(f"[vllm_inner_eval] vLLM ready at {self.base_url}")

    def _wait_ready(self, timeout: int) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False  # subprocess died
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(2)
        return False

    def register_lora(self, lora_name: str, lora_path: str) -> None:
        """Tell vLLM to load this adapter (if not already loaded)."""
        if lora_name in self._registered_loras:
            return
        url = f"{self.base_url}/v1/load_lora_adapter"
        body = {"lora_name": lora_name, "lora_path": str(Path(lora_path).resolve())}
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        self._registered_loras[lora_name] = lora_path

    def unregister_lora(self, lora_name: str) -> None:
        if lora_name not in self._registered_loras:
            return
        url = f"{self.base_url}/v1/unload_lora_adapter"
        r = requests.post(url, json={"lora_name": lora_name}, timeout=30)
        # 404 if already unloaded — fine
        if r.status_code not in (200, 404):
            r.raise_for_status()
        self._registered_loras.pop(lora_name, None)

    def generate(
        self,
        lora_name: str,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> List[str]:
        """OpenAI-compatible /v1/completions; routes to the named LoRA adapter."""
        url = f"{self.base_url}/v1/completions"
        body = {
            "model": lora_name,
            "prompt": prompts,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "n": 1,
        }
        r = requests.post(url, json=body, timeout=600)
        r.raise_for_status()
        data = r.json()
        return [c["text"] for c in data["choices"]]

    def eval_with_lora(
        self,
        leaf_id: int,
        lora_path: str,
        prompts: List[str],
        max_tokens: int = 64,
    ) -> List[str]:
        """Convenience: register LoRA → generate → unregister."""
        name = f"leaf_{leaf_id}"
        try:
            self.register_lora(name, lora_path)
            return self.generate(name, prompts, max_tokens=max_tokens, temperature=0.0)
        finally:
            # Auto-unregister to free LoRA slot for next leaf.
            try:
                self.unregister_lora(name)
            except Exception:
                pass

    def batch_generate_multi_lora(
        self,
        items: List[Dict],
        max_tokens: int = 64,
        chunk_size: Optional[int] = None,
    ) -> Dict[int, List[str]]:
        """Run batched eval across multiple LoRA adapters.

        Args:
            items: list of dicts each with keys: 'gid' (int), 'lora_path' (str),
                'prompts' (List[str]). Each adapter is registered, generated for
                its prompts, then unregistered (to free a slot for the next chunk).
            max_tokens: per-prompt completion budget.
            chunk_size: how many adapters to keep registered concurrently.
                Defaults to self.max_loras (vLLM caches them; chunking just
                bounds peak slot usage).

        Returns:
            {gid: list_of_decoded_strings} aligned with input prompts.
        """
        if chunk_size is None:
            chunk_size = max(1, self.max_loras - 4)  # keep a couple slots free

        results: Dict[int, List[str]] = {}
        for chunk_start in range(0, len(items), chunk_size):
            chunk = items[chunk_start:chunk_start + chunk_size]

            # Register every adapter in the chunk
            for it in chunk:
                self.register_lora(f"leaf_{int(it['gid'])}", it["lora_path"])

            # Generate one adapter at a time. vLLM's /v1/completions accepts
            # `model` per-request, so we call once per (adapter, batched_prompts).
            try:
                for it in chunk:
                    gid = int(it["gid"])
                    prompts = list(it["prompts"])
                    if not prompts:
                        results[gid] = []
                        continue
                    name = f"leaf_{gid}"
                    texts = self.generate(name, prompts, max_tokens=max_tokens, temperature=0.0)
                    results[gid] = texts
            finally:
                # Unregister all adapters in chunk to free slots
                for it in chunk:
                    try:
                        self.unregister_lora(f"leaf_{int(it['gid'])}")
                    except Exception:
                        pass

        return results

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        finally:
            self.proc = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


def maybe_start_server(
    base_model: str,
    gpu_id: int,
    *,
    log_path: Optional[str] = None,
) -> Optional[VLLMInnerEvalServer]:
    """Start a vLLM server, returning None on any failure (caller falls back to HF)."""
    if os.environ.get("HARP_INNER_EVAL", "").lower() != "vllm":
        return None
    if requests is None:
        print("[vllm_inner_eval] `requests` not installed; falling back to HF generate")
        return None
    try:
        return VLLMInnerEvalServer(base_model, gpu_id=gpu_id, log_path=log_path)
    except Exception as e:
        print(f"[vllm_inner_eval] Failed to start vLLM server, falling back to HF: {e}")
        return None
