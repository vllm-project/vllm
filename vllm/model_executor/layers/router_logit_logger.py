# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1 router-logit logging hook (research instrumentation, off by default).

The B1 benchmark (Q6) needs per-token MoE *router logits* to study expert routing
under async multi-replica serving. No public vLLM API exposes them
(``--enable-return-routed-experts`` gives only expert IDs and has perf issues), so the
MoE ``SparseMoeBlock.forward`` patches call ``maybe_log_router_logits`` right after the
gate. This module is the entire mechanism; the model edits are one line each.

Design constraints (see code/b1/TRACK_A.md → A2):
- **Kernel-free / pure Python.** No custom ops, so ``VLLM_USE_PRECOMPILED=1`` builds
  keep working.
- **Zero overhead when disabled.** Enabled only if ``VLLM_B1_ROUTER_LOG_DIR`` is set;
  the enabled flag is read once and cached, so the disabled path is one bool check.
- **Never crash a compiled graph.** Logging does ``.cpu()`` + file IO, which would
  break a ``torch.compile`` region, so the hook no-ops while compiling. Capture runs
  must use ``enforce_eager=True`` (the B1 inference spike already runs eager).
- **Bounded volume.** Forward calls are strided by ``VLLM_B1_ROUTER_LOG_SAMPLE`` and
  tokens per record are capped by ``VLLM_B1_ROUTER_LOG_MAX_TOKENS``.

Output: one append-only JSONL per process, ``router_logits.<pid>.jsonl`` in the log
dir. The orchestrator stamps ``replica_id`` / ``router_version`` (read here from
``VLLM_B1_REPLICA_ID`` / ``VLLM_B1_ROUTER_VERSION`` when set) so records join the run.

Env vars:
  VLLM_B1_ROUTER_LOG_DIR         enable + output directory (unset = disabled)
  VLLM_B1_ROUTER_LOG_SAMPLE      fraction of forward calls to log, (0, 1]; default 1.0
  VLLM_B1_ROUTER_LOG_MAX_TOKENS  max token rows per record; default 256 (0 = all)
  VLLM_B1_REPLICA_ID             optional tag copied into each record
  VLLM_B1_ROUTER_VERSION         optional tag copied into each record
"""

from __future__ import annotations

import os
import threading
import time

import torch

try:
    import orjson as _json

    def _dumps(o: dict) -> bytes:
        return _json.dumps(o) + b"\n"
except ModuleNotFoundError:  # stdlib fallback; vLLM may not ship orjson
    import json as _stdjson

    def _dumps(o: dict) -> bytes:
        return (_stdjson.dumps(o, separators=(",", ":")) + "\n").encode()


_lock = threading.Lock()
_state: dict | None = None  # lazily-built config + per-layer counters + fd
_DISABLED = object()  # sentinel: checked env, logging is off


def _is_compiling() -> bool:
    # torch.compiler.is_compiling exists on torch>=2.1; be defensive.
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    try:
        return bool(fn()) if fn is not None else False
    except Exception:
        return False


def _init_state():
    """Read env once and build logging state (or mark disabled)."""
    log_dir = os.getenv("VLLM_B1_ROUTER_LOG_DIR")
    if not log_dir:
        return _DISABLED
    os.makedirs(log_dir, exist_ok=True)
    try:
        sample = float(os.getenv("VLLM_B1_ROUTER_LOG_SAMPLE", "1.0"))
    except ValueError:
        sample = 1.0
    sample = min(max(sample, 0.0), 1.0)
    if sample <= 0.0:
        return _DISABLED
    stride = max(1, round(1.0 / sample))  # deterministic 1-in-stride sampling
    try:
        max_tokens = int(os.getenv("VLLM_B1_ROUTER_LOG_MAX_TOKENS", "256"))
    except ValueError:
        max_tokens = 256
    path = os.path.join(log_dir, f"router_logits.{os.getpid()}.jsonl")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    return {
        "fd": fd,
        "stride": stride,
        "max_tokens": max(0, max_tokens),
        "replica_id": os.getenv("VLLM_B1_REPLICA_ID"),
        "router_version": os.getenv("VLLM_B1_ROUTER_VERSION"),
        "counters": {},  # layer_id -> per-layer forward-call count
    }


def _get_state():
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _init_state()
    return _state


def maybe_log_router_logits(router_logits: torch.Tensor, layer_id: int) -> None:
    """Log a sampled subset of ``router_logits`` for ``layer_id`` (no-op when disabled).

    ``router_logits`` has shape ``(num_tokens, n_experts)``. Safe to call from a MoE
    forward unconditionally: returns immediately when disabled or while compiling.
    """
    state = _get_state()
    if state is _DISABLED:
        return
    if _is_compiling():  # never break a torch.compile graph; capture in eager mode
        return

    with _lock:
        n = state["counters"].get(layer_id, 0)
        state["counters"][layer_id] = n + 1
        if n % state["stride"] != 0:  # strided sampling of forward calls
            return

        logits = router_logits.detach()
        num_tokens = int(logits.shape[0])
        n_experts = int(logits.shape[1]) if logits.dim() > 1 else 0

        max_tokens = state["max_tokens"]
        if max_tokens and num_tokens > max_tokens:
            idx = torch.linspace(
                0, num_tokens - 1, steps=max_tokens, device=logits.device
            ).long()
            logits = logits.index_select(0, idx)

        # detached copy to CPU (float32 for stable JSON); bounded by sampling above
        sampled = logits.to("cpu", dtype=torch.float32).tolist()
        record = {
            "layer_id": int(layer_id),
            "call_idx": n,
            "t_wall": time.time(),
            "num_tokens": num_tokens,
            "n_experts": n_experts,
            "sampled_tokens": len(sampled),
            "router_logits": sampled,
            "pid": os.getpid(),
        }
        if state["replica_id"] is not None:
            record["replica_id"] = state["replica_id"]
        if state["router_version"] is not None:
            record["router_version"] = state["router_version"]
        os.write(state["fd"], _dumps(record))
