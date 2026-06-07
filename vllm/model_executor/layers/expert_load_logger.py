# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B2 expert-load logging hook (research instrumentation, off by default).

The B2 / TierShift benchmark needs **per-(layer, expert) token load over time** to
test whether the hot MoE-expert set drifts (Q1/Q5/Q6). The B1 router-logit hook
(``router_logit_logger``) dumps full ``(num_tokens, n_experts)`` logits — far too heavy
to leave on for a whole trace replay. This hook is the lightweight sibling: it does the
top-k expert selection itself, accumulates a per-expert *count* into windowed counters,
and flushes one small row per (layer, expert) at a fixed wall-clock cadence to
``expert.jsonl``.

It shares the same insertion point as the B1 hook — right after ``self.gate(...)`` in the
MoE block — and the same design constraints:

- **Kernel-free / pure Python.** No custom ops, so ``VLLM_USE_PRECOMPILED=1`` builds keep
  working. The top-k is a single ``torch.topk`` on the (already-computed) router logits.
- **Zero overhead when disabled.** Enabled only if ``VLLM_B2_EXPERT_LOG_DIR`` is set; the
  enabled flag is read once and cached, so the disabled path is one bool check.
- **Never crash a compiled graph.** ``.cpu()`` + file IO would break a ``torch.compile``
  region, so the hook no-ops while compiling. Capture runs must use ``enforce_eager=True``.
- **Bounded volume.** One row per (layer, expert) *per window*, not per forward. A 16-layer
  / 64-expert model emits ≤ 1024 short rows per window (default 1 s).

This is model-agnostic: the same one-line call goes in any MoE block's forward (OLMoE
here; ``qwen3_moe.py`` / ``mixtral.py`` for the Initial Plan 2 routing-granularity sweep).

Output: one append-only JSONL per process, ``expert.<pid>.jsonl`` in the log dir. Each row::

    window_id, t_start, t_end, layer_id, expert_id, token_load, n_experts, top_k,
    phase{prefill|decode|mixed}, segment_label, pid

``token_load`` is the number of *token-slots* routed to that expert in the window (a token
that selects k experts contributes 1 to each of its k experts). ``phase`` is a best-effort
prefill/decode tag from the vLLM forward context (Initial Plan 2 phase split).
``segment_label`` is copied from ``VLLM_B2_SEGMENT`` if set (single-segment captures); for a
drifting multi-segment replay leave it unset and join the windows to the replay's
``segments.jsonl`` by wall-clock time offline (``b2tel.telemetry.assign_segments``). The
``expert_class{consistent|temporal}`` field in the contract is derived *offline* (GEM
taxonomy), not emitted here.

Env vars:
  VLLM_B2_EXPERT_LOG_DIR   enable + output directory (unset = disabled)
  VLLM_B2_WINDOW_S         window length in seconds; default 1.0
  VLLM_B2_SAMPLE           fraction of forward calls to count, (0, 1]; default 1.0
  VLLM_B2_SEGMENT          optional segment label copied into each row
  VLLM_B2_REPLICA_ID       optional tag copied into each row
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
_state: dict | None = None  # lazily-built config + per-window counters + fd
_DISABLED = object()  # sentinel: checked env, logging is off


def _is_compiling() -> bool:
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    try:
        return bool(fn()) if fn is not None else False
    except Exception:
        return False


def _current_phase() -> str:
    """Best-effort prefill/decode tag for the current forward (Initial Plan 2: phase split).

    Reads vLLM's forward context attn-metadata if present; a batch that is all single-token
    sequences is ``decode``, all multi-token is ``prefill``, otherwise ``mixed``. Falls back
    to ``mixed`` on any version mismatch — never raises into the model forward.
    """
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        am = getattr(ctx, "attn_metadata", None)
        if am is None:
            return "mixed"
        # attn_metadata may be a dict keyed by layer-group on newer vLLM
        if isinstance(am, dict):
            am = next(iter(am.values()), None)
            if am is None:
                return "mixed"
        npref = getattr(am, "num_prefill_tokens", None)
        ndec = getattr(am, "num_decode_tokens", None)
        if npref is None and ndec is None:
            return "mixed"
        npref = int(npref or 0)
        ndec = int(ndec or 0)
        if npref and not ndec:
            return "prefill"
        if ndec and not npref:
            return "decode"
        return "mixed"
    except Exception:
        return "mixed"


def _init_state():
    """Read env once and build logging state (or mark disabled)."""
    log_dir = os.getenv("VLLM_B2_EXPERT_LOG_DIR")
    if not log_dir:
        return _DISABLED
    os.makedirs(log_dir, exist_ok=True)
    try:
        sample = float(os.getenv("VLLM_B2_SAMPLE", "1.0"))
    except ValueError:
        sample = 1.0
    sample = min(max(sample, 0.0), 1.0)
    if sample <= 0.0:
        return _DISABLED
    stride = max(1, round(1.0 / sample))  # deterministic 1-in-stride sampling
    try:
        window_s = float(os.getenv("VLLM_B2_WINDOW_S", "1.0"))
    except ValueError:
        window_s = 1.0
    window_s = max(window_s, 1e-3)
    path = os.path.join(log_dir, f"expert.{os.getpid()}.jsonl")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    return {
        "fd": fd,
        "stride": stride,
        "window_s": window_s,
        "segment": os.getenv("VLLM_B2_SEGMENT"),
        "replica_id": os.getenv("VLLM_B2_REPLICA_ID"),
        # mutable accumulator state (guarded by _lock):
        "call_counts": {},              # layer_id -> forward-call count (for striding)
        "counts": {},                   # (layer_id, phase) -> {expert_id: token_load}
        "n_experts": 0,
        "top_k": 0,
        "window_id": 0,
        "window_start_wall": None,      # time.time() at window open
    }


def _get_state():
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _init_state()
    return _state


def _flush_locked(state: dict, t_end: float) -> None:
    """Emit one row per (layer, expert) for the current window, then reset counters.

    Caller must hold ``_lock``. No-op if nothing was counted.
    """
    counts = state["counts"]
    t_start = state["window_start_wall"]
    if counts and t_start is not None:
        window_id = state["window_id"]
        n_experts = state["n_experts"]
        top_k = state["top_k"]
        segment = state["segment"]
        replica_id = state["replica_id"]
        fd = state["fd"]
        for (layer_id, phase), experts in counts.items():
            for expert_id, token_load in experts.items():
                if token_load <= 0:
                    continue
                record = {
                    "window_id": window_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    "token_load": int(token_load),
                    "n_experts": int(n_experts),
                    "top_k": int(top_k),
                    "phase": phase,
                    "pid": os.getpid(),
                }
                if segment is not None:
                    record["segment_label"] = segment
                if replica_id is not None:
                    record["replica_id"] = replica_id
                os.write(fd, _dumps(record))
        state["window_id"] = window_id + 1
    # reset for next window
    state["counts"] = {}
    state["window_start_wall"] = t_end


def maybe_log_expert_load(
    router_logits: torch.Tensor, layer_id: int, top_k: int
) -> None:
    """Accumulate per-expert token load for ``layer_id`` (no-op when disabled).

    ``router_logits`` has shape ``(num_tokens, n_experts)``. We take the top-``top_k``
    experts per token (top-k on logits == top-k on softmax, so this matches the model's
    selection) and add 1 per selected slot to that expert's running window count. Safe to
    call unconditionally from a MoE forward: returns immediately when disabled or while
    compiling. A new window is opened lazily and flushed when ``window_s`` elapses.
    """
    state = _get_state()
    if state is _DISABLED:
        return
    if _is_compiling():  # never break a torch.compile graph; capture in eager mode
        return
    if router_logits is None or router_logits.dim() < 2 or top_k <= 0:
        return

    with _lock:
        n = state["call_counts"].get(layer_id, 0)
        state["call_counts"][layer_id] = n + 1
        if n % state["stride"] != 0:  # strided sampling of forward calls
            return

        now = time.time()
        if state["window_start_wall"] is None:
            state["window_start_wall"] = now
        elif now - state["window_start_wall"] >= state["window_s"]:
            _flush_locked(state, now)  # closes the elapsed window, opens a fresh one

        logits = router_logits.detach()
        n_experts = int(logits.shape[-1])
        k = min(top_k, n_experts)
        # top-k expert ids per token, then count occurrences per expert
        sel = torch.topk(logits, k, dim=-1).indices.reshape(-1)
        binc = torch.bincount(sel, minlength=n_experts).to("cpu").tolist()

        state["n_experts"] = n_experts
        state["top_k"] = k
        phase = _current_phase()
        layer_counts = state["counts"].setdefault((layer_id, phase), {})
        for expert_id, c in enumerate(binc):
            if c:
                layer_counts[expert_id] = layer_counts.get(expert_id, 0) + c


def flush_expert_load() -> None:
    """Force-flush the current (partial) window. Call at end of a capture run so the
    final window isn't lost. No-op when disabled."""
    state = _get_state()
    if state is _DISABLED:
        return
    with _lock:
        _flush_locked(state, time.time())
