# SPDX-License-Identifier: Apache-2.0
"""Genesis P102 — Unified spec-decode metadata (TRT-LLM-style first-class).

Per Sander 2026-04-28: study TRT-LLM's first-class `SpecMetadata.is_cuda_graph`
architecture, adapt to Genesis vLLM Patches.

================================================================
PROBLEM
================================================================

Current Genesis approach: each spec-decode-aware patch (P67/P67b/P78/P98/P99)
re-derives "are we in spec-decode? cudagraph? warmup?" from local hints.
There is NO single source of truth, so v756 regression broke specifically
because P67 fired on chunked-prefill batches when spec-decode was OFF
(memory.md: "kernel hook misfires for chunked-prefill batches when
spec-decode is OFF; safety gate added").

Adding the next spec method (P82 SGLang-style, future Eagle3, DFlash)
requires updating 4+ scattered shape-check sites simultaneously — error-
prone and inconsistent.

================================================================
SOLUTION (TRT-LLM-inspired)
================================================================

Single dataclass `GenesisSpecMeta` holds the full spec-decode + cudagraph
state for the current step. Set by V1 GPUModelRunner.execute_model
prelude. Consumed by predicate functions (`should_dispatch_p67(...)`)
that replace scattered inline checks.

TRT-LLM reference: `tensorrt_llm/_torch/speculative/interface.py:330-400`
defines `SpecMetadata` base + `is_cuda_graph: bool` + `spec_dec_mode`
enum. Key TRT-LLM idiom (dflash.py:269):

  is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()

This canonical "warmup vs replay" distinction is what bites our P67.
With unified meta, dispatch becomes:

  if not should_dispatch_p67(...): fall through

================================================================
MIGRATION (3 phases)
================================================================

Phase 1 (THIS PATCH): assertion-only mode — predicates RECORD their
eligibility decision; existing inline checks remain authoritative;
disagreement logs WARNING but doesn't change behavior.

Phase 2 (next session): existing inline checks REPLACED by predicate
calls; original behavior preserved.

Phase 3 (later): remove inline checks entirely; spec_meta becomes
single source of truth.

Status: opt-in via `GENESIS_ENABLE_P102=1`. Default OFF first deploy.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Architectural ref: NVIDIA TensorRT-LLM SpecMetadata.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("genesis.spec_meta")


# ─── Data model ────────────────────────────────────────────────────────


@dataclass
class GenesisSpecMeta:
    """Unified spec-decode + cudagraph metadata for current step.

    Mirrors the *publicly observable* parts of TRT-LLM's `SpecMetadata`
    relevant to our patcher. Populated by V1 GPUModelRunner prelude.

    Fields:
        is_cuda_graph_capture: True between graph.begin/end_capture
        is_cuda_graph_replay: True when replaying captured graph
        current_query_len: 1 (no-spec/decode) or K+1 (spec verify)
        spec_method: "none" | "ngram" | "mtp" | "eagle3" | "dflash"
        runtime_K: 0 if disabled else N speculative tokens
        batch_size: requests in current batch
        is_chunked_prefill: True for chunked prefill (closes v756)
        step_index: monotonic step counter for diagnostics
    """
    is_cuda_graph_capture: bool = False
    is_cuda_graph_replay: bool = False
    current_query_len: int = 1
    spec_method: str = "none"
    runtime_K: int = 0
    batch_size: int = 0
    is_chunked_prefill: bool = False
    step_index: int = 0

    # Phase 1 telemetry: count predicate calls and disagreements
    _disagreement_count: int = field(default=0, repr=False)
    _predicate_calls: int = field(default=0, repr=False)


# ─── Thread-local global context ────────────────────────────────────────
# Single-process single-thread (vLLM scheduler is sequential per process).
# In TP, each worker has its own _CTX. _LOCK protects against test races.

_LOCK = threading.Lock()
_CTX: Optional[GenesisSpecMeta] = None


def current() -> GenesisSpecMeta:
    """Return current step's GenesisSpecMeta. Creates default if unset."""
    global _CTX
    if _CTX is None:
        with _LOCK:
            if _CTX is None:
                _CTX = GenesisSpecMeta()
    return _CTX


def set_step(meta: GenesisSpecMeta) -> None:
    """Set GenesisSpecMeta for the current step. Called by V1 runner prelude."""
    global _CTX
    with _LOCK:
        _CTX = meta


def reset_for_tests() -> None:
    """TESTS ONLY — clear the singleton context."""
    global _CTX
    with _LOCK:
        _CTX = None


def is_active() -> bool:
    """Returns True iff `GENESIS_ENABLE_P102` is set in env."""
    return os.environ.get("GENESIS_ENABLE_P102", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ─── Centralized predicates ─────────────────────────────────────────────
# Each predicate replaces a scattered inline check. Phase 1: predicates
# RECORD their decision; existing inline checks remain authoritative.


def should_dispatch_p67(
    *,
    Hq: int,
    Hk: int,
    head_size: int,
    max_query_len: int,
    max_seq_len: int,
    N: int,
    layer_kind: str = "turboquant",
    max_kp1: int = 16,
    max_prior: int = 4096,
    inline_decision: Optional[bool] = None,
) -> bool:
    """Centralized P67 dispatch predicate.

    Returns True iff:
    - GENESIS_ENABLE_P102 active
    - layer is TurboQuantAttentionImpl
    - K+1 spec verify shape (max_query_len in (1, max_kp1])
    - Has prior cached KV (max_seq_len > max_query_len)
    - Prior len within bake limit
    - Uniform K+1 batch (N % max_query_len == 0, N>0)
    - GQA shape valid (Hq>=8, head_size in {128,256}, Hq//Hk >= 2)
    - NOT chunked-prefill (closes v756)
    - Spec-decode runtime_K > 0
    """
    m = current()
    m._predicate_calls += 1

    decision = (
        m.runtime_K > 0
        and not m.is_chunked_prefill
        and m.spec_method in ("mtp", "ngram", "eagle3", "dflash")
        and layer_kind == "turboquant"
        and Hq >= 8
        and head_size in (128, 256)
        and (Hq // max(Hk, 1)) >= 2
        and 1 < max_query_len <= max_kp1
        and max_seq_len > max_query_len
        and (max_seq_len - max_query_len) <= max_prior
        and N > 0
        and (N % max_query_len) == 0
    )

    # Phase 1 assertion: log disagreements with inline check
    if inline_decision is not None and inline_decision != decision:
        m._disagreement_count += 1
        log.warning(
            "[Genesis P102] should_dispatch_p67 DISAGREEMENT: "
            "predicate=%s inline=%s | step=%d Hq=%d Hk=%d D=%d "
            "max_q=%d max_s=%d N=%d K=%d method=%s chunked=%s",
            decision, inline_decision, m.step_index, Hq, Hk, head_size,
            max_query_len, max_seq_len, N, m.runtime_K, m.spec_method,
            m.is_chunked_prefill,
        )

    return decision


def should_use_perlayer_workspace() -> bool:
    """P98 logic centralized: prefer per-layer cache during cudagraph replay."""
    m = current()
    m._predicate_calls += 1
    return m.is_cuda_graph_replay


def should_skip_tolist() -> bool:
    """P78 logic: skip .tolist() during cudagraph capture (graph-unsafe)."""
    m = current()
    m._predicate_calls += 1
    return m.is_cuda_graph_capture


def should_use_workspace_cache() -> bool:
    """P99 logic: use memoized WorkspaceManager when not in capture
    (capture path takes the no-cache branch for safety)."""
    m = current()
    m._predicate_calls += 1
    return not m.is_cuda_graph_capture


# ─── Diagnostics ────────────────────────────────────────────────────────


def get_telemetry() -> dict:
    """Return Phase-1 diagnostics: predicate calls + disagreement count.

    Used by tests + apply_all reporting to verify Phase 1 assertions.
    """
    m = current()
    return {
        "predicate_calls": m._predicate_calls,
        "disagreement_count": m._disagreement_count,
        "current_query_len": m.current_query_len,
        "spec_method": m.spec_method,
        "runtime_K": m.runtime_K,
        "is_cuda_graph_capture": m.is_cuda_graph_capture,
        "is_cuda_graph_replay": m.is_cuda_graph_replay,
        "is_chunked_prefill": m.is_chunked_prefill,
        "step_index": m.step_index,
    }


def log_telemetry_summary() -> None:
    """Log a one-line telemetry summary. Called periodically by runner."""
    t = get_telemetry()
    if t["predicate_calls"] == 0:
        return  # not active
    log.info(
        "[Genesis P102] step=%d predicate_calls=%d disagreements=%d "
        "(query_len=%d K=%d method=%s capture=%s replay=%s)",
        t["step_index"], t["predicate_calls"], t["disagreement_count"],
        t["current_query_len"], t["runtime_K"], t["spec_method"],
        t["is_cuda_graph_capture"], t["is_cuda_graph_replay"],
    )
