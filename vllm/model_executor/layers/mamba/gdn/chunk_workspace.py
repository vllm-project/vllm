# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLA gated-delta chunk-scan workspace accounting for KV cache sizing.

V1 ``profile_run`` only warms GDN prefill kernels at ``T = FLA_CHUNK_SIZE``
and never builds ``attn_metadata``, so the large ``h`` / ``v_new`` / ``w`` /
``u`` / ``A`` tensors used by a full ``max_num_batched_tokens`` prefill are
invisible to ``memory_profiling``. This module estimates that live-set and
reserves only the portion not already observed during profile / CUDA-graph
capture. See #54775.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import format_gib, format_mib

logger = init_logger(__name__)

# Triton/FLA allocates these tensors together in chunk_gated_delta_rule_fwd.
# ``o`` is produced later and can reuse freed memory; SSM ``final_state``
# is GDN cache, not this workspace.
_FLA_PREFILL_BACKENDS = frozenset({"triton"})


@dataclass(frozen=True)
class GdnWorkspaceSpec:
    num_value_heads_local: int
    key_dim: int
    value_dim: int
    dtype_itemsize: int
    chunk_size: int
    backend: str


@dataclass(frozen=True)
class GdnWorkspaceEstimate:
    num_tokens: int
    num_seqs: int
    num_chunks: int
    num_value_heads_local: int
    key_dim: int
    value_dim: int
    chunk_size: int
    elem_size: int
    h_bytes: int
    v_new_bytes: int
    w_bytes: int
    u_bytes: int
    a_bytes: int
    hv_only_bytes: int
    peak_live_bytes: int


def max_gdn_chunk_count(
    num_tokens: int,
    max_num_seqs: int,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> int:
    """Worst-case FLA chunk count for a token and sequence budget.

    Varlen ``NT = sum(ceil(seq_i / chunk_size))``. The maximum is
    ``(S - 1)`` length-1 sequences plus one sequence with the remainder.
    """
    if num_tokens <= 0 or chunk_size <= 0:
        return 0
    num_seqs = min(max(max_num_seqs, 1), num_tokens)
    return (num_seqs - 1) + math.ceil((num_tokens - num_seqs + 1) / chunk_size)


def estimate_gdn_chunk_workspace(
    num_tokens: int,
    num_value_heads_local: int,
    key_dim: int,
    value_dim: int,
    dtype_itemsize: int = 2,
    chunk_size: int = FLA_CHUNK_SIZE,
    num_seqs: int = 1,
    seq_lens: list[int] | None = None,
    worst_case_chunks: bool = False,
) -> GdnWorkspaceEstimate:
    """Estimate the FLA gated-delta prefill live-set.

    Covers ``h + v_new + w + u + A``. ``h`` scales with chunk count, not
    raw tokens. Packed varlen layout is ``B = 1``.

    Args:
        num_tokens: Total tokens in the batch (``T``).
        num_value_heads_local: Value heads on this TP rank.
        key_dim: Per-head key dim.
        value_dim: Per-head value dim.
        dtype_itemsize: Activation element size (bf16/fp16 is 2).
        chunk_size: FLA chunk size (``BT``, default 64).
        num_seqs: Sequence count when ``seq_lens`` is omitted.
        seq_lens: Explicit per-sequence lengths. Must sum to ``num_tokens``.
        worst_case_chunks: If true and ``seq_lens`` is omitted, use the
            maximum chunk count for ``(num_tokens, num_seqs)``.

    Returns:
        Breakdown of each tensor and the concurrent peak live-set.

    Raises:
        ValueError: If ``seq_lens`` does not sum to ``num_tokens``, or
            ``num_seqs`` is not positive when needed.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if seq_lens is not None:
        if sum(seq_lens) != num_tokens:
            raise ValueError("seq_lens must sum to num_tokens")
        num_chunks = sum(math.ceil(t / chunk_size) for t in seq_lens if t > 0)
        resolved_seqs = len(seq_lens)
    elif worst_case_chunks:
        if num_seqs <= 0:
            raise ValueError("num_seqs must be positive")
        num_chunks = max_gdn_chunk_count(num_tokens, num_seqs, chunk_size)
        resolved_seqs = min(num_seqs, max(num_tokens, 1)) if num_tokens else 0
    else:
        if num_seqs <= 0:
            raise ValueError("num_seqs must be positive")
        if num_tokens == 0:
            seq_lens = [0] * num_seqs
        elif num_tokens % num_seqs != 0:
            base = num_tokens // num_seqs
            rem = num_tokens % num_seqs
            seq_lens = [base + (1 if i < rem else 0) for i in range(num_seqs)]
        else:
            seq_lens = [num_tokens // num_seqs] * num_seqs
        num_chunks = sum(math.ceil(t / chunk_size) for t in seq_lens if t > 0)
        resolved_seqs = num_seqs

    h = num_value_heads_local
    k = key_dim
    v = value_dim
    e = dtype_itemsize
    h_bytes = num_chunks * h * v * k * e
    v_new_bytes = num_tokens * h * v * e
    w_bytes = num_tokens * h * k * e
    u_bytes = num_tokens * h * v * e
    a_bytes = num_tokens * h * chunk_size * 4
    hv_only = h_bytes + v_new_bytes
    peak_live = h_bytes + v_new_bytes + w_bytes + u_bytes + a_bytes
    return GdnWorkspaceEstimate(
        num_tokens=num_tokens,
        num_seqs=resolved_seqs,
        num_chunks=num_chunks,
        num_value_heads_local=h,
        key_dim=k,
        value_dim=v,
        chunk_size=chunk_size,
        elem_size=e,
        h_bytes=h_bytes,
        v_new_bytes=v_new_bytes,
        w_bytes=w_bytes,
        u_bytes=u_bytes,
        a_bytes=a_bytes,
        hv_only_bytes=hv_only,
        peak_live_bytes=peak_live,
    )


def uncovered_gdn_workspace_bytes(upper_bytes: int, covered_bytes: int) -> int:
    """Bytes to reserve beyond what profile / CUDA-graph already charged."""
    return max(0, upper_bytes - max(covered_bytes, 0))


def _format_workspace_bytes(n: int) -> str:
    n = max(n, 0)
    if n >= GiB_bytes:
        return f"{format_gib(n)} GiB"
    return f"{format_mib(n)} MiB"


def _dtype_itemsize(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize in (1, 2, 4, 8):
        return int(itemsize)
    return 2


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _spec_from_module(module: Any) -> GdnWorkspaceSpec | None:
    num_v = _positive_int(getattr(module, "num_v_heads", None))
    tp_size = getattr(module, "tp_size", 1)
    key_dim = _positive_int(getattr(module, "head_k_dim", None))
    value_dim = _positive_int(getattr(module, "head_v_dim", None))
    if num_v is None or key_dim is None or value_dim is None:
        return None
    cls_name = type(module).__name__
    if getattr(module, "gdn_prefill_backend", None) is None and (
        "GatedDeltaNet" not in cls_name and "GatedDelta" not in cls_name
    ):
        return None
    tp_size = max(int(tp_size), 1)
    backend = str(getattr(module, "gdn_prefill_backend", "triton") or "triton")
    model_config = getattr(module, "model_config", None)
    dtype = getattr(model_config, "dtype", torch.bfloat16)
    return GdnWorkspaceSpec(
        num_value_heads_local=num_v // tp_size,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype_itemsize=_dtype_itemsize(dtype),
        chunk_size=FLA_CHUNK_SIZE,
        backend=backend,
    )


def resolve_gdn_workspace_spec(
    vllm_config: Any,
    model: Any | None = None,
) -> GdnWorkspaceSpec | None:
    """Read GDN FLA workspace dims from a loaded layer or HF text config."""
    if model is not None:
        modules_fn = getattr(model, "modules", None)
        if callable(modules_fn):
            for module in modules_fn():
                spec = _spec_from_module(module)
                if spec is not None:
                    return spec
        spec = _spec_from_module(model)
        if spec is not None:
            return spec

    model_config = getattr(vllm_config, "model_config", None)
    text = getattr(model_config, "hf_text_config", None)
    if text is None:
        text = getattr(model_config, "hf_config", None)
    num_v = _positive_int(getattr(text, "linear_num_value_heads", None))
    key_dim = _positive_int(getattr(text, "linear_key_head_dim", None))
    value_dim = _positive_int(getattr(text, "linear_value_head_dim", None))
    if num_v is None or key_dim is None or value_dim is None:
        return None

    parallel_config = getattr(vllm_config, "parallel_config", None)
    tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)
    dtype = getattr(model_config, "dtype", torch.bfloat16)
    backend = "triton"
    if model is not None:
        for module in getattr(model, "modules", lambda: ())():
            layer_backend = getattr(module, "gdn_prefill_backend", None)
            if layer_backend is not None:
                backend = str(layer_backend)
                break
    return GdnWorkspaceSpec(
        num_value_heads_local=num_v // tp_size,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype_itemsize=_dtype_itemsize(dtype),
        chunk_size=FLA_CHUNK_SIZE,
        backend=backend,
    )


def apply_gdn_chunk_workspace_reservation(
    available_kv_bytes: int,
    vllm_config: Any,
    model: Any | None,
    covered_bytes: int,
    *,
    apply_reservation: bool = True,
) -> int:
    """Subtract uncovered FLA workspace from the KV-cache budget.

    No-op for non-GDN models and non-Triton/FLA prefill backends. Logs the
    upper bound, profile coverage, and extra reservation.

    Args:
        available_kv_bytes: KV bytes after regular memory profiling.
        vllm_config: Runtime config (scheduler, parallel, model).
        model: Loaded model, used to read local heads and backend.
        covered_bytes: Workspace bytes already charged during profile.
        apply_reservation: If false, log only (manual KV size path).

    Returns:
        Remaining KV bytes after the uncovered-delta reservation.

    Raises:
        RuntimeError: If the uncovered workspace exceeds ``available_kv_bytes``.
    """
    spec = resolve_gdn_workspace_spec(vllm_config, model)
    if spec is None:
        return available_kv_bytes
    if spec.backend not in _FLA_PREFILL_BACKENDS:
        logger.debug(
            "Skipping GDN chunk workspace reservation for backend %s.",
            spec.backend,
        )
        return available_kv_bytes

    scheduler = getattr(vllm_config, "scheduler_config", None)
    num_tokens = int(getattr(scheduler, "max_num_batched_tokens", 0) or 0)
    num_seqs = int(getattr(scheduler, "max_num_seqs", 1) or 1)
    if num_tokens <= 0 or spec.num_value_heads_local <= 0:
        return available_kv_bytes

    estimate = estimate_gdn_chunk_workspace(
        num_tokens=num_tokens,
        num_value_heads_local=spec.num_value_heads_local,
        key_dim=spec.key_dim,
        value_dim=spec.value_dim,
        dtype_itemsize=spec.dtype_itemsize,
        chunk_size=spec.chunk_size,
        num_seqs=num_seqs,
        worst_case_chunks=True,
    )
    additional = uncovered_gdn_workspace_bytes(estimate.peak_live_bytes, covered_bytes)
    logger.info(
        "GDN chunk workspace upper bound: %s (tokens=%d, chunks=%d, local_v_heads=%d)",
        _format_workspace_bytes(estimate.peak_live_bytes),
        estimate.num_tokens,
        estimate.num_chunks,
        spec.num_value_heads_local,
    )
    logger.info(
        "Workspace covered by profile:    %s",
        _format_workspace_bytes(max(covered_bytes, 0)),
    )
    logger.info(
        "Additional safety reservation:   %s",
        _format_workspace_bytes(additional),
    )
    if not apply_reservation:
        logger.warning(
            "Skipping GDN chunk workspace reservation because "
            "kv_cache_memory_bytes is set."
        )
        return available_kv_bytes

    remaining = available_kv_bytes - additional
    if remaining < 0:
        raise RuntimeError(
            "GDN chunk-scan workspace requires "
            f"{_format_workspace_bytes(additional)} beyond profile, but only "
            f"{_format_workspace_bytes(available_kv_bytes)} is left for the "
            "KV cache. Lower --max-num-batched-tokens or "
            "--gpu-memory-utilization."
        )
    return remaining
