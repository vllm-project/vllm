# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate and cached loader for DSV4.1 few-head sparse MLA prefill."""

from __future__ import annotations

from typing import Any

import torch

_FWD: Any = None


def fewhead_prefill_enabled() -> bool:
    """Return whether few-head sparse prefill is enabled.

    Returns:
        True when ``VLLM_DSV41_FEWHEAD_PREFILL`` is set (default on).

    """
    from vllm import envs

    return bool(envs.VLLM_DSV41_FEWHEAD_PREFILL)


def fewhead_min_sq() -> int:
    """Return the shortest query length that uses the few-head kernel.

    Returns:
        Non-negative token threshold. ``0`` means always on.

    """
    from vllm import envs

    return int(envs.VLLM_DSV41_FEWHEAD_MIN_SQ)


def should_use_fewhead_prefill(
    *,
    n_local_heads: int,
    padded_heads: int,
    s_q: int,
) -> bool:
    """Return whether this prefill chunk should use the native-head kernel.

    Args:
        n_local_heads: Unpadded local Q heads after TP.
        padded_heads: FlashMLA pad width (64 or 128).
        s_q: Query tokens in this chunk.

    Returns:
        True when the kernel is enabled, heads are padded, and ``s_q`` meets
        ``VLLM_DSV41_FEWHEAD_MIN_SQ``.

    """
    return (
        fewhead_prefill_enabled()
        and 0 < n_local_heads < padded_heads
        and s_q >= fewhead_min_sq()
    )


def _import_kernel():
    from vllm.models.deepseek_v41.nvidia.triton_fewhead_sparse_prefill import (
        fewhead_sparse_mla_fwd,
    )

    return fewhead_sparse_mla_fwd


def load_fewhead_kernel():
    """Load the few-head kernel once per process.

    Returns:
        The ``fewhead_sparse_mla_fwd`` callable.

    """
    global _FWD
    if _FWD is None:
        _FWD = _import_kernel()
    return _FWD


def run_fewhead_sparse_prefill(
    q_chunk: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
    out: torch.Tensor,
    n_local_heads: int,
) -> torch.Tensor:
    """Run native-head sparse prefill into the first ``n_local_heads`` slots.

    Args:
        q_chunk: Padded query ``[s_q, padded_heads, d]``.
        kv: Gathered KV ``[s_kv, 1, d]`` or ``[s_kv, d]``.
        indices: Combined top-k / SWA indices.
        sm_scale: Softmax scale.
        attn_sink: Padded sink logits.
        topk_length: Valid index prefix per token.
        out: Padded output buffer, same shape as ``q_chunk``.
        n_local_heads: Unpadded local Q heads.

    Returns:
        The ``out`` buffer.

    """
    fwd = load_fewhead_kernel()
    fwd(
        q_chunk[:, :n_local_heads],
        kv,
        indices,
        sm_scale,
        attn_sink=attn_sink[:n_local_heads],
        topk_length=topk_length,
        out=out[:, :n_local_heads],
    )
    return out
