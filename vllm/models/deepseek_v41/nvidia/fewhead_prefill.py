# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cached loader for DSV4.1 few-head sparse MLA prefill."""

from __future__ import annotations

from typing import Any

import torch

_FWD: Any = None


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
