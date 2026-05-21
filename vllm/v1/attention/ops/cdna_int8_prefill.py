# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CDNA HIP INT8 per-token-head paged prefill attention kernel wrapper.

Targets AMD CDNA (gfx942 / gfx950 / gfx90a). The compiled extension is
registered as ``torch.ops._C.paged_prefill_attn_cdna_int8`` when vLLM is
built with one of those architectures in ``PYTORCH_ROCM_ARCH``.

Phase 1 of the port from JartX's RDNA3 branch — see
``CDNA_INT8_INT4_PORT_PLAN.md`` at the repo root for the full plan.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

_available: bool | None = None


def is_available() -> bool:
    """True iff the CDNA INT8 prefill op is registered and runnable here."""
    global _available
    if _available is not None:
        return _available

    if not current_platform.is_rocm():
        _available = False
        return False

    # The platform helper exists on ROCm builds; import lazily so that
    # importing this module on non-ROCm targets is harmless.
    from vllm.platforms.rocm import on_mi3xx
    if not on_mi3xx():
        _available = False
        return False

    _available = (
        hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "paged_prefill_attn_cdna_int8")
    )
    return _available


def cdna_int8_paged_prefill(
    out: torch.Tensor,
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    k_cache: torch.Tensor,      # int8, [blocks, slots, kv_heads, head_size]
    v_cache: torch.Tensor,      # int8, same layout
    k_scale_cache: torch.Tensor,  # fp32, [blocks, slots, kv_heads]
    v_scale_cache: torch.Tensor,  # fp32, same layout
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
    causal: bool = True,
) -> None:
    """Run the CDNA INT8 per-token-head paged prefill attention in-place.

    Writes the attention output for the current chunk into ``out``.
    """
    torch.ops._C.paged_prefill_attn_cdna_int8(
        out, q, k_chunk, v_chunk,
        k_cache, v_cache, k_scale_cache, v_scale_cache,
        block_table, cu_seqlens_q, seq_lens,
        max_query_len, sm_scale, causal,
    )
