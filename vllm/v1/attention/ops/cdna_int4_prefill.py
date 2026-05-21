# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CDNA HIP INT4 per-token-head paged prefill attention kernel wrapper.

The cache layout is packed uint8 nibbles. The fp32 scale word is
steganographed: the low 4 bits hold the unsigned zero-point, the upper
28 bits hold the scale (with its low 4 mantissa bits zeroed). At load
time the kernel subtracts the zp from each nibble (giving a signed int in
[-15, 15]) so the inner MFMA loop is correction-free.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

_available: bool | None = None


def is_available() -> bool:
    global _available
    if _available is not None:
        return _available

    if not current_platform.is_rocm():
        _available = False
        return False

    from vllm.platforms.rocm import on_mi3xx
    if not on_mi3xx():
        _available = False
        return False

    _available = (
        hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "paged_prefill_attn_cdna_int4")
    )
    return _available


def pack_scale_zp(scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    """Steganograph (scale, zp) into a single fp32.

    Low 4 bits of the resulting fp32 carry the unsigned zero-point in [0..15].
    The remaining 28 bits carry the scale (with its 4 lowest mantissa bits
    forced to zero so the unpacked scale is bit-identical at load time).
    """
    assert scale.dtype == torch.float32 and zp.dtype == torch.int32
    bits = scale.contiguous().view(torch.int32)
    bits = (bits & ~0xF) | (zp & 0xF)
    return bits.view(torch.float32)


def cdna_int4_paged_prefill(
    out: torch.Tensor,
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    k_cache: torch.Tensor,      # uint8, [blocks, slots, kv_heads, head_size/2]
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,  # fp32 steganographed [blocks, slots, kv_heads]
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
    causal: bool = True,
) -> None:
    torch.ops._C.paged_prefill_attn_cdna_int4(
        out, q, k_chunk, v_chunk,
        k_cache, v_cache, k_scale_cache, v_scale_cache,
        block_table, cu_seqlens_q, seq_lens,
        max_query_len, sm_scale, causal,
    )
