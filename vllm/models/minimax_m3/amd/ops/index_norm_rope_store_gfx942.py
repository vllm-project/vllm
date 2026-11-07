# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 gfx942: index-branch Gemma RMSNorm + partial NeoX RoPE with the
MXFP4 index-K cache insert fused in (one launch instead of three:
norm/rope, ``dynamic_mxfp4_quant``, page store).

Q is bit-identical to ``index_norm_rope_gfx942``; the stored K bytes are
bit-identical to ``store_k_gfx942`` applied to that function's K.
"""

import torch

from vllm.models.minimax_m3.amd.ops.index_norm_rope_gfx942 import (
    _cos_sin_as,
    _gemma_norm,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import (
    K_CHUNK_BYTES,
    KERNEL_PAGE_TOKENS,
    MANAGER_BLOCK_BYTES,
    MANAGER_BLOCK_TOKENS,
    PAGE_DATA_BYTES,
    PAGE_SLOT_BYTES,
    _check_cache,
    store_k_row_gfx942,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _index_norm_rope_store_kernel(
    qkv_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_out_ptr,
    cache_ptr,
    slots_ptr,
    stride_qkv,
    q_start,
    k_start,
    stride_cs,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    PAGE_DATA: tl.constexpr,
    PAGE_SLOT: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dtype = q_out_ptr.dtype.element_ty
    half: tl.constexpr = ROT // 2
    pos = tl.load(pos_ptr + tok).to(tl.int64)
    if head < HEADS:
        src = qkv_ptr + tok * stride_qkv + q_start + head * HEAD_DIM
        x = tl.load(src + tl.arange(0, HEAD_DIM)).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        r = tl.arange(0, half)
        p = tl.arange(0, HEAD_DIM - ROT)
        y1 = _gemma_norm(src + r, q_w_ptr + r, rstd, dtype)
        y2 = _gemma_norm(src + half + r, q_w_ptr + half + r, rstd, dtype)
        yp = _gemma_norm(src + ROT + p, q_w_ptr + ROT + p, rstd, dtype)
        cos = tl.load(cos_sin_ptr + pos * stride_cs + r).to(dtype)
        sin = tl.load(cos_sin_ptr + pos * stride_cs + half + r).to(dtype)
        o1 = ((y1 * cos).to(dtype) - (y2 * sin).to(dtype)).to(dtype)
        o2 = ((y2 * cos).to(dtype) + (y1 * sin).to(dtype)).to(dtype)
        dst = q_out_ptr + (tok * HEADS + head) * HEAD_DIM
        tl.store(dst + r, o1)
        tl.store(dst + half + r, o2)
        tl.store(dst + ROT + p, yp)
    else:
        ksrc = qkv_ptr + tok * stride_qkv + k_start
        d = tl.arange(0, HEAD_DIM)
        kx = tl.load(ksrc + d).to(tl.float32)
        krstd = 1.0 / tl.sqrt(tl.sum(kx * kx, axis=0) / HEAD_DIM + eps)
        # Element d and its rotation partner, each normed exactly as the
        # split-half code rounds them.
        partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
        ky = _gemma_norm(ksrc + d, k_w_ptr + d, krstd, dtype)
        kyo = _gemma_norm(ksrc + partner, k_w_ptr + partner, krstd, dtype)
        ci = tl.where(d < ROT, d % half, 0)
        kcos = tl.load(cos_sin_ptr + pos * stride_cs + ci).to(dtype)
        ksin = tl.load(cos_sin_ptr + pos * stride_cs + half + ci).to(dtype)
        klo = ((ky * kcos).to(dtype) - (kyo * ksin).to(dtype)).to(dtype)
        khi = ((ky * kcos).to(dtype) + (kyo * ksin).to(dtype)).to(dtype)
        k = tl.where(d < half, klo, tl.where(d < ROT, khi, ky))
        slot = tl.load(slots_ptr + tok)
        store_k_row_gfx942(
            k.to(tl.float32),
            cache_ptr,
            slot,
            HEAD_DIM,
            BLOCK_TOKENS,
            PAGE_TOKENS,
            CHUNK_BYTES,
            PAGE_DATA,
            PAGE_SLOT,
            BLOCK_BYTES,
        )


def index_norm_rope_store_gfx942(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_cache: torch.Tensor | None,
    index_slot_mapping: torch.Tensor | None,
    *,
    q_start: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> torch.Tensor:
    """Normed and rotated index Q ``[tokens, num_heads * head_dim]``; index K
    is normed, rotated, MXFP4-quantized and written into ``index_cache``."""
    if qkv.stride(-1) != 1:
        raise ValueError("qkv must be row-contiguous")
    tokens = qkv.shape[0]
    cos_sin = _cos_sin_as(cos_sin_cache, qkv.dtype)
    q_out = qkv.new_empty((tokens, num_heads * head_dim))
    store = index_cache is not None and index_cache.numel() != 0
    cache = _check_cache(index_cache) if store else q_out
    if tokens:
        _index_norm_rope_store_kernel[(tokens, num_heads + (1 if store else 0))](
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin,
            positions,
            q_out,
            cache,
            index_slot_mapping if store else positions,
            qkv.stride(0),
            q_start,
            q_start + num_heads * head_dim,
            cos_sin.stride(0),
            eps,
            HEADS=num_heads,
            HEAD_DIM=head_dim,
            ROT=rotary_dim,
            BLOCK_TOKENS=MANAGER_BLOCK_TOKENS,
            PAGE_TOKENS=KERNEL_PAGE_TOKENS,
            CHUNK_BYTES=K_CHUNK_BYTES,
            PAGE_DATA=PAGE_DATA_BYTES,
            PAGE_SLOT=PAGE_SLOT_BYTES,
            BLOCK_BYTES=MANAGER_BLOCK_BYTES,
            num_warps=4,  # = gemma_rmsnorm reduction order (bit-exact)
        )
    return q_out
