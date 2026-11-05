# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 gfx942, MXFP4 K/V cache + MXFP4 index cache: the whole per-layer
pre-attention write path in ONE launch.

Programs ``(tok, j)``:
- ``j <= HEADS``: index Q head ``j`` / index K (norm + RoPE; K into its MXFP4
  index page) -- ``_index_norm_rope_store_kernel``;
- next ``HQ``: main Q head (per-head Gemma RMSNorm + partial NeoX RoPE) -> q;
- next ``HK``: main K head (norm + RoPE) quantized into its MXFP4 K/V page;
- next ``HK``: main V head quantized into its MXFP4 K/V page.

Every value is bit-identical to the eager ``MiniMAXGemmaRMSNorm`` +
``RotaryEmbedding.forward_native`` + ``store_kv_mxfp4_gfx942`` /
``store_k_gfx942`` path (same op order; 4-warp reductions like
``gemma_rmsnorm``).
"""

import torch

from vllm.models.minimax_m3.amd.ops.index_norm_rope_gfx942 import (
    _cos_sin_as,
    _gemma_norm,
)
from vllm.models.minimax_m3.amd.ops.index_norm_rope_store_gfx942 import (
    _index_norm_rope_store_kernel,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import _e2m1_rne
from vllm.models.minimax_m3.amd.ops.mxfp4_gfx942 import (
    K_CHUNK_BYTES,
    KERNEL_PAGE_TOKENS,
    MANAGER_BLOCK_BYTES,
    MANAGER_BLOCK_TOKENS,
    PAGE_DATA_BYTES,
    PAGE_SLOT_BYTES,
    _check_cache,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import HEAD_DIM, PAGE_BYTES, PAGE_TOKENS
from vllm.triton_utils import tl, triton


@triton.jit
def _store_kv_values(
    x,  # [D] fp32
    cache_ptr,
    slot,
    side,
    stride_cache_block,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
):
    # Same arithmetic as mxfp4_attn_gfx942._store_kv_mxfp4_kernel.
    x = tl.reshape(x, (D // 32, 32))
    amax = tl.max(tl.abs(x), axis=1)
    ratio = tl.maximum(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
    e8m0 = (ratio.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23
    inv = ((254 - e8m0) << 23).to(tl.float32, bitcast=True)
    y = x * inv[:, None]
    code = _e2m1_rne(tl.abs(y)) | tl.where(y < 0, 8, 0)
    code = tl.reshape(code, (D // 2, 2))
    lo, hi = tl.split(code)
    packed = (lo | (hi << 4)).to(tl.uint8)
    block = slot // BLOCK_TOKENS
    t = slot % BLOCK_TOKENS
    page = cache_ptr + block * stride_cache_block + side * PAGE
    tl.store(page + t * (D // 2) + tl.arange(0, D // 2), packed)
    tl.store(
        page + BLOCK_TOKENS * (D // 2) + t * (D // 32) + tl.arange(0, D // 32),
        e8m0.to(tl.uint8),
    )


@triton.jit
def _norm_rope_full(src, w_ptr, cos_sin_ptr, pos, stride_cs, eps, dtype: tl.constexpr,
                    HEAD_DIM: tl.constexpr, ROT: tl.constexpr):  # fmt: skip
    """All HEAD_DIM outputs of one head as a [HEAD_DIM] vector, each element
    rounded exactly as the split-half eager code rounds it."""
    half: tl.constexpr = ROT // 2
    d = tl.arange(0, HEAD_DIM)
    x = tl.load(src + d).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
    y = _gemma_norm(src + d, w_ptr + d, rstd, dtype)
    yo = _gemma_norm(src + partner, w_ptr + partner, rstd, dtype)
    ci = tl.where(d < ROT, d % half, 0)
    cos = tl.load(cos_sin_ptr + pos * stride_cs + ci).to(dtype)
    sin = tl.load(cos_sin_ptr + pos * stride_cs + half + ci).to(dtype)
    lo = ((y * cos).to(dtype) - (yo * sin).to(dtype)).to(dtype)
    hi = ((y * cos).to(dtype) + (yo * sin).to(dtype)).to(dtype)
    return tl.where(d < half, lo, tl.where(d < ROT, hi, y))


@triton.jit
def _qkv_index_store_kernel(
    qkv_ptr,
    iq_w_ptr,
    ik_w_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_sin_ptr,
    pos_ptr,
    iq_out_ptr,
    q_out_ptr,
    icache_ptr,
    islots_ptr,
    kv_cache_ptr,
    kv_slots_ptr,
    stride_qkv,
    iq_start,
    ik_start,
    k_start,
    v_start,
    stride_cs,
    stride_kv_block,
    eps,
    HEADS: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    PAGE_DATA: tl.constexpr,
    PAGE_SLOT: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    KV_BLOCK_TOKENS: tl.constexpr,
    KV_PAGE: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    dtype = q_out_ptr.dtype.element_ty
    if j <= HEADS:
        _index_norm_rope_store_kernel(
            qkv_ptr, iq_w_ptr, ik_w_ptr, cos_sin_ptr, pos_ptr, iq_out_ptr, icache_ptr,
            islots_ptr, stride_qkv, iq_start, ik_start, stride_cs, eps, HEADS,
            HEAD_DIM, ROT, BLOCK_TOKENS, PAGE_TOKENS, CHUNK_BYTES, PAGE_DATA,
            PAGE_SLOT, BLOCK_BYTES,
        )  # fmt: skip
    else:
        m = j - HEADS - 1
        pos = tl.load(pos_ptr + tok).to(tl.int64)
        if m < HQ:
            src = qkv_ptr + tok * stride_qkv + m * HEAD_DIM
            half: tl.constexpr = ROT // 2
            r = tl.arange(0, half)
            p = tl.arange(0, HEAD_DIM - ROT)
            x = tl.load(src + tl.arange(0, HEAD_DIM)).to(tl.float32)
            rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            y1 = _gemma_norm(src + r, q_w_ptr + r, rstd, dtype)
            y2 = _gemma_norm(src + half + r, q_w_ptr + half + r, rstd, dtype)
            yp = _gemma_norm(src + ROT + p, q_w_ptr + ROT + p, rstd, dtype)
            cos = tl.load(cos_sin_ptr + pos * stride_cs + r).to(dtype)
            sin = tl.load(cos_sin_ptr + pos * stride_cs + half + r).to(dtype)
            o1 = ((y1 * cos).to(dtype) - (y2 * sin).to(dtype)).to(dtype)
            o2 = ((y2 * cos).to(dtype) + (y1 * sin).to(dtype)).to(dtype)
            dst = q_out_ptr + (tok * HQ + m) * HEAD_DIM
            tl.store(dst + r, o1)
            tl.store(dst + half + r, o2)
            tl.store(dst + ROT + p, yp)
        else:
            side = m - HQ  # 0..2HK-1: K heads then V heads
            slot = tl.load(kv_slots_ptr + tok).to(tl.int64)
            if slot >= 0:
                if side < HK:
                    src = qkv_ptr + tok * stride_qkv + k_start + side * HEAD_DIM
                    kx = _norm_rope_full(
                        src, k_w_ptr, cos_sin_ptr, pos, stride_cs, eps, dtype,
                        HEAD_DIM, ROT,
                    )  # fmt: skip
                    x = kx.to(tl.float32)
                else:
                    src = qkv_ptr + tok * stride_qkv + v_start + (side - HK) * HEAD_DIM
                    x = tl.load(src + tl.arange(0, HEAD_DIM)).to(tl.float32)
                _store_kv_values(
                    x, kv_cache_ptr, slot, side, stride_kv_block, HEAD_DIM,
                    KV_BLOCK_TOKENS, KV_PAGE,
                )  # fmt: skip


def qkv_index_store_gfx942(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_cache: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q [tokens, HQ*D], index_q [tokens, HEADS*D])``; main K/V and index K
    are written into their MXFP4 caches. ``qkv`` = [q | k | v | index_q | index_k]."""
    if qkv.stride(-1) != 1:
        raise ValueError("qkv must be row-contiguous")
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    tokens = qkv.shape[0]
    cos_sin = _cos_sin_as(cos_sin_cache, qkv.dtype)
    q = qkv.new_empty((tokens, num_heads * head_dim))
    iq = qkv.new_empty((tokens, num_index_heads * head_dim))
    k_start = num_heads * head_dim
    v_start = k_start + num_kv_heads * head_dim
    iq_start = v_start + num_kv_heads * head_dim
    if tokens:
        _qkv_index_store_kernel[
            (tokens, num_index_heads + 1 + num_heads + 2 * num_kv_heads)
        ](
            qkv, index_q_norm_weight, index_k_norm_weight, q_norm_weight,
            k_norm_weight, cos_sin, positions, iq, q, _check_cache(index_cache),
            index_slot_mapping, kv_cache, kv_slot_mapping, qkv.stride(0), iq_start,
            iq_start + num_index_heads * head_dim, k_start, v_start,
            cos_sin.stride(0), kv_cache.stride(0), eps,
            HEADS=num_index_heads, HQ=num_heads, HK=num_kv_heads, HEAD_DIM=head_dim,
            ROT=rotary_dim, BLOCK_TOKENS=MANAGER_BLOCK_TOKENS,
            PAGE_TOKENS=KERNEL_PAGE_TOKENS, CHUNK_BYTES=K_CHUNK_BYTES,
            PAGE_DATA=PAGE_DATA_BYTES, PAGE_SLOT=PAGE_SLOT_BYTES,
            BLOCK_BYTES=MANAGER_BLOCK_BYTES, KV_BLOCK_TOKENS=PAGE_TOKENS,
            KV_PAGE=PAGE_BYTES,
            num_warps=4,  # = gemma_rmsnorm reduction order (bit-exact)
        )  # fmt: skip
    return q, iq
