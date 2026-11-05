# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 gfx942, MXFP4 K/V cache + MXFP4 index cache: one launch for
everything the layer writes after the fused main q/K norm + RoPE.

Programs ``(tok, j)``:
- ``j < HEADS``: index Q head ``j`` (Gemma RMSNorm + partial NeoX RoPE);
- ``j == HEADS``: index K, normed/rotated and stored into its MXFP4 page;
- ``j > HEADS``: main K (``j - HEADS - 1 < KVH``) or V row (read from the
  given normed/rotated ``k`` / raw ``v``), MXFP4-quantized into the main cache.

Bit-identical to ``index_norm_rope_store_gfx942`` + ``store_kv_mxfp4_gfx942``
(same ops, same order); it only removes the second launch.
"""

import torch

from vllm.models.minimax_m3.amd.ops.index_norm_rope_gfx942 import _cos_sin_as
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
def _store_kv_row(
    src,
    cache_ptr,
    slot_ptr,
    tok,
    side,
    stride_cache_block,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
):
    # Same arithmetic as mxfp4_attn_gfx942._store_kv_mxfp4_kernel.
    slot = tl.load(slot_ptr + tok).to(tl.int64)
    if slot >= 0:
        x = tl.load(src + tl.arange(0, D)).to(tl.float32)
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
def _index_kv_store_kernel(
    qkv_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_out_ptr,
    icache_ptr,
    islots_ptr,
    kv_cache_ptr,
    kv_slots_ptr,
    k_ptr,
    v_ptr,
    stride_k_tok,
    stride_k_head,
    stride_v_tok,
    stride_v_head,
    stride_qkv,
    q_start,
    k_start,
    stride_cs,
    stride_kv_block,
    eps,
    HEADS: tl.constexpr,
    KVH: tl.constexpr,
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
    if j <= HEADS:
        _index_norm_rope_store_kernel(
            qkv_ptr, q_w_ptr, k_w_ptr, cos_sin_ptr, pos_ptr, q_out_ptr, icache_ptr,
            islots_ptr, stride_qkv, q_start, k_start, stride_cs, eps, HEADS,
            HEAD_DIM, ROT, BLOCK_TOKENS, PAGE_TOKENS, CHUNK_BYTES, PAGE_DATA,
            PAGE_SLOT, BLOCK_BYTES,
        )  # fmt: skip
    else:
        side = j - HEADS - 1  # 0..2KVH-1: K heads then V heads
        head = side % KVH
        if side < KVH:
            src = k_ptr + tok * stride_k_tok + head * stride_k_head
        else:
            src = v_ptr + tok * stride_v_tok + head * stride_v_head
        _store_kv_row(
            src, kv_cache_ptr, kv_slots_ptr, tok, side, stride_kv_block, HEAD_DIM,
            KV_BLOCK_TOKENS, KV_PAGE,
        )  # fmt: skip


def index_kv_store_gfx942(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_cache: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,  # uint8 [blocks, 2*KVH, 128, 68]
    kv_slot_mapping: torch.Tensor,
    k: torch.Tensor,  # [tokens, KVH, 128] normed + rotated main K (any strides)
    v: torch.Tensor,  # [tokens, KVH, 128]
    *,
    q_start: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> torch.Tensor:
    """Index Q (returned), index K -> MXFP4 index pages, and main K/V (already
    normed/rotated in ``qkv``) -> MXFP4 K/V pages, in one launch."""
    if qkv.stride(-1) != 1:
        raise ValueError("qkv must be row-contiguous")
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("main K/V need a contiguous head dim")
    tokens = qkv.shape[0]
    cos_sin = _cos_sin_as(cos_sin_cache, qkv.dtype)
    q_out = qkv.new_empty((tokens, num_heads * head_dim))
    if tokens:
        _index_kv_store_kernel[(tokens, num_heads + 1 + 2 * num_kv_heads)](
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin,
            positions,
            q_out,
            _check_cache(index_cache),
            index_slot_mapping,
            kv_cache,
            kv_slot_mapping,
            k,
            v,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            qkv.stride(0),
            q_start,
            q_start + num_heads * head_dim,
            cos_sin.stride(0),
            kv_cache.stride(0),
            eps,
            HEADS=num_heads,
            KVH=num_kv_heads,
            HEAD_DIM=head_dim,
            ROT=rotary_dim,
            BLOCK_TOKENS=MANAGER_BLOCK_TOKENS,
            PAGE_TOKENS=KERNEL_PAGE_TOKENS,
            CHUNK_BYTES=K_CHUNK_BYTES,
            PAGE_DATA=PAGE_DATA_BYTES,
            PAGE_SLOT=PAGE_SLOT_BYTES,
            BLOCK_BYTES=MANAGER_BLOCK_BYTES,
            KV_BLOCK_TOKENS=PAGE_TOKENS,
            KV_PAGE=PAGE_BYTES,
            num_warps=4,  # = gemma_rmsnorm reduction order (bit-exact)
        )
    return q_out
