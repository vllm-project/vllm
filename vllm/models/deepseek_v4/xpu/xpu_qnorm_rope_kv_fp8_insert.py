# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU Triton replacement for fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

Does: Q per-head RMSNorm + GPT-J RoPE, KV GPT-J RoPE + UE8M0 FP8 quant + insert.
Uses the existing quantize_and_insert_k_cache for the FP8 portion.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2


@triton.jit
def _xpu_qnorm_rope_kernel(
    q_ptr,  # [num_tokens, num_heads, HEAD_DIM]
    kv_ptr,  # [num_tokens, HEAD_DIM]
    kv_out_ptr,  # [num_tokens, HEAD_DIM] bf16 (RoPE-applied kv for cache insert)
    position_ids_ptr,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Apply per-head RMSNorm + GPT-J RoPE on Q, GPT-J RoPE on KV.

    GPT-J interleaved format: pairs are (data[2i], data[2i+1]).
    cos_sin_cache layout: [max_pos, ROPE_DIM] with first HALF_ROPE=cos,
    second HALF_ROPE=sin.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)

    # Load cos/sin for this position
    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    if head_idx < num_heads:
        # ========== Q: per-head RMSNorm + GPT-J RoPE ==========
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM

        # Load full head
        offs = tl.arange(0, HEAD_DIM)
        q_vals = tl.load(q_base + offs).to(tl.float32)

        # RMSNorm (no weight)
        sq_sum = tl.sum(q_vals * q_vals, axis=0)
        rms = tl.rsqrt(sq_sum / HEAD_DIM + eps)
        q_vals = q_vals * rms

        # Store ONLY the NoPE portion (positions 0..NOPE_DIM-1)
        nope_mask = offs < NOPE_DIM
        tl.store(q_base + offs, q_vals.to(q_ptr.type.element_ty), mask=nope_mask)

        # GPT-J interleaved RoPE on the last ROPE_DIM dimensions:
        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

        # Re-load original values at rope positions and normalize
        q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
        q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms

        new_even = q_even * cos_val - q_odd * sin_val
        new_odd = q_even * sin_val + q_odd * cos_val

        # Store rotated RoPE values
        tl.store(q_base + even_offs, new_even.to(q_ptr.type.element_ty))
        tl.store(q_base + odd_offs, new_odd.to(q_ptr.type.element_ty))
    else:
        # ========== KV: GPT-J RoPE only ==========
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM

        # Copy full KV unchanged first
        offs = tl.arange(0, HEAD_DIM)
        kv_full = tl.load(kv_base + offs)
        tl.store(kv_out_base + offs, kv_full)

        # GPT-J interleaved RoPE on the last ROPE_DIM dimensions
        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

        kv_even = tl.load(kv_base + even_offs).to(tl.float32)
        kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)

        new_even = kv_even * cos_val - kv_odd * sin_val
        new_odd = kv_even * sin_val + kv_odd * cos_val

        tl.store(kv_out_base + even_offs, new_even.to(kv_out_ptr.type.element_ty))
        tl.store(kv_out_base + odd_offs, new_odd.to(kv_out_ptr.type.element_ty))


def xpu_qnorm_rope_kv_fp8_insert(
    q: torch.Tensor,  # [num_tokens, num_heads, HEAD_DIM] bf16, in-place
    kv: torch.Tensor,  # [num_tokens, HEAD_DIM] bf16
    swa_kv_cache: torch.Tensor,  # [num_blocks, block_size, 584] or flat uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, ROPE_DIM]
    eps: float,
    block_size: int,
):
    """XPU Triton: qnorm+rope on Q, rope on KV, then FP8 UE8M0 quant+insert."""
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens = q.shape[0]
    num_heads = q.shape[1]

    # Allocate temp buffer for RoPE-applied KV
    kv_roped = torch.empty_like(kv)

    # Grid: one program per (token, head_or_kv)
    # head_idx < num_heads: process Q head
    # head_idx == num_heads: process KV
    grid = (num_tokens, num_heads + 1)
    _xpu_qnorm_rope_kernel[grid](
        q,
        kv,
        kv_roped,
        positions,
        cos_sin_cache,
        eps,
        num_tokens,
        num_heads=num_heads,
        HEAD_DIM=HEAD_DIM,
        ROPE_DIM=ROPE_DIM,
        NOPE_DIM=NOPE_DIM,
        HALF_ROPE=HALF_ROPE,
    )

    # FP8 UE8M0 quant + paged insert (reuse existing Triton kernel)
    # swa_kv_cache may be [num_blocks, block_size, 584] or [num_blocks, flat]
    # quantize_and_insert_k_cache expects [num_blocks, block_bytes] uint8
    cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
    quantize_and_insert_k_cache(
        kv_roped,
        cache_2d,
        slot_mapping,
        block_size=block_size,
    )
