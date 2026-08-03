# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton replacement for the sm80+ fused Q-norm/RoPE/KV-insert C++ op.

Port of ``vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py`` for
FP16 plain-row KV caches on SM75: per-head RMSNorm + GPT-J RoPE on Q and GPT-J
RoPE + plain FP16 paged store on KV. The XPU op quantizes to FP8 via
``quantize_and_insert_k_cache``; the Turing cache is plain FP16, so the KV
branch writes the RoPE'd row straight into the paged cache.
"""

import torch

from vllm.models.deepseek_v4.turing.constants import (
    HALF_ROPE,
    HEAD_DIM,
    NOPE_DIM,
    ROPE_DIM,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _qnorm_rope_kv_fp16_insert_kernel(
    q_ptr,  # [num_tokens, num_heads, HEAD_DIM] fp16, in place
    kv_ptr,  # [num_tokens, HEAD_DIM] fp16
    kv_cache_ptr,  # [num_blocks, block_size, HEAD_DIM] fp16
    slot_mapping_ptr,  # [num_tokens] int64
    position_ids_ptr,  # [num_tokens] int64
    cos_sin_cache_ptr,  # [max_pos, ROPE_DIM] fp32
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Per-head RMSNorm + GPT-J RoPE on Q; GPT-J RoPE + paged store on KV.

    GPT-J interleaved format: pairs are (data[2i], data[2i+1]). The
    cos_sin_cache is [max_pos, ROPE_DIM] with the first HALF_ROPE entries cos
    and the last HALF_ROPE entries sin.
    """
    token_idx = tl.program_id(0)
    head_or_kv = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)

    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    offs = tl.arange(0, HEAD_DIM)
    even_offs = NOPE_DIM + rope_pair_idx * 2
    odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

    if head_or_kv < num_heads:
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_or_kv * HEAD_DIM
        q_vals = tl.load(q_base + offs).to(tl.float32)
        sq_sum = tl.sum(q_vals * q_vals, axis=0)
        rms = tl.rsqrt(sq_sum / HEAD_DIM + eps)
        tl.store(
            q_base + offs,
            (q_vals * rms).to(q_ptr.type.element_ty),
            mask=offs < NOPE_DIM,
        )
        q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
        q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms
        tl.store(
            q_base + even_offs,
            (q_even * cos_val - q_odd * sin_val).to(q_ptr.type.element_ty),
        )
        tl.store(
            q_base + odd_offs,
            (q_even * sin_val + q_odd * cos_val).to(q_ptr.type.element_ty),
        )
    else:
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_vals = tl.load(kv_base + offs)
        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot >= 0:
            block_idx = slot // block_size
            pos_in_block = slot % block_size
            cache_row = (
                kv_cache_ptr
                + block_idx.to(tl.int64) * block_size * HEAD_DIM
                + pos_in_block * HEAD_DIM
            )
            tl.store(cache_row + offs, kv_vals, mask=offs < NOPE_DIM)
            kv_even = tl.load(kv_base + even_offs).to(tl.float32)
            kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)
            tl.store(
                cache_row + even_offs,
                (kv_even * cos_val - kv_odd * sin_val).to(kv_cache_ptr.type.element_ty),
            )
            tl.store(
                cache_row + odd_offs,
                (kv_even * sin_val + kv_odd * cos_val).to(kv_cache_ptr.type.element_ty),
            )


def turing_qnorm_rope_kv_fp16_insert(
    q: torch.Tensor,  # [num_tokens, num_heads, HEAD_DIM] fp16, in place
    kv: torch.Tensor,  # [num_tokens, HEAD_DIM] fp16
    swa_kv_cache: torch.Tensor,  # [num_blocks, block_size, HEAD_DIM] fp16
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, ROPE_DIM] fp32
    eps: float,
    block_size: int,
) -> None:
    """Q-norm/RoPE + RoPE/KV plain FP16 paged insert for the Turing backend."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    cache_3d = swa_kv_cache.view(-1, block_size, -1)
    grid = (num_tokens, num_heads + 1)
    _qnorm_rope_kv_fp16_insert_kernel[grid](
        q,
        kv,
        cache_3d,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        num_tokens,
        num_heads=num_heads,
        block_size=block_size,
        HEAD_DIM=HEAD_DIM,
        ROPE_DIM=ROPE_DIM,
        NOPE_DIM=NOPE_DIM,
        HALF_ROPE=HALF_ROPE,
    )
