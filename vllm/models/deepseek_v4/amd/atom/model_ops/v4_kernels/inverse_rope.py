# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused inverse RoPE Triton kernel for V4 attention output.

Replaces the two-step `freqs_for_positions` + `_apply_rotary_emb(inverse=True)`
with a single kernel that indexes the cos/sin cache by positions and does the
inverse rotation in-place. GPT-J (interleaved) style only.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_rope_gptj_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    stride_x_s,
    stride_x_h,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    S,
    BLOCK_S: tl.constexpr,
    BLOCK_RD: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_RD)
    s_mask = s_offs < S

    pos = tl.load(pos_ptr + s_offs, mask=s_mask)

    # GPT-J with reuse_freqs_front_part: cos/sin have rd//2 entries,
    # each used for a pair of adjacent elements → index = d_offs // 2
    d_cos_offs = d_offs // 2
    cos_offs = pos[:, None] * stride_cos_s + d_cos_offs[None, :] * stride_cos_d
    cos_mask = s_mask[:, None] & (d_cos_offs < BLOCK_RD_HALF)[None, :]
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    x_offs = (
        s_offs[:, None] * stride_x_s + pid_h * stride_x_h + d_offs[None, :] * stride_x_d
    )
    x_mask = s_mask[:, None] & (d_offs < BLOCK_RD)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # GPT-J inverse: swap pairs and negate evens of (x * sin), then add x * cos.
    # Forward: out[2i]   =  x[2i]*cos - x[2i+1]*sin
    #          out[2i+1] =  x[2i]*sin + x[2i+1]*cos
    # Inverse: out[2i]   =  x[2i]*cos + x[2i+1]*sin
    #          out[2i+1] = -x[2i]*sin + x[2i+1]*cos
    x_sin = x * sin
    even_mask = (d_offs % 2 == 0)[None, :]
    x_negated = tl.where(even_mask, -x_sin, x_sin)
    x_negated = tl.reshape(x_negated, (BLOCK_S, BLOCK_RD_HALF, 2))
    x_negated = tl.flip(x_negated, 2)
    x_rotated = tl.reshape(x_negated, (BLOCK_S, BLOCK_RD))

    out = x * cos + x_rotated
    out = out.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + x_offs, out, mask=x_mask)


def inverse_rope_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """In-place inverse RoPE (GPT-J style) on the rope slice of attention output.

    Args:
        x: ``[num_tokens, n_heads, rotary_dim]`` bf16 — typically ``o[..., -rd:]``.
        cos: ``[max_seq_len, 1, 1, rotary_dim // 2]`` bf16 cache.
        sin: ``[max_seq_len, 1, 1, rotary_dim // 2]`` bf16 cache.
        positions: ``[num_tokens]`` int — per-token position ids.
    """
    S, H, rd = x.shape
    BLOCK_RD = triton.next_power_of_2(rd)
    BLOCK_S = 32
    grid = (H, triton.cdiv(S, BLOCK_S))
    _inverse_rope_gptj_kernel[grid](
        x,
        cos,
        sin,
        positions,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(3),
        S,
        BLOCK_S=BLOCK_S,
        BLOCK_RD=BLOCK_RD,
        BLOCK_RD_HALF=BLOCK_RD // 2,
        num_warps=4,
    )
