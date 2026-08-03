# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inverse GPT-J RoPE for the DeepSeek-V4 output projection.

The CUDA backends fuse inverse RoPE into an FP8 quantizing kernel
(``fused_inv_rope_fp8_quant``) that feeds DeepGEMM's FP8 einsum. Turing has no
FP8 tensor cores, so this plain FP16 inverse-RoPE kernel inverts the RoPE that
``turing/kv_insert.py`` applied to the rope tail of each head.
"""

import torch

from vllm.models.deepseek_v4.turing.constants import NOPE_DIM, ROPE_DIM
from vllm.triton_utils import tl, triton


@triton.jit
def _inverse_rope_kernel(
    o_ptr,  # [num_tokens, num_heads, HEAD_DIM] fp16, in place
    position_ids_ptr,  # [num_tokens] int64
    cos_sin_cache_ptr,  # [max_pos, ROPE_DIM] fp32
    num_tokens,
    o_stride_token,
    o_stride_head,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)

    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    o_base = o_ptr + token_idx * o_stride_token + head_idx * o_stride_head
    even_offs = NOPE_DIM + rope_pair_idx * 2
    odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1
    even = tl.load(o_base + even_offs).to(tl.float32)
    odd = tl.load(o_base + odd_offs).to(tl.float32)

    # Forward RoPE: e' = e*cos - o*sin, o' = e*sin + o*cos.
    # Inverse: e = e'*cos + o'*sin, o = o'*cos - e'*sin.
    inv_even = even * cos_val + odd * sin_val
    inv_odd = odd * cos_val - even * sin_val
    tl.store(o_base + even_offs, inv_even.to(o_ptr.type.element_ty))
    tl.store(o_base + odd_offs, inv_odd.to(o_ptr.type.element_ty))


def inverse_rope(
    o: torch.Tensor,  # [num_tokens, num_heads, head_dim] fp16, in place
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, ROPE_DIM] fp32
    nope_dim: int = NOPE_DIM,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    """Invert the GPT-J RoPE on the rope tail of each head, in place."""
    num_tokens, num_heads, head_dim = o.shape
    assert head_dim == nope_dim + rope_dim
    _inverse_rope_kernel[(num_tokens, num_heads)](
        o,
        positions,
        cos_sin_cache,
        num_tokens,
        o.stride(0),
        o.stride(1),
        ROPE_DIM=rope_dim,
        NOPE_DIM=nope_dim,
        HALF_ROPE=rope_dim // 2,
    )
    return o
