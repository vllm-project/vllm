# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 index-branch Gemma RMSNorm + partial NeoX RoPE in one launch.

Bit-exact with ``MiniMAXGemmaRMSNorm`` followed by
``RotaryEmbedding.forward_native``: the norm rounds once to the activation
dtype, and each rotation product and sum rounds to it as the eager ops do.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemma_norm(x_ptr, w_ptr, rstd, dtype: tl.constexpr):
    # Same operation order as gemma_rmsnorm, so the rounding matches.
    x = tl.load(x_ptr).to(tl.float32)
    return (x * rstd * (1.0 + tl.load(w_ptr).to(tl.float32))).to(dtype)


@triton.jit
def _index_norm_rope_kernel(
    qkv_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_qkv,
    q_start,
    k_start,
    stride_cs,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    is_k = head == HEADS
    src = (
        qkv_ptr + tok * stride_qkv + tl.where(is_k, k_start, q_start + head * HEAD_DIM)
    )
    w_ptr = tl.where(is_k, k_w_ptr, q_w_ptr)
    dst = tl.where(
        is_k, k_out_ptr + tok * HEAD_DIM, q_out_ptr + (tok * HEADS + head) * HEAD_DIM
    )
    dtype = dst.dtype.element_ty

    x = tl.load(src + tl.arange(0, HEAD_DIM)).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)

    half: tl.constexpr = ROT // 2
    r = tl.arange(0, half)
    p = tl.arange(0, HEAD_DIM - ROT)
    y1 = _gemma_norm(src + r, w_ptr + r, rstd, dtype)
    y2 = _gemma_norm(src + half + r, w_ptr + half + r, rstd, dtype)
    yp = _gemma_norm(src + ROT + p, w_ptr + ROT + p, rstd, dtype)

    pos = tl.load(pos_ptr + tok).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * stride_cs + r).to(dtype)
    sin = tl.load(cos_sin_ptr + pos * stride_cs + half + r).to(dtype)
    o1 = ((y1 * cos).to(dtype) - (y2 * sin).to(dtype)).to(dtype)
    o2 = ((y2 * cos).to(dtype) + (y1 * sin).to(dtype)).to(dtype)
    tl.store(dst + r, o1)
    tl.store(dst + half + r, o2)
    tl.store(dst + ROT + p, yp)


_cos_sin_cast: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}


def _cos_sin_as(cache: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The rope cache in ``dtype``, converted once rather than per call."""
    if cache.dtype == dtype:
        return cache
    key = (cache.data_ptr(), cache._version, dtype)
    out = _cos_sin_cast.get(key)
    if out is None:
        _cos_sin_cast.clear()
        out = _cos_sin_cast[key] = cache.to(dtype)
    return out


def index_norm_rope_gfx942(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    q_start: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normed and rotated index Q ``[tokens, num_heads * head_dim]`` and K
    ``[tokens, head_dim]``, read from the fused ``qkv`` at ``q_start``."""
    if qkv.stride(-1) != 1:
        raise ValueError("qkv must be row-contiguous")
    tokens = qkv.shape[0]
    cos_sin = _cos_sin_as(cos_sin_cache, qkv.dtype)
    q_out = qkv.new_empty((tokens, num_heads * head_dim))
    k_out = qkv.new_empty((tokens, head_dim))
    if tokens:
        _index_norm_rope_kernel[(tokens, num_heads + 1)](
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin,
            positions,
            q_out,
            k_out,
            qkv.stride(0),
            q_start,
            q_start + num_heads * head_dim,
            cos_sin.stride(0),
            eps,
            HEADS=num_heads,
            HEAD_DIM=head_dim,
            ROT=rotary_dim,
            num_warps=4,  # = gemma_rmsnorm reduction order (bit-exact)
        )
    return q_out, k_out
