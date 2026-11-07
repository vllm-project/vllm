# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 gfx942: main-attention per-head Gemma RMSNorm + partial NeoX
RoPE for q and k in one launch (writes normed/rotated q to ``q_out`` and k in
place of a [tokens, KVH, D] output). Same op order / roundings as
``MiniMAXGemmaRMSNorm`` + ``RotaryEmbedding.forward_native``.
"""

from vllm.models.minimax_m3.amd.ops.index_norm_rope_gfx942 import (
    _cos_sin_as,
    _gemma_norm,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _main_norm_rope_kernel(
    qkv_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_qkv,
    k_start,
    stride_cs,
    eps,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    is_k = h >= HQ
    hh = tl.where(is_k, h - HQ, h)
    src = qkv_ptr + tok * stride_qkv + tl.where(is_k, k_start, 0) + hh * HEAD_DIM
    w_ptr = tl.where(is_k, k_w_ptr, q_w_ptr)
    dst = tl.where(
        is_k,
        k_out_ptr + (tok * HK + hh) * HEAD_DIM,
        q_out_ptr + (tok * HQ + hh) * HEAD_DIM,
    )
    dtype = q_out_ptr.dtype.element_ty
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


def main_norm_rope_gfx942(
    qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, *,
    num_heads, num_kv_heads, head_dim, rotary_dim, eps,
):  # fmt: skip
    """Normed + rotated main q ``[tokens, HQ*D]`` and k ``[tokens, HK, D]``."""
    tokens = qkv.shape[0]
    cos_sin = _cos_sin_as(cos_sin_cache, qkv.dtype)
    q = qkv.new_empty((tokens, num_heads * head_dim))
    k = qkv.new_empty((tokens, num_kv_heads, head_dim))
    if tokens:
        _main_norm_rope_kernel[(tokens, num_heads + num_kv_heads)](
            qkv, q_norm_weight, k_norm_weight, cos_sin, positions, q, k,
            qkv.stride(0), num_heads * head_dim, cos_sin.stride(0), eps,
            HQ=num_heads, HK=num_kv_heads, HEAD_DIM=head_dim, ROT=rotary_dim,
            # gemma_rmsnorm reduces a 128-wide row with 4 warps; same here so
            # the fp32 sum (and hence every rounded output) is bit-identical.
            num_warps=4,
        )  # fmt: skip
    return q, k
