# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer RoPE and FP8 preparation with HY4's trailing rotary dimensions."""

import torch

from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


# One program per (token, head). Applies RoPE to the last 64 of 128 dims, computes
# one UE8M0 scale per head from the rotated and unrotated dims, stores the FP8 head,
# and writes weights[token, head] * scale * softmax_scale * head_scale so the logits
# kernel receives the dequantization and softmax scaling folded into its weights.
@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout in the last 64 dimensions.
        x0 = tl.load(q_base + 64 + offs32).to(tl.float32)
        x1 = tl.load(q_base + 96 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # Even/odd pairs in the last 64 dimensions.
        x0 = tl.load(q_base + 64 + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + 64 + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + 64 + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 96 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + 64 + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 64 + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare the indexer query and head weights for the FP8 MQA logits kernel.

    Args:
        positions: `(tokens,)` positions for the RoPE cache lookup.
        q: `(tokens, heads, 128)` BF16 indexer queries with the 64 rotary dims
            last.
        cos_sin_cache: `(max_pos, 64)` cache holding 32 cosines then 32 sines.
        weights: `(tokens, heads)` per-head indexer weights.
        softmax_scale: Attention scale folded into the output weights.
        head_scale: Head-count normalization folded into the output weights.
        is_neox: Rotary pair layout; NeoX pairs dim `i` with `i + 32`,
            otherwise adjacent dims are paired.

    Returns:
        `(q_fp8, weights_out)`: the rotated query in FP8 with one UE8M0 scale per
        token and head, and fp32 weights with that scale, `softmax_scale` and
        `head_scale` multiplied in.

    """
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return q_fp8, weights_out


# Four rows per program. LayerNorm (weight and bias) over the 128-dim key, then
# RoPE on the last 64 dims, with the partner index and sign chosen by the pair
# layout. Dims below 64 pass through unrotated.
@triton.jit
def _k_norm_rope(
    K,
    W,
    B,
    POS,
    CACHE,
    OUT,
    M,
    KS: tl.constexpr,
    CS: tl.constexpr,
    EPS: tl.constexpr,
    NEOX: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, 128)
    x = tl.load(K + rows[:, None] * KS + cols[None, :], rows[:, None] < M, 0).to(
        tl.float32
    )
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    mean = tl.sum(x, 1) / 128
    centered = x - mean[:, None]
    var = tl.sum(centered * centered, 1) / 128
    y = centered * tl.rsqrt(var + EPS)[:, None] * w[None, :] + b[None, :]
    y = y.to(K.dtype.element_ty).to(tl.float32)
    if NEOX:
        partner = tl.where(cols < 96, cols + 32, cols - 32)
        pair = (cols - 64) % 32
        negative = cols < 96
    else:
        partner = cols ^ 1
        pair = (cols - 64) // 2
        negative = cols % 2 == 0
    other = tl.gather(
        y, tl.broadcast_to(tl.minimum(partner, 127)[None, :], (4, 128)), 1
    )
    pos = tl.load(POS + rows, rows < M, 0)
    cos = tl.load(
        CACHE + pos[:, None] * CS + pair[None, :],
        (rows[:, None] < M) & (cols[None, :] >= 64),
        0,
    ).to(tl.float32)
    sin = tl.load(
        CACHE + pos[:, None] * CS + 32 + pair[None, :],
        (rows[:, None] < M) & (cols[None, :] >= 64),
        0,
    ).to(tl.float32)
    rotated = y * cos + tl.where(negative[None, :], -other, other) * sin
    out = tl.where(cols[None, :] < 64, y, rotated)
    tl.store(OUT + rows[:, None] * 128 + cols[None, :], out, rows[:, None] < M)


def fused_indexer_k_norm_rope(
    k: torch.Tensor,
    norm: LayerNorm,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Apply the indexer key LayerNorm and RoPE in one pass.

    Args:
        k: `(tokens, 128)` BF16 indexer keys with the 64 rotary dims last.
        norm: A `LayerNorm` module with `weight`, `bias` and `eps`.
        positions: `(tokens,)` row of each token in `cos_sin_cache`.
        cos_sin_cache: `(rows, 64)` cache holding 32 cosines then 32 sines.
        is_neox_style: Whether the RoPE pairs are NeoX-style halves.

    Returns:
        `(tokens, 128)` normalized and rotated keys in the dtype of `k`.

    """
    out = torch.empty((k.shape[0], 128), device=k.device, dtype=k.dtype)
    if k.shape[0]:
        _k_norm_rope[(triton.cdiv(k.shape[0], 4),)](
            k,
            norm.weight,
            norm.bias,
            positions,
            cos_sin_cache,
            out,
            k.shape[0],
            k.stride(0),
            cos_sin_cache.stride(0),
            norm.eps,
            is_neox_style,
            enable_fp_fusion=False,
        )
    return out
