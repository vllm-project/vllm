# SPDX-License-Identifier: Apache-2.0
"""Fused indexer-Q RoPE kernels used by reviewer accuracy/perf scripts."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 128
ROPE_DIM = 64
N_HEAD = 64
MAX_POS = 4096


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_quant_variant_kernel(
    pos_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    USE_FMA: tl.constexpr,
):
    index_q_rot_dim: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    index_q_nope_dim: tl.constexpr = INDEX_Q_HEAD_DIM - index_q_rot_dim
    tl.static_assert(index_q_nope_dim >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    rot_base = base_ptr + index_q_nope_dim
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    if USE_FMA:
        r_even = tl.fma(x_even, cos, -(x_odd * sin))
        r_odd = tl.fma(x_odd, cos, x_even * sin)
    else:
        r_even = x_even * cos - x_odd * sin
        r_odd = x_odd * cos + x_even * sin

    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if index_q_nope_dim > 0:
        nope_offset = tl.arange(0, index_q_nope_dim)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    if index_q_nope_dim > 0:
        tl.store(
            fp8_base_ptr + nope_offset,
            tl.div_rn(x_nope, index_q_scale).to(tl.float8e4nv),
        )
    fp8_rot_base = fp8_base_ptr + index_q_nope_dim
    tl.store(
        fp8_rot_base + half_offset * 2,
        tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv),
    )
    tl.store(
        fp8_rot_base + half_offset * 2 + 1,
        tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv),
    )

    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def launch_variant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    index_q_fp8: torch.Tensor,
    index_weights_out: torch.Tensor,
    *,
    use_fma: bool,
) -> None:
    """Launch one FP8 fused indexer-Q variant into preallocated outputs."""
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2
    assert index_q.shape[-1] == HEAD_DIM
    assert index_q_cos_sin_cache.shape[-1] == ROPE_DIM

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    _fused_indexer_q_rope_quant_variant_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        index_q.shape[-1],
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        USE_FMA=use_fma,
        num_warps=1,
    )


def run_variant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    *,
    use_fma: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one variant and allocate outputs like production FP8 path."""
    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    launch_variant(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_q_fp8,
        index_weights_out,
        use_fma=use_fma,
    )
    return index_q_fp8, index_weights_out
