# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter (https://github.com/ROCm/aiter) v0.1.28.post1,
# MIT License, Copyright (C) 2024-2026, Advanced Micro Devices, Inc.:
# aiter/ops/triton/rope/q_rope_mxfp4_quant.py,
# aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py and
# aiter/ops/triton/_triton_kernels/quant/quant.py

import torch

from vllm.triton_utils import tl, triton

_SCALE_GROUP = 32

PROJECT_BLOCK_H = 2
PROJECT_BLOCK_K = 1024
PROJECT_NUM_WARPS = 1


@triton.jit
def _mxfp4_ceil_scale_from_amax(amax):
    ratio = tl.maximum(amax, 6.0 * (2**-126)) * (1.0 / 6.0)
    bs_e8m0 = (ratio.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23
    quant_scale = ((254 - bs_e8m0) << 23).to(tl.float32, bitcast=True)
    return bs_e8m0.to(tl.uint8), quant_scale


@triton.jit
def _mxfp4_pack_cvt(
    qx,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    lo, hi = tl.split(qx.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
    return tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        args=[lo, hi, 1.0],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _mxfp4_quant_op(x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE):
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    bs_e8m0, quant_scale = _mxfp4_ceil_scale_from_amax(amax)
    x_fp4 = _mxfp4_pack_cvt(x * quant_scale, BLOCK_SIZE_N, BLOCK_SIZE_M)
    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _q_rope_mxfp4_quant_kernel(
    q_ptr,
    q_stride_t,
    q_stride_h,
    positions_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    q_fp4_ptr,
    q_fp4_stride_t,
    q_fp4_stride_h,
    q_scale_ptr,
    q_scale_stride_t,
    q_scale_stride_h,
    weights_ptr,
    weights_stride,
    weights_scale,
    weights_out_ptr,
    weights_out_stride,
    num_heads,
    HEAD_SIZE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    hidden_ptr,
    hidden_stride,
    proj_ptr,
    proj_stride,
    PROJECT: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2

    tok = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < num_heads
    dims = tl.arange(0, HEAD_SIZE)

    if PROJECT:
        ks = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
        for k in tl.static_range(0, HIDDEN, BLOCK_K):
            x = tl.load(hidden_ptr + tok * hidden_stride + k + ks).to(tl.float32)
            wp = tl.load(
                proj_ptr + heads[:, None] * proj_stride + k + ks[None, :],
                mask=head_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += wp * x[None, :]
        projected = tl.sum(acc, axis=1).to(hidden_ptr.dtype.element_ty)

    q = tl.load(
        q_ptr + tok * q_stride_t + heads[:, None] * q_stride_h + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    pos = tl.load(positions_ptr + tok)
    rope_pair = tl.arange(0, NUM_PAIRS) - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cs_base = cos_sin_ptr + pos * cos_sin_stride
    cos = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(tl.float32)
    sin = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0).to(tl.float32)
    even, odd = tl.split(tl.reshape(q, (BLOCK_H, NUM_PAIRS, 2)))
    rot_even = (even * cos - odd * sin).to(tl.bfloat16).to(tl.float32)
    rot_odd = (odd * cos + even * sin).to(tl.bfloat16).to(tl.float32)
    even = tl.where(is_rope, rot_even, even)
    odd = tl.where(is_rope, rot_odd, odd)
    q = tl.reshape(tl.join(even, odd), (BLOCK_H, HEAD_SIZE))

    q_fp4, q_scale = _mxfp4_quant_op(q, HEAD_SIZE, BLOCK_H, 32)
    fp4_dims = tl.arange(0, HEAD_SIZE // 2)
    tl.store(
        q_fp4_ptr
        + tok * q_fp4_stride_t
        + heads[:, None] * q_fp4_stride_h
        + fp4_dims[None, :],
        q_fp4,
        mask=head_mask[:, None],
    )
    scale_dims = tl.arange(0, HEAD_SIZE // 32)
    tl.store(
        q_scale_ptr
        + tok * q_scale_stride_t
        + heads[:, None] * q_scale_stride_h
        + scale_dims[None, :],
        q_scale,
        mask=head_mask[:, None],
    )

    if HAS_WEIGHTS:
        if PROJECT:
            w = projected
        else:
            w = tl.load(weights_ptr + tok * weights_stride + heads, mask=head_mask)
        tl.store(
            weights_out_ptr + tok * weights_out_stride + heads,
            w.to(tl.float32) * weights_scale,
            mask=head_mask,
        )


def _launch_config(num_tokens: int, num_heads: int, head_size: int) -> tuple[int, int]:
    rows = num_tokens * num_heads
    if rows <= 8192:
        block_h = 4
    elif rows <= 262144:
        block_h = 16
    else:
        block_h = 64
    block_h = min(triton.next_power_of_2(num_heads), block_h)
    return block_h, max(1, block_h * head_size // 2048)


def q_rope_mxfp4_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor | None = None,
    weights_scale: float = 1.0,
    *,
    hidden_states: torch.Tensor | None = None,
    weights_proj: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """GPT-J RoPE on the last rope_dim dims -> MXFP4 query, plus scaled weights."""
    if q.ndim != 3 or q.stride(2) != 1:
        raise ValueError(
            f"q must be [T, H, D] with unit last stride, got "
            f"{tuple(q.shape)} {q.stride()}"
        )
    num_tokens, num_heads, head_size = q.shape
    rope_dim = cos_sin_cache.shape[-1]
    if head_size % _SCALE_GROUP or rope_dim % 2 or rope_dim > head_size:
        raise ValueError(
            f"head_size {head_size} must be a multiple of "
            f"{_SCALE_GROUP} and hold the {rope_dim}-dim rope"
        )
    if positions.numel() != num_tokens:
        raise ValueError(
            f"positions {tuple(positions.shape)} must cover q's T={num_tokens}"
        )
    project = hidden_states is not None
    if hidden_states is not None and (
        weights is not None
        or weights_proj is None
        or not can_project_weights(hidden_states, weights_proj, num_heads)
        or hidden_states.shape[0] != num_tokens
    ):
        raise ValueError(
            f"pass weights, or [T={num_tokens}, K] hidden_states with an "
            f"[H={num_heads}, K] weights_proj that can_project_weights accepts"
        )
    if weights is not None and (
        tuple(weights.shape) != (num_tokens, num_heads) or weights.stride(1) != 1
    ):
        raise ValueError(
            f"weights {tuple(weights.shape)} must be q's [T={num_tokens}, "
            f"H={num_heads}] with unit head stride"
        )
    q_fp4 = torch.empty(
        (num_tokens, num_heads, head_size // 2), dtype=torch.uint8, device=q.device
    )
    q_scale = torch.empty(
        (num_tokens, num_heads, head_size // _SCALE_GROUP),
        dtype=torch.uint8,
        device=q.device,
    )
    weights_out = None
    if weights is not None or project:
        weights_out = torch.empty(
            (num_tokens, num_heads), dtype=torch.float32, device=q.device
        )
    if num_tokens == 0:
        return q_fp4, q_scale, weights_out

    if project:
        block_h, num_warps = PROJECT_BLOCK_H, PROJECT_NUM_WARPS
    else:
        block_h, num_warps = _launch_config(num_tokens, num_heads, head_size)
    _q_rope_mxfp4_quant_kernel[(num_tokens, triton.cdiv(num_heads, block_h))](
        q,
        q.stride(0),
        q.stride(1),
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp4,
        q_fp4.stride(0),
        q_fp4.stride(1),
        q_scale,
        q_scale.stride(0),
        q_scale.stride(1),
        weights,
        0 if weights is None else weights.stride(0),
        weights_scale,
        weights_out,
        0 if weights_out is None else weights_out.stride(0),
        num_heads,
        HEAD_SIZE=head_size,
        ROPE_DIM=rope_dim,
        BLOCK_H=block_h,
        HAS_WEIGHTS=weights_out is not None,
        hidden_ptr=hidden_states,
        hidden_stride=None if hidden_states is None else hidden_states.stride(0),
        proj_ptr=weights_proj,
        proj_stride=None if weights_proj is None else weights_proj.stride(0),
        PROJECT=project,
        HIDDEN=0 if hidden_states is None else hidden_states.shape[1],
        BLOCK_K=PROJECT_BLOCK_K,
        num_warps=num_warps,
    )
    return q_fp4, q_scale, weights_out


def can_project_weights(
    hidden_states: torch.Tensor, weights_proj: torch.Tensor, num_heads: int
) -> bool:
    """Whether `q_rope_mxfp4_quant` can project `num_heads` weights from these."""
    return (
        hidden_states.dim() == 2
        and weights_proj.dim() == 2
        and hidden_states.stride(1) == 1
        and weights_proj.stride(1) == 1
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and weights_proj.dtype == hidden_states.dtype
        and weights_proj.shape[0] == num_heads
        and weights_proj.shape[1] == hidden_states.shape[1]
        and hidden_states.shape[1] % PROJECT_BLOCK_K == 0
    )
