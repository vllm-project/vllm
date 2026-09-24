# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Triton iHC kernels for HY V4.

This module is intentionally separate from the CUDA implementation.  The
kernels use fp32 reductions and fixed launch shapes that are suitable for
HIP/Triton graph capture.
"""

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_BLOCK_K = 1024
_BLOCK_D = 1024
_PRE_STAGE1_WARPS = 8
_PRE_STAGE2_WARPS = 4
_POST_WARPS = 4


def triton_ihc_supported(x: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_rocm()
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and not envs.VLLM_BATCH_INVARIANT
    )


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    x = tl.load(x_ptr + token_idx * K_TOTAL + k_offsets, mask=k_mask, other=0.0).to(
        tl.float32
    )
    weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(
            weight_ptr + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        * x[None, :],
        axis=1,
    )
    post_mix = tl.sum(
        tl.load(
            weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        * x[None, :],
        axis=1,
    )
    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, tl.sum(x * x, axis=0))
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(partial + 1 + HC_POW2 + hc_idx, mask=hc_mask, other=0.0)

    x_row = x_ptr + token_idx * K_TOTAL
    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS
    post = (
        MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
        + HC_EPS
    )
    if hidden_block_idx == 0:
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += gate * channel
    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    post = tl.load(post_ptr + token_idx * HC_MULT + hc_idx, mask=hc_mask, other=0.0).to(
        tl.float32
    )
    offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < HIDDEN_SIZE
    x = tl.load(x_ptr + token_idx * HIDDEN_SIZE + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    residual_row = residual_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    output_row = output_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    for channel_idx in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_row + channel_idx * HIDDEN_SIZE + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        tl.store(
            output_row + channel_idx * HIDDEN_SIZE + offsets,
            (gate * x + residual).to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3 and x.is_cuda
    assert weight.dtype == torch.float32 and scale.dtype == torch.float32
    assert base.dtype == torch.float32
    assert weight.is_cuda and weight.ndim == 2
    # Static iHC weights are made contiguous once by
    # HYV4Model.process_weights_after_loading. Keep this guard for reloads or
    # alternate loaders that may provide a non-contiguous parameter.
    x = x.contiguous()
    scale, base = scale.contiguous(), base.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    hc_pow2 = triton.next_power_of_2(hc_mult)
    weight = weight.contiguous()
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,) and base.shape == (2 * hc_mult,)
    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
        )
    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    num_splits = triton.cdiv(k_total, _BLOCK_K)
    partial_stride = 1 + 2 * hc_pow2
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )
    _ihc_pre_stage1[(num_tokens, num_splits)](
        x,
        weight,
        partial,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        BLOCK_K=_BLOCK_K,
        PARTIAL_STRIDE=partial_stride,
        num_warps=_PRE_STAGE1_WARPS,
        enable_fp_fusion=False,
    )
    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        partial,
        scale,
        base,
        output,
        post,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_D=_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=_PRE_STAGE2_WARPS,
        enable_fp_fusion=False,
    )
    return output, post


@triton.jit
def _ihc_post_pre_stage1(
    x_ptr,
    residual_ptr,
    attn_post_ptr,
    weight_ptr,
    partial_ptr,
    y_ptr,
    K_TOTAL: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offsets < K_TOTAL
    channel_idx = offsets // HIDDEN_SIZE
    hidden_idx = offsets % HIDDEN_SIZE
    gates = tl.load(
        attn_post_ptr + token_idx * HC_MULT + channel_idx,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_idx,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + token_idx * K_TOTAL + offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    y = gates * x + residual
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + token_idx * K_TOTAL + offsets, y, mask=k_mask)
    y = y.to(tl.float32)

    weight_offsets = hc_idx[:, None] * K_TOTAL + offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0) * y[None, :],
        axis=1,
    )
    post_mix = tl.sum(
        tl.load(
            weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        * y[None, :],
        axis=1,
    )
    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, tl.sum(y * y, axis=0))
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


@triton.jit
def _ihc_post_pre_stage2(
    y_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(partial + 1 + HC_POW2 + hc_idx, mask=hc_mask, other=0.0)

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS
    post = (
        MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
        + HC_EPS
    )
    if hidden_block_idx == 0:
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)

    offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = offsets < HIDDEN_SIZE
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    y_row = y_ptr + token_idx * K_TOTAL
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            y_row + channel_idx * HIDDEN_SIZE + offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += gate * channel
    output = output.to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + token_idx * HIDDEN_SIZE + offsets, output, mask=hidden_mask)


@triton.jit
def _ihc_post_pre_rms_norm_stage2(
    y_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    norm_weight_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_HIDDEN_BLOCKS: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(partial + 1 + HC_POW2 + hc_idx, mask=hc_mask, other=0.0)

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS
    post = (
        MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
        + HC_EPS
    )
    tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)

    y_row = y_ptr + token_idx * K_TOTAL
    output_sum_squares = tl.zeros((), dtype=tl.float32)
    for hidden_block_idx in tl.static_range(NUM_HIDDEN_BLOCKS):
        offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        hidden_mask = offsets < HIDDEN_SIZE
        output = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for channel_idx in tl.static_range(HC_MULT):
            channel = tl.load(
                y_row + channel_idx * HIDDEN_SIZE + offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
            output += gate * channel
        output = output.to(output_ptr.dtype.element_ty).to(tl.float32)
        output_sum_squares += tl.sum(
            tl.where(hidden_mask, output * output, 0.0), axis=0
        )

    reciprocal_rms = tl.rsqrt(output_sum_squares / HIDDEN_SIZE + NORM_EPS)
    for hidden_block_idx in tl.static_range(NUM_HIDDEN_BLOCKS):
        offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        hidden_mask = offsets < HIDDEN_SIZE
        output = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for channel_idx in tl.static_range(HC_MULT):
            channel = tl.load(
                y_row + channel_idx * HIDDEN_SIZE + offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
            output += gate * channel
        output = output.to(output_ptr.dtype.element_ty).to(tl.float32)
        norm_weight = tl.load(
            norm_weight_ptr + offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            output_ptr + token_idx * HIDDEN_SIZE + offsets,
            (output * reciprocal_rms * norm_weight).to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


def triton_ihc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_post: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and residual.dim() == 3 and attn_post.dim() == 2
    assert x.is_cuda and residual.is_cuda and attn_post.is_cuda
    assert weight.is_cuda and scale.is_cuda and base.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert residual.dtype == x.dtype and attn_post.dtype == torch.float32
    assert weight.dtype == torch.float32 and scale.dtype == torch.float32
    assert base.dtype == torch.float32
    x, residual, attn_post = (
        x.contiguous(),
        residual.contiguous(),
        attn_post.contiguous(),
    )
    # The model makes static weights contiguous after loading; these guards
    # protect direct kernel callers and repeated/alternate loading paths.
    weight, scale, base = (
        weight.contiguous(),
        scale.contiguous(),
        base.contiguous(),
    )
    num_tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    k_total = hc_mult * hidden_size
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert attn_post.shape == (num_tokens, hc_mult)
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,) and base.shape == (2 * hc_mult,)
    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
            torch.empty_like(residual),
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    y = torch.empty_like(residual)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    num_splits = triton.cdiv(k_total, _BLOCK_K)
    partial_stride = 1 + 2 * hc_pow2
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )
    _ihc_post_pre_stage1[(num_tokens, num_splits)](
        x,
        residual,
        attn_post,
        weight,
        partial,
        y,
        K_TOTAL=k_total,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        BLOCK_K=_BLOCK_K,
        PARTIAL_STRIDE=partial_stride,
        num_warps=_PRE_STAGE1_WARPS,
        enable_fp_fusion=False,
    )
    _ihc_post_pre_stage2[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        y,
        partial,
        scale,
        base,
        output,
        post,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_D=_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=_PRE_STAGE2_WARPS,
        enable_fp_fusion=False,
    )
    return output, post, y


def triton_ihc_post_pre_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_post: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and residual.dim() == 3 and attn_post.dim() == 2
    assert x.is_cuda and residual.is_cuda and attn_post.is_cuda
    assert weight.is_cuda and scale.is_cuda and base.is_cuda
    assert norm_weight.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert residual.dtype == x.dtype and attn_post.dtype == torch.float32
    assert weight.dtype == torch.float32 and scale.dtype == torch.float32
    assert base.dtype == torch.float32
    x, residual, attn_post = (
        x.contiguous(),
        residual.contiguous(),
        attn_post.contiguous(),
    )
    weight, scale, base = (
        weight.contiguous(),
        scale.contiguous(),
        base.contiguous(),
    )
    # Normal serving makes this loaded parameter contiguous in the post-load
    # hook; keep this guard for direct callers and alternate loading paths.
    norm_weight = norm_weight.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    k_total = hc_mult * hidden_size
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert attn_post.shape == (num_tokens, hc_mult)
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,) and base.shape == (2 * hc_mult,)
    assert norm_weight.shape == (hidden_size,)
    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
            torch.empty_like(residual),
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    y = torch.empty_like(residual)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    num_splits = triton.cdiv(k_total, _BLOCK_K)
    num_hidden_blocks = triton.cdiv(hidden_size, _BLOCK_D)
    partial_stride = 1 + 2 * hc_pow2
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )
    _ihc_post_pre_stage1[(num_tokens, num_splits)](
        x,
        residual,
        attn_post,
        weight,
        partial,
        y,
        K_TOTAL=k_total,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        BLOCK_K=_BLOCK_K,
        PARTIAL_STRIDE=partial_stride,
        num_warps=_PRE_STAGE1_WARPS,
        enable_fp_fusion=False,
    )
    _ihc_post_pre_rms_norm_stage2[(num_tokens,)](
        y,
        partial,
        scale,
        base,
        norm_weight,
        output,
        post,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_D=_BLOCK_D,
        NUM_HIDDEN_BLOCKS=num_hidden_blocks,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=_PRE_STAGE2_WARPS,
        enable_fp_fusion=False,
    )
    return output, post, y


def triton_ihc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    assert x.dim() == 2 and x.is_cuda and residual.is_cuda and post.is_cuda
    assert post.dtype == torch.float32
    x, residual, post = x.contiguous(), residual.contiguous(), post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    if num_tokens == 0:
        return torch.empty_like(residual)
    output = torch.empty_like(residual)
    _ihc_post_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=_BLOCK_D,
        num_warps=_POST_WARPS,
        enable_fp_fusion=False,
    )
    return output
