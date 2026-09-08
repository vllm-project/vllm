# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton iHC pre / post / head kernels for HY V4.

Adapted from the SGLang HY V4 implementation:
https://github.com/sgl-project/sglang/pull/36805

The ops are registered as custom ops with fake impls
(torch.ops.vllm.hy_v4_ihc_pre / post / head) so they stay opaque to
torch.compile and are safe inside CUDA graphs. head is pre without the post
gates (HAS_POST=False) and shares its two kernels.
"""

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_K = 1024
_BLOCK_D = 1024


def triton_ihc_supported(x: torch.Tensor) -> bool:
    """Return whether the in-tree Triton path can run for this input."""
    return (
        HAS_TRITON
        and current_platform.is_cuda()
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
    HAS_POST: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    x = tl.load(
        x_ptr + token_idx * K_TOTAL + k_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)

    weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0) * x[None, :],
        axis=1,
    )
    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    if HAS_POST:
        post_mix = tl.sum(
            tl.load(
                weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
                mask=weight_mask,
                other=0.0,
            )
            * x[None, :],
            axis=1,
        )
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
    HAS_POST: tl.constexpr,
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
        if HAS_POST:
            post_mix += tl.load(
                partial + 1 + HC_POW2 + hc_idx,
                mask=hc_mask,
                other=0.0,
            )

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if HAS_POST:  # noqa: SIM102 - constexpr branch, keep apart from the runtime one
        if hidden_block_idx == 0:
            post_scale = tl.load(scale_ptr + 1)
            post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
            post = (
                MAGNITUDE
                * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
                + HC_EPS
            )
            tl.store(
                post_ptr + token_idx * HC_MULT + hc_idx,
                post,
                mask=hc_mask,
            )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

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
    post = tl.load(
        post_ptr + token_idx * HC_MULT + hc_idx,
        mask=hc_mask,
        other=0.0,
    )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    residual_row = residual_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    output_row = output_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    for channel_idx in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        output = channel_gate * x + residual
        tl.store(
            output_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


def _ihc_reduce(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    has_post: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared launch for pre (with post gates) and head (without)."""
    assert x.dim() == 3
    assert x.is_cuda and weight.is_cuda and scale.is_cuda and base.is_cuda
    assert weight.dtype == torch.float32
    assert scale.dtype == torch.float32 and base.dtype == torch.float32

    x = x.contiguous()
    assert weight.is_contiguous()
    scale = scale.contiguous()
    base = base.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    n_out = 2 * hc_mult if has_post else hc_mult
    assert weight.shape == (n_out, k_total)
    assert scale.shape == (2 if has_post else 1,)
    assert base.shape == (n_out,)

    post_rows = hc_mult if has_post else 0
    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, post_rows), dtype=torch.float32, device=x.device),
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, post_rows), dtype=torch.float32, device=x.device)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    num_splits = triton.cdiv(k_total, _BLOCK_K)
    partial_stride = 1 + (2 if has_post else 1) * hc_pow2
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
        HAS_POST=has_post,
        num_warps=8,
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
        HAS_POST=has_post,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, post


def _triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ihc_reduce(x, weight, scale, base, magnitude, hc_eps, norm_eps, True)


def _triton_ihc_pre_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = x.shape
    return (
        x.new_empty((num_tokens, hidden_size)),
        x.new_empty((num_tokens, hc_mult), dtype=torch.float32),
    )


def _triton_ihc_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    return _ihc_reduce(x, weight, scale, base, 1.0, hc_eps, norm_eps, False)[0]


def _triton_ihc_head_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = x.shape
    return x.new_empty((num_tokens, hidden_size))


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels and produce the post gates."""
    return torch.ops.vllm.hy_v4_ihc_pre(
        x, weight, scale, base, magnitude, hc_eps, norm_eps
    )


def triton_ihc_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    """Reduce iHC channels for the head layer (no post gates)."""
    return torch.ops.vllm.hy_v4_ihc_head(x, weight, scale, base, hc_eps, norm_eps)


def _triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    assert x.dim() == 2
    assert x.is_cuda and residual.is_cuda and post.is_cuda
    assert post.dtype == torch.float32

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    if num_tokens == 0:
        return torch.empty((0, hc_mult, hidden_size), dtype=x.dtype, device=x.device)

    output = torch.empty(
        (num_tokens, hc_mult, hidden_size), dtype=x.dtype, device=x.device
    )
    _ihc_post_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=_BLOCK_D,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def _triton_ihc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    return residual.new_empty(residual.shape)


def triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Scatter a sub-block output back over the iHC residual channels."""
    return torch.ops.vllm.hy_v4_ihc_post(x, residual, post)


direct_register_custom_op(
    op_name="hy_v4_ihc_pre",
    op_func=_triton_ihc_pre,
    fake_impl=_triton_ihc_pre_fake,
)
direct_register_custom_op(
    op_name="hy_v4_ihc_head",
    op_func=_triton_ihc_head,
    fake_impl=_triton_ihc_head_fake,
)
direct_register_custom_op(
    op_name="hy_v4_ihc_post",
    op_func=_triton_ihc_post,
    fake_impl=_triton_ihc_post_fake,
)
