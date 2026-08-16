# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


# Consumed by kimi_k3_triton_warmup.py during kernel_warmup().
def get_attn_res_triton_warmup_profiles(
    max_blocks: int,
) -> tuple[tuple[int, bool, int, bool], ...]:
    """Return the small-batch profiles that bypass the native kernel."""
    profiles = [
        (num_blocks, False, -1, True) for num_blocks in range(2, max_blocks + 1)
    ]
    profiles.extend(
        (block_write_idx, True, block_write_idx, True)
        for block_write_idx in range(2, max_blocks)
    )
    profiles.append((max_blocks, True, -1, False))
    return tuple(profiles)


@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    updated_prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row_idx * stride_delta_m + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix += delta
        # Match the BF16 prefix-add result before using it as a residual source.
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    # With only the prefix source, the AttnRes softmax is exactly one.
    if num_blocks == 0:
        mixed = updated_prefix
    else:
        # Reloading avoids keeping the full prefix vector live across the loop.
        if HAS_DELTA:
            tl.debug_barrier()
        input_qk_weight = tl.load(
            norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32) * tl.load(
            qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        mixed = tl.zeros((BLOCK_D,), tl.float32)

        num_sources = num_blocks + 1
        for source_tile in range(tl.cdiv(num_sources, BLOCK_L)):
            source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            block_ptrs = (
                blocks_ptr
                + row_idx * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + d_offsets[None, :]
            )
            prefix_ptrs = (
                prefix_ptr
                + row_idx * stride_prefix_m
                + source_offsets[:, None] * 0
                + d_offsets[None, :]
            )
            value_ptrs = tl.where(is_prefix[:, None], prefix_ptrs, block_ptrs)
            values = tl.load(
                value_ptrs,
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            block_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
            max_logit = new_max_logit

        mixed /= denominator
    output = mixed

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if APPLY_OUTPUT_NORM:
        output_reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        output = mixed * output_reciprocal_std * output_norm_weight
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix.shape
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    # The in-tree NVIDIA kernel covers the common fused-add + output-norm path;
    # Triton handles block boundaries and final pre-norm output. The native op
    # is only compiled for SM100 under CUDA >= 13, so a device check alone is
    # not enough to know it exists.
    if (
        hidden_size == 7168
        and delta is not None
        and output_norm_weight is not None
        and num_blocks > 0
        and block_write_idx < 0
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "kimi_k3_attn_res")
    ):
        return ops.kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            eps,
            output_norm_eps,
        )
    output = prefix.new_empty(prefix.shape)
    # Tuned on GB300: source tiling helps decode, while one-source tiles scale
    # better for prefill.
    # Keep get_attn_res_triton_warmup_profiles in sync with these fallbacks.
    if num_tokens >= 256 or num_blocks <= 1:
        block_l, num_warps = 1, 4
    else:
        block_l, num_warps = 4, 8
    _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        block_write_idx,
        eps,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output
