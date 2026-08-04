# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    FUSE_OUTPUT_NORM: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if num_blocks == 0:
        output = prefix
    else:
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
            block_offsets = tl.where(source_mask & ~is_prefix, source_offsets, 0)
            block_ptrs = (
                blocks_ptr
                + row_idx * stride_block_m
                + block_offsets[:, None] * stride_block_r
                + d_offsets[None, :]
            )
            block_values = tl.load(
                block_ptrs,
                mask=(source_mask[:, None] & ~is_prefix[:, None] & d_mask[None, :]),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            values = tl.where(is_prefix[:, None], prefix[None, :], block_values)
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

        output = mixed / denominator
    if FUSE_OUTPUT_NORM:
        output = output.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        output_reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, output * output, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        output *= output_reciprocal_std * output_norm_weight
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


def attn_res(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix.shape
    assert 0 < num_blocks <= blocks.shape[1]
    assert blocks.shape[0] == num_tokens
    assert norm_weight.numel() == hidden_size
    assert qk_weight.numel() == hidden_size
    assert prefix.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1

    output = prefix.new_empty(prefix.shape)
    if num_tokens == 0:
        return output

    if num_tokens >= 256 or num_blocks <= 1:
        block_l, num_warps = 1, 4
    else:
        block_l, num_warps = 4, 8
    _attn_res_kernel[(num_tokens,)](
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        norm_weight,
        output,
        prefix.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        eps,
        eps,
        FUSE_OUTPUT_NORM=False,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
    )
    return output


def attn_res_rmsnorm(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix.shape
    assert 0 <= num_blocks <= blocks.shape[1]
    assert blocks.shape[0] == num_tokens
    assert norm_weight.numel() == hidden_size
    assert qk_weight.numel() == hidden_size
    assert output_norm_weight.numel() == hidden_size
    assert prefix.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight.stride(-1) == 1

    output = prefix.new_empty(prefix.shape)
    if num_tokens == 0:
        return output

    block_l = 1 if num_blocks <= 1 else 4
    num_warps = 4 if num_blocks <= 2 else 8
    _attn_res_kernel[(num_tokens,)](
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        prefix.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        eps,
        output_norm_eps,
        FUSE_OUTPUT_NORM=True,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
    )
    return output
