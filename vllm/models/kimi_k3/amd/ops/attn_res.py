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

_DECODE_NUM_WARPS = 8
_PEEL_PREFIX_MIN_TOKENS = 256


@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_ptr,
    addend_ptr,
    prefix_sum_ptr,
    out_norm_weight_ptr,
    norm_output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    stride_addend_m: tl.constexpr,
    stride_prefix_sum_m: tl.constexpr,
    stride_norm_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    num_sources: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    out_norm_eps: tl.constexpr,
    HAS_ADD: tl.constexpr,
    HAS_NORM: tl.constexpr,
    STORE_MIX: tl.constexpr,
    PEEL_PREFIX: tl.constexpr,
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
    if HAS_ADD:
        # Round to bf16 before consuming the sum, so both the stored prefix-sum
        # and the mix below see exactly what the separate bf16 `prefix + addend`
        # op this replaces would have produced.
        prefix_sum = (
            prefix
            + tl.load(
                addend_ptr + row_idx * stride_addend_m + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(
            prefix_sum_ptr + row_idx * stride_prefix_sum_m + d_offsets,
            prefix_sum,
            mask=d_mask,
        )
        prefix = prefix_sum.to(tl.float32)
    input_qk_weight = tl.load(norm_weight_ptr + d_offsets, mask=d_mask, other=0.0).to(
        tl.float32
    ) * tl.load(qk_weight_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    if PEEL_PREFIX:
        # Seed the online-softmax state with the prefix, then iterate blocks.
        max_logit = tl.sum(prefix * input_qk_weight, axis=0) * tl.rsqrt(
            tl.sum(prefix * prefix, axis=0) * (1.0 / hidden_size) + eps
        )
        denominator = tl.full((), 1.0, tl.float32)
        mixed = prefix
    else:
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        mixed = tl.zeros((BLOCK_D,), tl.float32)

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
        block_values = tl.load(
            block_ptrs,
            mask=(source_mask[:, None] & ~is_prefix[:, None] & d_mask[None, :]),
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if PEEL_PREFIX:
            values = block_values
        else:
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
    if STORE_MIX:
        tl.store(
            output_ptr + row_idx * stride_output_m + d_offsets,
            output,
            mask=d_mask,
        )
    if HAS_NORM:
        # BLOCK_D already spans the whole hidden dimension, so this reduction is
        # workgroup-local. Round first to match the standalone bf16 norm input.
        mixed_bf16 = output.to(tl.bfloat16).to(tl.float32)
        reciprocal_std = tl.rsqrt(
            tl.sum(mixed_bf16 * mixed_bf16, axis=0) * (1.0 / hidden_size) + out_norm_eps
        )
        out_weight = tl.load(
            out_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            norm_output_ptr + row_idx * stride_norm_output_m + d_offsets,
            mixed_bf16 * reciprocal_std * out_weight,
            mask=d_mask,
        )


def _launch_attn_res(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
    addend: torch.Tensor | None,
    out_norm_weight: torch.Tensor | None,
    out_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_tokens, hidden_size = prefix.shape
    assert 0 < num_blocks <= blocks.shape[1]
    assert blocks.shape[0] == num_tokens
    assert norm_weight.numel() == hidden_size
    assert qk_weight.numel() == hidden_size
    assert prefix.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1

    if addend is not None:
        assert addend.shape == prefix.shape
        assert addend.stride(-1) == 1
        addend_arg = addend
        prefix_sum = prefix.new_empty(prefix.shape)
        addend_stride = addend.stride(0)
        prefix_sum_stride = prefix_sum.stride(0)
    else:
        addend_arg = prefix
        prefix_sum = None
        addend_stride = prefix_sum_stride = 0

    if out_norm_weight is not None:
        assert out_norm_weight.numel() == hidden_size
        assert out_norm_weight.stride(-1) == 1
        out_norm_weight_arg = out_norm_weight
    else:
        out_norm_weight_arg = norm_weight

    has_add = addend is not None
    has_norm = out_norm_weight is not None
    store_mix = not has_norm
    result = prefix.new_empty(prefix.shape)
    if num_tokens == 0:
        return result, prefix_sum

    peel_prefix = num_tokens >= _PEEL_PREFIX_MIN_TOKENS
    if peel_prefix:
        block_l, num_warps = 1, 4
    else:
        block_l = min(triton.next_power_of_2(num_blocks + 1), 16)
        num_warps = _DECODE_NUM_WARPS
    _attn_res_kernel[(num_tokens,)](
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        result,
        addend_arg,
        prefix_sum if prefix_sum is not None else prefix,
        out_norm_weight_arg,
        result,
        prefix.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        result.stride(0),
        addend_stride,
        prefix_sum_stride,
        result.stride(0),
        num_blocks,
        num_blocks if peel_prefix else num_blocks + 1,
        hidden_size,
        eps,
        out_norm_eps,
        HAS_ADD=has_add,
        HAS_NORM=has_norm,
        STORE_MIX=store_mix,
        PEEL_PREFIX=peel_prefix,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
    )
    return result, prefix_sum


def attn_res(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    output, _ = _launch_attn_res(
        prefix, blocks, norm_weight, qk_weight, num_blocks, eps, None, None, 0.0
    )
    return output


def attn_res_fused(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
    addend: torch.Tensor | None = None,
    out_norm_weight: torch.Tensor | None = None,
    out_norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply attn-res with an optional residual add and output RMSNorm.

    Returns ``(output, prefix_sum)``. ``output`` is the RMSNorm of the mixed
    result when ``out_norm_weight`` is given, and the raw mix otherwise;
    ``prefix_sum`` is ``prefix + addend`` when ``addend`` is given, and ``None``
    otherwise.
    """
    return _launch_attn_res(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        num_blocks,
        eps,
        addend,
        out_norm_weight,
        out_norm_eps,
    )
