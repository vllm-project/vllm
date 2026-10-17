# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.triton_utils import tl, triton


@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    output_scale_ptr,
    prefix_out_ptr,
    # Runtime strides let one compiled kernel serve all row strides. Do not add
    # them to do_not_specialize: the divisibility-by-16 hint keeps loads vectorized.
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_block_r,
    stride_output_m,
    stride_prefix_out_m,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    HAS_PREFIX_OUT: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    tl.assume(row_idx >= 0)
    tl.assume(stride_prefix_m > 0)
    tl.assume(stride_block_m > 0)
    tl.assume(stride_block_r > 0)
    tl.assume(stride_output_m > 0)
    # delta is absent on some launches and then has stride 0. Only the live
    # path is strictly positive.
    if HAS_DELTA:
        tl.assume(stride_delta_m > 0)
    if HAS_PREFIX_OUT:
        tl.assume(stride_prefix_out_m > 0)
    d_offsets = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_D), BLOCK_D), BLOCK_D
    )
    d_mask = d_offsets < hidden_size

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
        # Store that BF16 value directly so the write can pack; promoting it
        # back to fp32 before the store keeps the conversion on the scalar path.
        updated_prefix_bf16 = updated_prefix.to(prefix_ptr.dtype.element_ty)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix_bf16,
            mask=d_mask,
        )
        if HAS_PREFIX_OUT:
            # Same packed bf16 sum the separate auxiliary add used to write.
            tl.store(
                prefix_out_ptr + row_idx * stride_prefix_out_m + d_offsets,
                updated_prefix_bf16,
                mask=d_mask,
            )
        updated_prefix = updated_prefix_bf16.to(tl.float32)
    elif HAS_PREFIX_OUT:
        tl.store(
            prefix_out_ptr + row_idx * stride_prefix_out_m + d_offsets,
            updated_prefix.to(prefix_ptr.dtype.element_ty),
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix.to(blocks_ptr.dtype.element_ty),
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
        for source_tile in tl.range(
            0, tl.cdiv(num_sources, BLOCK_L), num_stages=LOOP_STAGES
        ):
            source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            # The prefix slot is not a block row. Point masked lanes at the last
            # real row so a vectorized D load does not use an out-of-bounds
            # address; those lanes are dropped by the mask and replaced below.
            block_row = tl.minimum(source_offsets, num_blocks - 1)
            block_ptrs = (
                blocks_ptr
                + row_idx * stride_block_m
                + block_row[:, None] * stride_block_r
                + d_offsets[None, :]
            )
            block_values = tl.load(
                block_ptrs,
                mask=(source_mask[:, None] & ~is_prefix[:, None] & d_mask[None, :]),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            values = tl.where(is_prefix[:, None], updated_prefix[None, :], block_values)
            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            # Scaled by log2(e) so the softmax can use exp2.
            logits = (
                tl.sum(values * input_qk_weight[None, :], axis=1)
                * reciprocal_std
                * 1.4426950408889634
            )
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp2(max_logit - new_max_logit)
            block_scales = tl.exp2(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
            max_logit = new_max_logit

        mixed /= denominator
    output = mixed

    if APPLY_OUTPUT_NORM:
        output_reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        output = mixed * output_reciprocal_std * output_norm_weight
    if QUANT_MAX > 0:
        # Preserve the rounding of the original BF16 output before quantizing.
        output = output.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        amax = tl.max(tl.where(d_mask, tl.abs(output), 0.0), axis=0)
        scale = tl.maximum(amax, 1e-10) * (1.0 / QUANT_MAX)
        inv_scale = 1.0 / scale
        output = tl.minimum(tl.maximum(output * inv_scale, -QUANT_MAX), QUANT_MAX)
        tl.store(output_scale_ptr + row_idx, scale)
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output.to(output_ptr.dtype.element_ty),
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
    *,
    quant_dtype: torch.dtype | None = None,
    prefix_snapshot: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = prefix.shape
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    if quant_dtype is not None:
        assert quant_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if prefix_snapshot is not None:
        assert prefix_snapshot.shape == prefix.shape
        assert prefix_snapshot.stride(-1) == 1
        assert prefix_snapshot.dtype == prefix.dtype
    output = torch.empty_like(
        prefix, dtype=quant_dtype or prefix.dtype, memory_format=torch.contiguous_format
    )
    scale = (
        torch.empty((num_tokens, 1), device=prefix.device, dtype=torch.float32)
        if quant_dtype is not None
        else None
    )
    if num_tokens == 0:
        if prefix_snapshot is not None:
            if delta is None:
                prefix_snapshot.copy_(prefix)
            else:
                torch.add(prefix, delta, out=prefix_snapshot)
        return output if scale is None else (output, scale)

    # Decode covers every source (num_blocks + the prefix) in one tile, so the
    # online softmax is a single pass. Prefill keeps one-source tiles and
    # software-pipelines the source loop instead.
    if num_tokens >= 256:
        block_l, num_warps, loop_stages = 1, 4, 2
    else:
        block_l, num_warps, loop_stages = (
            triton.next_power_of_2(num_blocks + 1),
            8,
            1,
        )
    _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        scale,
        prefix if prefix_snapshot is None else prefix_snapshot,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        0 if prefix_snapshot is None else prefix_snapshot.stride(0),
        num_blocks,
        hidden_size,
        block_write_idx,
        eps,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        HAS_PREFIX_OUT=prefix_snapshot is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=0.0 if quant_dtype is None else get_fp8_min_max()[1],
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        LOOP_STAGES=loop_stages,
        num_warps=num_warps,
        num_stages=2,
    )
    return output if scale is None else (output, scale)
