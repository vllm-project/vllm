# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.triton_utils import tl, triton

# The register-tile kernel holds next_pow2(num_blocks) x next_pow2(hidden_size)
# FP32 values per program; past this it spills. Kimi-K3 (8 bank rows, hidden
# 7168) sits exactly at the limit.
_REGTILE_MAX_ELEMENTS = 8 * 8192
# A full 8-row tile leaves room for one program per CU, so once the rows need a
# second wave (MI355X has 256 CUs) the online-softmax kernel's occupancy wins.
_REGTILE_8ROW_MAX_TOKENS = 256


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
    QUANT_MAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
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
        output,
        mask=d_mask,
    )


@triton.jit
def _attn_res_regtile_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    output_scale_ptr,
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
    QUANT_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Same contract as _attn_res_kernel, but all num_blocks bank rows are
    # loaded as one [BLOCK_R, BLOCK_D] register tile: every row load is in
    # flight at once and the softmax is a single max/exp/sum pass, instead of
    # one online-softmax step per BLOCK_L sources.
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    if num_blocks > 0:
        # Issue the bank loads first so they overlap the prefix work below.
        r_offsets = tl.arange(0, BLOCK_R)
        r_mask = r_offsets < num_blocks
        block_values = tl.load(
            blocks_ptr
            + row_idx * stride_block_m
            + r_offsets[:, None] * stride_block_r
            + d_offsets[None, :],
            mask=r_mask[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

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
        input_qk_weight = tl.load(
            norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32) * tl.load(
            qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        prefix_logit = tl.sum(updated_prefix * input_qk_weight, axis=0) * tl.rsqrt(
            tl.sum(updated_prefix * updated_prefix, axis=0) * (1.0 / hidden_size) + eps
        )
        block_logits = tl.sum(
            block_values * input_qk_weight[None, :], axis=1
        ) * tl.rsqrt(
            tl.sum(block_values * block_values, axis=1) * (1.0 / hidden_size) + eps
        )
        block_logits = tl.where(r_mask, block_logits, -float("inf"))
        max_logit = tl.maximum(tl.max(block_logits, axis=0), prefix_logit)
        block_scales = tl.exp(block_logits - max_logit)
        prefix_scale = tl.exp(prefix_logit - max_logit)
        denominator = tl.sum(block_scales, axis=0) + prefix_scale
        mixed = tl.sum(block_scales[:, None] * block_values, axis=0)
        mixed = (mixed + prefix_scale * updated_prefix) / denominator
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
    *,
    quant_dtype: torch.dtype | None = None,
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
    output = torch.empty_like(
        prefix, dtype=quant_dtype or prefix.dtype, memory_format=torch.contiguous_format
    )
    scale = (
        torch.empty((num_tokens, 1), device=prefix.device, dtype=torch.float32)
        if quant_dtype is not None
        else None
    )
    if num_tokens == 0:
        return output if scale is None else (output, scale)

    launch_args = (
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        scale,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
        quant_dtype,
    )
    block_r = triton.next_power_of_2(max(num_blocks, 1))
    if (
        envs.VLLM_K3_ATTN_RES_REGTILE
        and block_r * triton.next_power_of_2(hidden_size) <= _REGTILE_MAX_ELEMENTS
        # The tile is loaded before the prefix snapshot is stored.
        and not 0 <= block_write_idx < num_blocks
    ):
        # Up to 4 bank rows the register tile is faster at every token count.
        # With 8 rows it only is while every row has its own CU, so split the
        # rows between the two kernels. torch.compile traces one graph for all
        # token counts and drops shape guards, so a Python branch on
        # num_tokens would be frozen at the trace-time size; instead, compute
        # both grid sizes symbolically and let the unused one be empty (Triton
        # skips empty grids).
        if block_r <= 4:
            fallback_rows = 0
        else:
            fallback_rows = torch.sym_min(
                num_tokens,
                (num_tokens // (_REGTILE_8ROW_MAX_TOKENS + 1)) * num_tokens,
            )
        _launch_attn_res(
            *launch_args,
            num_rows=num_tokens - fallback_rows,
            regtile=True,
            block_l=0,
            # An 8-row tile needs the extra registers of 8 warps to not spill.
            num_warps=8 if block_r >= 8 else 4,
        )
        if block_r > 4:
            _launch_attn_res(
                *launch_args,
                num_rows=fallback_rows,
                regtile=False,
                block_l=1,
                num_warps=4,
            )
        return output if scale is None else (output, scale)

    # Source tiling helps decode, while one-source tiles scale better for prefill.
    if num_tokens >= 256 or num_blocks <= 1:
        block_l, num_warps = 1, 4
    else:
        block_l, num_warps = 4, 8
    _launch_attn_res(
        *launch_args,
        num_rows=num_tokens,
        regtile=False,
        block_l=block_l,
        num_warps=num_warps,
    )
    return output if scale is None else (output, scale)


def _launch_attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    output: torch.Tensor,
    scale: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    quant_dtype: torch.dtype | None,
    *,
    num_rows: int,
    regtile: bool,
    block_l: int,
    num_warps: int,
) -> None:
    hidden_size = prefix.shape[1]
    if regtile:
        kernel = _attn_res_regtile_kernel
        source_tiling = dict(BLOCK_R=triton.next_power_of_2(max(num_blocks, 1)))
    else:
        kernel = _attn_res_kernel
        source_tiling = dict(BLOCK_L=block_l)
    kernel[(num_rows,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        scale,
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
        QUANT_MAX=0.0 if quant_dtype is None else get_fp8_min_max()[1],
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
        **source_tiling,
    )


# Split AttnRes (VLLM_K3_ATTNRES_SPLIT): the bank rows a mix reads are final
# well before the mix itself, so their online-softmax state (running max, sum
# and weighted sum) can be computed off the critical path. A combine then folds
# in only the prefix source, reading 1 fp32 + 2 bf16 rows per token instead of
# the whole bank. One sweep serves two mixes that read the same bank rows: this
# layer's pre-MLP mix and the next layer's pre-attention mix.


@triton.jit
def _attn_res_partial_kernel(
    blocks_ptr,
    norm_weight_a_ptr,
    qk_weight_a_ptr,
    norm_weight_b_ptr,
    qk_weight_b_ptr,
    max_a_ptr,
    sum_a_ptr,
    acc_a_ptr,
    max_b_ptr,
    sum_b_ptr,
    acc_b_ptr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    eps_a: tl.constexpr,
    eps_b: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    qk_a = tl.load(norm_weight_a_ptr + d_offsets, mask=d_mask, other=0.0).to(
        tl.float32
    ) * tl.load(qk_weight_a_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    max_a = tl.full((), -float("inf"), tl.float32)
    sum_a = tl.zeros((), tl.float32)
    acc_a = tl.zeros((BLOCK_D,), tl.float32)
    if HAS_B:
        qk_b = tl.load(norm_weight_b_ptr + d_offsets, mask=d_mask, other=0.0).to(
            tl.float32
        ) * tl.load(qk_weight_b_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        max_b = tl.full((), -float("inf"), tl.float32)
        sum_b = tl.zeros((), tl.float32)
        acc_b = tl.zeros((BLOCK_D,), tl.float32)

    for source in range(num_blocks):
        values = tl.load(
            blocks_ptr + row_idx * stride_block_m + source * stride_block_r + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) * (1.0 / hidden_size)

        logit_a = tl.sum(values * qk_a, axis=0) * tl.rsqrt(mean_square + eps_a)
        new_max_a = tl.maximum(max_a, logit_a)
        old_scale_a = tl.exp(max_a - new_max_a)
        scale_a = tl.exp(logit_a - new_max_a)
        sum_a = sum_a * old_scale_a + scale_a
        acc_a = acc_a * old_scale_a + scale_a * values
        max_a = new_max_a
        if HAS_B:
            logit_b = tl.sum(values * qk_b, axis=0) * tl.rsqrt(mean_square + eps_b)
            new_max_b = tl.maximum(max_b, logit_b)
            old_scale_b = tl.exp(max_b - new_max_b)
            scale_b = tl.exp(logit_b - new_max_b)
            sum_b = sum_b * old_scale_b + scale_b
            acc_b = acc_b * old_scale_b + scale_b * values
            max_b = new_max_b

    tl.store(acc_a_ptr + row_idx * hidden_size + d_offsets, acc_a, mask=d_mask)
    tl.store(max_a_ptr + row_idx, max_a)
    tl.store(sum_a_ptr + row_idx, sum_a)
    if HAS_B:
        tl.store(acc_b_ptr + row_idx * hidden_size + d_offsets, acc_b, mask=d_mask)
        tl.store(max_b_ptr + row_idx, max_b)
        tl.store(sum_b_ptr + row_idx, sum_b)


@triton.jit
def _attn_res_combine_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    max_ptr,
    sum_ptr,
    acc_ptr,
    output_ptr,
    output_scale_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Same contract as _attn_res_kernel, with the bank sources replaced by
    # their precomputed online-softmax state (_attn_res_partial_kernel).
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
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

    input_qk_weight = tl.load(norm_weight_ptr + d_offsets, mask=d_mask, other=0.0).to(
        tl.float32
    ) * tl.load(qk_weight_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    prefix_logit = tl.sum(updated_prefix * input_qk_weight, axis=0) * tl.rsqrt(
        tl.sum(updated_prefix * updated_prefix, axis=0) * (1.0 / hidden_size) + eps
    )
    blocks_max = tl.load(max_ptr + row_idx)
    max_logit = tl.maximum(blocks_max, prefix_logit)
    blocks_scale = tl.exp(blocks_max - max_logit)
    prefix_scale = tl.exp(prefix_logit - max_logit)
    denominator = tl.load(sum_ptr + row_idx) * blocks_scale + prefix_scale
    acc = tl.load(acc_ptr + row_idx * hidden_size + d_offsets, mask=d_mask, other=0.0)
    mixed = (acc * blocks_scale + prefix_scale * updated_prefix) / denominator
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
        output,
        mask=d_mask,
    )


# Online-softmax state of a mix over its bank rows: (max [T], sum [T], acc
# [T, hidden]), all FP32.
AttnResPartial = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _empty_partial(blocks: torch.Tensor) -> AttnResPartial:
    num_tokens, _, hidden_size = blocks.shape
    return (
        torch.empty(num_tokens, device=blocks.device, dtype=torch.float32),
        torch.empty(num_tokens, device=blocks.device, dtype=torch.float32),
        torch.empty(
            (num_tokens, hidden_size), device=blocks.device, dtype=torch.float32
        ),
    )


def attn_res_partial_alloc(
    blocks: torch.Tensor, dual: bool
) -> tuple[AttnResPartial, AttnResPartial | None]:
    """Allocate the outputs of attn_res_partial.

    Kept separate so a caller running the partial on a side stream allocates
    on the consuming stream before the fork: the allocator orders reuse per
    stream, and the consumer joins the side stream before reading.
    """
    return _empty_partial(blocks), _empty_partial(blocks) if dual else None


def attn_res_partial(
    blocks: torch.Tensor,
    num_blocks: int,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    eps: float,
    out: AttnResPartial,
    *,
    next_norm_weight: torch.Tensor | None = None,
    next_qk_weight: torch.Tensor | None = None,
    next_eps: float = 0.0,
    next_out: AttnResPartial | None = None,
) -> None:
    """Online-softmax state of bank rows [0, num_blocks) for one or two mixes.

    The second (``next_*``) mix shares the bank sweep; both states feed
    attn_res_combine.
    """
    num_tokens, _, hidden_size = blocks.shape
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1 and qk_weight.stride(-1) == 1
    assert num_blocks > 0
    dual = next_out is not None
    if dual:
        assert next_norm_weight is not None and next_qk_weight is not None
        assert next_norm_weight.stride(-1) == 1 and next_qk_weight.stride(-1) == 1
    if num_tokens == 0:
        return
    max_a, sum_a, acc_a = out
    max_b, sum_b, acc_b = next_out if dual else out
    _attn_res_partial_kernel[(num_tokens,)](
        blocks,
        norm_weight,
        qk_weight,
        next_norm_weight if dual else norm_weight,
        next_qk_weight if dual else qk_weight,
        max_a,
        sum_a,
        acc_a,
        max_b,
        sum_b,
        acc_b,
        blocks.stride(0),
        blocks.stride(1),
        num_blocks,
        hidden_size,
        eps,
        next_eps if dual else 0.0,
        HAS_B=dual,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        # Two accumulators of a full hidden row need the extra registers.
        num_warps=8 if dual else 4,
        num_stages=2,
    )


def attn_res_combine(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    partial: AttnResPartial,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    *,
    quant_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """attn_res over a precomputed bank partial; same outputs and side effects
    (prefix update, block write) as attn_res with the partial's num_blocks."""
    num_tokens, hidden_size = prefix.shape
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    max_logit, denominator, acc = partial
    assert acc.shape == prefix.shape and acc.is_contiguous()
    if quant_dtype is not None:
        assert quant_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    output = torch.empty_like(
        prefix, dtype=quant_dtype or prefix.dtype, memory_format=torch.contiguous_format
    )
    scale = (
        torch.empty((num_tokens, 1), device=prefix.device, dtype=torch.float32)
        if quant_dtype is not None
        else None
    )
    if num_tokens == 0:
        return output if scale is None else (output, scale)

    _attn_res_combine_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        max_logit,
        denominator,
        acc,
        output,
        scale,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        hidden_size,
        block_write_idx,
        eps,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=0.0 if quant_dtype is None else get_fp8_min_max()[1],
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=4,
        num_stages=2,
    )
    return output if scale is None else (output, scale)
