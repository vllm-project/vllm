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
