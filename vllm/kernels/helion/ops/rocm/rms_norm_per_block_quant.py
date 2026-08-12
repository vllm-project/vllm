# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm implementation of fused RMSNorm and per-block quantization."""

import torch

import helion
import helion.language as hl

from vllm.kernels.helion.utils import (
    get_fp8_dtype,
    get_int8_min_max,
    get_int8_min_scaling_factor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)


def rms_norm_per_block_quant_baseline_rocm(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    assert result.dtype == get_fp8_dtype()
    assert scale_ub is None
    assert not is_scale_transposed
    if residual is None:
        quant, aiter_scale = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()(
            input, weight, epsilon, group_size
        )
    else:
        quant, aiter_residual, aiter_scale = (
            rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()(
                input, residual, weight, epsilon, group_size
            )
        )
        residual.copy_(aiter_residual)
    result.copy_(quant)
    scale.copy_(aiter_scale)


def rms_norm_per_block_quant_reference_rocm(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    num_tokens, hidden_size = input.shape
    groups_per_row = hidden_size // group_size
    quant_dtype = result.dtype
    if quant_dtype == torch.int8:
        qtype_min, qtype_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_min, qtype_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_max * 512.0)

    x = input.to(torch.float32)
    if residual is not None:
        x = x + residual
        residual.copy_(x.to(residual.dtype))

    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    x_grouped = (x * rms * weight).view(
        num_tokens, groups_per_row, group_size
    )
    s = torch.amax(torch.abs(x_grouped), dim=-1).to(torch.float32)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = (s * (1.0 / qtype_max)).clamp(min=min_scaling_factor)

    y = x_grouped * (1.0 / s[:, :, None])
    if quant_dtype == torch.int8:
        y = y.round()

    scale.copy_(s)
    result.copy_(
        y.clamp(qtype_min, qtype_max).view(num_tokens, hidden_size).to(result.dtype)
    )


def rms_norm_per_block_quant_rocm(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(num_tokens)
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    if scale.stride(1) > 1:
        assert is_scale_transposed

    fp8_dtype = get_fp8_dtype()
    assert result.dtype in [fp8_dtype, torch.int8]
    assert result.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert result.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype
    if residual is not None:
        assert residual.dtype == input.dtype
    assert group_size in [64, 128]

    quant_dtype = result.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)
    padded_hidden_size = helion.next_power_of_2(hidden_size)
    padded_groups_per_row = padded_hidden_size // group_size
    hl.specialize(padded_hidden_size)
    hl.specialize(padded_groups_per_row)

    for tile_m in hl.tile(num_tokens):
        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        n_idx = hl.arange(padded_hidden_size)
        n_mask = (n_idx < hidden_size)[None, :]
        x_blk = hl.load(
            input, [m_idx[:, None], n_idx[None, :]], extra_mask=n_mask
        ).to(torch.float32)
        if residual is not None:
            x_blk = x_blk + hl.load(
                residual,
                [m_idx[:, None], n_idx[None, :]],
                extra_mask=n_mask,
            )
            hl.store(
                residual,
                [m_idx[:, None], n_idx[None, :]],
                x_blk.to(residual.dtype),
                extra_mask=n_mask,
            )

        sum_squares = torch.sum(x_blk * x_blk, dim=-1)
        rms = torch.rsqrt(sum_squares * (1.0 / hidden_size) + epsilon)
        x_norm_blk = x_blk * rms[:, None] * hl.load(
            weight, [n_idx[None, :]], extra_mask=n_mask
        )
        x_grouped = x_norm_blk.view(
            tile_m.block_size, padded_groups_per_row, group_size
        )
        s_blk = torch.amax(torch.abs(x_grouped), dim=-1).to(torch.float32)

        if scale_ub is not None:
            s_blk = s_blk.clamp(max=hl.load(scale_ub, []))

        s_blk = (s_blk * (1.0 / qtype_max)).clamp(min=min_scaling_factor)
        scale[tile_m, hl.arange(groups_per_row)] = s_blk

        if quant_dtype == torch.int8:
            y_blk = (x_grouped * (1.0 / s_blk[:, :, None])).round()
        else:
            y_blk = x_grouped * (1.0 / s_blk[:, :, None])
        hl.store(
            result,
            [m_idx[:, None], n_idx[None, :]],
            y_blk.clamp(qtype_traits_min, qtype_traits_max)
            .view(tile_m.block_size, padded_hidden_size)
            .to(result.dtype),
            extra_mask=n_mask,
        )
