# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm implementation of fused SiLU-and-mul per-block quantization."""

import torch

import helion.language as hl

from vllm.kernels.helion.utils import (
    get_fp8_dtype,
    get_int8_min_max,
    get_int8_min_scaling_factor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)


def silu_and_mul_per_block_quant_baseline_rocm(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    assert out.dtype == get_fp8_dtype()
    assert scale_ub is None
    assert not is_scale_transposed
    quant, aiter_scales = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()(
        input, group_size
    )
    out.copy_(quant)
    scales.copy_(aiter_scales)


def silu_and_mul_per_block_quant_rocm(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    assert input.ndim == 2
    num_tokens, two_intermediate_size = input.shape
    hl.specialize(num_tokens)
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2
    assert out.shape == (num_tokens, intermediate_size)

    fp8_dtype = get_fp8_dtype()
    assert out.dtype in [fp8_dtype, torch.int8]
    if scale_ub is not None:
        assert out.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert scales.ndim == 2 and scales.dtype == torch.float32
    assert scales.shape[0] == num_tokens
    groups_per_row = scales.shape[1]
    hl.specialize(groups_per_row)
    assert (
        intermediate_size % group_size == 0
        and intermediate_size // group_size == groups_per_row
    )
    assert group_size in [64, 128]
    hl.specialize(group_size)
    assert input.stride()[-1] == 1
    assert out.stride()[-1] == 1

    quant_dtype = out.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)
    qtype_max = float(qtype_traits_max)

    input = input.view(num_tokens, -1, group_size)
    out = out.view(num_tokens, -1, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_a_blk = input[tile_m, tile_gn, tile_n].to(torch.float32)
        x_b_blk = hl.load(
            input,
            [tile_m, tile_gn.index + groups_per_row, tile_n],
            extra_mask=(tile_gn.index + groups_per_row < 2 * groups_per_row)[
                None, :, None
            ],
        ).to(torch.float32)
        x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
        s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)

        if scale_ub is not None:
            s_blk = s_blk.clamp(max=hl.load(scale_ub, []))
        s_blk = (s_blk * (1.0 / qtype_max)).clamp(min=min_scaling_factor)

        scales[tile_m, tile_gn] = s_blk
        if quant_dtype == torch.int8:
            y_blk = (x_blk * (1.0 / s_blk[:, :, None])).round()
        else:
            y_blk = x_blk * (1.0 / s_blk[:, :, None])

        out[tile_m, tile_gn, tile_n] = y_blk.clamp(
            qtype_traits_min, qtype_traits_max
        ).to(out.dtype)
