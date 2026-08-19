# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm implementation of fused RMSNorm and per-block quantization."""

import helion
import helion.language as hl
import torch

from vllm.kernels.helion.utils import get_fp8_dtype
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform

rms_norm_per_block_quant_baseline_rocm = None
if current_platform.is_rocm():
    from vllm._aiter_ops import rocm_aiter_ops

    rms_norm_per_block_quant_baseline_rocm = (
        rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()
    )


def rms_norm_per_block_quant_rocm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    num_tokens, hidden_size = x.shape
    hl.specialize(num_tokens)
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = hidden_size // group_size
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0
    fp8_dtype = get_fp8_dtype()
    result = torch.empty_like(x, dtype=fp8_dtype)
    scale = torch.empty(
        (num_tokens, groups_per_row), device=x.device, dtype=torch.float32
    )
    assert x.is_contiguous()
    assert x.dtype == weight.dtype
    assert group_size == 128

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
        x_blk = hl.load(x, [m_idx[:, None], n_idx[None, :]], extra_mask=n_mask).to(
            torch.float32
        )

        sum_squares = torch.sum(x_blk * x_blk, dim=-1)
        rms = torch.rsqrt(sum_squares * (1.0 / hidden_size) + variance_epsilon)
        x_norm_blk = (
            x_blk * rms[:, None] * hl.load(weight, [n_idx[None, :]], extra_mask=n_mask)
        )
        x_grouped = x_norm_blk.view(
            tile_m.block_size, padded_groups_per_row, group_size
        )
        s_blk = torch.amax(torch.abs(x_grouped), dim=-1).to(torch.float32)

        s_blk = (s_blk * (1.0 / qtype_max)).clamp(min=min_scaling_factor)
        group_idx = hl.arange(padded_groups_per_row)
        hl.store(
            scale,
            [m_idx[:, None], group_idx[None, :]],
            s_blk,
            extra_mask=(group_idx < groups_per_row)[None, :],
        )

        y_blk = x_grouped * (1.0 / s_blk[:, :, None])
        hl.store(
            result,
            [m_idx[:, None], n_idx[None, :]],
            y_blk.clamp(qtype_traits_min, qtype_traits_max)
            .view(tile_m.block_size, padded_hidden_size)
            .to(result.dtype),
            extra_mask=n_mask,
        )

    return result, scale
