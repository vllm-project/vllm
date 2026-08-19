# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm implementation of fused SiLU-and-mul per-block quantization."""

import helion.language as hl
import torch

from vllm.kernels.helion.utils import get_fp8_dtype
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform

silu_and_mul_per_block_quant_baseline_rocm = None
if current_platform.is_rocm():
    from vllm._aiter_ops import rocm_aiter_ops

    silu_and_mul_per_block_quant_baseline_rocm = (
        rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()
    )


def silu_and_mul_per_block_quant_rocm(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    num_tokens, two_intermediate_size = x.shape
    hl.specialize(num_tokens)
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2
    assert intermediate_size % group_size == 0
    assert group_size == 128
    hl.specialize(group_size)

    groups_per_row = intermediate_size // group_size
    hl.specialize(groups_per_row)
    out = torch.empty(
        (num_tokens, intermediate_size), device=x.device, dtype=get_fp8_dtype()
    )
    scales = torch.empty(
        (num_tokens, groups_per_row), device=x.device, dtype=torch.float32
    )

    qtype_min, qtype_max = get_fp8_min_max()
    # Match AITER's functional op: clamp the group absolute maximum before
    # converting it to a scale.  This differs from the CUDA kernel's FP8
    # minimum-scale convention and matters for very small activations.
    min_scaling_factor = 1.0e-10 / qtype_max

    x = x.view(num_tokens, -1, group_size)
    out = out.view(num_tokens, -1, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_a_blk = x[tile_m, tile_gn, tile_n].to(torch.float32)
        x_b_blk = hl.load(
            x,
            [tile_m, tile_gn.index + groups_per_row, tile_n],
            extra_mask=(tile_gn.index + groups_per_row < 2 * groups_per_row)[
                None, :, None
            ],
        ).to(torch.float32)
        x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
        s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
        s_blk = (s_blk * (1.0 / qtype_max)).clamp(min=min_scaling_factor)

        scales[tile_m, tile_gn] = s_blk
        y_blk = x_blk * (1.0 / s_blk[:, :, None])
        out[tile_m, tile_gn, tile_n] = y_blk.clamp(qtype_min, qtype_max).to(out.dtype)

    return out.view(num_tokens, intermediate_size), scales
