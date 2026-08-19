# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm implementation of per-token group FP8 quantization."""

import helion.language as hl
import torch

from vllm.kernels.helion.utils import get_fp8_dtype
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform

per_token_group_fp8_quant_baseline_rocm = None
if current_platform.is_rocm():
    from vllm._aiter_ops import rocm_aiter_ops

    per_token_group_fp8_quant_baseline_rocm = rocm_aiter_ops.get_group_quant_op()


def per_token_group_fp8_quant_rocm(
    x: torch.Tensor,
    group_size: int,
    transpose_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    num_tokens, hidden_size = x.shape
    hl.specialize(num_tokens)
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = hidden_size // group_size
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0
    fp8_min, fp8_max = get_fp8_min_max()
    eps = 1e-10
    output_q = torch.empty_like(x, dtype=get_fp8_dtype())
    output_s = torch.empty(
        (num_tokens, groups_per_row), device=x.device, dtype=torch.float32
    )

    x = x.view(num_tokens, -1, group_size)
    output_q = output_q.view(num_tokens, -1, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_blk = x[tile_m, tile_gn, tile_n]
        y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
        y_s_blk = y_s_blk / fp8_max

        y_q_blk = torch.clamp(x_blk * (1.0 / y_s_blk[:, :, None]), fp8_min, fp8_max).to(
            output_q.dtype
        )

        output_s[tile_m, tile_gn] = y_s_blk
        output_q[tile_m, tile_gn, tile_n] = y_q_blk

    return output_q.view(num_tokens, hidden_size), output_s
