# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm implementation of per-token group FP8 quantization."""

import torch

import helion.language as hl


def per_token_group_fp8_quant_baseline_rocm(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    assert not scale_ue8m0
    assert not dummy_is_scale_transposed
    assert not dummy_is_tma_aligned
    quant, scales = rocm_aiter_ops.group_fp8_quant(input, group_size)
    output_q.copy_(quant)
    output_s.copy_(scales)


def per_token_group_fp8_quant_rocm(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(num_tokens)
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = output_s.shape[1]
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert output_s.ndim == 2 and output_s.dtype == torch.float32

    input = input.view(num_tokens, -1, group_size)
    output_q = output_q.view(num_tokens, -1, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_blk = input[tile_m, tile_gn, tile_n]
        y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
        y_s_blk = y_s_blk / fp8_max

        if scale_ue8m0:
            y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))

        y_q_blk = torch.clamp(
            x_blk * (1.0 / y_s_blk[:, :, None]), fp8_min, fp8_max
        ).to(output_q.dtype)

        output_s[tile_m, tile_gn] = y_s_blk
        output_q[tile_m, tile_gn, tile_n] = y_q_blk
