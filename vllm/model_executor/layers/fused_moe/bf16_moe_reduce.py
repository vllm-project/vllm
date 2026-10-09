# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS FP4 finalization with the reference BF16 rounding points."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def supports_bf16_decode_fusion(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= 4
        and x.shape[1] == 2560
        and x.is_contiguous()
        and current_platform.is_device_capability(120, x.device.index)
    )


@triton.jit
def _weighted_sum_kernel(
    EXPERTS,
    WEIGHTS,
    OUT,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    APPLY_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_TOPK)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    values = tl.load(
        EXPERTS + (row * TOPK + slots[:, None]) * WIDTH + h[None, :],
        (slots[:, None] < TOPK) & (h[None, :] < WIDTH),
        other=0,
    ).to(tl.float32)
    if APPLY_WEIGHT:
        weights = tl.load(WEIGHTS + row * TOPK + slots, slots < TOPK, other=0)
        weights = weights.to(tl.bfloat16).to(tl.float32)
        # CUTLASS's current finalize rounds both weights and products to BF16.
        values = (values * weights[:, None]).to(tl.bfloat16).to(tl.float32)
    result = tl.sum(values, axis=0)
    tl.store(OUT + row * WIDTH + h, result, h < WIDTH)


def bf16_moe_weighted_sum(
    experts: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    block_h: int = 128,
) -> None:
    """Write CUTLASS's BF16 weighted expert sum directly into output.

    Inputs must be contiguous CUDA tensors: experts [M, topk, H] in token
    order after shuffle_rows, weights [M, topk], output [M, H]. This follows
    CUTLASS FP4 finalize rounding, not every MoE backend's weight semantics.
    """
    assert experts.is_cuda and experts.ndim == 3
    assert experts.dtype == output.dtype == torch.bfloat16
    assert weights.dtype in (torch.float32, torch.bfloat16)
    assert experts.device == weights.device == output.device
    assert experts.is_contiguous() and weights.is_contiguous()
    assert output.is_contiguous()
    m, topk, width = experts.shape
    assert m > 0 and 0 < topk <= 32 and width > 0
    assert weights.shape == (m, topk) and output.shape == (m, width)
    assert block_h in (128, 256, 512)
    with torch.accelerator.device_index(experts.device.index):
        _weighted_sum_kernel[(m, triton.cdiv(width, block_h))](
            experts,
            weights,
            output,
            width,
            topk,
            triton.next_power_of_2(topk),
            block_h,
            not apply_router_weight_on_input,
            num_warps=4,
            enable_fp_fusion=False,
        )
