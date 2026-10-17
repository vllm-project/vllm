# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental BF16 decode fusions, preserving materialized BF16 rounding."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


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
def _shared_gate_kernel(
    X,
    W,
    SHARED,
    OUT,
    WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    x = tl.load(X + row * WIDTH + k, k < WIDTH, other=0).to(tl.float32)
    w = tl.load(W + k, k < WIDTH, other=0).to(tl.float32)
    # F.linear and sigmoid each materialize a BF16 tensor in the reference.
    gate = tl.sum(x * w, 0).to(tl.bfloat16).to(tl.float32)
    scale = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    shared = tl.load(SHARED + row * WIDTH + h, h < WIDTH, other=0)
    tl.store(OUT + row * WIDTH + h, shared.to(tl.float32) * scale, h < WIDTH)


def _shared_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    shared: torch.Tensor,
    block_h: int,
) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2 and 0 < x.shape[1] <= 8192
    assert x.dtype == weight.dtype == shared.dtype == torch.bfloat16
    assert x.device == weight.device == shared.device
    assert x.is_contiguous() and weight.is_contiguous() and shared.is_contiguous()
    assert weight.shape == (1, x.shape[1]) and shared.shape == x.shape
    assert block_h in (256, 512, 1024, 4096)
    out = torch.empty_like(shared)
    with torch.accelerator.device_index(x.device.index):
        _shared_gate_kernel[(x.shape[0], triton.cdiv(x.shape[1], block_h))](
            x,
            weight,
            shared,
            out,
            x.shape[1],
            triton.next_power_of_2(x.shape[1]),
            block_h,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def _shared_gate_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    shared: torch.Tensor,
    block_h: int,
) -> torch.Tensor:
    return torch.empty_like(shared)


direct_register_custom_op(
    op_name="bf16_decode_shared_gate",
    op_func=_shared_gate,
    fake_impl=_shared_gate_fake,
)


def bf16_shared_expert_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    shared: torch.Tensor,
    block_h: int = 512,
) -> torch.Tensor:
    """Fuse a scalar BF16 projection, sigmoid, and shared-expert scaling.

    Inputs must be contiguous CUDA BF16 tensors: x/shared [M, H], weight
    [1, H]. FP32 reduction order can differ from the GEMM implementation.
    """
    return torch.ops.vllm.bf16_decode_shared_gate(x, weight, shared, block_h)


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
