# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _conv_residual_norm_kernel(
    X,
    DELTA,
    BASE,
    RESIDUAL,
    WEIGHT,
    OUTPUT,
    X_STRIDE: tl.constexpr,
    DELTA_ROW_STRIDE: tl.constexpr,
    DELTA_TAP_STRIDE: tl.constexpr,
    RESIDUAL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    TAPS: tl.constexpr,
    QUERY: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    group = col // GROUP
    coefficient = tl.load(BASE + col, mask, 0).to(tl.float32) + tl.load(
        DELTA + row * DELTA_ROW_STRIDE + group, mask, 0
    ).to(tl.float32)
    value = tl.load(X + row * X_STRIDE + col, mask, 0).to(tl.float32)
    result = coefficient * value
    for tap in tl.static_range(1, TAPS):
        valid = mask & (row % QUERY >= tap)
        coefficient = tl.load(BASE + tap * WIDTH + col, valid, 0).to(
            tl.float32
        ) + tl.load(
            DELTA + row * DELTA_ROW_STRIDE + tap * DELTA_TAP_STRIDE + group, valid, 0
        ).to(tl.float32)
        value = tl.load(X + (row - tap) * X_STRIDE + col, valid, 0).to(tl.float32)
        result = tl.fma(coefficient, value, result)
    result = result.to(X.dtype.element_ty).to(tl.float32)
    residual = tl.load(RESIDUAL + row * RESIDUAL_STRIDE + col, mask, 0).to(tl.float32)
    result = (result + residual).to(RESIDUAL.dtype.element_ty).to(tl.float32)
    tl.store(RESIDUAL + row * RESIDUAL_STRIDE + col, result, mask)
    variance = tl.sum(result * result, 0) / WIDTH
    weight = tl.load(WEIGHT + col, mask, 0).to(tl.float32)
    normalized = result * tl.rsqrt(variance + EPS) * weight
    tl.store(OUTPUT + row * WIDTH + col, normalized, mask)


def conv_residual_norm(
    hidden: torch.Tensor,
    coefficients: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    query_len: int,
    group_size: int,
    *,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finish grouped convolution and update the residual before RMSNorm."""
    rows, width = hidden.shape
    if num_warps is None:
        num_warps = 16 if rows <= 256 else (8 if rows < 4096 else 4)
    output = torch.empty_like(hidden)
    if rows:
        _conv_residual_norm_kernel[(rows,)](
            hidden,
            coefficients,
            base,
            residual,
            weight,
            output,
            hidden.stride(0),
            coefficients.stride(0),
            coefficients.stride(1),
            residual.stride(0),
            width,
            group_size,
            base.shape[0],
            query_len,
            eps,
            triton.next_power_of_2(width),
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return output, residual
