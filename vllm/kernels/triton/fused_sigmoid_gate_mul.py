# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sigmoid-gate-multiply Triton kernel: ``out = x * sigmoid(gate)``."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    scale_ptr,
    hidden,
    x_stride_t,
    gate_stride_t,
    out_stride_t,
    BLOCK: tl.constexpr,
    QUANT_FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden

    x = tl.load(x_ptr + row * x_stride_t + offs, mask=mask, other=0.0)
    g = tl.load(gate_ptr + row * gate_stride_t + offs, mask=mask, other=0.0)
    out = x.to(tl.float32) * tl.sigmoid(g.to(tl.float32))
    if QUANT_FP8:
        # Static FP8 from the rounded product, like scaled_fp8_quant.
        out = out.to(x_ptr.dtype.element_ty).to(tl.float32)
        out = tl.clamp(out * tl.math.div_rn(1.0, tl.load(scale_ptr)), -448.0, 448.0)
    tl.store(
        out_ptr + row * out_stride_t + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_sigmoid_gate_mul(
    x: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` in a single Triton kernel.

    Falls back to the eager ``x * torch.sigmoid(gate)`` sequence unless both
    inputs are same-shape, same-dtype, contiguous 2D CUDA tensors.

    Args:
        x: (num_tokens, hidden) tensor, e.g. attention output.
        gate: (num_tokens, hidden) pre-sigmoid gate.
        scale: optional static per-tensor FP8 scale. When given, the result is
            quantized to float8_e4m3fn; ineligible inputs take the eager
            sequence followed by scaled_fp8_quant.

    Returns:
        A fresh (num_tokens, hidden) tensor with x's dtype (or FP8) and device.

    """
    if (
        x.ndim != 2
        or x.shape != gate.shape
        or x.dtype != gate.dtype
        or not x.is_contiguous()
        or not gate.is_contiguous()
        or not x.is_cuda
        # The FP8 output is hardcoded to e4m3fn; ROCm GEMMs take fnuz.
        or (scale is not None and not current_platform.is_cuda())
    ):
        out = x * torch.sigmoid(gate)
        if scale is None:
            return out
        # Strided gate view (num_heads == 1 per TP rank): quantize the eager
        # result, matching the unfused chain.
        from vllm import _custom_ops as ops

        return ops.scaled_fp8_quant(out, scale, group_shape=(-1, -1))[0]

    num_tokens, hidden = x.shape
    out = torch.empty_like(x, dtype=x.dtype if scale is None else torch.float8_e4m3fn)
    if num_tokens == 0 or hidden == 0:
        return out

    if hidden <= 1024:
        block = triton.next_power_of_2(hidden)
        num_warps = 4
    elif hidden <= 4096:
        block = 1024
        num_warps = 4
    else:
        block = 2048
        num_warps = 8

    grid = (num_tokens, triton.cdiv(hidden, block))
    _fused_sigmoid_gate_mul_kernel[grid](
        x,
        gate,
        out,
        scale if scale is not None else x,
        hidden,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        BLOCK=block,
        QUANT_FP8=scale is not None,
        num_warps=num_warps,
    )
    return out
