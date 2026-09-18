# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of the fused post-norm / residual add / pre-norm op."""

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

from vllm import ir
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

# One program per row with the row held in registers, so the hidden size is
# capped at the largest block the kernel is tuned for.
_MAX_HIDDEN_SIZE = 16384
_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


@triton.jit
def _rms_norm_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    weight_residual_ptr,
    out_ptr,
    residual_out_ptr,
    x_row_stride,
    residual_row_stride,
    hidden_size,
    epsilon,
    HAS_WEIGHT: tl.constexpr,
    HAS_WEIGHT_RESIDUAL: tl.constexpr,
    WEIGHT_IS_FP32: tl.constexpr,
    WEIGHT_RESIDUAL_IS_FP32: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    # rms_norm(x, weight): variance in fp32, normalized value cast to the
    # weight dtype before the multiply, result cast to the input dtype.
    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    x_norm = x * (1.0 / tl.sqrt(variance + epsilon))
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        if WEIGHT_IS_FP32:
            x_norm = x_norm * weight.to(tl.float32)
        else:
            x_norm = x_norm.to(OUT_DTYPE) * weight
    x_norm = x_norm.to(OUT_DTYPE).to(tl.float32)

    # fused_add_rms_norm(x_norm, residual, weight_residual): add in fp32,
    # residual output rounded to the input dtype, variance from the fp32 sum.
    residual = tl.load(
        residual_ptr + row * residual_row_stride + offsets, mask=mask, other=0.0
    )
    summed = x_norm + residual.to(tl.float32)
    tl.store(
        residual_out_ptr + row * residual_row_stride + offsets,
        summed.to(OUT_DTYPE),
        mask=mask,
    )
    summed = tl.where(mask, summed, 0.0)
    variance = tl.sum(summed * summed, axis=0) / hidden_size
    out = summed * (1.0 / tl.sqrt(variance + epsilon))
    if HAS_WEIGHT_RESIDUAL:
        weight_residual = tl.load(weight_residual_ptr + offsets, mask=mask, other=0.0)
        if WEIGHT_RESIDUAL_IS_FP32:
            out = out * weight_residual.to(tl.float32)
        else:
            out = out.to(OUT_DTYPE) * weight_residual
    tl.store(out_ptr + row * x_row_stride + offsets, out.to(OUT_DTYPE), mask=mask)


_TL_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


def _weight_ok(x: Tensor, weight: Tensor | None) -> bool:
    return weight is None or (
        weight.ndim == 1
        and weight.numel() == x.shape[-1]
        and weight.dtype in (x.dtype, torch.float32)
        and weight.is_contiguous()
    )


def _supports_rms_norm_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    weight_residual: Tensor | None,
    epsilon: float,
) -> bool:
    del epsilon
    return (
        x.device.type == "cuda"
        and x.dtype in _SUPPORTED_DTYPES
        and x_residual.dtype == x.dtype
        and x.ndim >= 1
        and x.shape == x_residual.shape
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and 0 < x.shape[-1] <= _MAX_HIDDEN_SIZE
        and _weight_ok(x, weight)
        and _weight_ok(x, weight_residual)
    )


@triton_op("vllm::rms_norm_add_rms_norm_triton", mutates_args=())
def _rms_norm_add_rms_norm_triton_op(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    weight_residual: Tensor | None,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    hidden_size = x.shape[-1]
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x_residual)
    num_rows = x.numel() // hidden_size
    if num_rows == 0:
        return out, residual_out

    def grid(meta):
        return (num_rows,)

    block_size = triton.next_power_of_2(hidden_size)
    # 8 warps measured best at hidden 5376 (block 8192) on B300 for both
    # decode (1-16 tokens: 2.8 us vs 3.7 us with 4 warps) and prefill sizes
    # (2048 tokens: 18.6 us vs 28.0 us); 16 warps is slower at both.
    num_warps = 8
    wrap_triton(_rms_norm_add_rms_norm_kernel)[grid](
        x,
        x_residual,
        weight if weight is not None else x,
        weight_residual if weight_residual is not None else x,
        out,
        residual_out,
        hidden_size,
        hidden_size,
        hidden_size,
        epsilon,
        HAS_WEIGHT=weight is not None,
        HAS_WEIGHT_RESIDUAL=weight_residual is not None,
        WEIGHT_IS_FP32=weight is not None and weight.dtype == torch.float32,
        WEIGHT_RESIDUAL_IS_FP32=(
            weight_residual is not None and weight_residual.dtype == torch.float32
        ),
        OUT_DTYPE=_TL_DTYPES[x.dtype],
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out, residual_out


@ir.ops.rms_norm_add_rms_norm.register_impl(
    "triton",
    supported=HAS_TRITON and current_platform.is_cuda(),
    supports_args=_supports_rms_norm_add_rms_norm,
)
def rms_norm_add_rms_norm_triton(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    weight_residual: Tensor | None,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    return _rms_norm_add_rms_norm_triton_op(
        x, x_residual, weight, weight_residual, epsilon
    )
