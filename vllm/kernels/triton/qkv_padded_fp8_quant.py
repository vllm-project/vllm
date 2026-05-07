# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stride-aware FP8 quantization with head_dim padding for ViT attention.

Reads directly from non-contiguous QKV views using 3D strides and pads
head_dim to a multiple of 16 for cuDNN compatibility.
"""

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import round_up

_FP8_MIN, _FP8_MAX = get_fp8_min_max()


@triton.jit
def _quantize_pad_fp8_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_ys,
    stride_yh,
    stride_yd,
    num_heads,
    n_rows,
    n_cols,
    n_cols_padded,
    fp8_min,
    fp8_max,
    SKIP_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows
    mask_out = mask_m[:, None] & (offs_n[None, :] < n_cols_padded)
    mask_in = mask_m[:, None] & (offs_n[None, :] < n_cols)

    # Decompose flattened row into (token, head) for 3D stride indexing.
    s = offs_m // num_heads
    h = offs_m % num_heads

    x_ptrs = (
        x_ptr
        + s[:, None] * stride_xs
        + h[:, None] * stride_xh
        + offs_n[None, :] * stride_xd
    )
    x = tl.load(x_ptrs, mask=mask_in, other=0.0).to(tl.float32)
    if SKIP_SCALE:
        x_q = x
    else:
        scale = tl.load(scale_ptr)
        x_q = x / scale
    x_q = tl.clamp(x_q, fp8_min, fp8_max).to(y_ptr.dtype.element_ty)

    y_ptrs = (
        y_ptr
        + s[:, None] * stride_ys
        + h[:, None] * stride_yh
        + offs_n[None, :] * stride_yd
    )
    tl.store(y_ptrs, x_q, mask=mask_out)


def _get_fp8_pad_quant_config(padded_head_dim: int) -> tuple[int, int, int]:
    block_n = triton.next_power_of_2(padded_head_dim)
    block_n = max(16, min(block_n, 128))
    block_m = 16
    num_warps = 4
    return block_m, block_n, num_warps


def quantize_fp8_pad_head_dim_triton(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    skip_scale: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Quantize a 3D/4D tensor to FP8, padding head_dim to a multiple of 16.

    Reads directly from the input using its 3D strides, so non-contiguous
    views (e.g. Q/K/V slices from an interleaved QKV buffer) are handled
    without an extra copy.  Output is always a fresh contiguous tensor
    with shape (S, H, padded_D).
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required to quantize with head_dim padding.")

    original_shape = tensor.shape
    if tensor.dim() == 4:
        tensor = tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
    assert tensor.dim() == 3, f"Expected 3D input (S, H, D), got {tensor.dim()}D"
    S, H, D = tensor.shape
    padded_head_dim = round_up(D, 16)
    out_dtype = current_platform.fp8_dtype()
    output = torch.empty(
        (S, H, padded_head_dim),
        device=tensor.device,
        dtype=out_dtype,
    )

    scale_1d = scale.reshape(-1)
    n_rows = S * H

    if block_m is None or block_n is None or num_warps is None:
        block_m, block_n, num_warps = _get_fp8_pad_quant_config(padded_head_dim)

    grid = (
        triton.cdiv(n_rows, block_m),
        triton.cdiv(padded_head_dim, block_n),
    )

    _quantize_pad_fp8_kernel[grid](
        tensor,
        output,
        scale_1d,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        H,
        n_rows,
        D,
        padded_head_dim,
        _FP8_MIN,
        _FP8_MAX,
        SKIP_SCALE=skip_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )

    return output.view((*original_shape[:-1], padded_head_dim))


def quantize_fp8_maybe_pad_head_dim(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    fp8_quant: QuantFP8,
    skip_scale: bool = False,
) -> torch.Tensor:
    """Quantize a 3D/4D tensor to FP8, padding head_dim to a multiple of 16
    only when needed.

    Accepts (S, H, D) or (B, S, H, D) input. Uses ``fp8_quant`` (a
    :class:`QuantFP8` CustomOp) when head_dim is already aligned to 16
    (no padding); otherwise falls back to a stride-aware Triton kernel
    that pads head_dim to a multiple of 16.
    """
    head_dim = tensor.shape[-1]
    if head_dim % 16 != 0:
        return quantize_fp8_pad_head_dim_triton(tensor, scale, skip_scale=skip_scale)

    if skip_scale:
        return tensor.to(current_platform.fp8_dtype())

    # QuantFP8 expects 2D: flatten all dims except (H, D).
    orig_shape = tensor.shape
    total_tokens = tensor.numel() // (orig_shape[-1] * orig_shape[-2])
    tensor_2d = tensor.reshape(total_tokens, -1)
    fp8_tensor, _ = fp8_quant(tensor_2d, scale=scale)
    return fp8_tensor.reshape(orig_shape)
