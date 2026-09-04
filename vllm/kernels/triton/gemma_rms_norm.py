# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Gemma-style RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w).

The +1 offset and the weight multiply happen in fp32 and the result is cast
once at the end, matching ``GemmaRMSNorm.forward_native``. The vLLM C kernels
cannot be used directly because they require the weight to share the
activation dtype, and rounding ``1 + w`` to bf16 changes the output.
"""

import torch

from vllm.triton_utils import tl, triton

_MAX_BLOCK_SIZE = 65536


@triton.jit
def _gemma_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    x_row_stride,
    residual_row_stride,
    out_row_stride,
    n_cols,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)
    if HAS_RESIDUAL:
        residual_ptrs = residual_ptr + row * residual_row_stride + cols
        residual = tl.load(residual_ptrs, mask=mask, other=0.0)
        x = x + residual.to(tl.float32)
        tl.store(residual_ptrs, x.to(residual_ptr.dtype.element_ty), mask=mask)

    variance = tl.sum(x * x, axis=0) / n_cols
    x = x * tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    out = x * (weight.to(tl.float32) + 1.0)
    tl.store(
        out_ptr + row * out_row_stride + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def gemma_rms_norm_supported(hidden_size: int) -> bool:
    return triton.next_power_of_2(hidden_size) <= _MAX_BLOCK_SIZE


def gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns the normalized tensor; with ``residual`` given, ``x + residual``
    is also written back into ``residual`` (matching ``fused_add_rms_norm``)."""
    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    out = torch.empty_like(x_2d)
    n_rows = x_2d.shape[0]
    if n_rows == 0:
        return out.reshape(x.shape)

    if residual is not None:
        residual_2d = residual.view(-1, hidden_size)
        assert residual_2d.stride(-1) == 1
        residual_row_stride = residual_2d.stride(0)
    else:
        residual_2d = out
        residual_row_stride = 0

    block_size = triton.next_power_of_2(hidden_size)
    num_warps = min(max(block_size // 1024, 1), 16)
    _gemma_rms_norm_kernel[(n_rows,)](
        x_2d,
        residual_2d,
        weight,
        out,
        x_2d.stride(0),
        residual_row_stride,
        out.stride(0),
        hidden_size,
        eps,
        HAS_RESIDUAL=residual is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)
