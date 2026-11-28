# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-row FP8 quantization for the token-sharded all-gathers."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_DEQUANT_BLOCK = 8192


@triton.jit
def _quant_rows_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    x = tl.load(x_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / FP8_MAX
    tl.store(scale_ptr + t, s)
    tl.store(out_ptr + t * D + d, (x / s).to(out_ptr.dtype.element_ty), mask=dm)


@triton.jit
def _dequant_rows_kernel(
    x_ptr, scale_ptr, out_ptr, n, D: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + offs // D, mask=m, other=0.0)
    tl.store(out_ptr + offs, (x * s).to(out_ptr.dtype.element_ty), mask=m)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row of `x` with scale `max|row| / fp8_max`.

    Args:
        x: `[T, D]` contiguous rows.

    Returns:
        FP8 rows `[T, D]` and their fp32 scales `[T]`.

    """
    assert x.dim() == 2 and x.is_contiguous()
    T, D = x.shape
    dtype = current_platform.fp8_dtype()
    out = torch.empty((T, D), dtype=dtype, device=x.device)
    scale = torch.empty((T,), dtype=torch.float32, device=x.device)
    if T > 0:
        block_d = triton.next_power_of_2(D)
        _quant_rows_kernel[(T,)](
            x,
            out,
            scale,
            D=D,
            BLOCK_D=block_d,
            FP8_MAX=torch.finfo(dtype).max,
            num_warps=8 if block_d >= 4096 else 4,
        )
    return out, scale


def dequant_rows(
    x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of `quant_rows`: FP8 `[T, D]` times fp32 `scale` `[T]`."""
    assert x.dim() == 2 and x.is_contiguous() and scale.numel() == x.shape[0]
    T, D = x.shape
    out = torch.empty((T, D), dtype=dtype, device=x.device)
    n = T * D
    if n:
        _dequant_rows_kernel[(triton.cdiv(n, _DEQUANT_BLOCK),)](
            x, scale, out, n, D=D, BLOCK=_DEQUANT_BLOCK, num_warps=4
        )
    return out
