# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Gemma-style RMSNorm for AMD ROCm via Triton.

Gemma RMSNorm = normalize(x) * (1 + weight), computed in fp32. FlashInfer's
``gemma_rmsnorm`` / ``gemma_fused_add_rmsnorm`` CUDA kernels are unavailable on
ROCm, so the AMD path previously used a ~8-op PyTorch sequence (float cast, add,
pow, mean, rsqrt, two muls, cast) — each a separate kernel launch materializing
fp32 intermediates. These kernels collapse that into a single pass per row.

Two entry points:
  * ``gemma_rmsnorm(x, w, eps)``                 -> normalized tensor
  * ``gemma_fused_add_rmsnorm(x, res, w, eps)``  -> (normalized, x + res)

``gemma_rmsnorm_fp8`` / ``gemma_fused_add_rmsnorm_fp8`` additionally quantize
the normalized rows to FP8 with one dynamic scale per row, for a following
per-token FP8 GEMM, so the separate activation quant kernel is skipped.

Both normalize over the last dim and broadcast ``weight`` (shape [N]) over it,
so they serve both the full-hidden norms (input/post-attn/final) and the
per-head q_norm/k_norm (N == head_dim). Inputs may be non-contiguous views
(e.g. ``qkv.split`` slices); strides are passed through and outputs are written
contiguous.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_normed(
    out, out_ptr, scale_ptr, row, cols, mask, n_cols, fp8_max, act_dtype, FP8_OUT
):
    if FP8_OUT:
        # Round to the activation dtype first, as the unfused path does.
        out = out.to(act_dtype).to(tl.float32)
        # Floor the scale so an all-zero row does not divide by zero.
        scale = tl.maximum(tl.max(tl.abs(out), axis=0) / fp8_max, 1.0 / (fp8_max * 512))
        out = tl.clamp(out * (1.0 / scale), -fp8_max, fp8_max)
        tl.store(scale_ptr + row, scale)
    tl.store(
        out_ptr + row * n_cols + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    scale_ptr,
    n_cols,
    stride_row,
    stride_col,
    eps,
    fp8_max,
    BLOCK_N: tl.constexpr,
    FP8_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_row + cols * stride_col, mask=mask, other=0.0).to(
        tl.float32
    )
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rstd * (1.0 + w)
    _store_normed(
        out,
        out_ptr,
        scale_ptr,
        row,
        cols,
        mask,
        n_cols,
        fp8_max,
        x_ptr.dtype.element_ty,
        FP8_OUT,
    )


@triton.jit
def _gemma_fused_add_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    scale_ptr,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    eps,
    fp8_max,
    BLOCK_N: tl.constexpr,
    FP8_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(
        x_ptr + row * stride_xrow + cols * stride_xcol, mask=mask, other=0.0
    ).to(tl.float32)
    r = tl.load(
        res_ptr + row * stride_rrow + cols * stride_rcol, mask=mask, other=0.0
    ).to(tl.float32)
    s = x + r
    # residual_out is the pre-norm sum (consumed by the next layer's add).
    tl.store(
        res_out_ptr + row * n_cols + cols,
        s.to(res_out_ptr.dtype.element_ty),
        mask=mask,
    )
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = s * rstd * (1.0 + w)
    _store_normed(
        out,
        out_ptr,
        scale_ptr,
        row,
        cols,
        mask,
        n_cols,
        fp8_max,
        x_ptr.dtype.element_ty,
        FP8_OUT,
    )


def _num_warps(block_n: int) -> int:
    if block_n >= 4096:
        return 16
    if block_n >= 1024:
        return 8
    return 4


def _gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None,
    fp8_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Returns (normalized or its FP8 quant, per-row scale, x + residual)."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=fp8_dtype or x.dtype, device=x.device)
    scale = None
    fp8_max = 0.0
    if fp8_dtype is not None:
        scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        fp8_max = torch.finfo(fp8_dtype).max
    block_n = triton.next_power_of_2(n)
    num_warps = _num_warps(block_n)
    if fp8_dtype is not None:
        # The extra amax reduction is cheaper across fewer warps.
        num_warps = min(num_warps, 8)
    meta = dict(BLOCK_N=block_n, FP8_OUT=fp8_dtype is not None, num_warps=num_warps)
    if residual is None:
        _gemma_rmsnorm_kernel[(m,)](
            x2, weight, out, scale, n, x2.stride(0), x2.stride(1), eps, fp8_max, **meta
        )
        return out, scale, None
    r2 = residual.reshape(-1, n)
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _gemma_fused_add_rmsnorm_kernel[(m,)](
        x2,
        r2,
        weight,
        out,
        res_out,
        scale,
        n,
        x2.stride(0),
        x2.stride(1),
        r2.stride(0),
        r2.stride(1),
        eps,
        fp8_max,
        **meta,
    )
    return out, scale, res_out


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out, _, _ = _gemma_rmsnorm(x, weight, eps, None, None)
    return out.reshape(x.shape)


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, _, res_out = _gemma_rmsnorm(x, weight, eps, residual, None)
    assert res_out is not None
    return out.reshape(x.shape), res_out.reshape(x.shape)


def gemma_rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float, fp8_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the normalized rows as FP8 [M, N] and their scales [M, 1]."""
    out, scale, _ = _gemma_rmsnorm(x, weight, eps, None, fp8_dtype)
    assert scale is not None
    return out, scale


def gemma_fused_add_rmsnorm_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (FP8 normalized rows [M, N], scales [M, 1], x + residual)."""
    out, scale, res_out = _gemma_rmsnorm(x, weight, eps, residual, fp8_dtype)
    assert scale is not None and res_out is not None
    return out, scale, res_out.reshape(x.shape)
