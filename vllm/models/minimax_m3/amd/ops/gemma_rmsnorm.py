# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Gemma-style RMSNorm for AMD ROCm via Triton.

Gemma RMSNorm = normalize(x) * (1 + weight), computed in fp32. FlashInfer's
``gemma_rmsnorm`` / ``gemma_fused_add_rmsnorm`` CUDA kernels are unavailable on
ROCm, so the AMD path previously used a ~8-op PyTorch sequence (float cast, add,
pow, mean, rsqrt, two muls, cast) — each a separate kernel launch materializing
fp32 intermediates. These kernels collapse that into a single pass per row.

Entry points:
  * ``gemma_rmsnorm(x, w, eps)``                 -> normalized tensor
  * ``gemma_fused_add_rmsnorm(x, res, w, eps)``  -> (normalized, x + res)
  * ``gemma_fused_add_rmsnorm_mxfp8(x, res, w, eps)``
        -> (MXFP8 values, E8M0 scales, x + res): the fused add + norm with the
           per-32-block MXFP8 activation quant as an epilogue, so a following
           MXFP8 linear need not quantize its input again.

Both normalize over the last dim and broadcast ``weight`` (shape [N]) over it,
so they serve both the full-hidden norms (input/post-attn/final) and the
per-head q_norm/k_norm (N == head_dim). Inputs may be non-contiguous views
(e.g. ``qkv.split`` slices); strides are passed through and outputs are written
contiguous.
"""

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_cols,
    stride_row,
    stride_col,
    eps,
    BLOCK_N: tl.constexpr,
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
    tl.store(
        out_ptr + row * n_cols + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
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
    BLOCK_N: tl.constexpr,
    QUANT_MXFP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
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
    if QUANT_MXFP8:
        # Same math as mxfp8_e4m3_quantize on the input-dtype normed row, so
        # the result is bit-identical to norm -> bf16 -> quant.
        y = tl.reshape(
            out.to(x_ptr.dtype.element_ty).to(tl.float32), (BLOCK_N // 32, 32)
        )
        amax = tl.maximum(tl.max(tl.abs(y), axis=1), TINY)
        sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        # exp2(sb - 127) is subnormal at sb == 0 and flushes to zero on CDNA, so
        # scale by the (exact, normal) reciprocal instead of dividing.
        q = y * tl.exp2(127.0 - sb)[:, None]
        tl.store(
            out_ptr + row * n_cols + cols,
            tl.reshape(q, (BLOCK_N,)).to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        blocks = tl.arange(0, BLOCK_N // 32)
        tl.store(
            scale_ptr + row * (n_cols // 32) + blocks,
            sb.to(scale_ptr.dtype.element_ty),
            mask=blocks < n_cols // 32,
        )
    else:
        tl.store(
            out_ptr + row * n_cols + cols,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _num_warps(block_n: int) -> int:
    if block_n >= 4096:
        return 16
    if block_n >= 1024:
        return 8
    return 4


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    n = orig_shape[-1]
    x2 = x.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_rmsnorm_kernel[(m,)](
        x2,
        weight,
        out,
        n,
        x2.stride(0),
        x2.stride(1),
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return out.reshape(orig_shape)


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, _, res_out = _gemma_fused_add_rmsnorm(x, residual, weight, eps, False)
    return out.reshape(x.shape), res_out


def gemma_fused_add_rmsnorm_mxfp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gemma_fused_add_rmsnorm`` with the normed output quantized to MXFP8.

    Bit-identical to ``mxfp8_e4m3_quantize(gemma_fused_add_rmsnorm(...)[0])``.

    Returns:
        ``(x_q, x_scale, residual_out)``: ``x_q`` float8_e4m3fn ``[M, N]`` and
        ``x_scale`` uint8 E8M0 ``[M, N // 32]`` (row-major, not swizzled), with
        the leading dims of ``x`` flattened into ``M``; ``residual_out`` has
        the shape of ``x``.

    """
    n = x.shape[-1]
    assert n % MXFP8_BLOCK_SIZE == 0, (
        f"MXFP8 needs the hidden size to be a multiple of {MXFP8_BLOCK_SIZE}, got {n}"
    )
    x_q, x_scale, res_out = _gemma_fused_add_rmsnorm(x, residual, weight, eps, True)
    assert x_scale is not None
    return x_q, x_scale, res_out


def _gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_mxfp8: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    orig_shape = x.shape
    n = orig_shape[-1]
    x2 = x.reshape(-1, n)
    r2 = residual.reshape(-1, n)
    m = x2.shape[0]
    out_dtype = MXFP8_VALUE_DTYPE if quant_mxfp8 else x.dtype
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    scale = (
        torch.empty(
            (m, n // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
        )
        if quant_mxfp8
        else None
    )
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
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
        BLOCK_N=block_n,
        QUANT_MXFP8=quant_mxfp8,
        FP8_MAX=float(torch.finfo(MXFP8_VALUE_DTYPE).max),
        TINY=float(torch.finfo(torch.float32).tiny),
        num_warps=_num_warps(block_n),
    )
    return out, scale, res_out.reshape(orig_shape)
