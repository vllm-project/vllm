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

Both normalize over the last dim and broadcast ``weight`` (shape [N]) over it,
so they serve both the full-hidden norms (input/post-attn/final) and the
per-head q_norm/k_norm (N == head_dim). Inputs may be non-contiguous views
(e.g. ``qkv.split`` slices); strides are passed through and outputs are written
contiguous.
"""

import torch

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
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    eps,
    BLOCK_N: tl.constexpr,
    scale_ptr=None,
    QUANT_FP8: tl.constexpr = False,
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
    if QUANT_FP8:
        out = out.to(x_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.max(tl.where(mask, tl.abs(out), 0.0), axis=0) * (1.0 / 448.0)
        inverse = tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1",
            constraints="=v,v",
            args=[scale],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(scale_ptr + row, scale)
        out = out * inverse
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


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
    orig_shape = x.shape
    n = orig_shape[-1]
    x2 = x.reshape(-1, n)
    r2 = residual.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_fused_add_rmsnorm_kernel[(m,)](
        x2,
        r2,
        weight,
        out,
        res_out,
        n,
        x2.stride(0),
        x2.stride(1),
        r2.stride(0),
        r2.stride(1),
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return out.reshape(orig_shape), res_out.reshape(orig_shape)


def gemma_fused_add_rmsnorm_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gemma norm followed by AITER's dynamic per-token E4M3 quantization.

    The normalized values round to the input dtype before quantization,
    preserving the separate norm and quantization operations' precision.
    """
    n = x.shape[-1]
    x2, r2 = x.reshape(-1, n), residual.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    scales = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_fused_add_rmsnorm_kernel[(m,)](
        x2,
        r2,
        weight,
        out,
        res_out,
        n,
        x2.stride(0),
        x2.stride(1),
        r2.stride(0),
        r2.stride(1),
        eps,
        BLOCK_N=block_n,
        scale_ptr=scales,
        QUANT_FP8=True,
        num_warps=_num_warps(block_n),
    )
    return out.reshape(x.shape), scales, res_out.reshape(x.shape)
