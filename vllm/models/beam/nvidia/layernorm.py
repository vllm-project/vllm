# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam kernels preserving FP32 normalization and affine arithmetic."""

import torch
from torch import nn

from vllm.triton_utils import tl, triton


class RMSNorm(nn.Module):
    """Parameters for normalization performed directly by the model kernels."""

    def __init__(self, hidden_size: int, has_weight: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) if has_weight else None

    def forward(self, *args, **kwargs):
        raise RuntimeError("Call the model's fused normalization kernels directly")


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    STRIDE_TOKEN: tl.constexpr,
    STRIDE_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, BLOCK)
    token = row // HEADS
    head = row % HEADS
    x = tl.load(
        X + token * STRIDE_TOKEN + head * STRIDE_HEAD + idx, idx < N, other=0
    ).to(tl.float32)
    mean = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(mean + EPS)
    if HAS_WEIGHT:
        w = tl.load(W + idx, idx < N, other=0).to(tl.float32)
        y = y * w
    tl.store(Y + row * N + idx, y, idx < N)


def beam_rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """Normalize strided token/head rows, rounding only after the affine multiply."""
    assert x.ndim in (2, 3) and x.stride(-1) == 1
    n = x.shape[-1]
    heads = x.shape[1] if x.ndim == 3 else 1
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out
    _rms_norm_kernel[(x.numel() // n,)](
        x,
        weight,
        out,
        n,
        heads,
        x.stride(0),
        x.stride(-2) if x.ndim == 3 else n,
        eps,
        weight is not None,
        triton.next_power_of_2(n),
        num_warps=4 if n <= 1024 else 8,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _boundary_math(
    x,
    r,
    POST_W,
    NEXT_W,
    col,
    dtype: tl.constexpr,
    N: tl.constexpr,
    POST_EPS: tl.constexpr,
    NEXT_EPS: tl.constexpr,
    X_SCALE,
    R_SCALE,
    POST_NORM: tl.constexpr,
    POST_WEIGHT: tl.constexpr,
    NEXT_NORM: tl.constexpr,
    NEXT_WEIGHT: tl.constexpr,
    SCALED: tl.constexpr,
):
    if POST_NORM:
        x = x * tl.rsqrt(tl.sum(x * x, 0) / N + POST_EPS)
        if POST_WEIGHT:
            w = tl.load(POST_W + col, col < N, other=0).to(tl.float32)
            x = x * w
        x = x.to(dtype).to(tl.float32)
    if SCALED:
        x = (x * X_SCALE).to(dtype).to(tl.float32)
        r = (r * R_SCALE).to(dtype).to(tl.float32)
    h = (x + r).to(dtype)
    if NEXT_NORM:
        h_fp32 = tl.where(col < N, h.to(tl.float32), 0)
        y = h_fp32 * tl.rsqrt(tl.sum(h_fp32 * h_fp32, 0) / N + NEXT_EPS)
        if NEXT_WEIGHT:
            w = tl.load(NEXT_W + col, col < N, other=0).to(tl.float32)
            y = y * w
    else:
        y = h
    return h, y


@triton.jit
def _norm_residual_norm_kernel(
    X,
    R,
    POST_W,
    NEXT_W,
    H,
    Y,
    N: tl.constexpr,
    X_STRIDE: tl.constexpr,
    R_STRIDE: tl.constexpr,
    POST_EPS: tl.constexpr,
    NEXT_EPS: tl.constexpr,
    X_SCALE,
    R_SCALE,
    POST_NORM: tl.constexpr,
    POST_WEIGHT: tl.constexpr,
    NEXT_NORM: tl.constexpr,
    NEXT_WEIGHT: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    dtype = X.dtype.element_ty
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + row * X_STRIDE + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * R_STRIDE + col, col < N, other=0).to(dtype).to(tl.float32)
    h, y = _boundary_math(
        x,
        r,
        POST_W,
        NEXT_W,
        col,
        dtype,
        N,
        POST_EPS,
        NEXT_EPS,
        X_SCALE,
        R_SCALE,
        POST_NORM,
        POST_WEIGHT,
        NEXT_NORM,
        NEXT_WEIGHT,
        SCALED,
    )
    tl.store(H + row * N + col, h, col < N)
    if NEXT_NORM:
        tl.store(Y + row * N + col, y, col < N)


def beam_norm_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float | None,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float] | None,
    *,
    launch_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuse post-norm, scaled residual add and next pre-norm, keeping each cast."""
    assert x.ndim == residual.ndim == 2 and x.shape == residual.shape
    assert x.stride(-1) == residual.stride(-1) == 1
    hidden = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    normalized = torch.empty_like(hidden) if next_eps is not None else None
    n = x.shape[-1]
    if hidden.numel():
        _norm_residual_norm_kernel[(x.shape[0],)](
            x,
            residual,
            post_weight,
            next_weight,
            hidden,
            normalized,
            n,
            x.stride(0),
            residual.stride(0),
            post_eps or 0.0,
            next_eps or 0.0,
            *(scales if scales is not None else (1.0, 1.0)),
            post_eps is not None,
            post_weight is not None,
            next_eps is not None,
            next_weight is not None,
            scales is not None,
            triton.next_power_of_2(n),
            launch_pdl,
            num_warps=4 if launch_pdl or n <= 1024 else 8,
            enable_fp_fusion=False,
            launch_pdl=launch_pdl,
        )
    return hidden, normalized
