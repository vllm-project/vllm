# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2

from .kda import FusedRMSNormGated as FusedRMSNormGated


@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    H: tl.constexpr,  # number of heads
    g_stride_n,
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    p_x = x + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = residual + o_t[:, None] * D + o_d[None, :]
        b_x += tl.load(p_res, mask=m_x, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = residual_out + o_t[:, None] * D + o_d[None, :]
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), mask=m_x)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = mean + o_t
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), mask=m_t)
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = rstd + o_t
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    o_g = (o_t // H) * g_stride_n + (o_t % H) * D
    b_g = tl.load(
        g + o_g[:, None] + o_d[None, :],
        mask=(o_t[:, None] < T) & m_d[None, :],
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = y + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,  # number of columns in x
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
    H: int = 1,
    g_stride_n: int | None = None,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if g_stride_n is None:
        g_stride_n = D
    assert T % H == 0
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = (
        torch.empty((T,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if D <= 512:
        BT = 16
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            H=H,
            g_stride_n=g_stride_n,
            D=D,
            BD=BD,
            BT=BT,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            num_warps=8,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            num_warps=4,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.contiguous().reshape(-1, x.shape[-1])
    D = x.shape[-1]
    # The tiled kernel supports row-strided gates; kernel1 does not.
    if D <= 512:
        H = 1 if g.ndim == 2 else g.shape[-2]
        g = g.view(-1, H, D)
        g_stride_n = g.stride(0)
    else:
        g = g.contiguous()
        H = 1
        g_stride_n = D
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.contiguous().reshape(-1, residual.shape[-1])
    residual_dtype = (
        residual.dtype
        if residual is not None
        else (torch.float if residual_in_fp32 else None)
    )
    y, _, _, residual_out = layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        residual_dtype=residual_dtype,
        is_rms_norm=True,
        H=H,
        g_stride_n=g_stride_n,
    )
    y = y.reshape(x_shape_og)
    return y if not prenorm else (y, residual_out.reshape(x_shape_og))
