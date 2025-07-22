# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/state-spaces/mamba/blob/60dadf2e0ee730ac337035d5533de10bc26e4847/mamba_ssm/ops/triton/layernorm_gated.py

import torch

from vllm.triton_utils import tl, triton


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row: tl.int64,
    stride_y_row: tl.int64,
    stride_z_row: tl.int64,
    M: tl.int64,  # number of rows in X
    N: tl.int64,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(x,
                    weight,
                    bias,
                    eps,
                    z=None,
                    out=None,
                    group_size=None,
                    norm_before_gate=True,
                    is_rms_norm=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M, ), dtype=torch.float32,
                       device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](x,
                                           out,
                                           weight,
                                           bias,
                                           z,
                                           mean,
                                           rstd,
                                           x.stride(0),
                                           out.stride(0),
                                           z.stride(0) if z is not None else 0,
                                           M,
                                           group_size,
                                           eps,
                                           BLOCK_N=BLOCK_N,
                                           NORM_BEFORE_GATE=norm_before_gate,
                                           IS_RMS_NORM=is_rms_norm,
                                           num_warps=num_warps)
    return out, mean, rstd


def rms_norm_gated(x,
                   weight,
                   bias,
                   z=None,
                   eps=1e-6,
                   group_size=None,
                   norm_before_gate=True):
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y, _, _ = _layer_norm_fwd(x,
                              weight,
                              bias,
                              eps,
                              z=z,
                              group_size=group_size,
                              norm_before_gate=norm_before_gate,
                              is_rms_norm=True)

    return y.reshape(x_shape_og)
