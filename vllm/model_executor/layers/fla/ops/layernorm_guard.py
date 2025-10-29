# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Tri Dao
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2024, Tri Dao.

# ruff: noqa: E501
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2

from .utils import input_guard


def rms_norm_ref(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    upcast=True,
):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        x *= z * tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        y *= z * tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)


@lru_cache
def _get_sm_count(device: torch.device) -> int:
    """Get and cache the SM count for a given device."""
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


def calc_rows_per_block(M: int, device: torch.device) -> int:
    sm_count = _get_sm_count(device)
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, 4)
    return rows_per_block


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count
    rows_per_block = calc_rows_per_block(M, x.device)
    # Update grid to use rows_per_block
    grid = (cdiv(M, rows_per_block), ngroups)
    layer_norm_fwd_kernel[grid](
        x,
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
        ROWS_PER_BLOCK=rows_per_block,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out, mean, rstd


class LayerNormFn(torch.autograd.Function):
    @input_guard
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

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
        y, mean, rstd = layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)


def layernorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    return LayerNormFn.apply(
        x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm
    )


def rmsnorm_fn(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True
):
    return LayerNormFn.apply(
        x, weight, bias, z, eps, group_size, norm_before_gate, True
    )


class LayerNormGated(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return layernorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            group_size=self.group_size,
            eps=self.eps,
            norm_before_gate=self.norm_before_gate,
        )


class RMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )
