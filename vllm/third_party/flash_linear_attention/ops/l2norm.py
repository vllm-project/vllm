# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import WarmupChoices
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton

BT_LIST = [8, 16, 32, 64, 128]

USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))

_L2NORM_MBLOCK = 32


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["NB"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def l2norm_fwd_kernel2(
    X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, BD)[None, :]
    cmask = rindex < N
    mask = xmask & cmask
    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)


def _l2norm_block_dim(element_size: int, dim: int) -> int:
    return min(65536 // element_size, triton.next_power_of_2(dim))


def _l2norm_fwd_block_warmup_inputs(*, dtype: torch.dtype, dim: int) -> dict[str, Any]:
    block_dim = _l2norm_block_dim(dtype.itemsize, dim)
    num_rows: Any = WarmupChoices(1, 16, 2)
    x_aligned: Any = WarmupChoices(True, False)
    return dict(
        x=TritonWarmupTensor(dtype, aligned=x_aligned, shape=(num_rows, dim)),
        y=TritonWarmupTensor(dtype),
        eps=1e-6,
        block_dim=block_dim,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=l2norm_fwd_kernel2,
    warmup_inputs=_l2norm_fwd_block_warmup_inputs,
)
def l2norm_fwd_block(x, y, eps: float, block_dim: int) -> DispatchSpec:
    num_rows, dim = x.shape[0], x.shape[-1]
    return (triton.cdiv(num_rows, _L2NORM_MBLOCK),), dict(
        M=num_rows,
        N=dim,
        BD=block_dim,
        MBLOCK=_L2NORM_MBLOCK,
    )


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    BD = _l2norm_block_dim(x.element_size(), D)
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if not USE_DEFAULT_FLA_NORM:
        l2norm_fwd_block(x, y, eps, BD)
    else:
        if D <= 512:
            NB = triton.cdiv(T, 2048)

            def grid(meta):
                return (triton.cdiv(T, meta["BT"]),)

            l2norm_fwd_kernel[grid](
                x,
                y,
                eps,
                NB=NB,
                T=T,
                D=D,
                BD=BD,
            )
        else:
            l2norm_fwd_kernel1[(T,)](
                x,
                y,
                eps=eps,
                D=D,
                BD=BD,
            )

    return y.view(x_shape_og)
