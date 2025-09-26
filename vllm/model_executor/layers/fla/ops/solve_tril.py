# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
from typing import Optional

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .utils import input_guard


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4, 5]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(
            tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * 16, offset),
                            (16, 16), (1, 0))
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16),
                             (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(
        tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)

    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(p_Ai,
             b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(A, Ad, Ai, cu_seqlens, chunk_indices,
                                        T, H: tl.constexpr, BT: tl.constexpr,
                                        IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(
            tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0),
                               (16, 16), (1, 0))
    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 32, 0),
                                (16, 16), (1, 0))
    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0),
                                (16, 16), (1, 0))
    p_Ai_11 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32, 0),
                                (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16),
                                (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0),
                                (16, 16), (1, 0))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision='ieee'),
                    Ai_11,
                    input_precision='ieee')
    tl.store(p_Ai_11,
             Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_22,
             Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_21,
             Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8] for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(A, Ad, Ai, cu_seqlens, chunk_indices,
                                        T, H: tl.constexpr, BT: tl.constexpr,
                                        IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(
            tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    p_A_21 = tl.make_block_ptr(A, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0),
                               (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16),
                               (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0),
                               (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32),
                               (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16),
                               (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0),
                               (16, 16), (1, 0))
    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 64, 0),
                                (16, 16), (1, 0))
    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 64 + 16, 0),
                                (16, 16), (1, 0))
    p_Ad_33 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 64 + 32, 0),
                                (16, 16), (1, 0))
    p_Ad_44 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 64 + 48, 0),
                                (16, 16), (1, 0))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)

    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_33 = tl.load(p_Ad_33, boundary_check=(0, 1)).to(tl.float32)
    Ai_44 = tl.load(p_Ad_44, boundary_check=(0, 1)).to(tl.float32)

    Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision='ieee'),
                    Ai_11,
                    input_precision='ieee')
    Ai_32 = -tl.dot(tl.dot(Ai_33, A_32, input_precision='ieee'),
                    Ai_22,
                    input_precision='ieee')
    Ai_43 = -tl.dot(tl.dot(Ai_44, A_43, input_precision='ieee'),
                    Ai_33,
                    input_precision='ieee')

    Ai_31 = -tl.dot(Ai_33,
                    tl.dot(A_31, Ai_11, input_precision='ieee') +
                    tl.dot(A_32, Ai_21, input_precision='ieee'),
                    input_precision='ieee')
    Ai_42 = -tl.dot(Ai_44,
                    tl.dot(A_42, Ai_22, input_precision='ieee') +
                    tl.dot(A_43, Ai_32, input_precision='ieee'),
                    input_precision='ieee')
    Ai_41 = -tl.dot(Ai_44,
                    tl.dot(A_41, Ai_11, input_precision='ieee') +
                    tl.dot(A_42, Ai_21, input_precision='ieee') +
                    tl.dot(A_43, Ai_31, input_precision='ieee'),
                    input_precision='ieee')

    p_Ai_11 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64, 0),
                                (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 16),
                                (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32),
                                (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 48),
                                (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0),
                                (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0),
                                (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16),
                                (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0),
                                (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16),
                                (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32),
                                (16, 16), (1, 0))
    tl.store(p_Ai_11,
             Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_22,
             Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_33,
             Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_44,
             Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_21,
             Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_31,
             Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_32,
             Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_41,
             Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_42,
             Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_43,
             Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))

    fill_zeros = tl.zeros((16, 16), dtype=tl.float32)
    p_Ai_12 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64, 16),
                                (16, 16), (1, 0))
    p_Ai_13 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64, 32),
                                (16, 16), (1, 0))
    p_Ai_14 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64, 48),
                                (16, 16), (1, 0))
    p_Ai_23 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 32),
                                (16, 16), (1, 0))
    p_Ai_24 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 48),
                                (16, 16), (1, 0))
    p_Ai_34 = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 48),
                                (16, 16), (1, 0))
    tl.store(p_Ai_12,
             fill_zeros.to(p_Ai_12.dtype.element_ty,
                           fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_13,
             fill_zeros.to(p_Ai_13.dtype.element_ty,
                           fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_14,
             fill_zeros.to(p_Ai_14.dtype.element_ty,
                           fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_23,
             fill_zeros.to(p_Ai_23.dtype.element_ty,
                           fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_24,
             fill_zeros.to(p_Ai_24.dtype.element_ty,
                           fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))
    tl.store(p_Ai_34,
             fill_zeros.to(p_Ai_34.dtype.element_ty,
                           fp_downcast_rounding="rtne"),
             boundary_check=(0, 1))


@input_guard
def solve_tril(A: torch.Tensor,
               cu_seqlens: Optional[torch.Tensor] = None,
               output_dtype: torch.dtype = torch.float) -> torch.Tensor:
    """
    Compute the inverse of the lower triangular matrix
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, K]
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    Ad = torch.empty(B,
                     T,
                     H,
                     16,
                     device=A.device,
                     dtype=torch.float if BT != 16 else output_dtype)

    chunk_indices = prepare_chunk_indices(
        cu_seqlens, 16) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 16)
    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    if BT == 16:
        return Ad

    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel
    chunk_indices = prepare_chunk_indices(
        cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    return Ai
