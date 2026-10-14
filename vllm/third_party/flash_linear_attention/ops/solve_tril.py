# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import os

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import make_tensor_descriptor
from .utils import input_guard, is_amd, is_tma_supported

FLA_TRIL_PRECISION = os.environ.get("FLA_TRIL_PRECISION", "ieee")
ALLOWED_TRIL_PRECISIONS = ["ieee", "tf32"] if is_amd else ["ieee", "tf32", "tf32x3"]
assert FLA_TRIL_PRECISION in ALLOWED_TRIL_PRECISIONS, (
    f"FLA_TRIL_PRECISION must be one of {ALLOWED_TRIL_PRECISIONS}, but got {FLA_TRIL_PRECISION}"
)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    o_t = (i_t * 16 + o_i).to(tl.int64)
    m_t = o_t < T
    if not USE_TMA:
        p_A = A + o_t[:, None] * (H * BT) + (o_i + offset)[None, :]
        # [16, 16]
        b_A = tl.load(p_A, mask=m_t[:, None], other=0.0).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, 16], [H * 16, 1], [16, 16])
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
    b_A = -tl.where(m_A, b_A, 0)

    for i in range(2, min(16, T - i_t * 16)):
        # [16]
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = Ai + o_t[:, None] * (H * 16) + o_i[None, :]
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_t[:, None],
        )
    else:
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_t = (i_t * BT + o_i).to(tl.int64)
    if not USE_TMA:
        p_A_11 = A + o_t[:, None] * (H * BT) + o_i[None, :]
        p_A_22 = A + (o_t[:, None] + 16) * (H * BT) + (o_i[None, :] + 16)
        b_Ai_11 = tl.load(p_A_11, mask=(o_t[:, None] < T), other=0.0).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, mask=(o_t[:, None] + 16 < T), other=0.0).to(
            tl.float32
        )
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    if not USE_TMA:
        p_A_21 = A + (o_t[:, None] + 16) * (H * BT) + o_i[None, :]
        b_A_21 = tl.load(p_A_21, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = Ai + o_t[:, None] * (H * BT) + o_i[None, :]
        p_Ai_21 = Ai + (o_t[:, None] + 16) * (H * BT) + o_i[None, :]
        p_Ai_22 = Ai + (o_t[:, None] + 16) * (H * BT) + (o_i[None, :] + 16)
        tl.store(
            p_Ai_11,
            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] < T),
        )
        tl.store(
            p_Ai_22,
            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 16 < T),
        )
        tl.store(
            p_Ai_21,
            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 16 < T),
        )
    else:
        desc_o.store(
            [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_t = (i_t * BT + o_i).to(tl.int64)
    if not USE_TMA:
        p_A_11 = A + o_t[:, None] * (H * BT) + o_i[None, :]
        p_A_22 = A + (o_t[:, None] + 16) * (H * BT) + (o_i[None, :] + 16)
        p_A_33 = A + (o_t[:, None] + 32) * (H * BT) + (o_i[None, :] + 32)
        p_A_44 = A + (o_t[:, None] + 48) * (H * BT) + (o_i[None, :] + 48)
        b_Ai_11 = tl.load(p_A_11, mask=(o_t[:, None] < T), other=0.0).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, mask=(o_t[:, None] + 16 < T), other=0.0).to(
            tl.float32
        )
        b_Ai_33 = tl.load(p_A_33, mask=(o_t[:, None] + 32 < T), other=0.0).to(
            tl.float32
        )
        b_Ai_44 = tl.load(p_A_44, mask=(o_t[:, None] + 48 < T), other=0.0).to(
            tl.float32
        )
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    if not USE_TMA:
        p_A_21 = A + (o_t[:, None] + 16) * (H * BT) + o_i[None, :]
        p_A_31 = A + (o_t[:, None] + 32) * (H * BT) + o_i[None, :]
        p_A_32 = A + (o_t[:, None] + 32) * (H * BT) + (o_i[None, :] + 16)
        p_A_41 = A + (o_t[:, None] + 48) * (H * BT) + o_i[None, :]
        p_A_42 = A + (o_t[:, None] + 48) * (H * BT) + (o_i[None, :] + 16)
        p_A_43 = A + (o_t[:, None] + 48) * (H * BT) + (o_i[None, :] + 32)
        b_A_21 = tl.load(p_A_21, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)
        b_A_31 = tl.load(p_A_31, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
        b_A_32 = tl.load(p_A_32, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
        b_A_41 = tl.load(p_A_41, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
        b_A_42 = tl.load(p_A_42, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
        b_A_43 = tl.load(p_A_43, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = Ai + o_t[:, None] * (H * BT) + o_i[None, :]
        p_Ai_22 = Ai + (o_t[:, None] + 16) * (H * BT) + (o_i[None, :] + 16)
        p_Ai_33 = Ai + (o_t[:, None] + 32) * (H * BT) + (o_i[None, :] + 32)
        p_Ai_44 = Ai + (o_t[:, None] + 48) * (H * BT) + (o_i[None, :] + 48)
        p_Ai_21 = Ai + (o_t[:, None] + 16) * (H * BT) + o_i[None, :]
        p_Ai_31 = Ai + (o_t[:, None] + 32) * (H * BT) + o_i[None, :]
        p_Ai_32 = Ai + (o_t[:, None] + 32) * (H * BT) + (o_i[None, :] + 16)
        p_Ai_41 = Ai + (o_t[:, None] + 48) * (H * BT) + o_i[None, :]
        p_Ai_42 = Ai + (o_t[:, None] + 48) * (H * BT) + (o_i[None, :] + 16)
        p_Ai_43 = Ai + (o_t[:, None] + 48) * (H * BT) + (o_i[None, :] + 32)
        tl.store(
            p_Ai_11,
            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] < T),
        )
        tl.store(
            p_Ai_22,
            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 16 < T),
        )
        tl.store(
            p_Ai_33,
            b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 32 < T),
        )
        tl.store(
            p_Ai_44,
            b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 48 < T),
        )
        tl.store(
            p_Ai_21,
            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 16 < T),
        )
        tl.store(
            p_Ai_31,
            b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 32 < T),
        )
        tl.store(
            p_Ai_32,
            b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 32 < T),
        )
        tl.store(
            p_Ai_41,
            b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 48 < T),
        )
        tl.store(
            p_Ai_42,
            b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 48 < T),
        )
        tl.store(
            p_Ai_43,
            b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=(o_t[:, None] + 48 < T),
        )
    else:
        desc_o.store(
            [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        chunk_indices (torch.Tensor):
            Pre-computed chunk indices. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel

    merge_fn[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=is_tma_supported,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return Ai
