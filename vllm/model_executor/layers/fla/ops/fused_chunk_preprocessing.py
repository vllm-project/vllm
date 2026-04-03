# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .utils import input_guard


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [1, 2, 4]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_preprocessing_kernel(
    k,
    v,
    beta,
    g,
    g_cumsum,
    w,
    u,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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
    o_1 = i_t * BT + o_i
    o_2 = i_t * BT + 16 + o_i
    o_3 = i_t * BT + 32 + o_i
    o_4 = i_t * BT + 48 + o_i
    m_1 = o_1 < T
    m_2 = o_2 < T
    m_3 = o_3 < T
    m_4 = o_4 < T

    p_g_1 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,)
    )
    p_g_2 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,)
    )
    p_g_3 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,)
    )
    p_g_4 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,)
    )
    b_g_raw_1 = tl.load(p_g_1, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_g_raw_2 = tl.load(p_g_2, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_g_raw_3 = tl.load(p_g_3, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_g_raw_4 = tl.load(p_g_4, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_g_raw_1 = tl.where(m_1, b_g_raw_1, 0.0)
    b_g_raw_2 = tl.where(m_2, b_g_raw_2, 0.0)
    b_g_raw_3 = tl.where(m_3, b_g_raw_3, 0.0)
    b_g_raw_4 = tl.where(m_4, b_g_raw_4, 0.0)

    b_g_1 = tl.cumsum(b_g_raw_1, axis=0)
    b_g_1_total = tl.sum(b_g_raw_1, axis=0)
    b_g_2 = tl.cumsum(b_g_raw_2, axis=0) + b_g_1_total
    b_g_2_total = b_g_1_total + tl.sum(b_g_raw_2, axis=0)
    b_g_3 = tl.cumsum(b_g_raw_3, axis=0) + b_g_2_total
    b_g_3_total = b_g_2_total + tl.sum(b_g_raw_3, axis=0)
    b_g_4 = tl.cumsum(b_g_raw_4, axis=0) + b_g_3_total

    p_gc_1 = tl.make_block_ptr(
        g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,)
    )
    p_gc_2 = tl.make_block_ptr(
        g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,)
    )
    p_gc_3 = tl.make_block_ptr(
        g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,)
    )
    p_gc_4 = tl.make_block_ptr(
        g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,)
    )
    tl.store(p_gc_1, b_g_1.to(p_gc_1.dtype.element_ty), boundary_check=(0,))
    tl.store(p_gc_2, b_g_2.to(p_gc_2.dtype.element_ty), boundary_check=(0,))
    tl.store(p_gc_3, b_g_3.to(p_gc_3.dtype.element_ty), boundary_check=(0,))
    tl.store(p_gc_4, b_g_4.to(p_gc_4.dtype.element_ty), boundary_check=(0,))

    b_A_11 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_22 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_33 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_44 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_21 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_31 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_32 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_41 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_42 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_43 = tl.zeros([16, 16], dtype=tl.float32)

    p_beta_1 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,)
    )
    p_beta_2 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,)
    )
    p_beta_3 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,)
    )
    p_beta_4 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,)
    )
    b_beta_1 = tl.load(p_beta_1, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_beta_2 = tl.load(p_beta_2, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_beta_3 = tl.load(p_beta_3, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_beta_4 = tl.load(p_beta_4, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_beta_1 = tl.where(m_1, b_beta_1, 0.0)
    b_beta_2 = tl.where(m_2, b_beta_2, 0.0)
    b_beta_3 = tl.where(m_3, b_beta_3, 0.0)
    b_beta_4 = tl.where(m_4, b_beta_4, 0.0)

    for i_k in range(tl.cdiv(K, BK)):
        p_k_1 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 0, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_k_2 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 16, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_k_3 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 32, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_k_4 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 48, i_k * BK),
            (16, BK),
            (1, 0),
        )
        b_k_1 = tl.load(p_k_1, boundary_check=(0, 1), padding_option="zero")
        b_k_2 = tl.load(p_k_2, boundary_check=(0, 1), padding_option="zero")
        b_k_3 = tl.load(p_k_3, boundary_check=(0, 1), padding_option="zero")
        b_k_4 = tl.load(p_k_4, boundary_check=(0, 1), padding_option="zero")
        b_k_1 = tl.where(m_1[:, None], b_k_1, 0.0)
        b_k_2 = tl.where(m_2[:, None], b_k_2, 0.0)
        b_k_3 = tl.where(m_3[:, None], b_k_3, 0.0)
        b_k_4 = tl.where(m_4[:, None], b_k_4, 0.0)

        b_kb_1 = b_k_1 * b_beta_1[:, None]
        b_kb_2 = b_k_2 * b_beta_2[:, None]
        b_kb_3 = b_k_3 * b_beta_3[:, None]
        b_kb_4 = b_k_4 * b_beta_4[:, None]

        b_A_11 += tl.dot(b_kb_1.to(b_k_1.dtype), tl.trans(b_k_1))
        b_A_22 += tl.dot(b_kb_2.to(b_k_2.dtype), tl.trans(b_k_2))
        b_A_33 += tl.dot(b_kb_3.to(b_k_3.dtype), tl.trans(b_k_3))
        b_A_44 += tl.dot(b_kb_4.to(b_k_4.dtype), tl.trans(b_k_4))
        b_A_21 += tl.dot(b_kb_2.to(b_k_2.dtype), tl.trans(b_k_1))
        b_A_31 += tl.dot(b_kb_3.to(b_k_3.dtype), tl.trans(b_k_1))
        b_A_32 += tl.dot(b_kb_3.to(b_k_3.dtype), tl.trans(b_k_2))
        b_A_41 += tl.dot(b_kb_4.to(b_k_4.dtype), tl.trans(b_k_1))
        b_A_42 += tl.dot(b_kb_4.to(b_k_4.dtype), tl.trans(b_k_2))
        b_A_43 += tl.dot(b_kb_4.to(b_k_4.dtype), tl.trans(b_k_3))

    b_A_11 = b_A_11 * exp(b_g_1[:, None] - b_g_1[None, :])
    b_A_22 = b_A_22 * exp(b_g_2[:, None] - b_g_2[None, :])
    b_A_33 = b_A_33 * exp(b_g_3[:, None] - b_g_3[None, :])
    b_A_44 = b_A_44 * exp(b_g_4[:, None] - b_g_4[None, :])
    b_A_21 = b_A_21 * exp(b_g_2[:, None] - b_g_1[None, :])
    b_A_31 = b_A_31 * exp(b_g_3[:, None] - b_g_1[None, :])
    b_A_32 = b_A_32 * exp(b_g_3[:, None] - b_g_2[None, :])
    b_A_41 = b_A_41 * exp(b_g_4[:, None] - b_g_1[None, :])
    b_A_42 = b_A_42 * exp(b_g_4[:, None] - b_g_2[None, :])
    b_A_43 = b_A_43 * exp(b_g_4[:, None] - b_g_3[None, :])

    m_lt = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_A_11 = tl.where(m_lt & (m_1[:, None] & m_1[None, :]), b_A_11, 0)
    b_A_22 = tl.where(m_lt & (m_2[:, None] & m_2[None, :]), b_A_22, 0)
    b_A_33 = tl.where(m_lt & (m_3[:, None] & m_3[None, :]), b_A_33, 0)
    b_A_44 = tl.where(m_lt & (m_4[:, None] & m_4[None, :]), b_A_44, 0)
    b_A_21 = tl.where(m_2[:, None] & m_1[None, :], b_A_21, 0)
    b_A_31 = tl.where(m_3[:, None] & m_1[None, :], b_A_31, 0)
    b_A_32 = tl.where(m_3[:, None] & m_2[None, :], b_A_32, 0)
    b_A_41 = tl.where(m_4[:, None] & m_1[None, :], b_A_41, 0)
    b_A_42 = tl.where(m_4[:, None] & m_2[None, :], b_A_42, 0)
    b_A_43 = tl.where(m_4[:, None] & m_3[None, :], b_A_43, 0)

    b_Ai_11 = -tl.where(m_lt, b_A_11, 0)
    b_Ai_22 = -tl.where(m_lt, b_A_22, 0)
    b_Ai_33 = -tl.where(m_lt, b_A_33, 0)
    b_Ai_44 = -tl.where(m_lt, b_A_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.sum(tl.where((o_i == i)[:, None], b_A_11, 0), axis=0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.sum(tl.where((o_i == i - 16)[:, None], b_A_22, 0), axis=0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.sum(tl.where((o_i == i - 32)[:, None], b_A_33, 0), axis=0)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.sum(tl.where((o_i == i - 48)[:, None], b_A_44, 0), axis=0)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)

    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43), b_Ai_33)
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21),
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22) + tl.dot(b_A_43, b_Ai_32),
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31),
    )

    b_Ai_11 = b_Ai_11.to(k.dtype.element_ty)
    b_Ai_22 = b_Ai_22.to(k.dtype.element_ty)
    b_Ai_33 = b_Ai_33.to(k.dtype.element_ty)
    b_Ai_44 = b_Ai_44.to(k.dtype.element_ty)
    b_Ai_21 = b_Ai_21.to(k.dtype.element_ty)
    b_Ai_31 = b_Ai_31.to(k.dtype.element_ty)
    b_Ai_32 = b_Ai_32.to(k.dtype.element_ty)
    b_Ai_41 = b_Ai_41.to(k.dtype.element_ty)
    b_Ai_42 = b_Ai_42.to(k.dtype.element_ty)
    b_Ai_43 = b_Ai_43.to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_v_1 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 0, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_v_2 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 16, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_v_3 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 32, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_v_4 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 48, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_u_1 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 0, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_u_2 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 16, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_u_3 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 32, i_v * BV),
            (16, BV),
            (1, 0),
        )
        p_u_4 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 48, i_v * BV),
            (16, BV),
            (1, 0),
        )
        b_v_1 = tl.load(p_v_1, boundary_check=(0, 1), padding_option="zero")
        b_v_2 = tl.load(p_v_2, boundary_check=(0, 1), padding_option="zero")
        b_v_3 = tl.load(p_v_3, boundary_check=(0, 1), padding_option="zero")
        b_v_4 = tl.load(p_v_4, boundary_check=(0, 1), padding_option="zero")

        b_vb_1 = (b_v_1 * b_beta_1[:, None]).to(b_v_1.dtype)
        b_vb_2 = (b_v_2 * b_beta_2[:, None]).to(b_v_2.dtype)
        b_vb_3 = (b_v_3 * b_beta_3[:, None]).to(b_v_3.dtype)
        b_vb_4 = (b_v_4 * b_beta_4[:, None]).to(b_v_4.dtype)

        b_u_1 = tl.dot(b_Ai_11, b_vb_1, allow_tf32=False)
        b_u_2 = tl.dot(b_Ai_21, b_vb_1, allow_tf32=False) + tl.dot(
            b_Ai_22, b_vb_2, allow_tf32=False
        )
        b_u_3 = (
            tl.dot(b_Ai_31, b_vb_1, allow_tf32=False)
            + tl.dot(b_Ai_32, b_vb_2, allow_tf32=False)
            + tl.dot(b_Ai_33, b_vb_3, allow_tf32=False)
        )
        b_u_4 = (
            tl.dot(b_Ai_41, b_vb_1, allow_tf32=False)
            + tl.dot(b_Ai_42, b_vb_2, allow_tf32=False)
            + tl.dot(b_Ai_43, b_vb_3, allow_tf32=False)
            + tl.dot(b_Ai_44, b_vb_4, allow_tf32=False)
        )
        tl.store(p_u_1, b_u_1.to(p_u_1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u_2, b_u_2.to(p_u_2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u_3, b_u_3.to(p_u_3.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u_4, b_u_4.to(p_u_4.dtype.element_ty), boundary_check=(0, 1))

    b_scale_1 = b_beta_1 * exp(b_g_1)
    b_scale_2 = b_beta_2 * exp(b_g_2)
    b_scale_3 = b_beta_3 * exp(b_g_3)
    b_scale_4 = b_beta_4 * exp(b_g_4)

    for i_k in range(tl.cdiv(K, BK)):
        p_k_1 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 0, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_k_2 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 16, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_k_3 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 32, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_k_4 = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT + 48, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_w_1 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 0, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_w_2 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 16, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_w_3 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 32, i_k * BK),
            (16, BK),
            (1, 0),
        )
        p_w_4 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 48, i_k * BK),
            (16, BK),
            (1, 0),
        )
        b_k_1 = tl.load(p_k_1, boundary_check=(0, 1), padding_option="zero")
        b_k_2 = tl.load(p_k_2, boundary_check=(0, 1), padding_option="zero")
        b_k_3 = tl.load(p_k_3, boundary_check=(0, 1), padding_option="zero")
        b_k_4 = tl.load(p_k_4, boundary_check=(0, 1), padding_option="zero")

        b_kb_1 = (b_k_1 * b_scale_1[:, None]).to(b_k_1.dtype)
        b_kb_2 = (b_k_2 * b_scale_2[:, None]).to(b_k_2.dtype)
        b_kb_3 = (b_k_3 * b_scale_3[:, None]).to(b_k_3.dtype)
        b_kb_4 = (b_k_4 * b_scale_4[:, None]).to(b_k_4.dtype)

        b_w_1 = tl.dot(b_Ai_11, b_kb_1, allow_tf32=False)
        b_w_2 = tl.dot(b_Ai_21, b_kb_1, allow_tf32=False) + tl.dot(
            b_Ai_22, b_kb_2, allow_tf32=False
        )
        b_w_3 = (
            tl.dot(b_Ai_31, b_kb_1, allow_tf32=False)
            + tl.dot(b_Ai_32, b_kb_2, allow_tf32=False)
            + tl.dot(b_Ai_33, b_kb_3, allow_tf32=False)
        )
        b_w_4 = (
            tl.dot(b_Ai_41, b_kb_1, allow_tf32=False)
            + tl.dot(b_Ai_42, b_kb_2, allow_tf32=False)
            + tl.dot(b_Ai_43, b_kb_3, allow_tf32=False)
            + tl.dot(b_Ai_44, b_kb_4, allow_tf32=False)
        )
        tl.store(p_w_1, b_w_1.to(p_w_1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w_2, b_w_2.to(p_w_2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w_3, b_w_3.to(p_w_3.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w_4, b_w_4.to(p_w_4.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def fused_chunk_preprocessing_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused preprocessing path for each chunk:
      cumsum(g) -> A = beta * K * K^T -> A^(-1) -> recompute(w, u).

    Returns:
      g_cumsum, w, u
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    V = v.shape[-1]
    BT = chunk_size
    assert BT == 64, "fused_chunk_preprocessing_fwd currently only supports chunk_size=64"
    assert k.dtype == v.dtype, "fused_chunk_preprocessing_fwd expects k and v to share dtype"
    assert v.shape[:3] == beta.shape, "v and beta must agree on [B, T, H]"
    assert g.shape == beta.shape, "g and beta must both have shape [B, T, H]"
    if cu_seqlens is not None:
        assert k.shape[0] == 1, "Only batch size 1 is supported with cu_seqlens."

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_cumsum = torch.empty_like(g, dtype=torch.float32)
    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)
    fused_chunk_preprocessing_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        g_cumsum=g_cumsum,
        w=w,
        u=u,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return g_cumsum, w, u
