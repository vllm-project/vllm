# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The chunked gated delta rule math in this file follows the flash-linear-attention
# kernels vendored in
# vllm/third_party/flash_linear_attention (MIT license, Copyright (c) 2023-2025, Songlin
# Yang, Yu Zhang).
"""Chunked GDP prefill with m delta-rule updates per real token.

The math is the chunked gated delta rule over a virtual sequence of m steps per
real token, in chunks of 64 virtual steps. Kernels read the original projections
and state cache directly, avoiding expanded input tensors. Five stages compute:

1. L2-normalized keys for virtual steps and queries for real tokens.
2. Chunk-local decay sums and (I + A)^-1, with A strictly lower triangular.
3. The WY factors w and u, reading values from convolution output rows.
4. Recurrent state across chunks, including initial and checkpoint cache states.
5. Outputs for real tokens only, into the caller's output rows.
"""

import torch

from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# CUDA limits grid axes 1 and 2 to this many blocks.
_GRID_AXIS_LIMIT = 65535
CHUNK = 64  # virtual steps per chunk (FLA_CHUNK_SIZE)
L2_ROWS = (
    32  # l2norm_fwd_kernel2's row tile; the same tile gives the same reduction order
)


@triton.jit
def _gdp_kq_kernel(
    conv_ptr,
    conv_stride,
    q_ptr,
    k_ptr,
    T,
    eps,
    n_key_progs,
    M: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BD: tl.constexpr,
    L2_ROWS: tl.constexpr,
):
    """Keys of the [T * M * G] (token, step, group) rows and queries of the [T * G]
    rows,
    L2-normalized as
    `l2norm_fwd` does (same tile and expressions).
    """
    pid = tl.program_id(0)
    VALUE_W: tl.constexpr = M * H * DV
    KEY_W: tl.constexpr = M * G * DK
    col = tl.arange(0, BD)[None, :]
    if pid < n_key_progs:
        row = pid * L2_ROWS + tl.arange(0, L2_ROWS)[:, None]
        rmask = row < T * M * G
        mask = rmask & (col < DK)
        t = row // (M * G)
        src = (
            conv_ptr
            + t.to(tl.int64) * conv_stride
            + VALUE_W
            + (row % (M * G)) * DK
            + col
        )
        xs = tl.load(src, mask, other=0.0).to(tl.float32)
        square = tl.broadcast_to(xs * xs, [L2_ROWS, BD])
        square_sum = tl.sum(tl.where(rmask, square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        tl.store(
            k_ptr + row.to(tl.int64) * DK + col,
            (xs * rsqrt).to(k_ptr.dtype.element_ty),
            mask,
        )
    else:
        row = (pid - n_key_progs) * L2_ROWS + tl.arange(0, L2_ROWS)[:, None]
        rmask = row < T * G
        mask = rmask & (col < DK)
        t = row // G
        src = (
            conv_ptr
            + t.to(tl.int64) * conv_stride
            + VALUE_W
            + KEY_W
            + (row % G) * DK
            + col
        )
        xs = tl.load(src, mask, other=0.0).to(tl.float32)
        square = tl.broadcast_to(xs * xs, [L2_ROWS, BD])
        square_sum = tl.sum(tl.where(rmask, square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        tl.store(
            q_ptr + row.to(tl.int64) * DK + col,
            (xs * rsqrt).to(q_ptr.dtype.element_ty),
            mask,
        )


@triton.jit
def _beta(b, tok, step, i_h, m_t, stride_b, H: tl.constexpr):
    """sigmoid(b) of the steps, rounded to the activation dtype as the eager op does
    (zero
    on masked steps).
    """
    x = tl.load(b + tok * stride_b + step * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    beta = tl.math.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-x))
    return tl.where(m_t, beta, 0.0).to(b.dtype.element_ty, fp_downcast_rounding="rtne")


@triton.jit
def _inverse_16(b_A, o_i):
    """Invert I + A for a strictly lower triangular 16x16 block."""
    b_Ai = -b_A
    for i in range(2, 16):
        b_a = -tl.sum(tl.where((o_i == i)[:, None], b_A, 0.0), 0)  # row i of A
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    return b_Ai + (o_i[:, None] == o_i[None, :])


@triton.jit
def _solve_rows(
    r,
    k,
    b,
    gc,
    bos,
    bos_tok,
    t0,
    T,
    i_h,
    i_g,
    stride_b,
    M: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
):
    """Rows `16 r .. 16 r + 15` of the chunk: mask, decay cumsum, keys (two halves)
    and
    beta-scaled keys.
    """
    o_t = t0 + r * 16 + tl.arange(0, 16)  # virtual step in the sequence
    m_t = o_t < T
    b_g = tl.load(gc + (bos + o_t) * H + i_h, mask=m_t, other=0.0)
    b_beta = _beta(b, bos_tok + o_t // M, o_t % M, i_h, m_t, stride_b, H)
    o_k = tl.arange(0, 32)
    p_k = k + ((bos + o_t) * G + i_g)[:, None] * K + o_k[None, :]
    b_k0 = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
    b_k1 = tl.load(p_k + 32, mask=m_t[:, None] & (o_k + 32 < K)[None, :], other=0.0)
    return o_t, m_t, b_g, b_k0, b_k1, b_k0 * b_beta[:, None], b_k1 * b_beta[:, None]


@triton.jit
def _gram(b_kb0, b_kb1, b_k0, b_k1, b_g_row, b_g_col, mask):
    """Masked Gram block: beta_i (k_i . k_j) exp(g_i - g_j)."""
    b_A = tl.zeros([16, 16], dtype=tl.float32)
    b_A += tl.dot(b_kb0, tl.trans(b_k0))
    b_A += tl.dot(b_kb1, tl.trans(b_k1))
    b_A = b_A * tl.exp(b_g_row[:, None] - b_g_col[None, :])
    return tl.where(mask, b_A, 0)


@triton.jit(do_not_specialize=["stride_b", "stride_a"])
def _gdp_solve_kernel(
    k,  # [T * M, G, K] normalized keys of the virtual steps
    b,  # [tokens, M * H] raw write strengths (step major)
    a,  # [tokens, H] raw decay inputs
    A_log,  # [H]
    dt_bias,  # [H]
    gc,  # out [T * M, H] fp32 chunk-local cumulative log decay
    Ti,  # out [T * M, H, 64]: rows of (I + A)^-1, lower 16x16 blocks only
    cu_seqlens,  # [N + 1] virtual-step offsets
    chunk_indices,  # [chunks, 2] (sequence, chunk in sequence)
    stride_b,
    stride_a,
    M: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
):
    i_c, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_g = i_h // (H // G)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int64)
    t0 = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int64) * 64
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    bos_tok = bos // M

    # chunk-local cumulative log decay: g on each token's first step, zero on the others
    o_c = t0 + tl.arange(0, 64)
    m_c = o_c < T
    x = tl.load(a + (bos_tok + o_c // M) * stride_a + i_h, mask=m_c, other=0.0).to(
        tl.float32
    )
    x = x + tl.load(dt_bias + i_h).to(tl.float32)
    softplus = tl.where(
        x > 20.0, x, tl.extra.cuda.libdevice.log1p(tl.extra.cuda.libdevice.exp(x))
    )
    b_g = -tl.extra.cuda.libdevice.exp(tl.load(A_log + i_h).to(tl.float32)) * softplus
    b_g = tl.where(m_c & (o_c % M == 0), b_g, 0.0)
    tl.store(gc + (bos + o_c) * H + i_h, tl.cumsum(b_g, axis=0), mask=m_c)
    tl.debug_barrier()

    o_i = tl.arange(0, 16)
    o_t0, m_t0, b_g0, b_k00, b_k01, b_kb00, b_kb01 = _solve_rows(
        0, k, b, gc, bos, bos_tok, t0, T, i_h, i_g, stride_b, M, H, G, K
    )
    o_t1, m_t1, b_g1, b_k10, b_k11, b_kb10, b_kb11 = _solve_rows(
        1, k, b, gc, bos, bos_tok, t0, T, i_h, i_g, stride_b, M, H, G, K
    )
    o_t2, m_t2, b_g2, b_k20, b_k21, b_kb20, b_kb21 = _solve_rows(
        2, k, b, gc, bos, bos_tok, t0, T, i_h, i_g, stride_b, M, H, G, K
    )
    o_t3, m_t3, b_g3, b_k30, b_k31, b_kb30, b_kb31 = _solve_rows(
        3, k, b, gc, bos, bos_tok, t0, T, i_h, i_g, stride_b, M, H, G, K
    )

    # strictly lower Gram blocks A[r][c], c <= r
    m_lower = o_i[:, None] > o_i[None, :]
    b_A00 = _gram(
        b_kb00,
        b_kb01,
        b_k00,
        b_k01,
        b_g0,
        b_g0,
        m_lower & (m_t0[:, None] & m_t0[None, :]),
    )
    b_A11 = _gram(
        b_kb10,
        b_kb11,
        b_k10,
        b_k11,
        b_g1,
        b_g1,
        m_lower & (m_t1[:, None] & m_t1[None, :]),
    )
    b_A22 = _gram(
        b_kb20,
        b_kb21,
        b_k20,
        b_k21,
        b_g2,
        b_g2,
        m_lower & (m_t2[:, None] & m_t2[None, :]),
    )
    b_A33 = _gram(
        b_kb30,
        b_kb31,
        b_k30,
        b_k31,
        b_g3,
        b_g3,
        m_lower & (m_t3[:, None] & m_t3[None, :]),
    )
    b_A10 = _gram(
        b_kb10, b_kb11, b_k00, b_k01, b_g1, b_g0, m_t1[:, None] & m_t0[None, :]
    )
    b_A21 = _gram(
        b_kb20, b_kb21, b_k10, b_k11, b_g2, b_g1, m_t2[:, None] & m_t1[None, :]
    )
    b_A32 = _gram(
        b_kb30, b_kb31, b_k20, b_k21, b_g3, b_g2, m_t3[:, None] & m_t2[None, :]
    )
    b_A20 = _gram(
        b_kb20, b_kb21, b_k00, b_k01, b_g2, b_g0, m_t2[:, None] & m_t0[None, :]
    )
    b_A31 = _gram(
        b_kb30, b_kb31, b_k10, b_k11, b_g3, b_g1, m_t3[:, None] & m_t1[None, :]
    )
    b_A30 = _gram(
        b_kb30, b_kb31, b_k00, b_k01, b_g3, b_g0, m_t3[:, None] & m_t0[None, :]
    )

    # (I + A)^-1 by blocks
    b_T00 = _inverse_16(b_A00, o_i)
    b_T11 = _inverse_16(b_A11, o_i)
    b_T22 = _inverse_16(b_A22, o_i)
    b_T33 = _inverse_16(b_A33, o_i)
    b_T10 = -tl.dot(
        tl.dot(b_T11, b_A10, input_precision="ieee"), b_T00, input_precision="ieee"
    )
    b_T21 = -tl.dot(
        tl.dot(b_T22, b_A21, input_precision="ieee"), b_T11, input_precision="ieee"
    )
    b_T32 = -tl.dot(
        tl.dot(b_T33, b_A32, input_precision="ieee"), b_T22, input_precision="ieee"
    )
    b_T20 = -tl.dot(
        b_T22,
        tl.dot(b_A20, b_T00, input_precision="ieee")
        + tl.dot(b_A21, b_T10, input_precision="ieee"),
        input_precision="ieee",
    )
    b_T31 = -tl.dot(
        b_T33,
        tl.dot(b_A31, b_T11, input_precision="ieee")
        + tl.dot(b_A32, b_T21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_T30 = -tl.dot(
        b_T33,
        tl.dot(b_A30, b_T00, input_precision="ieee")
        + tl.dot(b_A31, b_T10, input_precision="ieee")
        + tl.dot(b_A32, b_T20, input_precision="ieee"),
        input_precision="ieee",
    )

    dt = Ti.dtype.element_ty
    p_T = Ti + (bos * H + i_h) * 64 + o_i[None, :]
    p0 = p_T + (o_t0 * (H * 64))[:, None]
    p1 = p_T + (o_t1 * (H * 64))[:, None]
    p2 = p_T + (o_t2 * (H * 64))[:, None]
    p3 = p_T + (o_t3 * (H * 64))[:, None]
    tl.store(p0, b_T00.to(dt, fp_downcast_rounding="rtne"), mask=m_t0[:, None])
    tl.store(p1, b_T10.to(dt, fp_downcast_rounding="rtne"), mask=m_t1[:, None])
    tl.store(p1 + 16, b_T11.to(dt, fp_downcast_rounding="rtne"), mask=m_t1[:, None])
    tl.store(p2, b_T20.to(dt, fp_downcast_rounding="rtne"), mask=m_t2[:, None])
    tl.store(p2 + 16, b_T21.to(dt, fp_downcast_rounding="rtne"), mask=m_t2[:, None])
    tl.store(p2 + 32, b_T22.to(dt, fp_downcast_rounding="rtne"), mask=m_t2[:, None])
    tl.store(p3, b_T30.to(dt, fp_downcast_rounding="rtne"), mask=m_t3[:, None])
    tl.store(p3 + 16, b_T31.to(dt, fp_downcast_rounding="rtne"), mask=m_t3[:, None])
    tl.store(p3 + 32, b_T32.to(dt, fp_downcast_rounding="rtne"), mask=m_t3[:, None])
    tl.store(p3 + 48, b_T33.to(dt, fp_downcast_rounding="rtne"), mask=m_t3[:, None])


@triton.jit(do_not_specialize=["stride_conv", "stride_b"])
def _gdp_wu_kernel(
    k,  # [T * M, G, K]
    conv,  # [tokens, ...] conv output rows; values of step j at columns j * H * V ..
    b,  # [tokens, M * H]
    gc,  # [T * M, H] fp32
    Ti,  # [T * M, H, 64]
    w,  # out [T * M, H, K]
    u,  # out [T * M, H, V]
    cu_seqlens,
    chunk_indices,
    stride_conv,
    stride_b,
    M: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """WY factors: u = (I + A)^-1 (beta v), w = (I + A)^-1 (beta exp(g) k)."""
    i_c, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_g = i_h // (H // G)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int64)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int64)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    o_c = tl.arange(0, 64)
    o_t = i_t * 64 + o_c
    m_t = o_t < T
    row = bos + o_t
    tok = bos // M + o_t // M
    step = o_t % M
    b_beta = _beta(b, tok, step, i_h, m_t, stride_b, H)
    # rows of the inverse; only its lower 16x16 blocks were stored
    b_A = tl.load(
        Ti + (row * H + i_h)[:, None] * 64 + o_c[None, :],
        mask=m_t[:, None] & ((o_c[None, :] // 16) <= (o_c[:, None] // 16)),
        other=0.0,
    )
    b_g = tl.exp(tl.load(gc + row * H + i_h, mask=m_t, other=0.0))

    o_v = tl.arange(0, BV)
    m_v = m_t[:, None] & (o_v < V)[None, :]
    b_v = tl.load(
        conv + (tok * stride_conv + step * (H * V) + i_h * V)[:, None] + o_v[None, :],
        mask=m_v,
        other=0.0,
    )
    b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False)
    tl.store(
        u + (row * H + i_h)[:, None] * V + o_v[None, :],
        b_u.to(u.dtype.element_ty),
        mask=m_v,
    )

    for i_k in range(tl.cdiv(K, 32)):
        o_k = i_k * 32 + tl.arange(0, 32)
        m_k = m_t[:, None] & (o_k < K)[None, :]
        b_k = tl.load(
            k + (row * G + i_g)[:, None] * K + o_k[None, :], mask=m_k, other=0.0
        )
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(
            w + (row * H + i_h)[:, None] * K + o_k[None, :],
            b_w.to(w.dtype.element_ty),
            mask=m_k,
        )


@triton.jit(do_not_specialize=["stride_ssm"])
def _gdp_state_kernel(
    k,  # [T * M, G, K]
    w,  # [T * M, H, K]
    u,  # [T * M, H, V]
    gc,  # [T * M, H] fp32
    h,  # out [chunks, H, V, K]: state at the start of every chunk
    v_new,  # out [T * M, H, V]
    ssm,  # [slots, H, V, K] state cache
    state_indices,  # [N]
    has_initial_state,  # [N]
    ckpt_offsets,  # [N] tokens, 0 for none
    ckpt_indices,  # [N] cache slots
    cu_seqlens,  # [N + 1] virtual-step offsets
    chunk_offsets,  # [N + 1] first chunk of each sequence
    stride_ssm,
    M: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NULL_BLOCK: tl.constexpr,
    HAS_CKPT: tl.constexpr,
):
    """State recurrence of one (sequence, value head) over its chunks, states in the
    cache
    in place.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    i_g = i_h // (H // G)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    NT = tl.cdiv(T, 64)
    boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    dt = k.dtype.element_ty
    st = ssm.dtype.element_ty

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k = tl.arange(0, 64)
    m_k = o_k < K
    m_h = m_v[:, None] & m_k[None, :]
    slot = tl.load(state_indices + i_n).to(tl.int64)
    p_s = ssm + slot * stride_ssm + (i_h * V + o_v)[:, None] * K + o_k[None, :]
    use_state = tl.load(has_initial_state + i_n).to(tl.int1)
    b_h = tl.zeros([BV, 64], dtype=tl.float32)
    b_h += tl.load(p_s, mask=m_h & use_state & (slot != NULL_BLOCK), other=0.0).to(
        tl.float32
    )

    ck_chunk = NT  # never reached
    if HAS_CKPT:
        ck_offset = tl.load(ckpt_offsets + i_n).to(tl.int64)
        ck_slot = tl.load(ckpt_indices + i_n).to(tl.int64)
        if (ck_slot != NULL_BLOCK) & (ck_offset > 0):
            ck_chunk = ck_offset * M // 64
        p_ck = ssm + ck_slot * stride_ssm + (i_h * V + o_v)[:, None] * K + o_k[None, :]

    for i_t in range(NT):
        if HAS_CKPT:  # noqa: SIM102 - constexpr guards the checkpoint pointer
            if i_t == ck_chunk:
                tl.store(p_ck, b_h.to(st), mask=m_h)
        tl.store(
            h + ((boh + i_t) * H + i_h) * (V * K) + o_v[:, None] * K + o_k[None, :],
            b_h.to(dt),
            mask=m_h,
        )
        o_t = i_t * 64 + tl.arange(0, 64)
        m_t = o_t < T
        row = bos + o_t
        b_w = tl.load(
            w + (row * H + i_h)[:, None] * K + o_k[None, :],
            mask=m_t[:, None] & m_k[None, :],
            other=0.0,
        )
        b_v = tl.dot(b_w, tl.trans(b_h).to(b_w.dtype))
        p_v = (row * H + i_h)[:, None] * V + o_v[None, :]
        b_v = tl.load(u + p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v
        tl.store(
            v_new + p_v,
            b_v.to(v_new.dtype.element_ty),
            mask=m_t[:, None] & m_v[None, :],
        )

        last_idx = min((i_t + 1) * 64, T) - 1
        b_g_last = tl.load(gc + (bos + last_idx) * H + i_h)
        b_g = tl.load(gc + row * H + i_h, mask=m_t, other=0.0)
        b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = tl.exp(b_g_last)
        b_h *= b_g_last
        b_v = b_v.to(dt)
        b_k = tl.load(
            k + o_k[:, None] + ((row * G + i_g) * K)[None, :],
            mask=m_k[:, None] & m_t[None, :],
            other=0.0,
        )
        b_h += tl.trans(tl.dot(b_k, b_v))

    if (slot != NULL_BLOCK) & (T > 0):
        tl.store(p_s, b_h.to(st), mask=m_h)
    if HAS_CKPT:  # noqa: SIM102 - constexpr guards checkpoint-only values
        if (ck_chunk == NT) & (ck_offset > 0) & (ck_slot != NULL_BLOCK):
            tl.store(p_ck, b_h.to(st), mask=m_h)


@triton.jit(do_not_specialize=["stride_o"])
def _gdp_output_kernel(
    q,  # [tokens, G, K] normalized queries
    k,  # [T * M, G, K]
    v_new,  # [T * M, H, V]
    h,  # [chunks, H, V, K]
    gc,  # [T * M, H] fp32
    o,  # out [tokens, H, V] rows of stride_o
    cu_seqlens,
    chunk_indices,
    scale,
    stride_o,
    M: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BQ: tl.constexpr,
):
    """Outputs of the tokens whose last step (the one the query reads after) lies in the
    chunk.
    """
    i_v, i_c, i_h = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_g = i_h // (H // G)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int64)
    c0 = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int64) * 64
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    bos_tok = bos // M

    o_t = c0 + tl.arange(0, 64)
    m_t = o_t < T
    o_q = (
        c0 + (M - 1 - c0 % M) + M * tl.arange(0, BQ)
    )  # last steps of the tokens ending in the chunk
    m_q = (o_q < c0 + 64) & (o_q < T)
    tok = bos_tok + o_q // M
    o_k = tl.arange(0, 64)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    b_q = tl.load(
        q + (tok * G + i_g)[:, None] * K + o_k[None, :],
        mask=m_q[:, None] & m_k[None, :],
        other=0.0,
    )
    b_k = tl.load(
        k + o_k[:, None] + ((bos + o_t) * G + i_g)[None, :] * K,
        mask=m_k[:, None] & m_t[None, :],
        other=0.0,
    )
    b_h = tl.load(
        h + (i_c * H + i_h) * (V * K) + o_v[:, None] * K + o_k[None, :],
        mask=m_v[:, None] & m_k[None, :],
        other=0.0,
    )
    b_o = tl.dot(b_q, tl.trans(b_h))  # [BQ, BV]
    b_A = tl.dot(b_q, b_k)  # [BQ, 64]

    b_g = tl.load(gc + (bos + o_t) * H + i_h, mask=m_t, other=0.0)
    b_gq = tl.load(gc + (bos + o_q) * H + i_h, mask=m_q, other=0.0)
    b_o = b_o * tl.exp(b_gq)[:, None]
    b_A = b_A * tl.exp(b_gq[:, None] - b_g[None, :])
    b_A = tl.where(
        (o_q[:, None] >= o_t[None, :]) & (m_q[:, None] & m_t[None, :]), b_A, 0
    )

    b_v = tl.load(
        v_new + ((bos + o_t) * H + i_h)[:, None] * V + o_v[None, :],
        mask=m_t[:, None] & m_v[None, :],
        other=0.0,
    )
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(
        o + (tok * stride_o)[:, None] + (i_h * V + o_v)[None, :],
        b_o.to(o.dtype.element_ty),
        mask=m_q[:, None] & m_v[None, :],
    )


def gdp_chunk_prefill_supported(
    num_householder: int, head_k_dim: int, head_v_dim: int
) -> bool:
    return (
        head_k_dim <= 64
        and head_v_dim <= 32
        and num_householder >= 1
        and torch.version.hip is None
    )


def _prefill_group(
    conv_out,
    b,
    a,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    has_initial_state,
    chunk_metadata,
    m,
    H,
    G,
    K,
    V,
    checkpoint_offsets,
    checkpoint_indices,
    out,
):
    T = conv_out.shape[0]
    if T == 0:
        return
    cu_seqlens, chunk_indices, chunk_offsets = chunk_metadata
    NT = chunk_indices.shape[0]
    N = cu_seqlens.shape[0] - 1
    dev, dt = conv_out.device, conv_out.dtype
    BV = 16 if V <= 16 else 32
    k = torch.empty(T * m, G, K, device=dev, dtype=dt)
    q = torch.empty(T, G, K, device=dev, dtype=dt)
    n_key = triton.cdiv(T * m * G, L2_ROWS)
    _gdp_kq_kernel[(n_key + triton.cdiv(T * G, L2_ROWS),)](
        conv_out,
        conv_out.stride(0),
        q,
        k,
        T,
        1e-6,
        n_key,
        M=m,
        H=H,
        G=G,
        DK=K,
        DV=V,
        BD=triton.next_power_of_2(K),
        L2_ROWS=L2_ROWS,
        num_warps=4,
    )
    gc = torch.empty(T * m, H, device=dev, dtype=torch.float32)
    Ti = torch.empty(T * m, H, CHUNK, device=dev, dtype=dt)
    _gdp_solve_kernel[(NT, H)](
        k,
        b,
        a,
        A_log,
        dt_bias,
        gc,
        Ti,
        cu_seqlens,
        chunk_indices,
        b.stride(0),
        a.stride(0),
        M=m,
        H=H,
        G=G,
        K=K,
        num_warps=1,
    )
    w = torch.empty(T * m, H, K, device=dev, dtype=dt)
    u = torch.empty(T * m, H, V, device=dev, dtype=dt)
    _gdp_wu_kernel[(NT, H)](
        k,
        conv_out,
        b,
        gc,
        Ti,
        w,
        u,
        cu_seqlens,
        chunk_indices,
        conv_out.stride(0),
        b.stride(0),
        M=m,
        H=H,
        G=G,
        K=K,
        V=V,
        BV=BV,
        num_warps=2,
        num_stages=2,
    )
    h = torch.empty(NT, H, V, K, device=dev, dtype=dt)
    v_new = torch.empty(T * m, H, V, device=dev, dtype=dt)
    has_ckpt = checkpoint_offsets is not None
    _gdp_state_kernel[(triton.cdiv(V, BV), N * H)](
        k,
        w,
        u,
        gc,
        h,
        v_new,
        ssm_state,
        state_indices,
        has_initial_state,
        checkpoint_offsets if has_ckpt else state_indices,
        checkpoint_indices if has_ckpt else state_indices,
        cu_seqlens,
        chunk_offsets,
        ssm_state.stride(0),
        M=m,
        H=H,
        G=G,
        K=K,
        V=V,
        BV=BV,
        NULL_BLOCK=NULL_BLOCK_ID,
        HAS_CKPT=has_ckpt,
        num_warps=4,
        num_stages=3,
    )
    _gdp_output_kernel[(triton.cdiv(V, BV), NT, H)](
        q,
        k,
        v_new,
        h,
        gc,
        out,
        cu_seqlens,
        chunk_indices,
        K**-0.5,
        out.stride(0),
        M=m,
        H=H,
        G=G,
        K=K,
        V=V,
        BV=BV,
        BQ=max(16, triton.next_power_of_2(triton.cdiv(CHUNK, m))),
        num_warps=2,
        num_stages=3,
    )


def gdp_chunk_prefill(
    conv_out: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    query_start: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_householder: int,
    num_heads: int,
    num_groups: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked prefill of a batch of sequences.

    conv_out / b / a: the prefill rows' tokens (unit column strides, any row stride).
    query_start: [N + 1]
    real-token offsets from 0. ssm_state: [slots, heads, head_v_dim, head_k_dim]; each
    sequence's state is read
    from `state_indices` where `has_initial_state` (zeros elsewhere) and its final
    state written back in place.
    checkpoint_offsets / checkpoint_indices: [N] int32; where the index is not the null
    block and the offset (real
    tokens, a multiple of `64 / gcd(64, m)`) is > 0, the state after that many tokens
    is stored in that slot.
    chunk_metadata: (cu_seqlens, chunk_indices, chunk_offsets) of the virtual sequences
    (`virtual_chunk_metadata`), used when the batch runs as one group; built here
    otherwise.
    Returns the per-token outputs [T, heads, head_v_dim] (`out` when given: unit
    strides within a row).
    """
    m, H, G, K, V = num_householder, num_heads, num_groups, head_k_dim, head_v_dim
    assert gdp_chunk_prefill_supported(m, K, V), (m, K, V)
    assert conv_out.stride(-1) == 1 and b.stride(-1) == 1 and a.stride(-1) == 1
    # The kernels index the per-sequence tensors with unit stride; the state indices can
    # be a column of the
    # block table (stride 1 + speculative blocks).
    assert state_indices.ndim == 1 and has_initial_state.ndim == 1
    state_indices = state_indices.contiguous()
    has_initial_state = has_initial_state.contiguous()
    if checkpoint_offsets is not None:
        assert checkpoint_indices is not None
        checkpoint_offsets = checkpoint_offsets.contiguous()
        checkpoint_indices = checkpoint_indices.contiguous()
    if chunk_metadata is not None:
        chunk_metadata = tuple(x.contiguous() for x in chunk_metadata)
    assert ssm_state.shape[1:] == (H, V, K) and ssm_state[0].is_contiguous()
    T = conv_out.shape[0]
    if out is None:
        out = conv_out.new_empty(T, H, V)
    assert out.shape == (T, H, V) and (T == 0 or out[0].is_contiguous())

    def metadata(real_cu):
        cu = (real_cu * m).to(torch.int32)
        return cu, prepare_chunk_indices(cu, CHUNK), prepare_chunk_offsets(cu, CHUNK)

    num_seqs = query_start.shape[0] - 1
    # The kernels launch one block per (sequence, value head) on a grid axis limited to
    # 65535 blocks.
    max_seqs = _GRID_AXIS_LIMIT // H
    if num_seqs <= max_seqs:
        _prefill_group(
            conv_out,
            b,
            a,
            A_log,
            dt_bias,
            ssm_state,
            state_indices,
            has_initial_state,
            chunk_metadata if chunk_metadata is not None else metadata(query_start),
            m,
            H,
            G,
            K,
            V,
            checkpoint_offsets,
            checkpoint_indices,
            out,
        )
        return out
    bounds = query_start.tolist()
    for s0 in range(0, num_seqs, max_seqs):
        s1 = min(s0 + max_seqs, num_seqs)
        t0, t1 = bounds[s0], bounds[s1]
        _prefill_group(
            conv_out[t0:t1],
            b[t0:t1],
            a[t0:t1],
            A_log,
            dt_bias,
            ssm_state,
            state_indices[s0:s1],
            has_initial_state[s0:s1],
            metadata(query_start[s0 : s1 + 1] - bounds[s0]),
            m,
            H,
            G,
            K,
            V,
            None if checkpoint_offsets is None else checkpoint_offsets[s0:s1],
            None if checkpoint_indices is None else checkpoint_indices[s0:s1],
            out[t0:t1],
        )
    return out
