# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Triple-fused (kkt ∘ solve_tril ∘ recompute_w_u) kernel.

Builds on wy_fast_fused.py by additionally absorbing the upstream
``chunk_scaled_dot_kkt_fwd`` kernel.  Saves the A writeback (~3.85 MB
at the trace shape), the A readback, plus one launch.

Inputs:
    k, v, beta, g_cumsum   — same as the unfused chain
    NO A tensor — computed in-kernel from k * β * kᵀ

Outputs:
    w, u                    — same shape and semantics as the unfused chain

The k tile is loaded twice per program (once for the kkt accumulation,
once for the w computation) because keeping K/BK tiles of size [BT, BK]
alive across both phases blows the VGPR budget.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .utils import is_navi

_DBLY_WARPS = [1, 2, 4] if is_navi else [2, 4, 8]
_DBLY_STAGES = [1, 2] if is_navi else [1, 2, 3, 4]


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in _DBLY_WARPS
        for num_stages in _DBLY_STAGES
        for BK in ([32, 64] if is_navi else [32, 64, 128])
        for BV in ([32, 64] if is_navi else [32, 64, 128])
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def kkt_solve_tril_recompute_w_u_kernel(
    k,
    v,
    beta,
    g,  # g_cumsum [B,T,H]
    w,  # output [B,T,H,K]
    u,  # output [B,T,H,V]
    A_scratch,  # fp32 scratch [B,T,H,BT] for in-kernel kkt round-trip
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,  # must be 64
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tl.static_assert(BT == 64, "doubly-fused kernel requires BT=64")

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

    # ---------- Load beta and g per [16] band ----------
    p_b0 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,)
    )
    p_b1 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,)
    )
    p_b2 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,)
    )
    p_b3 = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,)
    )
    bb0 = tl.load(p_b0, boundary_check=(0,))
    bb1 = tl.load(p_b1, boundary_check=(0,))
    bb2 = tl.load(p_b2, boundary_check=(0,))
    bb3 = tl.load(p_b3, boundary_check=(0,))

    p_g0 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,)
    )
    p_g1 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,)
    )
    p_g2 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,)
    )
    p_g3 = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,)
    )
    # Match singly-fused: keep g as bf16 through exp, no explicit fp32 cast.
    bg0 = bb0 * tl.exp(tl.load(p_g0, boundary_check=(0,)))
    bg1 = bb1 * tl.exp(tl.load(p_g1, boundary_check=(0,)))
    bg2 = bb2 * tl.exp(tl.load(p_g2, boundary_check=(0,)))
    bg3 = bb3 * tl.exp(tl.load(p_g3, boundary_check=(0,)))

    # ---------- KKT phase: single [BT, BT] matmul, exactly matching the
    # unfused chunk_scaled_dot_kkt_fwd reduction order, then write b_A to
    # an HBM scratch so the inversion code below can re-load 16×16 blocks
    # via standard make_block_ptr (Triton 3.6 has no static indexing into
    # reshaped tensors).  The scratch hits L2 since each program writes
    # then immediately re-reads its own slice. ----------
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta_full = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    k_off = (bos * Hg + i_h // (H // Hg)) * K
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + k_off, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta_full[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    # g-diff scaling and strict-lower mask, identical to the unfused kernel.
    p_g_full = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g_full = tl.load(p_g_full, boundary_check=(0,))
    b_A = b_A * exp(b_g_full[:, None] - b_g_full[None, :])
    m_A_full = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A_full, b_A, 0)

    # Store b_A to scratch as fp32 (matches solve_tril's input expectation).
    p_A_store = tl.make_block_ptr(
        A_scratch + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A_store, b_A, boundary_check=(0, 1))

    # Re-load per 16×16 block.  These hit L2 (just-written by us).
    A_blk = A_scratch + (bos * H + i_h) * BT
    p_A_11 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_A_22 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_A_33 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_A_44 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    p_A_21 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A_blk, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    b_A_11 = tl.load(p_A_11, boundary_check=(0, 1))
    b_A_22 = tl.load(p_A_22, boundary_check=(0, 1))
    b_A_33 = tl.load(p_A_33, boundary_check=(0, 1))
    b_A_44 = tl.load(p_A_44, boundary_check=(0, 1))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1))
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1))
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1))
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1))
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1))
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1))

    # ---------- Solve_tril: invert (I + A_lower) over the 4×4 grid -----------
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # Apply strict-lower mask to diagonal blocks (the original kkt kernel
    # stores zeros above the diagonal, so row-select must see zeros there).
    b_A_11 = tl.where(m_A, b_A_11, 0)
    b_A_22 = tl.where(m_A, b_A_22, 0)
    b_A_33 = tl.where(m_A, b_A_33, 0)
    b_A_44 = tl.where(m_A, b_A_44, 0)
    b_Ai_11 = -b_A_11
    b_Ai_22 = -b_A_22
    b_Ai_33 = -b_A_33
    b_Ai_44 = -b_A_44

    # Gauss-Jordan: re-load row i from scratch (matches singly-fused's HBM
    # load pattern exactly — same data path so the inversion is bit-
    # identical to the unfused chain through this point).
    A_blk = A_scratch + (bos * H + i_h) * BT
    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A_blk + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A_blk + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A_blk + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A_blk + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    # Off-diagonal Ai blocks in fp32 — matches singly-fused exactly to
    # avoid bf16 cascade rounding that previously broke end-to-end sanity.
    # tl.dot with fp32 inputs falls back to scalar fma chains on gfx11
    # (slower than bf16 mfma) but precision is preserved.
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43), b_Ai_33)
    b_Ai_31 = -tl.dot(b_Ai_33, tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21))
    b_Ai_42 = -tl.dot(b_Ai_44, tl.dot(b_A_42, b_Ai_22) + tl.dot(b_A_43, b_Ai_32))
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31),
    )

    # Cast Ai blocks to compute dtype with rtne rounding to match
    # solve_tril's HBM store exactly (fp_downcast_rounding="rtne").  Without
    # this the default truncation rounding produced enough drift in the
    # bf16 Ai values that errors compounded across 24 GDN layers and
    # broke end-to-end sanity on long-prefill inputs (e.g. the 1024×800
    # image embedding bench).
    out_dtype = w.dtype.element_ty
    b_Ai_11 = b_Ai_11.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_22 = b_Ai_22.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_33 = b_Ai_33.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_44 = b_Ai_44.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_21 = b_Ai_21.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_31 = b_Ai_31.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_32 = b_Ai_32.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_41 = b_Ai_41.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_42 = b_Ai_42.to(out_dtype, fp_downcast_rounding="rtne")
    b_Ai_43 = b_Ai_43.to(out_dtype, fp_downcast_rounding="rtne")

    # ---------- u = Ai · (β · v) ----------
    for i_v in range(tl.cdiv(V, BV)):
        pv0 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 0, i_v * BV),
            (16, BV),
            (1, 0),
        )
        pv1 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 16, i_v * BV),
            (16, BV),
            (1, 0),
        )
        pv2 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 32, i_v * BV),
            (16, BV),
            (1, 0),
        )
        pv3 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 48, i_v * BV),
            (16, BV),
            (1, 0),
        )
        bv0 = (tl.load(pv0, boundary_check=(0, 1)) * bb0[:, None]).to(out_dtype)
        bv1 = (tl.load(pv1, boundary_check=(0, 1)) * bb1[:, None]).to(out_dtype)
        bv2 = (tl.load(pv2, boundary_check=(0, 1)) * bb2[:, None]).to(out_dtype)
        bv3 = (tl.load(pv3, boundary_check=(0, 1)) * bb3[:, None]).to(out_dtype)

        u0 = tl.dot(b_Ai_11, bv0)
        u1 = tl.dot(b_Ai_21, bv0) + tl.dot(b_Ai_22, bv1)
        u2 = tl.dot(b_Ai_31, bv0) + tl.dot(b_Ai_32, bv1) + tl.dot(b_Ai_33, bv2)
        u3 = (
            tl.dot(b_Ai_41, bv0)
            + tl.dot(b_Ai_42, bv1)
            + tl.dot(b_Ai_43, bv2)
            + tl.dot(b_Ai_44, bv3)
        )

        pu0 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 0, i_v * BV),
            (16, BV),
            (1, 0),
        )
        pu1 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 16, i_v * BV),
            (16, BV),
            (1, 0),
        )
        pu2 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 32, i_v * BV),
            (16, BV),
            (1, 0),
        )
        pu3 = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + 48, i_v * BV),
            (16, BV),
            (1, 0),
        )
        tl.store(pu0, u0.to(pu0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu1, u1.to(pu1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu2, u2.to(pu2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu3, u3.to(pu3.dtype.element_ty), boundary_check=(0, 1))

    # ---------- w = Ai · (β · g · k) ----------
    # Re-loads k tiles (same as KKT phase) — keeping them in registers
    # across the entire kernel would blow VGPRs.
    for i_k in range(tl.cdiv(K, BK)):
        pk0 = tl.make_block_ptr(
            k + k_off, (T, K), (Hg * K, 1), (i_t * BT + 0, i_k * BK), (16, BK), (1, 0)
        )
        pk1 = tl.make_block_ptr(
            k + k_off, (T, K), (Hg * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        pk2 = tl.make_block_ptr(
            k + k_off, (T, K), (Hg * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        pk3 = tl.make_block_ptr(
            k + k_off, (T, K), (Hg * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )
        bk0 = (tl.load(pk0, boundary_check=(0, 1)) * bg0[:, None]).to(out_dtype)
        bk1 = (tl.load(pk1, boundary_check=(0, 1)) * bg1[:, None]).to(out_dtype)
        bk2 = (tl.load(pk2, boundary_check=(0, 1)) * bg2[:, None]).to(out_dtype)
        bk3 = (tl.load(pk3, boundary_check=(0, 1)) * bg3[:, None]).to(out_dtype)

        w0 = tl.dot(b_Ai_11, bk0)
        w1 = tl.dot(b_Ai_21, bk0) + tl.dot(b_Ai_22, bk1)
        w2 = tl.dot(b_Ai_31, bk0) + tl.dot(b_Ai_32, bk1) + tl.dot(b_Ai_33, bk2)
        w3 = (
            tl.dot(b_Ai_41, bk0)
            + tl.dot(b_Ai_42, bk1)
            + tl.dot(b_Ai_43, bk2)
            + tl.dot(b_Ai_44, bk3)
        )

        pw0 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 0, i_k * BK),
            (16, BK),
            (1, 0),
        )
        pw1 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 16, i_k * BK),
            (16, BK),
            (1, 0),
        )
        pw2 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 32, i_k * BK),
            (16, BK),
            (1, 0),
        )
        pw3 = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + 48, i_k * BK),
            (16, BK),
            (1, 0),
        )
        tl.store(pw0, w0.to(pw0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw1, w1.to(pw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw2, w2.to(pw2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw3, w3.to(pw3.dtype.element_ty), boundary_check=(0, 1))


def fused_kkt_solve_tril_recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for the kkt → solve_tril → recompute_w_u chain.

    BT must be 64.
    """
    assert BT == 64, f"doubly-fused path requires BT=64 (got {BT})"
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    # fp32 scratch for the in-kernel kkt → solve_tril round-trip.  Per-program
    # writes/reads land in L2 since the slice is small (B·T·H·BT·4 bytes,
    # ≈3.85 MB at the trace shape) and the access pattern is per-program-local.
    A_scratch = torch.empty(B, T, H, BT, dtype=torch.float32, device=k.device)
    extra = {"waves_per_eu": 2} if is_navi else {}
    kkt_solve_tril_recompute_w_u_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        g=g_cumsum,
        w=w,
        u=u,
        A_scratch=A_scratch,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        **extra,
    )
    return w, u
