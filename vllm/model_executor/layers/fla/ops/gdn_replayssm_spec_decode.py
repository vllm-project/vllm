# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from __future__ import annotations

import torch

from vllm.model_executor.layers.mamba.ops.replayssm_config import get_replayssm_config
from vllm.triton_utils import tl, triton


@triton.jit
def gdn_replayssm_spec_circular_kernel(
    mixed_qkv,  # [total_tokens, qkv_dim]  packed, channel-last (q|k|v)
    a,  # [total_tokens, HV]
    b,  # [total_tokens, HV]
    A_log,  # [HV] fp32
    dt_bias,  # [HV] fp32
    o,  # [total_tokens, HV, V]  preallocated output
    h0,  # [num_slots, HV, V, K]  checkpoint state (== ht, in-place)
    ht,  # [num_slots, HV, V, K]
    d_cache,  # [num_slots, HV, L, V]
    k_cache,  # [num_slots, H, L, K]
    g_cache,  # [num_slots, HV, L]  fp32
    query_start_loc,  # [B+1] int  packed cu_seqlens
    ssm_state_indices,  # [B] int  physical block per request
    write_pos,  # [num_slots] int32  block-keyed
    cache_base,  # [num_slots] int32  block-keyed circular origin
    is_flush_flags,  # [num_slots] int8  block-keyed
    scale,
    stride_mqkv_t: tl.constexpr,  # per-token stride of mixed_qkv
    stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_o_t: tl.constexpr,  # per-token stride of o (= HV*V)
    stride_state_slot: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_qsl: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_FLUSH: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    # --- per-request packed window ---
    bos = tl.load(query_start_loc + i_n * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc + (i_n + 1) * stride_qsl).to(tl.int64)
    spec_len = eos - bos  # full window length

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)

    # output pointer (packed): token (bos + o_s), value-head i_hv, dim o_v
    p_o = o + (bos + o_s[:, None]) * stride_o_t + i_hv * V + o_v[None, :]

    if IS_FLUSH:
        if state_idx <= NULL_BLOCK_ID:
            return
        b_is_flush = tl.load(is_flush_flags + state_idx) != 0
        if not b_is_flush:
            return
    else:
        if state_idx <= NULL_BLOCK_ID:
            full_mask = (o_s < spec_len)[:, None] & mask_v[None, :]
            tl.store(
                p_o,
                tl.zeros([BS, BV], dtype=tl.float32).to(p_o.dtype.element_ty),
                mask=full_mask,
            )
            return
        b_is_flush = tl.load(is_flush_flags + state_idx) != 0
        if b_is_flush:
            return

    b_write_pos = tl.load(write_pos + state_idx).to(tl.int64)
    b_cache_base = tl.load(cache_base + state_idx).to(tl.int32)

    mask_s = o_s < spec_len
    out_mask = mask_s[:, None] & mask_v[None, :]

    b_wp_i = b_write_pos.to(tl.int32)
    cache_valid = o_c < b_write_pos

    # CIRCULAR physical slots (addresses only; masks/cumsums stay logical).
    phys_c = (b_cache_base + o_c) & (MAX_CACHE_LEN - 1)  # [BC] history
    phys_spec = (b_cache_base + b_wp_i + o_s) & (MAX_CACHE_LEN - 1)  # [BS] spec

    # ------------------------------------------------------------------
    # Block 0: gates / beta / local cumsum + committed-history replay decay.
    # ------------------------------------------------------------------
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    a_s = tl.load(a + (bos + o_s) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    b_s = tl.load(b + (bos + o_s) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    x = a_s + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, tl.sigmoid(b_s), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    # committed-history replay decay from cached g (history loads -> phys_c)
    p_g_main = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c
    b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
    b_g_prefix = tl.cumsum(b_g_all, axis=0)
    b_g_total = tl.sum(b_g_all, axis=0)
    b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
    b_total_decay = tl.exp(b_g_total)

    p_d_main = d_cache + (
        state_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_c[None, :]) * V
        + o_v[:, None]
    )
    b_d_all = tl.load(
        p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0.0
    ).to(tl.float32)
    b_d_scaled = (b_d_all * b_replay_decay[None, :]).to(mixed_qkv.dtype.element_ty)

    if USE_QK_L2NORM_IN_KERNEL:
        qnorm_acc = tl.zeros([BS], dtype=tl.float32)
        knorm_acc = tl.zeros([BS], dtype=tl.float32)
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            ld = mask_s[:, None] & mask_kt[None, :]
            qn = tl.load(
                mixed_qkv
                + (bos + o_s[:, None]) * stride_mqkv_t
                + i_h * K
                + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            knn = tl.load(
                mixed_qkv
                + (bos + o_s[:, None]) * stride_mqkv_t
                + H * K
                + i_h * K
                + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            qnorm_acc += tl.sum(qn * qn, axis=1)
            knorm_acc += tl.sum(knn * knn, axis=1)
        q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-6), 0.0)
        k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-6), 0.0)
    else:
        q_rnorm = tl.where(mask_s, 1.0, 0.0)
        k_rnorm = tl.where(mask_s, 1.0, 0.0)

    # ------------------------------------------------------------------
    # K-Tiled Fused Projection and Intra-Spec Matrices (+ flush)
    # ------------------------------------------------------------------
    hw_q = tl.zeros([BV, BS], dtype=tl.float32)
    hw_k = tl.zeros([BV, BS], dtype=tl.float32)
    if not IS_FLUSH:
        scores_q = tl.zeros([BC, BS], dtype=tl.float32)
        scores_k = tl.zeros([BC, BS], dtype=tl.float32)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)

    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))

    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        ld_s = mask_s[:, None] & mask_kt[None, :]
        q_tile = tl.load(
            mixed_qkv + (bos + o_s[:, None]) * stride_mqkv_t + i_h * K + o_kt[None, :],
            mask=ld_s,
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            mixed_qkv
            + (bos + o_s[:, None]) * stride_mqkv_t
            + H * K
            + i_h * K
            + o_kt[None, :],
            mask=ld_s,
            other=0.0,
        ).to(tl.float32)
        q_tile = (q_tile * (q_rnorm * scale)[:, None]).to(mixed_qkv.dtype.element_ty)
        k_tile = (k_tile * k_rnorm[:, None]).to(mixed_qkv.dtype.element_ty)

        p_h0 = (
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        sc_tile = tl.load(
            p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0
        ).to(mixed_qkv.dtype.element_ty)
        # cached-key history load -> phys_c
        p_k = k_cache + (
            state_idx * stride_k_slot
            + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
            + o_kt[None, :]
        )
        khist_tile = tl.load(
            p_k, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
        ).to(mixed_qkv.dtype.element_ty)

        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT)
        kq_mat += tl.dot(k_tile, qT)

        if IS_FLUSH:
            sw_f = tl.dot(b_d_scaled, khist_tile, acc=b_total_decay * sc_tile.to(tl.float32))
            sw_tile = sw_f.to(mixed_qkv.dtype.element_ty)
            hw_q += tl.dot(sw_tile, qT)
            hw_k += tl.dot(sw_tile, kT)
            p_ht = (
                ht
                + state_idx * stride_state_slot
                + i_hv * V * K
                + o_v[:, None] * K
                + o_kt[None, :]
            )
            tl.store(p_ht, sw_tile, mask=mask_v[:, None] & mask_kt[None, :])
        else:
            hw_q += tl.dot(sc_tile, qT)
            hw_k += tl.dot(sc_tile, kT)
            scores_q += tl.dot(khist_tile, qT)
            scores_k += tl.dot(khist_tile, kT)

        if write_k:
            # spec key store -> phys_spec (circular)
            p_cur_k = k_cache + (
                state_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_kt[None, :]
            )
            tl.store(
                p_cur_k,
                k_tile,
                mask=mask_s[:, None]
                & mask_kt[None, :]
                & ((b_write_pos + o_s[:, None]) < MAX_CACHE_LEN),
            )

    if not IS_FLUSH:
        hw_q = b_total_decay * hw_q + tl.dot(b_d_scaled, scores_q.to(b_d_scaled.dtype))
        hw_k = b_total_decay * hw_k + tl.dot(b_d_scaled, scores_k.to(b_d_scaled.dtype))

    # ------------------------------------------------------------------
    # strictly-lower A and T = (I + A)^{-1}.
    # ------------------------------------------------------------------
    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)

    b_Ai = -A_mat
    for ii in range(2, BS):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)

    # ------------------------------------------------------------------
    # R and D_spec = R @ T^T.
    # ------------------------------------------------------------------
    p_v = (
        mixed_qkv
        + (bos + o_s[None, :]) * stride_mqkv_t
        + 2 * H * K
        + i_hv * V
        + o_v[:, None]
    )
    v_tile = tl.load(p_v, mask=mask_v[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    D_spec = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]

    # ------------------------------------------------------------------
    # outputs.
    # ------------------------------------------------------------------
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF

    tl.store(p_o, tl.trans(O_tile).to(p_o.dtype.element_ty), mask=out_mask)

    # ------------------------------------------------------------------
    # write speculative d / g at circular positions (phys_spec).
    # ------------------------------------------------------------------
    spec_pos = b_write_pos + o_s
    spec_store_mask = mask_s & (spec_pos < MAX_CACHE_LEN)
    p_cur_d = d_cache + (
        state_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    tl.store(
        p_cur_d,
        D_spec.to(p_cur_d.dtype.element_ty),
        mask=mask_v[:, None] & spec_store_mask[None, :],
    )
    if i_v == 0:
        p_cur_g = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
        tl.store(p_cur_g, g_s, mask=spec_store_mask)


@triton.jit
def _advance_gdn_spec_cursors_kernel(
    write_pos_ptr,
    cache_base_ptr,
    is_flush_ptr,
    num_accepted_ptr,
    state_batch_indices_ptr,
    n_rows,
    stride_sbi: tl.constexpr,
    stride_na: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    row_mask = offs < n_rows
    blk = tl.load(
        state_batch_indices_ptr + offs * stride_sbi, mask=row_mask, other=NULL_BLOCK_ID
    ).to(tl.int64)
    valid = row_mask & (blk > NULL_BLOCK_ID)

    write_pos = tl.load(write_pos_ptr + blk, mask=valid, other=0).to(tl.int32)
    cache_base = tl.load(cache_base_ptr + blk, mask=valid, other=0).to(tl.int32)
    is_flush_cur = tl.load(is_flush_ptr + blk, mask=valid, other=0).to(tl.int32)
    num_acc = tl.load(
        num_accepted_ptr + offs * stride_na, mask=valid, other=0
    ).to(tl.int32)

    total_commit = num_acc
    flush_now = (total_commit > 0) & (is_flush_cur != 0)

    new_base = tl.where(
        flush_now, (cache_base + write_pos) & (CACHE_BUF_LEN - 1), cache_base
    )
    new_wp = tl.where(is_flush_cur != 0, total_commit, write_pos + total_commit).to(
        tl.int32
    )
    # Early-flush one window early so every verify step satisfies
    # write_pos + spec_len <= max_cache_len (the spec window never overflows).
    next_is_flush = ((new_wp + 2 * MAX_SPEC_LEN) > MAX_CACHE_LEN).to(tl.int8)

    tl.store(write_pos_ptr + blk, new_wp, mask=valid)
    tl.store(cache_base_ptr + blk, new_base, mask=valid)
    tl.store(is_flush_ptr + blk, next_is_flush, mask=valid)


@triton.jit
def _reset_gdn_replayssm_spec_cursors_kernel(
    write_pos_ptr,
    cache_base_ptr,
    is_flush_ptr,
    do_reset_ptr,
    state_batch_indices_ptr,
    n_rows,
    stride_sbi: tl.constexpr,
    stride_reset: tl.constexpr,
    INIT_FLUSH: tl.constexpr,
    BLOCK: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    row_mask = offs < n_rows
    blk = tl.load(
        state_batch_indices_ptr + offs * stride_sbi, mask=row_mask, other=NULL_BLOCK_ID
    ).to(tl.int64)
    do_reset = tl.load(
        do_reset_ptr + offs * stride_reset, mask=row_mask, other=0
    ).to(tl.int32)
    do = row_mask & (blk > NULL_BLOCK_ID) & (do_reset != 0)

    tl.store(write_pos_ptr + blk, tl.zeros_like(blk).to(tl.int32), mask=do)
    tl.store(cache_base_ptr + blk, tl.zeros_like(blk).to(tl.int32), mask=do)
    tl.store(
        is_flush_ptr + blk,
        tl.full([BLOCK], INIT_FLUSH, dtype=tl.int8),
        mask=do,
    )


# ---------------------------------------------------------------------------
# Python wrappers.
# ---------------------------------------------------------------------------
def _launch_gdn_spec(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    out,
    checkpoint_state,
    d_cache,
    k_cache,
    g_cache,
    query_start_loc,
    ssm_state_indices,
    write_pos,
    cache_base,
    is_flush,
    scale,
    max_cache_len,
    max_spec_len,
    use_qk_l2norm_in_kernel,
    is_flush_kernel,
    block_v,
    num_warps,
    num_stages,
    nk,
    null_block_id,
):
    num_slots, HV, V, K = checkpoint_state.shape
    qkv_dim = mixed_qkv.shape[1]
    q_dim = (qkv_dim - HV * V) // 2
    H = q_dim // K
    B = query_start_loc.shape[0] - 1
    # max_cache_len is the logical flush threshold L; the physical pow2 ring is
    # d_cache.shape[2] = next_pow2(L) and wraps addresses / per-head strides.
    buf = d_cache.shape[2]
    assert buf & (buf - 1) == 0, "circular cache requires power-of-two buffer"

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"only NK_global=1 supported (K={K}, BK={BK}).")
    if BK % nk != 0:
        raise ValueError(f"nk={nk} must divide BK={BK}.")
    BKT = BK // nk
    if BKT < 16:
        raise ValueError(f"BKT={BKT} must be >=16 for tl.dot.")
    BV = block_v if block_v is not None else min(triton.next_power_of_2(V), 64)
    BS = max(4, triton.next_power_of_2(max_spec_len))
    # History block decoupled from the physical buffer: with L = B + max_spec_len
    # committed history never exceeds L - max_spec_len, so BC covers it while
    # staying small (the L=B+T win).
    BC = max(16, triton.next_power_of_2(max(1, max_cache_len - max_spec_len)))

    grid = (triton.cdiv(V, BV), B, HV)
    gdn_replayssm_spec_circular_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        checkpoint_state,
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        query_start_loc,
        ssm_state_indices,
        write_pos,
        cache_base,
        is_flush,
        scale,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        checkpoint_state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        query_start_loc.stride(0),
        ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BS=BS,
        BC=BC,
        NK=nk,
        BKT=BKT,
        MAX_CACHE_LEN=buf,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_FLUSH=is_flush_kernel,
        NULL_BLOCK_ID=null_block_id,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def gdn_replayssm_spec_decode(
    mixed_qkv: torch.Tensor,  # [total_tokens, qkv_dim]  post-conv packed (q|k|v)
    a: torch.Tensor,  # [total_tokens, HV]
    b: torch.Tensor,  # [total_tokens, HV]
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32
    checkpoint_state: torch.Tensor,  # [num_slots, HV, V, K]  (in-place h0==ht)
    d_cache: torch.Tensor,  # [num_slots, HV, buf, V]
    k_cache: torch.Tensor,  # [num_slots, H, buf, K]
    g_cache: torch.Tensor,  # [num_slots, HV, buf]  fp32
    out: torch.Tensor,  # [total_tokens, HV, V]  preallocated
    query_start_loc: torch.Tensor,  # [B+1] int
    ssm_state_indices: torch.Tensor,  # [B] int  physical block per request
    write_pos: torch.Tensor,  # [num_slots] int32  block-keyed
    cache_base: torch.Tensor,  # [num_slots] int32  block-keyed
    is_flush: torch.Tensor,  # [num_slots] int8  block-keyed
    max_cache_len: int,  # logical flush threshold L = B + max_spec_len
    max_spec_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    launch_mode: str = "both",
):
    """GDN cached speculative-decode on a CIRCULAR d/k/g cache (vLLM packed varlen).

    ``max_cache_len`` is the logical flush threshold L = B + max_spec_len; the
    physical pow2 buffer is ``d_cache.shape[2]`` = ``next_pow2(L)`` and the history
    block ``BC = next_pow2(L - max_spec_len)``. Two launches (verify + flush
    ``IS_FLUSH`` specializations) with device-side per-row routing keep the step
    CUDA-graph capturable. Cursors are advanced by ``commit_gdn_replayssm_spec``.
    """
    if scale is None:
        scale = checkpoint_state.shape[-1] ** -0.5
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)
    vb, vw, vnk, vns = get_replayssm_config("gdn_spec_verify", max_spec_len=max_spec_len)
    fb, fw, fnk, fns = get_replayssm_config("gdn_spec_flush", max_spec_len=max_spec_len)

    if launch_mode in ("both", "verify"):
        _launch_gdn_spec(
            mixed_qkv, a, b, A_log, dt_bias, out, checkpoint_state,
            d_cache, k_cache, g_cache, query_start_loc, ssm_state_indices,
            write_pos, cache_base, is_flush, scale, max_cache_len, max_spec_len,
            use_qk_l2norm_in_kernel, False, vb, vw, vns, vnk, null_block_id,
        )
    if launch_mode in ("both", "flush"):
        _launch_gdn_spec(
            mixed_qkv, a, b, A_log, dt_bias, out, checkpoint_state,
            d_cache, k_cache, g_cache, query_start_loc, ssm_state_indices,
            write_pos, cache_base, is_flush, scale, max_cache_len, max_spec_len,
            use_qk_l2norm_in_kernel, True, fb, fw, fns, fnk, null_block_id,
        )
    return out


def commit_gdn_replayssm_spec(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted: torch.Tensor,  # [n_rows] int  (already includes the bonus token)
    state_batch_indices: torch.Tensor,  # [n_rows] int  physical block per row
    max_cache_len: int,  # logical flush threshold L
    max_spec_len: int,
    cache_buf_len: int | None = None,  # physical pow2 buffer next_pow2(L)
    null_block_id: int = 0,
):
    """Advance the block-keyed cursors once per decode step (device-only)."""
    if cache_buf_len is None:
        cache_buf_len = triton.next_power_of_2(max_cache_len)
    n_rows = state_batch_indices.shape[0]
    BLOCK = triton.next_power_of_2(max(1, n_rows))
    _advance_gdn_spec_cursors_kernel[(1,)](
        write_pos,
        cache_base,
        is_flush,
        num_accepted,
        state_batch_indices,
        n_rows,
        stride_sbi=state_batch_indices.stride(0),
        stride_na=num_accepted.stride(0),
        MAX_CACHE_LEN=max_cache_len,
        MAX_SPEC_LEN=max_spec_len,
        CACHE_BUF_LEN=cache_buf_len,
        BLOCK=BLOCK,
        NULL_BLOCK_ID=null_block_id,
    )


def reset_gdn_replayssm_spec_cursors(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    do_reset: torch.Tensor,  # [n_rows] int/bool  1 for first-decode rows
    state_batch_indices: torch.Tensor,  # [n_rows] int
    max_cache_len: int,
    max_spec_len: int,
    null_block_id: int = 0,
):
    """Reset the cursors of first-decode rows (prefill->decode handoff)."""
    n_rows = state_batch_indices.shape[0]
    BLOCK = triton.next_power_of_2(max(1, n_rows))
    init_flush = 1 if 2 * max_spec_len > max_cache_len else 0
    _reset_gdn_replayssm_spec_cursors_kernel[(1,)](
        write_pos,
        cache_base,
        is_flush,
        do_reset,
        state_batch_indices,
        n_rows,
        stride_sbi=state_batch_indices.stride(0),
        stride_reset=do_reset.stride(0),
        INIT_FLUSH=init_flush,
        BLOCK=BLOCK,
        NULL_BLOCK_ID=null_block_id,
    )
