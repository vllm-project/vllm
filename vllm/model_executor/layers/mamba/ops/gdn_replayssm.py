# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# The verify-step math is adapted from TensorRT-LLM's
# `tensorrt_llm/_torch/modules/fla/cached_replay.py` (NVIDIA/TensorRT-LLM PR #16464,
# Apache-2.0), re-addressed to vLLM's ReplaySSM ring and trackers.
"""ReplaySSM for gated delta net (GDN) speculative decoding.

The baseline speculative-decode kernel stores the full `[HV, V, K]` recurrent state
after every verified token, so the next step can resume from the last accepted one.
ReplaySSM keeps the state at a checkpoint `h0` and caches, for each token since that
checkpoint, the solved delta-rule update `d` (`[HV, V]`), the L2-normalized key `k`
(`[H, K]`) and the cumulative log-decay `G` from the checkpoint (`[HV]`). The state
after the first `n` cached rows is

    h_n = exp(G[n-1]) * h0 + sum_{j<n} exp(G[n-1] - G[j]) * k_j d_j^T,

so a verify step reads `h0` and the short history and writes only the new rows.

The rings are the `(d, k, g)` tensors of the GDN ReplaySSM page, head-major as for
standard decode (`d_cache [HV, R, V]`, `k_cache [H, R, K]`, `g_cache [HV, R]`), but
`g` holds the cumulative log-decay `G` instead of per-token gates. Ring and trackers
follow the Mamba2 ReplaySSM contract (`replayssm_utils`): per physical state slot,
`ring_start` is the oldest live row and `num_committed` the number of accepted rows
since the checkpoint; rows live at `(ring_start + j) % R` with `R = B + 1 + T`. A
verify step of `q` tokens appends its rows after the committed ones. If
`num_committed + q > B` it also folds the committed rows into `h0` (the checkpoint
flush) and stores its rows' `G` relative to the new checkpoint. The runner advances
the trackers after rejection sampling with the same predicate.
"""

import torch

from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_spec import (
    _tl_dtype,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _split_hi_lo(x, DOT_DTYPE: tl.constexpr):
    hi = x.to(DOT_DTYPE)
    lo = (x - hi.to(tl.float32)).to(DOT_DTYPE)
    return hi, lo


@triton.jit
def _gdn_replayssm_verify_kernel(
    mixed_qkv,
    b,
    a,
    A_log,
    dt_bias,
    state,
    d_cache,
    k_cache,
    g_cache,
    ring_start,
    num_committed,
    slot_indices,
    cu_seqlens,
    out,
    stride_slot_indices,
    stride_qkv_row,
    stride_b_row,
    stride_a_row,
    stride_state_slot,
    stride_d_slot,
    stride_k_slot,
    stride_g_slot,
    scale,
    T: tl.constexpr,
    LOGICAL_WINDOW: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    STATE_FP32: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    # Dots run in the ring dtype (fp16 for bf16 activations).
    DOT_DTYPE: tl.constexpr = d_cache.dtype.element_ty

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    n_tok = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if n_tok <= 0:
        return
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    o_hist = tl.arange(0, BH)
    mask_t = o_t < n_tok
    p_out = out + ((bos + o_t[:, None]) * HV + i_hv) * V + o_v[None, :]

    slot = tl.load(slot_indices + i_n * stride_slot_indices).to(tl.int64)
    if slot == PAD_SLOT_ID:
        tl.store(
            p_out, tl.zeros([BT, BV], dtype=out.dtype.element_ty), mask=mask_t[:, None]
        )
        return

    start = tl.load(ring_start + slot)
    committed = tl.load(num_committed + slot)
    flush = (committed + n_tok) > LOGICAL_WINDOW
    is_hist = o_hist < committed
    hist_rows = ((start + o_hist) % R).to(tl.int64)
    last_row = ((start + committed - 1 + R) % R).to(tl.int64)

    d_base = d_cache + slot * stride_d_slot + i_hv * R * V
    k_base = k_cache + slot * stride_k_slot + i_h * R * K
    g_base = g_cache + slot * stride_g_slot + i_hv * R

    b_kh = tl.load(
        k_base + hist_rows[:, None] * K + o_k[None, :],
        mask=is_hist[:, None],
        other=0,
    )
    b_dh = tl.load(
        d_base + hist_rows[:, None] * V + o_v[None, :],
        mask=is_hist[:, None],
        other=0,
    )
    b_Gh = tl.load(g_base + hist_rows, mask=is_hist, other=0.0)
    g_start = tl.load(g_base + last_row, mask=committed > 0, other=0.0)

    # New tokens: post-conv q/k/v from the packed projection, raw gates from a/b.
    rows = bos + o_t
    p_qkv = mixed_qkv + rows[:, None] * stride_qkv_row
    b_q = tl.load(p_qkv + i_h * K + o_k[None, :], mask=mask_t[:, None], other=0)
    b_k = tl.load(p_qkv + (H + i_h) * K + o_k[None, :], mask=mask_t[:, None], other=0)
    b_v = tl.load(
        p_qkv + 2 * H * K + i_hv * V + o_v[None, :], mask=mask_t[:, None], other=0
    ).to(tl.float32)
    x = tl.load(a + rows * stride_a_row + i_hv, mask=mask_t, other=0.0).to(tl.float32)
    x += tl.load(dt_bias + i_hv).to(tl.float32)
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    neg_A = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    b_g = tl.where(mask_t, neg_A * softplus, 0.0)
    b_beta = tl.load(b + rows * stride_b_row + i_hv, mask=mask_t, other=0.0)
    b_beta = tl.where(mask_t, tl.sigmoid(b_beta.to(tl.float32)), 0.0)

    b_kf = b_k.to(tl.float32)
    b_qf = b_q.to(tl.float32)
    b_kn = (b_kf * tl.rsqrt(tl.sum(b_kf * b_kf, 1) + 1e-6)[:, None]).to(DOT_DTYPE)
    b_qn = (b_qf * (tl.rsqrt(tl.sum(b_qf * b_qf, 1) + 1e-6) * scale)[:, None]).to(
        DOT_DTYPE
    )

    b_G_local = tl.cumsum(b_g, 0)
    b_G = g_start + b_G_local

    # h0 tile as [K, BV] (the cache stores [HV, V, K]).
    p_h0 = (
        state
        + slot * stride_state_slot
        + i_hv * V * K
        + o_k[:, None]
        + o_v[None, :] * K
    )
    b_h = tl.load(p_h0)
    if STATE_FP32:
        b_h_hi, b_h_lo = _split_hi_lo(b_h, DOT_DTYPE)
        b_kh0 = tl.dot(b_kn, b_h_hi) + tl.dot(b_kn, b_h_lo)
        b_qh0 = tl.dot(b_qn, b_h_hi) + tl.dot(b_qn, b_h_lo)
    else:
        b_hd = b_h.to(DOT_DTYPE)
        b_kh0 = tl.dot(b_kn, b_hd)
        b_qh0 = tl.dot(b_qn, b_hd)

    hist_decay = tl.exp(b_G[:, None] - b_Gh[None, :])
    b_k_hist_coeff = tl.where(
        is_hist[None, :], tl.dot(b_kn, tl.trans(b_kh)) * hist_decay, 0.0
    )
    b_q_hist_coeff = tl.where(
        is_hist[None, :], tl.dot(b_qn, tl.trans(b_kh)) * hist_decay, 0.0
    )
    b_k_hist = tl.dot(b_k_hist_coeff.to(DOT_DTYPE), b_dh)
    b_q_hist = tl.dot(b_q_hist_coeff.to(DOT_DTYPE), b_dh)

    b_rhs = b_beta[:, None] * (b_v - tl.exp(b_G)[:, None] * b_kh0 - b_k_hist)
    new_decay = tl.exp(b_G[:, None] - b_G[None, :])
    b_A = tl.where(
        o_t[:, None] > o_t[None, :],
        b_beta[:, None] * tl.dot(b_kn, tl.trans(b_kn)) * new_decay,
        0.0,
    )
    # Forward substitution on the strictly lower T x T system (T <= 8).
    b_D = tl.zeros([BT, BV], dtype=tl.float32)
    for row in range(T):
        row_mask = o_t == row
        rhs_row = tl.sum(tl.where(row_mask[:, None], b_rhs, 0.0), axis=0)
        a_row = tl.sum(tl.where(row_mask[:, None], b_A, 0.0), axis=0)
        d_row = rhs_row - tl.sum(a_row[:, None] * b_D, axis=0)
        b_D += tl.where(row_mask[:, None], d_row[None, :], 0.0)

    b_q_new_coeff = tl.where(
        o_t[:, None] >= o_t[None, :], tl.dot(b_qn, tl.trans(b_kn)) * new_decay, 0.0
    )
    b_q_new = tl.zeros([BT, BV], dtype=tl.float32)
    for row in range(T):
        row_mask = o_t == row
        coeff_row = tl.sum(tl.where(row_mask[:, None], b_q_new_coeff, 0.0), axis=0)
        q_new_row = tl.sum(coeff_row[:, None] * b_D, axis=0)
        b_q_new += tl.where(row_mask[:, None], q_new_row[None, :], 0.0)
    b_o = tl.exp(b_G)[:, None] * b_qh0 + b_q_hist + b_q_new
    tl.store(p_out, b_o.to(out.dtype.element_ty), mask=mask_t[:, None])

    # Append this step's rows after the committed ones. After a flush they belong
    # to the new checkpoint, so their decay is local to this step.
    w_rows = ((start + committed + o_t) % R).to(tl.int64)
    tl.store(
        d_base + w_rows[:, None] * V + o_v[None, :],
        b_D.to(DOT_DTYPE),
        mask=mask_t[:, None],
    )
    if i_v == 0:
        tl.store(g_base + w_rows, tl.where(flush, b_G_local, b_G), mask=mask_t)
        if i_hv % (HV // H) == 0:
            tl.store(
                k_base + w_rows[:, None] * K + o_k[None, :],
                b_kn,
                mask=mask_t[:, None],
            )

    if flush:
        # Fold the committed rows into the checkpoint. Each program owns its
        # [K, BV] tile of h0, so the in-place store races with no reader.
        b_Dc = (
            b_dh.to(tl.float32)
            * tl.where(is_hist, tl.exp(g_start - b_Gh), 0.0)[:, None]
        )
        b_Dc_hi, b_Dc_lo = _split_hi_lo(b_Dc, DOT_DTYPE)
        b_kht = tl.trans(b_kh)
        b_hc = (
            b_h.to(tl.float32) * tl.exp(g_start)
            + tl.dot(b_kht, b_Dc_hi)
            + tl.dot(b_kht, b_Dc_lo)
        )
        tl.store(p_h0, b_hc.to(p_h0.dtype.element_ty))


def gdn_replayssm_verify(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    ring_start: torch.Tensor,
    num_committed: torch.Tensor,
    slot_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    *,
    max_query_len: int,
    logical_window: int,
    scale: float,
) -> None:
    """Verify `cu_seqlens.numel() - 1` speculative rows against their checkpoints.

    Args:
        mixed_qkv: `[tokens, 2*H*K + HV*V]` post-conv q/k/v (row stride allowed).
        a: `[tokens, HV]` raw decay gate (row stride allowed).
        b: `[tokens, HV]` raw beta gate (row stride allowed).
        A_log: `[HV]` log decay rate.
        dt_bias: `[HV]` decay bias.
        state: `[slots, HV, V, K]` checkpoints (fp32, bf16 or fp16), updated in
            place on a checkpoint flush.
        d_cache: `[slots, HV, R, V]` solved updates; its dtype is the dot dtype.
        k_cache: `[slots, H, R, K]` normalized keys, same dtype as `d_cache`.
        g_cache: `[slots, HV, R]` fp32 cumulative log-decay.
        ring_start: `[slots]` int32 oldest live ring row (read only).
        num_committed: `[slots]` int32 accepted rows since the checkpoint
            (read only).
        slot_indices: `[N]` live state slot of each row (`NULL_BLOCK_ID` skips).
        cu_seqlens: `[N + 1]` token offsets of the rows.
        out: `[tokens, HV, V]` core output before the gated RMSNorm.
        max_query_len: tokens per row `T` (num_spec + 1, at most 8).
        logical_window: replay history length `B` before a checkpoint flush.
        scale: query scale.

    """
    num_requests = cu_seqlens.numel() - 1
    if num_requests == 0:
        return
    HV, V = out.shape[-2:]
    K = state.shape[-1]
    H = k_cache.shape[1]
    R = d_cache.shape[2]
    T = max_query_len
    assert 1 <= T <= 8 and logical_window >= T and logical_window + T <= R
    assert triton.next_power_of_2(K) == K and triton.next_power_of_2(V) == V
    assert mixed_qkv.shape[-1] == 2 * H * K + HV * V and mixed_qkv.stride(-1) == 1
    assert a.stride(-1) == 1 and b.stride(-1) == 1 and out.is_contiguous()
    assert state.shape[1:] == (HV, V, K) and state.stride()[1:] == (V * K, K, 1)
    assert d_cache.shape[1:] == (HV, R, V) and d_cache.stride()[1:] == (R * V, V, 1)
    assert k_cache.shape[1:] == (H, R, K) and k_cache.stride()[1:] == (R * K, K, 1)
    assert g_cache.shape[1:] == (HV, R) and g_cache.stride()[1:] == (R, 1)
    assert d_cache.dtype == k_cache.dtype and g_cache.dtype == torch.float32

    # Value-tile width measured on H200 with Qwen3.5/3.6 heads. Wider tiles win
    # once the grid fills the GPU; fp32 states also keep a lo tile in registers.
    head_tiles = num_requests * HV
    state_fp32 = state.dtype == torch.float32
    if head_tiles <= 64:
        block_v = 32
    elif head_tiles <= 128 or (state_fp32 and head_tiles <= 512):
        block_v = 64
    else:
        block_v = V
    block_v = min(block_v, V)

    _gdn_replayssm_verify_kernel[(V // block_v, head_tiles)](
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        state,
        d_cache,
        k_cache,
        g_cache,
        ring_start,
        num_committed,
        slot_indices,
        cu_seqlens,
        out,
        slot_indices.stride(0),
        mixed_qkv.stride(0),
        b.stride(0),
        a.stride(0),
        state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        scale,
        T=T,
        LOGICAL_WINDOW=logical_window,
        R=R,
        BT=max(triton.next_power_of_2(T), 2),
        # tl.dot needs a reduction dim of at least 16.
        BH=max(triton.next_power_of_2(logical_window), 16),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=block_v,
        STATE_FP32=state_fp32,
        PAD_SLOT_ID=NULL_BLOCK_ID,
        num_warps=4,
        num_stages=1,
    )


@triton.jit(do_not_specialize=["num_reqs"])
def _gdn_replayssm_materialize_kernel(
    state_ptrs,
    state_slot_strides,
    d_cache_ptrs,
    d_cache_slot_strides,
    k_cache_ptrs,
    k_cache_slot_strides,
    g_cache_ptrs,
    g_cache_slot_strides,
    src_slots_ptr,
    dst_slots_ptr,
    ring_start_ptr,
    replay_prefix_len_ptr,
    active_request_indices_ptr,
    num_reqs,
    STATE_DTYPE: tl.constexpr,
    RING_DTYPE: tl.constexpr,
    MAX_NUM_REQS: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
):
    """Write `src`'s checkpoint advanced by a ring prefix into `dst`.

    One program per (active request, layer, value head x value tile).
    """
    pid_r = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_hv = tl.program_id(2)
    req = tl.load(active_request_indices_ptr + pid_r)
    if req < 0:
        return
    src = tl.load(src_slots_ptr + pid_l * MAX_NUM_REQS + req).to(tl.int64)
    dst = tl.load(dst_slots_ptr + pid_l * MAX_NUM_REQS + req).to(tl.int64)
    if (src == PAD_SLOT_ID) | (dst == PAD_SLOT_ID):
        return
    NV: tl.constexpr = V // BV
    i_hv = pid_hv // NV
    i_v = pid_hv % NV
    i_h = i_hv // (HV // H)

    state = tl.load(state_ptrs + pid_l).to(tl.pointer_type(STATE_DTYPE))
    state_stride = tl.load(state_slot_strides + pid_l)
    d_cache = tl.load(d_cache_ptrs + pid_l).to(tl.pointer_type(RING_DTYPE))
    k_cache = tl.load(k_cache_ptrs + pid_l).to(tl.pointer_type(RING_DTYPE))
    g_cache = tl.load(g_cache_ptrs + pid_l).to(tl.pointer_type(tl.float32))

    start = tl.load(ring_start_ptr + req)
    n = tl.load(replay_prefix_len_ptr + req)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    o_hist = tl.arange(0, BH)
    is_hist = o_hist < n
    rows = ((start + o_hist) % R).to(tl.int64)

    offs = i_hv * V * K + o_k[:, None] + o_v[None, :] * K
    b_h = tl.load(state + src * state_stride + offs).to(tl.float32)
    d_base = d_cache + src * tl.load(d_cache_slot_strides + pid_l) + i_hv * R * V
    k_base = k_cache + src * tl.load(k_cache_slot_strides + pid_l) + i_h * R * K
    g_base = g_cache + src * tl.load(g_cache_slot_strides + pid_l) + i_hv * R
    b_kh = tl.load(
        k_base + rows[:, None] * K + o_k[None, :],
        mask=is_hist[:, None],
        other=0,
    )
    b_dh = tl.load(
        d_base + rows[:, None] * V + o_v[None, :],
        mask=is_hist[:, None],
        other=0,
    ).to(tl.float32)
    b_Gh = tl.load(g_base + rows, mask=is_hist, other=0.0)
    g_last = tl.load(g_base + ((start + n - 1 + R) % R), mask=n > 0, other=0.0)
    b_Dc = b_dh * tl.where(is_hist, tl.exp(g_last - b_Gh), 0.0)[:, None]
    b_Dc_hi, b_Dc_lo = _split_hi_lo(b_Dc, RING_DTYPE)
    b_kht = tl.trans(b_kh)
    b_h = b_h * tl.exp(g_last) + tl.dot(b_kht, b_Dc_hi) + tl.dot(b_kht, b_Dc_lo)
    tl.store(state + dst * state_stride + offs, b_h.to(STATE_DTYPE))


def gdn_replayssm_materialize(
    state_ptrs: torch.Tensor,
    state_slot_strides: torch.Tensor,
    d_cache_ptrs: torch.Tensor,
    d_cache_slot_strides: torch.Tensor,
    k_cache_ptrs: torch.Tensor,
    k_cache_slot_strides: torch.Tensor,
    g_cache_ptrs: torch.Tensor,
    g_cache_slot_strides: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    active_request_indices: torch.Tensor,
    *,
    num_reqs: int,
    state_dtype: torch.dtype,
    ring_dtype: torch.dtype,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    ring_buffer_len: int,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    """Publish canonical prefix states for every GDN layer in one launch.

    The tables hold one base pointer and slot stride (in elements) per layer;
    `src_slots`/`dst_slots` are `[layers, max_num_reqs]`; the first
    `num_reqs` entries of `active_request_indices` select requests (`-1`
    skips); `replay_prefix_len[req]` rows from `ring_start[req]` are folded.
    """
    num_layers, max_num_reqs = src_slots.shape
    if num_reqs == 0 or num_layers == 0:
        return
    block_v = min(head_v_dim, 64)
    grid = (num_reqs, num_layers, num_v_heads * (head_v_dim // block_v))
    _gdn_replayssm_materialize_kernel[grid](
        state_ptrs,
        state_slot_strides,
        d_cache_ptrs,
        d_cache_slot_strides,
        k_cache_ptrs,
        k_cache_slot_strides,
        g_cache_ptrs,
        g_cache_slot_strides,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        active_request_indices,
        num_reqs,
        STATE_DTYPE=_tl_dtype(state_dtype),
        RING_DTYPE=_tl_dtype(ring_dtype),
        MAX_NUM_REQS=max_num_reqs,
        R=ring_buffer_len,
        BH=max(triton.next_power_of_2(ring_buffer_len), 16),
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BV=block_v,
        PAD_SLOT_ID=pad_slot_id,
        num_warps=4,
    )
