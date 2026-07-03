# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from __future__ import annotations

import torch

from vllm.model_executor.layers.mamba.ops.replayssm_config import (
    get_replayssm_config,
)
from vllm.triton_utils import tl, triton


@triton.jit
def fused_recurrent_gated_delta_rule_replayssm_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, h0, ht,
    d_cache, k_cache, g_cache, ssm_state_indices, write_pos, scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NK: tl.constexpr, BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    # Resolve the physical state slot; zero the output and bail for padded rows.
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty), mask=mask_v)
        return

    # Per-row buffer cursor and flush flag; valid (committed) cache positions.
    # vLLM: write_pos is per decode row (i_n), not per physical slot.
    b_write_pos = tl.load(write_pos + i_n).to(tl.int64)
    b_is_flush = b_write_pos == MAX_CACHE_LEN - 1
    cache_valid = o_c < b_write_pos

    # Gate for the current token: decay g, its exp alpha, and the beta mixing weight.
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    alpha_val = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    # Replay decay over the committed cache, from the cached per-step gates g.
    p_g_main = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + o_c
    b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
    b_g_prefix = tl.cumsum(b_g_all, axis=0)
    b_g_total = tl.sum(b_g_all, axis=0)
    b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
    b_total_decay = tl.exp(b_g_total)

    # Cached delta-rule update vectors d (K-independent), scaled by the replay decay.
    p_d_main = d_cache + state_idx * stride_d_slot + ((i_hv * MAX_CACHE_LEN + o_c[None, :]) * V + o_v[:, None])
    b_d_all = tl.load(p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0).to(tl.float32)
    b_d_scaled_tc = (b_d_all * b_replay_decay[None, :]).to(p_o.dtype.element_ty)  # [BV, BC]

    # Current token value (for the delta-rule update).
    v_off = (2 * H * K) + i_hv * V + o_v
    b_v = tl.load(mixed_qkv + i_n * stride_mixed_qkv_tok + v_off, mask=mask_v, other=0).to(tl.float32)

    # Optional q/k L2 norm: full-vector reciprocal norms (computed, not kept).
    if USE_QK_L2NORM_IN_KERNEL:
        o_kf = tl.arange(0, BK)
        mask_kf = o_kf < K
        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        qf = tl.load(p_mix + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
        kf = tl.load(p_mix + H * K + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
        q_rnorm = 1.0 / tl.sqrt(tl.sum(qf * qf) + 1e-6)
        k_rnorm = 1.0 / tl.sqrt(tl.sum(kf * kf) + 1e-6)
    else:
        q_rnorm = 1.0
        k_rnorm = 1.0

    # Reconstruct the state from the checkpoint + cached (d, k) in K tiles and read
    # it with the current q and k. K-tiling avoids holding a full [BV, BK] tile.
    # Also append the current key chunk to the ring cache (non-flush only).
    b_state_q = tl.zeros([BV], dtype=tl.float32)
    b_state_k = tl.zeros([BV], dtype=tl.float32)
    cur_kq = tl.zeros([1], dtype=tl.float32)
    write_k = (not b_is_flush) and (i_v == 0) and (i_hv == i_h * (HV // H))
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_c = tl.load(p_mix + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * q_rnorm
        k_c = tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * k_rnorm
        q_cs = q_c * scale
        cur_kq += tl.sum(k_c * q_cs)

        # Reconstruct this K tile of the state: S = total_decay * S_0 + d_scaled . k_cache.
        p_h0_c = h0 + state_idx * stride_init_state_token + i_hv * V * K + o_v[:, None] * K + o_kt[None, :]
        b_h0_c = tl.load(p_h0_c, mask=mask_v[:, None] & mask_kt[None, :], other=0).to(tl.float32)
        p_k_c = k_cache + state_idx * stride_k_slot + ((i_h * MAX_CACHE_LEN + o_c[:, None]) * K + o_kt[None, :])
        b_k_all_c = tl.load(p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0).to(p_o.dtype.element_ty)
        b_h_c = b_h0_c * b_total_decay + tl.dot(b_d_scaled_tc, b_k_all_c).to(tl.float32)  # [BV, BKT]

        # Read the state with q and k (accumulated across K tiles).
        b_state_q += tl.sum(b_h_c * q_cs[None, :], axis=1)
        b_state_k += tl.sum(b_h_c * k_c[None, :], axis=1)

        if write_k:
            p_cur_k = k_cache + state_idx * stride_k_slot + ((i_h * MAX_CACHE_LEN + b_write_pos) * K + o_kt)
            tl.store(p_cur_k, k_c.to(p_o.dtype.element_ty), mask=mask_kt & (b_write_pos < MAX_CACHE_LEN))

    # Current-token output: alpha*(S q) + d_cur * (k . q), with the new update
    # vector d_cur = beta * (v - alpha*(S k)).
    b_state_q *= alpha_val
    b_state_k *= alpha_val
    b_d_cur = beta_val * (b_v - b_state_k)
    b_o = b_state_q + b_d_cur * tl.sum(cur_kq)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    if b_is_flush:
        # Flush: fold the current token into the checkpoint, S_t = alpha*S + d_t k_t^T,
        # and persist it. Re-walk K chunks to rebuild S before applying the update.
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
            k_c = tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * k_rnorm
            p_h0_c = h0 + state_idx * stride_init_state_token + i_hv * V * K + o_v[:, None] * K + o_kt[None, :]
            b_h0_c = tl.load(p_h0_c, mask=mask_v[:, None] & mask_kt[None, :], other=0).to(tl.float32)
            p_k_c = k_cache + state_idx * stride_k_slot + ((i_h * MAX_CACHE_LEN + o_c[:, None]) * K + o_kt[None, :])
            b_k_all_c = tl.load(p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0).to(p_o.dtype.element_ty)
            b_h_c = b_h0_c * b_total_decay + tl.dot(b_d_scaled_tc, b_k_all_c).to(tl.float32)
            b_h_new_c = alpha_val * b_h_c + b_d_cur[:, None] * k_c[None, :]
            p_ht_c = ht + state_idx * stride_final_state_token + i_hv * V * K + o_v[:, None] * K + o_kt[None, :]
            tl.store(p_ht_c, b_h_new_c.to(p_ht_c.dtype.element_ty), mask=mask_v[:, None] & mask_kt[None, :])
    else:
        # Non-flush: append the current token's update vector d and gate g to the
        # cache (the k chunks were already written inside the loop above).
        p_cur_d = d_cache + state_idx * stride_d_slot + ((i_hv * MAX_CACHE_LEN + b_write_pos) * V + o_v)
        tl.store(p_cur_d, b_d_cur.to(p_cur_d.dtype.element_ty), mask=mask_v & (b_write_pos < MAX_CACHE_LEN))
        if i_v == 0:
            p_cur_g = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + b_write_pos
            tl.store(p_cur_g, g_val, mask=b_write_pos < MAX_CACHE_LEN)


def fused_recurrent_gated_delta_rule_replayssm(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    block_v: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    nk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached GDN autoregressive decode (one new token per sequence).

    Same call surface as ``fused_recurrent_gated_delta_rule_packed_decode``
    plus the three ring caches (``d_cache``/``k_cache``/``g_cache``) and the
    per-decode-row ``write_pos`` cursor. ``initial_state`` is both the
    checkpoint read (h0) and the (flush-only) checkpoint write (ht), in place.
    """
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")
    if write_pos.ndim != 1 or write_pos.dtype != torch.int32:
        raise ValueError("`write_pos` must be a 1D int32 tensor.")

    B = mixed_qkv.shape[0]
    num_state_slots, HV, V, K = initial_state.shape
    qkv_dim = mixed_qkv.shape[1]
    q_dim = (qkv_dim - HV * V) // 2
    if q_dim <= 0 or q_dim % K != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}, K={K}."
        )
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}.")
    max_cache_len = d_cache.shape[2]

    # Launch config (block_v, num_warps, num_stages, nk) from the L-keyed config
    # module; explicit kwargs override. Lets benchmarks/the config sweep pin it via
    # override_replayssm_config("gdn_decode", ...).
    cfg_bv, cfg_nw, cfg_ns, cfg_nk = get_replayssm_config(
        "gdn_decode", L=max_cache_len
    )
    if block_v is None:
        block_v = cfg_bv
    if num_warps is None:
        num_warps = cfg_nw
    if num_stages is None:
        num_stages = cfg_ns
    if nk is None:
        nk = cfg_nk

    # Cache shape sanity (per state slot): d=(HV, L, V), k=(H, L, K), g=(HV, L).
    if tuple(d_cache.shape[1:]) != (HV, max_cache_len, V):
        raise ValueError(
            f"`d_cache` per-slot shape must be {(HV, max_cache_len, V)} "
            f"(got {tuple(d_cache.shape[1:])})."
        )
    if tuple(k_cache.shape[1:]) != (H, max_cache_len, K):
        raise ValueError(
            f"`k_cache` per-slot shape must be {(H, max_cache_len, K)} "
            f"(got {tuple(k_cache.shape[1:])})."
        )
    if tuple(g_cache.shape[1:]) != (HV, max_cache_len):
        raise ValueError(
            f"`g_cache` per-slot shape must be {(HV, max_cache_len)} "
            f"(got {tuple(g_cache.shape[1:])})."
        )
    if g_cache.dtype != torch.float32:
        raise ValueError(f"`g_cache` must be float32 (got {g_cache.dtype}).")

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"Cached decode kernel only supports NK_global=1 (got K={K}, BK={BK}).")
    if BK % nk != 0:
        raise ValueError(f"nk={nk} must divide BK={BK}.")
    BKT = BK // nk
    if BKT < 16:
        raise ValueError(f"BKT={BKT} must be >=16 for tl.dot (nk={nk}, BK={BK}).")
    # K-tiling keeps the per-program tile small enough that BV=64 (NV=1, half the
    # grid -> fewer redundant cache/metadata loads) fits without register
    # spilling.
    BV = block_v if block_v is not None else min(triton.next_power_of_2(V), 64)
    BC = max(16, triton.next_power_of_2(max_cache_len))

    grid = (triton.cdiv(V, BV), B, HV)
    fused_recurrent_gated_delta_rule_replayssm_kernel[grid](
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, o=out,
        h0=initial_state, ht=initial_state,
        d_cache=d_cache, k_cache=k_cache, g_cache=g_cache,
        ssm_state_indices=ssm_state_indices, write_pos=write_pos, scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_d_slot=d_cache.stride(0),
        stride_k_slot=k_cache.stride(0),
        stride_g_slot=g_cache.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, BC=BC, NK=nk, BKT=BKT,
        MAX_CACHE_LEN=max_cache_len, SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, initial_state
