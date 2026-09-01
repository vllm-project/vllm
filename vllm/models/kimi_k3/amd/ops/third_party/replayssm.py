# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Ported from ATOM atom/model_ops/fla_ops/replayssm.py for vLLM Kimi-K3 KDA.
#
# Replay algebra, record layout and tile sizes track ROCm/ATOM#1883. The fold
# kernel is vLLM's own: ATOM's prefill rebuilds the checkpoint wholesale, while
# vLLM's chunked-prefill path reads the checkpoint as an initial state and so
# needs any pending records materialised into it first.

from __future__ import annotations

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

__all__ = [
    "PAD_SLOT_ID",
    "flush_threshold_ok",
    "replayssm_buffer_shapes",
    "replayssm_commit",
    "replayssm_fold",
    "replayssm_sigmoid_gating_delta_rule",
]

PAD_SLOT_ID = -1

#: Replay-GEMM arithmetic, keyed by the record buffer's dtype. See `_replay_dot`
#: for what the modes do and why the split is only one-sided.
_DOT_MODE_BY_RECORD_DTYPE = {
    torch.bfloat16: 2,
    torch.float16: 3,
}


def _replay_dot_mode(record_dtype: torch.dtype) -> int:
    """How to contract the records against the checkpoint on a flush.

    A 16-bit record buffer replays on the bf16 matrix cores: fp32 MFMA runs at
    an eighth of the bf16 rate on CDNA, and upcasting a bf16 record to fp32
    cannot add information it never had. An fp32 buffer is the other way round
    -- a bf16 hi/lo pair tops out near 16 mantissa bits and would lose real
    precision -- so those callers keep the fp32 contraction.
    """
    return _DOT_MODE_BY_RECORD_DTYPE.get(record_dtype, 0)


# The flush predicate below is evaluated in two places -- the layer kernel, to
# decide whether *this* step folds, and the commit kernel, to re-derive what the
# *previous* step decided, since the forward never touches the cursor. They must
# agree exactly, so both use the same expression; do not inline a variant.
#
# Firing at `h + 2T > L` rather than `h + T > L` keeps at least one full window
# free: a step landing at h = L-T followed by a full accept would otherwise
# leave a single free slot and truncate the next window to one draft. It also
# guarantees h + T <= L, so appends never run off the end of the buffer.


def flush_threshold_ok(cache_len: int, max_query_len: int) -> bool:
    """``cache_len`` must hold two full windows or the invariant above breaks."""
    return cache_len >= 2 * max_query_len


def replayssm_buffer_shapes(
    cache_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    is_kda: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Per-slot record buffer shapes: (k, u, g), head-major.

    Head-minor put a head's consecutive records ``num_v_heads * head_dim``
    apart, so a rebuild read h scattered 256 B chunks over a 128 KiB span.
    Hoisting the head axis above the record axis makes a head's whole slab
    contiguous and leaves the per-step append contiguous too.
    """
    return (
        (num_v_heads, cache_len, head_k_dim),
        (num_v_heads, cache_len, head_v_dim),
        (num_v_heads, cache_len, head_k_dim) if is_kda else (num_v_heads, cache_len),
    )


# --------------------------------------------------------------------------- #
# Record replay                                                               #
# --------------------------------------------------------------------------- #
#
# Unrolling the recurrence over the h committed records,
#
#     S_{j+1} = diag(exp(g_j)) S_j + u_j (x) k_j
#
# leaves a form with no sequential dependency in it at all:
#
#     S_h = exp(C) * S_0 + sum_j exp(C - C_j) * u_j (x) k_j,   C_j = sum_{i<=j} g_i
#
# so the replay is ONE diagonal scale plus ONE GEMM rather than h dependent
# rank-1 updates. What makes this legal is that the buffer stores `u` (already
# delta-corrected) and not raw `v`: with `v`, each `u_j = beta_j (v_j - S_{j-1}^T
# k_j)` would depend on the previous state and the chain could not be cut.
#
# The serial form measured 4.95 us per record at K = V = 128 -- not bandwidth,
# but h dependent full-tile multiplies with no instruction-level parallelism to
# hide the latency.


@triton.jit
def _replay_tiles(
    buf_k,
    buf_u,
    buf_g,
    slot,
    h,
    stride_bufk_slot,
    stride_bufk_pos,
    stride_bufk_hv,
    stride_bufu_slot,
    stride_bufu_pos,
    stride_bufu_hv,
    stride_bufg_slot,
    stride_bufg_pos,
    stride_bufg_hv,
    i_hv,
    o_k,
    o_v,
    mask_k,
    mask_v,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """The h committed records as dense ``[BH, *]`` tiles, ready for one GEMM.

    Returns ``(b_kw, b_u, b_decay)`` with ``b_kw[j] = exp(C - C_j) * k_j``,
    ``b_u[j] = u_j`` and ``b_decay = exp(C)``, the factor the incoming
    checkpoint is scaled by.

    Rows ``j >= h`` are zeroed rather than skipped, so they contribute nothing
    to the GEMM and no caller has to branch on how many records were live.
    Their gate loads default to 0, which leaves both the running sum and the
    total untouched -- ``b_ctot`` is the sum over the whole tile precisely
    because the padding is additively neutral.
    """
    o_h = tl.arange(0, BH)
    m_h = o_h < h

    if IS_KDA:
        b_g = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h[:, None] * stride_bufg_pos
            + i_hv * stride_bufg_hv
            + o_k[None, :],
            mask=m_h[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        # Scalar gate: broadcast the per-record value across K so the cumsum and
        # the weighting below stay one code path.
        b_g1 = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h * stride_bufg_pos
            + i_hv * stride_bufg_hv,
            mask=m_h,
            other=0.0,
        ).to(tl.float32)
        b_g = b_g1[:, None] + tl.zeros([BH, BK], dtype=tl.float32)

    # Gates are log-decays (<= 0), so C - C_j <= 0 and every weight is in (0, 1]
    # -- the exponentials cannot overflow however long the buffer gets.
    b_c = tl.cumsum(b_g, axis=0)
    b_ctot = tl.sum(b_g, axis=0)
    b_w = exp(b_ctot[None, :] - b_c)

    b_k = tl.load(
        buf_k
        + slot * stride_bufk_slot
        + o_h[:, None] * stride_bufk_pos
        + i_hv * stride_bufk_hv
        + o_k[None, :],
        mask=m_h[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    b_u = tl.load(
        buf_u
        + slot * stride_bufu_slot
        + o_h[:, None] * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_v[None, :],
        mask=m_h[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    return b_k * b_w, b_u, exp(b_ctot)


@triton.jit
def _replay_dot(
    lhs,
    rhs,
    SPLIT_LHS: tl.constexpr,
    DOT_MODE: tl.constexpr,
):
    """``tl.dot(lhs, rhs)`` for the replay contraction.

    ``SPLIT_LHS`` says which side is ``b_kw``, the only operand genuinely wider
    than the record buffer: ``b_u`` is a raw record load, so narrowing it back
    to the buffer's own dtype is exact, while ``b_kw = k * w`` is a product and
    is not.

    ``DOT_MODE`` picks the arithmetic:
      0 -- one fp32 dot. What an fp32 record buffer needs: bf16 hi/lo tops out
           near 16 mantissa bits and cannot carry fp32 records faithfully.
      1 -- one bf16 dot. ~8 mantissa bits; too lossy, attribution only.
      2 -- bf16 hi/lo on the ``b_kw`` side only, 2 dots. For a bf16 record
           buffer the other side's lo term is identically zero, so this is the
           whole split; measured 3.6e-06 relative against an fp64 reference,
           553x tighter than mode 1, for 2.9 us over it.
      3 -- also splits the record side, 3 dots (lo*lo is negligible). Needed
           when the buffer is 16-bit but not bf16, i.e. fp16.
    """
    if DOT_MODE == 0:
        acc = tl.dot(lhs, rhs)
    elif SPLIT_LHS:
        b_hi = lhs.to(tl.bfloat16)
        b_other = rhs.to(tl.bfloat16)
        acc = tl.dot(b_hi, b_other)
        if DOT_MODE >= 2:
            b_lo = (lhs - b_hi.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_lo, b_other)
        if DOT_MODE == 3:
            b_other_lo = (rhs - b_other.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_hi, b_other_lo)
    else:
        b_hi = rhs.to(tl.bfloat16)
        b_other = lhs.to(tl.bfloat16)
        acc = tl.dot(b_other, b_hi)
        if DOT_MODE >= 2:
            b_lo = (rhs - b_hi.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_other, b_lo)
        if DOT_MODE == 3:
            b_other_lo = (lhs - b_other.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_other_lo, b_hi)
    return acc


# --------------------------------------------------------------------------- #
# Commit kernel -- runs once per forward, shared by every linear-attn layer    #
# --------------------------------------------------------------------------- #


@triton.jit(do_not_specialize=["N", "T_MAX", "CAP"])
def _replayssm_commit_kernel(
    write_pos,
    slot_idx,
    num_accepted,
    N,
    T_MAX,
    CAP,
):
    i_n = tl.program_id(0)
    if i_n >= N:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    h = tl.load(write_pos + slot)
    # vLLM zeroes the cursor and flags pending_reset instead of parking it at a
    # sentinel, so this branch is normally inert. Kept because it is the only
    # thing standing between a negative cursor and a record row outside the
    # slot, and it costs one predicate.
    if h < 0:
        tl.store(write_pos + slot, 0)
        return
    # Re-derive the previous forward's flush decision. The forward never mutates
    # write_pos, so `h` here is exactly what that forward branched on.
    prev_flushed = h + 2 * T_MAX > CAP
    base = tl.where(prev_flushed, 0, h)
    tl.store(write_pos + slot, base + tl.load(num_accepted + i_n))


def replayssm_commit(
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    num_accepted: torch.Tensor,
    max_query_len: int,
    cache_len: int,
) -> None:
    """Advance each sequence's record cursor by the previous step's accepts.

    Call exactly once per forward, before any linear-attention layer runs.
    Device-side so the accepted counts never round-trip to the host.
    """
    n = slot_idx.numel()
    if n == 0:
        return
    _replayssm_commit_kernel[(n,)](
        write_pos,
        slot_idx,
        num_accepted,
        n,
        max_query_len,
        cache_len,
        num_warps=1,
    )


# --------------------------------------------------------------------------- #
# Fold kernel -- materialise pending records before the chunk/prefill path     #
# --------------------------------------------------------------------------- #


@triton.jit(do_not_specialize=["N"])
def _replayssm_fold_kernel(
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    fold_len,
    slot_idx,
    N,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    stride_ckpt_slot: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufu_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    DOT_MODE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if i_n >= N:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    h = tl.load(fold_len + i_n).to(tl.int32)
    if h <= 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_ckpt = (
        ckpt + slot * stride_ckpt_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    )
    b_h = tl.load(p_ckpt, mask=mask_h, other=0.0).to(tl.float32)

    b_kw, b_ru, b_decay = _replay_tiles(
        buf_k,
        buf_u,
        buf_g,
        slot,
        h,
        stride_bufk_slot,
        stride_bufk_pos,
        stride_bufk_hv,
        stride_bufu_slot,
        stride_bufu_pos,
        stride_bufu_hv,
        stride_bufg_slot,
        stride_bufg_pos,
        stride_bufg_hv,
        i_hv,
        o_k,
        o_v,
        mask_k,
        mask_v,
        K,
        V,
        BK,
        BH,
        True,
    )
    # KDA keeps the state transposed as [BV, BK], so the contraction is u^T @ k~.
    b_h = b_h * b_decay[None, :]
    b_h += _replay_dot(tl.trans(b_ru), b_kw, False, DOT_MODE)

    tl.store(p_ckpt, b_h.to(p_ckpt.dtype.element_ty), mask=mask_h)


def replayssm_fold(
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    fold_len: torch.Tensor,
    slot_idx: torch.Tensor,
) -> None:
    """Collapse each slot's committed records into its checkpoint, in place.

    Kernels that consume the recurrent state directly (the chunk/prefill path)
    cannot see the records, so a slot leaving the ReplaySSM decode path must
    have them materialized first. ``fold_len`` is the cursor snapshot taken when
    the step's metadata was built; the caller is responsible for zeroing
    ``write_pos`` for these slots so every layer folds the same records.
    """
    n = slot_idx.numel()
    if n == 0:
        return
    HV, V, K = ckpt.shape[1], ckpt.shape[2], ckpt.shape[3]
    cap = buf_k.shape[2]
    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "K must fit one block"
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    # No max_query_len here, so bound the replay tile by the whole buffer rather
    # than by the pre-flush cursor ceiling. Rows past `fold_len` are masked out
    # and contribute nothing, so an over-wide tile is slower, never wrong.
    BH = max(16, triton.next_power_of_2(cap))

    _replayssm_fold_kernel[(NV, n * HV)](
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        fold_len=fold_len,
        slot_idx=slot_idx,
        N=n,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BH=BH,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0),
        stride_bufk_pos=buf_k.stride(2),
        stride_bufk_hv=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0),
        stride_bufu_pos=buf_u.stride(2),
        stride_bufu_hv=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0),
        stride_bufg_pos=buf_g.stride(2),
        stride_bufg_hv=buf_g.stride(1),
        DOT_MODE=_replay_dot_mode(buf_u.dtype),
        num_warps=1,
        num_stages=3,
    )


# --------------------------------------------------------------------------- #
# Fused rebuild + decode/verify kernel                                        #
# --------------------------------------------------------------------------- #


@triton.jit
def _kda_gate(A_log_ptr, a_ptr, dt_bias_ptr, mask_k, LOWER_BOUND, USE_LOWER_BOUND):
    x = tl.load(a_ptr, mask=mask_k, other=0.0).to(tl.float32) + tl.load(
        dt_bias_ptr, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_A = tl.load(A_log_ptr).to(tl.float32)
    if USE_LOWER_BOUND:
        return LOWER_BOUND * tl.sigmoid(tl.exp(b_A) * x)
    softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    return -tl.exp(b_A) * softplus_x


@triton.jit(do_not_specialize=["N", "T_TOT", "T_MAX", "CAP"])
def _replayssm_kda_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    LOWER_BOUND,
    N,
    T_TOT,
    T_MAX,
    CAP,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    stride_ckpt_slot: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufu_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    DOT_MODE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return

    h = tl.load(write_pos + slot).to(tl.int32)
    # Graph capture wires the buffers up and replays dummy batches without
    # committing, so the cursor can still be negative here; folding nothing and
    # writing from 0 is both the right answer and what keeps `base` off -1,
    # which would index a record row outside this slot.
    h = tl.maximum(h, 0)
    do_flush = h + 2 * T_MAX > CAP
    base = tl.where(do_flush, 0, h).to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]  # [BV, BK] -- transposed layout

    # ---- 1. rebuild S_h -----------------------------------------------------
    p_ckpt_hv = (
        ckpt + slot * stride_ckpt_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    )
    b_h = tl.load(p_ckpt_hv, mask=mask_h, other=0.0).to(tl.float32)

    b_kw, b_ru, b_decay = _replay_tiles(
        buf_k,
        buf_u,
        buf_g,
        slot,
        h,
        stride_bufk_slot,
        stride_bufk_pos,
        stride_bufk_hv,
        stride_bufu_slot,
        stride_bufu_pos,
        stride_bufu_hv,
        stride_bufg_slot,
        stride_bufg_pos,
        stride_bufg_hv,
        i_hv,
        o_k,
        o_v,
        mask_k,
        mask_v,
        K,
        V,
        BK,
        BH,
        True,
    )
    # KDA keeps the state transposed as [BV, BK], so the contraction is u^T @ k~.
    b_h = b_h * b_decay[None, :]
    b_h += _replay_dot(tl.trans(b_ru), b_kw, False, DOT_MODE)

    if do_flush:
        tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=mask_h)

    # ---- 2. this step's tokens ---------------------------------------------
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_a = a + (bos * HV + i_hv) * K + o_k
    p_b = b + bos * HV + i_hv
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv * K + o_k

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = _kda_gate(p_A_log, p_a, p_dt_bias, mask_k, LOWER_BOUND, USE_LOWER_BOUND)
        b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h *= exp(b_g)[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        tl.store(
            p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mask_v
        )

        pos = base + i_t
        p_bu = (
            buf_u
            + slot * stride_bufu_slot
            + pos * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v
        )
        tl.store(p_bu, b_v.to(p_bu.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + pos * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg = (
                buf_g
                + slot * stride_bufg_slot
                + pos * stride_bufg_pos
                + i_hv * stride_bufg_hv
                + o_k
            )
            tl.store(p_bg, b_g.to(p_bg.dtype.element_ty), mask=mask_k)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_a += stride_a_token
        p_b += stride_b_token


def replayssm_sigmoid_gating_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    o: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """ReplaySSM for the KDA (Kimi-K3) linear-attention layer.

    Drop-in for :func:`fused_sigmoid_gating_delta_rule_update` on the decode /
    verify path: same fused gating, same ``[slot, HV, V, K]`` state layout, but
    the pool holds one checkpoint per request instead of one state per
    speculative token.

    ``o`` may be passed to write the output in place (K3 does this).
    """
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[2]
    if scale is None:
        scale = K**-0.5
    assert flush_threshold_ok(cap, max_query_len), (
        f"replayssm cache_len={cap} must be >= 2*max_query_len={2 * max_query_len}"
    )

    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "K must fit one block"
    # KDA tile width. Measured on Kimi-K3 (tp=8, conc=64, per-launch from a
    # trace): BV=16 25.8 us, BV=32 23.3 us, BV=64 25.7 us against a 22.3 us
    # baseline, and BV=32 with num_warps=4 -- the width and warp count the
    # baseline `fused_sigmoid_gating` kernel itself uses -- is 36.2 us, so
    # matching the baseline's launch config is the worst of the options here.
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    # Replay tile height: the cursor can reach cap - max_query_len before a
    # flush resets it, and `tl.dot` wants at least 16 rows on CDNA, so a short
    # buffer pads rather than shrinks.
    BH = max(16, triton.next_power_of_2(cap - max_query_len))

    q, k, v, a, b = (x.contiguous() for x in (q, k, v, a, b))
    out = q.new_empty(1, T_tot, HV, V) if o is None else o.unsqueeze(0)

    _replayssm_kda_fwd_kernel[(NV, N * HV)](
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=scale,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        N=N,
        T_TOT=T_tot,
        T_MAX=max_query_len,
        CAP=cap,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BH=BH,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0),
        stride_bufk_pos=buf_k.stride(2),
        stride_bufk_hv=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0),
        stride_bufu_pos=buf_u.stride(2),
        stride_bufu_hv=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0),
        stride_bufg_pos=buf_g.stride(2),
        stride_bufg_hv=buf_g.stride(1),
        stride_a_token=a.stride(-3),
        stride_b_token=b.stride(-2),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        DOT_MODE=_replay_dot_mode(buf_u.dtype),
        num_warps=1,
        num_stages=3,
    )
    return out
