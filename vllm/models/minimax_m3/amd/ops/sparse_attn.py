# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm block-sparse GQA prefill kernel for MiniMax-M3.

Only the prefill path is specialized here: each 128-token KV block is split into
SUB_K-token sub-tiles to right-size the per-block QK/PV GEMMs. Decode, the FP8
dtype set and the sparse block size come unchanged from ``common.ops``.

On gfx950 the attend is also grouped. The kernel is KV-gather bound (a load-only
twin costs 103% of it) and neighbouring query tokens select mostly the same
blocks, so ``_SPARSE_ATTN_BLOCK_Q`` consecutive tokens share a program and walk
the *union* of their selections, fetching each block once per group rather than
once per token. Groups selecting too widely for that to pay off fall back to
``_gqa_sparse_fwd_kernel``, one token per program. Every grid comes from shapes
and each program routes itself, so no launch decision reads a device value.
"""

import torch

from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _FP8_DTYPES,
    _KV_SCALE_NONE,
    SPARSE_BLOCK_SIZE,
    _kv_scale_args,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms.rocm import on_gfx950, on_mi3xx
from vllm.triton_utils import tl, triton

__all__ = ["minimax_m3_sparse_attn", "minimax_m3_sparse_attn_decode"]


# Sub-tile width for the one-token-per-program kernel's QK/PV GEMMs, gfx950 -> 64
# and everything else -> 32. Must divide SPARSE_BLOCK_SIZE.
_SPARSE_ATTN_SUB_K = SPARSE_BLOCK_SIZE // 2 if on_gfx950() else SPARSE_BLOCK_SIZE // 4

# The union path's accumulator is _SPARSE_ATTN_BLOCK_Q times taller, so it needs
# a narrower sub-tile to keep two waves in flight over the next block's fetch.
_SPARSE_ATTN_UNION_SUB_K = SPARSE_BLOCK_SIZE // 4

# Query tokens per program for the prefill attend. Tuned on gfx950 only; 1
# restores the one-token-per-program path exactly, including its numerics.
_SPARSE_ATTN_BLOCK_Q = 4 if on_gfx950() else 1

# Locality guard: groups selecting more than this fraction of their
# BLOCK_SIZE_Q * topk candidate slots fall back to one token at a time. With
# gather cost g and per-row compute w, the union costs |U|*(g + B*w) against
# B*topk*(g + w) per token, so break-even is a *fraction* of the candidates,
# independent of topk, which an absolute count would not track. Scanned on gfx950
# over |union| 16..56 for best worst case, clearing the ~22 of 64 a live 60k
# prefill selects; FP8 crosses earlier as its gathers are half the bytes.
_SPARSE_ATTN_UNION_MAX_FRAC_FP8 = 0.42
_SPARSE_ATTN_UNION_MAX_FRAC_DEFAULT = 0.60

# Union members pack into one int32, block id low and membership mask above: the
# inner loop is latency bound on scalar loads, so this saves one load per member.
_UNION_BITS_SHIFT = 16
_UNION_BLK_MASK = (1 << _UNION_BITS_SHIFT) - 1

_SPARSE_ATTN_PREFILL_KWARG: dict | None = None


def _sparse_attn_prefill_kwargs(*, use_union: bool) -> dict:
    """MFMA + pipeline launch params for the sub-tiled prefill kernels.

    ``matrix_instr_nonkdim=16`` selects MFMA_16x16 and ``num_stages=1`` fits LDS.
    The union path takes ``num_warps=2`` for its taller accumulator, one token per
    program keeps 1. Cached: arch is fixed per process.
    """
    global _SPARSE_ATTN_PREFILL_KWARG
    if _SPARSE_ATTN_PREFILL_KWARG is None:
        kwarg: dict = {}
        if on_mi3xx():
            kwarg = {"matrix_instr_nonkdim": 16, "num_stages": 1}
            if not on_gfx950():
                # Deprecated on gfx950, where Triton forces it to 1 and warns.
                kwarg["kpack"] = 2
        _SPARSE_ATTN_PREFILL_KWARG = kwarg
    if not _SPARSE_ATTN_PREFILL_KWARG:
        # Non-CDNA AMD: no MFMA knobs, num_warps left at Triton's default as on
        # the pre-union path. gfx950 is a subset of CDNA, so the assert holds.
        assert not use_union
        return {}
    return {**_SPARSE_ATTN_PREFILL_KWARG, "num_warps": 2 if use_union else 1}


@triton.jit
def _group_row(q_start, pid_b, group_in_req, GROUP_Q: tl.constexpr):
    """Union-buffer row for group ``group_in_req`` of request ``pid_b``.

    Indexed by group, not (request, max_query_len), which would size the buffers
    by batch * max_query_len rather than total_q. Requests cannot overlap because
    ``s // G + ceil(L / G) <= (s + L) // G + 1``, so one spare row per request
    suffices and no prefix sum is needed.
    """
    return q_start // GROUP_Q + pid_b + group_in_req


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        # one wave per 32 candidate slots: the dedup compares every slot against
        # every other, so its tile is (BLOCK_SIZE_Q * next_pow2(topk))^2
        "num_warps": lambda args: max(
            1, min(8, args["BLOCK_SIZE_Q"] * args["BLOCK_SIZE_T"] // 32)
        ),
    }
)
@triton.jit
def _union_build_kernel(
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk], per query TOKEN
    u_blk_ptr,  # out: [num_kv_heads, num_groups, BLOCK_SIZE_N] int32
    u_len_ptr,  # out: [num_kv_heads, num_groups] int32, |union|
    cu_seqlens_q,
    max_topk,
    stride_th,
    stride_tn,
    stride_tk,
    stride_ub_h,
    stride_ub_g,
    stride_ul_h,
    num_groups,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BITS_SHIFT: tl.constexpr,  # membership mask sits above this bit
):
    """Per query group, compact the group's selected blocks into their union.

    Writes the distinct ids it selected, dense in ``[0, u_len)``, each packed with
    a mask of which queries selected it, so the attend gets both the page and its
    per-query predicate from one int32. Reads ``topk_idx`` as the selection kernel
    writes it (per token, -1 padded).
    """
    tl.static_assert(
        BITS_SHIFT + BLOCK_SIZE_Q < 31,
        "block id and membership mask must fit in a non-negative int32",
    )
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_Q * BLOCK_SIZE_T
    pid_g = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_tok0 = pid_g * BLOCK_SIZE_Q
    if q_tok0 >= q_len:
        return
    row = _group_row(q_start, pid_b, pid_g, BLOCK_SIZE_Q)
    off_g = tl.arange(0, BLOCK_SIZE_Q)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    t_grp = tl.load(
        t_ptr
        + (q_start + q_tok0 + off_g[:, None]) * stride_tn
        + pid_kh * stride_th
        + off_t[None, :] * stride_tk,
        mask=((q_tok0 + off_g[:, None]) < q_len) & (off_t[None, :] < max_topk),
        other=-1,
    ).to(tl.int32)
    cand = tl.reshape(t_grp, (BLOCK_SIZE_N,))
    slot = tl.arange(0, BLOCK_SIZE_N)
    same_block = cand[:, None] == cand[None, :]
    # keep each distinct id at its earliest slot, and compact to dense positions
    is_first = (
        tl.sum(tl.where((slot[None, :] < slot[:, None]) & same_block, 1, 0), axis=1)
        == 0
    )
    is_first = is_first & (cand >= 0)
    dest_slot = tl.maximum(tl.cumsum(is_first.to(tl.int32), axis=0) - 1, 0)
    # bit r of member_mask[s] <-> query r of the group selected cand[s] itself
    selected_by = tl.max(
        tl.reshape(same_block.to(tl.int32), (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_T)),
        axis=2,
    )
    member_mask = tl.sum(selected_by << off_g[None, :], axis=1)
    # row < num_groups holds only while cu_seqlens_q[-1] == total_q; mask instead
    # of trusting it, since these stores would otherwise run off the buffer.
    fits = row < num_groups
    u_off = pid_kh * stride_ub_h + row * stride_ub_g
    tl.store(
        u_blk_ptr + u_off + dest_slot,
        cand | (member_mask << BITS_SHIFT),
        mask=is_first & fits,
    )
    tl.store(
        u_len_ptr + pid_kh * stride_ul_h + row,
        tl.sum(is_first.to(tl.int32), axis=0),
        mask=fits,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_union_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    u_blk_ptr,  # per-group union members (block id + mask), _union_build_kernel
    u_len_ptr,  # per-group |union|, also the guard's routing key
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    gqa_group_size,
    head_dim,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_ub_h,
    stride_ub_g,
    stride_ul_h,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    SUB_K: tl.constexpr,  # KV sub-tile width
    UNION_MAX: tl.constexpr,  # locality guard threshold
    BITS_SHIFT: tl.constexpr,  # membership mask sits above this bit of u_blk
):
    """BLOCK_SIZE_Q consecutive query tokens per program, over their block union.

    Keeps only groups whose |union| is within UNION_MAX; ``_gqa_sparse_fwd_kernel``
    takes exactly the complement, so the two launches partition the groups and
    every query is attended exactly once.
    """
    BLOCK_SIZE_QH: tl.constexpr = BLOCK_SIZE_Q * BLOCK_SIZE_H
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_g = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_len = (q_len + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    if pid_g >= q_block_len:
        return
    row = _group_row(q_start, pid_b, pid_g, BLOCK_SIZE_Q)
    u_len = tl.load(u_len_ptr + pid_kh * stride_ul_h + row)
    if u_len > UNION_MAX:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    off_g = tl.arange(0, BLOCK_SIZE_Q)
    q_tok0 = pid_g * BLOCK_SIZE_Q
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(q_tok0, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    # A query skips union members it did not select, so a block can be fully
    # masked for some rows: -inf as the running max would give NaN, hence the
    # finite sentinel and a linear normalizer.
    m_i = tl.full((BLOCK_SIZE_QH,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
    NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K
    BLK_MASK: tl.constexpr = (1 << BITS_SHIFT) - 1
    u_off = u_blk_ptr + pid_kh * stride_ub_h + row * stride_ub_g
    for s in tl.range(u_len):
        packed = tl.load(u_off + s)
        blk = packed & BLK_MASK
        # bit r above BITS_SHIFT <-> the group's r-th query selected this block
        in_set = (((packed >> BITS_SHIFT) >> off_g) & 1) != 0
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        for sub_i in range(NUM_SUB):
            off_sub = tl.arange(0, SUB_K) + sub_i * SUB_K
            pos_sub = c + off_sub
            pos_mask_sub = pos_sub < seq_len
            k_sub = tl.load(
                kv_base
                + off_sub[None, :] * stride_kv_pos
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask_sub[None, :],
                other=0.0,
            )
            if USE_FP8:
                k_sub = k_sub.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k_sub = (k_sub * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_sub) * stride_ks_t,
                        mask=pos_mask_sub,
                        other=1.0,
                    )
                    k_sub = (k_sub * k_scale[None, :]).to(q.dtype)
            off_q_sub = off_g[:, None] + q_tok0 + prefix_len - off_sub[None, :]
            qk_sub = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, SUB_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk_sub += tl.where(off_q_sub[:, None, :] >= c, 0, float("-inf"))
            # sparsity: a query attends only the blocks IT selected
            qk_sub += tl.where(in_set[:, None, None], 0, float("-inf"))
            qk_sub = tl.reshape(qk_sub, BLOCK_SIZE_QH, SUB_K)
            qk_sub += tl.dot(q, k_sub) * sm_scale_log2e
            qk_sub += tl.where(pos_mask_sub[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk_sub, axis=1))
            # finite m_i keeps this 1.0 (not NaN) for an all-masked block
            alpha = tl.exp2(m_i - m_ij)
            p_sub = tl.exp2(qk_sub - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p_sub, axis=1)
            acc_o = acc_o * alpha[:, None]
            v_sub = tl.load(
                kv_base
                + off_sub[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_mask_sub[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v_sub = v_sub.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v_sub = (v_sub * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_sub) * stride_vs_t,
                        mask=pos_mask_sub,
                        other=1.0,
                    )
                    v_sub = (v_sub * v_scale[:, None]).to(q.dtype)
            acc_o += tl.dot(p_sub.to(v_sub.dtype), v_sub)
            m_i = m_ij
    # l_i is 0 for padding rows past q_len, which the store drops, and for a row
    # that selected nothing, which a well-formed topk_idx never produces.
    acc_o = acc_o / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(q_tok0, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk], per query TOKEN
    u_len_ptr,  # per-group |union|, read only to route; unused if not GUARDED
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_ul_h,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
    SUB_K: tl.constexpr,
    GUARDED: tl.constexpr,  # take only the groups the union path gave up on
    UNION_MAX: tl.constexpr,
    GROUP_Q: tl.constexpr,  # union group size, i.e. the prologue's BLOCK_SIZE_Q
):
    """One query token per program, over that token's own top-k blocks.

    Unguarded this is the whole prefill attend. Guarded it is the union path's
    fallback, taking only tokens whose group's |union| exceeded UNION_MAX, so a
    badly-selecting layer costs about what it did before. Kept separate rather
    than a constexpr branch in the union kernel: sharing the loop body cost 12%
    on the prefix=0 chunk, and one program per group measured 0.50-0.91x.
    """
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if pid_q >= q_len:
        return
    if GUARDED:
        row = _group_row(q_start, pid_b, pid_q // GROUP_Q, GROUP_Q)
        if tl.load(u_len_ptr + pid_kh * stride_ul_h + row) <= UNION_MAX:
            return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    off_t = tl.arange(0, BLOCK_SIZE_T)
    NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K
    t_ptr_q = t_ptr + (q_start + pid_q) * stride_tn + pid_kh * stride_th
    topk_idx = tl.load(t_ptr_q + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_H, BLOCK_SIZE_D)
    # A selected block's first sub-tile holds its own start position, which every
    # query selecting it can see, so the running max is finite from the first
    # iteration and log space is safe here (given real_topk >= 1).
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    for _ in tl.range(real_topk):
        blk = tl.load(t_ptr_q).to(tl.int32)
        t_ptr_q = t_ptr_q + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        for sub_i in range(NUM_SUB):
            off_sub = tl.arange(0, SUB_K) + sub_i * SUB_K
            pos_sub = c + off_sub
            pos_mask_sub = pos_sub < seq_len
            k_sub = tl.load(
                kv_base
                + off_sub[None, :] * stride_kv_pos
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask_sub[None, :],
                other=0.0,
            )
            if USE_FP8:
                k_sub = k_sub.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k_sub = (k_sub * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_sub) * stride_ks_t,
                        mask=pos_mask_sub,
                        other=1.0,
                    )
                    k_sub = (k_sub * k_scale[None, :]).to(q.dtype)
            qk_sub = tl.zeros((BLOCK_SIZE_H, SUB_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk_sub += tl.where(
                (pid_q + prefix_len - off_sub)[None, :] >= c, 0, float("-inf")
            )
            qk_sub += tl.dot(q, k_sub) * sm_scale_log2e
            qk_sub += tl.where(pos_mask_sub[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk_sub, axis=1))
            p_sub = tl.exp2(qk_sub - m_ij[:, None])
            l_ij = tl.sum(p_sub, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v_sub = tl.load(
                kv_base
                + off_sub[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_mask_sub[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v_sub = v_sub.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v_sub = (v_sub * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_sub) * stride_vs_t,
                        mask=pos_mask_sub,
                        other=1.0,
                    )
                    v_sub = (v_sub * v_scale[:, None]).to(q.dtype)
            acc_o += tl.dot(p_sub.to(v_sub.dtype), v_sub)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    acc_o = tl.reshape(acc_o, 1, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention over the selected blocks.

    On gfx950 a program owns ``_SPARSE_ATTN_BLOCK_Q`` consecutive query tokens and
    walks the union of their selections, guarded by a one-token-per-program
    fallback; elsewhere a program owns one token. ``topk_idx`` is per token either
    way. Requires ``cu_seqlens_q[-1] == q.shape[0]``, since the union buffers are
    sized from ``total_q``.
    """
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    block_size_q = _SPARSE_ATTN_BLOCK_Q
    attend_args = dict(
        q_ptr=q,
        kv_cache_ptr=kv_cache,
        k_scale_ptr=k_scale_arg,
        v_scale_ptr=v_scale_arg,
        o_ptr=output,
        block_table_ptr=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        gqa_group_size=gqa_group_size,
        head_dim=head_dim,
        sm_scale=sm_scale,
        stride_qn=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kv_blk=kv_cache.stride(0),
        stride_kv_h=kv_cache.stride(1),
        stride_kv_pos=kv_cache.stride(2),
        stride_kv_d=kv_cache.stride(3),
        stride_ks_h=stride_ks_h,
        stride_ks_t=stride_ks_t,
        stride_vs_h=stride_vs_h,
        stride_vs_t=stride_vs_t,
        stride_on=output.stride(0),
        stride_oh=output.stride(1),
        stride_od=output.stride(2),
        stride_bt_b=block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
    )

    def launch_per_token(union_len, union_max, group_q):
        """One query token per program. Guarded when a union buffer is supplied."""
        _gqa_sparse_fwd_kernel[(max_query_len, num_kv_heads, batch)](
            t_ptr=topk_idx,
            u_len_ptr=union_len if union_len is not None else topk_idx,
            max_topk=topk,
            stride_th=topk_idx.stride(0),
            stride_tn=topk_idx.stride(1),
            stride_tk=topk_idx.stride(2),
            stride_ul_h=union_len.stride(0) if union_len is not None else 1,
            SUB_K=_SPARSE_ATTN_SUB_K,
            GUARDED=union_len is not None,
            UNION_MAX=union_max,
            GROUP_Q=group_q,
            **attend_args,
            **_sparse_attn_prefill_kwargs(use_union=False),
        )

    if block_size_q == 1:
        # Groups are tokens: nothing to deduplicate, nothing to route.
        launch_per_token(None, 0, 1)
        return

    # Block ids share an int32 with the membership mask, low bits first.
    assert block_table.shape[1] <= _UNION_BLK_MASK, (
        f"{block_table.shape[1]} blocks exceeds the {_UNION_BLK_MASK} addressable "
        "by the packed union representation"
    )
    if use_fp8:
        max_frac = _SPARSE_ATTN_UNION_MAX_FRAC_FP8
    else:
        max_frac = _SPARSE_ATTN_UNION_MAX_FRAC_DEFAULT
    union_max = int(block_size_q * topk * max_frac)
    capacity = block_size_q * triton.next_power_of_2(topk)
    # One row per query group; each request rounds up to at most one extra.
    num_groups = triton.cdiv(total_q, block_size_q) + batch
    i32 = {"dtype": torch.int32, "device": q.device}
    u_blk = torch.empty((num_kv_heads, num_groups, capacity), **i32)
    u_len = torch.empty((num_kv_heads, num_groups), **i32)
    grid_groups = (triton.cdiv(max_query_len, block_size_q), num_kv_heads, batch)
    _union_build_kernel[grid_groups](
        topk_idx,
        u_blk,
        u_len,
        cu_seqlens_q,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        u_blk.stride(0),
        u_blk.stride(1),
        u_len.stride(0),
        num_groups,
        BLOCK_SIZE_Q=block_size_q,
        BITS_SHIFT=_UNION_BITS_SHIFT,
    )
    _gqa_sparse_union_fwd_kernel[grid_groups](
        u_blk_ptr=u_blk,
        u_len_ptr=u_len,
        stride_ub_h=u_blk.stride(0),
        stride_ub_g=u_blk.stride(1),
        stride_ul_h=u_len.stride(0),
        BLOCK_SIZE_Q=block_size_q,
        SUB_K=_SPARSE_ATTN_UNION_SUB_K,
        UNION_MAX=union_max,
        BITS_SHIFT=_UNION_BITS_SHIFT,
        **attend_args,
        **_sparse_attn_prefill_kwargs(use_union=True),
    )
    # |union| <= candidates, so a threshold at or above that took every group.
    if union_max < block_size_q * topk:
        launch_per_token(u_len, union_max, block_size_q)
