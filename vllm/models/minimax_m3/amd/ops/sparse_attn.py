# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm gfx942/gfx950 block-sparse GQA prefill kernel for MiniMax-M3.

Only the prefill path is specialized on CDNA: each 128-token KV block is split
into SUB_K-token sub-tiles to right-size the per-block QK/PV MFMAs, and a group
of ``BLOCK_SIZE_Q`` consecutive query tokens shares one program, iterating the
*union* of their selected blocks so each block is fetched once for the whole
group instead of once per query token. Everything else -- the decode split-K
kernels, the FP8 dtype set, the sparse block size -- is reused unchanged from
``common.ops.sparse_attn``.

The union matters because the kernel is purely KV-gather bound: measured on
gfx950 a load-only twin with all the attention math deleted costs 103% of the
full kernel, so the arithmetic is free and only fetched bytes count. The
lightning indexer's per-token top-k sets overlap heavily between neighbouring
query tokens -- measured over every call of a live 60000-token prefill, the union
of 4 consecutive tokens' sets holds 21.8 of a possible 64 blocks on average -- so
the union cuts the KV traffic ~3x while growing the QK/PV work only by
|union| / topk, most of which the MFMAs absorb for free by getting 4x taller.

Union size is a property of the indexer's output, so it varies by layer, and a
few layers select badly enough that the extra QK/PV work costs more than the
traffic it saves (measured break-even: |union| = 37.8-42.0 of 64). The attend kernel
is therefore launched twice over the same shape-constant grid, once per path, and
each program reads its group's |union| and keeps only the groups its path owns
(see _SPARSE_ATTN_UNION_MAX). The two launches partition the query groups, so
the decision costs no host-device synchronization and the launch sequence is
fixed at trace time -- it captures into a CUDA graph unchanged.
"""

import torch

from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _FP8_DTYPES,
    _KV_SCALE_NONE,
    SPARSE_BLOCK_SIZE,
    _kv_scale_args,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms.rocm import on_mi3xx
from vllm.triton_utils import tl, triton

__all__ = ["minimax_m3_sparse_attn", "minimax_m3_sparse_attn_decode"]


# Sub-tile width for the prefill kernel's per-block QK/PV GEMMs. Must divide
# SPARSE_BLOCK_SIZE. 32 on both CDNA archs, re-tuned jointly with num_warps once
# the union made each program four query tokens tall: at group 4 the widest
# sub-tile that used to pay off (64 on gfx950) now overshoots, because the
# accumulator is 4x taller and a narrower sub-tile is what keeps the working set
# small enough to keep two waves in flight over the next block's fetch. Measured
# over the served prefill chunk geometries: 1.03x on the FP8 cache and 1.14x on
# BF16 against SUB_K=64 / num_warps=4, faster in every geometry of both.
_SPARSE_ATTN_SUB_K = SPARSE_BLOCK_SIZE // 4

# Union members are packed into one int32: the block id in the low bits, the
# per-query membership mask above it. The inner loop is latency bound on its
# scalar loads, so folding the mask into the id it travels with removes a load
# per union member (and a whole [.., BLOCK_SIZE_N] int32 tensor per call).
# 16 bits of id covers 65536 blocks == 8.4M tokens of context, well past the
# 133120 the model is served at; the mask needs BLOCK_SIZE_Q bits above that.
_UNION_BITS_SHIFT = 16
_UNION_BLK_MASK = (1 << _UNION_BITS_SHIFT) - 1

# Query tokens per program for the prefill attend. The program walks the union of
# the group's selected blocks, so KV traffic falls by the group's dedup factor
# while QK/PV work grows only by |union| / topk. 4 is the measured optimum on
# gfx950 at the 32k-token prefill chunk: it takes ~3.2x off the KV traffic, which
# is enough to move the kernel off the gather wall, and past that point the extra
# masked QK/PV work costs more than the traffic it saves. 1 restores the
# one-token-per-program behaviour exactly. Non-CDNA AMD archs keep 1 (untuned).
#
# 8 was measured and rejected, which is worth recording because the arithmetic
# looks like it should win: |union| is 25.96 of 128 at group 8 against 21.75 of 64
# at group 4, i.e. 1.68x less KV traffic per query token. It does not pay, and not
# for the reason one would guess:
#   * it is not spilling -- group 8 at num_warps=4 compiles to 256 VGPRs and 0
#     spill bytes (group 4: 156 and 0)
#   * the extra masked QK/PV work can be removed entirely, by fetching at group 8
#     but computing in sub-blocks of 4 and skipping a sub-block whose 4 queries all
#     missed the block. Per query token that makes compute |union(sub-block)| and
#     traffic |union(group)| / group, i.e. group-4 compute at group-8 traffic, in
#     ONE pass over the KV. Implemented and measured: it delivers exactly that,
#     achieved bandwidth falling 6.33 -> 3.94 TB/s, and the time does not move.
#   * what kills it is that group 8 halves the program count and nearly doubles the
#     per-program register footprint, and this kernel is latency-bound rather than
#     byte-bound at group 4, so there is no bandwidth headroom to convert. Halving
#     the program count at IDENTICAL traffic and compute (num_q_loop=2) costs
#     1.10x on its own, which is about what the traffic saving is worth.
# Blended over 456 live calls the group-8 variant came out at 0.985x of this one.
_SPARSE_ATTN_BLOCK_Q = 4 if on_mi3xx() else 1

# Locality guard: query groups whose union exceeds this many blocks are attended
# one query token at a time instead, because past that point the union's extra
# masked QK/PV work costs more than the KV traffic it saves.
#
# Measured on gfx950 at the served 2.30 GiB KV allocation. Break-even against
# the guarded fallback is 37.8, 42.0 and 40.2 blocks across the two single-
# sequence chunks and the production-dominant ragged geometry. Blended time is
# flat from 30 through 44, but the worst case starts moving above 38.
#
# 38 is therefore the top of the blended plateau and the last threshold before
# the tail regresses: 1.799-1.801x blended over 456 live calls, with no call
# below 0.95x. A single constant is sufficient across the measured geometries.
# Set above BLOCK_SIZE_Q * topk to disable the fallback launch entirely.
_SPARSE_ATTN_UNION_MAX = 38

_SPARSE_ATTN_PREFILL_KWARG: dict | None = None


def _sparse_attn_prefill_kwargs(use_union: bool = True) -> dict:
    """MFMA + pipeline launch params for the sub-tiled prefill kernel.

    gfx942 and gfx950 share the same params: ``matrix_instr_nonkdim=16`` /
    ``kpack=2`` select the MFMA_16x16 path and ``num_stages=1`` fits LDS and is
    fastest in the sweep. ``num_warps`` is tuned per path: the union path's
    ``num_warps=4`` is tuned jointly with ``_SPARSE_ATTN_BLOCK_Q``, because the
    union makes each program do ``|union| / topk`` more QK/PV work than a single
    query token did, and four waves both split the
    ``BLOCK_SIZE_Q * gqa_group_size`` row accumulator across enough registers to
    avoid spilling and give the scheduler something to run while the next KV
    block is in flight (1.10x over ``num_warps=1`` at group 4). The guard's
    one-token-per-program fallback has a quarter of the rows and keeps
    ``num_warps=1``. Only the sub-tile width (``_SPARSE_ATTN_SUB_K``) differs by
    arch. Empty on other AMD archs. Cached: arch is fixed per process.
    """
    global _SPARSE_ATTN_PREFILL_KWARG
    if _SPARSE_ATTN_PREFILL_KWARG is None:
        kwarg: dict = {}
        if on_mi3xx():
            kwarg = {
                "matrix_instr_nonkdim": 16,
                "kpack": 2,
                "num_stages": 1,
            }
        _SPARSE_ATTN_PREFILL_KWARG = kwarg
    if not _SPARSE_ATTN_PREFILL_KWARG:
        return _SPARSE_ATTN_PREFILL_KWARG
    return {**_SPARSE_ATTN_PREFILL_KWARG, "num_warps": 2 if use_union else 1}


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_N": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["max_topk"]),
        # one wave per 32 candidate slots: the dedup compares all BLOCK_SIZE_N
        # slots against each other, so the tile it works on is BLOCK_SIZE_N^2
        "num_warps": lambda args: max(1, min(8, args["BLOCK_SIZE_N"] // 32)),
    }
)
@triton.jit
def _union_build_kernel(
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk], per query TOKEN
    u_blk_ptr,  # out: [num_kv_heads, batch, num_q_blocks, BLOCK_SIZE_N] int32
    u_len_ptr,  # out: [num_kv_heads, batch, num_q_blocks] int32, |union|
    cu_seqlens_q,
    max_topk,
    stride_th,
    stride_tn,
    stride_tk,
    stride_ub_h,
    stride_ub_b,
    stride_ub_g,
    stride_ul_h,
    stride_ul_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BITS_SHIFT: tl.constexpr,  # membership mask sits above this bit
):
    """Per query group, compact the group's selected blocks into their union.

    Writes, for each group of ``BLOCK_SIZE_Q`` consecutive query tokens, the
    distinct block ids it selected (``u_blk``, dense in ``[0, u_len)``), each
    packed with a bitmask of which queries in the group actually selected it
    (bit ``r`` above ``BITS_SHIFT`` <-> the group's r-th query). This keeps the
    attend kernel's inner loop free of any cross-lane work: it reads one int32
    per union member and turns it into both the page to fetch and the per-query
    predicate with two shifts. Reads ``topk_idx`` exactly as the selection kernel
    writes it (per query token, -1 padded), so the indexer's contract is
    untouched.
    """
    pid_g = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    len_ptr = u_len_ptr + pid_kh * stride_ul_h + pid_b * stride_ul_b + pid_g
    q_tok0 = pid_g * BLOCK_SIZE_Q
    if q_tok0 >= q_len:
        tl.store(len_ptr, 0)
        return
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
    tf = tl.reshape(t_grp, (BLOCK_SIZE_N,))
    sid = tl.arange(0, BLOCK_SIZE_N)
    eq = tf[:, None] == tf[None, :]
    # keep each distinct id at its first occurrence, and compact by rank
    first = tl.sum(tl.where((sid[None, :] < sid[:, None]) & eq, 1, 0), axis=1) == 0
    first = first & (tf >= 0)
    rank = tl.maximum(tl.cumsum(first.to(tl.int32), axis=0) - 1, 0)
    # bit r of bits[s] <-> query r of the group has tf[s] in its own top-k
    has = tl.max(
        tl.reshape(eq.to(tl.int32), (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_T)),
        axis=2,
    )
    bits = tl.sum(has << off_g[None, :], axis=1)
    u_off = pid_kh * stride_ub_h + pid_b * stride_ub_b + pid_g * stride_ub_g
    tl.store(u_blk_ptr + u_off + rank, tf | (bits << BITS_SHIFT), mask=first)
    tl.store(len_ptr, tl.sum(first.to(tl.int32), axis=0))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
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
    num_kv_heads,
    gqa_group_size,
    head_dim,
    num_q_loop,
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
    stride_ub_b,
    stride_ub_g,
    stride_ul_h,
    stride_ul_b,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    SUB_K: tl.constexpr,  # CDNA only: KV sub-tile width (see _IS_MI3XX)
    UNION_MAX: tl.constexpr,  # locality guard threshold (_SPARSE_ATTN_UNION_MAX)
    BITS_SHIFT: tl.constexpr,  # membership mask sits above this bit of u_blk
    BLK_MASK: tl.constexpr,  # block id occupies the bits below BITS_SHIFT
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    # Query blocks, not query tokens: BLOCK_SIZE_Q consecutive tokens per program.
    q_block_len = (q_len + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    u_len_base = u_len_ptr + pid_kh * stride_ul_h + pid_b * stride_ul_b
    # Locality guard: drop the groups whose union is too big to pay for itself.
    # _gqa_sparse_dense_fwd_kernel takes exactly those, testing the same |union|
    # against the same threshold the other way round, so the two launches
    # partition the query groups -- none attended twice, none dropped -- with no
    # host-side branch on a device value (see minimax_m3_sparse_attn).
    if tl.load(u_len_base + pid_q) > UNION_MAX:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    off_g = tl.arange(0, BLOCK_SIZE_Q)
    u_base = pid_kh * stride_ub_h + pid_b * stride_ub_b
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        q_tok0 = pid_q_j * BLOCK_SIZE_Q  # first query token of this group
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        # A query in the group skips the blocks it did not select, so a block can
        # be fully masked for some rows. -inf as the running max would then give
        # exp2(-inf - -inf) = NaN, so the max starts at a finite sentinel and the
        # normalizer is accumulated linearly instead of in log space.
        m_i = tl.full((BLOCK_SIZE_QH,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

        # CDNA: process each 128-token KV block in SUB_K-token sub-tiles so
        # each QK/PV MFMA is right-sized. Numerically equivalent to the dense
        # path below (flash-softmax reassociation).
        NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K
        # Walk the UNION of the group's selected blocks: each distinct block is
        # fetched once for the whole group instead of once per query token. The
        # union was compacted by _union_build_kernel, so this costs the same two
        # scalar loads per block that the one-token-per-program version paid.
        u_off = u_blk_ptr + u_base + pid_q_j * stride_ub_g
        for s in tl.range(tl.load(u_len_base + pid_q_j)):
            packed = tl.load(u_off + s)
            blk = packed & BLK_MASK
            # bit r above BITS_SHIFT <-> the group's r-th query selected this block
            in_set = ((packed >> BITS_SHIFT) >> off_g) & 1 != 0
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
                off_q_sub = (
                    tl.arange(0, BLOCK_SIZE_Q)[:, None]
                    + q_tok0
                    + prefix_len
                    - off_sub[None, :]
                )
                qk_sub = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, SUB_K), dtype=tl.float32)
                # causal: q_abs_pos - k_off >= block_start (c)
                qk_sub += tl.where(off_q_sub[:, None, :] >= c, 0, float("-inf"))
                # sparsity: this query only attends to the blocks IT selected,
                # not to the rest of the group's union
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
        # l_i is 0 only for padding rows past q_len, which the store drops.
        acc_o = acc_o / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
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
def _gqa_sparse_dense_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk], per query TOKEN
    u_len_ptr,  # per-group |union|, read only to route (see UNION_MAX)
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
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
    stride_ul_b,
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
    UNION_MAX: tl.constexpr,  # locality guard threshold (_SPARSE_ATTN_UNION_MAX)
    GROUP_Q: tl.constexpr,  # union group size, i.e. the prologue's BLOCK_SIZE_Q
):
    """One query token per program, over that token's own top-k blocks.

    The guard's other half: it takes the query groups whose union grew too large
    for _gqa_sparse_fwd_kernel to pay for, and attends them the way the non-CDNA
    kernel does, so a badly-selecting layer costs what it cost before the union
    lever existed instead of more.

    This is deliberately a second kernel rather than a constexpr branch inside
    the union one. Sharing the loop body was tried first and cost 12% of the
    shipped kernel on the prefix=0 chunk while costing nothing on the other --
    that chunk's 33 GB of gathers land in 2.45 ms, i.e. 13.6 TB/s, which is twice
    MALL bandwidth, so it runs out of cache and is limited by waves in flight,
    where whatever the shared body did to the register allocation shows up. As
    its own kernel the fallback keeps the shipped kernel's own allocation and
    measures within 0.3% of it, which is what makes the guard able to recover a
    bad call fully rather than only partly.
    """
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)  # one query TOKEN per program
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:
        return
    # Locality guard, the complement of the one in _gqa_sparse_fwd_kernel: keep
    # only the tokens whose query group the union path gave up on. Same buffer,
    # same threshold, opposite comparison, so the two launches partition the
    # groups exactly and neither needs to know what the other decided.
    u_len_base = u_len_ptr + pid_kh * stride_ul_h + pid_b * stride_ul_b
    if tl.load(u_len_base + pid_q // GROUP_Q) <= UNION_MAX:
        return
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    t_ptr_q = t_ptr + (q_start + pid_q) * stride_tn + pid_kh * stride_th
    off_t = tl.arange(0, BLOCK_SIZE_T)
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
    # Log-space normalizer, exactly as the non-CDNA kernel accumulates it. The
    # union kernel cannot use this -- a block masked away for a whole query would
    # give exp2(-inf - -inf) = NaN -- but here the first sub-tile of a selected
    # block always contains the block's own start position, which every query that
    # selected the block can see, so the running max is finite from the first
    # iteration on. Keeping the shipped form matters: the finite-sentinel version
    # measured 11% slower than it on the prefix=0 chunk.
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K
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

    Each program owns ``_SPARSE_ATTN_BLOCK_Q`` consecutive query tokens and
    iterates the union of their selected blocks; ``topk_idx`` stays per query
    token, so the indexer's output contract is unchanged.
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
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )
    # Compact each query group's selected blocks into their union first, so the
    # attend kernel below fetches every distinct block once per group instead of
    # once per query token. topk_idx itself is read as-is (per query token).
    block_size_q = _SPARSE_ATTN_BLOCK_Q
    num_q_blocks = triton.cdiv(max_query_len, block_size_q)
    n_slot = block_size_q * triton.next_power_of_2(topk)
    u_blk = torch.empty(
        (num_kv_heads, batch, num_q_blocks, n_slot),
        dtype=torch.int32,
        device=q.device,
    )
    u_len = torch.empty(
        (num_kv_heads, batch, num_q_blocks), dtype=torch.int32, device=q.device
    )
    _union_build_kernel[(num_q_blocks, num_kv_heads, batch)](
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
        u_blk.stride(2),
        u_len.stride(0),
        u_len.stride(1),
        BLOCK_SIZE_Q=block_size_q,
        BITS_SHIFT=_UNION_BITS_SHIFT,
    )
    # Locality guard: attend the groups whose union is small enough to pay for
    # itself with one program per group over the union, and the rest one query
    # token at a time. Each program decides from its own group's |union|, so both
    # launches are issued unconditionally over shape-derived grids and nothing
    # reads a device value on the host -- this captures into a CUDA graph. A
    # group belongs to exactly one launch, so the two write disjoint rows of
    # `output` and their order does not matter.
    _gqa_sparse_fwd_kernel[(num_q_blocks, num_kv_heads, batch)](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        u_blk,
        u_len,
        output,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        u_blk.stride(0),
        u_blk.stride(1),
        u_blk.stride(2),
        u_len.stride(0),
        u_len.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=block_size_q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        SUB_K=_SPARSE_ATTN_SUB_K,
        UNION_MAX=_SPARSE_ATTN_UNION_MAX,
        BITS_SHIFT=_UNION_BITS_SHIFT,
        BLK_MASK=_UNION_BLK_MASK,
        **_sparse_attn_prefill_kwargs(),
    )
    # The other half of the guard, over one query token per program. Issued
    # unconditionally: which groups it does work for is decided per program from
    # u_len, so the host never reads a device value, the grid comes from
    # max_query_len, and the launch sequence is the same every call.
    if n_slot > _SPARSE_ATTN_UNION_MAX and block_size_q > 1:
        _gqa_sparse_dense_fwd_kernel[(max_query_len, num_kv_heads, batch)](
            q,
            kv_cache,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            u_len,
            output,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            topk,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h,
            stride_ks_t,
            stride_vs_h,
            stride_vs_t,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            u_len.stride(0),
            u_len.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            SUB_K=_SPARSE_ATTN_SUB_K,
            UNION_MAX=_SPARSE_ATTN_UNION_MAX,
            GROUP_Q=block_size_q,
            **_sparse_attn_prefill_kwargs(use_union=False),
        )
