# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing stages: biased sigmoid top-k of one token, and the expert sort.

Top-k writes at device scope (``st_wt``) because the sort may run on another
XCD. The sort's scattered rows are plain stores (device-scope ones cost ~6 us
at M=64); the routed flag after them releases, and a gemm tile on another XCD
only drops its L1 before reading them.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    key_value,
    lds_f32,
    lds_i32,
    lds_or,
    order_key,
    popc,
    popc64,
    rcp,
    route_mark,
    st_plain,
    st_wt,
    wave_kth_key,
)
from vllm.models.kimi_k3.amd.mono.common.plan import BM, N_WAVES, THREADS, WAVE

LOG2E = 1.4426950408889634


@flyc.jit
def topk_token(
    lds_base,
    arg_logits,
    arg_bias,
    arg_tw,
    arg_ti,
    arg_out,
    tok,
    tid,
    lane,
    wave,
    arg_trace,
    *,
    NE,
    TOPK,
    D_HIDDEN,
    TRACE,
    L_PART,
    L_PIV,
    L_WTOT,
    L_CV,
    L_CI,
    L_CS,
    L_SEL,
):
    """Biased sigmoid top-k of one token over the whole workgroup; zeroes the
    token's output row for gemm2's atomic adds.

    Matches aiter's topk_reg_kernel: score = sigmoid(logit), select on
    score + bias, ties to the lower expert id, weights = selected scores
    renormalized to sum 1.

    The pivot is the largest over waves of each wave's TOPK-th largest lane
    maximum (bisection on ballots), so every winner reaches it. The candidates
    at or above it are compacted by ballot + popcount and ranked: with at most
    WAVE candidates each wave ranks lane c against a quarter of them and wave 0
    sums the partial ranks; more candidates take the all-pairs rank.
    """
    EPW = NE // N_WAVES
    EPL = (EPW + WAVE - 1) // WAVE
    s_piv = lds_f32(lds_base, L_PIV)
    wtot = lds_i32(lds_base, L_WTOT)
    s_cv = lds_f32(lds_base, L_CV)
    s_ci = lds_i32(lds_base, L_CI)
    s_cs = lds_f32(lds_base, L_CS)
    s_sel = lds_f32(lds_base, L_SEL)

    def tmark(k):
        if const_expr(TRACE):
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            route_mark(tid, arg_trace, k)

    tmark(2)
    row64 = tok * fx.Int32(D_HIDDEN // 4)
    for i in range_constexpr((D_HIDDEN // 4 + THREADS - 1) // THREADS):
        if const_expr((i + 1) * THREADS <= D_HIDDEN // 4):
            st_wt(arg_out, row64 + tid + fx.Int32(i * THREADS), fx.Int64(0), 8)
        else:
            if tid + fx.Int32(i * THREADS) < fx.Int32(D_HIDDEN // 4):
                st_wt(arg_out, row64 + tid + fx.Int32(i * THREADS), fx.Int64(0), 8)

    logits = global_typed_ptr(arg_logits, T.f32)
    bias = global_typed_ptr(arg_bias, T.f32)
    neg_inf = fx.Float32(float("-inf"))
    row = tok * fx.Int32(NE)
    ids, sig, choice = [], [], []
    for j in range_constexpr(EPL):
        e = wave * fx.Int32(EPW) + fx.Int32(j * WAVE) + lane
        ok = fx.Int32(j * WAVE) + lane < fx.Int32(EPW)
        ec = ok.select(e, fx.Int32(0))
        x = fx.Float32(logits[row + ec])
        s = rcp(fx.Float32(1.0) + (x * fx.Float32(-LOG2E)).exp2())
        # A NaN score would make the pivot NaN and select nothing, leaving the
        # token's ids unwritten for the sort (padded graph rows can be NaN).
        s = (s == s).select(s, fx.Float32(0.0))
        ids.append(e)
        sig.append(s)
        choice.append(ok.select(s + fx.Float32(bias[ec]), neg_inf))
    # Candidate slots past n_cand read as losers: ordered before the
    # compaction writes by the pivot barrier.
    if tid < fx.Int32(WAVE):
        s_cv[tid] = neg_inf
        s_ci[tid] = fx.Int32(0x7FFFFFFF)
    tmark(3)

    lmax = choice[0]
    for j in range_constexpr(1, EPL):
        lmax = lmax.maximumf(choice[j])
    wpiv = key_value(wave_kth_key(order_key(lmax), TOPK))
    if lane == fx.Int32(0):
        s_piv[wave] = wpiv
    gpu.barrier()
    pv = fx.Float32(s_piv[0])
    for wv in range_constexpr(1, N_WAVES):
        pv = pv.maximumf(fx.Float32(s_piv[fx.Int32(wv)]))

    tmark(4)
    # Slot of each hit: earlier hits of the wave (j-major, then lane) by
    # ballot + popcount, plus the hits of lower waves.
    hits = [choice[j] >= pv for j in range_constexpr(EPL)]
    lt = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
    run = fx.Int32(0)
    slots = []
    for j in range_constexpr(EPL):
        mask = fx.Int64(rocdl.ballot(T.i64, hits[j]))
        slots.append(run + popc64(mask & lt))
        run = run + popc64(mask)
    if lane == fx.Int32(0):
        wtot[wave] = run
    gpu.barrier()
    wbase = fx.Int32(0)
    n_cand = fx.Int32(0)
    for wv in range_constexpr(N_WAVES):
        t = fx.Int32(wtot[fx.Int32(wv)])
        wbase = wbase + (fx.Int32(wv) < wave).select(t, fx.Int32(0))
        n_cand = n_cand + t
    for j in range_constexpr(EPL):
        if hits[j]:
            s_cv[wbase + slots[j]] = choice[j]
            s_ci[wbase + slots[j]] = ids[j]
            s_cs[wbase + slots[j]] = sig[j]
    gpu.barrier()
    tmark(5)

    ti = arg_ti
    tw = arg_tw
    QPW = WAVE // N_WAVES
    s_part = lds_i32(lds_base, L_PART)
    fast = n_cand <= fx.Int32(WAVE)
    if fast:
        mv = fx.Float32(s_cv[lane])
        mi = fx.Int32(s_ci[lane])
        part = fx.Int32(0)
        for j in range_constexpr(QPW):
            ov = fx.Float32(s_cv[wave * fx.Int32(QPW) + fx.Int32(j)])
            oi = fx.Int32(s_ci[wave * fx.Int32(QPW) + fx.Int32(j)])
            beats = (ov > mv) | ((ov == mv) & (oi < mi))
            part = part + beats.select(fx.Int32(1), fx.Int32(0))
        s_part[tid] = part
    n_slow = fast.select(fx.Int32(0), n_cand)
    for c in range(tid, n_slow, fx.Int32(THREADS)):
        mv = fx.Float32(s_cv[c])
        mi = fx.Int32(s_ci[c])
        rank = fx.Int32(0)
        for q in range(fx.Int32(0), n_cand, fx.Int32(1)):
            ov = fx.Float32(s_cv[q])
            oi = fx.Int32(s_ci[q])
            beats = (ov > mv) | ((ov == mv) & (oi < mi))
            rank = rank + beats.select(fx.Int32(1), fx.Int32(0))
        if rank < fx.Int32(TOPK):
            st_wt(ti, tok * fx.Int32(TOPK) + rank, mi, 4)
            s_sel[rank] = fx.Float32(s_cs[c])
    gpu.barrier()
    tmark(6)
    if fast & (wave == fx.Int32(0)):
        rank = fx.Int32(s_part[lane])
        for wv in range_constexpr(1, N_WAVES):
            rank = rank + fx.Int32(s_part[lane + fx.Int32(wv * WAVE)])
        sel = (lane < n_cand) & (rank < fx.Int32(TOPK))
        sv = sel.select(fx.Float32(s_cs[lane]), fx.Float32(0.0))
        tot = sv
        for sh in range_constexpr(6):
            tot = tot + tot.shuffle_xor(fx.Int32(1 << sh), fx.Int32(WAVE))
        if sel:
            st_wt(ti, tok * fx.Int32(TOPK) + rank, fx.Int32(s_ci[lane]), 4)
            st_wt(
                tw,
                tok * fx.Int32(TOPK) + rank,
                (tot > fx.Float32(0.0)).select(sv / tot, fx.Float32(0.0)),
                4,
            )
    if (wave == fx.Int32(0)) & (n_slow != fx.Int32(0)):
        mine = lane < fx.Int32(TOPK)
        sv = mine.select(fx.Float32(s_sel[lane & fx.Int32(TOPK - 1)]), fx.Float32(0.0))
        tot = sv
        for sh in range_constexpr(6):
            tot = tot + tot.shuffle_xor(fx.Int32(1 << sh), fx.Int32(WAVE))
        if mine:
            st_wt(
                tw,
                tok * fx.Int32(TOPK) + lane,
                (tot > fx.Float32(0.0)).select(sv / tot, fx.Float32(0.0)),
                4,
            )
    tmark(7)


@flyc.jit
def sort_routes(
    lds_base,
    arg_tw,
    arg_ti,
    arg_stids,
    arg_sw,
    arg_eids,
    arg_cumsum,
    arg_mind,
    i32_M,
    tid,
    lane,
    wave,
    arg_trace,
    *,
    NE,
    TOPK,
    M_MAX,
    TRACE,
    L_CNT,
    L_BMAP,
    L_BFILL,
):
    """Expert-sorted layout as moe_sort_quant's one-shot sort writes it.

    Experts ascend; each takes ceil(routes / BM) consecutive m-blocks and pads
    its last block with token M. An expert has at most M_MAX routes (one per
    token). Bitmap k marks the experts with more than k * BM routes, so an
    expert's first m-block is the number of set bits below it over all bitmaps:
    a scan over NE / 32 words that every wave does on its own, by ballots over
    the bits of the per-word counts, with no scan over all NE experts.
    """
    CNT_WORDS = (NE + THREADS - 1) // THREADS * THREADS
    EPT = CNT_WORDS // THREADS
    RPT = (M_MAX * TOPK + THREADS - 1) // THREADS
    NMAP = (M_MAX + BM - 1) // BM
    BW = (NE + 31) // 32
    PC_BITS = (32 * NMAP).bit_length()
    assert BW <= WAVE and NMAP * BW <= THREADS and THREADS <= CNT_WORDS
    cnt = lds_i32(lds_base, L_CNT)
    bmap = lds_i32(lds_base, L_BMAP)
    bfill = lds_i32(lds_base, L_BFILL)

    def tmark(k):
        if const_expr(TRACE):
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            route_mark(tid, arg_trace, k)

    for i in range_constexpr(EPT):
        cnt[tid + fx.Int32(i * THREADS)] = fx.Int32(0)
    if tid < fx.Int32(NMAP * BW):
        bmap[tid] = fx.Int32(0)
    gpu.barrier()

    n_routes = i32_M * fx.Int32(TOPK)
    ti = global_typed_ptr(arg_ti, T.i32)
    tw = global_typed_ptr(arg_tw, T.f32)
    lds_i64 = fx.Int64(lds_base)
    es, ws, olds = [], [], []
    for k in range_constexpr(RPT):
        r = tid + fx.Int32(k * THREADS)
        has = r < n_routes
        rc = has.select(r, fx.Int32(0))
        e = fx.Int32(ti[rc])
        es.append(e)
        ws.append(fx.Float32(tw[rc]))
        # Lanes without a route add 0 to a word of their own (not yet used
        # fill words): one shared word would serialise their atomics.
        slot_w = has.select(fx.Int32(L_CNT) + e, fx.Int32(L_BFILL) + tid)
        old = fx.Int32(
            comm_ops.atomic_add_lds(
                lds_i64 + fx.Int64(slot_w) * fx.Int64(4),
                has.select(fx.Int32(1), fx.Int32(0)),
            )
        )
        olds.append(old)
        if has & ((old & fx.Int32(BM - 1)) == fx.Int32(0)):
            word = (old >> fx.Int32(BM.bit_length() - 1)) * fx.Int32(BW) + (
                e >> fx.Int32(5)
            )
            lds_or(
                lds_i64 + fx.Int64(L_BMAP * 4) + fx.Int64(word) * fx.Int64(4),
                fx.Int32(1) << (e & fx.Int32(31)),
            )
    gpu.barrier()
    tmark(10)

    lw = lane < fx.Int32(BW)
    lwc = lw.select(lane, fx.Int32(0))
    pc = fx.Int32(0)
    for m in range_constexpr(NMAP):
        pc = pc + popc(bmap[lwc + fx.Int32(m * BW)])
    pc = lw.select(pc, fx.Int32(0))
    pc_bits = [
        fx.Int64(
            rocdl.ballot(T.i64, ((pc >> fx.Int32(b)) & fx.Int32(1)) != fx.Int32(0))
        )
        for b in range_constexpr(PC_BITS)
    ]
    total = fx.Int32(0)
    for b in range_constexpr(PC_BITS):
        total = total + (popc64(pc_bits[b]) << fx.Int32(b))

    for k in range_constexpr(RPT):
        r = tid + fx.Int32(k * THREADS)
        e = es[k]
        old = olds[k]
        w5 = e >> fx.Int32(5)
        below = (fx.Int32(1) << (e & fx.Int32(31))) - fx.Int32(1)
        lt_w = (fx.Int64(1) << fx.Int64(w5)) - fx.Int64(1)
        base = fx.Int32(0)
        for b in range_constexpr(PC_BITS):
            base = base + (popc64(pc_bits[b] & lt_w) << fx.Int32(b))
        for m in range_constexpr(NMAP):
            base = base + popc(fx.Int32(bmap[w5 + fx.Int32(m * BW)]) & below)
        if r < n_routes:
            row = base * fx.Int32(BM) + old
            tok = r // fx.Int32(TOPK)
            slot = r - tok * fx.Int32(TOPK)
            st_plain(arg_stids, row, tok | (slot << fx.Int32(24)), 4)
            st_plain(arg_mind, row, tok, 4)
            st_plain(arg_sw, row, ws[k], 4)
            if old == fx.Int32(0):
                ce = fx.Int32(cnt[e])
                for m in range_constexpr(NMAP):
                    if ce > fx.Int32(m * BM):
                        st_plain(arg_eids, base + fx.Int32(m), e, 4)
                        left = ce - fx.Int32(m * BM)
                        bfill[base + fx.Int32(m)] = (left < fx.Int32(BM)).select(
                            left, fx.Int32(BM)
                        )
    if tid == fx.Int32(0):
        st_plain(arg_cumsum, fx.Int32(0), total * fx.Int32(BM), 4)
        st_plain(arg_cumsum, fx.Int32(1), i32_M, 4)
    gpu.barrier()
    tmark(11)

    for rr in range(tid, total * fx.Int32(BM), fx.Int32(THREADS)):
        b = rr >> fx.Int32(BM.bit_length() - 1)
        if (rr & fx.Int32(BM - 1)) >= fx.Int32(bfill[b]):
            st_plain(arg_stids, rr, i32_M, 4)
            st_plain(arg_mind, rr, i32_M, 4)
            st_plain(arg_sw, rr, fx.Float32(0.0), 4)
