# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's 5 layers and final norm as one launch for 3..8 requests.

A draft step of up to 64 query rows on one TP4 rank (layout `layout.W_*`). The
differences from `layer`:

- GEMVs: token tiles of 16 rows (a runtime loop; two tiles' activations loaded together)
  reuse each CTA's weights in the same wave split and order, so a row's result matches
  `layer`'s bitwise; o_proj / down partials go to the row's owner rank (row % 4) only.
- Attention: request r gets 48 // n CTAs; CTA c takes a contiguous range of the
  window's blocks, newest first (the first two hold the new K / V rows), with an
  online softmax and the next block's K / V loaded during the current one; one
  partial a CTA.
- Norms: reduce-scatter + all-gather. N CTA j sums the 4 ranks' partials of row
  4 j + rank, adds the residual, norms and quantizes it, and writes the row as
  tagged words into its own symmetric buffer; every rank's N CTA j then pulls rows
  j, j + 16, .. from their owners into local scratch and flags them. The final
  bf16 rows are gathered into `h_out` the same way.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64

from vllm.models.minimax_m3.amd.mono.drafter import attention as att
from vllm.models.minimax_m3.amd.mono.drafter import device as dev
from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.models.minimax_m3.amd.mono.drafter.device import F32, I32, I64
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    CM_SYS,
    fp8_pack4,
    kernel_symbol,
    memrealtime,
    traced,
    uniform,
    xshfl,
)

POOL_WORDS = 5 * att.TILE_WORDS + 512
_STREAM = fx.Stream(None)
NEG_INF = float("-inf")


def ag_put_x8(o, t, Y, tag, sym, tid):
    """Publish row `t` (16 values a thread) as MXFP8 in this rank's gathered rows at
    `sym + Y`: e4m3 words at pairs 4 tid .., 4 e8m0 scales a word from pair 1536."""
    am = fmath.absf(o[0])
    for e in range_constexpr(1, 16):
        am = fx.max(am, fmath.absf(o[e]))
    am = fx.max(am, xshfl(am, 1))
    sb, rs = dev.mx_scale(am)
    wd = [
        fp8_pack4(
            o[4 * m] * rs, o[4 * m + 1] * rs, o[4 * m + 2] * rs, o[4 * m + 3] * rs
        )
        for m in range(4)
    ]
    s0 = sb.to(fx.Int32)
    s2 = xshfl(s0, 2)
    sw = s0 | (s2 << 8) | (xshfl(s0, 4) << 16) | (xshfl(s2, 4) << 24)
    row = (Y + t * layout.W_AGW * 8) // 4  # words
    if tid < 384:
        dev.st(
            sym,
            row + tid * 8,
            fx.Vector.from_elements([wd[0], tag, wd[1], tag], fx.Int32),
        )
        dev.st(
            sym,
            row + tid * 8 + 4,
            fx.Vector.from_elements([wd[2], tag, wd[3], tag], fx.Int32),
        )
        if tid % 8 == 0:
            dev.st(sym, row + (1536 + tid // 8) * 2, dev.pair(sw, tag))


ag_put_x8 = traced(ag_put_x8)


def ag_copy_x8(t, src, Y, tag, x8, xs8, tid, dl, health, rank, site):
    """Copy row `t` from its owner's gathered rows (`src`: the owner's peer base)
    into x8[t] / xs8[t], polling on the tagged words."""
    if tid < 384:
        w, fl = dev.poll_pairs(
            src, Y // 8 + t * layout.W_AGW + tid * 4, 4, tag, dl, health, CM_SYS
        )
        dev.fail(health, fl, rank, site)
        dev.st(
            x8,
            (t * layout.HIDDEN + tid * 16) // 4,
            fx.Vector.from_elements(w, fx.Int32),
            CM_DEV,
        )
    if (tid >= 384) & (tid < 432):
        js = tid - 384
        w, fl = dev.poll_pairs(
            src, Y // 8 + t * layout.W_AGW + 1536 + js, 1, tag, dl, health, CM_SYS
        )
        dev.fail(health, fl, rank, site)
        dev.st(xs8, (t * 192 + js * 4) // 4, w[0], CM_DEV)


ag_copy_x8 = traced(ag_copy_x8)


def ag_put_bf(o, t, Y, tag, sym, tid):
    """Publish final row `t` (16 values a thread) as bf16 pairs at `sym + Y`."""
    if tid < 384:
        row = (Y + (t * layout.ROW_PAIRS + tid * 8) * 8) // 4
        for q in range_constexpr(4):
            dev.st(
                sym,
                row + 4 * q,
                fx.Vector.from_elements(
                    [
                        dev.bfpair(o[4 * q], o[4 * q + 1]),
                        tag,
                        dev.bfpair(o[4 * q + 2], o[4 * q + 3]),
                        tag,
                    ],
                    fx.Int32,
                ),
            )


ag_put_bf = traced(ag_put_bf)


def ag_copy_bf(t, src, Y, tag, out, tid, dl, health, rank, site):
    if tid < 384:
        w, fl = dev.poll_pairs(
            src, Y // 8 + t * layout.ROW_PAIRS + tid * 8, 8, tag, dl, health, CM_SYS
        )
        dev.fail(health, fl, rank, site)
        dev.st(out, t * 3072 + tid * 8, fx.Vector.from_elements(w[:4], fx.Int32))
        dev.st(out, t * 3072 + tid * 8 + 4, fx.Vector.from_elements(w[4:], fx.Int32))


ag_copy_bf = traced(ag_copy_bf)


def build_wide(yw, wait_ticks=1_000_000_000):
    """The `@flyc.jit` launcher of the wide drafter kernel for one TP4 rank.

    `yw`: the byte offset of its symmetric regions in the peer buffer.
    """
    WF = layout.W_F_OFF
    WL = layout.W_L_OFF

    @fx.struct
    class Smem:
        pool: fx.Array[fx.Int32, POOL_WORDS, 16]

    @flyc.kernel(
        name=kernel_symbol("minimax_m3_dspark_drafter_wide", y=yw),
        known_block_size=[layout.THREADS, 1, 1],
    )
    def drafter_wide(
        x_in: Int64,
        h_out: Int64,
        tab: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_gu: Int64,
        s_gu: Int64,
        w_d: Int64,
        s_d: Int64,
        fin_ln: Int64,
        cos_sin: Int64,
        block_table: Int64,
        positions: Int64,
        slots: Int64,
        seq_lens: Int64,
        scratch: Int64,
        sym: Int64,
        pe0: Int64,
        pe1: Int64,
        pe2: Int64,
        pe3: Int64,
        step: Int64,
        health: Int64,
        rank: Int32,
        bt_width: Int32,
        n_rows: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        g4 = lane // 16
        l16 = lane % 16
        pool = fx.SharedAllocator().allocate(Smem).peek().pool.ptr
        st_ = uniform(dev.ld(step, 0))
        dl = memrealtime() + wait_ticks

        def tptr(layer, i):
            v = dev.ld(tab, (layer * 16 + i) * 2, 2)
            return (I64(uniform(v[1])) << 32) | I64(fx.Uint32(uniform(v[0])))

        def flag(name):
            return scratch + I64(WF[name])

        def owner(t):
            o = t % 4
            return (o == 0).select(pe0, (o == 1).select(pe1, (o == 2).select(pe2, pe3)))

        if bid < layout.NG:
            # ======================================================= G: the GEMVs
            red = fx.recast_iter(fx.Float32, pool)
            ii = fx.min(tid, 255)
            rr = ii // 16
            kk = ii % 16
            nt = (n_rows + 15) // 16

            def tile_row(t):
                """The lane's token row of tile t; rows past n_rows repeat the last."""
                return 16 * t + fx.min(l16, fx.max(n_rows - 16 * t, I32(1)) - 1)

            def rs_push(t, col0, Y, tag):
                """Push token tile `t`'s rows col0 .. col0 + 31, summed over the
                waves, as bf16 pairs to each token row's owner rank (row % 4)."""
                k = ii // 16
                rp = ii % 16
                tt = rp // 8
                r0 = (2 * rp) % 16
                lo = dev.red_sum(red, tt, r0, k)
                hi = dev.red_sum(red, tt, r0 + 1, k)
                pw = dev.pair(dev.bfpair(lo, hi), tag)
                row = 16 * t + k
                own = row % 4
                wd = (
                    Y
                    + rank * layout.RS_STRIDE
                    + ((row // 4) * layout.ROW_PAIRS + col0 // 2 + rp) * 8
                ) // 4
                if (tid < 256) & (row < n_rows):
                    if own == 0:
                        dev.st(pe0, wd, pw)
                    if own == 1:
                        dev.st(pe1, wd, pw)
                    if own == 2:
                        dev.st(pe2, wd, pw)
                    if own == 3:
                        dev.st(pe3, wd, pw)

            for layer in range(layout.LAYERS):
                layer = I32(layer)
                # per-lane values re-derived each layer (not hoisted out of the loop:
                # live across it they spill)
                tid = dev.opaque(tid, layer)
                lane = tid % 64
                wave = tid // 64
                g4 = lane // 16
                l16 = lane % 16
                ii = fx.min(tid, 255)
                rr = ii // 16
                kk = ii % 16
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * layout.W_LAYER_BYTES
                qkv_w = w_qkv + I64(layer) * layout.QKV_W_B
                qkv_s = s_qkv + I64(layer) * layout.QKV_S_B
                o_w = w_o + I64(layer) * layout.O_W_B
                o_s = s_o + I64(layer) * layout.O_S_B
                gu_w = w_gu + I64(layer) * layout.GU_W_B
                gu_s = s_gu + I64(layer) * layout.GU_S_B
                d_w = w_d + I64(layer) * layout.D_W_B
                d_s = s_d + I64(layer) * layout.D_S_B
                # ---- QKV: rows 12 bid ..; token tiles two at a time
                wq = [
                    dev.w_chunk(
                        qkv_w,
                        qkv_s,
                        bid * layout.QKV_R + fx.min(l16, layout.QKV_R - 1),
                        6144,
                        wave,
                        g4,
                        c,
                    )
                    for c in range(6)
                ]
                dev.wait_flags(flag("x8f"), n_rows, tag, tid, dl, health, rank, 1)
                for t in range(0, nt, 2):
                    t = I32(t)
                    xs2 = [
                        [
                            dev.w_chunk(
                                lb + WL["x8"],
                                lb + WL["xs"],
                                tile_row(fx.min(t + h, nt - 1)),
                                6144,
                                wave,
                                g4,
                                c,
                            )
                            for c in range(6)
                        ]
                        for h in range(2)
                    ]
                    for h in range_constexpr(2):
                        if t + h < nt:
                            dev.red_store(
                                red, 0, dev.mma_chain(wq, xs2[h], 6), wave, g4, l16
                            )
                            gpu.barrier()
                            v = dev.red_sum(red, 0, rr, kk)
                            row = 16 * (t + h) + kk
                            if (tid < 256) & (rr < layout.QKV_R) & (row < n_rows):
                                dev.st16(
                                    lb + WL["qkv"],
                                    row * layout.QKV_N + bid * layout.QKV_R + rr,
                                    dev.bf16_bits(v),
                                    CM_DEV,
                                )
                            gpu.barrier()
                dev.publish(flag("qkvf"), bid, tag, tid)
                wo = [
                    [
                        dev.w_chunk(
                            o_w,
                            o_s,
                            bid * layout.O_R + 16 * tt + l16,
                            2048,
                            wave,
                            g4,
                            c,
                        )
                        for c in range(2)
                    ]
                    for tt in range(2)
                ]
                wu = [
                    dev.w_chunk(
                        gu_w,
                        gu_s,
                        layout.IR + bid * layout.GU_R + l16,
                        6144,
                        wave,
                        g4,
                        c,
                    )
                    for c in range(6)
                ]
                # ---- o_proj: rows 32 bid .., partials to the rows' owners
                dev.wait_flags(flag("attf"), n_rows * 4, tag, tid, dl, health, rank, 2)
                for t in range(0, nt, 2):
                    t = I32(t)
                    xs2 = [
                        [
                            dev.w_chunk(
                                lb + WL["a8"],
                                lb + WL["as"],
                                tile_row(fx.min(t + h, nt - 1)),
                                2048,
                                wave,
                                g4,
                                c,
                            )
                            for c in range(2)
                        ]
                        for h in range(2)
                    ]
                    for h in range_constexpr(2):
                        if t + h < nt:
                            dev.red_store(
                                red, 0, dev.mma_chain(wo[0], xs2[h], 2), wave, g4, l16
                            )
                            dev.red_store(
                                red, 1, dev.mma_chain(wo[1], xs2[h], 2), wave, g4, l16
                            )
                            gpu.barrier()
                            rs_push(t + h, bid * layout.O_R, yw + layout.Y_RSO, tag)
                            gpu.barrier()
                wg = [
                    dev.w_chunk(gu_w, gu_s, bid * layout.GU_R + l16, 6144, wave, g4, c)
                    for c in range(6)
                ]
                # ---- gate / up: rows 16 bid .. of each
                dev.wait_flags(flag("h8f"), n_rows, tag, tid, dl, health, rank, 3)
                for t in range(0, nt):
                    t = I32(t)
                    xb = [
                        dev.w_chunk(
                            lb + WL["h8"], lb + WL["hs"], tile_row(t), 6144, wave, g4, c
                        )
                        for c in range(6)
                    ]
                    dev.red_store(red, 0, dev.mma_chain(wg, xb, 6), wave, g4, l16)
                    dev.red_store(red, 1, dev.mma_chain(wu, xb, 6), wave, g4, l16)
                    gpu.barrier()
                    g = dev.bf16r(dev.red_sum(red, 0, rr, kk))
                    u = dev.bf16r(dev.red_sum(red, 1, rr, kk))
                    g = fx.min(g, F32(7.0))
                    u = fx.min(fx.max(u, F32(-7.0)), F32(7.0))
                    a = g * (1.0 / (1.0 + dev.exp(-(g * 1.702)))) * (u + 1.0)
                    if (tid < 256) & (16 * t + kk < n_rows):
                        dev.st16(
                            lb + WL["act"],
                            (16 * t + kk) * layout.IR + bid * layout.GU_R + rr,
                            dev.bf16_bits(a),
                            CM_DEV,
                        )
                    gpu.barrier()
                dev.publish(flag("actf"), bid, tag, tid)
                wd = [
                    [
                        dev.w_chunk(
                            d_w,
                            d_s,
                            bid * layout.D_R + 16 * tt + l16,
                            3072,
                            wave,
                            g4,
                            c,
                        )
                        for c in range(3)
                    ]
                    for tt in range(2)
                ]
                # ---- down: rows 32 bid .., partials to the rows' owners
                dev.wait_flags(flag("actf"), layout.NG, tag, tid, dl, health, rank, 4)
                for t in range(0, nt, 2):
                    t = I32(t)
                    xs2 = [
                        [
                            dev.xq_chunk(
                                lb + WL["act"],
                                tile_row(fx.min(t + h, nt - 1)),
                                3072,
                                wave,
                                g4,
                                l16,
                                c,
                            )
                            for c in range(3)
                        ]
                        for h in range(2)
                    ]
                    for h in range_constexpr(2):
                        if t + h < nt:
                            dev.red_store(
                                red, 0, dev.mma_chain(wd[0], xs2[h], 3), wave, g4, l16
                            )
                            dev.red_store(
                                red, 1, dev.mma_chain(wd[1], xs2[h], 3), wave, g4, l16
                            )
                            gpu.barrier()
                            rs_push(t + h, bid * layout.D_R, yw + layout.Y_RSD, tag)
                            gpu.barrier()
        elif bid < layout.NORM0:
            # ======================================================= L: attention
            ml = fx.recast_iter(fx.Float32, pool) + 5 * att.TILE_WORDS
            n_reqs = n_rows // layout.QPR
            cpr = layout.W_NA // n_reqs
            pidx = bid - layout.ATT0
            for layer in range(layout.LAYERS):
                layer = I32(layer)
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * layout.W_LAYER_BYTES
                cache = (tptr(layer, layout.T_CACHE), I32(tptr(layer, layout.T_NBLK)))
                qn = tptr(layer, layout.T_Q_NORM)
                kn = tptr(layer, layout.T_K_NORM)
                if pidx < cpr * n_reqs:
                    r = pidx // cpr
                    c = pidx % cpr
                    p0, p6, lo, hi, b0w, b1w = att.window(r, positions, seq_lens)
                    nb = b1w - b0w + 1
                    q0 = (c * nb) // cpr
                    cnt = ((c + 1) * nb) // cpr - q0
                    bA = b1w - q0
                    nope = b1w + 1  # past hi: an all-zero tile
                    kw0, vw0 = att.kv_load(
                        cache,
                        block_table,
                        bt_width,
                        r,
                        (cnt >= 1).select(bA, nope),
                        lo,
                        hi,
                        tid,
                    )
                    kw1, vw1 = att.kv_load(
                        cache,
                        block_table,
                        bt_width,
                        r,
                        (cnt >= 2).select(bA - 1, nope),
                        lo,
                        hi,
                        tid,
                    )
                    pre = att.qkv_pre(r, qn, kn, cos_sin, positions, slots, tid)
                    att.kv_store(pool, att.TK0, att.TV0, kw0, vw0, tid)
                    att.kv_store(pool, att.TK1, att.TV1, kw1, vw1, tid)
                    dev.wait_flags(
                        flag("qkvf"), layout.NG, tag, tid, dl, health, rank, 5
                    )
                    att.qkv_rows(
                        r, bA, bA - 1, cnt, pre, lb + WL["qkv"], cache, pool, tid
                    )
                    gpu.barrier()
                    qb = att.q_operand(pool, fx.min(wave, 6), l16, g4)
                    m_run = F32(NEG_INF)
                    l_run = F32(0.0)
                    o_run = [F32(0.0)] * 32
                    # blocks bA, bA - 1, .. in tile pairs K0 / V0 (even), K1 / V1 (odd);
                    # block it + 2's tiles load into registers during block it, then
                    # refill the tile pair block it used
                    for it in range(0, cnt, 2):
                        it = I32(it)
                        for h in range_constexpr(2):
                            if it + h < cnt:
                                kn_, vn_ = att.kv_load(
                                    cache,
                                    block_table,
                                    bt_width,
                                    r,
                                    (it + h + 2 < cnt).select(bA - it - h - 2, nope),
                                    lo,
                                    hi,
                                    tid,
                                )
                                blk = bA - it - h
                                tk = att.TK0 if h == 0 else att.TK1
                                tv = att.TV0 if h == 0 else att.TV1
                                interior = (
                                    (blk * layout.BLOCK > p6 - layout.WINDOW)
                                    & (blk * layout.BLOCK >= lo)
                                    & ((blk + 1) * layout.BLOCK - 1 <= hi)
                                )
                                edge = (
                                    (blk * layout.BLOCK <= p6 - layout.WINDOW)
                                    | (blk * layout.BLOCK < lo)
                                    | ((blk + 1) * layout.BLOCK - 1 > hi)
                                )
                                if (wave < 7) & interior:
                                    m_run, l_run, o_run = att.block_acc(
                                        pool,
                                        qb,
                                        tk,
                                        tv,
                                        blk,
                                        wave,
                                        p0,
                                        lo,
                                        hi,
                                        l16,
                                        g4,
                                        m_run,
                                        l_run,
                                        o_run,
                                        False,
                                    )
                                if (wave < 7) & edge:
                                    m_run, l_run, o_run = att.block_acc(
                                        pool,
                                        qb,
                                        tk,
                                        tv,
                                        blk,
                                        wave,
                                        p0,
                                        lo,
                                        hi,
                                        l16,
                                        g4,
                                        m_run,
                                        l_run,
                                        o_run,
                                        True,
                                    )
                                gpu.barrier()
                                if it + h + 2 < cnt:
                                    att.kv_store(pool, tk, tv, kn_, vn_, tid)
                        gpu.barrier()
                    if wave < 7:
                        ap = lb + I64(WL["ap"]) + I64(pidx) * layout.AP_STRIDE
                        att.store_partial(
                            ap, wave, l16, g4, m_run * att.SM_SCALE, l_run, o_run
                        )
                    dev.publish(flag("apf"), pidx, tag, tid)
                # ---- merges: (row t, heads 4 hq ..), task = 4 t + hq
                for task in range(pidx, n_rows * 4, layout.W_NA):
                    task = I32(task)
                    t = task // 4
                    hq = task % 4
                    r = t // layout.QPR
                    dev.wait_flags(
                        flag("apf") + I64(r * cpr * 8),
                        cpr,
                        tag,
                        tid,
                        dl,
                        health,
                        rank,
                        6,
                    )
                    att.merge_task(
                        t,
                        hq,
                        cpr,
                        lb + I64(WL["ap"]) + I64(r * cpr) * layout.AP_STRIDE,
                        lb + WL["a8"],
                        lb + WL["as"],
                        ml,
                        tid,
                        lane,
                        layout.W_MAXP,
                    )
                    dev.publish(flag("attf"), task, tag, tid)
        else:
            # ======================================================= N: row norms
            j = bid - layout.NORM0
            t_own = 4 * j + rank
            res = pool
            nred = fx.recast_iter(fx.Float32, pool) + 3072

            def gather_x8(Y, tag, lb, x8n, xsn, fname, site):
                for c in range_constexpr(4):
                    t = j + 16 * c
                    if t < n_rows:
                        ag_copy_x8(
                            t,
                            owner(t),
                            yw + Y,
                            tag,
                            lb + WL[x8n],
                            lb + WL[xsn],
                            tid,
                            dl,
                            health,
                            rank,
                            site,
                        )
                        dev.publish(flag(fname), t, tag, tid)

            def post_norm(layer, tag, lb):
                post_w = dev.norm_w(tptr(layer, layout.T_POST_LN), tid)
                if t_own < n_rows:
                    v = dev.row_sum4(
                        sym, yw + layout.Y_RSO, j, tid, tag, dl, health, rank, 8
                    )
                    o = dev.norm_row(v, res, post_w, True, nred, tid, lane, wave)
                    ag_put_x8(o, t_own, yw + layout.Y_AGH, tag, sym, tid)
                gather_x8(layout.Y_AGH, tag, lb, "h8", "hs", "h8f", 11)

            # layer 0: the input rows
            tag = st_ * 8 + 1
            in_w = dev.norm_w(tptr(0, layout.T_IN_LN), tid)
            if t_own < n_rows:
                o = dev.norm_row(
                    dev.row_x(x_in, t_own, tid), res, in_w, False, nred, tid, lane, wave
                )
                ag_put_x8(o, t_own, yw + layout.Y_AGX, tag, sym, tid)
            gather_x8(layout.Y_AGX, tag, scratch, "x8", "xs", "x8f", 10)
            post_norm(I32(0), tag, scratch)
            for layer in range(1, layout.LAYERS):
                layer = I32(layer)
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * layout.W_LAYER_BYTES
                in_w = dev.norm_w(tptr(layer, layout.T_IN_LN), tid)
                if t_own < n_rows:
                    v = dev.row_sum4(
                        sym, yw + layout.Y_RSD, j, tid, tag - 1, dl, health, rank, 7
                    )
                    o = dev.norm_row(v, res, in_w, True, nred, tid, lane, wave)
                    ag_put_x8(o, t_own, yw + layout.Y_AGX, tag, sym, tid)
                gather_x8(layout.Y_AGX, tag, lb, "x8", "xs", "x8f", 10)
                post_norm(layer, tag, lb)
            ftag = st_ * 8 + layout.LAYERS + 1
            fin_w = dev.norm_w(fin_ln, tid)
            if t_own < n_rows:
                v = dev.row_sum4(
                    sym,
                    yw + layout.Y_RSD,
                    j,
                    tid,
                    st_ * 8 + layout.LAYERS,
                    dl,
                    health,
                    rank,
                    9,
                )
                o = dev.norm_row(v, res, fin_w, True, nred, tid, lane, wave)
                ag_put_bf(o, t_own, yw + layout.Y_AGF, ftag, sym, tid)
            for c in range_constexpr(4):
                t = j + 16 * c
                if t < n_rows:
                    ag_copy_bf(
                        t,
                        owner(t),
                        yw + layout.Y_AGF,
                        ftag,
                        h_out,
                        tid,
                        dl,
                        health,
                        rank,
                        12,
                    )

    @flyc.jit
    def launch(
        x_in: Int64,
        h_out: Int64,
        tab: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_gu: Int64,
        s_gu: Int64,
        w_d: Int64,
        s_d: Int64,
        fin_ln: Int64,
        cos_sin: Int64,
        block_table: Int64,
        positions: Int64,
        slots: Int64,
        seq_lens: Int64,
        scratch: Int64,
        sym: Int64,
        pe0: Int64,
        pe1: Int64,
        pe2: Int64,
        pe3: Int64,
        step: Int64,
        health: Int64,
        rank: Int32,
        bt_width: Int32,
        n_rows: Int32,
        stream: fx.Stream = _STREAM,
    ):
        drafter_wide(
            x_in,
            h_out,
            tab,
            w_qkv,
            s_qkv,
            w_o,
            s_o,
            w_gu,
            s_gu,
            w_d,
            s_d,
            fin_ln,
            cos_sin,
            block_table,
            positions,
            slots,
            seq_lens,
            scratch,
            sym,
            pe0,
            pe1,
            pe2,
            pe3,
            step,
            health,
            rank,
            bt_width,
            n_rows,
        ).launch(grid=(layout.BLOCKS,), block=(layout.THREADS,), stream=stream)

    return launch
