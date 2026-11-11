# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's 5 layers and final norm as one persistent launch.

For a draft step of up to 2 requests (14 query rows) on one TP4 rank. A layer
(tag ``step * 8 + layer + 1``), by CTA role (``layout``):

- N: the input row (the embeddings, or the sum of the 4 ranks' down partials) plus
  the residual, input norm -> MXFP8 x8 + flag;
- G: QKV (weights loaded before the wait) -> qkv bf16 + flag;
- L: a partition loads its K / V tiles at the layer's start, then norms and ropes
  q and the new K / V rows, inserts them, and runs FP8 attention over its blocks
  -> partials + flag; the merge tasks (a row, 4 heads) -> MXFP8 attn + flag;
- G: o_proj -> bf16 partials pushed to every rank's symmetric buffer;
- N: the 4 ranks' o_proj partials summed, residual add, post-attention norm ->
  h8 + flag;
- G: gate / up -> SwiGLU-OAI -> act + flag; down (act quantized in registers) ->
  partials pushed to every rank.

After the last layer, N runs the final norm into ``h_out`` (bf16).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.minimax_m3.amd.mono.drafter import attention as att
from vllm.models.minimax_m3.amd.mono.drafter import device as dev
from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.models.minimax_m3.amd.mono.drafter.device import F32, I32, I64
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    kernel_symbol,
    memrealtime,
    uniform,
)

# LDS: L 5 tiles + merge staging; G 4096 partial floats; N the residual + 8 floats
POOL_WORDS = 5 * att.TILE_WORDS + 512
_STREAM = fx.Stream(None)


def build_drafter(wait_ticks=1_000_000_000):
    """The ``@flyc.jit`` launcher of the drafter kernel for one TP4 rank."""

    @fx.struct
    class Smem:
        pool: fx.Array[fx.Int32, POOL_WORDS, 16]

    @flyc.kernel(
        name=kernel_symbol("minimax_m3_dspark_drafter", rows=14),
        known_block_size=[layout.THREADS, 1, 1],
    )
    def drafter(
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
            """Entry i of layer ``layer``'s pointer table row (uniform)."""
            v = dev.ld(tab, (layer * 16 + i) * 2, 2)
            return (I64(uniform(v[1])) << 32) | I64(fx.Uint32(uniform(v[0])))

        def flag(off):
            return scratch + I64(off)

        if bid < layout.NG:
            # ======================================================= G: the GEMVs
            red = fx.recast_iter(fx.Float32, pool)
            ii = fx.min(tid, 255)
            rr = ii // 16
            kk = ii % 16
            xrow = fx.min(l16, n_rows - 1)

            def push_pairs(Y, col0, tag):
                """Push tiles 0 / 1 (rows col0 .. col0 + 31), summed over the waves,
                as bf16 pairs to every rank's slot Y[rank][token][col0 / 2 ..]."""
                k = ii // 16
                rp = ii % 16
                t = rp // 8
                r0 = (2 * rp) % 16
                lo = dev.red_sum(red, t, r0, k)
                hi = dev.red_sum(red, t, r0 + 1, k)
                pw = dev.pair(dev.bfpair(lo, hi), tag)
                if (tid < 256) & (k < n_rows):
                    wd = (
                        Y + ((rank * 16 + k) * layout.ROW_PAIRS + col0 // 2 + rp) * 8
                    ) // 4
                    dev.st(pe0, wd, pw)
                    dev.st(pe1, wd, pw)
                    dev.st(pe2, wd, pw)
                    dev.st(pe3, wd, pw)

            for layer in range(layout.LAYERS):
                layer = I32(layer)
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * layout.LAYER_BYTES
                qkv_w = w_qkv + I64(layer) * layout.QKV_W_B
                qkv_s = s_qkv + I64(layer) * layout.QKV_S_B
                o_w = w_o + I64(layer) * layout.O_W_B
                o_s = s_o + I64(layer) * layout.O_S_B
                gu_w = w_gu + I64(layer) * layout.GU_W_B
                gu_s = s_gu + I64(layer) * layout.GU_S_B
                d_w = w_d + I64(layer) * layout.D_W_B
                d_s = s_d + I64(layer) * layout.D_S_B
                # ---- QKV: rows 12 bid .. (lanes past row 11 repeat it, unused)
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
                dev.wait_flags(flag(layout.F_X8), n_rows, tag, tid, dl, health, rank, 1)
                xb = [
                    dev.w_chunk(
                        lb + layout.R_X8, lb + layout.R_XS, xrow, 6144, wave, g4, c
                    )
                    for c in range(6)
                ]
                dev.red_store(red, 0, dev.mma_chain(wq, xb, 6), wave, g4, l16)
                gpu.barrier()
                v = dev.red_sum(red, 0, rr, kk)
                if (tid < 256) & (rr < layout.QKV_R) & (kk < n_rows):
                    dev.st16(
                        lb + layout.R_QKV,
                        kk * layout.QKV_N + bid * layout.QKV_R + rr,
                        dev.bf16_bits(v),
                        CM_DEV,
                    )
                # publish before the next weights go out: its drain would wait for them
                dev.publish(flag(layout.F_QKV), bid, tag, tid)
                wo = [
                    [
                        dev.w_chunk(
                            o_w, o_s, bid * layout.O_R + 16 * t + l16, 2048, wave, g4, c
                        )
                        for c in range(2)
                    ]
                    for t in range(2)
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
                # ---- o_proj: rows 32 bid ..
                dev.wait_flags(
                    flag(layout.F_ATT), n_rows * 4, tag, tid, dl, health, rank, 2
                )
                xb = [
                    dev.w_chunk(
                        lb + layout.R_A8, lb + layout.R_AS, xrow, 2048, wave, g4, c
                    )
                    for c in range(2)
                ]
                acc0 = dev.mma_chain(wo[0], xb, 2)
                acc1 = dev.mma_chain(wo[1], xb, 2)
                wg = [
                    dev.w_chunk(gu_w, gu_s, bid * layout.GU_R + l16, 6144, wave, g4, c)
                    for c in range(6)
                ]
                dev.red_store(red, 0, acc0, wave, g4, l16)
                dev.red_store(red, 1, acc1, wave, g4, l16)
                gpu.barrier()
                push_pairs(layout.Y_OP, bid * layout.O_R, tag)
                # ---- gate / up: rows 16 bid .. of each
                dev.wait_flags(flag(layout.F_H8), n_rows, tag, tid, dl, health, rank, 3)
                xb = [
                    dev.w_chunk(
                        lb + layout.R_H8, lb + layout.R_HS, xrow, 6144, wave, g4, c
                    )
                    for c in range(6)
                ]
                accg = dev.mma_chain(wg, xb, 6)
                accu = dev.mma_chain(wu, xb, 6)
                dev.red_store(red, 0, accg, wave, g4, l16)
                dev.red_store(red, 1, accu, wave, g4, l16)
                gpu.barrier()
                g = dev.bf16r(dev.red_sum(red, 0, rr, kk))
                u = dev.bf16r(dev.red_sum(red, 1, rr, kk))
                g = fx.min(g, F32(7.0))
                u = fx.min(fx.max(u, F32(-7.0)), F32(7.0))
                a = g * (1.0 / (1.0 + dev.exp(-(g * 1.702)))) * (u + 1.0)
                if (tid < 256) & (kk < n_rows):
                    dev.st16(
                        lb + layout.R_ACT,
                        kk * layout.IR + bid * layout.GU_R + rr,
                        dev.bf16_bits(a),
                        CM_DEV,
                    )
                dev.publish(flag(layout.F_ACT), bid, tag, tid)
                wd = [
                    [
                        dev.w_chunk(
                            d_w, d_s, bid * layout.D_R + 16 * t + l16, 3072, wave, g4, c
                        )
                        for c in range(3)
                    ]
                    for t in range(2)
                ]
                # ---- down: rows 32 bid ..
                dev.wait_flags(
                    flag(layout.F_ACT), layout.NG, tag, tid, dl, health, rank, 4
                )
                xb = [
                    dev.xq_chunk(lb + layout.R_ACT, xrow, 3072, wave, g4, l16, c)
                    for c in range(3)
                ]
                acc0 = dev.mma_chain(wd[0], xb, 3)
                acc1 = dev.mma_chain(wd[1], xb, 3)
                dev.red_store(red, 0, acc0, wave, g4, l16)
                dev.red_store(red, 1, acc1, wave, g4, l16)
                gpu.barrier()
                push_pairs(layout.Y_DP, bid * layout.D_R, tag)
        elif bid < layout.NORM0:
            # ======================================================= L: attention
            ml = fx.recast_iter(fx.Float32, pool) + 5 * att.TILE_WORDS
            n_reqs = n_rows // layout.QPR
            bpp = n_reqs
            npr = layout.NP // n_reqs
            pidx = bid - layout.ATT0
            for layer in range(layout.LAYERS):
                layer = I32(layer)
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * layout.LAYER_BYTES
                cache = (tptr(layer, layout.T_CACHE), I32(tptr(layer, layout.T_NBLK)))
                qn = tptr(layer, layout.T_Q_NORM)
                kn = tptr(layer, layout.T_K_NORM)
                # ---- partition: request r's blocks b0w + c bpp .. (+ bpp - 1)
                if (pidx < layout.NP) & (pidx // npr < n_reqs):
                    r = pidx // npr
                    c = pidx % npr
                    p0, p6, lo, hi, b0w, b1w = att.window(r, positions, seq_lens)
                    blk0 = b0w + c * bpp
                    if blk0 <= b1w:
                        use2 = (bpp == 2) & (blk0 + 1 <= b1w)
                        blk1 = use2.select(blk0 + 1, b1w + 1)
                        kw0, vw0 = att.kv_load(
                            cache, block_table, bt_width, r, blk0, lo, hi, tid
                        )
                        kw1, vw1 = att.kv_load(
                            cache, block_table, bt_width, r, blk1, lo, hi, tid
                        )
                        pre = att.qkv_pre(r, qn, kn, cos_sin, positions, slots, tid)
                        att.kv_store(pool, att.TK0, att.TV0, kw0, vw0, tid)
                        att.kv_store(pool, att.TK1, att.TV1, kw1, vw1, tid)
                        dev.wait_flags(
                            flag(layout.F_QKV), layout.NG, tag, tid, dl, health, rank, 5
                        )
                        att.qkv_rows(
                            r,
                            blk0,
                            blk0 + 1,
                            bpp,
                            pre,
                            lb + layout.R_QKV,
                            cache,
                            pool,
                            tid,
                        )
                        gpu.barrier()
                        # a partial a block (request r's block b at slot r 34 + b)
                        ap = (
                            lb
                            + I64(layout.R_AP)
                            + I64(r * layout.MAXB + blk0 - b0w) * layout.AP_STRIDE
                        )
                        if wave < 7:
                            qb = att.q_operand(pool, wave, l16, g4)
                            for h in range_constexpr(2):
                                blk = blk0 + h
                                go = (h == 0) | use2
                                # interior: every key of the block is in every row's
                                # window and below seq_len
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
                                tk = att.TK0 if h == 0 else att.TK1
                                tv = att.TV0 if h == 0 else att.TV1
                                aph = ap + I64(h * layout.AP_STRIDE)
                                if go & interior:
                                    m, ls, o = att.block_math(
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
                                        False,
                                    )
                                    att.store_partial(aph, wave, l16, g4, m, ls, o)
                                if go & edge:
                                    m, ls, o = att.block_math(
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
                                        True,
                                    )
                                    att.store_partial(aph, wave, l16, g4, m, ls, o)
                        dev.publish(flag(layout.F_AP), pidx, tag, tid)
                # ---- merges: (row t, heads 4 hq ..), task = 4 t + hq
                for task in range(pidx, n_rows * 4, layout.NATT):
                    task = I32(task)
                    t = task // 4
                    hq = task % 4
                    r = t // layout.QPR
                    p0, p6, lo, hi, b0w, b1w = att.window(r, positions, seq_lens)
                    # a padded request of a FULL graph replay has an empty window
                    # (seq_len 0): no partitions. Its merges wait for this layer's
                    # QKV as partitions do; unpaced, they would retag the attention
                    # flags (shared by the layers) before the GEMVs read them.
                    na = fx.max(b1w - b0w + 1, I32(0))
                    nparts = (na + bpp - 1) // bpp  # this request's partition CTAs
                    if nparts > 0:
                        dev.wait_flags(
                            flag(layout.F_AP) + I64(r * npr * 8),
                            nparts,
                            tag,
                            tid,
                            dl,
                            health,
                            rank,
                            6,
                        )
                    else:
                        dev.wait_flags(
                            flag(layout.F_QKV), layout.NG, tag, tid, dl, health, rank, 6
                        )
                    att.merge_task(
                        t,
                        hq,
                        na,
                        lb + I64(layout.R_AP) + I64(r * layout.MAXB) * layout.AP_STRIDE,
                        lb + layout.R_A8,
                        lb + layout.R_AS,
                        ml,
                        tid,
                        lane,
                        layout.MAXB,
                    )
                    dev.publish(flag(layout.F_ATT), task, tag, tid)
        elif bid - layout.NORM0 < n_rows:
            # ======================================================= N: row norms
            t = bid - layout.NORM0
            res = pool
            nred = fx.recast_iter(fx.Float32, pool) + 3072

            def post_norm(layer, tag, lb):
                post_w = dev.norm_w(tptr(layer, layout.T_POST_LN), tid)
                v = dev.row_sum4(sym, layout.Y_OP, t, tid, tag, dl, health, rank, 8)
                o = dev.norm_row(v, res, post_w, True, nred, tid, lane, wave)
                dev.quant_store(o, t, lb + layout.R_H8, lb + layout.R_HS, tid)
                dev.publish(flag(layout.F_H8), t, tag, tid)

            # layer 0: the input rows
            tag = st_ * 8 + 1
            in_w = dev.norm_w(tptr(0, layout.T_IN_LN), tid)
            o = dev.norm_row(
                dev.row_x(x_in, t, tid), res, in_w, False, nred, tid, lane, wave
            )
            dev.quant_store(o, t, scratch + layout.R_X8, scratch + layout.R_XS, tid)
            dev.publish(flag(layout.F_X8), t, tag, tid)
            post_norm(I32(0), tag, scratch)
            for layer in range(1, layout.LAYERS):
                layer = I32(layer)
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * layout.LAYER_BYTES
                in_w = dev.norm_w(tptr(layer, layout.T_IN_LN), tid)
                v = dev.row_sum4(sym, layout.Y_DP, t, tid, tag - 1, dl, health, rank, 7)
                o = dev.norm_row(v, res, in_w, True, nred, tid, lane, wave)
                dev.quant_store(o, t, lb + layout.R_X8, lb + layout.R_XS, tid)
                dev.publish(flag(layout.F_X8), t, tag, tid)
                post_norm(layer, tag, lb)
            fin_w = dev.norm_w(fin_ln, tid)
            v = dev.row_sum4(
                sym, layout.Y_DP, t, tid, st_ * 8 + layout.LAYERS, dl, health, rank, 9
            )
            o = dev.norm_row(v, res, fin_w, True, nred, tid, lane, wave)
            dev.store_bf(o, t, h_out, tid)

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
        drafter(
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
