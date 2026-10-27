# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501, SIM102

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/k5m.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""HM: this rank's routed experts for S = 32 / 64 / 96 / 128 rows as one persistent launch
between HA and HB (k5.py); one rank of a TP = EP = 8 group.

Inputs from HA: h [S, 4096] bf16 (pre_expert_norm output), top-8 ids [S, 8] and weights [S, 8].
Every CTA builds the same tables: the picks of each local expert in token order, split into
virtual experts of at most 16 picks (one MFMA N tile). Tasks: (virtual expert v, row tile j),
256 a virtual expert (gate_up: 16 gate + 16 up rows; down: 16 output rows); CTA b owns tasks
[b V, (b+1) V) (V virtual experts), so it spans at most two virtual experts and keeps their
picks' h (gate_up) / a (down) rows in LDS. Weights MXFP4 (w13 / w2 + E8M0 scales) in vLLM's
shuffled layouts, streamed with a prefetch queue, decoded in registers
(v_cvt_scalef32_pk_bf16_fp4), bf16 MFMA.

  gate_up: a = bf16(silu(bf16 g) bf16 u) -> scratch A (tagged pairs, pick rows)
  down:    z = bf16(w_post bf16(y)) -> scratch Z (tagged), per-pick sums of squares a CTA
  rstd:    CTA c: the post-expert-norm rstd of picks c, c + 256, .. (tagged)
  final:   CTA b, columns 16b ..: part[t] = sum over t's local picks (k order) of
           bf16(z rstd) w -> pushed to the owner rank of the column block (b / 32) = HB's
           reduce-scatter input (+ flags), and part_out (a local copy).
Same tag as HA / HB of the layer (epoch * 64 + layer + 1).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from . import k5
from .dev import (
    BLOCKS,
    CM_DEV,
    CM_NT,
    CM_SYS,
    THREADS,
    WAVES,
    ballot,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    e8m0_f32,
    fp4x8_bf16,
    fresh,
    hw_exp2,
    hw_rsq,
    lanes_below,
    ld_f32,
    ld_i32,
    ld_i32_if,
    ld_vec_if,
    mb_poll,
    mb_put,
    mb_put_sys,
    mfma_bf16,
    permlane_swap,
    popcount,
    readlane,
    st_if,
    traced,
    uniform,
    wave_incl_scan,
)
from .k3 import _lay, sel_peer

DM = 4096
INTER = 4096
E = 256
W = 8  # ranks in the TP = EP group
EL = E // W  # local experts
TOPK = 8
FCH = 256
POST_EPS = 1e-8
HROW = DM + 8
AROW = INTER + 8
SSN = 257  # ss partial slots a pick (CTAs covering one virtual expert, at most 256 / V + 1)
REGION = {"a": 31, "z": 32, "ss": 33, "rs": 34}


@dataclass(frozen=True)
class BuildM:
    S: int = 32  # rows (32 / 64 / 96 / 128)


def build_vllm(S, W=8):
    """The HM build vLLM launches for S rows of a TP = EP = 8 group."""
    assert S in (32, 64, 96, 128) and W == 8, (S, W)
    return BuildM(S=S)


def scratch_layout(S):
    P = TOPK * S
    return _lay(
        [
            ("a", P * (INTER // 2) * 8),
            ("z", P * (DM // 2) * 8),
            ("ss", P * SSN * 8),
            ("rs", P * 8),
        ]
    )


_BUILDS = {}


def build(key: BuildM):
    if key in _BUILDS:
        return _BUILDS[key]
    _KEYSTR = repr(key)
    S = key.S
    assert S in (32, 64, 96, 128), S
    CPR = BLOCKS // W  # 16-column blocks a rank owns in the EP reduce-scatter
    P = TOPK * S
    assert EL % WAVES == 0
    VMAX = EL + P // 16
    lay, _ = scratch_layout(S)
    play, _ = k5.peer_layout(S, W)  # HM addresses moep / mf of the token-owner layout
    PF_GU, PF_DN = 1, 2  # gate_up / down tasks in flight
    UNR = S != 96  # down task loop as a ring (unroll 3, fixed slots)
    W13_BYTES = EL * 2 * INTER * (DM // 2)
    W13S_BYTES = EL * 2 * INTER * (DM // 32)
    W2_BYTES = EL * DM * (INTER // 2)
    W2S_BYTES = EL * DM * (INTER // 32)
    tag_par = lambda tag: tag & 1

    @fx.struct
    class RouteM:
        nu: fx.Array[fx.Int32, EL, 16]
        vstart: fx.Array[fx.Int32, EL, 16]
        unused: fx.Array[fx.Int32, 3 * 256, 16]  # keeps the tuned LDS offsets
        ve_u: fx.Array[fx.Int32, VMAX, 16]
        ve_n: fx.Array[fx.Int32, VMAX, 16]
        ve_off: fx.Array[fx.Int32, VMAX, 16]
        nve: fx.Array[fx.Int32, 4, 16]
        pinfo: fx.Array[fx.Int32, P, 16]
        pk: fx.Array[fx.Int32, 2 * 16, 16]
        ssacc: fx.Array[fx.Float32, 2 * 16, 16]
        rsl: fx.Array[fx.Float32, P, 16]
        pad: fx.Array[fx.Int64, 8, 16]  # unused: keeps the tuned LDS offsets

    @fx.struct
    class MSetup:
        lists: fx.Array[fx.Int32, EL * S, 16]

    @fx.struct
    class MGu:
        h: fx.Array[fx.Int32, 16 * HROW // 2, 16]
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]

    @fx.struct
    class MDn:
        a: fx.Array[fx.Int32, 16 * AROW // 2, 16]
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]

    @fx.union
    class MLds:
        s_: MSetup
        g_: MGu
        d_: MDn

    name = f"muse_k5m_s{S}"

    # ------------------------------------------------------------ tables
    @traced
    def setup(ids, rank, tid, lane, wave, rt, lists, nrows):
        tid = fresh(tid)
        NC = P // 64
        for q in range_constexpr((P + THREADS - 1) // THREADS):
            p = tid + THREADS * q
            if p < P:
                fx.ptr_store(fx.Int32(-1), rt["pinfo"] + p)
        # every wave: the weight slot of every pick, 64 picks a register
        ev = []
        for c in range_constexpr(NC):
            real = ((c * 64 + lane) // TOPK) < nrows  # padded rows route nowhere
            e = ld_i32(ids, c * 64 + lane) - rank * EL
            ev.append(((e >= 0) & (e < EL) & real).select(e, fx.Int32(-1)))
        # wave w: slots w, w + 8, ..: pick lists in pick (= token) order
        for uu in range_constexpr(EL // WAVES):
            u = wave + WAVES * uu
            cnt = fx.Int32(0)
            for c in range_constexpr(NC):
                hit = ev[c] == u
                b = ballot(hit)
                pos = cnt + lanes_below(b, lane)
                if hit:
                    fx.ptr_store(fx.Int32(c * 64) + lane, lists + (u * S + pos))
                cnt = cnt + popcount(b)
            if lane == 0:
                fx.ptr_store(cnt, rt["nu"] + u)
        gpu.barrier()
        # wave 0, lane u (+ 64 hh): groups of 16 -> virtual experts (slot order)
        if wave == 0:
            vcarry = fx.Int32(0)
            for hh in range_constexpr((EL + 63) // 64):
                u = lane + 64 * hh
                n = (u < EL).select(
                    fx.ptr_load(rt["nu"] + fx.min(u, EL - 1)), fx.Int32(0)
                )
                ng = (n + 15) // 16
                incl = wave_incl_scan(ng, lane) + vcarry
                st = incl - ng
                if u < EL:
                    fx.ptr_store(st, rt["vstart"] + u)
                for gg in range_constexpr(S // 16):
                    if gg < ng:
                        v = st + gg
                        fx.ptr_store(u, rt["ve_u"] + v)
                        fx.ptr_store(fx.Int32(16 * gg), rt["ve_off"] + v)
                        fx.ptr_store(fx.min(n - 16 * gg, 16), rt["ve_n"] + v)
                vcarry = readlane(incl, 63)
            if lane == 0:
                fx.ptr_store(vcarry, rt["nve"] + 0)
        gpu.barrier()
        # pinfo[pick] = virtual expert * 32 + slot
        for q in range_constexpr((EL * S + THREADS - 1) // THREADS):
            x = tid + THREADS * q
            u = fx.min(x // S, EL - 1)
            i = x % S
            if (x < EL * S) & (i < fx.ptr_load(rt["nu"] + u)):
                p = fx.ptr_load(lists + (u * S + i))
                v = fx.ptr_load(rt["vstart"] + u) + i // 16
                fx.ptr_store(v * 32 + i % 16, rt["pinfo"] + p)
        gpu.barrier()

    @traced
    def own_picks(rt, lists, v0, v1, tid):
        """pk[h][n]: the picks of the CTA's virtual experts v0 (h = 0), v1 (h = 1)."""
        if tid < 32:
            h = tid // 16
            n = tid % 16
            v = (h == 0).select(v0, v1)
            ok = (v < uniform(fx.ptr_load(rt["nve"] + 0))) & (
                n < fx.ptr_load(rt["ve_n"] + fx.min(v, VMAX - 1))
            )
            pk = fx.Int32(0)
            if ok:
                vv = fx.min(v, VMAX - 1)
                pk = fx.ptr_load(
                    lists
                    + (
                        fx.ptr_load(rt["ve_u"] + vv) * S
                        + fx.ptr_load(rt["ve_off"] + vv)
                        + n
                    )
                )
            fx.ptr_store(ok.select(pk, fx.Int32(-1)), rt["pk"] + tid)
            fx.ptr_store(fx.Float32(0.0), rt["ssacc"] + tid)
        gpu.barrier()

    # ------------------------------------------------------------ gate_up
    def gu_loads(w13, w13s, rt, task, lane, wave, ok):
        """Task (v, j): waves 0-3 the 16 gate rows 16j .., waves 4-7 the up rows; wave w the
        K chunks 4 (w % 4) + i (256 K each): 2 x dwordx4 + 1 scale dword a chunk."""
        r = lane % 16
        g = lane // 16
        task = ok.select(task, fx.Int32(0))
        v = task >> 8
        j = task & 255
        e = uniform(fx.ptr_load(rt["ve_u"] + fx.min(v, VMAX - 1)))
        rg = wave // 4
        ops = []
        for i in range_constexpr(4):
            c = 4 * (wave % 4) + i
            rbase = e * (2 * INTER) + rg * INTER + j * 16
            wb = (rbase * (DM // 2) + (2 * c) * 1024 + lane * 16) // 4
            ops.append(ld_vec_if(ok, w13, wb, 4, W13_BYTES, CM_NT))
            ops.append(ld_vec_if(ok, w13, wb + 256, 4, W13_BYTES, CM_NT))
            sdw = (
                (rbase >> 5) * (8 * (DM // 32))
                + 64 * c
                + 16 * (2 * (g % 2) + g // 2)
                + r
            )
            ops.append(ld_i32_if(ok, w13s, sdw, W13S_BYTES))
        return ops

    def fp4_mma(ops, base, bl, brow, lane, acc, c, jpar=None):
        """Chunk c (256 K) of a 16-row group times LDS B rows lane % 16: 8 MFMA steps.
        ``jpar``: the task's 16-row group parity (j & 1), for the shuffled scales."""
        g = lane // 16
        t = lane % 16
        sw = ops[base + 2]
        sa_, sb_ = permlane_swap(32, sw, sw)
        shs = (g // 2) * 16 + jpar * 8
        s0 = e8m0_f32((sa_ >> shs) & 0xFF)
        s1 = e8m0_f32((sb_ >> shs) & 0xFF)
        # shuffled, contiguous loads: permlane32_swap then permlane16_swap give the MFMA operand (as k3 fp4_mma)
        x0, x1 = [], []
        for jj in range_constexpr(4):
            a_, b_ = permlane_swap(32, ops[base][jj], ops[base + 1][jj])
            p_, q_ = permlane_swap(16, a_, b_)
            x0.append(p_)
            x1.append(q_)
        xs = [x0, x1]
        for d in range_constexpr(8):
            a = fp4x8_bf16(xs[d // 4][d % 4], s0 if d < 4 else s1)
            kk = FCH * c + 64 * g + 8 * d
            b = fx.Vector(
                fx.ptr_load(
                    bl + (t * brow + kk) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            ).bitcast(fx.BFloat16)
            acc = mfma_bf16(a, b, acc)
        return acc

    @traced
    def load_rows_h(h, rt, hsel, hl, tid):
        """LDS row n := h row of the token of pick pk[hsel][n] (zeros for an empty slot)."""
        tid = fresh(tid)
        NQ = 16 * (DM // 8) // THREADS
        for half in range_constexpr(2):  # 8 x 16 B in flight a thread (VGPR budget)
            vs = []
            for q in range_constexpr(NQ // 2):
                x = tid + THREADS * (half * (NQ // 2) + q)
                n = x // (DM // 8)
                kk = (x % (DM // 8)) * 8
                pk = fx.ptr_load(rt["pk"] + (hsel * 16 + n))
                vs.append(
                    ld_vec_if(pk >= 0, h, ((pk >> 3) * DM + kk) // 2, 4, S * DM * 2)
                )
            for q in range_constexpr(NQ // 2):
                x = tid + THREADS * (half * (NQ // 2) + q)
                n = x // (DM // 8)
                kk = (x % (DM // 8)) * 8
                fx.ptr_store(vs[q], hl + (n * HROW + kk) // 2)
        gpu.barrier()

    @traced
    def gu_compute(ops, task, it, hsel, scratch, tag, rt, tid, lane, wave, hl, red):
        tid = fresh(tid)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(4):
            acc = fp4_mma(
                ops,
                3 * i,
                hl,
                HROW,
                lane,
                acc,
                4 * (wave % 4) + i,
                jpar=(task & 1),
            )
        par = (it & 1) * (WAVES * 64 * 4)
        fx.ptr_store(acc, red + (par + (wave * 64 + lane) * 4))
        gpu.barrier()
        if tid < 128:
            rp = tid % 8
            n = tid // 8
            j = task & 255
            pk = fx.ptr_load(rt["pk"] + (hsel * 16 + n))
            av = []
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                gs = fx.Float32(0.0)
                us = fx.Float32(0.0)
                for w in range_constexpr(4):
                    off = par + (w * 64 + 16 * (rr // 4) + n) * 4 + rr % 4
                    gs = gs + fx.ptr_load(red + off)
                    us = us + fx.ptr_load(red + (off + 4 * 64 * 4))
                gb = bf16_round(gs)
                ub = bf16_round(us)
                sg = bf16_round(
                    gb / (fx.Float32(1.0) + hw_exp2(gb * fx.Float32(-LOG2E)))
                )
                av.append(sg * ub)
            if pk >= 0:
                mb_put(
                    tag,
                    scratch + fx.Int64(lay["a"][0]),
                    pk * (INTER // 2) + j * 8 + rp,
                    bf16_pair(av[0], av[1]),
                )

    # ------------------------------------------------------------ down
    def dn_loads(w2, w2s, w_post, rt, task, lane, wave, ok):
        r = lane % 16
        g = lane // 16
        task = ok.select(task, fx.Int32(0))
        v = task >> 8
        j = task & 255
        e = uniform(fx.ptr_load(rt["ve_u"] + fx.min(v, VMAX - 1)))
        rbase = e * DM + j * 16
        ops = []
        for i in range_constexpr(2):
            c = 2 * wave + i
            wb = (rbase * (INTER // 2) + (2 * c) * 1024 + lane * 16) // 4
            ops.append(ld_vec_if(ok, w2, wb, 4, W2_BYTES, CM_NT))
            ops.append(ld_vec_if(ok, w2, wb + 256, 4, W2_BYTES, CM_NT))
            sdw = (
                (rbase >> 5) * (8 * (INTER // 32))
                + 64 * c
                + 16 * (2 * (g % 2) + g // 2)
                + r
            )
            ops.append(ld_i32_if(ok, w2s, sdw, W2S_BYTES))
        ops.append(ld_i32_if(ok, w_post, 8 * j + lane % 8, DM * 2))
        return ops

    @traced
    def load_rows_a(scratch, tag, health, rt, hsel, al, tid):
        """LDS row n := A row of pick pk[hsel][n] (polled; zeros for an empty slot)."""
        tid = fresh(tid)
        NQ = 16 * (INTER // 4) // THREADS
        for qb in range_constexpr(NQ // 8):  # 8 specs (16 pairs) a round: VGPR budget
            specs, oks, dsts = [], [], []
            for q in range_constexpr(8):
                x = tid + THREADS * (qb * 8 + q)
                n = x // (INTER // 4)
                sidx = x % (INTER // 4)
                pk = fx.ptr_load(rt["pk"] + (hsel * 16 + n))
                specs.append(
                    (
                        scratch + fx.Int64(lay["a"][0]),
                        fx.max(pk, 0) * (INTER // 2) + 2 * sidx,
                        2,
                    )
                )
                oks.append(pk >= 0)
                dsts.append(al + (n * AROW + 4 * sidx) // 2)
            got = mb_poll(tag, health, specs, REGION["a"], batch=8, oks=oks)
            for q in range_constexpr(8):
                fx.ptr_store(
                    fx.Vector.from_elements([got[q][0], got[q][1]], fx.Int32), dsts[q]
                )
        gpu.barrier()

    @traced
    def dn_compute(ops, task, it, hsel, scratch, tag, rt, tid, lane, wave, al, red):
        tid = fresh(tid)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(2):
            acc = fp4_mma(
                ops,
                3 * i,
                al,
                AROW,
                lane,
                acc,
                2 * wave + i,
                jpar=(task & 1),
            )
        par = (it & 1) * (WAVES * 64 * 4)
        fx.ptr_store(acc, red + (par + (wave * 64 + lane) * 4))
        gpu.barrier()
        if tid < 128:
            rp = tid % 8
            n = tid // 8
            j = task & 255
            pk = fx.ptr_load(rt["pk"] + (hsel * 16 + n))
            wpw = ops[6]
            zv = []
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                ys = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    ys = ys + fx.ptr_load(
                        red + (par + (w * 64 + 16 * (rr // 4) + n) * 4 + rr % 4)
                    )
                wp = bf_lo(wpw) if h2 == 0 else bf_hi(wpw)
                zv.append(bf16_round(wp * bf16_round(ys)))
            ss = butterfly(zv[0] * zv[0] + zv[1] * zv[1], (1, 2, 4))
            if pk >= 0:
                mb_put(
                    tag,
                    scratch + fx.Int64(lay["z"][0]),
                    pk * (DM // 2) + j * 8 + rp,
                    bf16_pair(zv[0], zv[1]),
                )
                if rp == 0:
                    sa = rt["ssacc"] + (hsel * 16 + n)
                    fx.ptr_store(fx.ptr_load(sa) + ss, sa)

    # ------------------------------------------------------------ rstd + final
    @traced
    def stage_rstd(scratch, tag, health, rt, tid, bid):
        """CTA c: picks c, c + 256, ..: rstd = rsqrt(sum of the per-CTA partials / 4096 + eps),
        8 threads a pick (slots part + 8 q)."""
        tid = fresh(tid)
        V = uniform(fx.ptr_load(rt["nve"] + 0))
        part = tid % 8
        for hh in range_constexpr((P // BLOCKS) * 8 // THREADS + 1):
            pi = bid + BLOCKS * (tid // 8 + (THREADS // 8) * hh)
            if pi < P:
                pinf = fx.ptr_load(rt["pinfo"] + pi)
                if pinf >= 0:
                    v = pinf >> 5
                    cf = (v * 256) // V
                    nc = (v * 256 + 255) // V - cf + 1
                    specs = [
                        (
                            scratch + fx.Int64(lay["ss"][0]),
                            pi * SSN + fx.min(part + 8 * q, SSN - 1),
                            1,
                        )
                        for q in range(SSN // 8 + 1)
                    ]
                    oks = [(part + 8 * q) < nc for q in range(SSN // 8 + 1)]
                    got = mb_poll(tag, health, specs, REGION["ss"], batch=17, oks=oks)
                    s_ = fx.Float32(0.0)
                    for q in range_constexpr(SSN // 8 + 1):
                        s_ = s_ + got[q][0].bitcast(fx.Float32)
                    s_ = butterfly(s_, (1, 2, 4))
                    if part == 0:
                        mb_put(
                            tag,
                            scratch + fx.Int64(lay["rs"][0]),
                            pi,
                            hw_rsq(s_ * fx.Float32(1.0 / DM) + fx.Float32(POST_EPS)),
                        )

    @traced
    def stage_final(scratch, sc, part_out, pl, tag, health, rt, tid, bid, rank):
        tid = fresh(tid)
        par = tag_par(tag)
        # rstd of every local pick (polled), then CTA b's 16 columns of every token
        for q in range_constexpr((P + THREADS - 1) // THREADS):
            p = tid + THREADS * q
            if p < P:
                if fx.ptr_load(rt["pinfo"] + p) >= 0:
                    got = mb_poll(
                        tag,
                        health,
                        [(scratch + fx.Int64(lay["rs"][0]), p, 1)],
                        REGION["rs"],
                    )
                    fx.ptr_store(got[0][0].bitcast(fx.Float32), rt["rsl"] + p)
        gpu.barrier()
        pd = sel_peer(pl, uniform(bid // CPR))
        for it_ in range_constexpr((8 * S + THREADS - 1) // THREADS):
            e0 = tid + THREADS * it_
            ok = e0 < 8 * S
            e = fx.min(e0, 8 * S - 1)
            t = e // 8
            rp = e % 8
            specs, oks, wts, rss = [], [], [], []
            for k in range_constexpr(TOPK):
                p = t * TOPK + k
                loc = fx.ptr_load(rt["pinfo"] + p) >= 0
                specs.append(
                    (scratch + fx.Int64(lay["z"][0]), p * (DM // 2) + bid * 8 + rp, 1)
                )
                oks.append(loc & ok)
                wts.append(loc.select(ld_f32(sc, p), fx.Float32(0.0)))
                rss.append(fx.ptr_load(rt["rsl"] + p))
            got = mb_poll(tag, health, specs, REGION["z"], oks=oks)
            a0 = fx.Float32(0.0)
            a1 = fx.Float32(0.0)
            for k in range_constexpr(TOPK):
                zw = got[k][0]
                a0 = a0 + oks[k].select(
                    bf16_round(bf_lo(zw) * rss[k]) * wts[k], fx.Float32(0.0)
                )
                a1 = a1 + oks[k].select(
                    bf16_round(bf_hi(zw) * rss[k]) * wts[k], fx.Float32(0.0)
                )
            w2 = bf16_pair(a0, a1)
            st_if(ok, w2, part_out, (t * DM + bid * 16 + 2 * rp) // 2)
            st_if(
                ok,
                w2,
                pd + fx.Int64(play["moep"][0]),
                (((par * W + rank) * BLOCKS + bid) * S + t) * 8 + rp,
                CM_SYS,
            )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put_sys(
                tag,
                pd + fx.Int64(play["mf"][0]),
                (par * W + rank) * BLOCKS + bid,
                fx.Int32(1),
            )

    @traced
    def hm_body(
        h,
        ids,
        sc,
        w13,
        w13s,
        w2,
        w2s,
        w_post,
        part_out,
        scratch,
        epoch,
        health,
        tl,
        peers,
        layer,
        rank,
        nrows,
        w13r,
        w13sr,
        w2r,
        w2sr,
        eptab,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        alloc = fx.SharedAllocator()
        rl = alloc.allocate(RouteM).peek()
        ml = alloc.allocate(MLds)
        ms, mg, md = ml.s_.peek(), ml.g_.peek(), ml.d_.peek()
        rt = {
            "nu": rl.nu.ptr,
            "vstart": rl.vstart.ptr,
            "ve_u": rl.ve_u.ptr,
            "ve_n": rl.ve_n.ptr,
            "ve_off": rl.ve_off.ptr,
            "nve": rl.nve.ptr,
            "pinfo": rl.pinfo.ptr,
            "pk": rl.pk.ptr,
            "ssacc": rl.ssacc.ptr,
            "rsl": rl.rsl.ptr,
        }
        pl = list(peers)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        setup(ids, rank, tid, lane, wave, rt, ms.lists.ptr, nrows)
        V = uniform(fx.ptr_load(rt["nve"] + 0))
        t0 = bid * V
        t1 = t0 + V
        v0 = t0 >> 8
        v1 = (fx.max(t1, 1) - 1) >> 8
        q = []  # (a function-level name: the scf.while rewriter carries only names bound at this level)
        q = [
            gu_loads(w13, w13s, rt, t0 + d, lane, wave, (t0 + d) < t1)
            for d in range(PF_GU)
        ]
        own_picks(rt, ms.lists.ptr, v0, v1, tid)
        # gate_up over virtual expert v0, then (if the range crosses) v1
        load_rows_h(h, rt, 0, mg.h.ptr, tid)
        task = t0
        it = fx.Int32(0)
        tsplit = fx.min(t1, (v0 + 1) << 8)
        while task < tsplit:
            nxt = gu_loads(w13, w13s, rt, task + PF_GU, lane, wave, (task + PF_GU) < t1)
            gu_compute(
                q[0],
                task,
                it,
                0,
                scratch,
                tag,
                rt,
                tid,
                lane,
                wave,
                mg.h.ptr,
                mg.red.ptr,
            )
            q = q[1:] + [nxt]
            task = task + 1
            it = it + 1
        if task < t1:
            gpu.barrier()
            load_rows_h(h, rt, 1, mg.h.ptr, tid)
        while task < t1:
            nxt = gu_loads(w13, w13s, rt, task + PF_GU, lane, wave, (task + PF_GU) < t1)
            gu_compute(
                q[0],
                task,
                it,
                1,
                scratch,
                tag,
                rt,
                tid,
                lane,
                wave,
                mg.h.ptr,
                mg.red.ptr,
            )
            q = q[1:] + [nxt]
            task = task + 1
            it = it + 1
        # down: same tasks
        dq = []
        dq = [
            dn_loads(w2, w2s, w_post, rt, t0 + d, lane, wave, (t0 + d) < t1)
            for d in range(PF_DN)
        ]
        gpu.barrier()
        load_rows_a(scratch, tag, health, rt, 0, md.a.ptr, tid)
        task = t0
        it = fx.Int32(0)
        if const_expr(UNR):
            # ring: three tasks an iteration in fixed slots (dq[0], dq[1], dc): two tasks' loads stay in
            # flight during each compute, no rotation copy waits for its data (the tail: the loop below)
            while task + 2 < tsplit:
                dc = dn_loads(
                    w2, w2s, w_post, rt, task + 2, lane, wave, (task + 2) < t1
                )
                dn_compute(
                    dq[0],
                    task,
                    it,
                    0,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    md.a.ptr,
                    md.red.ptr,
                )
                da = dn_loads(
                    w2, w2s, w_post, rt, task + 3, lane, wave, (task + 3) < t1
                )
                dn_compute(
                    dq[1],
                    task + 1,
                    it + 1,
                    0,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    md.a.ptr,
                    md.red.ptr,
                )
                db = dn_loads(
                    w2, w2s, w_post, rt, task + 4, lane, wave, (task + 4) < t1
                )
                dn_compute(
                    dc,
                    task + 2,
                    it + 2,
                    0,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    md.a.ptr,
                    md.red.ptr,
                )
                dq = [da, db]
                task = task + 3
                it = it + 3
        while task < tsplit:
            nxt = dn_loads(
                w2, w2s, w_post, rt, task + PF_DN, lane, wave, (task + PF_DN) < t1
            )
            dn_compute(
                dq[0],
                task,
                it,
                0,
                scratch,
                tag,
                rt,
                tid,
                lane,
                wave,
                md.a.ptr,
                md.red.ptr,
            )
            dq = dq[1:] + [nxt]
            task = task + 1
            it = it + 1
        if task < t1:
            gpu.barrier()
            load_rows_a(scratch, tag, health, rt, 1, md.a.ptr, tid)
        if const_expr(UNR):
            # ring: three tasks an iteration in fixed slots (dq[0], dq[1], dc): two tasks' loads stay in
            # flight during each compute, no rotation copy waits for its data (the tail: the loop below)
            while task + 2 < t1:
                dc = dn_loads(
                    w2, w2s, w_post, rt, task + 2, lane, wave, (task + 2) < t1
                )
                dn_compute(
                    dq[0],
                    task,
                    it,
                    1,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    md.a.ptr,
                    md.red.ptr,
                )
                da = dn_loads(
                    w2, w2s, w_post, rt, task + 3, lane, wave, (task + 3) < t1
                )
                dn_compute(
                    dq[1],
                    task + 1,
                    it + 1,
                    1,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    md.a.ptr,
                    md.red.ptr,
                )
                db = dn_loads(
                    w2, w2s, w_post, rt, task + 4, lane, wave, (task + 4) < t1
                )
                dn_compute(
                    dc,
                    task + 2,
                    it + 2,
                    1,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    md.a.ptr,
                    md.red.ptr,
                )
                dq = [da, db]
                task = task + 3
                it = it + 3
        while task < t1:
            nxt = dn_loads(
                w2, w2s, w_post, rt, task + PF_DN, lane, wave, (task + PF_DN) < t1
            )
            dn_compute(
                dq[0],
                task,
                it,
                1,
                scratch,
                tag,
                rt,
                tid,
                lane,
                wave,
                md.a.ptr,
                md.red.ptr,
            )
            dq = dq[1:] + [nxt]
            task = task + 1
            it = it + 1
        gpu.barrier()
        # per-CTA sums of squares of the picks of v0 / v1 -> slot (b - first CTA of the virtual expert)
        # (the down's epilogue: ssacc is the down's CTA-local sum)
        if (tid < 32) & (V > 0):
            hs = tid // 16
            pk = fx.ptr_load(rt["pk"] + tid)
            v = (hs == 0).select(v0, v1)
            okh = (pk >= 0) & ((hs == 0) | (v1 != v0))
            cf = (v * 256) // fx.max(V, 1)
            if okh:
                mb_put(
                    tag,
                    scratch + fx.Int64(lay["ss"][0]),
                    pk * SSN + (bid - cf),
                    fx.ptr_load(rt["ssacc"] + tid),
                )
        stage_rstd(scratch, tag, health, rt, tid, bid)
        stage_final(scratch, sc, part_out, pl, tag, health, rt, tid, bid, rank)

    # The kernel and its launcher. tl (a timeline buffer) and w13r .. eptab (expert replicas) are
    # part of the launch signature the decoder passes; this build does not read them.

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def hm(
        h: Int64,
        ids: Int64,
        sc: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_post: Int64,
        part_out: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        tl: Int64,
        peer0: Int64,
        peer1: Int64,
        peer2: Int64,
        peer3: Int64,
        peer4: Int64,
        peer5: Int64,
        peer6: Int64,
        peer7: Int64,
        layer: Int32,
        rank: Int32,
        nrows: Int32,
        w13r: Int64,
        w13sr: Int64,
        w2r: Int64,
        w2sr: Int64,
        eptab: Int64,
    ):
        hm_body(
            h,
            ids,
            sc,
            w13,
            w13s,
            w2,
            w2s,
            w_post,
            part_out,
            scratch,
            epoch,
            health,
            tl,
            (
                peer0,
                peer1,
                peer2,
                peer3,
                peer4,
                peer5,
                peer6,
                peer7,
            ),
            layer,
            rank,
            nrows,
            w13r,
            w13sr,
            w2r,
            w2sr,
            eptab,
        )

    @flyc.jit
    def launch(
        h: Int64,
        ids: Int64,
        sc: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_post: Int64,
        part_out: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        tl: Int64,
        peer0: Int64,
        peer1: Int64,
        peer2: Int64,
        peer3: Int64,
        peer4: Int64,
        peer5: Int64,
        peer6: Int64,
        peer7: Int64,
        layer: Int32,
        rank: Int32,
        nrows: Int32,
        w13r: Int64,
        w13sr: Int64,
        w2r: Int64,
        w2sr: Int64,
        eptab: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _KEYSTR
        hm(
            h,
            ids,
            sc,
            w13,
            w13s,
            w2,
            w2s,
            w_post,
            part_out,
            scratch,
            epoch,
            health,
            tl,
            peer0,
            peer1,
            peer2,
            peer3,
            peer4,
            peer5,
            peer6,
            peer7,
            layer,
            rank,
            nrows,
            w13r,
            w13sr,
            w2r,
            w2sr,
            eptab,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch
