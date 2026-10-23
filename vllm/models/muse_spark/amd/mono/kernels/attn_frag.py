# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B002, E501, E741, SIM102, SIM108

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/attn_frag.py):
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
"""The verify attention of one Muse Spark layer as a fragment of the persistent kernels: K3
and HA run it at their head (attn = "swa" / "full"), so the layer's attention and its decoder
work are one launch.

Shapes (a TP8 rank): 16 q heads, 2 kv heads (QPK = 8), head dim 64, 8 query tokens a request,
NR requests (K3 S8: 1, S16: 2; HA S32: 4). K / V bf16 in the layer's vLLM LBHNC cache
[block][2][16][K 64 | V 64] (block 16); the page table is the layer group's block_table
(int32 [NR][stride]); seq_used = vLLM's seq_lens (int32 [NR], keys incl. the 8 new ones).
q [8 NR, 16, 64] bf16 (row stride 1024, without the QK scale); out attn_o [8 NR, 1024] bf16.
SWA: vLLM's window (2047, 0), keys qpos - key < 2048, and only keys in the window are read
(vLLM frees the blocks before it: their block-table entries are stale).

Split (every CTA): group = (request, kv head), CPG CTAs a group; CTA j of a group takes tiles
[lo_t + j tps, +tps) of 64 keys (tps = ceil(n_t / CPG) at run time) for all 64 rows (8 tokens x 8
heads) of the kv head. 8 waves = 4 row quarters (16 rows) x 2 tile lanes; a step = 2 tiles.
K / V tiles: global -> VGPR (PD steps ahead; one 64-bit block base a wave, a wave loads the 16
keys of one block) -> LDS. S^T = K Q^T (16x16x32 bf16 MFMA, a lane holds one q row), online
softmax (exp2 domain, the -inf -> 0 rule), O^T = V^T P^T (P^T lane-local by a key permutation
inside the 32-key k block; V^T from LDS with ds_read_b64_tr_b16). The 2 tile lanes merge in LDS;
the CTA writes (m, l, unnormalized o) for 64 rows (fp32, device scope) + one tagged flag.

Merge: item = (group, row) -> one output row of 64 d. NR = 1: CTA 2i merges item i with 512
threads (32 segment lanes x 16 d-quads); NR >= 2: NR / 2 items a CTA. Poll the group's flags,
load the partials, merge (max trick) per thread, over lanes and over waves, out = o / l -> bf16
-> attn_o and the same rows as tagged pairs (P_AO), which the consumer (K3 stage O, HA ng)
polls directly.

Descriptor (int64 words, device memory; vllm_desc): 0 q, 1 the KV cache base, 2 its block stride
in elements (a value), 3 block table, 4 seq_lens, 5 attn_o (out), 6 part (scratch of
part_bytes()), 7 block-table row stride (elements), 8 real requests (requests >= it are graph
padding: no split, zero output rows, their flags still set), 9 the softmax scale x log2(e) as
fp32 bits. A fragment's descriptor is row `layer` of a table of TABLE_ROWS rows.
"""

import math
from types import SimpleNamespace

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .dev import (
    BLOCKS,
    CM_DEV,
    CM_NT,
    OOB_BOUND,
    THREADS,
    bf16_pair,
    hw_exp2,
    ld_i32,
    ld_i32_if,
    ld_vec,
    ld_vec_if,
    mb_poll,
    mb_put,
    mb_put2,
    mfma_bf16,
    permlane_swap,
    rsrc,
    traced,
    uniform,
    xred,
)

HKV, QPK, HD, BS, TILE, NQ = 2, 8, 64, 16, 64, 8
NH = HKV * QPK
WIN_SWA = 2048  # vLLM's window (2047, 0) -> keys qpos - key < 2048
LOG2E = 1.0 / math.log(2.0)
KP = 72  # LDS row stride of a K / V tile (bf16)
KPW = KP // 2
NEG = float("-inf")
PGMAX = 512  # page ids a CTA staged in one round (128 tiles)
VLLM_MAX_KEYS = (
    1 << 20
)  # keys a request the page staging covers (max_position_embeddings)
REGION_IDS = {"aseg": 40}
DESC_WORDS = 10
TABLE_ROWS = 64

# parts of the scratch buffer (bytes); the segment count is always BLOCKS (G x CPG <= 256)
P_O = 0  # partial o, fp32 [seg][row][64]
P_ML = BLOCKS * 64 * HD * 4  # (m, l) [seg][row]
P_SF = P_ML + BLOCKS * 64 * 2 * 4  # one tagged flag a segment
P_IF = BLOCKS * 64 * (HD + 4) * 8  # one tagged flag an output row
P_AO = P_IF + 16 * 128 * 8  # attn_o as tagged pairs [row][512 words]
P_END = P_AO + 128 * 512 * 8
assert P_SF + BLOCKS * 8 <= P_IF


def part_bytes():
    return P_END


def default_cpg(NR, swa):
    g = NR * HKV
    if swa:
        return min(BLOCKS // g, 17)  # 2 tiles a CTA (34 tiles at most in the window)
    return BLOCKS // g


def make(NR, swa, maxk=VLLM_MAX_KEYS):
    """The fragment for NR requests; swa = SWA (window 2048) or full attention. It stages enough
    page ids a CTA for maxk keys a request (PGN)."""
    KCM = (
        0 if swa else CM_NT
    )  # K / V load cache policy: cached for SWA, nontemporal for full

    G = NR * HKV
    CPG = default_cpg(NR, swa)
    NST = -(
        -(-(-(34 if swa else 512) // CPG)) // 2
    )  # steps a CTA at the window / at 32k context
    PD = max(1, min(4, NST))  # K / V steps in flight
    assert G * CPG <= BLOCKS
    NI = NR * NQ * NH  # output rows
    if NI >= BLOCKS:
        IPC = NI // BLOCKS
        STR = 1
    else:
        IPC = 1
        STR = BLOCKS // NI
    TPI = THREADS // IPC
    WPI = TPI // 64
    SL = TPI // 16
    SPT = -(-CPG // SL)
    assert IPC * TPI == THREADS and WPI >= 1 and NI <= 16 * 128 and PGMAX == THREADS
    WIN = WIN_SWA if swa else 0
    PGN = PGMAX  # page ids staged a CTA (LDS); full attention: maxk keys a request (PGN // THREADS rounds)
    if not swa:
        tiles = -(-(-(-maxk // TILE)) // CPG)  # tiles a CTA at maxk keys
        PGN = max(PGMAX, -(-tiles * (TILE // BS) // THREADS) * THREADS)

    @fx.struct
    class Lds:
        kv: fx.Array[fx.Int32, 2 * 2 * 64 * KPW, 16]  # [tile lane][K | V][key][KP bf16]
        mrg: fx.Array[
            fx.Float32, 4 * 64 * 18, 16
        ]  # tile lane 1 -> 0: [quarter][lane][m, l, acc 16]
        mg2: fx.Array[fx.Float32, 8 * 16 * 6, 16]  # merge: [wave][d quad][M, L, A 4]
        pg: fx.Array[
            fx.Int32, PGN, 16
        ]  # the CTA's page ids, staged once (off the K / V vmcnt chain)

    def fmax(a, b):
        return (a > b).select(a, b)

    def fadd(a, b):
        return a + b

    def p64(dv, i):
        lo = uniform(dv[2 * i])
        hi = uniform(dv[2 * i + 1])
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def desc(dptr):
        dv = []
        for i in range(DESC_WORDS // 2):
            v4 = ld_vec(dptr, 4 * i, 4)
            dv += [v4[e] for e in range(4)]
        return SimpleNamespace(
            q=p64(dv, 0),
            k=p64(dv, 1),
            v=p64(dv, 2),
            pt=p64(dv, 3),
            used=p64(dv, 4),
            out=p64(dv, 5),
            part=p64(dv, 6),
            pts=uniform(dv[14]),
            nreq=uniform(dv[16]),
            qks=uniform(dv[18]).bitcast(fx.Float32),
        )

    def geom(d, req):
        """Run-time key-tile range of request req (uniform)."""
        seq_len = uniform(ld_i32(d.used, req))
        ctx = seq_len - NQ
        if const_expr(swa):
            lo = fx.max(ctx - WIN + 1, 0)
        else:
            lo = fx.Int32(0)
        lo_t = lo // TILE
        hi_t = (seq_len + TILE - 1) // TILE
        n_t = hi_t - lo_t
        tps = fx.max((n_t + CPG - 1) // CPG, 1)
        nseg = (n_t + tps - 1) // tps
        return seq_len, ctx, lo_t, hi_t, tps, nseg

    def tr16(kvb, byte):
        ptr = fx.to_llvm_ptr(fx.inttoptr(kvb.type, fx.Int64(byte)))
        return fx.Vector(
            rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), ptr).result, (4,), fx.BFloat16
        )

    def kv_load(d, t0, t1, seq_len, kvh, step, pgl, ltl, lkey, lch, wlo=None):
        tile = t0 + 2 * step + ltl
        kg = tile * TILE + lkey
        ok = (tile < t1) & (kg < seq_len)
        pg = fx.ptr_load(
            pgl + fx.min((2 * step + ltl) * (TILE // BS) + lkey // BS, PGN - 1)
        )
        if const_expr(swa):
            ok = ok & (
                kg >= wlo
            )  # keys before the window: no access (vLLM frees their blocks)
        base = d.k + (
            fx.Int64(uniform(pg)) * d.v + fx.Int64(kvh * (BS * 2 * HD))
        ) * fx.Int64(2)
        off = (
            (kg % BS) * HD + 8 * lch
        )  # dwords: row (kg % 16) of 128 bf16, channels [16 lch, 16 lch + 16)
        kv = [
            ld_vec_if(ok, base, off + o, 4, OOB_BOUND, KCM)
            for o in (0, 4, HD // 2, HD // 2 + 4)
        ]
        return fx.Vector.from_elements(
            [kv[j][i] for j in range(4) for i in range(4)], fx.Int32
        )

    def step(
        d,
        s,
        kvc,
        m,
        l,
        acc,
        t0,
        t1,
        seq_len,
        kvh,
        pgl,
        ltl,
        lkey,
        lch,
        wtl,
        r16,
        g,
        qpos,
        qb,
        kvl,
        kvbase,
        wlo=None,
    ):
        """One step (2 tiles): K / V of step s (kvc) -> LDS, the loads of step s + PD issued, then this wave's tile."""
        gpu.barrier()
        for kind in range_constexpr(2):
            words = [kvc[8 * kind + i] for i in range(8)]
            b0 = ((ltl * 2 + kind) * 64 + lkey) * KPW + 8 * lch
            fx.ptr_store(fx.Vector.from_elements(words[:4], fx.Int32), kvl + b0)
            fx.ptr_store(fx.Vector.from_elements(words[4:], fx.Int32), kvl + b0 + 4)
        gpu.barrier()
        rocdl.sched_barrier(0)
        kvn = kv_load(d, t0, t1, seq_len, kvh, s + PD, pgl, ltl, lkey, lch, wlo)
        rocdl.sched_barrier(0)
        tile = t0 + 2 * s + wtl
        kb = tile * TILE
        tin = tile < t1
        # S^T = K Q^T: lane holds q row r16, keys 16 n + 4 g + i
        xs = []
        mx = fx.Float32(NEG)
        for n in range_constexpr(4):
            sv = fx.Vector.filled(4, 0.0, fx.Float32)
            for c in range_constexpr(2):
                a = fx.Vector(
                    fx.ptr_load(
                        kvl
                        + ((wtl * 2) * 64 + 16 * n + r16) * KPW
                        + (32 * c + 8 * g) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                ).bitcast(fx.BFloat16)
                sv = mfma_bf16(a, qb[c], sv)
            for i in range_constexpr(4):
                key = kb + 16 * n + 4 * g + i
                ok = tin & (key <= qpos)
                if const_expr(swa):
                    ok = ok & ((qpos - key) < WIN)
                x = ok.select(sv[i] * d.qks, fx.Float32(NEG))
                xs.append(x)
                mx = fmax(mx, x)
        mx = xred(mx, 16, fmax)
        mx = xred(mx, 32, fmax)
        mj = fmax(m, mx)
        mj = (mj > fx.Float32(NEG)).select(mj, fx.Float32(0.0))
        ps = [hw_exp2(x - mj) for x in xs]
        alpha = hw_exp2(m - mj)
        psum = fx.Float32(0.0)
        for x in ps:
            psum = psum + x
        l = l * alpha + psum
        acc = [
            fx.Vector.from_elements([acc[j][i] * alpha for i in range(4)], fx.Float32)
            for j in range(4)
        ]
        m = mj
        # O^T += V^T P^T
        for c in range_constexpr(2):
            pb = fx.Vector.from_elements(
                [
                    bf16_pair(ps[(2 * c) * 4 + 0], ps[(2 * c) * 4 + 1]),
                    bf16_pair(ps[(2 * c) * 4 + 2], ps[(2 * c) * 4 + 3]),
                    bf16_pair(ps[(2 * c + 1) * 4 + 0], ps[(2 * c + 1) * 4 + 1]),
                    bf16_pair(ps[(2 * c + 1) * 4 + 2], ps[(2 * c + 1) * 4 + 3]),
                ],
                fx.Int32,
            ).bitcast(fx.BFloat16)
            for j in range_constexpr(4):
                row1 = 32 * c + 4 * g + r16 // 4
                col = 16 * j + 4 * (r16 % 4)
                vb = kvbase + (((wtl * 2 + 1) * 64 + row1) * KP + col) * 2
                lo_ = tr16(kvl, vb)
                hi_ = tr16(kvl, vb + 16 * KP * 2)
                a = lo_.shuffle(hi_, [0, 1, 2, 3, 4, 5, 6, 7])
                acc[j] = mfma_bf16(a, pb, acc[j])
        return m, l, acc, kvn

    @traced
    def split(d, tag, tid, bid, lane, wave, L):
        kvl = L.kv.ptr
        kvbase = fx.Int32(fx.ptrtoint(kvl))
        grp = bid // CPG
        jseg = bid % CPG
        req = grp // HKV
        kvh = grp % HKV
        seq_len, ctx, lo_t, hi_t, tps, nseg = geom(d, req)
        t0 = lo_t + jseg * tps
        t1 = fx.min(t0 + tps, hi_t)
        wlo = None
        if const_expr(swa):
            wlo = fx.max(ctx - WIN + 1, 0)  # the first key in the window
        if (bid < G * CPG) & (req < d.nreq):
            if t0 < t1:
                nst = (t1 - t0 + 1) // 2
                # load lanes: tile lane, key, 16 B chunk
                ltl = tid // 256
                lkey = (tid % 256) // 4
                lch = tid % 4

                # this wave: row quarter qr (16 q rows), tile lane wtl
                qr = wave % 4
                wtl = wave // 4
                r16 = lane % 16
                g = lane // 16
                m_row = 16 * qr + r16
                tok = m_row // QPK
                head = kvh * QPK + m_row % QPK
                qpos = ctx + tok
                qb = [
                    ld_vec(
                        d.q,
                        ((NQ * req + tok) * (NH * HD) + head * HD + 32 * c + 8 * g)
                        // 2,
                        4,
                    ).bitcast(fx.BFloat16)
                    for c in range(2)
                ]
                pgl = L.pg.ptr
                for r in range_constexpr(
                    1, PGN // THREADS
                ):  # full attention past 128 tiles a CTA
                    tr = tid + r * THREADS
                    pk = t0 * (TILE // BS) + tr
                    pok = (tr < (t1 - t0) * (TILE // BS)) & (pk * BS < seq_len)
                    fx.ptr_store(
                        ld_i32_if(pok, d.pt, req * d.pts + pk, OOB_BOUND), pgl + tr
                    )
                pk = t0 * (TILE // BS) + tid  # page tid of the CTA's range
                pok = (tid < (t1 - t0) * (TILE // BS)) & (pk * BS < seq_len)
                if const_expr(swa):
                    pok = pok & (
                        (pk + 1) * BS > wlo
                    )  # blocks before the window: stale ids, not read
                pgv = ld_i32_if(pok, d.pt, req * d.pts + pk, OOB_BOUND)
                fx.ptr_store(pgv, pgl + fx.min(tid, PGMAX - 1))
                gpu.barrier()
                kvq = []
                for i in range_constexpr(
                    PD
                ):  # in step order: the scheduler must not move slot 0 behind the others
                    kvq.append(
                        kv_load(d, t0, t1, seq_len, kvh, i, pgl, ltl, lkey, lch, wlo)
                    )
                    rocdl.sched_barrier(0)
                m = fx.Float32(NEG)
                l = fx.Float32(0.0)
                acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
                s = fx.Int32(0)
                while s < nst:
                    # PD steps a trip with fixed register slots (a rotated list makes LLVM copy the slots at the back
                    # edge behind a vmcnt wait); steps past nst are fully masked (no loads, no change of m / l / acc)
                    for j in range_constexpr(PD):
                        m, l, acc, kvn = step(
                            d,
                            s + j,
                            kvq[j],
                            m,
                            l,
                            acc,
                            t0,
                            t1,
                            seq_len,
                            kvh,
                            pgl,
                            ltl,
                            lkey,
                            lch,
                            wtl,
                            r16,
                            g,
                            qpos,
                            qb,
                            kvl,
                            kvbase,
                            wlo,
                        )
                        kvq = kvq[:j] + [kvn] + kvq[j + 1 :]
                    s = s + PD
                lt = xred(l, 16, fadd)
                lt = xred(lt, 32, fadd)
                mrg = L.mrg.ptr
                mb = (qr * 64 + lane) * 18
                if wtl == 1:
                    fx.ptr_store(m, mrg + mb)
                    fx.ptr_store(lt, mrg + mb + 1)
                    for j in range_constexpr(4):
                        for i in range_constexpr(4):
                            fx.ptr_store(acc[j][i], mrg + mb + 2 + 4 * j + i)
                gpu.barrier()
                if wtl == 0:
                    m1 = fx.ptr_load(mrg + mb)
                    l1 = fx.ptr_load(mrg + mb + 1)
                    mm = fmax(m, m1)
                    a0 = hw_exp2(m - mm)
                    a1 = hw_exp2(m1 - mm)
                    lf = lt * a0 + l1 * a1
                    seg = grp * CPG + jseg
                    po = d.part + fx.Int64(P_O)
                    for j in range_constexpr(4):
                        ov = [
                            acc[j][i] * a0 + fx.ptr_load(mrg + mb + 2 + 4 * j + i) * a1
                            for i in range(4)
                        ]
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [v_.bitcast(fx.Int32) for v_ in ov], fx.Int32
                            ),
                            rsrc(po),
                            ((seg * 64 + m_row) * HD + 16 * j + 4 * g),
                            cache_modifier=CM_DEV,
                        )
                    if g == 0:
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [mm.bitcast(fx.Int32), lf.bitcast(fx.Int32)],
                                fx.Int32,
                            ),
                            rsrc(d.part + fx.Int64(P_ML)),
                            (seg * 64 + m_row) * 2,
                            cache_modifier=CM_DEV,
                        )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == 0:
                    mb_put(tag, d.part + fx.Int64(P_SF), grp * CPG + jseg, fx.Int32(1))

    def comb(Ma, La, Aa, Mb, Lb, Ab):
        M = fmax(Ma, Mb)
        wa = (Ma > fx.Float32(NEG)).select(hw_exp2(Ma - M), fx.Float32(0.0))
        wb = (Mb > fx.Float32(NEG)).select(hw_exp2(Mb - M), fx.Float32(0.0))
        return M, La * wa + Lb * wb, [Aa[i] * wa + Ab[i] * wb for i in range(4)]

    def swap(off, v):
        a, b = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
        return a.bitcast(fx.Float32), b.bitcast(fx.Float32)

    @traced
    def merge(d, tag, health, tid, bid, lane, wave, L):
        if const_expr(STR > 1):
            has = (bid % STR) == 0
            item = bid // STR
        else:
            has = bid >= 0
            item = bid * IPC + tid // TPI
        if has:
            itl = tid // TPI
            tt = tid % TPI
            sl = tt // 16
            dq = tt % 16
            grp = item // 64
            row = item % 64
            req = grp // HKV
            kvh = grp % HKV
            real = req < d.nreq
            seq_len, ctx, lo_t, hi_t, tps, nseg = geom(d, fx.min(req, d.nreq - 1))
            mls, os_ = [], []
            sf = d.part + fx.Int64(P_SF)
            specs, oks = [], []
            for k in range_constexpr(SPT):
                sg = sl + SL * k
                specs.append((sf, grp * CPG + sg, 1))
                oks.append((sg < nseg) & real)
            mb_poll(tag, health, specs, REGION_IDS["aseg"], oks=oks)
            for k in range_constexpr(SPT):
                seg = grp * CPG + sl + SL * k
                mls.append(
                    ld_vec_if(
                        oks[k],
                        d.part + fx.Int64(P_ML),
                        (seg * 64 + row) * 2,
                        2,
                        P_SF - P_ML,
                        CM_DEV,
                    )
                )
                os_.append(
                    ld_vec_if(
                        oks[k],
                        d.part + fx.Int64(P_O),
                        (seg * 64 + row) * HD + 4 * dq,
                        4,
                        P_ML,
                        CM_DEV,
                    )
                )
            M = fx.Float32(NEG)
            for k in range_constexpr(SPT):
                M = oks[k].select(fmax(M, mls[k][0].bitcast(fx.Float32)), M)
            Ls = fx.Float32(0.0)
            A = [fx.Float32(0.0) for _ in range(4)]
            for k in range_constexpr(SPT):
                w = oks[k].select(
                    hw_exp2(mls[k][0].bitcast(fx.Float32) - M), fx.Float32(0.0)
                )
                Ls = Ls + w * mls[k][1].bitcast(fx.Float32)
                A = [A[i] + w * os_[k][i].bitcast(fx.Float32) for i in range(4)]
            for off in (16, 32):
                ma, mb_ = swap(off, M)
                la, lb = swap(off, Ls)
                aa, ab = [], []
                for i in range(4):
                    x_, y_ = swap(off, A[i])
                    aa.append(x_)
                    ab.append(y_)
                M, Ls, A = comb(ma, la, aa, mb_, lb, ab)
            g2 = L.mg2.ptr
            wi = tt // 64
            if lane < 16:
                b = ((itl * WPI + wi) * 16 + dq) * 6
                fx.ptr_store(M, g2 + b)
                fx.ptr_store(Ls, g2 + b + 1)
                for i in range_constexpr(4):
                    fx.ptr_store(A[i], g2 + b + 2 + i)
            gpu.barrier()
            if tt < 16:
                b0 = ((itl * WPI) * 16 + dq) * 6
                M = fx.ptr_load(g2 + b0)
                Ls = fx.ptr_load(g2 + b0 + 1)
                A = [fx.ptr_load(g2 + b0 + 2 + i) for i in range(4)]
                for w_ in range_constexpr(1, WPI):
                    b = ((itl * WPI + w_) * 16 + dq) * 6
                    M, Ls, A = comb(
                        M,
                        Ls,
                        A,
                        fx.ptr_load(g2 + b),
                        fx.ptr_load(g2 + b + 1),
                        [fx.ptr_load(g2 + b + 2 + i) for i in range(4)],
                    )
                f = fx.Float32(1.0) / (Ls > fx.Float32(0.0)).select(Ls, fx.Float32(1.0))
                tok = row // QPK
                head = kvh * QPK + row % QPK
                w2 = fx.Vector.from_elements(
                    [bf16_pair(A[0] * f, A[1] * f), bf16_pair(A[2] * f, A[3] * f)],
                    fx.Int32,
                )
                bo.buffer_store(
                    w2,
                    rsrc(d.out),
                    ((NQ * req + tok) * (NH * HD) + head * HD + 4 * dq) // 2,
                    cache_modifier=CM_DEV,
                )
                mb_put2(
                    tag,
                    d.part + fx.Int64(P_AO),
                    ((NQ * req + tok) * (NH * HD) + head * HD + 4 * dq) // 2,
                    w2[0],
                    w2[1],
                )
                rocdl.s_waitcnt(0)
                if tt == 0:
                    mb_put(tag, d.part + fx.Int64(P_IF), item, fx.Int32(1))

    def run(dptr, tag, health, tid, bid, lane, wave, L):
        """The whole fragment; returns (the attn_o pointer, the tagged attn_o rows)."""
        d = desc(dptr)
        split(d, tag, tid, bid, lane, wave, L)
        merge(d, tag, health, tid, bid, lane, wave, L)
        return d.out, d.part + fx.Int64(P_AO)

    return SimpleNamespace(
        Lds=Lds,
        run=run,
        key=f"attnfrag_nr{NR}_{'swa' if swa else 'full'}_c{CPG}_p{PD}_k{KCM}_{_SRC_HASH}",
    )


with open(__file__, "rb") as _f:
    import hashlib as _hl

    _SRC_HASH = _hl.sha1(_f.read()).hexdigest()[:12]


def vllm_scale_word(scale):
    """Descriptor word 9: scale x log2(e) as fp32 bits (an int64 word)."""
    import struct

    return struct.unpack("<i", struct.pack("<f", float(scale) * LOG2E))[0]


def vllm_desc(q, kv_cache, block_table, seq_lens, out, part, nreq, scale):
    """The 10 descriptor words of one layer: q [8 nreq.., 16 x 64] bf16 rows,
    kv_cache the layer's [block][2][16][128] bf16 view (block stride = kv_cache.stride(0)), block_table int32
    [>= nreq][stride] of the layer's group, seq_lens int32 [>= nreq], out attn_o [rows, 1024] bf16, part the scratch
    (int32, part_bytes()), nreq the requests of the step (graph padding included: seq_len 0 = no keys, zero rows)."""
    import torch

    assert (
        q.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and q.shape[-1] * (q.shape[-2] if q.dim() == 3 else 1) % (NH * HD) == 0
    )
    assert kv_cache.dtype == torch.bfloat16 and tuple(kv_cache.shape[1:]) == (
        HKV,
        BS,
        2 * HD,
    ), kv_cache.shape
    assert tuple(kv_cache.stride()[1:]) == (BS * 2 * HD, 2 * HD, 1), kv_cache.stride()
    assert (
        block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and block_table.shape[0] >= nreq
    )
    assert (
        seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
        and seq_lens.shape[0] >= nreq
    )
    assert (
        out.dtype == torch.bfloat16
        and out.is_contiguous()
        and part.numel() * part.element_size() >= part_bytes()
    )
    return [
        q.data_ptr(),
        kv_cache.data_ptr(),
        kv_cache.stride(0),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        out.data_ptr(),
        part.data_ptr(),
        block_table.stride(0),
        nreq,
        vllm_scale_word(scale),
    ]
