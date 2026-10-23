# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B002, B008, E501, E741, F841, SIM102, SIM108

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
"""S4 (#8 / #10): the DFlash verify attention as a fragment of the persistent kernels (TP8 shapes),
plus a standalone one-launch kernel with the same code (tests / timing).

Shapes (a TP8 rank): 16 q heads, 2 kv heads (QPK = 8), head dim 64, 8 query tokens a request,
NR requests (K3 S8: 1, K5 S32: 4). KV cache fp8 E4M3, paged [block][16][2][64] (k_scale =
v_scale = 1), page table int32 [NR][stride], seq_used int64 [NR] (keys incl. the 8 new ones).
q [8 NR, 16, 64] bf16 (row stride 1024); out attn_o [8 NR, 1024] bf16.

Split (every CTA): group = (request, kv head), CPG CTAs a group; CTA j of a group takes tiles
[lo_t + j tps, +tps) of 64 keys (tps = ceil(n_t / CPG) at run time) for all 64 rows (8 tokens x 8
heads) of the kv head. 8 waves = 4 row quarters (16 rows) x 2 tile lanes; a step = 2 tiles.
K / V tiles: global -> VGPR (one step ahead; page ids two steps ahead) -> bf16 -> LDS.
S^T = K Q^T (16x16x32 bf16 MFMA, a lane holds one q row), online softmax as mk/attn6._split
(exp2 domain, the -inf -> 0 rule), O^T = V^T P^T (P^T lane-local by a key permutation inside the
32-key k block; V^T from LDS with ds_read_b64_tr_b16). The 2 tile lanes merge in LDS; the CTA
writes (m, l, unnormalized o) for 64 rows (fp32, device scope) + one tagged flag.

Merge: item = (group, row) -> one output row of 64 d. NR = 1: CTA 2i merges item i with 512
threads (32 segment lanes x 16 d-quads); NR >= 2: NR / 2 items a CTA. Poll the group's flags,
load the partials, merge (max trick) per thread, over lanes and over waves, out = o / l -> bf16
-> attn_o, one tagged flag an item. Fused builds: every CTA then waits for all items and
invalidates its caches before stage O reads attn_o with plain loads.

Descriptor (int64 words, device memory): 0 q, 1 k_cache, 2 v_cache, 3 page table, 4 seq_used,
5 attn_o (out), 6 part (scratch of part_bytes()), 7 page-table row stride (elements), 8 real requests
(requests >= it are graph padding: no split, zero output rows, their flags still set), 9 unused.

vLLM build (kvl = "lbhnc", make(..., kvl="lbhnc")): K / V bf16 in the layer's LBHNC cache [block][2][16][K 64 | V 64]
(block 16): word 1 = the cache base, word 2 = its block stride in elements (a value); the page table is the layer
group's block_table (int32), seq_used = vLLM's seq_lens (int32, keys incl. the 8 new ones), word 9 = the softmax scale
x log2(e) as fp32 bits (vLLM: 0.125 x qk_scale_factor; q comes without the QK scale). SWA: vLLM's window (2047, 0),
keys qpos - key < 2048 (VLLM_WIN_SWA), and only keys in the window are read (vLLM frees the blocks before it: their
block-table entries are stale). One 64-bit block base a wave (a wave loads the 16 keys of one block).
"""

import math
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from .dev import (
    BLOCKS,
    CM_DEV,
    CM_NT,
    OOB_BOUND,
    THREADS,
    bf16_pair,
    bf_hi,
    bf_lo,
    cache_inv,
    hw_exp2,
    ld_i32,
    ld_i32_if,
    ld_vec,
    ld_vec_if,
    mb_poll,
    mb_put,
    mb_put2,
    memrealtime_after,
    mfma_bf16,
    permlane_swap,
    rsrc,
    st_if,
    traced,
    uniform,
    xred,
)

HKV, QPK, HD, BS, TILE, NQ = 2, 8, 64, 16, 64, 8
NH = HKV * QPK
WIN_SWA = 2047  # SGLang window (2046, 0) -> keys qpos - key < 2047
VLLM_WIN_SWA = 2048  # vLLM window (2047, 0) -> keys qpos - key < 2048
SCALE = 0.125
LOG2E = 1.0 / math.log(2.0)
KP = 72  # LDS row stride of a K / V tile (bf16)
KPW = KP // 2
NEG = float("-inf")
PGMAX = 512  # page ids a CTA (128 tiles: 262k keys a request at CPG = 32)
VLLM_MAX_KEYS = (
    1 << 20
)  # vLLM builds: keys a request the page staging covers (Muse Spark max_position_embeddings)
REGION_IDS = {"aseg": 40, "aitem": 41, "aall": 42}
DESC_WORDS = 10
ROWB = (
    DESC_WORDS * 8
)  # bytes a descriptor row; a fragment's descriptor is row `layer` of a table
PPC = 128  # plan: page ids a CTA (plan=1)
PLW = 4 + PPC  # plan row a CTA: seq_len, t0, t1, nseg, pages
PLAN_HDR = 64  # plan header: (seq_len, nseg) a group (<= 16 groups x 2 words), padded
PLAN_WORDS = PLAN_HDR + BLOCKS * PLW
PLAN_OFF = 8192  # bytes from the table start to plan 0 (swa); plan 1 (full) follows
TABLE_ROWS = 64

# parts of the scratch buffer (bytes); the segment count is always BLOCKS (G x CPG <= 256)
PROW = HD + 4  # tagged pairs a partial row: o[64], m, l, (pad)
P_O = 0  # tag = 1: tagged pairs [seg][row][PROW]; tag = 0: fp32 o [seg][row][64]
P_ML = BLOCKS * 64 * HD * 4  # tag = 0: (m, l) [seg][row]
P_SF = P_ML + BLOCKS * 64 * 2 * 4  # tag = 0: one tagged flag a segment
P_IF = BLOCKS * 64 * PROW * 8
P_AO = (
    P_IF + 16 * 128 * 8
)  # opt aot=1: attn_o as tagged pairs [row][512 words] (stage O / ng poll it directly)
P_TL = (
    P_AO + 128 * 512 * 8
)  # timeline probe (opt tl=1): [CTA][16] s_memrealtime words (int64)
P_END = P_TL + BLOCKS * 16 * 8
assert P_SF + BLOCKS * 8 <= P_IF
DEFAULT_OPTS = "cm=-1,tag=0,dbg=0,aot=1"  # cm -1: NT (2) for full attention, cached (0) for SWA (S4 bisect); aot: O polls rows


def parse_opts(opts):
    o = dict(x.split("=") for x in (DEFAULT_OPTS + "," + (opts or "")).split(",") if x)
    return {k: int(v) for k, v in o.items()}


def part_bytes():
    return P_END


def default_cpg(NR, swa):
    g = NR * HKV
    if swa:
        return min(BLOCKS // g, 17)  # 2 tiles a CTA (34 tiles at most in the window)
    return BLOCKS // g


def make(NR, swa, cpg=0, pd=0, opts=None, kvl="", maxk=VLLM_MAX_KEYS):
    """The fragment for NR requests; swa = SWA (window 2047) or full attention. pd: K / V steps in flight.
    kvl = "lbhnc": the vLLM build (bf16 LBHNC cache, int32 seq_lens, scale from the descriptor, window 2048), which
    stages enough page ids a CTA for maxk keys a request (PGN; SGLang builds: PGMAX, 262k keys at CPG = 32)."""
    assert kvl in ("", "lbhnc"), kvl
    VL = kvl == "lbhnc"
    O = parse_opts(opts)
    KCM, TAG = (
        O["cm"],
        O["tag"],
    )  # K / V load cache policy; partials as tagged pairs (1) or data + flag (0)
    if KCM < 0:
        KCM = 0 if swa else CM_NT
    DBG = O["dbg"]  # timing probes only: 1 = no softmax / MFMA work, 2 = no K / V loads
    TLP = O.get("tl", 0)
    PB = O.get(
        "pb", 0
    )  # partial o as bf16 o / l (numerics: one more rounding a partial)
    PLAN = O.get(
        "plan", 0
    )  # per-step plan (seq_len, tile range, page ids a CTA) next to the descriptor table
    PLANT = 0 if swa else 1
    AOT = O.get(
        "aot", 0
    )  # attn_o also as tagged pairs: the consumer (stage O / ng) polls the rows, no wait_all hop
    assert not (VL and PLAN), "plan=1: SGLang only"
    assert not (
        PB and TAG
    )  # timeline probe: per-CTA stamps into part + P_TL (thread 0)

    def st(d, tid, bid, k, dep):
        if const_expr(TLP):
            v = memrealtime_after(dep)
            st_if(
                tid == 0,
                fx.Vector.from_elements(
                    [fx.Int32(v & 0xFFFFFFFF), fx.Int32(v >> 32)], fx.Int32
                ),
                d.part + fx.Int64(P_TL),
                (bid * 16 + k) * 2,
            )

    G = NR * HKV
    CPG = cpg or default_cpg(NR, swa)
    NST = -(
        -(-(-(34 if swa else 512) // CPG)) // 2
    )  # steps a CTA at the window / at 32k context
    PD = pd or max(1, min(4, NST))
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
    WIN = (VLLM_WIN_SWA if VL else WIN_SWA) if swa else 0
    PGN = PGMAX  # page ids staged a CTA (LDS); vLLM full attention: maxk keys a request (PGN // THREADS rounds)
    if VL and not swa:
        tiles = -(-(-(-maxk // TILE)) // CPG)  # tiles a CTA at maxk keys
        PGN = max(PGMAX, -(-tiles * (TILE // BS) // THREADS) * THREADS)
    QKS = SCALE * LOG2E

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

    def fp8x4_bf16(dw):
        lo = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp8(
                T.vec(2, T.bf16),
                as_ir_value(fx.Int32(dw)),
                as_ir_value(fx.Float32(1.0)),
                False,
            )
        ).bitcast(fx.Int32)[0]
        hi = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp8(
                T.vec(2, T.bf16),
                as_ir_value(fx.Int32(dw)),
                as_ir_value(fx.Float32(1.0)),
                True,
            )
        ).bitcast(fx.Int32)[0]
        return lo, hi

    def p64(dv, i):
        lo = uniform(dv[2 * i])
        hi = uniform(dv[2 * i + 1])
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def desc(dptr, layer=None):
        pl = None
        if const_expr(
            PLAN
        ):  # the plan row of this CTA and the group headers: loads issued with the descriptor's
            pl = (
                dptr
                - fx.Int64(layer) * fx.Int64(ROWB)
                + fx.Int64(PLAN_OFF + PLANT * PLAN_WORDS * 4)
            )
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
            plan=pl,
            qks=uniform(dv[18]).bitcast(fx.Float32) if VL else None,
        )

    def geom(d, req):
        """Run-time key-tile range of request req (uniform)."""
        seq_len = uniform(
            ld_i32(d.used, req if VL else req * 2)
        )  # vLLM int32, SGLang int64 (low word)
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
        if const_expr(
            VL
        ):  # bf16 LBHNC: the wave's 16 keys are one block (one 64-bit base a wave)
            if const_expr(swa):
                ok = ok & (
                    kg >= wlo
                )  # keys before the window: no access (vLLM frees their blocks)
            if const_expr(DBG == 2):
                ok = ok & (seq_len < 0)
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
        off = (pg * (BS * HKV * HD) + (kg % BS) * (HKV * HD) + kvh * HD + 16 * lch) // 4
        if const_expr(DBG == 2):
            ok = ok & (seq_len < 0)
        kk = ld_vec_if(
            ok, d.k, off, 4, OOB_BOUND, KCM
        )  # keys past seq_len / the CTA tiles: zeros, no access
        vv = ld_vec_if(ok, d.v, off, 4, OOB_BOUND, KCM)
        return fx.Vector.from_elements(
            [kk[i] for i in range(4)] + [vv[i] for i in range(4)], fx.Int32
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
            words = []
            if const_expr(VL):
                words = [kvc[8 * kind + i] for i in range(8)]
            else:
                for i in range_constexpr(4):
                    lo_, hi_ = fp8x4_bf16(kvc[4 * kind + i])
                    words += [lo_, hi_]
            b0 = ((ltl * 2 + kind) * 64 + lkey) * KPW + 8 * lch
            fx.ptr_store(fx.Vector.from_elements(words[:4], fx.Int32), kvl + b0)
            fx.ptr_store(fx.Vector.from_elements(words[4:], fx.Int32), kvl + b0 + 4)
        gpu.barrier()
        rocdl.sched_barrier(0)
        kvn = kv_load(d, t0, t1, seq_len, kvh, s + PD, pgl, ltl, lkey, lch, wlo)
        rocdl.sched_barrier(0)
        if const_expr(DBG == 1):
            return m, l, acc, kvn
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
                x = ok.select(
                    sv[i] * (d.qks if const_expr(VL) else fx.Float32(QKS)),
                    fx.Float32(NEG),
                )
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
        if const_expr(PLAN):
            pr = ld_vec(
                d.plan, PLAN_HDR + bid * PLW, 4
            )  # seq_len, t0, t1, nseg (t0 = t1 = 0: no work)
            seq_len = uniform(pr[0])
            ctx = seq_len - NQ
            t0 = uniform(pr[1])
            t1 = uniform(pr[2])
            pgw = ld_i32_if(
                tid < PPC,
                d.plan,
                PLAN_HDR + bid * PLW + 4 + fx.min(tid, PPC - 1),
                OOB_BOUND,
            )
        else:
            seq_len, ctx, lo_t, hi_t, tps, nseg = geom(d, req)
            t0 = lo_t + jseg * tps
            t1 = fx.min(t0 + tps, hi_t)
        wlo = None
        if const_expr(VL and swa):
            wlo = fx.max(ctx - WIN + 1, 0)  # the first key in the window
        if (bid < G * CPG) & (req < d.nreq):
            st(d, tid, bid, 2, seq_len)
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
                ):  # vLLM full attention past 128 tiles a CTA
                    tr = tid + r * THREADS
                    pk = t0 * (TILE // BS) + tr
                    pok = (tr < (t1 - t0) * (TILE // BS)) & (pk * BS < seq_len)
                    fx.ptr_store(
                        ld_i32_if(pok, d.pt, req * d.pts + pk, OOB_BOUND), pgl + tr
                    )
                pk = t0 * (TILE // BS) + tid  # page tid of the CTA's range
                pok = (tid < (t1 - t0) * (TILE // BS)) & (pk * BS < seq_len)
                if const_expr(VL and swa):
                    pok = pok & (
                        (pk + 1) * BS > wlo
                    )  # blocks before the window: stale ids, not read
                if const_expr(PLAN):
                    big = (t1 - t0) * (
                        TILE // BS
                    ) > PPC  # more pages than a plan row holds: load them here
                    pgv = big.select(
                        ld_i32_if(pok & big, d.pt, req * d.pts + pk, OOB_BOUND), pgw
                    )
                else:
                    pgv = ld_i32_if(pok, d.pt, req * d.pts + pk, OOB_BOUND)
                fx.ptr_store(pgv, pgl + fx.min(tid, PGMAX - 1))
                gpu.barrier()
                st(d, tid, bid, 3, fx.ptr_load(pgl))
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
                st(d, tid, bid, 5, m.bitcast(fx.Int32))
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
                    if const_expr(TAG):
                        pb_ = (seg * 64 + m_row) * PROW
                        for j in range_constexpr(4):
                            ov = [
                                acc[j][i] * a0
                                + fx.ptr_load(mrg + mb + 2 + 4 * j + i) * a1
                                for i in range(4)
                            ]
                            mb_put2(
                                tag,
                                po,
                                pb_ + 16 * j + 4 * g,
                                ov[0].bitcast(fx.Int32),
                                ov[1].bitcast(fx.Int32),
                            )
                            mb_put2(
                                tag,
                                po,
                                pb_ + 16 * j + 4 * g + 2,
                                ov[2].bitcast(fx.Int32),
                                ov[3].bitcast(fx.Int32),
                            )
                        if g == 0:
                            mb_put2(
                                tag,
                                po,
                                pb_ + HD,
                                mm.bitcast(fx.Int32),
                                lf.bitcast(fx.Int32),
                            )
                    elif const_expr(
                        PB
                    ):  # o / l as bf16 (half the bytes; one rounding a partial)
                        rl = fx.Float32(1.0) / (lf > fx.Float32(0.0)).select(
                            lf, fx.Float32(1.0)
                        )
                        for j in range_constexpr(4):
                            ov = [
                                (
                                    acc[j][i] * a0
                                    + fx.ptr_load(mrg + mb + 2 + 4 * j + i) * a1
                                )
                                * rl
                                for i in range(4)
                            ]
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [bf16_pair(ov[0], ov[1]), bf16_pair(ov[2], ov[3])],
                                    fx.Int32,
                                ),
                                rsrc(po),
                                ((seg * 64 + m_row) * HD + 16 * j + 4 * g) // 2,
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
                    else:
                        for j in range_constexpr(4):
                            ov = [
                                acc[j][i] * a0
                                + fx.ptr_load(mrg + mb + 2 + 4 * j + i) * a1
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
                if const_expr(not TAG):
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    st(d, tid, bid, 6, tid)
                    if tid == 0:
                        mb_put(
                            tag, d.part + fx.Int64(P_SF), grp * CPG + jseg, fx.Int32(1)
                        )

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
            if const_expr(PLAN):
                nseg = uniform(ld_i32(d.plan, grp * 2 + 1))
            else:
                seq_len, ctx, lo_t, hi_t, tps, nseg = geom(d, fx.min(req, d.nreq - 1))
            po = d.part + fx.Int64(P_O)
            mls, os_ = [], []
            if const_expr(TAG):
                specs, oks = [], []
                for k in range_constexpr(SPT):
                    sg = sl + SL * k
                    pb_ = ((grp * CPG + sg) * 64 + row) * PROW
                    specs += [
                        (po, pb_ + HD, 2),
                        (po, pb_ + 4 * dq, 2),
                        (po, pb_ + 4 * dq + 2, 2),
                    ]
                    ok = (sg < nseg) & real
                    oks += [ok, ok, ok]
                got = mb_poll(tag, health, specs, REGION_IDS["aseg"], oks=oks)
                for k in range_constexpr(SPT):
                    mls.append(got[3 * k])
                    os_.append(got[3 * k + 1] + got[3 * k + 2])
                oks = [oks[3 * k] for k in range(SPT)]
            else:
                sf = d.part + fx.Int64(P_SF)
                specs, oks = [], []
                for k in range_constexpr(SPT):
                    sg = sl + SL * k
                    specs.append((sf, grp * CPG + sg, 1))
                    oks.append((sg < nseg) & real)
                pw = mb_poll(tag, health, specs, REGION_IDS["aseg"], oks=oks)
                st(d, tid, bid, 7, pw[0][0])
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
                    if const_expr(PB):
                        ob = ld_vec_if(
                            oks[k],
                            d.part + fx.Int64(P_O),
                            ((seg * 64 + row) * HD + 4 * dq) // 2,
                            2,
                            P_ML,
                            CM_DEV,
                        )
                        lk = mls[k][1].bitcast(fx.Float32)
                        os_.append(
                            [
                                (bf_lo(ob[0]) * lk).bitcast(fx.Int32),
                                (bf_hi(ob[0]) * lk).bitcast(fx.Int32),
                                (bf_lo(ob[1]) * lk).bitcast(fx.Int32),
                                (bf_hi(ob[1]) * lk).bitcast(fx.Int32),
                            ]
                        )
                    else:
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
                st(d, tid, bid, 8, f.bitcast(fx.Int32))
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
                if const_expr(AOT):
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
                st(d, tid, bid, 9, tid)

    @traced
    def wait_all(d, tag, health, tid, wave):
        if tid < NI:
            mb_poll(
                tag, health, [(d.part + fx.Int64(P_IF), tid, 1)], REGION_IDS["aall"]
            )
        gpu.barrier()
        if wave == 0:
            cache_inv(False)
        gpu.barrier()

    def run(dptr, tag, health, tid, bid, lane, wave, L, fused, layer=None):
        """The whole fragment; returns the attn_o pointer (Int64). fused: wait for every row. layer: the row of dptr
        in its descriptor table (plan=1 finds the plan from it)."""
        t0_ = memrealtime_after(tid) if const_expr(TLP) else None
        d = desc(dptr, layer)
        if const_expr(TLP):
            st_if(
                tid == 0,
                fx.Vector.from_elements(
                    [fx.Int32(t0_ & 0xFFFFFFFF), fx.Int32(t0_ >> 32)], fx.Int32
                ),
                d.part + fx.Int64(P_TL),
                (bid * 16) * 2,
            )
        st(d, tid, bid, 1, fx.Int32(d.pts))
        split(d, tag, tid, bid, lane, wave, L)
        merge(d, tag, health, tid, bid, lane, wave, L)
        if const_expr(fused and not AOT):
            wait_all(d, tag, health, tid, wave)
            st(d, tid, bid, 10, tid)
        if const_expr(AOT):
            return d.out, d.part + fx.Int64(P_AO)
        return d.out

    return SimpleNamespace(
        Lds=Lds,
        run=run,
        split=split,
        merge=merge,
        desc=desc,
        CPG=CPG,
        PD=PD,
        O=O,
        WIN=WIN,
        AOT=AOT,
        NI=NI,
        G=G,
        PGN=PGN,
        max_keys=(10**9 if swa else PGN // (TILE // BS) * CPG * TILE),
        key=f"attnfrag_nr{NR}_{'swa' if swa else 'full'}_c{CPG}_p{PD}_k{KCM}t{TAG}d{DBG}l{TLP}b{PB}n{PLAN}a{AOT}"
        + ("_kvlbhnc" if VL else "")
        + f"_{_SRC_HASH}",
    )


_SA = {}
with open(__file__, "rb") as _f:
    import hashlib as _hl

    _SRC_HASH = _hl.sha1(_f.read()).hexdigest()[:12]


def build_standalone(NR, swa, cpg=0, mode=0, pd=0, kvl=""):
    """Mode 0: split + merge; 1: empty kernel (the launch baseline, same LDS); 2: split only."""
    k = (NR, swa, cpg, mode, pd, kvl)
    if k in _SA:
        return _SA[k]
    F = make(NR, swa, cpg, pd, kvl=kvl)
    _KEYSTR = F.key + f"_m{mode}"

    @flyc.kernel(
        name="s4_" + F.key.rsplit("_", 1)[0] + f"_m{mode}",
        known_block_size=[THREADS, 1, 1],
    )
    def kern(dptr: Int64, epoch: Int64, health: Int64, layer: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        L = fx.SharedAllocator().allocate(F.Lds).peek()
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        if const_expr(mode == 0):
            _ = F.run(dptr, tag, health, tid, bid, lane, wave, L, False, layer)
        if const_expr(mode == 1):
            if tid == 0:
                bo.buffer_store(tag, rsrc(health), 7)
        if const_expr(mode == 2):
            F.split(F.desc(dptr, layer), tag, tid, bid, lane, wave, L)

    @flyc.jit
    def launch(
        dptr: Int64,
        epoch: Int64,
        health: Int64,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _KEYSTR
        kern(dptr, epoch, health, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _SA[k] = launch
    return launch


_PLAN_K = {}


def _plan_kernel():
    if "k" not in _PLAN_K:
        import triton
        import triton.language as tl

        @triton.jit
        def plan(
            pt_ptr,
            pts,
            used_ptr,
            out_ptr,
            nreq,
            CPG: tl.constexpr,
            G: tl.constexpr,
            WIN: tl.constexpr,
            PPCc: tl.constexpr,
            PLWc: tl.constexpr,
            HDR: tl.constexpr,
        ):
            b = tl.program_id(0)
            grp = b // CPG
            j = b % CPG
            req = grp // 2
            ok_cta = (b < G * CPG) & (req < nreq)
            reqc = tl.minimum(req, nreq - 1)
            sl = tl.load(used_ptr + reqc).to(tl.int32)
            ctx = sl - 8
            if WIN > 0:
                lo = tl.maximum(ctx - WIN + 1, 0)
            else:
                lo = sl * 0
            lo_t = lo // 64
            hi_t = (sl + 63) // 64
            n_t = hi_t - lo_t
            tps = tl.maximum((n_t + CPG - 1) // CPG, 1)
            nseg = (n_t + tps - 1) // tps
            t0 = lo_t + j * tps
            t1 = tl.minimum(t0 + tps, hi_t)
            t0 = tl.where(ok_cta, t0, 0)
            t1 = tl.where(ok_cta & (t1 > t0), t1, t0)
            row = out_ptr + HDR + b * PLWc
            tl.store(row + 0, sl)
            tl.store(row + 1, t0)
            tl.store(row + 2, t1)
            tl.store(row + 3, nseg)
            p = tl.arange(0, PPCc)
            pk = t0 * 4 + p
            okp = (p < (t1 - t0) * 4) & (pk * 16 < sl)
            pg = tl.load(pt_ptr + reqc.to(tl.int64) * pts + pk, mask=okp, other=0)
            tl.store(row + 4 + p, pg)
            if b < G:  # group header b: request b // 2
                rq = tl.minimum(b // 2, nreq - 1)
                s2 = tl.load(used_ptr + rq).to(tl.int32)
                c2 = s2 - 8
                if WIN > 0:
                    l2 = tl.maximum(c2 - WIN + 1, 0)
                else:
                    l2 = s2 * 0
                n2 = (s2 + 63) // 64 - l2 // 64
                tp2 = tl.maximum((n2 + CPG - 1) // CPG, 1)
                tl.store(out_ptr + b * 2, s2)
                tl.store(out_ptr + b * 2 + 1, (n2 + tp2 - 1) // tp2)

        _PLAN_K["k"] = plan
    return _PLAN_K["k"]


def run_plan(table, NR, swa, pt, used, cpg=0):
    """The per-step plan of one attention type into its slot after the descriptor table (one small launch)."""
    G = NR * HKV
    CPG = cpg or default_cpg(NR, swa)
    out = table.view(-1).view(
        table.dtype
    )  # int64 table; the plan is int32 words at PLAN_OFF
    w32 = table.view(-1).view(__import__("torch").int32)
    base = PLAN_OFF // 4 + (0 if swa else 1) * PLAN_WORDS
    _plan_kernel()[(BLOCKS,)](
        pt,
        pt.stride(0),
        used,
        w32[base:],
        used.shape[0],
        CPG=CPG,
        G=G,
        WIN=WIN_SWA if swa else 0,
        PPCc=PPC,
        PLWc=PLW,
        HDR=PLAN_HDR,
        num_warps=1,
    )
    return out


def table_words():
    return (PLAN_OFF + 2 * PLAN_WORDS * 4) // 8


class Host:
    """Host side: one descriptor (row `layer` of its own table, with the two plan slots after it) and the scratch of
    one fragment instance (one per launch site)."""

    def __init__(self, dev):
        import torch

        self.torch = torch
        self.table = torch.zeros(table_words(), dtype=torch.int64, device=dev)
        self.desc = self.table[:DESC_WORDS]
        self.part = torch.zeros(part_bytes() // 4, dtype=torch.int32, device=dev)

    def set(self, q, k, v, pt, used, out, layer=3, cpg=0):
        t = self.torch
        assert q.dtype == t.bfloat16 and q.is_contiguous() and q.shape[1:] == (NH, HD)
        assert (
            k.dtype == t.float8_e4m3fn
            and k.shape[1:] == (BS, HKV, HD)
            and k.is_contiguous()
            and v.is_contiguous()
        )
        assert (
            pt.dtype == t.int32
            and pt.stride(1) == 1
            and used.dtype == t.int64
            and out.is_contiguous()
        )
        vals = [
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            pt.data_ptr(),
            used.data_ptr(),
            out.data_ptr(),
            self.part.data_ptr(),
            pt.stride(0),
            used.shape[0],
            0,
        ]
        self.desc = self.table[layer * DESC_WORDS : (layer + 1) * DESC_WORDS]
        self.desc.copy_(t.tensor(vals, dtype=t.int64))
        NR = q.shape[0] // NQ
        for swa in (True, False):
            run_plan(self.table, NR, swa, pt, used, cpg)
        return self.desc


class LayerDescs:
    """Server side (sglang_mono): the descriptors of every layer of one K3 / K5 set as one device table (row i =
    layer i, then the two plan slots), refreshed once a forward with one H2D copy from a pinned host buffer plus the
    plan launches. In graph capture the copy is a graph node that reads its own pinned buffer (written only while
    capturing, so later eager steps of other shapes do not change what the replays copy); eager steps use another
    pinned buffer and a blocking copy."""

    def __init__(self, dev, NR):
        import torch

        self.torch, self.dev, self.NR = torch, dev, NR
        self.part = torch.zeros(part_bytes() // 4, dtype=torch.int32, device=dev)
        self.table = {}  # rows -> device table (one a graph bucket / eager shape)
        self.pinned = {}
        # pinned host buffers for the capture-time copies: allocated here, because a pinned (host) allocation is not
        # permitted while a stream is capturing (hipErrorStreamCaptureUnsupported); one is taken per captured shape
        self._cap_pool = [
            torch.zeros(TABLE_ROWS, DESC_WORDS, dtype=torch.int64, pin_memory=True)
            for _ in range(8)
        ]

    def fill(self, key, rows, plans=()):
        """rows: {layer: (q, k, v, pt, used, out)}; plans: [(swa, pt, used)] -> the device table."""
        t = self.torch
        cap = t.cuda.is_current_stream_capturing()
        pk = (key, cap)
        if pk not in self.pinned:
            if cap:
                if not self._cap_pool:
                    raise RuntimeError(
                        "attn_frag.LayerDescs: more captured shapes than pre-allocated pinned buffers"
                    )
                self.pinned[pk] = self._cap_pool.pop()
            else:
                self.pinned[pk] = t.zeros(
                    TABLE_ROWS, DESC_WORDS, dtype=t.int64, pin_memory=True
                )
            if key not in self.table:
                self.table[key] = t.zeros(table_words(), dtype=t.int64, device=self.dev)
        h = self.pinned[pk]
        for i, (q, k, v, pt, used, out) in rows.items():
            assert (
                q.dtype == t.bfloat16
                and pt.dtype == t.int32
                and pt.stride(1) == 1
                and used.dtype == t.int64
            )
            h[i] = t.tensor(
                [
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                    pt.data_ptr(),
                    used.data_ptr(),
                    out.data_ptr(),
                    self.part.data_ptr(),
                    pt.stride(0),
                    used.shape[0],
                    0,
                ],
                dtype=t.int64,
            )
        d = self.table[key]
        d[: TABLE_ROWS * DESC_WORDS].view(TABLE_ROWS, DESC_WORDS).copy_(
            h, non_blocking=cap
        )  # eager: blocking
        for swa, pt, used in plans:
            run_plan(d, self.NR, swa, pt, used)
        return d

    @staticmethod
    def row(table, i):
        return table[i * DESC_WORDS : (i + 1) * DESC_WORDS]


def vllm_scale_word(scale):
    """Descriptor word 9 of the vLLM build: scale x log2(e) as fp32 bits (an int64 word)."""
    import struct

    return struct.unpack("<i", struct.pack("<f", float(scale) * LOG2E))[0]


def vllm_desc(q, kv_cache, block_table, seq_lens, out, part, nreq, scale):
    """The 10 descriptor words of the vLLM build (kvl = "lbhnc") for one layer: q [8 nreq.., 16 x 64] bf16 rows,
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
