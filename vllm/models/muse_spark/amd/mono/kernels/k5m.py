# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501, F841, SIM102

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
"""HM (milestone 5): this rank's routed experts for S = 32 / 64 / 128 rows as one persistent
launch between HA and HB (replaces the library MoE: align, gate_up, down, finalize, and HB's
partial push).

Inputs from HA: h [S, 4096] bf16 (pre_expert_norm output), top-8 ids [S, 8] and weights [S, 8].
Every CTA builds the same tables: the picks of each local expert in token order, split into
virtual experts of at most 16 picks (one MFMA N tile). Tasks: (virtual expert v, row tile j),
256 a virtual expert (gate_up: 16 gate + 16 up rows; down: 16 output rows); CTA b owns tasks
[b V, (b+1) V) (V virtual experts), so it spans at most two virtual experts and keeps their
picks' h (gate_up) / a (down) rows in LDS. Weights MXFP4 (w13 / w2 + E8M0 scales), streamed
with a prefetch queue, decoded in registers (v_cvt_scalef32_pk_bf16_fp4), bf16 MFMA.

  gate_up: a = bf16(silu(bf16 g) bf16 u) -> scratch A (tagged pairs, pick rows)
  down:    z = bf16(w_post bf16(y)) -> scratch Z (tagged), per-pick sums of squares a CTA
  rstd:    CTA c: the post-expert-norm rstd of picks c, c + 256, .. (tagged)
  final:   CTA b, columns 16b ..: part[t] = sum over t's local picks (k order) of
           bf16(z rstd) w -> pushed to the owner rank of the column quarter (b / 64) = HB's
           reduce-scatter input (+ flags), and part_out (debug copy).
Same tag as HA / HB of the layer (epoch * 64 + layer + 1).
"""

import math
import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from . import k5 as k5m
from . import portion as _portion
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
    bpermute,
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
    min_tree,
    permlane_swap,
    popcount,
    readlane,
    rsrc,
    st_if,
    stamp,
    traced,
    uniform,
    wave_incl_scan,
    xred,
)

DM = 4096
INTER = 4096
E = 256
EL = 64  # TP4 value; builds use E // W
TOPK = 8
FCH = 256
POST_EPS = 1e-8
LOG2E = 1.0 / math.log(2.0)
HROW = DM + 8
AROW = INTER + 8
SSN = 257  # ss partial slots a pick (CTAs covering one virtual expert, at most 256 / V + 1)
REGION = {"a": 31, "z": 32, "ss": 33, "rs": 34}


@dataclass(frozen=True)
class BuildM:
    S: int = 32
    pf_gu: int = 1
    pf_dn: int = 2
    timeline: bool = False
    eplb: int = 0  # M7: replica expert slots (mk/eplb.py)
    W: int = 4  # TP = EP group (4 / 8)
    only: str = (
        ""  # ablation: stage mask (mk/portion.py; "" = every stage, the shipped kernel)
    )
    unr: int = int(
        os.environ.get("MK5M_UNR", "-1")
    )  # down task-loop ring (unroll 3, fixed slots; PF_DN 2): 1 on, 0 off, -1 auto (on at W = 8, S != 96)
    # vLLM layouts of the MXFP4 experts, read in place (mk/shuffle.py; = mk/k3.py Build.shuffled_w / shuffled_s):
    # w13 / w2 in aiter shuffle_weight((16, 16)), w13s / w2s in aiter e8m0_shuffle. False: the native layouts
    shuffled_w: bool = False
    shuffled_s: bool = False


def build_vllm(S, W=8):
    """The HM build vLLM launches (no env reads): TEP8, no EPLB, the experts in vLLM's layouts read in place."""
    assert S in (32, 64, 96, 128) and W == 8, (S, W)
    return BuildM(
        S=S,
        pf_gu=1,
        pf_dn=2,
        timeline=False,
        eplb=0,
        W=W,
        only="",
        unr=-1,
        shuffled_w=True,
        shuffled_s=True,
    )


def _lay(items):
    out, off = {}, 0
    for name, n in items:
        off = (off + 255) // 256 * 256
        out[name] = (off, n)
        off += n
    return out, (off + 4095) // 4096 * 4096


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
    W = key.W
    assert W in (4, 8) and S in (32, 64, 96, 128) and S % W == 0, (W, S)
    EL = E // W  # local experts (64 / 32)
    CPR = BLOCKS // W  # 16-column blocks a rank owns in the EP reduce-scatter (64 / 32)
    MASKW = (1 << W) - 1  # eplb holder mask bits
    SB = 8 + W  # eplb flist packing shift
    P = TOPK * S
    RB = key.eplb
    NU = EL + RB  # weight slots: own 0 .. EL-1, replicas EL .. NU-1
    assert NU % WAVES == 0
    VMAX = NU + P // 16
    lay, _ = scratch_layout(S)
    play, _ = k5m.peer_layout(S, W)
    TL = key.timeline
    PF_GU, PF_DN = key.pf_gu, key.pf_dn
    UNRD = ((W == 8 and S != 96) if key.unr < 0 else key.unr == 1) and PF_DN == 2
    W13_BYTES = EL * 2 * INTER * (DM // 2)
    W13S_BYTES = EL * 2 * INTER * (DM // 32)
    W2_BYTES = EL * DM * (INTER // 2)
    W2S_BYTES = EL * DM * (INTER // 32)
    tag_par = lambda tag: tag & 1  # noqa: E731

    @fx.struct
    class RouteM:
        nu: fx.Array[fx.Int32, NU, 16]
        vstart: fx.Array[fx.Int32, NU, 16]
        wi: fx.Array[fx.Int32, 256, 16]
        gpk: fx.Array[fx.Int32, 256, 16]
        flist: fx.Array[fx.Int32, 256, 16]
        ve_u: fx.Array[fx.Int32, VMAX, 16]
        ve_n: fx.Array[fx.Int32, VMAX, 16]
        ve_off: fx.Array[fx.Int32, VMAX, 16]
        nve: fx.Array[fx.Int32, 4, 16]
        pinfo: fx.Array[fx.Int32, P, 16]
        pk: fx.Array[fx.Int32, 2 * 16, 16]
        ssacc: fx.Array[fx.Float32, 2 * 16, 16]
        rsl: fx.Array[fx.Float32, P, 16]
        tls: fx.Array[fx.Int64, 8, 16]

    @fx.struct
    class MSetup:
        lists: fx.Array[fx.Int32, NU * S, 16]

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

    name = (
        f"muse_k5m_s{S}_g{PF_GU}_d{PF_DN}_t{int(TL)}_eb{RB}"
        + ("" if W == 4 else f"_w{W}")
        + ("_u" if UNRD else "")
        + _portion.name_tag(key.only)
        + ("_shw" if key.shuffled_w else "")
        + ("_shs" if key.shuffled_s else "")
    )
    SHUF_W, SHUF_S = key.shuffled_w, key.shuffled_s
    ONLY = _portion.stage_set(key.only, _portion.HM_STAGES, _portion.K5_DEPS)

    def on(stage):  # stage active in this build (const at trace time)
        return ONLY is None or stage in ONLY

    def sel_peer(pl, d):
        p = pl[0]
        for r_ in range_constexpr(1, W):
            p = (d == r_).select(pl[r_], p)
        return p

    # ------------------------------------------------------------ tables
    @traced
    def eplb_assign(ids, rank, tid, lane, wave, rt, nrows, ept):
        """mk/k3.py eplb_assign on the global ids (identical on every CTA of every rank)."""
        if tid < 256:
            fx.ptr_store(fx.Int32(0), rt["gpk"] + tid)
        gpu.barrier()
        for q in range_constexpr((P + THREADS - 1) // THREADS):
            p = tid + THREADS * q
            if (p < P) & ((p // TOPK) < nrows):
                fx.ptr_store(
                    fx.Int32(1), rt["gpk"] + fx.min(fx.max(ld_i32(ids, p), 0), 255)
                )
        gpu.barrier()
        if wave == 0:
            cnt = [fx.Int32(0) for _ in range(W)]
            carry = fx.Int32(0)
            for hh in range_constexpr(4):
                e = lane + 64 * hh
                f = fx.ptr_load(rt["gpk"] + e) != 0
                t = ept[hh]
                m = t & MASKW
                single = (m & (m - 1)) == 0
                for r_ in range_constexpr(W):
                    cnt[r_] = cnt[r_] + popcount(ballot(f & single & (m == (1 << r_))))
                mine1 = f & single & (m == (fx.Int32(1) << rank))
                fx.ptr_store(mine1.select(e - rank * EL, fx.Int32(-1)), rt["wi"] + e)
                flex = (f & (~single)).select(fx.Int32(1), fx.Int32(0))
                incl = wave_incl_scan(flex, lane) + carry
                if flex != 0:
                    fx.ptr_store(
                        e | (m << 8) | ((t >> 8) << SB), rt["flist"] + (incl - flex)
                    )
                carry = readlane(incl, 63)
            nfx = uniform(carry)
            if const_expr(W == 8):
                # TP8: lane r < W holds rank r's key cnt * W + r; the flexible experts are read from
                # registers (lane l: entries l, l + 64, ..), the chosen rank of entry i goes to lane
                # i % 64 of rs[i / 64], and the wi stores run after the loop on all lanes (no LDS
                # access, no lane-0 branch, no `rank` in the serial loop). Same greedy, same ties
                # (lower count, then lower rank) as the one-lane loop below: bit-identical tables.
                kk = fx.Int32(0)
                for r_ in range_constexpr(W):
                    kk = kk + (lane == r_).select(cnt[r_], fx.Int32(0))
                kk = kk * W + lane
                bitv = (lane < W).select(
                    fx.Int32(256) << (lane & 7), fx.Int32(0)
                )  # holder bit of rank `lane`
                bigv = fx.Int32(1 << 20)
                fl0 = fx.ptr_load(rt["flist"] + lane)
                fl1 = fx.ptr_load(rt["flist"] + (lane + 64))
                fl2 = fx.ptr_load(rt["flist"] + (lane + 128))
                fl3 = fx.ptr_load(rt["flist"] + (lane + 192))
                rs0 = fx.Int32(-1)
                rs1 = fx.Int32(-1)
                rs2 = fx.Int32(-1)
                rs3 = fx.Int32(-1)
                iv = fx.Int32(0)
                while iv < nfx:
                    ch = iv >> 6
                    flv = (ch == 0).select(
                        fl0, (ch == 1).select(fl1, (ch == 2).select(fl2, fl3))
                    )
                    vv = readlane(flv, iv & 63)
                    kb = ((vv & bitv) != 0).select(kk, bigv)
                    for off in (4, 2, 1):
                        kb = xred(kb, off, fx.min)
                    rv = readlane(kb, 0) & (W - 1)
                    kk = kk + (lane == rv).select(fx.Int32(W), fx.Int32(0))
                    rs0 = (lane == iv).select(rv, rs0)
                    rs1 = (lane + 64 == iv).select(rv, rs1)
                    rs2 = (lane + 128 == iv).select(rv, rs2)
                    rs3 = (lane + 192 == iv).select(rv, rs3)
                    iv = iv + 1
                fls = [fl0, fl1, fl2, fl3]
                rss = [rs0, rs1, rs2, rs3]
                for c in range_constexpr(4):
                    flc = fls[c]
                    rsc = rss[c]
                    ev = flc & 255
                    ownv = (ev // EL) == rank
                    wsl = ownv.select(ev - rank * EL, fx.Int32(EL - 1) + (flc >> SB))
                    if (lane + 64 * c) < nfx:
                        fx.ptr_store(
                            (rsc == rank).select(wsl, fx.Int32(-1)), rt["wi"] + ev
                        )
            else:
                if lane == 0:
                    cs = list(cnt)
                    i = fx.Int32(0)
                    while i < nfx:
                        v = fx.ptr_load(rt["flist"] + i)
                        e = v & 255
                        m = (v >> 8) & MASKW
                        big = fx.Int32(1 << 20)
                        ks = [
                            ((m & (1 << r_)) != 0).select(cs[r_] * W + r_, big)
                            for r_ in range(W)
                        ]
                        kb = min_tree(
                            ks
                        )  # pairwise (W = 4: min(min(k0, k1), min(k2, k3)))
                        r = kb & (W - 1)
                        cs = [
                            cs[r_] + (r == r_).select(fx.Int32(1), fx.Int32(0))
                            for r_ in range(W)
                        ]
                        own = (e // EL) == rank
                        wslot = own.select(e - rank * EL, fx.Int32(EL - 1) + (v >> SB))
                        fx.ptr_store(
                            (r == rank).select(wslot, fx.Int32(-1)), rt["wi"] + e
                        )
                        i = i + 1
        gpu.barrier()

    @traced
    def setup(ids, rank, tid, lane, wave, rt, lists, nrows, ept=None):
        tid = fresh(tid)
        NC = P // 64
        for q in range_constexpr((P + THREADS - 1) // THREADS):
            p = tid + THREADS * q
            if p < P:
                fx.ptr_store(fx.Int32(-1), rt["pinfo"] + p)
        if const_expr(RB > 0):
            eplb_assign(ids, rank, tid, lane, wave, rt, nrows, ept)
        # every wave: the weight slot of every pick, 64 picks a register
        ev = []
        for c in range_constexpr(NC):
            real = ((c * 64 + lane) // TOPK) < nrows  # padded rows route nowhere
            if const_expr(RB > 0):
                e = fx.ptr_load(
                    rt["wi"] + fx.min(fx.max(ld_i32(ids, c * 64 + lane), 0), 255)
                )
                ev.append(real.select(e, fx.Int32(-1)))
            else:
                e = ld_i32(ids, c * 64 + lane) - rank * EL
                ev.append(((e >= 0) & (e < EL) & real).select(e, fx.Int32(-1)))
        # wave w: slots w, w + 8, ..: pick lists in pick (= token) order
        for uu in range_constexpr(NU // WAVES):
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
            for hh in range_constexpr((NU + 63) // 64):
                u = lane + 64 * hh
                n = (u < NU).select(
                    fx.ptr_load(rt["nu"] + fx.min(u, NU - 1)), fx.Int32(0)
                )
                ng = (n + 15) // 16
                incl = wave_incl_scan(ng, lane) + vcarry
                st = incl - ng
                if u < NU:
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
        for q in range_constexpr((NU * S + THREADS - 1) // THREADS):
            x = tid + THREADS * q
            u = fx.min(x // S, NU - 1)
            i = x % S
            if (x < NU * S) & (i < fx.ptr_load(rt["nu"] + u)):
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
        if const_expr(RB > 0):
            isr = e >= EL
            w13 = isr.select(rt["w13r"], w13)
            w13s = isr.select(rt["w13sr"], w13s)
            e = isr.select(e - EL, e)
        rg = wave // 4
        ops = []
        for i in range_constexpr(4):
            c = 4 * (wave % 4) + i
            row = e * (2 * INTER) + rg * INTER + j * 16 + r
            rowc = e * (2 * INTER) + rg * INTER + j * 16 + lane // 8
            rbase = (
                (e * (2 * INTER) + rg * INTER + j * 16)
                if const_expr(SHUF_W or SHUF_S)
                else None
            )
            if const_expr(
                SHUF_W
            ):  # lane L: the 16 B at 16 L of the 1 KB block (mk/k3.py gu_loads)
                wb = (rbase * (DM // 2) + (2 * c) * 1024 + lane * 16) // 4
                ops.append(ld_vec_if(ok, w13, wb, 4, W13_BYTES, CM_NT))
                ops.append(ld_vec_if(ok, w13, wb + 256, 4, W13_BYTES, CM_NT))
            else:
                wb = (rowc * (DM // 2) + 128 * c + 16 * (lane % 8)) // 4
                ops.append(ld_vec_if(ok, w13, wb, 4, W13_BYTES, CM_NT))
                ops.append(
                    ld_vec_if(ok, w13, wb + 8 * (DM // 2) // 4, 4, W13_BYTES, CM_NT)
                )
            if const_expr(
                SHUF_S
            ):  # e8m0_shuffle: one 256-B load a chunk (mk/k3.py gu_loads)
                sdw = (
                    (rbase >> 5) * (8 * (DM // 32))
                    + 64 * c
                    + 16 * (2 * (g % 2) + g // 2)
                    + r
                )
                ops.append(ld_i32_if(ok, w13s, sdw, W13S_BYTES))
            else:
                ops.append(
                    ld_i32_if(
                        ok,
                        w13s,
                        (row * (DM // 32) + 8 * c + 4 * (g // 2)) // 4,
                        W13S_BYTES,
                    )
                )
        return ops

    def fp4_mma(ops, base, bl, brow, lane, acc, c, jpar=None):
        """Chunk c (256 K) of a 16-row group times LDS B rows lane % 16: 8 MFMA steps.
        ``jpar``: the task's 16-row group parity (j & 1), for the shuffled scales (SHUF_S)."""
        g = lane // 16
        t = lane % 16
        sw = ops[base + 2]
        if const_expr(SHUF_S):  # see mk/k3.py fp4_mma
            sa_, sb_ = permlane_swap(32, sw, sw)
            shs = (g // 2) * 16 + jpar * 8
            s0 = e8m0_f32((sa_ >> shs) & 0xFF)
            s1 = e8m0_f32((sb_ >> shs) & 0xFF)
        else:
            sh = (g % 2) * 16
            s0 = e8m0_f32((sw >> sh) & 0xFF)
            s1 = e8m0_f32((sw >> (sh + 8)) & 0xFF)
        r = lane % 16
        src = 8 * (r % 8) + 2 * g
        hi = r >= 8
        xs = []
        if const_expr(SHUF_W):
            # shuffled, contiguous loads: permlane32_swap then permlane16_swap give the MFMA operand (mk/k3.py fp4_mma)
            x0, x1 = [], []
            for jj in range_constexpr(4):
                a_, b_ = permlane_swap(32, ops[base][jj], ops[base + 1][jj])
                p_, q_ = permlane_swap(16, a_, b_)
                x0.append(p_)
                x1.append(q_)
            xs = [x0, x1]
        elif const_expr(W == 8):
            # TP8 (R5 #3): one bpermute an output dword after a 1-lane DPP pre-mix (see mk/k3.py fp4_mma)
            odd = (lane & 1) == 1
            hv = hi.select(fx.Int32(1), fx.Int32(0))
            x0, x1 = [], []
            for jj in range_constexpr(4):
                a0 = fx.Int32(ops[base][jj])
                a1 = fx.Int32(ops[base + 1][jj])
                m_ = odd.select(
                    fx.Int32(update_dpp_i32(a1, a1, 0x111, 0xF, 0xF, False)), a0
                )
                n_ = odd.select(
                    a0, fx.Int32(update_dpp_i32(a1, a1, 0x101, 0xF, 0xF, False))
                )
                x0.append(bpermute(m_, src + hv))
                x1.append(bpermute(n_, src + 1 - hv))
            xs = [x0, x1]
        else:
            for h2 in range_constexpr(2):
                xv = []
                for jj in range_constexpr(4):
                    pa = bpermute(ops[base][jj], src + h2)
                    pb = bpermute(ops[base + 1][jj], src + h2)
                    xv.append(hi.select(pb, pa))
                xs.append(xv)
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
                jpar=(task & 1) if SHUF_S else None,
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
        if const_expr(RB > 0):
            isr = e >= EL
            w2 = isr.select(rt["w2r"], w2)
            w2s = isr.select(rt["w2sr"], w2s)
            e = isr.select(e - EL, e)
        row = e * DM + j * 16 + r
        rowc = e * DM + j * 16 + lane // 8
        rbase = (e * DM + j * 16) if const_expr(SHUF_W or SHUF_S) else None
        ops = []
        for i in range_constexpr(2):
            c = 2 * wave + i
            if const_expr(SHUF_W):
                wb = (rbase * (INTER // 2) + (2 * c) * 1024 + lane * 16) // 4
                ops.append(ld_vec_if(ok, w2, wb, 4, W2_BYTES, CM_NT))
                ops.append(ld_vec_if(ok, w2, wb + 256, 4, W2_BYTES, CM_NT))
            else:
                wb = (rowc * (INTER // 2) + 128 * c + 16 * (lane % 8)) // 4
                ops.append(ld_vec_if(ok, w2, wb, 4, W2_BYTES, CM_NT))
                ops.append(
                    ld_vec_if(ok, w2, wb + 8 * (INTER // 2) // 4, 4, W2_BYTES, CM_NT)
                )
            if const_expr(SHUF_S):
                sdw = (
                    (rbase >> 5) * (8 * (INTER // 32))
                    + 64 * c
                    + 16 * (2 * (g % 2) + g // 2)
                    + r
                )
                ops.append(ld_i32_if(ok, w2s, sdw, W2S_BYTES))
            else:
                ops.append(
                    ld_i32_if(
                        ok,
                        w2s,
                        (row * (INTER // 32) + 8 * c + 4 * (g // 2)) // 4,
                        W2S_BYTES,
                    )
                )
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
                jpar=(task & 1) if SHUF_S else None,
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
        rocdl.s_waitcnt(0)
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
            "wi": rl.wi.ptr,
            "gpk": rl.gpk.ptr,
            "flist": rl.flist.ptr,
            "w13r": w13r,
            "w13sr": w13sr,
            "w2r": w2r,
            "w2sr": w2sr,
        }
        tls = rl.tls.ptr
        pl = list(peers)
        stamp(TL, tls, tid, 0)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        ept = None
        if const_expr(RB > 0 and on("m_setup")):
            ept = [ld_i32(eptab, lane + 64 * hh) for hh in range(4)]
        if const_expr(on("m_setup")):
            setup(ids, rank, tid, lane, wave, rt, ms.lists.ptr, nrows, ept)
        V = uniform(fx.ptr_load(rt["nve"] + 0))
        t0 = bid * V
        t1 = t0 + V
        v0 = t0 >> 8
        v1 = (fx.max(t1, 1) - 1) >> 8
        q = []  # (a function-level name: the scf.while rewriter carries only names bound at this level)
        if const_expr(on("m_gate_up")):
            q = [
                gu_loads(w13, w13s, rt, t0 + d, lane, wave, (t0 + d) < t1)
                for d in range(PF_GU)
            ]
        if const_expr(on("m_setup")):
            own_picks(rt, ms.lists.ptr, v0, v1, tid)
        stamp(TL, tls, tid, 1)
        # gate_up over virtual expert v0, then (if the range crosses) v1
        if const_expr(on("m_gate_up")):
            load_rows_h(h, rt, 0, mg.h.ptr, tid)
        task = t0
        it = fx.Int32(0)
        tsplit = fx.min(t1, (v0 + 1) << 8)
        if const_expr(on("m_gate_up")):
            while task < tsplit:
                nxt = gu_loads(
                    w13, w13s, rt, task + PF_GU, lane, wave, (task + PF_GU) < t1
                )
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
                nxt = gu_loads(
                    w13, w13s, rt, task + PF_GU, lane, wave, (task + PF_GU) < t1
                )
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
        stamp(TL, tls, tid, 2)
        # down: same tasks
        dq = []
        if const_expr(on("m_down")):
            dq = [
                dn_loads(w2, w2s, w_post, rt, t0 + d, lane, wave, (t0 + d) < t1)
                for d in range(PF_DN)
            ]
            gpu.barrier()
            load_rows_a(scratch, tag, health, rt, 0, md.a.ptr, tid)
        stamp(TL, tls, tid, 3)
        if const_expr(on("m_down")):
            task = t0
            it = fx.Int32(0)
            if const_expr(UNRD):
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
            if const_expr(UNRD):
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
        stamp(TL, tls, tid, 4)
        # per-CTA sums of squares of the picks of v0 / v1 -> slot (b - first CTA of the virtual expert)
        # (the down's epilogue: ssacc is the down's CTA-local sum, so it runs with m_down)
        if const_expr(on("m_down")):
            if (tid < 32) & (V > 0):
                hs = tid // 16
                n = tid % 16
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
        if const_expr(on("m_rstd")):
            stage_rstd(scratch, tag, health, rt, tid, bid)
        stamp(TL, tls, tid, 5)
        if const_expr(on("m_final")):
            stage_final(scratch, sc, part_out, pl, tag, health, rt, tid, bid, rank)
        stamp(TL, tls, tid, 6)
        if const_expr(TL):  # noqa: SIM102
            if tid < 8:
                v_ = fx.ptr_load(tls + tid)
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [fx.Int32(v_ & 0xFFFFFFFF), fx.Int32(v_ >> 32)], fx.Int32
                    ),
                    rsrc(tl),
                    (bid * 8 + tid) * 2,
                )

    # kernel and launcher: peer0 .. peer3 at W = 4 (the TP4 signature), peer0 .. peer7 at W = 8
    if W == 4:

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
                layer,
                rank,
                nrows,
                w13r,
                w13sr,
                w2r,
                w2sr,
                eptab,
            ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)
    else:

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
