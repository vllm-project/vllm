# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501, SIM102

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/k5.py):
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
"""HA and HB: one Muse Spark decoder layer for S = 32 / 64 / 96 / 128 rows as two persistent
launches around HM (k5m.py, the local experts); one rank of a TP = EP = 8 group.

    [attention] -> HA -> HM -> HB

HA  optional: this layer's verify attention (attn_frag.py) at the head; o-norm x gate
    (distributed) -> o_proj (rows 32 b .., K = this rank's 1024) -> the bf16 partials go to
    the owner rank -> attention boundary (post_attn_norm, gated residual, pre-FFN norm) ->
    ffn_in -> router (fp32, 32 experts a rank) and pre_expert_proj (512 rows a rank), K in 8
    parts over 8-CTA groups -> logits and h_raw all-gathered -> top-8 (CTA t routes token t)
    -> h = pre_expert_norm(h_raw) (CTA t: token t) -> h, top-k ids / weights for HM.
HB  EP all-reduce of HM's partial (reduce-scatter to the column-block owner + all-gather) ->
    post_expert_proj (this rank's 1024 rows, K in 4 parts) -> post-FFN norm, gated residual,
    next input norm -> xn -> qkv + gate (K eighths) -> head finishers (qk-norm, RoPE,
    q out, bf16 K / V into vLLM's LBHNC KV cache).

The boundaries (Build.bnd). S > 32, token owners: rank r owns tokens t = r (mod 8); the o_proj
partials and the post_expert_proj outputs go to the owner rank of each token, one CTA a token
does the whole row (all-reduce sum, both norms, gated residual) and the ffn_in / xn rows are
all-gathered. The residual stream lives token-sharded (rank r: its S / 8 rows, fp32, all 8192
columns); HA reads it as rows so the first layer can read a full replicated stream (s_full);
HB writes it the same way and (``gather``) all-gathers it. S = 32, column blocks: the o_proj
partials go to the owner of the column block, which reduces and all-gathers them; both
boundaries run on all 256 CTAs (token groups x column groups, one-hop token norms) and the
stream is replicated.

Every row tile is 16 tokens (MFMA N); dense stages keep their weights in VGPRs and loop over
the S / 16 row tiles with the activations read straight from L2 / peer memory in MFMA
layout. Tags: tag = epoch * 64 + layer + 1 (HA and HB of a layer share it, regions are
disjoint); peer regions double buffered by tag parity.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from .dev import (
    BLOCKS,
    CM_DEV,
    CM_SYS,
    THREADS,
    WAVES,
    _add,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fresh,
    hw_exp2,
    hw_rsq,
    hw_sqrt,
    ld_f32,
    ld_i32,
    ld_vec,
    mb_poll,
    mb_put,
    mb_put_if,
    mb_put_sys,
    mfma_bf16,
    readlane,
    rsrc,
    st_if,
    traced,
    uniform,
    wave_sum,
    xred,
)
from .k3 import _lay, bf_mma_operand, bf_tile_load, route_token, sel_peer

W = 8  # ranks in the TP = EP group
HID = 8192
QH = HID // W  # hidden columns a rank owns (= stage-A K part = post-proj rows)
DM = 4096
E = 256
EL = E // W  # local experts = router experts a rank = stage-A groups
KQ = BLOCKS // EL  # stage-A K parts
OWN = 16 // KQ  # pre-proj rows a CTA owns after the K-part exchange
NRP = OWN // 2  # their row pairs
NTILES = QH // 16  # post-proj 16-row tiles a rank
KSP = BLOCKS // NTILES  # post-proj K split
TOPK = 8
ROUTE_EPS = 1e-15
PRE_EPS = 1e-5
POST_EPS = 1e-8
KCH = 128
HQ, HKV, HD = 128 // W, 16 // W, 64  # q / kv heads a rank
QROWS, KROWS = HQ * HD, HKV * HD
QKV_ROWS = QROWS + 2 * KROWS
QKVG_ROWS = QKV_ROWS + QROWS
OK_K = HID // W  # o_proj K a rank
NQTILES = QKVG_ROWS // 16  # qkvg 16-row tiles a rank
# qkvg tiles a CTA group (32 groups x 8 K eighths; 144 of 160 real)
NQT = -(-NQTILES // 32)
TL_POINTS = 32
# loads of peer-written data (after its system-scope flag): device scope = past the per-XCD L2,
# served by the memory-side cache, which every write to this memory passes
XCM = CM_DEV
YCM = CM_DEV  # activations re-read by many CTAs
ACM = CM_DEV  # stage A's x tiles
HEALTH_WORDS = 8 + 8 * BLOCKS
REGION_IDS = {
    "yf": 1,
    "opf": 2,
    "xf": 6,
    "logit": 7,
    "px": 8,
    "hf": 9,
    "mf": 10,
    "mmf": 11,
    "py": 12,
    "ytf": 13,
    "xnf": 16,
    "qx": 17,
    "qkv": 18,
    "sgf": 19,
    # the column-block boundaries (bnd)
    "opb": 20,
    "orb": 21,
    "yb": 22,
    "xinbf": 23,
    "xnbf": 24,
    "ssa4": 25,
    "ssg4": 26,
    "gpa": 27,
    "gpg": 28,
}


@dataclass(frozen=True)
class Build:
    S: int = 32  # rows (32 / 64 / 96 / 128)
    next_rope: bool = True  # the next layer has RoPE (SWA) or not (full, NoPE)
    # this layer's verify attention fused at the HA head ("swa" / "full")
    attn: str = ""

    @property
    def bnd(self):
        """The boundaries: 3 at S = 32 (o reduce-scattered to the column-block owner and
        all-gathered, both boundaries on all 256 CTAs with one-hop token norms over column
        groups, the stream replicated); 0 (S > 32): the token-owner boundaries (one CTA a
        token, the stream token-sharded and all-gathered by HB)."""
        return 3 if self.S == 32 else 0


def scratch_layout(S, world=W, bnd=0):
    assert world == W, world
    TS = S // W
    extra = (
        []
        if not bnd
        else [
            # the mid stream, replicated [t][col] fp32 (each CTA its 32 columns)
            ("smidb", S * HID * 4),
            ("xinb", S * HID * 2),
            ("xinbf", BLOCKS * 8),  # ffn_in [t][col] bf16 + a flag a CTA
            ("xnb", S * HID * 2),
            ("xnbf", BLOCKS * 8),  # the next layer's input [t][col] bf16 + a flag a CTA
            ("ssa4", (4 * S * BLOCKS + 2 * S) * 8),
            ("ssg4", (4 * S * BLOCKS + 2 * S) * 8),  # partials, rstd
            ("gpa", BLOCKS * 16 * 8),
            # [CTA][4 tokens][4] partials of the AB / G
            ("gpg", BLOCKS * 16 * 8),
        ]
    )
    return _lay(
        [
            ("y", S * OK_K * 2),
            ("yf", BLOCKS * 8),
            ("smid", TS * HID * 4),
            ("px", EL * KQ * 16 * S * 8),
            ("m", S * DM * 2),
            ("mmf", BLOCKS * 8),
            # [K part - 1][tile][row][t]
            ("py", max(128, (KSP - 1) * NTILES) * 16 * S * 8),
            ("qx", 32 * 8 * NQT * 16 * S * 8),
            ("qkv", S * (QKV_ROWS // 2) * 8),
        ]
        + extra
    )


def peer_layout(S, world=W, bnd=0):
    assert world == W, world
    TS = S // W
    extra = (
        []
        if not bnd
        else [
            # o_proj partials to the block owner [par][src][t][own blk][pair]
            ("opb", 2 * W * S * (BLOCKS // W) * 16 * 8),
            # the owner's reduced o, all-gathered [par][t][pair] tagged
            ("orb", 2 * S * (HID // 2) * 8),
            # stage F's y, all-gathered [par][t][pair] tagged
            ("yb", 2 * S * (HID // 2) * 8),
        ]
    )
    # the bnd regions go last: HM (k5m) addresses moep / mf with the layout of the plain build
    return _lay(
        [
            ("op", 2 * W * TS * (HID // 2) * 4),
            ("opf", 2 * W * BLOCKS * 8),
            ("xin", 2 * S * (HID // 2) * 4),
            ("xf", 2 * S * 8),
            ("logit", 2 * KQ * S * E * 8),
            ("hraw", 2 * S * (DM // 2) * 4),
            ("hf", 2 * W * BLOCKS * 8),
            ("moep", 2 * W * S * (DM // 2) * 4),
            ("mf", 2 * W * BLOCKS * 8),
            ("moem", 2 * S * (DM // 2) * 4),
            ("mmf", 2 * BLOCKS * 8),
            ("yt", 2 * TS * (HID // 2) * 4),
            ("ytf", 2 * W * NTILES * 8),
            ("xn", 2 * S * (HID // 2) * 4),
            ("xnf", 2 * S * 8),
            ("sg", 2 * S * HID * 4),
            ("sgf", 2 * S * 8),
        ]
        + extra
    )


_BUILDS = {}


def build(key: Build):
    if key in _BUILDS:
        return _BUILDS[key]
    _KEYSTR = repr(key)  # FlyDSL disk-cache key (see k3.build)
    S = key.S
    assert S in (32, 64, 96, 128), S
    NT = S // 16
    TS = S // W  # tokens a rank owns in the token-owner boundaries
    CO = OK_K // KCH // WAVES  # o_proj chunks a wave
    CA = QH // KCH // WAVES  # stage-A pre-proj chunks a wave
    CP = (DM // KSP) // KCH // WAVES  # post-proj chunks a wave
    KPT = QH // THREADS  # router K a thread
    NFQ = W * BLOCKS // THREADS  # producer flags a thread (opf / hf polls)
    QPF = 3  # qkvg tiles issued before stage G (of NQT)
    CPR = BLOCKS // W  # CTAs (16-column blocks) a rank owns in the EP reduce-scatter
    assert TS * W == S and NFQ * THREADS == W * BLOCKS and W * NTILES * 8 >= 512
    BND = key.bnd != 0  # the column-block boundaries (S = 32)
    QR1 = S == 32  # QKV5 reduces both 16-token row tiles in one LDS round
    lay, _ = scratch_layout(S, W, key.bnd)
    play, _ = peer_layout(S, W, key.bnd)
    tag_par = lambda tag: tag & 1

    @fx.struct
    class RouteLds:
        rid: fx.Array[fx.Int32, TOPK, 16]
        rw: fx.Array[fx.Float32, TOPK, 16]
        sel: fx.Array[fx.Int32, TOPK * 3 + 8, 16]

    @fx.struct
    class HaO:
        red: fx.Array[fx.Float32, NT * WAVES * 2 * 64 * 4, 16]

    XAROW = QH + 8  # LDS row of the x slice tile (bf16)

    @fx.struct
    class HaA:
        x: fx.Array[fx.Int32, 16 * XAROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tok: fx.Array[fx.Float32, WAVES * 16 + 8 + S, 16]
        own: fx.Array[fx.Float32, NT * OWN * 16, 16]

    FA = None
    if key.attn:  # the fused attention fragment (attn_frag.py)
        from . import attn_frag

        assert key.attn in ("swa", "full"), key
        FA = attn_frag.make(S // 8, key.attn == "swa")
        FaLds = FA.Lds
        # the fragment's source hash: the disk cache does not follow FA.run
        _KEYSTR = _KEYSTR + FA.key

    @fx.union
    class HaLds:
        o_: HaO
        a_: HaA
        if FA is not None:
            fa_: FaLds

    @fx.struct
    class HbF:
        red: fx.Array[fx.Float32, NT * WAVES * 64 * 4, 16]

    @fx.struct
    class HbQ:
        red: fx.Array[fx.Float32, (NT if QR1 else 1) * WAVES * NQT * 64 * 4, 16]
        own: fx.Array[fx.Float32, NT * NQT * 2 * 16, 16]

    @fx.union
    class HbLds:
        f_: HbF
        q_: HbQ

    @fx.struct
    class Misc:
        pad: fx.Array[fx.Int64, TL_POINTS, 16]  # unused: keeps the tuned LDS offsets
        # bnd: tok_rstd2 uses rs[2S .. 2S + 32)
        rs: fx.Array[fx.Float32, 2 * S + (32 if BND else 8), 16]
        # bnd: the polled group partials + wave sums
        pr: fx.Array[fx.Float32, 512 + 64 if BND else 8, 16]
        red4: fx.Array[fx.Float32, 32, 16]

    name_a = f"muse_k5a_s{S}" + (f"_fa{key.attn}" if key.attn else "")
    name_b = f"muse_k5b_s{S}_r{int(key.next_rope)}"

    # ------------------------------------------------------------ shared helpers
    def red_at(red, base_tile, rr, tt):
        """LDS offset of row rr (0..15), token tt (0..15) of MFMA tile slot base_tile."""
        return red + ((base_tile * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4)

    @traced
    def block_sum(v, lane, wave, red4, slot):
        """Sum of v over the CTA's 512 threads (fixed order: lanes, then waves)."""
        v = wave_sum(v)
        if lane == 0:
            fx.ptr_store(v, red4 + (slot * 8 + wave))
        gpu.barrier()
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red4 + (slot * 8 + w))
        return tot

    # ============================================================ HA
    def o_loads(w_o, bid, lane, wave):
        ops = []
        for i in range_constexpr(2):
            for ci in range_constexpr(CO):
                c = wave + WAVES * ci
                ops.append(bf_tile_load(w_o, bid * 32 + 16 * i, OK_K, c * KCH, lane))
        return ops

    @traced
    def stage_ng(attn_o, attn_g, g_stride, scratch, tag, tid, bid, nrows, ao=None):
        """Y = bf16(o rms(o) sigmoid(g)) per 64-wide head, distributed: 8 lanes a (token, head)."""
        tid = fresh(tid)
        gt = bid * THREADS + tid
        item = gt // 8
        if item < S * HQ:
            t = item // HQ
            kk = (item % HQ) * HD + (gt % 8) * 8
            # padded rows (t >= nrows) repeat the last real row
            tl_ = fx.min(t, nrows - 1)
            if const_expr(ao is None):
                xv = ld_vec(attn_o, (tl_ * OK_K + kk) // 2, 4)
            else:  # the fused attention: the merged rows as tagged pairs
                got = mb_poll(
                    tag,
                    ao[1],
                    [
                        (ao[0], (tl_ * OK_K + kk) // 2, 2),
                        (ao[0], (tl_ * OK_K + kk) // 2 + 2, 2),
                    ],
                    43,
                )
                xv = got[0] + got[1]
            gv = ld_vec(attn_g, (tl_ * g_stride + kk) // 2, 4)
            ss = fx.Float32(0.0)
            for d in range_constexpr(4):
                ss = ss + bf_lo(xv[d]) * bf_lo(xv[d]) + bf_hi(xv[d]) * bf_hi(xv[d])
            ss = butterfly(ss, (1, 2, 4))
            rstd = fx.Float32(1.0) / hw_sqrt(
                ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS)
            )
            ow = []
            for d in range_constexpr(4):
                s0 = fx.Float32(1.0) / (
                    fx.Float32(1.0) + hw_exp2(bf_lo(gv[d]) * fx.Float32(-LOG2E))
                )
                s1 = fx.Float32(1.0) / (
                    fx.Float32(1.0) + hw_exp2(bf_hi(gv[d]) * fx.Float32(-LOG2E))
                )
                ow.append(bf16_pair(bf_lo(xv[d]) * rstd * s0, bf_hi(xv[d]) * rstd * s1))
            bo.buffer_store(
                fx.Vector.from_elements(ow, fx.Int32),
                rsrc(scratch + fx.Int64(lay["y"][0])),
                (t * OK_K + kk) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if (tid == 0) & (bid < S * HQ * 8 // THREADS):
            mb_put(tag, scratch + fx.Int64(lay["yf"][0]), bid, fx.Int32(1))

    @traced
    def stage_o(o_ops, scratch, pl, tag, health, tid, bid, lane, wave, rank, red):
        """o_proj rows 32b .. (this rank's K slice) for all S tokens; the bf16 partial goes to
        the owner rank of the column block (bnd: b // CPR) or of each token (t % W)."""
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        if tid < S * HQ * 8 // THREADS:
            mb_poll(
                tag,
                health,
                [(scratch + fx.Int64(lay["yf"][0]), tid, 1)],
                REGION_IDS["yf"],
            )
        gpu.barrier()
        ybase = scratch + fx.Int64(lay["y"][0])
        # MFMA A operands once (reused by every row tile)
        oa = [
            [
                bf_mma_operand(o_ops[i * CO + ci], qq, lane).bitcast(fx.BFloat16)
                for qq in range(4)
            ]
            for i in range(2)
            for ci in range(CO)
        ]

        def ob_loads(rt):
            return [
                ld_vec(
                    ybase,
                    ((16 * rt + t) * OK_K + (wave + WAVES * ci) * KCH + 8 * g + 32 * qq)
                    // 2,
                    4,
                    YCM,
                ).bitcast(fx.BFloat16)
                for ci in range(CO)
                for qq in range(4)
            ]

        # row groups of 2 tiles (32 tokens): MFMA -> barrier -> (next group's loads) -> reduce + push,
        # so the XGMI pushes of a group overlap the next group's loads and MFMAs
        NG = NT // 2
        nxt = ob_loads(0) + ob_loads(1)
        for gi in range_constexpr(NG):
            cur = nxt
            for r2 in range_constexpr(2):
                rt = 2 * gi + r2
                accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
                for ci in range_constexpr(CO):
                    for i in range_constexpr(2):
                        for qq in range_constexpr(4):
                            accs[i] = mfma_bf16(
                                oa[i * CO + ci][qq],
                                cur[r2 * (4 * CO) + ci * 4 + qq],
                                accs[i],
                            )
                for i in range_constexpr(2):
                    fx.ptr_store(
                        accs[i], red + (((rt * WAVES + wave) * 2 + i) * 64 + lane) * 4
                    )
            gpu.barrier()
            if const_expr(gi + 1 < NG):
                nxt = ob_loads(2 * gi + 2) + ob_loads(2 * gi + 3)
            if const_expr(BND):
                # the CTA's whole [32 columns x tokens] tile to the owner of block b (b // 32), tagged pairs
                e = tid
                pp = e % 16
                tt32 = e // 16
                rt_ = 2 * gi + tt32 // 16
                t16 = tt32 % 16
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * pp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red_at(red, (rt_ * WAVES + w) * 2 + rr // 16, rr % 16, t16)
                        )
                    v.append(s_)
                pdo = sel_peer(pl, uniform(bid // CPR))
                mb_put_sys(
                    tag,
                    pdo + fx.Int64(play["opb"][0]),
                    ((par * W + rank) * S + 16 * rt_ + t16) * (CPR * 16)
                    + (bid % CPR) * 16
                    + pp,
                    bf16_pair(v[0], v[1]),
                )
            else:
                e = tid
                # owner t % 8: wave w pushes owner w's 2 tokens of both tiles of the group
                rt_ = 2 * gi + (e // 32) % 2
                pp = e % 16
                dw = uniform(wave)
                tt = dw + W * ((e // 16) % 2)
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * pp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red_at(red, (rt_ * WAVES + w) * 2 + rr // 16, rr % 16, tt)
                        )
                    v.append(s_)
                t_ = 16 * rt_ + tt
                # one destination a wave (SGPR peer select): no branch around the store (a branch makes
                # the waitcnt pass drain everything), and the links carry every group in parallel
                idx = (((par * W + rank) * BLOCKS + bid) * TS + t_ // W) * 16 + pp
                w2 = bf16_pair(v[0], v[1])
                bo.buffer_store(
                    w2,
                    rsrc(sel_peer(pl, dw) + fx.Int64(play["op"][0])),
                    idx,
                    cache_modifier=CM_SYS,
                )
        if const_expr(not BND):  # bnd: tagged pairs, no flags
            rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
            gpu.barrier()
            if tid == 0:
                for pb in pl:
                    mb_put_sys(
                        tag,
                        pb + fx.Int64(play["opf"][0]),
                        (par * W + rank) * BLOCKS + bid,
                        fx.Int32(1),
                    )

    @traced
    def stage_ab(
        own_peer,
        pl,
        scratch,
        stream_in,
        s_full,
        nrows,
        alpha_a,
        beta_a,
        w_pffn,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        rank,
        red4,
    ):
        """CTA j < S / W: token t = W j + rank, all 8192 columns (16 a thread): the W o_proj
        partials (rank order) -> o, post_attn_norm (weightless, 1e-8), gated residual (mid stream,
        local), pre_feedforward_layernorm -> the ffn_in row pushed to every rank + flag."""
        tid = fresh(tid)
        par = tag_par(tag)
        j = bid
        t = W * j + rank
        # replicated [rows, 8192] or token-sharded
        srow = (s_full != 0).select(fx.min(t, nrows - 1), j)
        st, al, be, wfs = [], [], [], []
        for h in range_constexpr(2):
            cq = 8 * (tid + THREADS * h)
            wfs.append(ld_vec(w_pffn, cq // 2, 4))
            for h2 in range_constexpr(2):
                v4 = ld_vec(stream_in, srow * HID + cq + 4 * h2, 4)
                a4 = ld_vec(alpha_a, cq + 4 * h2, 4)
                b4 = ld_vec(beta_a, cq + 4 * h2, 4)
                for d in range_constexpr(4):
                    st.append(v4[d].bitcast(fx.Float32))
                    al.append(a4[d].bitcast(fx.Float32))
                    be.append(b4[d].bitcast(fx.Float32))
        specs = [
            (own_peer + fx.Int64(play["opf"][0]), par * W * BLOCKS + NFQ * tid + q, 1)
            for q in range(NFQ)
        ]
        mb_poll(tag, health, specs, REGION_IDS["opf"], cm=CM_SYS)
        gpu.barrier()
        a = [fx.Float32(0.0) for _ in range(16)]
        for src in range_constexpr(W):
            for h in range_constexpr(2):
                # 8-column chunk: columns 8q .. 8q + 8 (block q / 4)
                q = tid + THREADS * h
                wv = ld_vec(
                    own_peer + fx.Int64(play["op"][0]),
                    (((par * W + src) * BLOCKS + q // 4) * TS + j) * 16 + 4 * (q % 4),
                    4,
                    XCM,
                )
                for d in range_constexpr(4):
                    k = 8 * h + 2 * d
                    if const_expr(src == 0):
                        a[k] = bf_lo(wv[d])
                        a[k + 1] = bf_hi(wv[d])
                    else:
                        a[k] = a[k] + bf_lo(wv[d])
                        a[k + 1] = a[k + 1] + bf_hi(wv[d])
        o = [bf16_round(x) for x in a]
        ss = fx.Float32(0.0)
        for k in range_constexpr(16):
            ss = ss + o[k] * o[k]
        r1 = hw_rsq(
            block_sum(ss, lane, wave, red4, 0) * fx.Float32(1.0 / HID)
            + fx.Float32(POST_EPS)
        )
        sv = [al[k] * st[k] + be[k] * bf16_round(o[k] * r1) for k in range(16)]
        for h in range_constexpr(2):
            cq = 8 * (tid + THREADS * h)
            for h2 in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [sv[8 * h + 4 * h2 + d].bitcast(fx.Int32) for d in range(4)],
                        fx.Int32,
                    ),
                    rsrc(scratch + fx.Int64(lay["smid"][0])),
                    j * HID + cq + 4 * h2,
                    cache_modifier=CM_DEV,
                )
        ss2 = fx.Float32(0.0)
        for k in range_constexpr(16):
            ss2 = ss2 + sv[k] * sv[k]
        r2 = hw_rsq(
            block_sum(ss2, lane, wave, red4, 1) * fx.Float32(1.0 / HID)
            + fx.Float32(PRE_EPS)
        )
        for h in range_constexpr(2):
            cq = 8 * (tid + THREADS * h)
            wf = wfs[h]
            xw = [
                bf16_pair(
                    (sv[8 * h + 2 * d] * r2) * bf_lo(wf[d]),
                    (sv[8 * h + 2 * d + 1] * r2) * bf_hi(wf[d]),
                )
                for d in range(4)
            ]
            xv = fx.Vector.from_elements(xw, fx.Int32)
            for pb in pl:
                bo.buffer_store(
                    xv,
                    rsrc(pb + fx.Int64(play["xin"][0])),
                    (par * S + t) * (HID // 2) + cq // 2,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            for pb in pl:
                mb_put_sys(tag, pb + fx.Int64(play["xf"][0]), par * S + t, fx.Int32(1))

    def a_loads(w_r, w_pre, bid, tid, lane, wave):
        """CTA b = (group rb = b / KQ, K part kq = b % KQ): router expert rb of this rank (fp32,
        KPT K a thread) and pre_expert_proj rows 16 rb .. (CA chunk tiles a wave)."""
        kq = bid % KQ
        rb = bid // KQ
        wr = ld_vec(w_r, rb * HID + kq * QH + KPT * tid, KPT)
        wp = []
        for ci in range_constexpr(CA):
            c = wave + WAVES * ci
            wp.append(bf_tile_load(w_pre, rb * 16, HID, kq * QH + c * KCH, lane))
        return wr, wp

    @traced
    def stage_a(
        wr,
        wp,
        own_peer,
        pl,
        scratch,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        rank,
        red,
        tok,
        own,
        xl,
    ):
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        kq = bid % KQ
        rb = bid // KQ
        # ffn_in is local; the 32 CTAs of this K part wrote its columns
        if const_expr(BND):
            if tid < 32:
                fa = (tid // 4) * 32 + 4 * kq + tid % 4
                mb_poll(
                    tag,
                    health,
                    [(scratch + fx.Int64(lay["xinbf"][0]), fa, 1)],
                    REGION_IDS["xinbf"],
                )
        else:
            if tid < S:
                mb_poll(
                    tag,
                    health,
                    [(own_peer + fx.Int64(play["xf"][0]), par * S + tid, 1)],
                    REGION_IDS["xf"],
                    cm=CM_SYS,
                )
        gpu.barrier()
        if const_expr(BND):
            xb = scratch + fx.Int64(lay["xinb"][0])
            xpar = 0
        else:
            xb = own_peer + fx.Int64(play["xin"][0])
            xpar = par
        wf = [wr[d].bitcast(fx.Float32) for d in range(KPT)]
        wa = [
            [bf_mma_operand(wp[ci], qq, lane).bitcast(fx.BFloat16) for qq in range(4)]
            for ci in range(CA)
        ]
        QV = QH // 8  # 16-B vectors a token row of the x tile
        NXQ = 16 * QV // THREADS  # of them a thread

        def ax_loads(rt):
            return [
                ld_vec(
                    xb,
                    (xpar * S + 16 * rt + (tid + THREADS * q) // QV) * (HID // 2)
                    + (kq * QH + 8 * ((tid + THREADS * q) % QV)) // 2,
                    4,
                    ACM,
                )
                for q in range(NXQ)
            ]

        xnext = ax_loads(0)
        for rt in range_constexpr(NT):
            # the tile's x slice (16 tokens x QH) -> LDS, NXQ x 16 B a thread
            xs = xnext
            for q in range_constexpr(NXQ):
                e = tid + THREADS * q
                fx.ptr_store(xs[q], xl + ((e // QV) * XAROW + 8 * (e % QV)) // 2)
            gpu.barrier()
            if const_expr(rt + 1 < NT):
                xnext = ax_loads(rt + 1)
            # router: thread's 4 K of every token of the tile; all LDS reads first, then the
            # 16 wave reductions interleaved (ILP)
            xws = [fx.ptr_load(xl + (t16 * XAROW + 2 * tid) // 2) for t16 in range(16)]
            bvs = []
            for ci in range_constexpr(CA):
                c = wave + WAVES * ci
                for qq in range_constexpr(4):
                    kk = c * KCH + 8 * g + 32 * qq
                    bvs.append(
                        fx.Vector(
                            fx.ptr_load(
                                xl + (t * XAROW + kk) // 2,
                                result_type=fx.Vector.make_type(4, fx.Int32),
                            )
                        ).bitcast(fx.BFloat16)
                    )
            prs = [bf_lo(xw) * wf[0] + bf_hi(xw) * wf[1] for xw in xws]
            for off in (32, 16, 8, 4, 2, 1):
                prs = [xred(v, off, _add) for v in prs]
            if lane < 16:
                v_ = prs[0]
                for t16 in range_constexpr(1, 16):
                    v_ = (lane == t16).select(prs[t16], v_)
                fx.ptr_store(v_, tok + (wave * 16 + lane))
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for ci in range_constexpr(CA):
                for qq in range_constexpr(4):
                    acc = mfma_bf16(wa[ci][qq], bvs[ci * 4 + qq], acc)
            fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            # branch-free stores (dropped out-of-range offsets): no waitcnt drain at a join
            t16 = tid % 16
            v = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                v = v + fx.ptr_load(tok + (w * 16 + t16))
            if tid < 16:
                # logits of the CTA's expert, pushed after the loop
                fx.ptr_store(v, tok + (WAVES * 16 + 8 + rt * 16 + t16))
            rp = (tid // 16) % 8
            tt = tid % 16
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                s_ = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    s_ = s_ + fx.ptr_load(red_at(red, w, rr, tt))
                mine = (rr // OWN) == kq
                if (tid < 128) & mine:
                    fx.ptr_store(s_, own + ((rt * OWN + rr - OWN * kq) * 16 + tt))
                mb_put_if(
                    (tid < 128) & (~mine),
                    tag,
                    scratch + fx.Int64(lay["px"][0]),
                    ((rb * KQ + kq) * 16 + rr) * S + 16 * rt + tt,
                    s_,
                )
        gpu.barrier()
        if tid < S:
            lv = fx.ptr_load(tok + (WAVES * 16 + 8 + tid))
            for pb in pl:
                mb_put_sys(
                    tag,
                    pb + fx.Int64(play["logit"][0]),
                    ((par * KQ + kq) * E + rank * EL + rb) * S + tid,
                    lv,
                )
        for it_ in range_constexpr((NRP * 16 * NT + THREADS - 1) // THREADS):
            e = tid + THREADS * it_
            if e < NRP * 16 * NT:
                rt_ = e // (NRP * 16)
                rp = e % NRP
                tt = (e // NRP) % 16
                specs, oks = [], []
                for src in range_constexpr(KQ):
                    for h2 in range_constexpr(2):
                        specs.append(
                            (
                                scratch + fx.Int64(lay["px"][0]),
                                ((rb * KQ + src) * 16 + OWN * kq + 2 * rp + h2) * S
                                + 16 * rt_
                                + tt,
                                1,
                            )
                        )
                        oks.append(fx.Int32(src) != kq)
                got = mb_poll(tag, health, specs, REGION_IDS["px"], oks=oks)
                tot = []
                for h2 in range_constexpr(2):
                    mine = fx.ptr_load(own + ((rt_ * OWN + 2 * rp + h2) * 16 + tt))
                    s_ = fx.Float32(0.0)
                    for src in range_constexpr(KQ):
                        s_ = s_ + (fx.Int32(src) == kq).select(
                            mine, got[2 * src + h2][0].bitcast(fx.Float32)
                        )
                    tot.append(s_)
                hw = bf16_pair(tot[0], tot[1])
                hidx = (
                    (((par * W + rank) * EL + rb) * KQ + kq) * S + 16 * rt_ + tt
                ) * NRP + rp
                for pb in pl:
                    bo.buffer_store(
                        hw,
                        rsrc(pb + fx.Int64(play["hraw"][0])),
                        hidx,
                        cache_modifier=CM_SYS,
                    )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            for pb in pl:
                mb_put_sys(
                    tag,
                    pb + fx.Int64(play["hf"][0]),
                    (par * W + rank) * BLOCKS + bid,
                    fx.Int32(1),
                )

    @traced
    def stage_route_h(
        own_peer,
        bias,
        w_pren,
        h_out,
        ids_out,
        sc_out,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        rt,
        tok,
    ):
        """CTA t < S: token t's top-8 (sigmoid + bias, radix top-8, renorm) -> ids / weights,
        then h = pre_expert_norm(h_raw) of token t."""
        tid = fresh(tid)
        par = tag_par(tag)
        t = bid
        if wave == 0:
            bias_v = [ld_f32(bias, lane + 64 * i) for i in range(4)]
            specs = [
                (
                    own_peer + fx.Int64(play["logit"][0]),
                    ((par * KQ + kq) * E + lane + 64 * i) * S + t,
                    1,
                )
                for i in range(4)
                for kq in range(KQ)
            ]
            got = mb_poll(tag, health, specs, REGION_IDS["logit"], cm=CM_SYS)
            lgs = []
            for i in range_constexpr(4):
                v = got[i * KQ][0].bitcast(fx.Float32)
                for kq in range_constexpr(1, KQ):
                    v = v + got[i * KQ + kq][0].bitcast(fx.Float32)
                lgs.append(v.bitcast(fx.Int32))
            route_token([0], bias_v, [lgs], tid, lane, rt)
            if lane < TOPK:
                bo.buffer_store(
                    fx.ptr_load(rt["rid"] + lane), rsrc(ids_out), t * TOPK + lane
                )
                bo.buffer_store(
                    fx.ptr_load(rt["rw"] + lane), rsrc(sc_out), t * TOPK + lane
                )
        # h_raw of token t from every rank (4 x 256 producer CTAs)
        specs = [
            (
                own_peer + fx.Int64(play["hf"][0]),
                (par * W + (NFQ * tid + q) // BLOCKS) * BLOCKS
                + (NFQ * tid + q) % BLOCKS,
                1,
            )
            for q in range(NFQ)
        ]
        mb_poll(tag, health, specs, REGION_IDS["hf"], cm=CM_SYS)
        gpu.barrier()
        kk = tid * 8
        hsrc = kk // (DM // W)
        hloc = kk % (DM // W)
        hb0 = (
            (((par * W + hsrc) * EL + hloc // 16) * KQ + (hloc % 16) // OWN) * S + t
        ) * NRP
        # 8 columns = 4 owner CTAs (K parts) of 2 rows each: one pair a part
        hv = fx.Vector.from_elements(
            [
                ld_i32(own_peer + fx.Int64(play["hraw"][0]), hb0 + o * S, XCM)
                for o in range(4)
            ],
            fx.Int32,
        )
        wv = ld_vec(w_pren, kk // 2, 4)
        ss = fx.Float32(0.0)
        for d in range_constexpr(4):
            ss = ss + bf_lo(hv[d]) * bf_lo(hv[d]) + bf_hi(hv[d]) * bf_hi(hv[d])
        ss = wave_sum(ss)
        if lane == 0:
            fx.ptr_store(ss, tok + wave)
        gpu.barrier()
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(tok + w)
        rs = hw_rsq(tot * fx.Float32(1.0 / DM) + fx.Float32(PRE_EPS))
        outw = []
        for d in range_constexpr(4):
            outw.append(
                bf16_pair(
                    bf_lo(hv[d]) * rs * bf_lo(wv[d]), bf_hi(hv[d]) * rs * bf_hi(wv[d])
                )
            )
        bo.buffer_store(
            fx.Vector.from_elements(outw, fx.Int32), rsrc(h_out), (t * DM + kk) // 2
        )

    # ------------------------------------------------------------ bnd: column-group boundaries
    # CTA b = token group tg = b / 32 (tokens 4 tg .. 4 tg + 4) x column group cb = b % 32 (columns 256 cb .. + 256);
    # thread = (token tq = tid / 128, column pair pp = tid % 128). The token norms reduce in the CTA (2 waves a
    # token), then over the 32 CTAs of the token group in ONE hop: every CTA of the group reads the group's 32 x 4 x 4
    # partials (fixed order cb = 0 .. 31) instead of a reducer CTA and a second hop.
    @traced
    def group_rstd2(
        scratch,
        region,
        tag,
        health,
        tid,
        lane,
        wave,
        bid,
        q1,
        q2,
        q3,
        q4,
        eps1,
        eps2,
        pr,
    ):
        """-> (r1, r2) of this thread's token from the 4 norm partials of every CTA of the token group."""
        tg = bid // 32
        base = scratch + fx.Int64(lay[region][0])
        qs = [wave_sum(q) for q in (q1, q2, q3, q4)]
        if lane == 0:
            for k in range_constexpr(4):
                fx.ptr_store(qs[k], pr + (512 + wave * 4 + k))
        gpu.barrier()
        # token tq = tid / 4, partial k = tid % 4: the token's 2 waves in order
        if tid < 16:
            v = fx.ptr_load(pr + (512 + (2 * (tid // 4)) * 4 + tid % 4)) + fx.ptr_load(
                pr + (512 + (2 * (tid // 4) + 1) * 4 + tid % 4)
            )
            mb_put(tag, base, bid * 16 + tid, v)
        src = tid // 16  # (column group, token, partial) of the group, one a thread
        got = mb_poll(
            tag,
            health,
            [(base, (tg * 32 + src) * 16 + tid % 16, 1)],
            REGION_IDS[region],
        )
        fx.ptr_store(got[0][0].bitcast(fx.Float32), pr + tid)
        gpu.barrier()
        if tid < 4:  # token tid: the 32 groups in order, then the two norms
            p = []
            for k in range_constexpr(4):
                a_ = fx.ptr_load(pr + (tid * 4 + k))
                for c_ in range_constexpr(1, 32):
                    a_ = a_ + fx.ptr_load(pr + (c_ * 16 + tid * 4 + k))
                p.append(a_)
            r1 = hw_rsq(p[0] * fx.Float32(1.0 / HID) + fx.Float32(eps1))
            q = p[1] + fx.Float32(2.0) * r1 * p[2] + r1 * r1 * p[3]
            r2 = hw_rsq(q * fx.Float32(1.0 / HID) + fx.Float32(eps2))
            fx.ptr_store(r1, pr + (512 + 32 + tid))
            fx.ptr_store(r2, pr + (512 + 36 + tid))
        gpu.barrier()
        tq = tid // 128
        return fx.ptr_load(pr + (512 + 32 + tq)), fx.ptr_load(pr + (512 + 36 + tq))

    @traced
    def stage_ab_c(
        own_peer,
        pl,
        scratch,
        stream_in,
        nrows,
        alpha_a,
        beta_a,
        w_pffn,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        rank,
        pr,
    ):
        """The attention boundary with the column-group mapping: the block owners (b / 32 == rank, block
        b = 32 columns, all tokens) reduce the o_proj partials and all-gather the result; CTA (tg, cb) then
        does the boundary of tokens 4 tg .. x columns 256 cb .."""
        tid = fresh(tid)
        par = tag_par(tag)
        if (
            bid // CPR
        ) == rank:  # the block owner: 8 partials in rank order -> bf16 -> every rank
            pp_ = tid % 16
            tt_ = tid // 16
            specs = [
                (
                    own_peer + fx.Int64(play["opb"][0]),
                    ((par * W + src) * S + tt_) * (CPR * 16) + (bid % CPR) * 16 + pp_,
                    1,
                )
                for src in range(W)
            ]
            got = mb_poll(tag, health, specs, REGION_IDS["opb"], cm=CM_SYS)
            a0 = bf_lo(got[0][0])
            a1 = bf_hi(got[0][0])
            for src in range_constexpr(1, W):
                a0 = a0 + bf_lo(got[src][0])
                a1 = a1 + bf_hi(got[src][0])
            for pb in pl:
                mb_put_sys(
                    tag,
                    pb + fx.Int64(play["orb"][0]),
                    (par * S + tt_) * (HID // 2) + bid * 16 + pp_,
                    bf16_pair(a0, a1),
                )
        tg = bid // 32
        cb = bid % 32
        tt = 4 * tg + tid // 128
        pp = tid % 128
        c = cb * 256 + 2 * pp
        st = ld_vec(stream_in, fx.min(tt, nrows - 1) * HID + c, 2)
        al = ld_vec(alpha_a, c, 2)
        be = ld_vec(beta_a, c, 2)
        wf = ld_i32(w_pffn, c // 2)
        got = mb_poll(
            tag,
            health,
            [
                (
                    own_peer + fx.Int64(play["orb"][0]),
                    (par * S + tt) * (HID // 2) + c // 2,
                    1,
                )
            ],
            REGION_IDS["orb"],
            cm=CM_SYS,
        )
        o0 = bf_lo(got[0][0])
        o1 = bf_hi(got[0][0])
        a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
        a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
        b0 = be[0].bitcast(fx.Float32) * o0
        b1 = be[1].bitcast(fx.Float32) * o1
        r1, r2 = group_rstd2(
            scratch,
            "gpa",
            tag,
            health,
            tid,
            lane,
            wave,
            bid,
            o0 * o0 + o1 * o1,
            a0 * a0 + a1 * a1,
            a0 * b0 + a1 * b1,
            b0 * b0 + b1 * b1,
            POST_EPS,
            PRE_EPS,
            pr,
        )
        n0 = bf16_round(o0 * r1)
        n1 = bf16_round(o1 * r1)
        s0 = a0 + be[0].bitcast(fx.Float32) * n0
        s1 = a1 + be[1].bitcast(fx.Float32) * n1
        bo.buffer_store(
            fx.Vector.from_elements(
                [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
            ),
            rsrc(scratch + fx.Int64(lay["smidb"][0])),
            tt * HID + c,
            cache_modifier=CM_DEV,
        )
        bo.buffer_store(
            bf16_pair((s0 * r2) * bf_lo(wf), (s1 * r2) * bf_hi(wf)),
            rsrc(scratch + fx.Int64(lay["xinb"][0])),
            (tt * HID + c) // 2,
            cache_modifier=CM_DEV,
        )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xinbf"][0]), bid, fx.Int32(1))

    @traced
    def stage_g_c(
        own_peer,
        scratch,
        w_pff,
        alpha,
        beta,
        w_in,
        stream_out,
        so_stride,
        so_row,
        sfull_out,
        gather,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        pr,
    ):
        """The FFN boundary with the column-group mapping (CTA (tg, cb): tokens 4 tg .. x columns
        256 cb ..)."""
        tid = fresh(tid)
        par = tag_par(tag)
        tg = bid // 32
        cb = bid % 32
        tt = 4 * tg + tid // 128
        pp = tid % 128
        c = cb * 256 + 2 * pp
        st = ld_vec(scratch + fx.Int64(lay["smidb"][0]), tt * HID + c, 2, CM_DEV)
        wpf = ld_i32(w_pff, c // 2)
        al = ld_vec(alpha, c, 2)
        be = ld_vec(beta, c, 2)
        wi = ld_i32(w_in, c // 2)
        got = mb_poll(
            tag,
            health,
            [
                (
                    own_peer + fx.Int64(play["yb"][0]),
                    (par * S + tt) * (HID // 2) + c // 2,
                    1,
                )
            ],
            REGION_IDS["yb"],
            cm=CM_SYS,
        )
        yw = got[0][0]
        z0 = bf16_round(bf_lo(wpf) * bf_lo(yw))
        z1 = bf16_round(bf_hi(wpf) * bf_hi(yw))
        a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
        a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
        b0 = be[0].bitcast(fx.Float32) * z0
        b1 = be[1].bitcast(fx.Float32) * z1
        r1, r2 = group_rstd2(
            scratch,
            "gpg",
            tag,
            health,
            tid,
            lane,
            wave,
            bid,
            z0 * z0 + z1 * z1,
            a0 * a0 + a1 * a1,
            a0 * b0 + a1 * b1,
            b0 * b0 + b1 * b1,
            POST_EPS,
            PRE_EPS,
            pr,
        )
        m0 = bf16_round(z0 * r1)
        m1 = bf16_round(z1 * r1)
        s0 = a0 + be[0].bitcast(fx.Float32) * m0
        s1 = a1 + be[1].bitcast(fx.Float32) * m1
        sw = fx.Vector.from_elements(
            [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
        )
        bo.buffer_store(sw, rsrc(stream_out), tt * so_stride + so_row + c)
        if gather != 0:
            bo.buffer_store(sw, rsrc(sfull_out), tt * HID + c)
        xw = bf16_pair((s0 * r2) * bf_lo(wi), (s1 * r2) * bf_hi(wi))
        bo.buffer_store(
            xw,
            rsrc(scratch + fx.Int64(lay["xnb"][0])),
            (tt * HID + c) // 2,
            cache_modifier=CM_DEV,
        )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xnbf"][0]), bid, fx.Int32(1))

    @traced
    def ha_body(
        attn_o,
        attn_g,
        stream_in,
        w_o,
        alpha_a,
        beta_a,
        w_pffn,
        w_r,
        bias,
        w_pre,
        w_pren,
        h_out,
        ids_out,
        sc_out,
        scratch,
        epoch,
        health,
        tl,
        peers,
        layer,
        rank,
        g_stride,
        s_full,
        nrows,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        alloc = fx.SharedAllocator()
        misc = alloc.allocate(Misc).peek()
        rl = alloc.allocate(RouteLds).peek()
        sl = alloc.allocate(HaLds)
        so, sa = sl.o_.peek(), sl.a_.peek()
        rt = {"rid": rl.rid.ptr, "rw": rl.rw.ptr, "sel": rl.sel.ptr}
        pl = list(peers)
        own_peer = sel_peer(pl, rank)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        ao = None
        if const_expr(FA is None):
            o_ops = o_loads(w_o, bid, lane, wave)
        else:
            # attn_o is the fragment's descriptor; ng polls the merged rows (tagged pairs)
            attn_o, ao_t = FA.run(
                attn_o, tag, health, tid, bid, lane, wave, sl.fa_.peek()
            )
            ao = (ao_t, health)
            o_ops = o_loads(w_o, bid, lane, wave)
        stage_ng(attn_o, attn_g, g_stride, scratch, tag, tid, bid, nrows, ao)
        stage_o(
            o_ops,
            scratch,
            pl,
            tag,
            health,
            tid,
            bid,
            lane,
            wave,
            rank,
            so.red.ptr,
        )
        wr, wp = a_loads(w_r, w_pre, bid, tid, lane, wave)
        if const_expr(BND):  # all 256 CTAs, column groups
            stage_ab_c(
                own_peer,
                pl,
                scratch,
                stream_in,
                nrows,
                alpha_a,
                beta_a,
                w_pffn,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                rank,
                misc.pr.ptr,
            )
        else:
            if bid < TS:
                stage_ab(
                    own_peer,
                    pl,
                    scratch,
                    stream_in,
                    s_full,
                    nrows,
                    alpha_a,
                    beta_a,
                    w_pffn,
                    tag,
                    health,
                    tid,
                    bid,
                    lane,
                    wave,
                    rank,
                    misc.red4.ptr,
                )
        stage_a(
            wr,
            wp,
            own_peer,
            pl,
            scratch,
            tag,
            health,
            tid,
            bid,
            lane,
            wave,
            rank,
            sa.red.ptr,
            sa.tok.ptr,
            sa.own.ptr,
            sa.x.ptr,
        )
        if bid < S:
            stage_route_h(
                own_peer,
                bias,
                w_pren,
                h_out,
                ids_out,
                sc_out,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                rt,
                sa.tok.ptr,
            )

    # ============================================================ HB
    def pp_loads(w_pp, bid, lane, wave):
        """post_expert_proj tile b / KSP (16 of this rank's QH rows), K part b % KSP."""
        kh = bid % KSP
        ops = []
        for ci in range_constexpr(CP):
            c = wave + WAVES * ci
            ops.append(
                bf_tile_load(
                    w_pp, (bid // KSP) * 16, DM, kh * (DM // KSP) + c * KCH, lane
                )
            )
        return ops

    @traced
    def stage_ar(own_peer, pl, tag, health, tid, bid, rank):
        """CTA b of the column-block owner (b / CPR == rank): the W ranks' partials of columns
        16b .., rank order (fp32) -> bf16 m -> every rank's MOEM (all-gather, plain) + flag."""
        tid = fresh(tid)
        par = tag_par(tag)
        if tid < W:
            mb_poll(
                tag,
                health,
                [
                    (
                        own_peer + fx.Int64(play["mf"][0]),
                        (par * W + tid) * BLOCKS + bid,
                        1,
                    )
                ],
                REGION_IDS["mf"],
                cm=CM_SYS,
            )
        gpu.barrier()
        for it_ in range_constexpr((8 * S + THREADS - 1) // THREADS):
            e0 = tid + THREADS * it_
            ok = e0 < 8 * S
            e = fx.min(e0, 8 * S - 1)
            t = e // 8
            p = e % 8
            a0 = fx.Float32(0.0)
            a1 = fx.Float32(0.0)
            for src in range_constexpr(W):
                w_ = ld_i32(
                    own_peer + fx.Int64(play["moep"][0]),
                    (((par * W + src) * BLOCKS + bid) * S + t) * 8 + p,
                    XCM,
                )
                if const_expr(src == 0):
                    a0 = bf_lo(w_)
                    a1 = bf_hi(w_)
                else:
                    a0 = a0 + bf_lo(w_)
                    a1 = a1 + bf_hi(w_)
            mw = bf16_pair(a0, a1)
            for pb in pl:
                st_if(
                    ok,
                    mw,
                    pb + fx.Int64(play["moem"][0]),
                    ((par * BLOCKS + bid) * S + t) * 8 + p,
                    CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            for pb in pl:
                mb_put_sys(
                    tag, pb + fx.Int64(play["mmf"][0]), par * BLOCKS + bid, fx.Int32(1)
                )

    @traced
    def stage_post(ppo, pl, scratch, tag, health, tid, bid, lane, wave, rank, red):
        """post_expert_proj tile b / KSP over K part b % KSP for all S tokens; the K parts are
        summed by the kh = 0 CTA -> y (this rank's QH hidden columns) + flag."""
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        kh = bid % KSP
        tile = bid // KSP
        own_peer = sel_peer(pl, rank)
        if tid < BLOCKS // KSP:
            mb_poll(
                tag,
                health,
                [
                    (
                        own_peer + fx.Int64(play["mmf"][0]),
                        par * BLOCKS + kh * (BLOCKS // KSP) + tid,
                        1,
                    )
                ],
                REGION_IDS["mmf"],
                cm=CM_SYS,
            )
        gpu.barrier()
        mb_ = own_peer + fx.Int64(play["moem"][0])
        pa = [
            [bf_mma_operand(ppo[ci], qq, lane).bitcast(fx.BFloat16) for qq in range(4)]
            for ci in range(CP)
        ]

        def pb_loads(rt):
            # MOEM block layout [par][b][t][16 columns]: columns kk .. kk+8 = block kk / 16, half (kk % 16) / 8
            return [
                ld_vec(
                    mb_,
                    (
                        (
                            par * BLOCKS
                            + (
                                kh * (DM // KSP)
                                + (wave + WAVES * ci) * KCH
                                + 8 * g
                                + 32 * qq
                            )
                            // 16
                        )
                        * S
                        + 16 * rt
                        + t
                    )
                    * 8
                    + 4 * (g % 2),
                    4,
                    XCM,
                ).bitcast(fx.BFloat16)
                for ci in range(CP)
                for qq in range(4)
            ]

        nxt = pb_loads(0)
        for rt in range_constexpr(NT):
            cur = nxt
            if const_expr(rt + 1 < NT):
                nxt = pb_loads(rt + 1)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for ci in range_constexpr(CP):
                for qq in range_constexpr(4):
                    acc = mfma_bf16(pa[ci][qq], cur[ci * 4 + qq], acc)
            fx.ptr_store(acc, red + ((rt * WAVES + wave) * 64 + lane) * 4)
        gpu.barrier()
        if const_expr(BND):
            # bnd: y of the tile (16 columns of this rank x every token) to every rank, tagged pairs at the global
            # column rank * QH + 16 tile + 2 rp; thread = (row tile rt_, token tt, row pair rp)
            e0 = tid
            if e0 < NT * 128:
                rt_ = e0 // 128
                tt = (e0 // 8) % 16
                rp = e0 % 8
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * rp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(red_at(red, rt_ * WAVES + w, rr, tt))
                    v.append(s_)
                if kh != 0:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            scratch + fx.Int64(lay["py"][0]),
                            (((kh - 1) * NTILES + tile) * 16 + 2 * rp + h2) * S
                            + 16 * rt_
                            + tt,
                            v[h2],
                        )
                else:
                    got = mb_poll(
                        tag,
                        health,
                        [
                            (
                                scratch + fx.Int64(lay["py"][0]),
                                (((q - 1) * NTILES + tile) * 16 + 2 * rp + h2) * S
                                + 16 * rt_
                                + tt,
                                1,
                            )
                            for q in range(1, KSP)
                            for h2 in range(2)
                        ],
                        REGION_IDS["py"],
                    )
                    y0 = v[0]
                    y1 = v[1]
                    for q in range_constexpr(KSP - 1):
                        y0 = y0 + got[2 * q][0].bitcast(fx.Float32)
                        y1 = y1 + got[2 * q + 1][0].bitcast(fx.Float32)
                    w2 = bf16_pair(y0, y1)
                    for pb in pl:
                        mb_put_sys(
                            tag,
                            pb + fx.Int64(play["yb"][0]),
                            (par * S + 16 * rt_ + tt) * (HID // 2)
                            + (rank * QH + tile * 16 + 2 * rp) // 2,
                            w2,
                        )
        else:
            for it_ in range_constexpr((NT + 3) // 4):
                e = tid
                # 4 tiles a round; wave w: owner w, tiles 4 it + k (k < 4), the owner's 2 tokens of each
                dw = uniform(wave)
                k_ = (e // 16) % 4
                rt_ = 4 * it_ + k_
                rp = e % 8
                tt = dw + W * ((e // 8) % 2)
                if rt_ < NT:
                    v = []
                    for h2 in range_constexpr(2):
                        rr = 2 * rp + h2
                        s_ = fx.Float32(0.0)
                        for w in range_constexpr(WAVES):
                            s_ = s_ + fx.ptr_load(red_at(red, rt_ * WAVES + w, rr, tt))
                        v.append(s_)
                    # K parts 1 .. KSP-1 put their partial (slot kh - 1); part 0 sums the parts in order
                    if kh != 0:
                        for h2 in range_constexpr(2):
                            mb_put(
                                tag,
                                scratch + fx.Int64(lay["py"][0]),
                                (((kh - 1) * NTILES + tile) * 16 + 2 * rp + h2) * S
                                + 16 * rt_
                                + tt,
                                v[h2],
                            )
                    else:
                        got = mb_poll(
                            tag,
                            health,
                            [
                                (
                                    scratch + fx.Int64(lay["py"][0]),
                                    (((q - 1) * NTILES + tile) * 16 + 2 * rp + h2) * S
                                    + 16 * rt_
                                    + tt,
                                    1,
                                )
                                for q in range(1, KSP)
                                for h2 in range(2)
                            ],
                            REGION_IDS["py"],
                        )
                        y0 = v[0]
                        y1 = v[1]
                        for q in range_constexpr(KSP - 1):
                            y0 = y0 + got[2 * q][0].bitcast(fx.Float32)
                            y1 = y1 + got[2 * q + 1][0].bitcast(fx.Float32)
                        t_ = 16 * rt_ + tt
                        idx = (
                            ((par * W + rank) * NTILES + tile) * TS + t_ // W
                        ) * 8 + rp
                        w2 = bf16_pair(y0, y1)
                        bo.buffer_store(
                            w2,
                            rsrc(sel_peer(pl, dw) + fx.Int64(play["yt"][0])),
                            idx,
                            cache_modifier=CM_SYS,
                        )
            rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
            gpu.barrier()
            if (tid == 0) & (kh == 0):
                for pb in pl:
                    mb_put_sys(
                        tag,
                        pb + fx.Int64(play["ytf"][0]),
                        (par * W + rank) * NTILES + tile,
                        fx.Int32(1),
                    )

    @traced
    def stage_g(
        own_peer,
        pl,
        scratch,
        w_pff,
        alpha,
        beta,
        w_in,
        stream_out,
        so_stride,
        so_row,
        gather,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        rank,
        red4,
    ):
        """CTA j < S / W: token t = W j + rank, all 8192 columns: post_feedforward_layernorm
        (input-scaled, eps 1e-8), gated residual on the mid stream, next input_layernorm ->
        stream row (token-sharded; ``gather``: to every rank too), xn row to every rank + flag."""
        tid = fresh(tid)
        par = tag_par(tag)
        j = bid
        t = W * j + rank
        st, al, be, wps, wis = [], [], [], [], []
        for h in range_constexpr(2):
            cq = 8 * (tid + THREADS * h)
            wps.append(ld_vec(w_pff, cq // 2, 4))
            wis.append(ld_vec(w_in, cq // 2, 4))
            for h2 in range_constexpr(2):
                v4 = ld_vec(
                    scratch + fx.Int64(lay["smid"][0]),
                    j * HID + cq + 4 * h2,
                    4,
                    CM_DEV,
                )
                a4 = ld_vec(alpha, cq + 4 * h2, 4)
                b4 = ld_vec(beta, cq + 4 * h2, 4)
                for d in range_constexpr(4):
                    st.append(v4[d].bitcast(fx.Float32))
                    al.append(a4[d].bitcast(fx.Float32))
                    be.append(b4[d].bitcast(fx.Float32))
        mb_poll(
            tag,
            health,
            [(own_peer + fx.Int64(play["ytf"][0]), par * W * NTILES + tid, 1)],
            REGION_IDS["ytf"],
            cm=CM_SYS,
        )
        gpu.barrier()
        z = []
        for h in range_constexpr(2):
            q = tid + THREADS * h
            ysrc = (8 * q) // QH
            yloc = (8 * q) % QH
            yv = ld_vec(
                own_peer + fx.Int64(play["yt"][0]),
                (((par * W + ysrc) * NTILES + yloc // 16) * TS + j) * 8 + 4 * (q % 2),
                4,
                XCM,
            )
            wv = wps[h]
            for d in range_constexpr(4):
                z.append(bf16_round(bf_lo(wv[d]) * bf_lo(yv[d])))
                z.append(bf16_round(bf_hi(wv[d]) * bf_hi(yv[d])))
        ss = fx.Float32(0.0)
        for k in range_constexpr(16):
            ss = ss + z[k] * z[k]
        r1 = hw_rsq(
            block_sum(ss, lane, wave, red4, 0) * fx.Float32(1.0 / HID)
            + fx.Float32(POST_EPS)
        )
        sv = [al[k] * st[k] + be[k] * bf16_round(z[k] * r1) for k in range(16)]
        for h in range_constexpr(2):
            cq = 8 * (tid + THREADS * h)
            for h2 in range_constexpr(2):
                swv = fx.Vector.from_elements(
                    [sv[8 * h + 4 * h2 + d].bitcast(fx.Int32) for d in range(4)],
                    fx.Int32,
                )
                bo.buffer_store(
                    swv, rsrc(stream_out), j * so_stride + so_row + cq + 4 * h2
                )
                if gather != 0:
                    for pb in pl:
                        bo.buffer_store(
                            swv,
                            rsrc(pb + fx.Int64(play["sg"][0])),
                            (par * S + t) * HID + cq + 4 * h2,
                            cache_modifier=CM_SYS,
                        )
        ss2 = fx.Float32(0.0)
        for k in range_constexpr(16):
            ss2 = ss2 + sv[k] * sv[k]
        r2 = hw_rsq(
            block_sum(ss2, lane, wave, red4, 1) * fx.Float32(1.0 / HID)
            + fx.Float32(PRE_EPS)
        )
        for h in range_constexpr(2):
            cq = 8 * (tid + THREADS * h)
            wi = wis[h]
            xw = [
                bf16_pair(
                    (sv[8 * h + 2 * d] * r2) * bf_lo(wi[d]),
                    (sv[8 * h + 2 * d + 1] * r2) * bf_hi(wi[d]),
                )
                for d in range(4)
            ]
            xv = fx.Vector.from_elements(xw, fx.Int32)
            for pb in pl:
                bo.buffer_store(
                    xv,
                    rsrc(pb + fx.Int64(play["xn"][0])),
                    (par * S + t) * (HID // 2) + cq // 2,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            for pb in pl:
                mb_put_sys(tag, pb + fx.Int64(play["xnf"][0]), par * S + t, fx.Int32(1))
                if gather != 0:
                    mb_put_sys(
                        tag, pb + fx.Int64(play["sgf"][0]), par * S + t, fx.Int32(1)
                    )

    def q_loads(w_qkvg, bid, lane, wave, n0, n1):
        k8 = bid % 8
        gq = bid // 8
        return [
            bf_tile_load(
                w_qkvg,
                fx.min(NQT * gq + n, NQTILES - 1) * 16,
                HID,
                k8 * (HID // 8) + wave * KCH,
                lane,
            )
            for n in range(n0, n1)
        ]

    @traced
    def stage_qkv(
        q_ops,
        own_peer,
        scratch,
        g_out,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        red,
        own,
    ):
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        k8 = bid % 8
        gq = bid // 8
        # xn is local; the 32 CTAs of this K eighth wrote its columns
        if const_expr(BND):
            if tid < 32:
                fq = (tid // 4) * 32 + 4 * k8 + tid % 4
                mb_poll(
                    tag,
                    health,
                    [(scratch + fx.Int64(lay["xnbf"][0]), fq, 1)],
                    REGION_IDS["xnbf"],
                )
        else:
            if tid < S:
                mb_poll(
                    tag,
                    health,
                    [(own_peer + fx.Int64(play["xnf"][0]), par * S + tid, 1)],
                    REGION_IDS["xnf"],
                    cm=CM_SYS,
                )
        gpu.barrier()
        if const_expr(BND):
            xb = scratch + fx.Int64(lay["xnb"][0])
            xpar = 0
        else:
            xb = own_peer + fx.Int64(play["xn"][0])
            xpar = par
        qa = [
            [bf_mma_operand(q_ops[n], qq, lane).bitcast(fx.BFloat16) for qq in range(4)]
            for n in range(NQT)
        ]

        def qb_loads(rt):
            return [
                ld_vec(
                    xb,
                    (xpar * S + 16 * rt + t) * (HID // 2)
                    + (k8 * (HID // 8) + wave * KCH + 8 * g + 32 * qq) // 2,
                    4,
                    XCM,
                ).bitcast(fx.BFloat16)
                for qq in range(4)
            ]

        if const_expr(QR1):
            # the MFMAs of both row tiles first (red holds both), then one reduction round
            bnext = qb_loads(0)
            for rt in range_constexpr(NT):
                bs = bnext
                if const_expr(rt + 1 < NT):
                    bnext = qb_loads(rt + 1)
                for n in range_constexpr(NQT):
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for qq in range_constexpr(4):
                        acc = mfma_bf16(qa[n][qq], bs[qq], acc)
                    fx.ptr_store(
                        acc, red + (((rt * WAVES + wave) * NQT + n) * 64 + lane) * 4
                    )
            gpu.barrier()
            for it_ in range_constexpr((NT * NQT * 128 + THREADS - 1) // THREADS):
                e0 = tid + THREADS * it_
                okq = e0 < NT * NQT * 128
                e = fx.min(e0, NT * NQT * 128 - 1)
                rt = e // (NQT * 128)
                n = (e // 128) % NQT
                rp = (e // 16) % 8
                tt = e % 16
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * rp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red_at(red, (rt * WAVES + w) * NQT + n, rr, tt)
                        )
                    v.append(s_)
                if okq & (rp == k8):
                    fx.ptr_store(v[0], own + (((rt * NQT + n) * 2) * 16 + tt))
                    fx.ptr_store(v[1], own + (((rt * NQT + n) * 2 + 1) * 16 + tt))
                for h2 in range_constexpr(2):
                    mb_put_if(
                        okq & (rp != k8),
                        tag,
                        scratch + fx.Int64(lay["qx"][0]),
                        (((gq * 8 + k8) * NQT + n) * 16 + 2 * rp + h2) * S
                        + 16 * rt
                        + tt,
                        v[h2],
                    )
        else:
            bnext = qb_loads(0)
            for rt in range_constexpr(NT):
                bs = bnext
                if const_expr(rt + 1 < NT):
                    bnext = qb_loads(rt + 1)
                for n in range_constexpr(NQT):
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for qq in range_constexpr(4):
                        acc = mfma_bf16(qa[n][qq], bs[qq], acc)
                    fx.ptr_store(acc, red + ((wave * NQT + n) * 64 + lane) * 4)
                gpu.barrier()
                for it_ in range_constexpr((NQT * 128 + THREADS - 1) // THREADS):
                    e0 = tid + THREADS * it_
                    okq = e0 < NQT * 128
                    e = fx.min(e0, NQT * 128 - 1)
                    n = e // 128
                    rp = (e // 16) % 8
                    tt = e % 16
                    v = []
                    for h2 in range_constexpr(2):
                        rr = 2 * rp + h2
                        s_ = fx.Float32(0.0)
                        for w in range_constexpr(WAVES):
                            s_ = s_ + fx.ptr_load(red_at(red, w * NQT + n, rr, tt))
                        v.append(s_)
                    if okq & (rp == k8):
                        fx.ptr_store(v[0], own + (((rt * NQT + n) * 2) * 16 + tt))
                        fx.ptr_store(v[1], own + (((rt * NQT + n) * 2 + 1) * 16 + tt))
                    for h2 in range_constexpr(2):
                        mb_put_if(
                            okq & (rp != k8),
                            tag,
                            scratch + fx.Int64(lay["qx"][0]),
                            (((gq * 8 + k8) * NQT + n) * 16 + 2 * rp + h2) * S
                            + 16 * rt
                            + tt,
                            v[h2],
                        )
                gpu.barrier()  # red is rewritten by the next row tile
        gpu.barrier()
        for it_ in range_constexpr((NQT * 16 * NT + THREADS - 1) // THREADS):
            e = tid + THREADS * it_
            if e < NQT * 16 * NT:
                rt_ = e // (NQT * 16)
                n = (e // 16) % NQT
                tt = e % 16
                specs, oks = [], []
                for src in range_constexpr(8):
                    for h2 in range_constexpr(2):
                        specs.append(
                            (
                                scratch + fx.Int64(lay["qx"][0]),
                                (((gq * 8 + src) * NQT + n) * 16 + 2 * k8 + h2) * S
                                + 16 * rt_
                                + tt,
                                1,
                            )
                        )
                        oks.append(fx.Int32(src) != k8)
                got = mb_poll(tag, health, specs, REGION_IDS["qx"], oks=oks)
                tot = []
                for h2 in range_constexpr(2):
                    mine = fx.ptr_load(own + (((rt_ * NQT + n) * 2 + h2) * 16 + tt))
                    s_ = fx.Float32(0.0)
                    for src in range_constexpr(8):
                        s_ = s_ + (fx.Int32(src) == k8).select(
                            mine, got[2 * src + h2][0].bitcast(fx.Float32)
                        )
                    tot.append(s_)
                row = (NQT * gq + n) * 16 + 2 * k8
                w2 = bf16_pair(tot[0], tot[1])
                if row < QKV_ROWS:
                    mb_put(
                        tag,
                        scratch + fx.Int64(lay["qkv"][0]),
                        (16 * rt_ + tt) * (QKV_ROWS // 2) + row // 2,
                        w2,
                    )
                else:
                    st_if(
                        row < QKVG_ROWS,
                        w2,
                        g_out,
                        ((16 * rt_ + tt) * QROWS + row - QKV_ROWS) // 2,
                    )

    @traced
    def finish_vllm(
        scratch,
        q_out,
        k_cache,
        v_cache,
        slots,
        cos_sin,
        pos,
        tag,
        health,
        tid,
        hh,
        p,
        tt,
        nrows,
    ):
        """The finish of head hh (uniform), token tt (lanes 0-31 / 32-63 of a wave: two tokens): q / k as
        vLLM's fused_muse_spark_qknorm_rope_kv_insert (weightless qk-norm, GPT-J RoPE with fp32 cos / sin,
        all fp32, one RNE bf16 rounding; no qk_scale); K / V as one bf16 dword a lane into the LBHNC cache,
        element (slot >> 4) * bstride + (kv * 16 + (slot & 15)) * 128 + 64 isv + 2p, bstride = v_cache (a
        value); each half wave stores with its own wave-uniform 64-bit block base (readlane of its slot);
        slot < 0 and tt >= nrows skipped. Rows tt >= nrows are not real rows (their q is written, their KV
        is not)."""
        row0 = (hh * HD) // 2 + p
        got = mb_poll(
            tag,
            health,
            [(scratch + fx.Int64(lay["qkv"][0]), tt * (QKV_ROWS // 2) + row0, 1)],
            REGION_IDS["qkv"],
        )
        x0 = bf_lo(got[0][0])
        x1 = bf_hi(got[0][0])
        ss = butterfly(x0 * x0 + x1 * x1, (1, 2, 4, 8, 16))
        rs = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS))
        isv = hh >= HQ + HKV
        isq = hh < HQ
        n0 = x0 * rs
        n1 = x1 * rs
        if const_expr(key.next_rope):
            ps = ld_i32(pos, fx.min(tt, nrows - 1) * 2)
            c = ld_f32(cos_sin, ps * HD + p)
            sn = ld_f32(cos_sin, ps * HD + HD // 2 + p)
            r0 = n0 * c - n1 * sn
            r1 = n0 * sn + n1 * c
            n0 = r0
            n1 = r1
        o0 = isv.select(x0, n0)
        o1 = isv.select(x1, n1)
        if isq:
            bo.buffer_store(bf16_pair(o0, o1), rsrc(q_out), (tt * QROWS) // 2 + row0)
        else:
            kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
            slot = ld_i32(slots, fx.min(tt, nrows - 1) * 2)
            for half in range_constexpr(2):
                sl = readlane(slot, 32 * half)
                tl_ = readlane(tt, 32 * half)
                mine = ((tid % 64) >= 32) if half == 1 else ((tid % 64) < 32)
                if (sl >= 0) & (tl_ < nrows):
                    base = k_cache + (
                        fx.Int64(sl >> 4) * v_cache
                        + fx.Int64(
                            (kv * 16 + (sl & 15)) * 128
                            + isv.select(fx.Int32(HD), fx.Int32(0))
                        )
                    ) * fx.Int64(2)
                    if mine:
                        bo.buffer_store(bf16_pair(o0, o1), rsrc(base), p)

    @traced
    def head_finish(
        scratch,
        q_out,
        k_cache,
        v_cache,
        slots,
        cos_sin,
        pos,
        tag,
        health,
        tid,
        bid,
        nrows,
    ):
        """CTA hh < HQ + 2 HKV finishes head hh (q heads, then k, then v) of the next layer for all
        S tokens."""
        tid = fresh(tid)
        hh = bid
        p = tid % 32
        for tg in range_constexpr(NT):
            if bid < HQ + 2 * HKV:
                finish_vllm(
                    scratch,
                    q_out,
                    k_cache,
                    v_cache,
                    slots,
                    cos_sin,
                    pos,
                    tag,
                    health,
                    tid,
                    hh,
                    p,
                    16 * tg + tid // 32,
                    nrows,
                )

    @traced
    def stage_gather(own_peer, sfull_out, tag, health, tid, bid):
        """CTA t < S copies the all-gathered stream row t into sfull_out."""
        tid = fresh(tid)
        par = tag_par(tag)
        t = bid
        if tid == 0:
            mb_poll(
                tag,
                health,
                [(own_peer + fx.Int64(play["sgf"][0]), par * S + t, 1)],
                REGION_IDS["sgf"],
                cm=CM_SYS,
            )
        gpu.barrier()
        for h in range_constexpr(4):
            v = ld_vec(
                own_peer + fx.Int64(play["sg"][0]),
                (par * S + t) * HID + tid * 16 + 4 * h,
                4,
                XCM,
            )
            bo.buffer_store(v, rsrc(sfull_out), t * HID + tid * 16 + 4 * h)

    @traced
    def hb_body(
        part,
        w_pp,
        w_pff,
        alpha,
        beta,
        w_in,
        w_qkvg,
        stream_out,
        sfull_out,
        q_out,
        g_out,
        k_cache,
        v_cache,
        slots,
        cos_sin,
        pos,
        scratch,
        epoch,
        health,
        tl,
        peers,
        layer,
        rank,
        so_stride,
        so_off,
        gather,
        nrows,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        alloc = fx.SharedAllocator()
        misc = alloc.allocate(Misc).peek()
        sl = alloc.allocate(HbLds)
        sf, sq = sl.f_.peek(), sl.q_.peek()
        pl = list(peers)
        own_peer = sel_peer(pl, rank)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        ppo = pp_loads(w_pp, bid, lane, wave)
        if bid // CPR == rank:
            stage_ar(own_peer, pl, tag, health, tid, bid, rank)
        stage_post(
            ppo, pl, scratch, tag, health, tid, bid, lane, wave, rank, sf.red.ptr
        )
        qo = q_loads(w_qkvg, bid, lane, wave, 0, QPF)
        if const_expr(BND):  # all 256 CTAs, column groups
            stage_g_c(
                own_peer,
                scratch,
                w_pff,
                alpha,
                beta,
                w_in,
                stream_out,
                so_stride,
                so_off,
                sfull_out,
                gather,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                misc.pr.ptr,
            )
        else:
            if bid < TS:
                stage_g(
                    own_peer,
                    pl,
                    scratch,
                    w_pff,
                    alpha,
                    beta,
                    w_in,
                    stream_out,
                    so_stride,
                    so_off,
                    gather,
                    tag,
                    health,
                    tid,
                    bid,
                    lane,
                    wave,
                    rank,
                    misc.red4.ptr,
                )
        qo = qo + q_loads(w_qkvg, bid, lane, wave, QPF, NQT)
        stage_qkv(
            qo,
            own_peer,
            scratch,
            g_out,
            tag,
            health,
            tid,
            bid,
            lane,
            wave,
            sq.red.ptr,
            sq.own.ptr,
        )
        head_finish(
            scratch,
            q_out,
            k_cache,
            v_cache,
            slots,
            cos_sin,
            pos,
            tag,
            health,
            tid,
            bid,
            nrows,
        )
        if const_expr(not BND):  # the column-block G wrote sfull_out itself
            if (gather != 0) & (bid < nrows):
                stage_gather(own_peer, sfull_out, tag, health, tid, bid)

    # The kernels and their launchers. tl (a timeline buffer) is part of the launch signature the
    # decoder passes; these builds do not read it.

    @flyc.kernel(name=name_a, known_block_size=[THREADS, 1, 1])
    def ha(
        attn_o: Int64,
        attn_g: Int64,
        stream_in: Int64,
        w_o: Int64,
        alpha_a: Int64,
        beta_a: Int64,
        w_pffn: Int64,
        w_r: Int64,
        bias: Int64,
        w_pre: Int64,
        w_pren: Int64,
        h_out: Int64,
        ids_out: Int64,
        sc_out: Int64,
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
        g_stride: Int32,
        s_full: Int32,
        nrows: Int32,
    ):
        ha_body(
            attn_o,
            attn_g,
            stream_in,
            w_o,
            alpha_a,
            beta_a,
            w_pffn,
            w_r,
            bias,
            w_pre,
            w_pren,
            h_out,
            ids_out,
            sc_out,
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
            g_stride,
            s_full,
            nrows,
        )

    @flyc.kernel(name=name_b, known_block_size=[THREADS, 1, 1])
    def hb(
        part: Int64,
        w_pp: Int64,
        w_pff: Int64,
        alpha: Int64,
        beta: Int64,
        w_in: Int64,
        w_qkvg: Int64,
        stream_out: Int64,
        sfull_out: Int64,
        q_out: Int64,
        g_out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        slots: Int64,
        cos_sin: Int64,
        pos: Int64,
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
        so_stride: Int32,
        so_off: Int32,
        gather: Int32,
        nrows: Int32,
    ):
        hb_body(
            part,
            w_pp,
            w_pff,
            alpha,
            beta,
            w_in,
            w_qkvg,
            stream_out,
            sfull_out,
            q_out,
            g_out,
            k_cache,
            v_cache,
            slots,
            cos_sin,
            pos,
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
            so_stride,
            so_off,
            gather,
            nrows,
        )

    @flyc.jit
    def launch_a(
        attn_o: Int64,
        attn_g: Int64,
        stream_in: Int64,
        w_o: Int64,
        alpha_a: Int64,
        beta_a: Int64,
        w_pffn: Int64,
        w_r: Int64,
        bias: Int64,
        w_pre: Int64,
        w_pren: Int64,
        h_out: Int64,
        ids_out: Int64,
        sc_out: Int64,
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
        g_stride: Int32,
        s_full: Int32,
        nrows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _KEYSTR
        ha(
            attn_o,
            attn_g,
            stream_in,
            w_o,
            alpha_a,
            beta_a,
            w_pffn,
            w_r,
            bias,
            w_pre,
            w_pren,
            h_out,
            ids_out,
            sc_out,
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
            g_stride,
            s_full,
            nrows,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    @flyc.jit
    def launch_b(
        part: Int64,
        w_pp: Int64,
        w_pff: Int64,
        alpha: Int64,
        beta: Int64,
        w_in: Int64,
        w_qkvg: Int64,
        stream_out: Int64,
        sfull_out: Int64,
        q_out: Int64,
        g_out: Int64,
        k_cache: Int64,
        v_cache: Int64,
        slots: Int64,
        cos_sin: Int64,
        pos: Int64,
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
        so_stride: Int32,
        so_off: Int32,
        gather: Int32,
        nrows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _KEYSTR
        hb(
            part,
            w_pp,
            w_pff,
            alpha,
            beta,
            w_in,
            w_qkvg,
            stream_out,
            sfull_out,
            q_out,
            g_out,
            k_cache,
            v_cache,
            slots,
            cos_sin,
            pos,
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
            so_stride,
            so_off,
            gather,
            nrows,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch_a, launch_b)
    return _BUILDS[key]


def build_vllm(S=32, W=8, next_rope=True, attn=""):
    """The HA / HB build vLLM launches for S rows (32 / 64 / 96 / 128) of a TP = EP = 8 group;
    next_rope: the next layer has RoPE (HB); attn = "swa" / "full": this layer's verify attention
    fused at the HA head."""
    assert W == 8 and S in (32, 64, 96, 128), (S, W)
    return Build(S=S, next_rope=bool(next_rope), attn=attn)
