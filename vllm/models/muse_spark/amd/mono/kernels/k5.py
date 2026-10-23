# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501, F841, SIM102

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
"""K5 (milestone 5): the decode layer at S = 32 / 64 / 128 rows as two persistent launches
around the library MoE (muse_moe_flydsl), rung "hybrid" of the ladder.

    K1 (library attention) -> HA -> library MoE (align, gate_up, down, finalize) -> HB

HA  o-norm x gate (distributed) -> o_proj (rows 32b.., K = this rank's 2048) -> the bf16
    partial tile goes to the rank that owns those hidden columns (reduce-scatter) ->
    attention boundary on this rank's hidden quarter (post_attn_norm, gated residual,
    pre-FFN norm; per-token sums of squares reduced over the 256 CTAs and the 4 ranks)
    -> ffn_in quarter all-gathered (plain bf16 + flags) -> router (fp32, 64 experts a rank)
    and pre_expert_proj (1024 rows a rank), K quarters over 4-CTA groups -> logits and
    h_raw all-gathered -> top-8 (CTA t routes token t) -> h = pre_expert_norm(h_raw)
    (CTA t: token t) -> h, topk ids / weights for the library MoE.
HB  EP all-reduce of the library's partial (one-shot push + distributed sum) ->
    post_expert_proj (2048 rows = this rank's hidden quarter, K halves) -> post-FFN norm,
    gated residual, next input norm on the quarter -> xn quarter all-gathered -> qkv+gate
    (K eighths, 9 tiles a CTA) -> head finishers (qk-norm, x qk_scale, RoPE, fp8 KV write).

Token-sharded boundaries: rank r owns tokens [r S/4, (r+1) S/4). The o_proj partials and the
post_expert_proj outputs go to the owner rank of each token; one CTA a token then does the
whole row (all-reduce sum, both norms, gated residual) with no cross-CTA reduction, and the
ffn_in / xn rows are all-gathered. The residual stream lives token-sharded (rank r: its S/4
rows, fp32, full 8192 columns); HA reads it as rows (s_row_off, stride 8192) so the first layer
can read a full replicated stream; HB writes it the same way and (``gather``) all-gathers it.

Every row tile is 16 tokens (MFMA N); dense stages keep their weights in VGPRs and loop over
the S / 16 row tiles with the activations read straight from L2 / peer memory in MFMA
layout. Tags: tag = epoch * 64 + layer + 1 (HA and HB of a layer share it, regions are
disjoint); peer regions double buffered by tag parity.
"""

import math
import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from . import portion as _portion
from .dev import (
    BLOCKS,
    CM_DEV,
    CM_NT,
    CM_SYS,
    THREADS,
    WAVES,
    _add,
    ballot,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    bpermute,
    butterfly,
    cache_inv,
    fp8x2,
    fresh,
    hw_exp2,
    hw_rsq,
    hw_sqrt,
    lanes_below,
    ld_f32,
    ld_i32,
    ld_vec,
    mb_poll,
    mb_put,
    mb_put_if,
    mb_put_sys,
    mfma_bf16,
    popcount,
    readlane,
    rsrc,
    st_if,
    stamp,
    stamp_dep,
    traced,
    uniform,
    wave_sum,
    xred,
    xshfl,
)

HID = 8192
QH = HID // 4  # hidden columns a rank owns (W = 4; builds use geom5(W))
DM = 4096
E = 256
EL = E // 4
TOPK = 8
WMAX = 8
MULT = 0.17677669529663687
ROUTE_EPS = 1e-15
PRE_EPS = 1e-5
POST_EPS = 1e-8
LOG2E = 1.0 / math.log(2.0)
KCH = 128
HQ, HKV, HD = 32, 4, 64
QROWS, KROWS = HQ * HD, HKV * HD
QKV_ROWS = QROWS + 2 * KROWS
QKVG_ROWS = QKV_ROWS + QROWS
OK_K = 2048
QK_SCALE = 5.4730064863875
TL_POINTS = 32
# loads of peer-written data (after its system-scope flag): device scope = past the per-XCD L2,
# served by the memory-side cache, which every write to this memory passes (MK5_XCM=sys: system scope)
XCM = CM_SYS if os.environ.get("MK5_XCM", "dev") == "sys" else CM_DEV
# activations re-read by many CTAs: cached loads after an acquire invalidate (MK5_YCM=dev: device scope)
YCM = 0 if os.environ.get("MK5_YCM", "dev") == "l2" else CM_DEV
ACM = 0 if os.environ.get("MK5_ACM", "dev") == "l2" else XCM  # stage A's x tiles
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
    # bnd builds (K3-style boundaries at W = 8, S = 32)
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
class Geom5:
    """World-size dependent constants of HA / HB (TP = EP = W, token owner t % W)."""

    W: int
    TS: int  # tokens a rank owns (S / W)
    QH: int  # hidden columns a rank owns (= stage-A K part = post-proj rows)
    EL: int  # local experts = router experts a rank = stage-A groups
    KQ: int  # stage-A K parts (BLOCKS / EL): 4 / 8
    OWN: int  # pre-proj rows a CTA owns after the K-part exchange (16 / KQ): 4 / 2
    NRP: int  # their row pairs: 2 / 1
    NTILES: int  # post-proj 16-row tiles a rank (QH / 16): 128 / 64
    KSP: int  # post-proj K split (BLOCKS / NTILES): 2 / 4
    HQ: int
    HKV: int
    QROWS: int
    QKV_ROWS: int
    QKVG_ROWS: int
    OK_K: int
    NQT: int  # qkvg tiles a CTA group (32 groups x 8 K eighths): 9 / 5 (TP8: 144 of 160 real)
    NQTILES: int


def geom5(W, S=None):
    assert W in (4, 8) and W <= WMAX, W
    el = E // W
    kq = BLOCKS // el
    own = 16 // kq
    qh = HID // W
    hq, hkv = 128 // W, 16 // W
    qr = hq * HD
    qkv = qr + 2 * hkv * HD
    nqtiles = (qkv + qr) // 16
    return Geom5(
        W=W,
        TS=(S // W if S else 0),
        QH=qh,
        EL=el,
        KQ=kq,
        OWN=own,
        NRP=own // 2,
        NTILES=qh // 16,
        KSP=BLOCKS // (qh // 16),
        HQ=hq,
        HKV=hkv,
        QROWS=qr,
        QKV_ROWS=qkv,
        QKVG_ROWS=qkv + qr,
        OK_K=8192 // W,
        NQT=-(-nqtiles // 32),
        NQTILES=nqtiles,
    )


@dataclass(frozen=True)
class Build:
    S: int = 32
    W: int = 4
    next_rope: bool = True
    qkv_pf: int = 9  # qkvg tiles issued before stage G (of 9)
    timeline: bool = False
    stop: int = (
        99  # debug: HA stops after stage `stop` (1 ng, 2 o, 3 ab, 4 a, 5 route/h)
    )
    hm: bool = False  # the MoE is HM (k5m), which pushes the EP partial itself: HB skips its push
    only: str = ""  # ablation: stage mask (mk/portion.py; "" = every stage, the shipped kernels)
    # TEP8 c = 4 boundaries (W = 8, S = 32 only): 1 = #11 the K3-style attention boundary (O5 reduce-scatter to the
    # column-block owner + all-gather of the reduced o, the boundary on all 256 CTAs, local ffn_in for A5; the
    # stream input is replicated); 2 = #11 + #1 the replicated FFN boundary (post5's y to every rank, G on all 256
    # CTAs, the full stream out, local xn for QKV5, no gather5). 0 = the token-owner boundaries (shipped).
    bnd: int = (
        0  # 3 = 2 with the column-group boundary mapping (#20: one-hop token norms)
    )
    qr1: bool = False  # W = 8, S = 32: QKV5 reduces both 16-token row tiles in one LDS round (R2 #4; bit-identical)
    attn: str = ""  # S4 (MK_FUSED_ATTN=1): this layer's verify attention fused at the HA head ("swa" / "full")
    attn_olate: bool = False  # S4 #10 A/B: issue the o_proj weights after the fused attention instead of before
    # vLLM build flags (as k3.Build; defaults = the SGLang kernels, same binaries as before the flags)
    mult1: bool = False  # HA router MULT = 1 (vLLM folds output_multiplier into the router weight)
    vqk: bool = False  # HB q / k finish as vLLM's fused qk-norm + RoPE: no qk_scale, fp32, one bf16 rounding
    kvl: str = ""  # HB KV store: "" = fp8 [slot][H][64] K / V (SGLang); "lbhnc" = bf16 vLLM LBHNC (see k3.Build.kvl)


def _lay(items):
    out, off = {}, 0
    for name, n in items:
        off = (off + 255) // 256 * 256
        out[name] = (off, n)
        off += n
    return out, (off + 4095) // 4096 * 4096


def scratch_layout(S, W=4, bnd=0):
    G = geom5(W, S)
    TS = G.TS
    extra = (
        []
        if not bnd
        else [
            (
                "smidb",
                S * HID * 4,
            ),  # the mid stream, replicated [t][col] fp32 (each CTA its 32 columns)
            ("xinb", S * HID * 2),
            ("xinbf", BLOCKS * 8),  # ffn_in [t][col] bf16 + a flag a CTA
            ("xnb", S * HID * 2),
            ("xnbf", BLOCKS * 8),  # the next layer's input [t][col] bf16 + a flag a CTA
            ("ssa4", (4 * S * BLOCKS + 2 * S) * 8),
            ("ssg4", (4 * S * BLOCKS + 2 * S) * 8),  # ONEPASS partials, rstd
            ("gpa", BLOCKS * 16 * 8),
            (
                "gpg",
                BLOCKS * 16 * 8,
            ),  # bnd 3: [CTA][4 tokens][4] partials of the AB / G
        ]
    )
    return _lay(
        [
            ("y", S * G.OK_K * 2),
            ("yf", BLOCKS * 8),
            ("smid", TS * HID * 4),
            ("px", G.EL * G.KQ * 16 * S * 8),
            ("m", S * DM * 2),
            ("mmf", BLOCKS * 8),
            (
                "py",
                max(128, (G.KSP - 1) * G.NTILES) * 16 * S * 8,
            ),  # [K part - 1][tile][row][t]
            ("qx", 32 * 8 * G.NQT * 16 * S * 8),
            ("qkv", S * (G.QKV_ROWS // 2) * 8),
        ]
        + extra
    )


def peer_layout(S, W=4, bnd=0):
    G = geom5(W, S)
    TS = G.TS
    extra = (
        []
        if not bnd
        else [
            (
                "opb",
                2 * W * S * (BLOCKS // W) * 16 * 8,
            ),  # o_proj partials to the block owner [par][src][t][own blk][pair]
            (
                "orb",
                2 * S * (HID // 2) * 8,
            ),  # the owner's reduced o, all-gathered [par][t][pair] tagged
            (
                "yb",
                2 * S * (HID // 2) * 8,
            ),  # post5's y, all-gathered [par][t][pair] tagged (bnd 2)
        ]
    )
    # the bnd regions go last: HM (k5m) addresses moep / mf with the layout of the plain build
    return _lay(
        [
            ("op", 2 * W * TS * (HID // 2) * 4),
            ("opf", 2 * W * BLOCKS * 8),
            ("xin", 2 * S * (HID // 2) * 4),
            ("xf", 2 * S * 8),
            ("logit", 2 * G.KQ * S * E * 8),
            ("hraw", 2 * S * (DM // 2) * 4),
            ("hf", 2 * W * BLOCKS * 8),
            ("moep", 2 * W * S * (DM // 2) * 4),
            ("mf", 2 * W * BLOCKS * 8),
            ("moem", 2 * S * (DM // 2) * 4),
            ("mmf", 2 * BLOCKS * 8),
            ("yt", 2 * TS * (HID // 2) * 4),
            ("ytf", 2 * W * G.NTILES * 8),
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
    W = key.W
    assert S in (32, 64, 96, 128) and S % W == 0, (W, S)
    G = geom5(W, S)
    NT = S // 16
    TS = G.TS  # tokens a rank owns in the boundaries
    QH, EL, KQ, OWN, NRP, NTILES, KSP = G.QH, G.EL, G.KQ, G.OWN, G.NRP, G.NTILES, G.KSP
    HQ, HKV, QROWS, QKV_ROWS, OK_K, NQT = (
        G.HQ,
        G.HKV,
        G.QROWS,
        G.QKV_ROWS,
        G.OK_K,
        G.NQT,
    )
    CO = OK_K // KCH // WAVES  # o_proj chunks a wave: 2 / 1
    CA = QH // KCH // WAVES  # stage-A pre-proj chunks a wave: 2 / 1
    CP = (DM // KSP) // KCH // WAVES  # post-proj chunks a wave: 2 / 1
    KPT = QH // THREADS  # router K a thread: 4 / 2
    NFQ = W * BLOCKS // THREADS  # producer flags a thread (opf / hf polls): 2 / 4
    QPF = min(key.qkv_pf, NQT)
    CPR = (
        BLOCKS // W
    )  # CTAs (16-column blocks) a rank owns in the EP reduce-scatter: 64 / 32
    assert TS * W == S and NFQ * THREADS == W * BLOCKS and W * NTILES * 8 >= 512
    BND = key.bnd
    assert BND in (0, 1, 2, 3) and (BND == 0 or (W == 8 and S == 32)), (
        "bnd: TEP8 32-row builds only",
        W,
        S,
        BND,
    )
    BG = BND >= 2  # the replicated FFN boundary (#1)
    QR1 = key.qr1
    assert not QR1 or (W == 8 and S == 32), "qr1: TEP8 32-row builds only"
    lay, _ = scratch_layout(S, W, BND)
    play, _ = peer_layout(S, W, BND)
    TL = key.timeline
    tag_par = lambda tag: tag & 1  # noqa: E731

    @fx.struct
    class RouteLds:
        rid: fx.Array[fx.Int32, TOPK, 16]
        rw: fx.Array[fx.Float32, TOPK, 16]
        sel: fx.Array[fx.Int32, TOPK * 3 + 8, 16]

    @fx.struct
    class HaO:
        red: fx.Array[fx.Float32, NT * WAVES * 2 * 64 * 4, 16]

    XAROW = QH + 8  # LDS row of the x quarter tile (bf16)

    @fx.struct
    class HaA:
        x: fx.Array[fx.Int32, 16 * XAROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tok: fx.Array[fx.Float32, WAVES * 16 + 8 + S, 16]
        own: fx.Array[fx.Float32, NT * OWN * 16, 16]

    FA = None
    if key.attn:  # S4: the fused attention fragment (mk/attn_frag.py), TP8 only
        from . import attn_frag

        assert W == 8 and key.attn in ("swa", "full"), key
        # kvl = lbhnc: the vLLM build (bf16 LBHNC cache)
        FA = attn_frag.make(S // 8, key.attn == "swa", kvl=key.kvl)
        FaLds = FA.Lds
        _KEYSTR = (
            _KEYSTR + FA.key
        )  # the fragment's source hash: the disk cache does not follow FA.run

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
        tls: fx.Array[fx.Int64, TL_POINTS, 16]
        rs: fx.Array[
            fx.Float32, 2 * S + (32 if BND else 8), 16
        ]  # bnd: tok_rstd2 uses rs[2S .. 2S + 32)
        pr: fx.Array[
            fx.Float32, 512 + 64 if BND == 3 else 8, 16
        ]  # bnd 3: the polled group partials + wave sums
        red4: fx.Array[fx.Float32, 32, 16]

    WN = "" if W == 4 else f"_w{W}"  # the TP4 kernel names are unchanged
    BN = (f"_bnd{BND}" if BND else "") + ("_qr1" if QR1 else "")
    name_a = (
        f"muse_k5a_s{S}_t{int(TL)}_x{key.stop}_c{XCM}_y{YCM}_a{ACM}_i{os.environ.get('MK5_NOINV', '0')}{WN}"
        + BN
        + _portion.name_tag(key.only)
        + (f"_fa{key.attn}{int(key.attn_olate)}" if key.attn else "")
        + ("_m1" if key.mult1 else "")
        + (f"_kv{key.kvl}" if key.attn and key.kvl else "")
    )
    name_b = (
        f"muse_k5b_r{int(key.next_rope)}q{key.qkv_pf}_s{S}_t{int(TL)}_c{XCM}_m{int(key.hm)}{WN}"
        + BN
        + _portion.name_tag(key.only)
        + ("_vqk" if key.vqk else "")
        + (f"_kv{key.kvl}" if key.kvl else "")
    )
    RMULT = 1.0 if key.mult1 else MULT  # the router sigmoid's logit multiplier
    VQK = key.vqk
    VKV = key.kvl == "lbhnc"
    assert key.kvl in ("", "lbhnc"), key.kvl
    # fused attention: SGLang (fp8 KV, q x qk_scale) or vLLM (bf16 LBHNC KV, q without qk_scale)
    assert not key.attn or VQK == VKV, key
    ONLY = _portion.stage_set(
        key.only, _portion.K5A_STAGES + _portion.K5B_STAGES, _portion.K5_DEPS
    )

    def on(stage):  # stage active in this build (const at trace time)
        return ONLY is None or stage in ONLY

    @traced
    def flush_range(tls, tl, tid, bid, lo, hi):
        if const_expr(TL):  # noqa: SIM102
            if (tid >= lo) & (tid < hi):
                v = fx.ptr_load(tls + tid)
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [fx.Int32(v & 0xFFFFFFFF), fx.Int32(v >> 32)], fx.Int32
                    ),
                    rsrc(tl),
                    (bid * TL_POINTS + tid) * 2,
                )

    def sel_peer(pl, d):
        p = pl[0]
        for r_ in range_constexpr(1, W):
            p = (d == r_).select(pl[r_], p)
        return p

    # ------------------------------------------------------------ shared helpers
    def bf_mma_operand(ops, qq, lane):
        """Coalesced bf16 weight registers (rows 8 blk + lane / 8, 16 B at 128 h + 16 (lane % 8))
        -> the MFMA A operand of step qq (K 32 qq + 8 g of row r)."""
        r = lane % 16
        g = lane // 16
        h = qq // 2
        src = 8 * (r % 8) + 4 * (qq % 2) + g
        hi = r >= 8
        ws = []
        for jj in range_constexpr(4):
            pa = bpermute(ops[h][jj], src)
            pb = bpermute(ops[2 + h][jj], src)
            ws.append(hi.select(pb, pa))
        return fx.Vector.from_elements(ws, fx.Int32)

    def bf_tile_load(wbase, row16, ldk, k, lane):
        """16 rows (row16 ..) x 128 K (k ..) of a row-major bf16 [*, ldk] weight, coalesced."""
        out = []
        for blk in range_constexpr(2):
            rowc = row16 + 8 * blk + lane // 8
            for h in range_constexpr(2):
                out.append(
                    ld_vec(
                        wbase, (rowc * ldk + k + 64 * h + 8 * (lane % 8)) // 2, 4, CM_NT
                    )
                )
        return out

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

    def okey(v):
        b = v.bitcast(fx.Int32)
        neg = b < fx.Int32(0)
        return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))

    @traced
    def route_token(tkns, bias_v, lgss, tid, lane, rt, tls=None):
        """Tokens ``tkns`` (this wave's, all valid), interleaved for ILP: sigmoid +
        bias, radix top-8, rank, renorm -> rid / rw."""
        T_ = len(tkns)
        keys, sigs = [], []
        for x_ in range_constexpr(T_):
            kx, sx = [], []
            for i in range_constexpr(4):
                lg = lgss[x_][i].bitcast(fx.Float32)
                sg = fx.Float32(1.0) / (
                    fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-RMULT * LOG2E))
                )
                sc = sg + bias_v[i]
                live = sc == sc  # noqa: PLR0124  (NaN ranks last)
                kx.append(okey(live.select(sc, fx.Float32(-3.0e38))))
                sx.append((sg == sg).select(sg, fx.Float32(0.0)))  # noqa: PLR0124
            keys.append(kx)
            sigs.append(sx)
        th = [fx.Uint32(0) for _ in range(T_)]
        for bit in range_constexpr(31, -1, -1):
            for x_ in range_constexpr(T_):
                cand = th[x_] | fx.Uint32(1 << bit)
                n = fx.Int32(0)
                for i in range_constexpr(4):
                    n = n + popcount(ballot(keys[x_][i] >= cand))
                th[x_] = (n >= TOPK).select(cand, th[x_])
        if const_expr(tls is not None):
            stamp_dep(TL, tls, tid, 12, fx.Int32(th[0]))
        for x_ in range_constexpr(T_):
            tkn = tkns[x_]
            n_gt = fx.Int32(0)
            for i in range_constexpr(4):
                n_gt = n_gt + popcount(ballot(keys[x_][i] > th[x_]))
            need = fx.Int32(TOPK) - n_gt
            eq_before = fx.Int32(0)
            sel_before = fx.Int32(0)
            sel_base = rt["sel"] + tkn * (TOPK * 3)
            for i in range_constexpr(4):
                eqb = ballot(keys[x_][i] == th[x_])
                eq_rank = eq_before + lanes_below(eqb, lane)
                sel = (keys[x_][i] > th[x_]) | (
                    (keys[x_][i] == th[x_]) & (eq_rank < need)
                )
                selb = ballot(sel)
                pos = sel_before + lanes_below(selb, lane)
                if sel:
                    fx.ptr_store(keys[x_][i].bitcast(fx.Int32), sel_base + pos * 3)
                    fx.ptr_store(fx.Int32(lane + 64 * i), sel_base + (pos * 3 + 1))
                    fx.ptr_store(
                        sigs[x_][i].bitcast(fx.Int32), sel_base + (pos * 3 + 2)
                    )
                eq_before = eq_before + popcount(eqb)
                sel_before = sel_before + popcount(selb)
        q = fx.min(lane, TOPK - 1)
        mks, mes, mss, ranks = [], [], [], []
        for x_ in range_constexpr(T_):
            sel_base = rt["sel"] + tkns[x_] * (TOPK * 3)
            mk = fx.Uint32(fx.ptr_load(sel_base + q * 3))
            me = fx.ptr_load(sel_base + (q * 3 + 1))
            ms = fx.ptr_load(sel_base + (q * 3 + 2)).bitcast(fx.Float32)
            rank_ = fx.Int32(0)
            for j in range_constexpr(TOPK):
                ok = fx.Uint32(fx.ptr_load(sel_base + j * 3))
                oe = fx.ptr_load(sel_base + (j * 3 + 1))
                before = (ok > mk) | ((ok == mk) & (oe < me))
                rank_ = rank_ + before.select(fx.Int32(1), fx.Int32(0))
            mks.append(mk)
            mes.append(me)
            mss.append(ms)
            ranks.append(rank_)
        if const_expr(tls is not None):
            stamp_dep(TL, tls, tid, 13, ranks[0])
        # serial sum of the 8 sigmoids in rank order (+ eps), IEEE division
        for x_ in range_constexpr(T_):
            if lane < TOPK:
                fx.ptr_store(mss[x_], rt["rw"] + (tkns[x_] * TOPK + ranks[x_]))
        for x_ in range_constexpr(T_):
            tkn = tkns[x_]
            total = fx.Float32(0.0)
            for j in range_constexpr(TOPK):
                total = total + fx.ptr_load(rt["rw"] + (tkn * TOPK + j))
            total = total + fx.Float32(ROUTE_EPS)
            if lane < TOPK:
                fx.ptr_store(
                    fx.min(fx.max(mes[x_], 0), fx.Int32(E - 1)),
                    rt["rid"] + (tkn * TOPK + ranks[x_]),
                )
                fx.ptr_store(mss[x_] / total, rt["rw"] + (tkn * TOPK + ranks[x_]))

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
            tl_ = fx.min(
                t, nrows - 1
            )  # padded rows (t >= nrows) repeat the last real row
            if const_expr(ao is None):
                xv = ld_vec(attn_o, (tl_ * OK_K + kk) // 2, 4)
            else:  # S4 fused attention (aot=1): the merged rows as tagged pairs
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
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if (tid == 0) & (bid < S * HQ * 8 // THREADS):
            mb_put(tag, scratch + fx.Int64(lay["yf"][0]), bid, fx.Int32(1))

    @traced
    def stage_o(o_ops, scratch, pl, tag, health, tid, bid, lane, wave, rank, red, tls):
        """o_proj rows 32b .. (this rank's K slice) for all S tokens; the bf16 partial
        goes to the owner rank of those hidden columns (bid / 64)."""
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
        if const_expr((YCM == 0) & (os.environ.get("MK5_NOINV", "0") != "1")):
            if wave == 0:
                cache_inv(False)
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
            if const_expr(gi == 0):
                stamp_dep(TL, tls, tid, 24, accs[0][0].bitcast(fx.Int32))
            gpu.barrier()
            if const_expr(gi + 1 < NG):
                nxt = ob_loads(2 * gi + 2) + ob_loads(2 * gi + 3)
            if const_expr(BND):
                # #11: the CTA's whole [32 columns x tokens] tile to the owner of block b (b // 32), tagged pairs
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
                if const_expr(W == 4):
                    # owner of token t is rank t % 4: wave w pushes the 4 tokens of owner w % 4 of tile w / 4
                    rt_ = 2 * gi + e // 256
                    pp = e % 16
                    dw = uniform(wave % 4)
                    tt = dw + 4 * ((e // 16) % 4)
                else:
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
                # the waitcnt pass drain everything), and the 4 links carry every group in parallel
                idx = (((par * W + rank) * BLOCKS + bid) * TS + t_ // W) * 16 + pp
                w2 = bf16_pair(v[0], v[1])
                bo.buffer_store(
                    w2,
                    rsrc(sel_peer(pl, dw) + fx.Int64(play["op"][0])),
                    idx,
                    cache_modifier=CM_SYS,
                )
        stamp(TL, tls, tid, 25)
        if const_expr(not BND):  # bnd: tagged pairs, no flags
            rocdl.s_waitcnt(0)
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
        tls,
    ):
        """CTA j < S/4: token t = rank S/4 + j, all 8192 columns (16 a thread): the 4 o_proj
        partials (rank order) -> o, post_attn_norm (weightless, 1e-8), gated residual (mid stream,
        local), pre_feedforward_layernorm -> the ffn_in row pushed to every rank + flag."""
        tid = fresh(tid)
        par = tag_par(tag)
        j = bid
        t = W * j + rank
        srow = (s_full != 0).select(
            fx.min(t, nrows - 1), j
        )  # replicated [rows, 8192] or token-sharded
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
        stamp(TL, tls, tid, 15)
        a = [fx.Float32(0.0) for _ in range(16)]
        for src in range_constexpr(W):
            for h in range_constexpr(2):
                q = (
                    tid + THREADS * h
                )  # 8-column chunk: columns 8q .. 8q + 8 (block q / 4)
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
        stamp(TL, tls, tid, 16)
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
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            for pb in pl:
                mb_put_sys(tag, pb + fx.Int64(play["xf"][0]), par * S + t, fx.Int32(1))

    def a_loads(w_r, w_pre, bid, tid, lane, wave):
        """CTA b = (group rb = b / KQ, K part kq = b % KQ): router expert rb of this rank (fp32,
        KPT = 4 / 2 K a thread) and pre_expert_proj rows 16 rb .. (CA = 2 / 1 chunk tiles a wave)."""
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
        tls,
        xl,
    ):
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        kq = bid % KQ
        rb = bid // KQ
        if const_expr(
            BND
        ):  # #11: ffn_in is local; the 32 CTAs of this K part wrote its columns
            if tid < 32:
                if const_expr(BND == 3):
                    fa = (tid // 4) * 32 + 4 * kq + tid % 4
                else:
                    fa = kq * 32 + tid
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
        if const_expr(ACM == 0):
            if wave == 0:
                cache_inv(True)
                rocdl.s_waitcnt(0)
            gpu.barrier()
        stamp(TL, tls, tid, 13)
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
        QV = QH // 8  # 16-B vectors a token row of the x tile: 256 / 128
        NXQ = 16 * QV // THREADS  # of them a thread: 8 / 4

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
            # the tile's x quarter (16 tokens x 2048) -> LDS, 8 x 16 B a thread
            xs = xnext
            for q in range_constexpr(NXQ):
                e = tid + THREADS * q
                fx.ptr_store(xs[q], xl + ((e // QV) * XAROW + 8 * (e % QV)) // 2)
            gpu.barrier()
            if const_expr(rt + 1 < NT):
                xnext = ax_loads(rt + 1)
            if const_expr(rt == 0):
                stamp(TL, tls, tid, 18)
            if const_expr(rt == 1):
                stamp(TL, tls, tid, 26)
            # router: thread's 4 K of every token of the tile; all LDS reads first, then the
            # 16 wave reductions interleaved (ILP)
            if const_expr(KPT == 4):
                xws = [
                    fx.Vector(
                        fx.ptr_load(
                            xl + (t16 * XAROW + 4 * tid) // 2,
                            result_type=fx.Vector.make_type(2, fx.Int32),
                        )
                    )
                    for t16 in range(16)
                ]
            else:
                xws = [
                    fx.ptr_load(xl + (t16 * XAROW + 2 * tid) // 2) for t16 in range(16)
                ]
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
            if const_expr(KPT == 4):
                prs = [
                    bf_lo(xw[0]) * wf[0]
                    + bf_hi(xw[0]) * wf[1]
                    + bf_lo(xw[1]) * wf[2]
                    + bf_hi(xw[1]) * wf[3]
                    for xw in xws
                ]
            else:
                prs = [bf_lo(xw) * wf[0] + bf_hi(xw) * wf[1] for xw in xws]
            for off in (32, 16, 8, 4, 2, 1):
                prs = [xred(v, off, _add) for v in prs]
            if lane < 16:
                v_ = prs[0]
                for t16 in range_constexpr(1, 16):
                    v_ = (lane == t16).select(prs[t16], v_)
                fx.ptr_store(v_, tok + (wave * 16 + lane))
            if const_expr(rt == 0):
                stamp_dep(TL, tls, tid, 22, prs[0].bitcast(fx.Int32))
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for ci in range_constexpr(CA):
                for qq in range_constexpr(4):
                    acc = mfma_bf16(wa[ci][qq], bvs[ci * 4 + qq], acc)
            fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            if const_expr(rt == 0):
                stamp(TL, tls, tid, 19)
            if const_expr(rt == 1):
                stamp(TL, tls, tid, 27)
            # branch-free stores (dropped out-of-range offsets): no waitcnt drain at a join
            t16 = tid % 16
            v = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                v = v + fx.ptr_load(tok + (w * 16 + t16))
            if tid < 16:
                fx.ptr_store(
                    v, tok + (WAVES * 16 + 8 + rt * 16 + t16)
                )  # logits of the CTA's expert, pushed after the loop
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
            if const_expr(rt == 1):
                stamp_dep(TL, tls, tid, 28, s_.bitcast(fx.Int32))
        gpu.barrier()
        stamp(TL, tls, tid, 20)
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
        stamp(TL, tls, tid, 21)
        rocdl.s_waitcnt(0)
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
        if const_expr(W == 4):
            h0 = ld_vec(own_peer + fx.Int64(play["hraw"][0]), hb0, 2, XCM)
            h1 = ld_vec(own_peer + fx.Int64(play["hraw"][0]), hb0 + 2 * S, 2, XCM)
            hv = fx.Vector.from_elements([h0[0], h0[1], h1[0], h1[1]], fx.Int32)
        else:
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

    # ============================================================ bnd (W = 8, S = 32): K3-style boundaries
    def put4_b(tag, scratch, region, tt, bid, q1, q2, q3, q4):
        """The 4 ONEPASS partials of token tt, CTA bid -> scratch region [t][CTA][4] (tagged), as K3 put4."""
        base = scratch + fx.Int64(lay[region][0])
        i0 = (tt * BLOCKS + bid) * 4
        bo.buffer_store(
            fx.Vector.from_elements(
                [q1.bitcast(fx.Int32), tag, q2.bitcast(fx.Int32), tag], fx.Int32
            ),
            rsrc(base),
            i0 * 2,
            cache_modifier=CM_DEV,
        )
        bo.buffer_store(
            fx.Vector.from_elements(
                [q3.bitcast(fx.Int32), tag, q4.bitcast(fx.Int32), tag], fx.Int32
            ),
            rsrc(base),
            (i0 + 2) * 2,
            cache_modifier=CM_DEV,
        )

    @traced
    def tok_rstd2_b(
        scratch, region, tag, health, tid, lane, rs_out, eps1, eps2, bid, red, tls=None
    ):
        """K3 tok_rstd2: one round for two norms of a token. Reducer CTA 255 - t sums the 256 CTAs' 4 partials in a
        fixed order: r1 = rsqrt(P1 / HID + eps1), r2 = rsqrt((P2 + 2 r1 P3 + r1^2 P4) / HID + eps2).
        rs_out[t] = r1, rs_out[S + t] = r2 on every CTA (red: WAVES x 4 floats of LDS)."""
        rbase = scratch + fx.Int64(lay[region][0])
        tred = bid >= BLOCKS - S
        if tred:
            t = BLOCKS - 1 - bid
            va = fx.Float32(0.0)
            vb = fx.Float32(0.0)
            if tid < 2 * BLOCKS:
                got = mb_poll(
                    tag,
                    health,
                    [(rbase, (t * BLOCKS + tid // 2) * 4 + 2 * (tid % 2), 2)],
                    REGION_IDS[region],
                )
                va = got[0][0].bitcast(fx.Float32)
                vb = got[0][1].bitcast(fx.Float32)
            va = butterfly(va, (32, 16, 8, 4, 2))  # lanes of one parity: 32 CTAs
            vb = butterfly(vb, (32, 16, 8, 4, 2))
            if lane < 2:
                fx.ptr_store(va, red + (tid // 64) * 4 + 2 * lane)
                fx.ptr_store(vb, red + (tid // 64) * 4 + 2 * lane + 1)
            gpu.barrier()
            if const_expr(tls is not None):
                stamp_dep(
                    TL, tls, tid, 30, va.bitcast(fx.Int32)
                )  # the reducer has all 256 partials
            if tid == 0:
                p = [fx.Float32(0.0) for _ in range(4)]
                for w in range_constexpr(WAVES):
                    for j in range_constexpr(4):
                        p[j] = p[j] + fx.ptr_load(red + w * 4 + j)
                r1 = hw_rsq(p[0] * fx.Float32(1.0 / HID) + fx.Float32(eps1))
                q = p[1] + fx.Float32(2.0) * r1 * p[2] + r1 * r1 * p[3]
                r2 = hw_rsq(q * fx.Float32(1.0 / HID) + fx.Float32(eps2))
                mb_put(tag, rbase, 4 * S * BLOCKS + 2 * t, r1)
                mb_put(tag, rbase, 4 * S * BLOCKS + 2 * t + 1, r2)
            if const_expr(tls is not None):
                stamp(TL, tls, tid, 31)  # published
        if tid < S:
            got2 = mb_poll(
                tag, health, [(rbase, 4 * S * BLOCKS + 2 * tid, 2)], REGION_IDS[region]
            )
            fx.ptr_store(got2[0][0].bitcast(fx.Float32), rs_out + tid)
            fx.ptr_store(got2[0][1].bitcast(fx.Float32), rs_out + (S + tid))
        gpu.barrier()

    @traced
    def stage_ab_b(
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
        rank,
        rsl,
        tls,
    ):
        """#11, CTA b = hidden columns 32b .. on every rank, thread = (token tid / 16, column pair tid % 16):
        the block owner (b / 32 == rank) sums the 8 o_proj partials in rank order (fp32 -> bf16, the all-reduce
        output) and all-gathers it; every CTA: post_attn_norm (weightless, 1e-8), gated residual on the
        replicated fp32 stream, pre_feedforward_layernorm, one ONEPASS token-norm round over the 256 CTAs
        -> the mid stream and ffn_in (local, + a flag a CTA)."""
        tid = fresh(tid)
        par = tag_par(tag)
        pp = tid % 16
        tt = tid // 16
        c = bid * 32 + 2 * pp
        st = ld_vec(
            stream_in, fx.min(tt, nrows - 1) * HID + c, 2
        )  # padded rows repeat the last real row
        al = ld_vec(alpha_a, c, 2)
        be = ld_vec(beta_a, c, 2)
        wf = ld_i32(w_pffn, c // 2)
        if (
            bid // CPR
        ) == rank:  # the block owner: 8 partials in rank order -> bf16 -> every rank
            specs = [
                (
                    own_peer + fx.Int64(play["opb"][0]),
                    ((par * W + src) * S + tt) * (CPR * 16) + (bid % CPR) * 16 + pp,
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
            wr = bf16_pair(a0, a1)
            for pb in pl:
                mb_put_sys(
                    tag,
                    pb + fx.Int64(play["orb"][0]),
                    (par * S + tt) * (HID // 2) + bid * 16 + pp,
                    wr,
                )
        stamp(TL, tls, tid, 15)
        got = mb_poll(
            tag,
            health,
            [
                (
                    own_peer + fx.Int64(play["orb"][0]),
                    (par * S + tt) * (HID // 2) + bid * 16 + pp,
                    1,
                )
            ],
            REGION_IDS["orb"],
            cm=CM_SYS,
        )
        o0 = bf_lo(got[0][0])
        o1 = bf_hi(got[0][0])
        stamp_dep(TL, tls, tid, 29, got[0][0])  # orr arrived
        a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
        a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
        b0 = be[0].bitcast(fx.Float32) * o0
        b1 = be[1].bitcast(fx.Float32) * o1
        q1 = butterfly(o0 * o0 + o1 * o1, (1, 2, 4, 8))
        q2 = butterfly(a0 * a0 + a1 * a1, (1, 2, 4, 8))
        q3 = butterfly(a0 * b0 + a1 * b1, (1, 2, 4, 8))
        q4 = butterfly(b0 * b0 + b1 * b1, (1, 2, 4, 8))
        if pp == 0:
            put4_b(tag, scratch, "ssa4", tt, bid, q1, q2, q3, q4)
        tok_rstd2_b(
            scratch,
            "ssa4",
            tag,
            health,
            tid,
            lane,
            rsl,
            POST_EPS,
            PRE_EPS,
            bid,
            rsl + 2 * S,
            tls,
        )
        stamp(TL, tls, tid, 16)
        r1 = fx.ptr_load(rsl + tt)
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
        r2 = fx.ptr_load(rsl + (S + tt))
        bo.buffer_store(
            bf16_pair((s0 * r2) * bf_lo(wf), (s1 * r2) * bf_hi(wf)),
            rsrc(scratch + fx.Int64(lay["xinb"][0])),
            (tt * HID + c) // 2,
            cache_modifier=CM_DEV,
        )
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xinbf"][0]), bid, fx.Int32(1))

    @traced
    def stage_g_b(
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
        rsl,
        tls,
    ):
        """#1, CTA b = hidden columns 32b .. on every rank, thread = (token, column pair): post5's y of these
        columns (all-gathered by rank b / 32), post_feedforward_layernorm (input-scaled, 1e-8), gated residual on
        the replicated mid stream, next input_layernorm, one ONEPASS round -> the full stream (stream_out; also
        sfull_out when gather), xn (local, + a flag a CTA)."""
        tid = fresh(tid)
        par = tag_par(tag)
        pp = tid % 16
        tt = tid // 16
        c = bid * 32 + 2 * pp
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
                    (par * S + tt) * (HID // 2) + bid * 16 + pp,
                    1,
                )
            ],
            REGION_IDS["yb"],
            cm=CM_SYS,
        )
        stamp(TL, tls, tid, 17)
        yw = got[0][0]
        z0 = bf16_round(bf_lo(wpf) * bf_lo(yw))
        z1 = bf16_round(bf_hi(wpf) * bf_hi(yw))
        a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
        a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
        b0 = be[0].bitcast(fx.Float32) * z0
        b1 = be[1].bitcast(fx.Float32) * z1
        q1 = butterfly(z0 * z0 + z1 * z1, (1, 2, 4, 8))
        q2 = butterfly(a0 * a0 + a1 * a1, (1, 2, 4, 8))
        q3 = butterfly(a0 * b0 + a1 * b1, (1, 2, 4, 8))
        q4 = butterfly(b0 * b0 + b1 * b1, (1, 2, 4, 8))
        if pp == 0:
            put4_b(tag, scratch, "ssg4", tt, bid, q1, q2, q3, q4)
        tok_rstd2_b(
            scratch,
            "ssg4",
            tag,
            health,
            tid,
            lane,
            rsl,
            POST_EPS,
            PRE_EPS,
            bid,
            rsl + 2 * S,
        )
        stamp(TL, tls, tid, 23)
        r1 = fx.ptr_load(rsl + tt)
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
        r2 = fx.ptr_load(rsl + (S + tt))
        xw = bf16_pair((s0 * r2) * bf_lo(wi), (s1 * r2) * bf_hi(wi))
        bo.buffer_store(
            xw,
            rsrc(scratch + fx.Int64(lay["xnb"][0])),
            (tt * HID + c) // 2,
            cache_modifier=CM_DEV,
        )
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xnbf"][0]), bid, fx.Int32(1))

    # ------------------------------------------------------------ bnd 3: column-group boundaries (#20)
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
        tls=None,
        sa=None,
        sb=None,
    ):
        """-> (r1, r2) of this thread's token from the 4 ONEPASS partials of every CTA of the token group."""
        tg = bid // 32
        cb = bid % 32
        base = scratch + fx.Int64(lay[region][0])
        qs = [wave_sum(q) for q in (q1, q2, q3, q4)]
        if lane == 0:
            for k in range_constexpr(4):
                fx.ptr_store(qs[k], pr + (512 + wave * 4 + k))
        gpu.barrier()
        if (
            tid < 16
        ):  # token tq = tid / 4, partial k = tid % 4: the token's 2 waves in order
            v = fx.ptr_load(pr + (512 + (2 * (tid // 4)) * 4 + tid % 4)) + fx.ptr_load(
                pr + (512 + (2 * (tid // 4) + 1) * 4 + tid % 4)
            )
            mb_put(tag, base, bid * 16 + tid, v)
        if const_expr(sa is not None):
            stamp(TL, sa[0], tid, sa[1])
        src = tid // 16  # (column group, token, partial) of the group, one a thread
        got = mb_poll(
            tag,
            health,
            [(base, (tg * 32 + src) * 16 + tid % 16, 1)],
            REGION_IDS[region],
        )
        fx.ptr_store(got[0][0].bitcast(fx.Float32), pr + tid)
        gpu.barrier()
        if const_expr(sb is not None):
            stamp(TL, sb[0], tid, sb[1])
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
        tls,
    ):
        """#11 with the column-group mapping: the block owners (b / 32 == rank, block b = 32 columns, all tokens)
        reduce + all-gather as in bnd 2; CTA (tg, cb) then does the boundary of tokens 4 tg .. x columns 256 cb .."""
        tid = fresh(tid)
        par = tag_par(tag)
        if (
            (bid // CPR) == rank
        ):  # the block owner: 8 partials in rank order -> bf16 -> every rank (as bnd 2)
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
        stamp(TL, tls, tid, 15)
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
        stamp_dep(TL, tls, tid, 29, got[0][0])  # orr arrived
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
            tls,
            (tls, 30),
            (tls, 31),
        )
        stamp(TL, tls, tid, 16)
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
        rocdl.s_waitcnt(0)
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
        tls,
    ):
        """#1 with the column-group mapping (CTA (tg, cb): tokens 4 tg .. x columns 256 cb ..)."""
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
        stamp(TL, tls, tid, 17)
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
        stamp(TL, tls, tid, 23)
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
        rocdl.s_waitcnt(0)
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
        tls = misc.tls.ptr
        rt = {"rid": rl.rid.ptr, "rw": rl.rw.ptr, "sel": rl.sel.ptr}
        pl = list(peers)
        own_peer = sel_peer(pl, rank)
        stamp(TL, tls, tid, 0)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        if const_expr(on("O5") and (FA is None or not key.attn_olate)):
            o_ops = o_loads(
                w_o, bid, lane, wave
            )  # S4 #10: with FA, the o_proj weights land during the attention
        if const_expr(
            FA is not None
        ):  # S4 #8: attn_o is the fragment's descriptor; the real pointer comes back
            attn_o = FA.run(
                attn_o, tag, health, tid, bid, lane, wave, sl.fa_.peek(), True, layer
            )
            if const_expr(FA.AOT):  # ng polls the tagged rows (no all-rows wait)
                attn_o, ao_t = attn_o
        if const_expr(on("O5") and FA is not None and key.attn_olate):
            o_ops = o_loads(w_o, bid, lane, wave)
        if const_expr(on("ng")):
            stage_ng(
                attn_o,
                attn_g,
                g_stride,
                scratch,
                tag,
                tid,
                bid,
                nrows,
                (ao_t, health) if (FA is not None and FA.AOT) else None,
            )
        stamp(TL, tls, tid, 1)
        if const_expr(key.stop >= 2 and on("O5")):
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
                tls,
            )
        stamp(TL, tls, tid, 2)
        if const_expr(key.stop >= 3):
            if const_expr(on("A5")):
                wr, wp = a_loads(w_r, w_pre, bid, tid, lane, wave)
            if const_expr(on("AB5")):
                if const_expr(BND == 3):  # #11 + #20: all 256 CTAs, column groups
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
                        tls,
                    )
                elif const_expr(BND):  # #11: all 256 CTAs
                    stage_ab_b(
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
                        rank,
                        misc.rs.ptr,
                        tls,
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
                            tls,
                        )
            stamp(TL, tls, tid, 3)
            if const_expr(key.stop >= 4 and on("A5")):
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
                    tls,
                    sa.x.ptr,
                )
        stamp(TL, tls, tid, 4)
        if const_expr(key.stop >= 5 and on("route_h")):
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
        stamp(TL, tls, tid, 5)
        flush_range(tls, tl, tid, bid, 0, 6)
        flush_range(tls, tl, tid, bid, 13, 14)
        flush_range(tls, tl, tid, bid, 15, 17)
        flush_range(tls, tl, tid, bid, 18, 23)
        flush_range(tls, tl, tid, bid, 24, 29)
        if const_expr(BND):
            flush_range(tls, tl, tid, bid, 29, 32)  # bnd AB round stamps

    # ============================================================ HB
    def pp_loads(w_pp, bid, lane, wave):
        """post_expert_proj tile b / KSP (16 of this rank's QH rows), K part b % KSP (2 / 4 parts)."""
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
    def stage_push(part, pl, tag, tid, bid, rank):
        """CTA b: columns 16b .. of the library's EP partial -> the MOEP of the rank owning that
        column quarter (b / 64; reduce-scatter, plain) + flag."""
        tid = fresh(tid)
        par = tag_par(tag)
        pd = sel_peer(pl, uniform(bid // CPR))
        for it_ in range_constexpr((8 * S + THREADS - 1) // THREADS):
            e0 = tid + THREADS * it_
            ok = e0 < 8 * S
            e = fx.min(e0, 8 * S - 1)
            t = e // 8
            p = e % 8
            w_ = ld_i32(part, (t * DM + bid * 16 + 2 * p) // 2)
            st_if(
                ok,
                w_,
                pd + fx.Int64(play["moep"][0]),
                (((par * W + rank) * BLOCKS + bid) * S + t) * 8 + p,
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
    def stage_ar(own_peer, pl, tag, health, tid, bid, rank):
        """CTA b of the column-quarter owner (b / 64 == rank): the 4 ranks' partials of columns
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
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            for pb in pl:
                mb_put_sys(
                    tag, pb + fx.Int64(play["mmf"][0]), par * BLOCKS + bid, fx.Int32(1)
                )

    @traced
    def stage_post(ppo, pl, scratch, tag, health, tid, bid, lane, wave, rank, red):
        """post_expert_proj tile b / 2 over K half b % 2 for all S tokens; the K halves are
        summed by the kh = 0 CTA -> y (this rank's hidden quarter, local) + flag."""
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
        if const_expr(BG):
            # #1: y of the tile (16 columns of this rank x every token) to every rank, tagged pairs at the global
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
                if const_expr(W == 4):
                    # 4 tiles a round; wave w: owner w % 4, tiles 4 it + 2 (w / 4) + k, 4 tokens of that owner
                    dw = uniform(wave % 4)
                    k_ = (e // 32) % 2
                    rt_ = 4 * it_ + 2 * (wave // 4) + k_
                    rp = e % 8
                    tt = dw + 4 * ((e // 8) % 4)
                else:
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
                    if const_expr(KSP == 2):
                        if kh == 1:
                            for h2 in range_constexpr(2):
                                mb_put(
                                    tag,
                                    scratch + fx.Int64(lay["py"][0]),
                                    (tile * 16 + 2 * rp + h2) * S + 16 * rt_ + tt,
                                    v[h2],
                                )
                        else:
                            got = mb_poll(
                                tag,
                                health,
                                [
                                    (
                                        scratch + fx.Int64(lay["py"][0]),
                                        (tile * 16 + 2 * rp + h2) * S + 16 * rt_ + tt,
                                        1,
                                    )
                                    for h2 in range(2)
                                ],
                                REGION_IDS["py"],
                            )
                            y0 = v[0] + got[0][0].bitcast(fx.Float32)
                            y1 = v[1] + got[1][0].bitcast(fx.Float32)
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
                    else:
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
                                        (((q - 1) * NTILES + tile) * 16 + 2 * rp + h2)
                                        * S
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
            rocdl.s_waitcnt(0)
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
        tls,
    ):
        """CTA j < S/4: token t = rank S/4 + j, all 8192 columns: post_feedforward_layernorm
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
                if const_expr(
                    BND
                ):  # bnd 1: the mid stream is replicated (smidb [t][col])
                    v4 = ld_vec(
                        scratch + fx.Int64(lay["smidb"][0]),
                        t * HID + cq + 4 * h2,
                        4,
                        CM_DEV,
                    )
                else:
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
        stamp(TL, tls, tid, 17)
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
        stamp_dep(TL, tls, tid, 23, sv[0].bitcast(fx.Int32))
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
        rocdl.s_waitcnt(0)
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
        if const_expr(NQT * 32 == G.NQTILES):
            return [
                bf_tile_load(
                    w_qkvg, (NQT * gq + n) * 16, HID, k8 * (HID // 8) + wave * KCH, lane
                )
                for n in range(n0, n1)
            ]
        return [
            bf_tile_load(
                w_qkvg,
                fx.min(NQT * gq + n, G.NQTILES - 1) * 16,
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
        tls,
    ):
        tid = fresh(tid)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        t = lane % 16
        k8 = bid % 8
        gq = bid // 8
        if const_expr(
            BG
        ):  # #1 / #3: xn is local; the 32 CTAs of this K eighth wrote its columns
            if tid < 32:
                if const_expr(BND == 3):
                    fq = (tid // 4) * 32 + 4 * k8 + tid % 4
                else:
                    fq = k8 * 32 + tid
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
        stamp(TL, tls, tid, 14)
        if const_expr(BG):
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
            # R2 #4: the MFMAs of both row tiles first (red holds both), then one reduction round
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
                    if const_expr(NQT * 32 == G.NQTILES):
                        bo.buffer_store(
                            w2,
                            rsrc(g_out),
                            ((16 * rt_ + tt) * QROWS + row - QKV_ROWS) // 2,
                        )
                    else:
                        st_if(
                            row < G.QKVG_ROWS,
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
        """head_finish of head hh (uniform), token tt (lanes 0-31 / 32-63 of a wave: two tokens) with the vLLM flags.
        VQK: as vLLM's fused_muse_spark_qknorm_rope_kv_insert (weightless qk-norm, GPT-J RoPE with fp32 cos / sin,
        all fp32, one RNE bf16 rounding; no qk_scale). VKV: one bf16 dword a lane into the LBHNC cache, element
        (slot >> 4) * bstride + (kv * 16 + (slot & 15)) * 128 + 64 isv + 2p, bstride = v_cache (a value); each half
        wave stores with its own wave-uniform 64-bit block base (readlane of its slot); slot < 0 and tt >= nrows
        skipped. Rows tt >= nrows are not real rows (their q is written like the stock finish, their KV is not)."""
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
        if const_expr(VQK):
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
        else:
            n0 = bf16_round(x0 * rs)
            n1 = bf16_round(x1 * rs)
            n0 = isq.select(bf16_round(n0 * fx.Float32(QK_SCALE)), n0)
            n1 = isq.select(bf16_round(n1 * fx.Float32(QK_SCALE)), n1)
            if const_expr(key.next_rope):
                ps = ld_i32(pos, fx.min(tt, nrows - 1) * 2)
                c = bf16_round(ld_f32(cos_sin, ps * HD + p))
                sn = bf16_round(ld_f32(cos_sin, ps * HD + HD // 2 + p))
                r0 = bf16_round(bf16_round(n0 * c) - bf16_round(n1 * sn))
                r1 = bf16_round(bf16_round(n1 * c) + bf16_round(n0 * sn))
                n0 = r0
                n1 = r1
        o0 = isv.select(x0, n0)
        o1 = isv.select(x1, n1)
        if isq:
            bo.buffer_store(bf16_pair(o0, o1), rsrc(q_out), (tt * QROWS) // 2 + row0)
        else:
            kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
            slot = ld_i32(slots, fx.min(tt, nrows - 1) * 2)
            if const_expr(VKV):
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
            else:
                if const_expr(VQK):  # vLLM: the fp8 of the bf16-rounded value
                    o0 = bf16_round(o0)
                    o1 = bf16_round(o1)
                w8 = fp8x2(o0, o1)
                w8n = xshfl(w8, 1)
                if ((p % 2) == 0) & (tt < nrows):
                    off = (slot * (HKV * HD) + kv * HD + 2 * p) // 4
                    dw = w8 | (w8n << 16)
                    if isv:
                        bo.buffer_store(dw, rsrc(v_cache), off)
                    else:
                        bo.buffer_store(dw, rsrc(k_cache), off)

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
        """CTA hh < 40 finishes head hh (32 q, 4 k, 4 v) of the next layer for all S tokens."""
        tid = fresh(tid)
        hh = bid
        p = tid % 32
        if const_expr(VQK or VKV):
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
        else:
            for tg in range_constexpr(NT):
                tt = 16 * tg + tid // 32
                if bid < HQ + 2 * HKV:
                    row0 = (hh * HD) // 2 + p
                    got = mb_poll(
                        tag,
                        health,
                        [
                            (
                                scratch + fx.Int64(lay["qkv"][0]),
                                tt * (QKV_ROWS // 2) + row0,
                                1,
                            )
                        ],
                        REGION_IDS["qkv"],
                    )
                    x0 = bf_lo(got[0][0])
                    x1 = bf_hi(got[0][0])
                    ss = butterfly(x0 * x0 + x1 * x1, (1, 2, 4, 8, 16))
                    rs = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS))
                    isv = hh >= HQ + HKV
                    isq = hh < HQ
                    n0 = bf16_round(x0 * rs)
                    n1 = bf16_round(x1 * rs)
                    n0 = isq.select(bf16_round(n0 * fx.Float32(QK_SCALE)), n0)
                    n1 = isq.select(bf16_round(n1 * fx.Float32(QK_SCALE)), n1)
                    if const_expr(key.next_rope):
                        ps = ld_i32(pos, fx.min(tt, nrows - 1) * 2)
                        c = bf16_round(ld_f32(cos_sin, ps * HD + p))
                        sn = bf16_round(ld_f32(cos_sin, ps * HD + HD // 2 + p))
                        r0 = bf16_round(bf16_round(n0 * c) - bf16_round(n1 * sn))
                        r1 = bf16_round(bf16_round(n1 * c) + bf16_round(n0 * sn))
                        n0 = r0
                        n1 = r1
                    o0 = isv.select(x0, n0)
                    o1 = isv.select(x1, n1)
                    if isq:
                        bo.buffer_store(
                            bf16_pair(o0, o1), rsrc(q_out), (tt * QROWS) // 2 + row0
                        )
                    else:
                        w8 = fp8x2(o0, o1)
                        w8n = xshfl(w8, 1)
                        kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
                        slot = ld_i32(slots, fx.min(tt, nrows - 1) * 2)
                        if ((p % 2) == 0) & (tt < nrows):
                            off = (slot * (HKV * HD) + kv * HD + 2 * p) // 4
                            dw = w8 | (w8n << 16)
                            if isv:
                                bo.buffer_store(dw, rsrc(v_cache), off)
                            else:
                                bo.buffer_store(dw, rsrc(k_cache), off)

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
        tls = misc.tls.ptr
        pl = list(peers)
        own_peer = sel_peer(pl, rank)
        stamp(TL, tls, tid, 0)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        if const_expr(on("post5")):
            ppo = pp_loads(w_pp, bid, lane, wave)
        if const_expr(
            (not key.hm) and on("ar5")
        ):  # the library-MoE push (not in the HM build)
            stage_push(part, pl, tag, tid, bid, rank)
        stamp(TL, tls, tid, 6)
        if const_expr(on("ar5")):
            if bid // CPR == rank:
                stage_ar(own_peer, pl, tag, health, tid, bid, rank)
        stamp(TL, tls, tid, 7)
        if const_expr(on("post5")):
            stage_post(
                ppo, pl, scratch, tag, health, tid, bid, lane, wave, rank, sf.red.ptr
            )
        stamp(TL, tls, tid, 8)
        if const_expr(on("QKV5")):
            qo = q_loads(w_qkvg, bid, lane, wave, 0, QPF)
        if const_expr(on("G5")):
            if const_expr(BND == 3):  # #1 + #20: all 256 CTAs, column groups
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
                    tls,
                )
            elif const_expr(BND == 2):  # #1: all 256 CTAs
                stage_g_b(
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
                    misc.rs.ptr,
                    tls,
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
                        tls,
                    )
        stamp(TL, tls, tid, 9)
        if const_expr(on("QKV5")):
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
                tls,
            )
        stamp(TL, tls, tid, 10)
        if const_expr(on("finish5")):
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
        stamp(TL, tls, tid, 11)
        if const_expr(on("gather5") and not BG):  # bnd 2 / 3: G wrote sfull_out itself
            if (gather != 0) & (bid < nrows):
                stage_gather(own_peer, sfull_out, tag, health, tid, bid)
        stamp(TL, tls, tid, 12)
        flush_range(tls, tl, tid, bid, 6, 13)
        flush_range(tls, tl, tid, bid, 14, 15)
        flush_range(tls, tl, tid, bid, 17, 18)
        flush_range(tls, tl, tid, bid, 23, 24)

    # kernels and launchers: peer0 .. peer3 at W = 4 (the TP4 signature), peer0 .. peer7 at W = 8
    if W == 4:

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
                layer,
                rank,
                so_stride,
                so_off,
                gather,
                nrows,
            ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)
    else:

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


def build_from_env(S=32, **over):
    kw = dict(
        S=S,
        W=over.pop("W", 4),
        next_rope=over.pop("next_rope", True),
        qkv_pf=int(os.environ.get("MK_QKV_PF", "3")),
        timeline=os.environ.get("MK_TIMELINE", "0") == "1",
        stop=int(os.environ.get("MK5_STOP", "99")),
        hm=os.environ.get("MK5_HM", "1") == "1",
        only=os.environ.get("MK5_ONLY", ""),
    )
    W_ = kw["W"] if "W" not in over else over["W"]
    # MK5_BND: the TEP8 c = 4 boundaries (W = 8, S = 32 only; other builds ignore it)
    # default at W = 8, S = 32: #11 + #1 with the column-group one-hop norms (#20); MK5_BND=2: the reducer-round
    # version, MK5_BND=0: the token-owner boundaries (shipped)
    if int(os.environ.get("MK5_BND", "3")) and W_ == 8 and S == 32:
        kw["bnd"] = int(os.environ.get("MK5_BND", "3"))
    if (
        os.environ.get("MK5_QR1", "1") == "1" and W_ == 8 and S == 32
    ):  # default on (R2 #4, bit-identical)
        kw["qr1"] = True
    if (
        os.environ.get("MK_FUSED_ATTN", "0") == "1" and W_ == 8
    ):  # the caller passes attn= the layer's type
        kw["attn_olate"] = (
            os.environ.get("MK_FA_OLATE", "1") == "1"
        )  # S4: #10 (weights during the attention) did not help
    else:
        over.pop("attn", None)
    # vLLM flags (default off): MK_VLLM=1 turns on all three; MK_MULT1 / MK_VQK / MK_KVL set them one by one
    vl = os.environ.get("MK_VLLM", "0") == "1"
    for name, env, conv in (
        ("mult1", "MK_MULT1", lambda v: v == "1"),
        ("vqk", "MK_VQK", lambda v: v == "1"),
        ("kvl", "MK_KVL", str),
    ):
        v = os.environ.get(env)
        if v is not None:
            kw[name] = conv(v)
        elif vl:
            kw[name] = "lbhnc" if name == "kvl" else True
    kw.update(over)
    return Build(**kw)


def build_vllm(S=32, W=8, next_rope=True, **over):
    """HA / HB for vLLM main (TEP8, S = 32 / 64 / 96 / 128). No env reads: the shipped TEP8 knobs (qkv_pf 3, hm, at
    S = 32 bnd 3 + qr1, else the token-owner boundaries; no fused attention) plus the vLLM flags (mult1, vqk,
    kvl = "lbhnc"). The module-level MK5_XCM / MK5_YCM / MK5_ACM / MK5_NOINV env reads stay (defaults = shipped)."""
    assert W == 8 and S in (32, 64, 96, 128), (S, W)
    kw = dict(
        S=S,
        W=W,
        next_rope=bool(next_rope),
        qkv_pf=3,
        timeline=False,
        stop=99,
        hm=True,
        only="",
        bnd=3 if S == 32 else 0,
        qr1=S == 32,
        attn="",
        attn_olate=False,
        mult1=True,
        vqk=True,
        kvl="lbhnc",
    )
    if over.get("attn"):
        # the fused verify attention (vLLM's bf16 cache): o_proj weights after it, as SGLang ships
        kw["attn_olate"] = True
    kw.update(over)
    return Build(**kw)
