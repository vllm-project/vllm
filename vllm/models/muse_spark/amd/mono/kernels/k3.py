# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501, SIM102, SIM108

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/k3.py):
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
"""K3 (milestone 3): the decode layer's K2 launch = everything between two attention
cores: o-norm x gate -> o_proj -> in-kernel all-reduce -> post-attention norm + gated
residual + pre-FFN norm -> the M2 MoE half (k2full) -> the next layer's qkv+gate GEMV
-> qk-norm (x qk_scale on q) -> RoPE (SWA layers) -> fp8 KV-cache write -> q, gate out.
Stages added to k2full: O (o_proj, partial push), AB (attention boundary on CTA b's 32
hidden columns), QKV (K split in eighths, 9 tiles a CTA, head finishers).

(k2full docstring follows)
K2-full (milestone 2): the whole MoE half of a Muse Spark layer for one rank of a
TP = EP = W group, one persistent FlyDSL launch a layer, 256 CTAs x 512 threads,
S in {8, 16} rows. Every region, slot count and tag is laid out for WMAX = 4 ranks;
W (2 or 4) is a build constant (rank r owns experts [r E/W, (r+1) E/W)).

    x = ffn_in [S, 8192] bf16, stream [S, 8192] fp32 (both replicated)
      A  router + pre_expert_proj, sharded (``shard``; else replicated): rank r computes
         the logits of experts [r 256/W ..) and h_raw columns [r 4096/W ..), K split in
         quarters over 4-CTA groups; logit partials (tagged pairs) and h_raw (plain bf16 +
         one flag a CTA) are pushed to every rank's peer buffer (system scope)
      B  route (every CTA of every rank, identical inputs -> identical top-8)
      C  gate_up, D down, E post-norm + weighted sum: the M1 K2-core (moe_block.py)
      E' EP all-reduce: CTA b pushes its [S x 16] partial tile into slot [parity][src]
         of every rank (tagged bf16 pairs), polls the W slots of its own columns, sums
         them in rank order (fp32) -> m (bf16, local) + a local flag
      F  post_expert_proj, sharded (rank r: output rows [r 8192/W ..)): weights in flight
         before the all-reduce is waited for; y tiles pushed to every rank (tagged)
      G  CTA b owns hidden columns 32b .. 32b+32 on every rank: post_feedforward_layernorm
         (input-scaled, eps 1e-8), gated residual (alpha / beta from gate / 0.3) on the
         fp32 stream, the next layer's input_layernorm -> new stream (fp32) + next
         attention input (bf16); the two per-token reductions are intra-GPU hops.

Peer regions are double buffered by tag parity; tag = epoch * 64 + layer + 1 with the
epoch bumped in lockstep by every rank once per forward.
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
    fp8x2,
    fresh,
    hw_exp2,
    hw_rsq,
    hw_sqrt,
    lanes_below,
    ld_f32,
    ld_i32,
    ld_i32_if,
    ld_vec,
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
    stamp_dep,
    stamp_flush,
    traced,
    uniform,
    wave_incl_scan,
    wave_sum,
    xred,
    xshfl,
)

HID = 8192
DM = 4096
INTER = 4096
E = 256
TOPK = 8
WMAX = (
    8  # largest TP = EP group a build supports (regions are sized from the build's W)
)
MULT = 0.17677669529663687
ROUTE_EPS = 1e-15
PRE_EPS = 1e-5
POST_EPS = 1e-8
LOG2E = 1.0 / math.log(2.0)
XPAD = 8
HROW = DM + XPAD
AROW = INTER + XPAD
KCH = 128
FCH = 256
XQ = HID // 4
XQROW = XQ + XPAD
NG = 64  # stage-A groups of 4 CTAs (K quarters)
TL_POINTS = 16
REGION_IDS = {
    "logit": 1,
    "hf": 2,
    "a": 3,
    "z": 4,
    "ss": 5,
    "px": 6,
    "moep": 7,
    "mf": 8,
    "py": 9,
    "y": 10,
    "ss1": 11,
    "ss2g": 12,
}
HEALTH_WORDS = 8 + 8 * BLOCKS
HQ, HKV, HD = 32, 4, 64  # per rank (TP4); QKVG rows: q 2048 | k 256 | v 256 | gate 2048
QROWS, KROWS = HQ * HD, HKV * HD
QKV_ROWS = QROWS + 2 * KROWS  # 2560
QKVG_ROWS = QKV_ROWS + QROWS  # 4608
OK_K = 2048  # o_proj K a rank
# (the module-level values above are the TP4 ones, kept for the host side; builds use geom(W))


def bf_tile_load_mma(wbase, row16, ldk, k, lane, ok=None, nbytes=None, cm=CM_NT):
    """16 rows (row16 ..) x 128 K (k ..) of a row-major bf16 [*, ldk] weight, loaded straight in the
    16x16x32 bf16 MFMA A-operand layout: element qq (of 4) is the operand of MFMA step qq (lane l = row
    l % 16, K 32 qq + 8 (l / 16) .. + 8), 16 rows x 64 B an instruction. The same operand values as
    bf_tile_load + bf_mma_operand (bit-identical results) without the 8 ds_bpermute a step.
    ok (traced bool, optional): when false no memory access (zeros); nbytes must then bound the buffer."""
    rowm = row16 + lane % 16
    kk = k + 8 * (lane // 16)
    if ok is None:
        return [
            ld_vec(wbase, (rowm * ldk + kk + 32 * qq) // 2, 4, cm) for qq in range(4)
        ]
    return [
        ld_vec_if(ok, wbase, (rowm * ldk + kk + 32 * qq) // 2, 4, nbytes, cm)
        for qq in range(4)
    ]


@dataclass(frozen=True)
class Geom:
    """Every world-size dependent constant of K3 (and its layouts) for a TP = EP group of W."""

    W: int
    HQ: int  # q heads a rank (128 / W)
    HKV: int  # kv heads a rank (16 / W)
    QROWS: int
    KROWS: int
    QKV_ROWS: int
    QKVG_ROWS: int
    OK_K: int  # o_proj K a rank (8192 / W)
    KQ: int  # stage-A K split (CTAs a group): 4 at W <= 4, 8 at W = 8
    NG: int  # stage-A groups (BLOCKS / KQ)
    XQ: int  # stage-A K a CTA (HID / KQ)
    NQT: int  # qkvg 16-row tiles a CTA group (32 groups x 8 K eighths; the tail of the last group padded)
    NQTILES: int  # qkvg 16-row tiles a rank


def geom(W):
    assert W in (2, 4, 8) and W <= WMAX, W
    hq, hkv = 128 // W, 16 // W
    qr, kr = hq * HD, hkv * HD
    qkv = qr + 2 * kr
    qkvg = qkv + qr
    kq = 4 if W <= 4 else 8
    nqtiles = qkvg // 16
    return Geom(
        W=W,
        HQ=hq,
        HKV=hkv,
        QROWS=qr,
        KROWS=kr,
        QKV_ROWS=qkv,
        QKVG_ROWS=qkvg,
        OK_K=8192 // W,
        KQ=kq,
        NG=BLOCKS // kq,
        XQ=HID // kq,
        NQT=-(-nqtiles // 32),
        NQTILES=nqtiles,
    )


QK_SCALE = 5.4730064863875
REGION_IDS.update({"orr": 24, "mrr": 25, "ssa4p": 26})
REGION_IDS.update(
    {
        "op": 13,
        "ssa1": 14,
        "ssa2": 15,
        "xf": 16,
        "xnf": 17,
        "qx": 18,
        "qkv": 19,
        "ssa4": 20,
        "ssg4": 21,
    }
)


@dataclass(frozen=True)
class Build:
    S: int = 8
    W: int = 4  # ranks in the TP = EP group
    shard: bool = (
        True  # router / pre_expert_proj / post_expert_proj sharded over the ranks
    )
    pf_gu: int = 2
    pf_dn: int = 2
    gusplit: bool = False
    ac_cap: int = 0
    pp_early: bool = True
    next_rope: bool = True  # the next layer has RoPE (SWA) or not (full, NoPE)
    qkv_pf: int = 9  # qkvg chunk-tiles issued before stage G (of 9)  # post_expert_proj weights issued before the final stage (else after its push)
    timeline: bool = False
    stop: int = 99
    onepass: bool = (
        False  # M7: attention boundary and stage G with one token-reduction round each
    )
    eplb: int = 0  # M7: replica expert slots a rank (0 = plain EP); see mk/eplb.py
    xrs: bool = False  # TP8: o_proj and EP all-reduces as reduce-scatter to the block owner + all-gather (tp8 probe)
    unr: int = -1  # MoE task-loop ring (unroll 3, fixed slots; PF 2): 1 on, 0 off, -1 auto (on at W = 8, S = 8)
    only: str = (
        ""  # ablation: stage mask (mk/portion.py; "" = every stage, the shipped kernel)
    )
    # #12 (W = 8, XRS, ONEPASS): the AB block owners compute the 4 norm partials of their block and send them with the
    # all-gather (peer region ssa4p); the per-CTA put4 hop goes away (bit-identical to the plain build)
    p12: bool = False
    s3q: int = 0  # W = 8 only: stage QKV bits (0 = the TP4-shape stage QKV + head finishers); see stage_qkv8
    s3o: int = 0  # W = 8 only: stage O bits: 1 = o_proj weights in the MFMA operand layout (bf_tile_load_mma)
    attn: str = ""  # S4 (MK_FUSED_ATTN=1): this layer's verify attention fused at the head ("swa" / "full"); "" = K1 outside
    attn_olate: bool = False  # S4 #10 A/B: issue the o_proj weights after the fused attention instead of before
    # vLLM layouts of the MXFP4 experts, read in place (mk/shuffle.py): w13 / w2 in aiter shuffle_weight((16, 16)),
    # w13s / w2s in aiter e8m0_shuffle (2-D view [E N, K / 32]). False: the native checkpoint layouts
    shuffled_w: bool = False
    shuffled_s: bool = False
    # vLLM (first PR) build flags; the defaults are the SGLang kernel (same binaries as before the flags)
    mult1: bool = False  # router MULT = 1: vLLM folds output_multiplier into the fp32 router weight (model.py:281-302)
    # q / k finish as vLLM's fused_muse_spark_qknorm_rope_kv_insert: q without qk_scale (vLLM puts it into the softmax
    # scale), qk-norm and RoPE in fp32 with fp32 cos / sin, one bf16 rounding at the end (W = 8, s3q FUSE only)
    vqk: bool = False
    # KV store: "" = fp8 E4M3 K and V buffers [slot][H][64] (SGLang); "lbhnc" = bf16 into vLLM's per-layer LBHNC cache
    # [block][H][16][K 64 | V 64] (block size 16): k_cache = the layer's cache, v_cache = its block stride in elements,
    # slots = the layer's slot_mapping (int64), slot < 0 skipped, 64-bit block base (W = 8, s3q FUSE only)
    kvl: str = ""


def _lay(items):
    out, off = {}, 0
    for name, n in items:
        off = (off + 255) // 256 * 256
        out[name] = (off, n)
        off += n
    return out, (off + 4095) // 4096 * 4096


def scratch_layout(S, W=4):
    G = geom(W)
    P = S * TOPK
    return _lay(
        [
            ("px", G.NG * G.KQ * 64 * S * 8),  # [group][K part][row < RT <= 64][t]
            ("a", P * (INTER // 2) * 8),
            ("z", P * (DM // 2) * 8),
            ("ss", P * 256 * 8),
            ("ss2", P * 256 * 8),
            ("m", S * DM * 2),  # all-reduced MoE output (bf16, plain)
            ("mf", BLOCKS * 8),  # m tile flags
            ("py", 512 * 16 * S * 8),  # post-proj K-half partials
            (
                "ss1",
                (S * BLOCKS + S) * 8,
            ),  # post-FFN norm partial sums of squares [t][CTA], then rstd[t]
            (
                "ss2g",
                (S * BLOCKS + S) * 8,
            ),  # next input-norm partial sums of squares [t][CTA]
            ("route", P * 8),
            ("xin", S * HID * 2),  # ffn_in (bf16 plain), CTA b's 32 columns each
            ("xf", BLOCKS * 8),  # its flags
            ("smid", S * HID * 4),  # stream after the attention residual (fp32 plain)
            ("ssa1", (S * BLOCKS + S) * 8),
            ("ssa2", (S * BLOCKS + S) * 8),
            ("xn", S * HID * 2),  # next layer's attention input (bf16 plain)
            ("xnf", BLOCKS * 8),
            (
                "ssa4",
                (4 * S * BLOCKS + 2 * S) * 8,
            ),  # onepass AB: 4 partials [t][CTA][4], then (r1, r2)[t]
            ("ssg4", (4 * S * BLOCKS + 2 * S) * 8),  # onepass G
            (
                "qx",
                32 * 8 * G.NQT * 16 * S * 8,
            ),  # qkvg K-eighth partials [group][src][tile row][t]
            (
                "qkv",
                S * (G.QKV_ROWS // 2) * 8,
            ),  # q / k / v rows (tagged bf16 pairs) for the head finishers
        ]
    )


def peer_layout(S, W=4):
    """Every rank's peer buffer (IPC-shared), sized for the build's W ranks, [parity] first."""
    G = geom(W)
    return _lay(
        [
            ("logit", 2 * G.KQ * S * E * 8),  # [par][kq][t][e] tagged
            ("hraw", 2 * S * DM * 2),  # [par][t][col] bf16 plain
            ("hf", 2 * W * BLOCKS * 8),  # [par][src][CTA] tagged flags
            ("moep", 2 * W * S * (DM // 2) * 8),  # [par][src][t][pair] tagged
            ("y", 2 * S * (HID // 2) * 8),  # [par][t][pair] tagged
            (
                "op",
                2 * W * S * (HID // 2) * 8,
            ),  # o_proj partials [par][src][t][pair] tagged
            (
                "orr",
                2 * S * (HID // 2) * 8,
            ),  # xrs: the owner's reduced o_proj [par][t][pair] tagged (all-gather)
            (
                "mrr",
                2 * S * (DM // 2) * 8,
            ),  # xrs: the owner's reduced EP sum [par][t][pair] tagged (all-gather)
            (
                "ssa4p",
                2 * 4 * S * BLOCKS * 8,
            ),  # #12: the AB owners' norm partials [par][t][CTA][4] tagged (all-gather)
        ]
    )


_BUILDS = {}


def build(key: Build):
    if key in _BUILDS:
        return _BUILDS[key]
    # FlyDSL 0.3.2 keys its disk cache on sources + *scalar* closure values; the Build
    # dataclass is not a scalar, so every knob read as key.x would share one binary.
    # Referencing this string from the launcher puts the whole key into the cache key.
    _KEYSTR = repr(key)
    S = key.S
    W = key.W
    assert W in (4, 8) and S in (8, 16), "K3: the attention side is TP4 or TP8"
    G = geom(W)
    HQ, HKV, QROWS, QKV_ROWS, OK_K = G.HQ, G.HKV, G.QROWS, G.QKV_ROWS, G.OK_K
    KQ, NG, XQ, NQT = G.KQ, G.NG, G.XQ, G.NQT
    XQROW = XQ + XPAD
    CA = XQ // KCH // WAVES  # stage-A pre-proj chunks a wave (2 / 1)
    CO = OK_K // KCH // WAVES  # o_proj chunks a wave (2 / 1)
    QPF = min(key.qkv_pf, NQT)
    MASKW = (1 << W) - 1  # eplb holder mask bits
    SB = 8 + W  # eplb flist packing: e | mask << 8 | (replica slot + 1) << SB
    SH = W if key.shard else 1  # shard count of the router / pre / post GEMVs
    EL = E // W  # local experts
    RB = key.eplb  # replica slots
    NU = EL + RB  # weight slots: own experts 0 .. EL-1, replicas EL .. NU-1
    P = S * TOPK
    RROUNDS = (S + WAVES - 1) // WAVES
    AC_CAP = key.ac_cap if key.ac_cap > 0 else S
    lay, _ = scratch_layout(S, W)
    play, _ = peer_layout(S, W)
    PF_GU, PF_DN = key.pf_gu, key.pf_dn
    TL = key.timeline
    W13_BYTES = EL * 2 * INTER * (DM // 2)
    W13S_BYTES = EL * 2 * INTER * (DM // 32)
    W2_BYTES = EL * DM * (INTER // 2)
    W2S_BYTES = EL * DM * (INTER // 32)
    # stage A: a KQ-CTA group (K parts) covers RT pre-proj rows and EG router experts
    RS_PRE = DM // SH
    RT = RS_PRE // NG  # 16 / 32 / 64 (W = 8: 512 / 32 = 16)
    NT_A = RT // 16  # MFMA tiles a CTA
    OWN = RT // KQ  # rows a CTA owns after the K-part exchange
    EG = (E // SH) // NG  # router experts a group: 1 / 2 / 4 (W = 8: 1)
    TPE = THREADS // EG  # threads an expert
    KPT = XQ // TPE  # router K a thread (4 / 8 / 16; W = 8: 2)
    assert (
        NT_A >= 1
        and OWN >= 2
        and OWN % 2 == 0
        and EG >= 1
        and KPT in (2, 4)
        or KPT % 8 == 0
    ), (RT, OWN, EG, KPT)
    # stage F: post_expert_proj rows RS_POST a rank, 16-row tiles
    RS_POST = HID // SH
    NTL = RS_POST // 16
    TPC = max(1, NTL // BLOCKS)  # tiles a CTA
    KSP = max(
        1, BLOCKS // NTL
    )  # K split of a tile (2 when 4-way sharded, 4 when 8-way)
    KR = DM // KSP  # K a CTA reads
    CPW = KR // KCH // WAVES  # chunks a wave a tile
    MROW = KR + XPAD
    tag_par = lambda tag: tag & 1  # noqa: E731

    @fx.struct
    class RouteLds:
        rid: fx.Array[fx.Int32, S * TOPK, 16]
        rw: fx.Array[fx.Float32, S * TOPK, 16]
        uexp: fx.Array[fx.Int32, NU, 16]
        kof: fx.Array[fx.Int32, NU * S, 16]
        nu: fx.Array[fx.Int32, 4, 16]
        sel: fx.Array[fx.Int32, max(S * TOPK * 3, NU), 16]
        plist: fx.Array[fx.Int32, P, 16]
        pidx: fx.Array[fx.Int32, P, 16]
        np_: fx.Array[fx.Int32, 4, 16]
        rstd: fx.Array[fx.Float32, P, 16]
        uslot: fx.Array[fx.Int32, NU, 16]
        wi: fx.Array[
            fx.Int32, E, 16
        ]  # eplb: weight slot of expert e on this rank this step, -1 = not here
        gpk: fx.Array[fx.Int32, E, 16]  # eplb: expert picked by any token
        flist: fx.Array[fx.Int32, E, 16]
        ssacc: fx.Array[fx.Float32, 2 * S, 16]
        upk: fx.Array[fx.Int32, 2 * S, 16]
        upn: fx.Array[fx.Int32, 4, 16]
        tok: fx.Array[fx.Float32, WAVES * S + 8, 16]
        rs: fx.Array[fx.Float32, 2 * S + 32, 16]

    @fx.struct
    class StageA:
        x: fx.Array[fx.Int32, S * XQROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * NT_A * 64 * 4, 16]
        own: fx.Array[fx.Float32, OWN * S, 16]

    @fx.struct
    class StageC:
        h: fx.Array[fx.Int32, S * HROW // 2, 16]
        red: fx.Array[fx.Float32, 2 * (1 if key.gusplit else 2) * WAVES * 64 * 4, 16]

    @fx.struct
    class StageD:
        a: fx.Array[fx.Int32, S * AROW // 2, 16]
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]

    @fx.struct
    class StageE:
        z: fx.Array[fx.Float32, P * 16, 16]

    @fx.struct
    class StageF:
        mm: fx.Array[fx.Int32, S * MROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * TPC * 64 * 4, 16]

    YROW = OK_K + XPAD
    XQ8 = HID // 8  # qkvg K eighth
    assert CA >= 1 and CO >= 1 and NQT * 32 * 16 >= G.QKVG_ROWS
    X8ROW = XQ8 + XPAD

    @fx.struct
    class StageO:
        y: fx.Array[fx.Int32, S * YROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]

    @fx.struct
    class StageQ:
        xq: fx.Array[fx.Int32, S * X8ROW // 2, 16]
        red: fx.Array[fx.Float32, WAVES * NQT * 64 * 4, 16]
        own: fx.Array[fx.Float32, NQT * 2 * S, 16]

    FA = None
    if key.attn:  # S4: the fused attention fragment (mk/attn_frag.py), TP8 only
        from . import attn_frag

        assert W == 8 and key.attn in ("swa", "full"), key
        FA = attn_frag.make(S // 8, key.attn == "swa")
        FaLds = FA.Lds
        _KEYSTR = (
            _KEYSTR + FA.key
        )  # the fragment's source hash: the disk cache does not follow FA.run

    @fx.union
    class StageLds:
        o_: StageO
        q_: StageQ
        a_: StageA
        c_: StageC
        d_: StageD
        e_: StageE
        f_: StageF
        if FA is not None:
            fa_: FaLds

    @fx.struct
    class Misc:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    RING = (W == 8 and S == 8) if key.unr < 0 else key.unr == 1
    UNR = RING and key.pf_gu == 2
    UNRD = RING and key.pf_dn == 2
    name = (
        f"muse_k3_r{int(key.next_rope)}q{key.qkv_pf}_s{S}_w{W}_sh{int(key.shard)}_g{PF_GU}_d{PF_DN}_gs{int(key.gusplit)}_ac{key.ac_cap}_pe{int(key.pp_early)}"
        f"_t{int(TL)}_x{key.stop}_op{int(key.onepass)}_eb{key.eplb}"
        + ("_xrs" if key.xrs else "")
        + ("_u" if RING else "")
        + ("_p12" if key.p12 else "")
        + _portion.name_tag(key.only)
        + (f"_q8x{key.s3q}" if key.s3q else "")
        + (f"_o8x{key.s3o}" if key.s3o else "")
        + (f"_fa{key.attn}{int(key.attn_olate)}" if key.attn else "")
        + ("_shw" if key.shuffled_w else "")
        + ("_shs" if key.shuffled_s else "")
        + ("_m1" if key.mult1 else "")
        + ("_vqk" if key.vqk else "")
        + (f"_kv{key.kvl}" if key.kvl else "")
    )
    SHUF_W, SHUF_S = key.shuffled_w, key.shuffled_s
    RMULT = 1.0 if key.mult1 else MULT  # the router sigmoid's logit multiplier
    VQK = key.vqk
    VKV = key.kvl == "lbhnc"
    assert key.kvl in ("", "lbhnc"), key.kvl
    if (
        VQK or VKV
    ):  # the vLLM finish lives in the FUSE owners of stage_qkv8 (not in head_finish / the fused attention)
        assert W == 8 and key.s3q & 1 and not key.attn, key
    XRS = key.xrs
    ONLY = _portion.stage_set(key.only, _portion.K3_STAGES, _portion.K3_DEPS)

    def on(stage):  # stage active in this build (const at trace time)
        return ONLY is None or stage in ONLY

    CPRX = BLOCKS // W  # xrs: CTA blocks a rank owns (owner of block b = b // CPRX)

    def peers_of(pl):
        return pl[:W]

    def targets(pl, rank):
        """Ranks a sharded producer pushes to: all of them (replicated: only itself)."""
        return pl[:W] if key.shard else None

    # ------------------------------------------------------------ stage A (sharded)
    def bf_mma_operand(ops, qq, lane):
        """Coalesced bf16 weight registers (4 instructions: rows 8 blk + lane / 8, 16 B at
        128 h + 16 (lane % 8)) -> the MFMA A operand of step qq (K 32 qq + 8 g of row r)."""
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

    @traced
    def stage_a(
        x,
        w_r,
        bias,
        w_pre,
        scratch,
        peer,
        pl,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        rank,
        xl,
        red,
        own,
        tok,
        tls,
    ):
        """CTA b = (group rb = b / KQ, K part kq = b % KQ): router experts EG rb .. and
        pre-proj rows RT rb .. of this rank's shard over K part kq."""
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        lane = fresh(lane)
        g = lane // 16
        kq = bid % KQ
        rb = bid // KQ
        k0 = kq * XQ
        par = tag_par(tag)
        nx = S * XQ // 8
        # weights first (data independent), then the ffn_in quarter's 64 flags, then x
        ei = tid // TPE
        kr = (tid % TPE) * KPT
        wr = [
            ld_vec(w_r, (EG * rb + ei) * HID + k0 + kr + 4 * q, 4)
            for q in range(KPT // 4)
        ]
        if const_expr(KPT == 2):
            wr = [ld_vec(w_r, (EG * rb + ei) * HID + k0 + kr, 2)]
        bias_v = [ld_f32(bias, lane + 64 * i) for i in range(4)]
        wp = []
        for nt in range_constexpr(NT_A):
            for ci in range_constexpr(CA):
                c = wave + WAVES * ci
                wp.append(
                    bf_tile_load(w_pre, rb * RT + nt * 16, HID, k0 + c * KCH, lane)
                )
        if tid < BLOCKS // KQ:  # the CTAs owning this K part's columns (32 a CTA)
            mb_poll(
                tag,
                health,
                [(scratch + fx.Int64(lay["xf"][0]), kq * (BLOCKS // KQ) + tid, 1)],
                REGION_IDS["xf"],
            )
        gpu.barrier()
        xv = []
        for i in range_constexpr((nx + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, nx - 1)
            t = e // (XQ // 8)
            kk = (e % (XQ // 8)) * 8
            xv.append(
                ld_vec(
                    scratch + fx.Int64(lay["xin"][0]),
                    (t * HID + k0 + kk) // 2,
                    4,
                    CM_DEV,
                )
            )
        for i in range_constexpr((nx + THREADS - 1) // THREADS):
            e = tid + THREADS * i
            if e < nx:
                t = e // (XQ // 8)
                kk = (e % (XQ // 8)) * 8
                fx.ptr_store(xv[i], xl + (t * XQROW + kk) // 2)
        gpu.barrier()
        wf = []
        for q in range_constexpr(KPT // 4):
            for d in range_constexpr(4):
                wf.append(wr[q][d].bitcast(fx.Float32))
        if const_expr(KPT == 2):
            wf = [wr[0][0].bitcast(fx.Float32), wr[0][1].bitcast(fx.Float32)]
        for t in range_constexpr(S):
            pr = fx.Float32(0.0)
            for hh in range_constexpr(KPT // 8):
                xw = fx.Vector(
                    fx.ptr_load(
                        xl + (t * XQROW + kr + 8 * hh) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                for d in range_constexpr(4):
                    pr = pr + bf_lo(xw[d]) * wf[8 * hh + 2 * d]
                    pr = pr + bf_hi(xw[d]) * wf[8 * hh + 2 * d + 1]
            if const_expr(KPT == 4):
                xw2 = fx.Vector(
                    fx.ptr_load(
                        xl + (t * XQROW + kr) // 2,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                )
                for d in range_constexpr(2):
                    pr = pr + bf_lo(xw2[d]) * wf[2 * d]
                    pr = pr + bf_hi(xw2[d]) * wf[2 * d + 1]
            if const_expr(KPT == 2):
                xw1 = fx.ptr_load(xl + (t * XQROW + kr) // 2)
                pr = pr + bf_lo(xw1) * wf[0]
                pr = pr + bf_hi(xw1) * wf[1]
            pr = wave_sum(pr)
            if lane == 0:
                fx.ptr_store(pr, tok + (wave * S + t))
        gpu.barrier()
        if tid < EG * S:
            e2 = tid // S
            t = tid % S
            v = fx.Float32(0.0)
            for w in range_constexpr(WAVES // EG):
                v = v + fx.ptr_load(tok + ((e2 * (WAVES // EG) + w) * S + t))
            eg = (rank * (E // SH) if key.shard else 0) + EG * rb + e2
            idx = ((par * KQ + kq) * S + t) * E + eg
            if const_expr(key.shard):
                for pb in pl:
                    mb_put_sys(tag, pb + fx.Int64(play["logit"][0]), idx, v)
            else:
                mb_put_sys(tag, peer + fx.Int64(play["logit"][0]), idx, v)
        stamp(TL, tls, tid, 1)
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NT_A)]
        t = fx.min(lane % 16, S - 1)
        for nt in range_constexpr(NT_A):
            for ci in range_constexpr(CA):
                c = wave + WAVES * ci
                for qq in range_constexpr(4):
                    kk = c * KCH + 8 * g + 32 * qq
                    b = fx.Vector(
                        fx.ptr_load(
                            xl + (t * XQROW + kk) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    ).bitcast(fx.BFloat16)
                    accs[nt] = mfma_bf16(
                        bf_mma_operand(wp[nt * CA + ci], qq, lane).bitcast(fx.BFloat16),
                        b,
                        accs[nt],
                    )
        for nt in range_constexpr(NT_A):
            fx.ptr_store(accs[nt], red + ((wave * NT_A + nt) * 64 + lane) * 4)
        gpu.barrier()
        # thread (row pair rp, token tt) over the group's RT rows: this quarter's partial
        if tid < (RT // 2) * S:
            rp = tid // S
            tt = tid % S
            v = []
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                s_ = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    s_ = s_ + fx.ptr_load(
                        red
                        + (
                            ((w * NT_A + rr // 16) * 64 + 16 * ((rr % 16) // 4) + tt)
                            * 4
                            + rr % 4
                        )
                    )
                v.append(s_)
            owner = (2 * rp) // OWN
            if owner == kq:
                fx.ptr_store(v[0], own + ((2 * rp - kq * OWN) * S + tt))
                fx.ptr_store(v[1], own + ((2 * rp + 1 - kq * OWN) * S + tt))
            else:
                for h2 in range_constexpr(2):
                    mb_put(
                        tag,
                        scratch + fx.Int64(lay["px"][0]),
                        ((rb * KQ + kq) * 64 + 2 * rp + h2) * S + tt,
                        v[h2],
                    )
        gpu.barrier()
        # own rows: the KQ parts in order -> bf16 h_raw -> every rank's HRAW
        if tid < (OWN // 2) * S:
            rp = tid // S
            tt = tid % S
            specs, oks = [], []
            for src in range_constexpr(KQ):
                for h2 in range_constexpr(2):
                    specs.append(
                        (
                            scratch + fx.Int64(lay["px"][0]),
                            ((rb * KQ + src) * 64 + kq * OWN + 2 * rp + h2) * S + tt,
                            1,
                        )
                    )
                    oks.append(fx.Int32(src) != kq)
            got = mb_poll(tag, health, specs, REGION_IDS["px"], oks=oks)
            tot = []
            for h2 in range_constexpr(2):
                s_ = fx.Float32(0.0)
                for src in range_constexpr(KQ):
                    mine = fx.ptr_load(own + ((2 * rp + h2) * S + tt))
                    s_ = s_ + (fx.Int32(src) == kq).select(
                        mine, got[2 * src + h2][0].bitcast(fx.Float32)
                    )
                tot.append(s_)
            col = (rank * RS_PRE if key.shard else 0) + rb * RT + kq * OWN + 2 * rp
            hw = bf16_pair(tot[0], tot[1])
            idx = (par * S * DM + tt * DM + col) // 2
            if const_expr(key.shard):
                for pb in pl:
                    bo.buffer_store(
                        hw,
                        rsrc(pb + fx.Int64(play["hraw"][0])),
                        idx,
                        cache_modifier=CM_SYS,
                    )
            else:
                bo.buffer_store(
                    hw,
                    rsrc(peer + fx.Int64(play["hraw"][0])),
                    idx,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            fidx = (par * W + rank) * BLOCKS + bid
            if const_expr(key.shard):
                for pb in pl:
                    mb_put_sys(tag, pb + fx.Int64(play["hf"][0]), fidx, fx.Int32(1))
            else:
                mb_put_sys(tag, peer + fx.Int64(play["hf"][0]), fidx, fx.Int32(1))
        stamp(TL, tls, tid, 2)
        return bias_v

    @traced
    def route_poll(peer, tag, health, lane, wave):
        """This wave's tokens' 256 logits from the local peer buffer: KQ K-part partials
        each, summed in part order (identical on every CTA of every rank)."""
        par = tag_par(tag)
        specs = [
            (
                peer + fx.Int64(play["logit"][0]),
                ((par * KQ + kq) * S + fx.min(wave + WAVES * rr, S - 1)) * E
                + lane
                + 64 * i,
                1,
            )
            for rr in range(RROUNDS)
            for i in range(4)
            for kq in range(KQ)
        ]
        got = mb_poll(
            tag,
            health,
            specs,
            REGION_IDS["logit"],
            batch=min(len(specs), 32),
            cm=CM_SYS,
        )  # W = 8 (KQ = 8): 2 polls of 32
        out = []
        for rr in range_constexpr(RROUNDS):
            lgs = []
            for i in range_constexpr(4):
                base = (rr * 4 + i) * KQ
                v = got[base][0].bitcast(fx.Float32)
                for kq in range_constexpr(1, KQ):
                    v = v + got[base + kq][0].bitcast(fx.Float32)
                lgs.append(v.bitcast(fx.Int32))
            out.append(lgs)
        return out

    @traced
    def h_wait_issue(peer, w_pren, tag, health, tid, rank):
        """Every producing rank's HRAW flags, then this thread's h_raw (cols kk .. kk+8
        of every token) from the local peer buffer and its w_pre values."""
        par = tag_par(tag)
        NSRC = W if key.shard else 1
        nf = NSRC * BLOCKS
        specs, oks = [], []
        for i in range_constexpr((nf + THREADS - 1) // THREADS):
            f = fx.min(tid + THREADS * i, nf - 1)
            src = (f // BLOCKS) if key.shard else rank
            specs.append(
                (
                    peer + fx.Int64(play["hf"][0]),
                    (par * W + src) * BLOCKS + f % BLOCKS,
                    1,
                )
            )
            oks.append((tid + THREADS * i) < nf)
        mb_poll(tag, health, specs, REGION_IDS["hf"], oks=oks, cm=CM_SYS)
        gpu.barrier()
        kk = (tid % (DM // 8)) * 8
        wv = ld_vec(w_pren, kk // 2, 4)
        hv = []
        for t in range_constexpr(S):
            hv.append(
                ld_vec(
                    peer + fx.Int64(play["hraw"][0]),
                    (par * S * DM + t * DM + kk) // 2,
                    4,
                    CM_SYS,
                )
            )
        return hv, wv

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

    @traced
    def stage_route(
        scratch,
        bias_v,
        lgs_r,
        tag,
        health,
        rank,
        tid,
        lane,
        wave,
        rt,
        tls=None,
        ept=None,
    ):
        """Every token's routing (wave w: tokens w, w + 8, ...), then the local union."""
        if const_expr(RB > 0):
            if tid < E:
                fx.ptr_store(fx.Int32(0), rt["gpk"] + tid)
        if const_expr(S <= WAVES):
            if wave < S:
                route_token([wave], bias_v, [lgs_r[0]], tid, lane, rt, tls)
        else:
            # S a multiple of 8: wave w routes tokens w, w + 8, .. interleaved
            route_token(
                [wave + WAVES * rr for rr in range(RROUNDS)],
                bias_v,
                lgs_r,
                tid,
                lane,
                rt,
                tls,
            )
        gpu.barrier()
        if const_expr(tls is not None):
            stamp(TL, tls, tid, 14)
        # the union U of local experts (sorted), kof[u][t]; picks list
        e0 = rank * EL
        if const_expr(RB > 0):
            eplb_assign(rt, rank, tid, lane, wave, ept)
        # parallel: picks flag their experts, a scan numbers U, picks scatter kof
        for ii in range_constexpr((NU * S + THREADS - 1) // THREADS):
            ix = tid + THREADS * ii
            if ix < NU * S:
                fx.ptr_store(fx.Int32(-1), rt["kof"] + ix)
        if tid < NU:
            fx.ptr_store(fx.Int32(0), rt["sel"] + tid)
        gpu.barrier()
        for ii in range_constexpr((P + THREADS - 1) // THREADS):
            pq = tid + THREADS * ii
            if pq < P:
                eid = slot_of(rt, fx.ptr_load(rt["rid"] + pq), e0)
                if eid >= 0:
                    fx.ptr_store(fx.Int32(1), rt["sel"] + eid)
        gpu.barrier()
        if wave == 0:
            ucarry = fx.Int32(0)
            for hh in range_constexpr((NU + 63) // 64):
                el_ = lane + 64 * hh
                f = fx.ptr_load(rt["sel"] + fx.min(el_, NU - 1))
                f = (el_ < NU).select(f, fx.Int32(0))
                incl = wave_incl_scan(f, lane) + ucarry
                slot = incl - f
                if f != 0:
                    fx.ptr_store(el_, rt["uexp"] + slot)
                if el_ < NU:
                    fx.ptr_store(slot, rt["uslot"] + el_)
                ucarry = readlane(incl, 63)
            if lane == 63:
                fx.ptr_store(ucarry, rt["nu"] + 0)
        if wave == 1:
            # local picks p = t*8 + k in (t, k) order (lane p, then p + 64, ...)
            carry = fx.Int32(0)
            for hh in range_constexpr((P + 63) // 64):
                pl_ = lane + 64 * hh
                pp = fx.min(pl_, P - 1)
                eid = slot_of(rt, fx.ptr_load(rt["rid"] + pp), e0)
                loc = ((eid >= 0) & (pl_ < P)).select(fx.Int32(1), fx.Int32(0))
                incl = wave_incl_scan(loc, lane) + carry
                if loc != 0:
                    fx.ptr_store(pl_, rt["plist"] + (incl - loc))
                if pl_ < P:
                    fx.ptr_store(incl - loc, rt["pidx"] + pp)
                carry = readlane(incl, 63)
            if lane == 63:
                fx.ptr_store(carry, rt["np_"] + 0)
        gpu.barrier()
        if const_expr(True):
            for ii in range_constexpr((P + THREADS - 1) // THREADS):
                pq = tid + THREADS * ii
                if pq < P:
                    eid = slot_of(rt, fx.ptr_load(rt["rid"] + pq), e0)
                    if eid >= 0:
                        slot = fx.ptr_load(rt["uslot"] + eid)
                        fx.ptr_store(
                            fx.Int32(pq % TOPK), rt["kof"] + (slot * S + pq // TOPK)
                        )
            gpu.barrier()

    def slot_of(rt, e, e0):
        """Weight slot of global expert e on this rank this step (-1: not computed here)."""
        if const_expr(RB > 0):
            return fx.ptr_load(rt["wi"] + fx.min(fx.max(e, 0), E - 1))
        el_ = e - e0
        return ((el_ >= 0) & (el_ < EL)).select(el_, fx.Int32(-1))

    @traced
    def eplb_assign(rt, rank, tid, lane, wave, ept):
        """Identical on every CTA of every rank (same routing, same table): an expert held by
        one rank (its owner) goes there; an expert with replicas goes, in expert order, to the
        holder with the fewest experts so far (ties: lower rank). eptab[e] = holder mask
        (bits 0-3) | (replica slot on this rank + 1) << 8 (ept: wave 0's 4 values a lane, loaded
        before the logits poll). wi[e] := weight slot here or -1. gpk was zeroed before the
        routing barrier."""
        for ii in range_constexpr((P + THREADS - 1) // THREADS):
            pq = tid + THREADS * ii
            if pq < P:
                fx.ptr_store(
                    fx.Int32(1),
                    rt["gpk"] + fx.min(fx.max(fx.ptr_load(rt["rid"] + pq), 0), E - 1),
                )
        gpu.barrier()
        if wave == 0:
            cnt = [fx.Int32(0) for _ in range(W)]
            carry = fx.Int32(0)
            for hh in range_constexpr(E // 64):
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
                    # packed: e | mask << 8 | (replica slot + 1) << 12
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

    def gu_loads(w13, w13s, rt, task, lane, wave, ok):
        """Task (u, j)'s w13 operands for this wave: gate / up row groups x the
        wave's 2 chunks of 256 K: 2 dwordx4 + 1 scale dword each. ``ok`` false:
        zeros, no memory access (a lookahead past the CTA's last task)."""
        r = lane % 16
        g = lane // 16
        task = ok.select(task, fx.Int32(0))
        u = task >> 8
        j = task & 255
        e = uniform(fx.ptr_load(rt["uexp"] + u))
        if const_expr(RB > 0):
            isr = e >= EL
            w13 = isr.select(rt["w13r"], w13)
            w13s = isr.select(rt["w13sr"], w13s)
            e = isr.select(e - EL, e)
        ops = []
        if const_expr(key.gusplit):
            # waves 0-3 the gate rows, 4-7 the up rows; wave w the chunks 4 (w % 4) + i
            rgs = [wave // 4]
            cs = [4 * (wave % 4) + i for i in range(4)]
        else:
            rgs = list(range(2))
            cs = [2 * wave + i for i in range(2)]
        for rg in rgs:
            row = e * (2 * INTER) + rg * INTER + j * 16 + r
            rowc = e * (2 * INTER) + rg * INTER + j * 16 + lane // 8
            rbase = (
                (e * (2 * INTER) + rg * INTER + j * 16)
                if const_expr(SHUF_W or SHUF_S)
                else None
            )
            for c in cs:
                if const_expr(SHUF_W):
                    # vLLM shuffle_weight((16, 16)): lane L reads the 16 B at 16 L of the 1 KB block (16 rows x
                    # K bytes 64 (2 c + q) ..): one instruction = 1 KB contiguous; fp4_mma's permlane swaps map
                    # the pieces to the MFMA lanes (mk/shuffle.kernel_w_dword / consumer_w_src)
                    wb = (rbase * (DM // 2) + (2 * c) * 1024 + lane * 16) // 4
                    ops.append(ld_vec_if(ok, w13, wb, 4, W13_BYTES, CM_NT))
                    ops.append(ld_vec_if(ok, w13, wb + 256, 4, W13_BYTES, CM_NT))
                elif const_expr(True):
                    wb = (rowc * (DM // 2) + 128 * c + 16 * (lane % 8)) // 4
                    ops.append(ld_vec_if(ok, w13, wb, 4, W13_BYTES, CM_NT))
                    ops.append(
                        ld_vec_if(ok, w13, wb + 8 * (DM // 2) // 4, 4, W13_BYTES, CM_NT)
                    )
                else:
                    wb = (row * (DM // 2) + 128 * c + 32 * g) // 4
                    ops.append(ld_vec_if(ok, w13, wb, 4, W13_BYTES, CM_NT))
                    ops.append(ld_vec_if(ok, w13, wb + 4, 4, W13_BYTES, CM_NT))
                if const_expr(SHUF_S):
                    # e8m0_shuffle: lane 16 g + r reads the dword (rbase // 32, c, K block % 4 = 2 (g % 2) + g // 2,
                    # row r): rows r, r + 16 x K blocks of 4 apart; 64 lanes = 256 B contiguous (kernel_s_dword)
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

    def fp4_mma(ops, base, bl, brow, lane, wave, acc, i, c=None, bsel=None, jpar=None):
        """Chunk c (default 2 wave + i) of one row group: 8 MFMA steps. ``bsel``
        (row base, valid count): B column n reads LDS row base + n, zero past count.
        ``jpar``: the task's 16-row group parity (j & 1), for the shuffled scales (SHUF_S)."""
        g = lane // 16
        t = fx.min(lane % 16, S - 1)
        bok = None
        if bsel is not None:
            # columns past the expert's picks read some row: their outputs are never stored
            t = fx.min(bsel[0] + lane % 16, S - 1)
        if c is None:
            c = 2 * wave + i
        sw = ops[base + 2]
        if const_expr(SHUF_S):
            # a = the dword of the lane's lower-half partner (K block % 4 = 2 (g % 2)), b = of the upper-half one
            # (2 (g % 2) + 1); the byte = 2 (g // 2) (K block + 4) + the row group parity (rows + 16)
            sa_, sb_ = permlane_swap(32, sw, sw)
            shs = (g // 2) * 16 + jpar * 8
            s0 = e8m0_f32((sa_ >> shs) & 0xFF)
            s1 = e8m0_f32((sb_ >> shs) & 0xFF)
        else:
            sh = (g % 2) * 16
            s0 = e8m0_f32((sw >> sh) & 0xFF)
            s1 = e8m0_f32((sw >> (sh + 8)) & 0xFF)
        if const_expr(SHUF_W):
            # loaded (shuffled): vector q lane 16 p + r' = row r', piece 4 q + p; MFMA lane 16 g + r wants row r
            # pieces 2 g, 2 g + 1: X0 = [V0 p0, V0 p2, V1 p0, V1 p2], X1 = [V0 p1, V0 p3, V1 p1, V1 p3] (16-lane
            # rows) = permlane32_swap then permlane16_swap, no bpermute (mk/shuffle.consumer_w_src)
            x0, x1 = [], []
            for jj in range_constexpr(4):
                a_, b_ = permlane_swap(32, ops[base][jj], ops[base + 1][jj])
                p_, q_ = permlane_swap(16, a_, b_)
                x0.append(p_)
                x1.append(q_)
            xs = [x0, x1]
        elif const_expr(True):
            # loaded: lane 8 r' + p holds row r' (+8 in the 2nd vector) bytes 16p ..;
            # MFMA lane 16 g + r wants row r pieces 2g, 2g+1 (= the uncoalesced layout)
            r = lane % 16
            src = 8 * (r % 8) + 2 * g
            hi = r >= 8
            xs = []
            if const_expr(W == 8):
                # TP8 (R5 #3): pre-mix the two vectors with a 1-lane DPP shift (inside 16-lane rows), then ONE
                # bpermute an output dword: M = odd ? ops1[l - 1] : ops0[l], N = odd ? ops0[l] : ops1[l + 1];
                # h2 = 0 reads M at src + hi, h2 = 1 reads N at src + 1 - hi (the same values as below)
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
            if const_expr(True):
                word = xs[d // 4][d % 4]
            else:
                word = ops[base + d // 4][d % 4]
            a = fp4x8_bf16(word, s0 if d < 4 else s1)
            kk = FCH * c + 64 * g + 8 * d
            if bok is not None:
                bw = fx.Vector(
                    fx.ptr_load(
                        bl + (t * brow + kk) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                bw = fx.Vector.from_elements(
                    [bok.select(bw[d2], fx.Int32(0)) for d2 in range(4)], fx.Int32
                )
                b = bw.bitcast(fx.BFloat16)
            else:
                b = fx.Vector(
                    fx.ptr_load(
                        bl + (t * brow + kk) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                ).bitcast(fx.BFloat16)
            acc = mfma_bf16(a, b, acc)
        return acc

    @traced
    def load_h_norm(hv, wv, tid, lane, wave, hl, tok):
        """Rstd per token from the full h_raw, h = bf16(h_raw rstd w) -> LDS."""
        kk = (tid % (DM // 8)) * 8
        for t in range_constexpr(S):
            ss = fx.Float32(0.0)
            for d in range_constexpr(4):
                lo = bf_lo(hv[t][d])
                hi = bf_hi(hv[t][d])
                ss = ss + lo * lo + hi * hi
            ss = wave_sum(ss)
            if lane == 0:
                fx.ptr_store(ss, tok + (wave * S + t))
        gpu.barrier()
        for t in range_constexpr(S):
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(tok + (w * S + t))
            rs = hw_rsq(tot * fx.Float32(1.0 / DM) + fx.Float32(PRE_EPS))
            outw = []
            for d in range_constexpr(4):
                a0 = bf_lo(hv[t][d]) * rs * bf_lo(wv[d])
                a1 = bf_hi(hv[t][d]) * rs * bf_hi(wv[d])
                outw.append(bf16_pair(a0, a1))
            fx.ptr_store(
                fx.Vector.from_elements(outw, fx.Int32), hl + (t * HROW + kk) // 2
            )
        gpu.barrier()

    @traced
    def gu_compute(ops, task, it, scratch, tag, rt, tid, lane, wave, hl, red):
        if const_expr(key.gusplit):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(4):
                acc = fp4_mma(
                    ops,
                    3 * i,
                    hl,
                    HROW,
                    lane,
                    wave,
                    acc,
                    i,
                    c=4 * (wave % 4) + i,
                    jpar=(task & 1) if SHUF_S else None,
                )
            par = (it & 1) * (WAVES * 64 * 4)
            fx.ptr_store(acc, red + (par + (wave * 64 + lane) * 4))
        else:
            acc_g = fx.Vector.filled(4, 0.0, fx.Float32)
            acc_u = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(2):
                acc_g = fp4_mma(
                    ops,
                    3 * i,
                    hl,
                    HROW,
                    lane,
                    wave,
                    acc_g,
                    i,
                    jpar=(task & 1) if SHUF_S else None,
                )
                acc_u = fp4_mma(
                    ops,
                    6 + 3 * i,
                    hl,
                    HROW,
                    lane,
                    wave,
                    acc_u,
                    i,
                    jpar=(task & 1) if SHUF_S else None,
                )
            par = (it & 1) * (2 * WAVES * 64 * 4)
            fx.ptr_store(acc_g, red + (par + (wave * 64 + lane) * 4))
            fx.ptr_store(acc_u, red + (par + WAVES * 64 * 4 + (wave * 64 + lane) * 4))
        gpu.barrier()
        if tid < 8 * S:
            rp = tid % 8
            tt = tid // 8
            u = task >> 8
            j = task & 255
            k = fx.ptr_load(rt["kof"] + (u * S + tt))
            av = []
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                gs = fx.Float32(0.0)
                us = fx.Float32(0.0)
                if const_expr(key.gusplit):
                    for w in range_constexpr(4):
                        off = par + (w * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4
                        gs = gs + fx.ptr_load(red + off)
                        us = us + fx.ptr_load(red + (off + 4 * 64 * 4))
                else:
                    for w in range_constexpr(WAVES):
                        off = par + (w * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4
                        gs = gs + fx.ptr_load(red + off)
                        us = us + fx.ptr_load(red + (off + WAVES * 64 * 4))
                gb = bf16_round(gs)
                ub = bf16_round(us)
                sg = bf16_round(
                    gb / (fx.Float32(1.0) + hw_exp2(gb * fx.Float32(-LOG2E)))
                )
                av.append(sg * ub)
            if k >= 0:
                pick = tt * TOPK + k
                mb_put(
                    tag,
                    scratch + fx.Int64(lay["a"][0]),
                    pick * (INTER // 2) + j * 8 + rp,
                    bf16_pair(av[0], av[1]),
                )

    def dn_loads(w2, w2s, w_post, rt, task, lane, wave, ok):
        r = lane % 16
        g = lane // 16
        task = ok.select(task, fx.Int32(0))
        u = task >> 8
        j = task & 255
        e = uniform(fx.ptr_load(rt["uexp"] + u))
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
            if const_expr(
                SHUF_W
            ):  # lane L: the 16 B at 16 L of the 1 KB block (see gu_loads)
                wb = (rbase * (INTER // 2) + (2 * c) * 1024 + lane * 16) // 4
                ops.append(ld_vec_if(ok, w2, wb, 4, W2_BYTES, CM_NT))
                ops.append(ld_vec_if(ok, w2, wb + 256, 4, W2_BYTES, CM_NT))
            elif const_expr(True):
                wb = (rowc * (INTER // 2) + 128 * c + 16 * (lane % 8)) // 4
                ops.append(ld_vec_if(ok, w2, wb, 4, W2_BYTES, CM_NT))
                ops.append(
                    ld_vec_if(ok, w2, wb + 8 * (INTER // 2) // 4, 4, W2_BYTES, CM_NT)
                )
            else:
                wb = (row * (INTER // 2) + 128 * c + 32 * g) // 4
                ops.append(ld_vec_if(ok, w2, wb, 4, W2_BYTES, CM_NT))
                ops.append(ld_vec_if(ok, w2, wb + 4, 4, W2_BYTES, CM_NT))
            if const_expr(SHUF_S):  # see gu_loads
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
        # the epilogue's w_post pair (rows 16j + 2 (lane % 8), +1): loaded with the weights,
        # not in the epilogue, where it would wait behind the whole prefetch queue
        ops.append(ld_i32_if(ok, w_post, 8 * j + lane % 8, DM * 2))
        return ops

    @traced
    def dn_build_picks(rt, u0, u1, tid):
        """upk[h][n]: the picks (t*8 + k, token order) of expert slot u0 + h; upn[h]."""
        if tid < 2:
            u = fx.min(u0 + tid, NU - 1)
            vu = (tid == 0) | (u1 != u0)
            n = fx.Int32(0)
            for tt in range_constexpr(S):
                k = fx.ptr_load(rt["kof"] + (u * S + tt))
                ok = vu & (k >= 0)
                if ok:
                    fx.ptr_store(fx.Int32(tt * TOPK) + k, rt["upk"] + (tid * S + n))
                n = n + ok.select(fx.Int32(1), fx.Int32(0))
            fx.ptr_store(n, rt["upn"] + tid)
        gpu.barrier()

    @traced
    def load_ac(scratch, tag, health, rt, second, buf, tid):
        """Compact A: LDS row m = u0's pick m (m < n0), then u1's picks (when both fit
        in S rows); ``second``: u1's picks from row 0 (the overflow reload)."""
        n0 = uniform(fx.ptr_load(rt["upn"] + 0))
        n1 = uniform(fx.ptr_load(rt["upn"] + 1))
        fit = (n0 + n1) <= AC_CAP
        specs, oks, dsts = [], [], []
        for m in range_constexpr(S):
            if const_expr(second):
                pk = fx.ptr_load(rt["upk"] + (S + fx.Int32(m)))
                ok = fx.Int32(m) < n1
            else:
                in0 = fx.Int32(m) < n0
                pk = in0.select(
                    fx.ptr_load(rt["upk"] + m),
                    fx.ptr_load(rt["upk"] + (S + fx.max(fx.Int32(m) - n0, 0))),
                )
                ok = in0 | (fit & (fx.Int32(m) < n0 + n1))
            pk = ok.select(pk, fx.Int32(0))
            for qq in range_constexpr(INTER // 4 // THREADS):
                sidx = tid + THREADS * qq
                specs.append(
                    (scratch + fx.Int64(lay["a"][0]), pk * (INTER // 2) + 2 * sidx, 2)
                )
                oks.append(ok)
                dsts.append(buf + (m * AROW + 4 * sidx) // 2)
        got = mb_poll(tag, health, specs, REGION_IDS["a"], batch=16, oks=oks)
        for n in range_constexpr(len(specs)):
            fx.ptr_store(
                fx.Vector.from_elements([got[n][0], got[n][1]], fx.Int32), dsts[n]
            )

    @traced
    def dn_switch(scratch, tag, health, rt, task, u0, u1, buf, tid):
        """Overflow (u0's + u1's picks > S rows): reload u1's rows at its first task."""
        n0 = uniform(fx.ptr_load(rt["upn"] + 0))
        n1 = uniform(fx.ptr_load(rt["upn"] + 1))
        if (u1 != u0) & ((n0 + n1) > AC_CAP):
            gpu.barrier()
            load_ac(scratch, tag, health, rt, True, buf, tid)
            gpu.barrier()

    @traced
    def dn_compute_c(ops, task, it, ubase, scratch, tag, rt, tid, lane, wave, al, red):
        """Compact-A down task: B column n = pick n of the task's expert."""
        u = task >> 8
        j = task & 255
        h = (u - ubase) & 1
        n0 = uniform(fx.ptr_load(rt["upn"] + 0))
        n1 = uniform(fx.ptr_load(rt["upn"] + 1))
        fit = (n0 + n1) <= AC_CAP
        rowb = (h == 0).select(fx.Int32(0), fit.select(n0, fx.Int32(0)))
        nv = (h == 0).select(n0, n1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(2):
            acc = fp4_mma(
                ops,
                3 * i,
                al,
                AROW,
                lane,
                wave,
                acc,
                i,
                bsel=(rowb, nv),
                jpar=(task & 1) if SHUF_S else None,
            )
        par = (it & 1) * (WAVES * 64 * 4)
        fx.ptr_store(acc, red + (par + (wave * 64 + lane) * 4))
        gpu.barrier()
        if tid < 8 * S:
            rp = tid % 8
            n = tid // 8
            pk = fx.ptr_load(rt["upk"] + (h * S + fx.min(n, S - 1)))
            tt = pk >> 3
            k = pk & 7
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
            ss = zv[0] * zv[0] + zv[1] * zv[1]
            ss = butterfly(ss, (1, 2, 4))
            if n < nv:
                pick = tt * TOPK + k
                mb_put(
                    tag,
                    scratch + fx.Int64(lay["z"][0]),
                    pick * (DM // 2) + j * 8 + rp,
                    bf16_pair(zv[0], zv[1]),
                )
                if rp == 0:
                    sa_ = rt["ssacc"] + (h * S + tt)
                    fx.ptr_store(fx.ptr_load(sa_) + ss, sa_)

    @traced
    def stage_final(scratch, out, tag, health, rt, tid, bid, lane, zl):
        npk = uniform(fx.ptr_load(rt["np_"] + 0))
        part = tid % 8
        for hh in range_constexpr((P * 8 + THREADS - 1) // THREADS):
            pi = tid // 8 + (THREADS // 8) * hh
            if pi < npk:
                final_pick(scratch, tag, health, rt, pi, part, bid, zl)

    @traced
    def final_pick(scratch, tag, health, rt, pi, part, bid, zl):
        pick = fx.ptr_load(rt["plist"] + pi)
        zspec = (scratch + fx.Int64(lay["z"][0]), pick * (DM // 2) + bid * 8 + part, 1)
        if const_expr(True):
            # the CTA-aggregated partials of the pick's expert: CTAs cf .. cl
            nu_ = uniform(fx.ptr_load(rt["nu"] + 0))
            el_ = slot_of(rt, fx.ptr_load(rt["rid"] + pick), rt["rank"] * EL)
            u = fx.ptr_load(rt["uslot"] + fx.min(fx.max(el_, 0), NU - 1))
            cf = (u * 256) // nu_
            nc = (u * 256 + 255) // nu_ - cf + 1
            specs = [
                (
                    scratch + fx.Int64(lay["ss2"][0]),
                    pick * 256 + fx.min(part + 8 * q, 255),
                    1,
                )
                for q in range(32)
            ]
            oks = [(part + 8 * q) < nc for q in range(32)]
            got = mb_poll(
                tag,
                health,
                specs + [zspec],
                REGION_IDS["ss"],
                batch=33,
                oks=oks + [fx.Boolean(True)],
            )
            s_ = fx.Float32(0.0)
            for q in range_constexpr(32):
                s_ = s_ + got[q][0].bitcast(fx.Float32)
            zw = got[32][0]
        else:
            specs = [
                (scratch + fx.Int64(lay["ss"][0]), pick * 256 + part * 32 + 2 * q, 2)
                for q in range(16)
            ]
            got = mb_poll(tag, health, specs + [zspec], REGION_IDS["ss"], batch=17)
            s_ = fx.Float32(0.0)
            for q in range_constexpr(16):
                s_ = s_ + got[q][0].bitcast(fx.Float32)
                s_ = s_ + got[q][1].bitcast(fx.Float32)
            zw = got[16][0]
        s_ = butterfly(s_, (1, 2, 4))
        fx.ptr_store(bf_lo(zw), zl + (pi * 16 + 2 * part))
        fx.ptr_store(bf_hi(zw), zl + (pi * 16 + 2 * part + 1))
        if part == 0:
            fx.ptr_store(
                hw_rsq(s_ * fx.Float32(1.0 / DM) + fx.Float32(POST_EPS)),
                rt["rstd"] + pi,
            )

    # ------------------------------------------------------------ stage E': partial push + EP all-reduce
    @traced
    def final_push(scratch, part_out, pl, tag, rt, tid, bid, lane, zl, rank):
        """CTA b's [S x 16] EP partial (slot order, the reference rounding) -> part_out
        (debug, local) and slot [parity][rank] of every rank's MOEP."""
        gpu.barrier()
        par = tag_par(tag)
        if tid < 8 * S:
            rp = tid % 8
            tt = tid // 8
            ov = []
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                acc = fx.Float32(0.0)
                for k in range_constexpr(TOPK):
                    p = tt * TOPK + k
                    ix = fx.ptr_load(rt["pidx"] + p)
                    eid = fx.ptr_load(rt["rid"] + p)
                    isl = slot_of(rt, eid, rank * EL) >= 0
                    ixc = fx.min(fx.max(ix, 0), P - 1)
                    zz = fx.ptr_load(zl + (ixc * 16 + rr))
                    rs = fx.ptr_load(rt["rstd"] + ixc)
                    yn = bf16_round(zz * rs)
                    acc = acc + isl.select(
                        yn * fx.ptr_load(rt["rw"] + p), fx.Float32(0.0)
                    )
                ov.append(acc)
            w = bf16_pair(ov[0], ov[1])
            bo.buffer_store(w, rsrc(part_out), (tt * DM + bid * 16 + 2 * rp) // 2)
            if const_expr(XRS):  # reduce-scatter: only the block owner gets the partial
                pd = pl[0]
                for r_ in range_constexpr(1, W):
                    pd = ((bid // CPRX) == r_).select(pl[r_], pd)
                mb_put_sys(
                    tag,
                    pd + fx.Int64(play["moep"][0]),
                    ((par * W + rank) * S + tt) * (DM // 2) + bid * 8 + rp,
                    w,
                )
            else:
                for src_pb in pl:
                    mb_put_sys(
                        tag,
                        src_pb + fx.Int64(play["moep"][0]),
                        ((par * W + rank) * S + tt) * (DM // 2) + bid * 8 + rp,
                        w,
                    )

    @traced
    def ar_reduce(scratch, own_peer, tag, health, tid, bid, pl=None, rank=None):
        """The W ranks' slots of this CTA's columns, summed in rank order (fp32) ->
        bf16 m (local, plain) -> drain -> this CTA's MF flag."""
        par = tag_par(tag)
        if const_expr(XRS):
            if (tid < 8 * S) & (
                (bid // CPRX) == rank
            ):  # the block owner: W slots in rank order -> every rank
                rp = tid % 8
                tt = tid // 8
                specs = [
                    (
                        own_peer + fx.Int64(play["moep"][0]),
                        ((par * W + src) * S + tt) * (DM // 2) + bid * 8 + rp,
                        1,
                    )
                    for src in range(W)
                ]
                got = mb_poll(tag, health, specs, REGION_IDS["moep"], cm=CM_SYS)
                lo = bf_lo(got[0][0])
                hi = bf_hi(got[0][0])
                for src in range_constexpr(1, W):
                    lo = lo + bf_lo(got[src][0])
                    hi = hi + bf_hi(got[src][0])
                wr = bf16_pair(lo, hi)
                for pb in pl:
                    mb_put_sys(
                        tag,
                        pb + fx.Int64(play["mrr"][0]),
                        (par * S + tt) * (DM // 2) + bid * 8 + rp,
                        wr,
                    )
        if tid < 8 * S:
            rp = tid % 8
            tt = tid // 8
            if const_expr(XRS):
                got = mb_poll(
                    tag,
                    health,
                    [
                        (
                            own_peer + fx.Int64(play["mrr"][0]),
                            (par * S + tt) * (DM // 2) + bid * 8 + rp,
                            1,
                        )
                    ],
                    REGION_IDS["mrr"],
                    cm=CM_SYS,
                )
                lo = bf_lo(got[0][0])
                hi = bf_hi(got[0][0])
            else:
                specs = [
                    (
                        own_peer + fx.Int64(play["moep"][0]),
                        ((par * W + src) * S + tt) * (DM // 2) + bid * 8 + rp,
                        1,
                    )
                    for src in range(W)
                ]
                got = mb_poll(tag, health, specs, REGION_IDS["moep"], cm=CM_SYS)
                lo = bf_lo(got[0][0])
                hi = bf_hi(got[0][0])
                for src in range_constexpr(1, W):
                    lo = lo + bf_lo(got[src][0])
                    hi = hi + bf_hi(got[src][0])
            bo.buffer_store(
                bf16_pair(lo, hi),
                rsrc(scratch + fx.Int64(lay["m"][0])),
                (tt * DM + bid * 16 + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["mf"][0]), bid, fx.Int32(1))

    # ------------------------------------------------------------ stage F: post_expert_proj
    def pp_tile(bid, i):
        return bid // KSP if KSP > 1 else bid * TPC + i

    def pp_loads(w_pp, bid, lane, wave):
        k0 = (bid % KSP) * KR
        ops = []
        for i in range_constexpr(TPC):
            for ci in range_constexpr(CPW):
                c = wave + WAVES * ci
                ops.append(
                    bf_tile_load(w_pp, pp_tile(bid, i) * 16, DM, k0 + c * KCH, lane)
                )
        return ops

    @traced
    def post_proj(
        w_pp_ops, scratch, peer, pl, tag, health, tid, bid, lane, wave, rank, ml, red
    ):
        # lane math through an opaque move: with MROW == XQROW LLVM otherwise reuses stage A's
        # LDS addresses here, holding them across the whole kernel (W=4, S=16: 8 spills)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        kh = bid % KSP
        k0 = kh * KR
        nfl = KR // 16
        specs, oks = [], []
        for i in range_constexpr((nfl + THREADS - 1) // THREADS):
            f = fx.min(tid + THREADS * i, nfl - 1)
            specs.append((scratch + fx.Int64(lay["mf"][0]), k0 // 16 + f, 1))
            oks.append((tid + THREADS * i) < nfl)
        mb_poll(tag, health, specs, REGION_IDS["mf"], oks=oks)
        gpu.barrier()
        nm = S * KR // 8
        for i in range_constexpr((nm + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, nm - 1)
            t = e // (KR // 8)
            kk = (e % (KR // 8)) * 8
            v = ld_vec(
                scratch + fx.Int64(lay["m"][0]), (t * DM + k0 + kk) // 2, 4, CM_DEV
            )
            if tid + THREADS * i < nm:
                fx.ptr_store(v, ml + (t * MROW + kk) // 2)
        gpu.barrier()
        t = fx.min(lane % 16, S - 1)
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(TPC)]
        for i in range_constexpr(TPC):
            for ci in range_constexpr(CPW):
                c = wave + WAVES * ci
                for qq in range_constexpr(4):
                    kk = c * KCH + 8 * g + 32 * qq
                    b = fx.Vector(
                        fx.ptr_load(
                            ml + (t * MROW + kk) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    ).bitcast(fx.BFloat16)
                    accs[i] = mfma_bf16(
                        bf_mma_operand(w_pp_ops[i * CPW + ci], qq, lane).bitcast(
                            fx.BFloat16
                        ),
                        b,
                        accs[i],
                    )
        for i in range_constexpr(TPC):
            fx.ptr_store(accs[i], red + ((wave * TPC + i) * 64 + lane) * 4)
        gpu.barrier()
        if tid < TPC * 8 * S:
            i = tid // (8 * S)
            rp = (tid // S) % 8
            tt = tid % S
            v = []
            for h2 in range_constexpr(2):
                rr = 2 * rp + h2
                s_ = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    s_ = s_ + fx.ptr_load(
                        red + (((w * TPC + i) * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4)
                    )
                v.append(s_)
            tile = bid // KSP if KSP > 1 else bid * TPC + i
            if const_expr(KSP == 2):
                if kh == 1:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            scratch + fx.Int64(lay["py"][0]),
                            (tile * 16 + 2 * rp + h2) * S + tt,
                            v[h2],
                        )
                else:
                    got = mb_poll(
                        tag,
                        health,
                        [
                            (
                                scratch + fx.Int64(lay["py"][0]),
                                (tile * 16 + 2 * rp + h2) * S + tt,
                                1,
                            )
                            for h2 in range(2)
                        ],
                        REGION_IDS["py"],
                    )
                    y0 = v[0] + got[0][0].bitcast(fx.Float32)
                    y1 = v[1] + got[1][0].bitcast(fx.Float32)
                    col = (rank * RS_POST if key.shard else 0) + tile * 16 + 2 * rp
                    idx = (par * S + tt) * (HID // 2) + col // 2
                    if const_expr(key.shard):
                        for pb in pl:
                            mb_put_sys(
                                tag, pb + fx.Int64(play["y"][0]), idx, bf16_pair(y0, y1)
                            )
                    else:
                        mb_put_sys(
                            tag, peer + fx.Int64(play["y"][0]), idx, bf16_pair(y0, y1)
                        )
            elif const_expr(KSP > 2):
                # K parts 1 .. KSP-1 put their partial (slot kh - 1); part 0 sums all parts in order
                if kh != 0:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            scratch + fx.Int64(lay["py"][0]),
                            ((kh - 1) * NTL * 16 + tile * 16 + 2 * rp + h2) * S + tt,
                            v[h2],
                        )
                else:
                    got = mb_poll(
                        tag,
                        health,
                        [
                            (
                                scratch + fx.Int64(lay["py"][0]),
                                ((q - 1) * NTL * 16 + tile * 16 + 2 * rp + h2) * S + tt,
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
                    col = (rank * RS_POST if key.shard else 0) + tile * 16 + 2 * rp
                    idx = (par * S + tt) * (HID // 2) + col // 2
                    if const_expr(key.shard):
                        for pb in pl:
                            mb_put_sys(
                                tag, pb + fx.Int64(play["y"][0]), idx, bf16_pair(y0, y1)
                            )
                    else:
                        mb_put_sys(
                            tag, peer + fx.Int64(play["y"][0]), idx, bf16_pair(y0, y1)
                        )
            else:
                col = (rank * RS_POST if key.shard else 0) + tile * 16 + 2 * rp
                idx = (par * S + tt) * (HID // 2) + col // 2
                if const_expr(key.shard):
                    for pb in pl:
                        mb_put_sys(
                            tag, pb + fx.Int64(play["y"][0]), idx, bf16_pair(v[0], v[1])
                        )
                else:
                    mb_put_sys(
                        tag, peer + fx.Int64(play["y"][0]), idx, bf16_pair(v[0], v[1])
                    )

    # ------------------------------------------------------------ stage G: post-FFN norm, residual, next norm
    @traced
    def tok_rstd(scratch, region, tag, health, tid, lane, rs_out, eps, bid, red):
        """rs_out[t] = rsqrt(sum of the 256 CTA partials of token t / HID + eps). One
        reducer CTA per token (CTA 255 - t) polls the 256 partials (no hot spot), sums
        them in a fixed order and publishes rstd[t]; every CTA polls the S values."""
        rbase = scratch + fx.Int64(lay[region][0])
        tred = bid >= BLOCKS - S
        if tred:
            t = BLOCKS - 1 - bid
            v = fx.Float32(0.0)
            if tid < BLOCKS:
                got = mb_poll(
                    tag, health, [(rbase, t * BLOCKS + tid, 1)], REGION_IDS[region]
                )
                v = got[0][0].bitcast(fx.Float32)
            v = wave_sum(v)
            if lane == 0:
                fx.ptr_store(v, red + tid // 64)
            gpu.barrier()
            if tid == 0:
                tot = fx.Float32(0.0)
                for w in range_constexpr(BLOCKS // 64):
                    tot = tot + fx.ptr_load(red + w)
                mb_put(
                    tag,
                    rbase,
                    S * BLOCKS + t,
                    hw_rsq(tot * fx.Float32(1.0 / HID) + fx.Float32(eps)),
                )
        if tid < S:
            got2 = mb_poll(
                tag, health, [(rbase, S * BLOCKS + tid, 1)], REGION_IDS[region]
            )
            fx.ptr_store(got2[0][0].bitcast(fx.Float32), rs_out + tid)
        gpu.barrier()

    @traced
    def tok_rstd2(
        scratch,
        region,
        tag,
        health,
        tid,
        lane,
        rs_out,
        eps1,
        eps2,
        bid,
        red,
        pbase=None,
        pcm=CM_DEV,
    ):
        """One round for two norms of a token. Every CTA put 4 partials of token t at
        ((t * BLOCKS + CTA) * 4 + j): P1 = sum o^2, P2 = sum (a s)^2, P3 = sum (a s)(b o),
        P4 = sum (b o)^2 for s' = a s + b (o r1). Reducer CTA 255 - t sums the 256 CTAs in a
        fixed order: r1 = rsqrt(P1 / HID + eps1), r2 = rsqrt((P2 + 2 r1 P3 + r1^2 P4) / HID + eps2)
        (r2 from the unrounded o r1). rs_out[t] = r1, rs_out[S + t] = r2 on every CTA."""
        rbase = scratch + fx.Int64(lay[region][0])
        prb = (
            rbase if pbase is None else pbase
        )  # #12: the partials come from the own peer buffer (the AB owners)
        tred = bid >= BLOCKS - S
        if tred:
            t = BLOCKS - 1 - bid
            va = fx.Float32(0.0)
            vb = fx.Float32(0.0)
            if tid < 2 * BLOCKS:
                got = mb_poll(
                    tag,
                    health,
                    [(prb, (t * BLOCKS + tid // 2) * 4 + 2 * (tid % 2), 2)],
                    REGION_IDS[region],
                    cm=pcm,
                )
                va = got[0][0].bitcast(fx.Float32)
                vb = got[0][1].bitcast(fx.Float32)
            va = butterfly(va, (32, 16, 8, 4, 2))  # lanes of one parity: 32 CTAs
            vb = butterfly(vb, (32, 16, 8, 4, 2))
            if lane < 2:
                fx.ptr_store(va, red + (tid // 64) * 4 + 2 * lane)
                fx.ptr_store(vb, red + (tid // 64) * 4 + 2 * lane + 1)
            gpu.barrier()
            if tid == 0:
                p = [fx.Float32(0.0) for _ in range(4)]
                for w in range_constexpr(WAVES):
                    for j in range_constexpr(4):
                        p[j] = p[j] + fx.ptr_load(red + w * 4 + j)
                # red slot order: lane 0 -> (P1, P2), lane 1 -> (P3, P4)
                r1 = hw_rsq(p[0] * fx.Float32(1.0 / HID) + fx.Float32(eps1))
                q = p[1] + fx.Float32(2.0) * r1 * p[2] + r1 * r1 * p[3]
                r2 = hw_rsq(q * fx.Float32(1.0 / HID) + fx.Float32(eps2))
                mb_put(tag, rbase, 4 * S * BLOCKS + 2 * t, r1)
                mb_put(tag, rbase, 4 * S * BLOCKS + 2 * t + 1, r2)
        if tid < S:
            got2 = mb_poll(
                tag, health, [(rbase, 4 * S * BLOCKS + 2 * tid, 2)], REGION_IDS[region]
            )
            fx.ptr_store(got2[0][0].bitcast(fx.Float32), rs_out + tid)
            fx.ptr_store(got2[0][1].bitcast(fx.Float32), rs_out + (S + tid))
        gpu.barrier()

    def put4(tag, scratch, region, tt, bid, q1, q2, q3, q4):
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
    def stage_g(
        peer,
        scratch,
        stream_in,
        w_pff,
        alpha,
        beta,
        w_in,
        stream_out,
        xn_out,
        tag,
        health,
        tid,
        bid,
        lane,
        rsl,
        tls,
    ):
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        par = tag_par(tag)
        pp = tid % 16
        tt = fx.min(tid // 16, S - 1)
        live = tid < 16 * S
        c = bid * 32 + 2 * pp
        # data-independent loads first
        st = ld_vec(scratch + fx.Int64(lay["smid"][0]), tt * HID + c, 2, CM_DEV)
        wpf = ld_i32(w_pff, c // 2)
        al = ld_vec(alpha, c, 2)
        be = ld_vec(beta, c, 2)
        wi = ld_i32(w_in, c // 2)
        z0 = fx.Float32(0.0)
        z1 = fx.Float32(0.0)
        if live:
            got = mb_poll(
                tag,
                health,
                [
                    (
                        peer + fx.Int64(play["y"][0]),
                        (par * S + tt) * (HID // 2) + bid * 16 + pp,
                        1,
                    )
                ],
                REGION_IDS["y"],
                cm=CM_SYS,
            )
            yw = got[0][0]
            z0 = bf16_round(bf_lo(wpf) * bf_lo(yw))
            z1 = bf16_round(bf_hi(wpf) * bf_hi(yw))
        if const_expr(key.onepass):
            a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
            a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
            b0 = be[0].bitcast(fx.Float32) * z0
            b1 = be[1].bitcast(fx.Float32) * z1
            q1 = butterfly(z0 * z0 + z1 * z1, (1, 2, 4, 8))
            q2 = butterfly(a0 * a0 + a1 * a1, (1, 2, 4, 8))
            q3 = butterfly(a0 * b0 + a1 * b1, (1, 2, 4, 8))
            q4 = butterfly(b0 * b0 + b1 * b1, (1, 2, 4, 8))
            if live & (pp == 0):
                put4(tag, scratch, "ssg4", tt, bid, q1, q2, q3, q4)
            tok_rstd2(
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
            r1 = fx.ptr_load(rsl + tt)
            m0 = bf16_round(z0 * r1)
            m1 = bf16_round(z1 * r1)
            s0 = a0 + be[0].bitcast(fx.Float32) * m0
            s1 = a1 + be[1].bitcast(fx.Float32) * m1
            if live:
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
                    ),
                    rsrc(stream_out),
                    tt * HID + c,
                )
        else:
            ss = butterfly(z0 * z0 + z1 * z1, (1, 2, 4, 8))
            if live & (pp == 0):
                mb_put(tag, scratch + fx.Int64(lay["ss1"][0]), tt * BLOCKS + bid, ss)
            tok_rstd(
                scratch, "ss1", tag, health, tid, lane, rsl, POST_EPS, bid, rsl + 2 * S
            )
            r1 = fx.ptr_load(rsl + tt)
            m0 = bf16_round(z0 * r1)
            m1 = bf16_round(z1 * r1)
            s0 = (
                al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
                + be[0].bitcast(fx.Float32) * m0
            )
            s1 = (
                al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
                + be[1].bitcast(fx.Float32) * m1
            )
            if live:
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
                    ),
                    rsrc(stream_out),
                    tt * HID + c,
                )
            ss2 = butterfly(s0 * s0 + s1 * s1, (1, 2, 4, 8))
            if live & (pp == 0):
                mb_put(tag, scratch + fx.Int64(lay["ss2g"][0]), tt * BLOCKS + bid, ss2)
            tok_rstd(
                scratch,
                "ss2g",
                tag,
                health,
                tid,
                lane,
                rsl + S,
                PRE_EPS,
                bid,
                rsl + 2 * S,
            )
        r2 = fx.ptr_load(rsl + (S + tt))
        if live:
            xw = bf16_pair((s0 * r2) * bf_lo(wi), (s1 * r2) * bf_hi(wi))
            bo.buffer_store(xw, rsrc(xn_out), (tt * HID + c) // 2)
            bo.buffer_store(
                xw,
                rsrc(scratch + fx.Int64(lay["xn"][0])),
                (tt * HID + c) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xnf"][0]), bid, fx.Int32(1))

    # ------------------------------------------------------------ stage O: o-norm x gate, o_proj, push
    def o_loads(w_o, bid, lane, wave):
        """o_proj rows 32b .. 32b+32 (2 tiles) x this wave's CO chunks of K = OK_K (2048 / 1024)."""
        ops = []
        for i in range_constexpr(2):
            for ci in range_constexpr(CO):
                c = wave + WAVES * ci
                if const_expr(key.s3o & 1):
                    ops.append(
                        bf_tile_load_mma(w_o, bid * 32 + 16 * i, OK_K, c * KCH, lane)
                    )
                else:
                    ops.append(
                        bf_tile_load(w_o, bid * 32 + 16 * i, OK_K, c * KCH, lane)
                    )
        return ops

    @traced
    def stage_o(
        o_ops,
        attn_o,
        attn_g,
        g_stride,
        pl,
        tag,
        tid,
        bid,
        lane,
        wave,
        rank,
        yl,
        red,
        ao=None,
    ):
        """Y = bf16(o rms(o) sigmoid(g)) per 64-wide head (SGLang's _rms_norm_gate: fp32,
        one rounding) into LDS for every CTA, then o_proj rows 32b.. (this rank's K slice),
        the bf16 partial pushed into slot [par][rank] of every rank's OP."""
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        lane = fresh(lane)
        par = tag_par(tag)
        g = lane // 16
        npc = S * OK_K // 8
        for i in range_constexpr((npc + THREADS - 1) // THREADS):
            e = tid + THREADS * i
            t = fx.min(e // (OK_K // 8), S - 1)
            kk = (e % (OK_K // 8)) * 8
            if const_expr(ao is None):
                xv = ld_vec(attn_o, (t * OK_K + kk) // 2, 4)
            else:  # S4 fused attention (aot=1): the merged rows as tagged pairs
                got = mb_poll(
                    tag,
                    ao[1],
                    [
                        (ao[0], (t * OK_K + kk) // 2, 2),
                        (ao[0], (t * OK_K + kk) // 2 + 2, 2),
                    ],
                    43,
                )
                xv = got[0] + got[1]
            gv = ld_vec(attn_g, (t * g_stride + kk) // 2, 4)
            ss = fx.Float32(0.0)
            for d in range_constexpr(4):
                ss = ss + bf_lo(xv[d]) * bf_lo(xv[d]) + bf_hi(xv[d]) * bf_hi(xv[d])
            ss = butterfly(ss, (1, 2, 4))  # the head's 8 lanes
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
            if e < npc:
                fx.ptr_store(
                    fx.Vector.from_elements(ow, fx.Int32), yl + (t * YROW + kk) // 2
                )
        gpu.barrier()
        t = fx.min(lane % 16, S - 1)
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for i in range_constexpr(2):
            for ci in range_constexpr(CO):
                c = wave + WAVES * ci
                for qq in range_constexpr(4):
                    kk = c * KCH + 8 * g + 32 * qq
                    b = fx.Vector(
                        fx.ptr_load(
                            yl + (t * YROW + kk) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    ).bitcast(fx.BFloat16)
                    if const_expr(key.s3o & 1):
                        aop = o_ops[i * CO + ci][qq]
                    else:
                        aop = bf_mma_operand(o_ops[i * CO + ci], qq, lane)
                    accs[i] = mfma_bf16(aop.bitcast(fx.BFloat16), b, accs[i])
        for i in range_constexpr(2):
            fx.ptr_store(accs[i], red + ((wave * 2 + i) * 64 + lane) * 4)
        gpu.barrier()
        if tid < 16 * S:
            pp = tid % 16  # row pair of the CTA's 32 rows
            tt = tid // 16
            v = []
            for h2 in range_constexpr(2):
                rr = 2 * pp + h2
                s_ = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    s_ = s_ + fx.ptr_load(
                        red
                        + (
                            ((w * 2 + rr // 16) * 64 + 16 * ((rr % 16) // 4) + tt) * 4
                            + rr % 4
                        )
                    )
                v.append(s_)
            w2 = bf16_pair(v[0], v[1])
            if const_expr(XRS):  # reduce-scatter: only the block owner gets the partial
                pd = pl[0]
                for r_ in range_constexpr(1, W):
                    pd = ((bid // CPRX) == r_).select(pl[r_], pd)
                mb_put_sys(
                    tag,
                    pd + fx.Int64(play["op"][0]),
                    ((par * W + rank) * S + tt) * (HID // 2) + bid * 16 + pp,
                    w2,
                )
            else:
                for pb in pl:
                    mb_put_sys(
                        tag,
                        pb + fx.Int64(play["op"][0]),
                        ((par * W + rank) * S + tt) * (HID // 2) + bid * 16 + pp,
                        w2,
                    )

    # ------------------------------------------------------------ stage AB: attention boundary
    @traced
    def stage_ab(
        peer,
        scratch,
        stream_pre,
        alpha_a,
        beta_a,
        w_pffn,
        tag,
        health,
        tid,
        bid,
        lane,
        rsl,
        tls,
        pl=None,
        rank=None,
    ):
        """CTA b's hidden columns 32b ..: the W o_proj partials in rank order (fp32 -> bf16,
        the all-reduce output), post_attn_norm (weightless, 1e-8), gated residual on the
        fp32 stream, pre_feedforward_layernorm -> ffn_in (local, + flag) and the mid stream."""
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        par = tag_par(tag)
        pp = tid % 16
        tt = fx.min(tid // 16, S - 1)
        live = tid < 16 * S
        c = bid * 32 + 2 * pp
        st = ld_vec(stream_pre, tt * HID + c, 2)
        al = ld_vec(alpha_a, c, 2)
        be = ld_vec(beta_a, c, 2)
        wf = ld_i32(w_pffn, c // 2)
        o0 = fx.Float32(0.0)
        o1 = fx.Float32(0.0)
        if const_expr(XRS):
            if live & (
                (bid // CPRX) == rank
            ):  # the block owner: W partials in rank order -> bf16 -> every rank
                specs = [
                    (
                        peer + fx.Int64(play["op"][0]),
                        ((par * W + src) * S + tt) * (HID // 2) + bid * 16 + pp,
                        1,
                    )
                    for src in range(W)
                ]
                got = mb_poll(tag, health, specs, REGION_IDS["op"], cm=CM_SYS)
                a0 = bf_lo(got[0][0])
                a1 = bf_hi(got[0][0])
                for src in range_constexpr(1, W):
                    a0 = a0 + bf_lo(got[src][0])
                    a1 = a1 + bf_hi(got[src][0])
                wr = bf16_pair(a0, a1)
                if const_expr(
                    key.p12
                ):  # #12: this block's 4 norm partials (the values every CTA b computes)
                    oo0 = bf_lo(wr)
                    oo1 = bf_hi(wr)
                    pa0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
                    pa1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
                    pb0 = be[0].bitcast(fx.Float32) * oo0
                    pb1 = be[1].bitcast(fx.Float32) * oo1
                    p1 = butterfly(oo0 * oo0 + oo1 * oo1, (1, 2, 4, 8))
                    p2 = butterfly(pa0 * pa0 + pa1 * pa1, (1, 2, 4, 8))
                    p3 = butterfly(pa0 * pb0 + pa1 * pb1, (1, 2, 4, 8))
                    p4 = butterfly(pb0 * pb0 + pb1 * pb1, (1, 2, 4, 8))
                    if pp == 0:
                        i0 = ((par * S + tt) * BLOCKS + bid) * 4
                        for pb in pl:
                            base = pb + fx.Int64(play["ssa4p"][0])
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [
                                        p1.bitcast(fx.Int32),
                                        tag,
                                        p2.bitcast(fx.Int32),
                                        tag,
                                    ],
                                    fx.Int32,
                                ),
                                rsrc(base),
                                i0 * 2,
                                cache_modifier=CM_SYS,
                            )
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [
                                        p3.bitcast(fx.Int32),
                                        tag,
                                        p4.bitcast(fx.Int32),
                                        tag,
                                    ],
                                    fx.Int32,
                                ),
                                rsrc(base),
                                (i0 + 2) * 2,
                                cache_modifier=CM_SYS,
                            )
                for pb in pl:
                    mb_put_sys(
                        tag,
                        pb + fx.Int64(play["orr"][0]),
                        (par * S + tt) * (HID // 2) + bid * 16 + pp,
                        wr,
                    )
            if live:
                got = mb_poll(
                    tag,
                    health,
                    [
                        (
                            peer + fx.Int64(play["orr"][0]),
                            (par * S + tt) * (HID // 2) + bid * 16 + pp,
                            1,
                        )
                    ],
                    REGION_IDS["orr"],
                    cm=CM_SYS,
                )
                o0 = bf_lo(got[0][0])
                o1 = bf_hi(got[0][0])
        elif live:
            specs = [
                (
                    peer + fx.Int64(play["op"][0]),
                    ((par * W + src) * S + tt) * (HID // 2) + bid * 16 + pp,
                    1,
                )
                for src in range(W)
            ]
            got = mb_poll(tag, health, specs, REGION_IDS["op"], cm=CM_SYS)
            a0 = bf_lo(got[0][0])
            a1 = bf_hi(got[0][0])
            for src in range_constexpr(1, W):
                a0 = a0 + bf_lo(got[src][0])
                a1 = a1 + bf_hi(got[src][0])
            o0 = bf16_round(a0)
            o1 = bf16_round(a1)
        if const_expr(key.onepass):
            a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
            a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
            b0 = be[0].bitcast(fx.Float32) * o0
            b1 = be[1].bitcast(fx.Float32) * o1
            q1 = butterfly(o0 * o0 + o1 * o1, (1, 2, 4, 8))
            q2 = butterfly(a0 * a0 + a1 * a1, (1, 2, 4, 8))
            q3 = butterfly(a0 * b0 + a1 * b1, (1, 2, 4, 8))
            q4 = butterfly(b0 * b0 + b1 * b1, (1, 2, 4, 8))
            if const_expr(key.p12):  # #12: the owners sent the partials with orr
                tok_rstd2(
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
                    pbase=peer
                    + fx.Int64(play["ssa4p"][0])
                    + fx.Int64(par * S * BLOCKS * 4 * 8),
                    pcm=CM_SYS,
                )
            else:
                if live & (pp == 0):
                    put4(tag, scratch, "ssa4", tt, bid, q1, q2, q3, q4)
                tok_rstd2(
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
                )
            r1 = fx.ptr_load(rsl + tt)
            n0 = bf16_round(o0 * r1)
            n1 = bf16_round(o1 * r1)
            s0 = a0 + be[0].bitcast(fx.Float32) * n0
            s1 = a1 + be[1].bitcast(fx.Float32) * n1
            if live:
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
                    ),
                    rsrc(scratch + fx.Int64(lay["smid"][0])),
                    tt * HID + c,
                    cache_modifier=CM_DEV,
                )
        else:
            ss = butterfly(o0 * o0 + o1 * o1, (1, 2, 4, 8))
            if live & (pp == 0):
                mb_put(tag, scratch + fx.Int64(lay["ssa1"][0]), tt * BLOCKS + bid, ss)
            tok_rstd(
                scratch, "ssa1", tag, health, tid, lane, rsl, POST_EPS, bid, rsl + 2 * S
            )
            r1 = fx.ptr_load(rsl + tt)
            n0 = bf16_round(o0 * r1)
            n1 = bf16_round(o1 * r1)
            s0 = (
                al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
                + be[0].bitcast(fx.Float32) * n0
            )
            s1 = (
                al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
                + be[1].bitcast(fx.Float32) * n1
            )
            if live:
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [s0.bitcast(fx.Int32), s1.bitcast(fx.Int32)], fx.Int32
                    ),
                    rsrc(scratch + fx.Int64(lay["smid"][0])),
                    tt * HID + c,
                    cache_modifier=CM_DEV,
                )
            ss2 = butterfly(s0 * s0 + s1 * s1, (1, 2, 4, 8))
            if live & (pp == 0):
                mb_put(tag, scratch + fx.Int64(lay["ssa2"][0]), tt * BLOCKS + bid, ss2)
            tok_rstd(
                scratch,
                "ssa2",
                tag,
                health,
                tid,
                lane,
                rsl + S,
                PRE_EPS,
                bid,
                rsl + 2 * S,
            )
        r2 = fx.ptr_load(rsl + (S + tt))
        if live:
            bo.buffer_store(
                bf16_pair((s0 * r2) * bf_lo(wf), (s1 * r2) * bf_hi(wf)),
                rsrc(scratch + fx.Int64(lay["xin"][0])),
                (tt * HID + c) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xf"][0]), bid, fx.Int32(1))

    # ------------------------------------------------------------ stage QKV: next layer's qkv+gate
    def q_loads(w_qkvg, bid, lane, wave, n0, n1):
        """Chunk-tiles n0 .. n1 (of NQT = 9 / 5) of this CTA: tile NQT (b / 8) + n, K eighth b % 8,
        this wave's 128-K chunk. Tiles past the rank's last (TP8: 144 of 160) read tile 0 (never
        stored)."""
        k8 = bid % 8
        gq = bid // 8
        if const_expr(NQT * 32 == G.NQTILES):
            return [
                bf_tile_load(
                    w_qkvg, (NQT * gq + n) * 16, HID, k8 * XQ8 + wave * KCH, lane
                )
                for n in range(n0, n1)
            ]
        return [
            bf_tile_load(
                w_qkvg,
                fx.min(NQT * gq + n, G.NQTILES - 1) * 16,
                HID,
                k8 * XQ8 + wave * KCH,
                lane,
            )
            for n in range(n0, n1)
        ]

    @traced
    def stage_qkv(
        q_ops, scratch, g_out, tag, health, tid, bid, lane, wave, xl, red, own
    ):
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        lane = fresh(lane)
        g = lane // 16
        k8 = bid % 8
        gq = bid // 8
        # the next layer's input slice (columns 1024 k8 ..): CTAs 32 k8 .. +32 own them
        if tid < 32:
            mb_poll(
                tag,
                health,
                [(scratch + fx.Int64(lay["xnf"][0]), k8 * 32 + tid, 1)],
                REGION_IDS["xnf"],
            )
        gpu.barrier()
        nx = S * XQ8 // 8
        for i in range_constexpr((nx + THREADS - 1) // THREADS):
            e = tid + THREADS * i
            t = fx.min(e // (XQ8 // 8), S - 1)
            kk = (e % (XQ8 // 8)) * 8
            v = ld_vec(
                scratch + fx.Int64(lay["xn"][0]),
                (t * HID + k8 * XQ8 + kk) // 2,
                4,
                CM_DEV,
            )
            if e < nx:
                fx.ptr_store(v, xl + (t * X8ROW + kk) // 2)
        gpu.barrier()
        t = fx.min(lane % 16, S - 1)
        for n in range_constexpr(NQT):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for qq in range_constexpr(4):
                kk = wave * KCH + 8 * g + 32 * qq
                b = fx.Vector(
                    fx.ptr_load(
                        xl + (t * X8ROW + kk) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                ).bitcast(fx.BFloat16)
                acc = mfma_bf16(
                    bf_mma_operand(q_ops[n], qq, lane).bitcast(fx.BFloat16), b, acc
                )
            fx.ptr_store(acc, red + ((wave * NQT + n) * 64 + lane) * 4)
        gpu.barrier()
        # thread (tile n, row pair rp, token tt): this K eighth's partial; row pair k8 is ours
        for it_ in range_constexpr((NQT * 8 * S + THREADS - 1) // THREADS):
            e = tid + THREADS * it_
            if e < NQT * 8 * S:
                n = e // (8 * S)
                rp = (e // S) % 8
                tt = e % S
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * rp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red
                            + (((w * NQT + n) * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4)
                        )
                    v.append(s_)
                if rp == k8:
                    fx.ptr_store(v[0], own + ((n * 2) * S + tt))
                    fx.ptr_store(v[1], own + ((n * 2 + 1) * S + tt))
                else:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            scratch + fx.Int64(lay["qx"][0]),
                            (((gq * 8 + k8) * NQT + n) * 16 + 2 * rp + h2) * S + tt,
                            v[h2],
                        )
        gpu.barrier()
        # owner threads (tile n, token tt): rows 2 k8, 2 k8 + 1 of tile n, the 8 eighths in order
        if tid < NQT * S:
            n = tid // S
            tt = tid % S
            specs, oks = [], []
            for src in range_constexpr(8):
                for h2 in range_constexpr(2):
                    specs.append(
                        (
                            scratch + fx.Int64(lay["qx"][0]),
                            (((gq * 8 + src) * NQT + n) * 16 + 2 * k8 + h2) * S + tt,
                            1,
                        )
                    )
                    oks.append(fx.Int32(src) != k8)
            got = mb_poll(tag, health, specs, REGION_IDS["qx"], oks=oks)
            tot = []
            for h2 in range_constexpr(2):
                s_ = fx.Float32(0.0)
                for src in range_constexpr(8):
                    mine = fx.ptr_load(own + ((n * 2 + h2) * S + tt))
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
                    tt * (QKV_ROWS // 2) + row // 2,
                    w2,
                )
            else:
                if const_expr(NQT * 32 == G.NQTILES):
                    bo.buffer_store(w2, rsrc(g_out), (tt * QROWS + row - QKV_ROWS) // 2)
                else:
                    st_if(
                        row < G.QKVG_ROWS, w2, g_out, (tt * QROWS + row - QKV_ROWS) // 2
                    )

    # ------------------------------------------------------------ S3: stage QKV at W = 8 (s3q != 0)
    # bits of key.s3q:
    #   1 FUSE   head-aligned tile map + token owners + the head finish in the owners (no finisher hop):
    #            group gq < NH = HQ + 2 HKV holds the 4 tiles of head gq (q heads, then k, then v) + gate tile
    #            NQKVT + gq; groups NH .. hold the other gate tiles (5 a group, the tail padded). After the K-split
    #            partial sums, CTA k8 of the group owns the tokens t = k8 (mod 8) for all its rows, so it holds
    #            whole heads: qk-norm, x qk_scale, RoPE, q / fp8 k, v stores as head_finish (same ops, same
    #            order: bit-identical). The qkv rows are still written to the scratch region (checks).
    #   2 XDIRECT the x eighth goes to registers as the MFMA B operand (no LDS staging, one barrier less)
    #  16 MLAYOUT the weights are loaded in the MFMA A-operand layout (lane = row l % 16, K 8 (l / 16) + 32 qq:
    #            16 rows x 64 B an instruction) instead of the coalesced 8 rows x 128 B + 8 ds_bpermute a step
    #            (bf_mma_operand); the same operand values (bit-identical)
    S3Q = key.s3q
    S3_FUSE, S3_XD, S3_ML = bool(S3Q & 1), bool(S3Q & 2), bool(S3Q & 16)
    NH = HQ + 2 * HKV
    NQKVT = QKV_ROWS // 16
    if S3Q:
        assert W == 8 and NQT == 5 and NH <= 32 and S % 8 == 0 and S3Q & ~19 == 0, (
            W,
            NQT,
            NH,
            S,
            S3Q,
        )

    def q8_tile(gq, n):
        """Tile index (traced) of chunk-tile n of group gq (FUSE map, else the shipped one)."""
        if const_expr(not S3_FUSE):
            return NQT * gq + n
        head = gq < NH
        if const_expr(n < 4):
            return head.select(4 * gq + n, NQKVT + NH + NQT * (gq - NH) + n)
        return head.select(NQKVT + gq, NQKVT + NH + NQT * (gq - NH) + n)

    def q8_loads(w_qkvg, bid, lane, wave, n0, n1):
        k8 = bid % 8
        gq = bid // 8
        out = []
        for n in range_constexpr(n0, n1):
            tl = q8_tile(gq, n)
            row16 = fx.min(tl, G.NQTILES - 1) * 16
            if const_expr(S3_ML):
                out.append(
                    bf_tile_load_mma(w_qkvg, row16, HID, k8 * XQ8 + wave * KCH, lane)
                )
            else:
                out.append(
                    bf_tile_load(w_qkvg, row16, HID, k8 * XQ8 + wave * KCH, lane)
                )
        return out

    @traced
    def vllm_finish(w2, hh, tt, p, q_out, k_cache, v_cache, slots, cos_sin, pos):
        """The FUSE owner's head finish with the vLLM flags (VQK and / or VKV): head hh (uniform), token tt (uniform
        in the wave), lane p = pair (2p, 2p + 1) of the head; w2 = the bf16 pair of the qkv GEMV.
        VQK: as vLLM's fused_muse_spark_qknorm_rope_kv_insert: weightless qk-norm, then GPT-J RoPE in fp32 with the
        fp32 cos / sin, one RNE bf16 rounding (bf16_pair); q without qk_scale. V passes through (the bf16 GEMV).
        VKV: K / V pairs as one bf16 dword into the LBHNC cache, element ((slot >> 4) * bstride + (kv * 16 +
        (slot & 15)) * 128 + 64 isv + 2p), bstride = v_cache (the block stride in elements, a value, not a pointer);
        slot < 0 (padded rows) skipped. The block base is 64-bit and wave-uniform (one buffer resource a wave)."""
        x0 = bf_lo(w2)
        x1 = bf_hi(w2)
        ss = butterfly(x0 * x0 + x1 * x1, (1, 2, 4, 8, 16))
        rs = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS))
        isv = hh >= HQ + HKV
        isq = hh < HQ
        if const_expr(VQK):
            n0 = x0 * rs
            n1 = x1 * rs
            if const_expr(key.next_rope):
                ps = ld_i32(pos, tt * 2)  # int64 positions: low word
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
                ps = ld_i32(pos, tt * 2)
                c = bf16_round(ld_f32(cos_sin, ps * HD + p))
                sn = bf16_round(ld_f32(cos_sin, ps * HD + HD // 2 + p))
                r0 = bf16_round(bf16_round(n0 * c) - bf16_round(n1 * sn))
                r1 = bf16_round(bf16_round(n1 * c) + bf16_round(n0 * sn))
                n0 = r0
                n1 = r1
        o0 = isv.select(x0, n0)
        o1 = isv.select(x1, n1)
        row0 = (hh * HD) // 2 + p
        if isq:
            bo.buffer_store(bf16_pair(o0, o1), rsrc(q_out), (tt * QROWS) // 2 + row0)
        else:
            kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
            if const_expr(VKV):
                slot = uniform(
                    ld_i32(slots, tt * 2)
                )  # int64 slot_mapping: low word (-1 = padding)
                if slot >= 0:
                    eoff = fx.Int64(slot >> 4) * v_cache + fx.Int64(
                        (kv * 16 + (slot & 15)) * 128
                        + isv.select(fx.Int32(HD), fx.Int32(0))
                    )
                    bo.buffer_store(
                        bf16_pair(o0, o1), rsrc(k_cache + eoff * fx.Int64(2)), p
                    )
            else:
                if const_expr(VQK):  # vLLM: the fp8 of the bf16-rounded value
                    o0 = bf16_round(o0)
                    o1 = bf16_round(o1)
                w8 = fp8x2(o0, o1)
                w8n = xshfl(w8, 1)
                slot = ld_i32(slots, tt * 2)
                if (p % 2) == 0:
                    off = (slot * (HKV * HD) + kv * HD + 2 * p) // 4
                    dw = w8 | (w8n << 16)
                    if isv:
                        bo.buffer_store(dw, rsrc(v_cache), off)
                    else:
                        bo.buffer_store(dw, rsrc(k_cache), off)

    @traced
    def stage_qkv8(
        q_ops,
        scratch,
        q_out,
        g_out,
        k_cache,
        v_cache,
        slots,
        cos_sin,
        pos,
        tag,
        health,
        tid,
        bid,
        lane,
        wave,
        xl,
        red,
        own,
    ):
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        lane = fresh(lane)
        g = lane // 16
        k8 = bid % 8
        gq = bid // 8
        if tid < 32:
            mb_poll(
                tag,
                health,
                [(scratch + fx.Int64(lay["xnf"][0]), k8 * 32 + tid, 1)],
                REGION_IDS["xnf"],
            )
        gpu.barrier()
        t = fx.min(lane % 16, S - 1)
        xb = []
        if const_expr(S3_XD):
            for qq in range_constexpr(4):
                kk = wave * KCH + 8 * g + 32 * qq
                xb.append(
                    ld_vec(
                        scratch + fx.Int64(lay["xn"][0]),
                        (t * HID + k8 * XQ8 + kk) // 2,
                        4,
                        CM_DEV,
                    )
                )
        else:
            nx = S * XQ8 // 8
            for i in range_constexpr((nx + THREADS - 1) // THREADS):
                e = tid + THREADS * i
                tx = fx.min(e // (XQ8 // 8), S - 1)
                kk = (e % (XQ8 // 8)) * 8
                v = ld_vec(
                    scratch + fx.Int64(lay["xn"][0]),
                    (tx * HID + k8 * XQ8 + kk) // 2,
                    4,
                    CM_DEV,
                )
                if e < nx:
                    fx.ptr_store(v, xl + (tx * X8ROW + kk) // 2)
        if const_expr(S3_XD):
            rocdl.sched_barrier(0)  # keep the x loads issued here, before the MFMA loop
        if const_expr(not S3_XD):
            gpu.barrier()
        for n in range_constexpr(NQT):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for qq in range_constexpr(4):
                if const_expr(S3_XD):
                    b = xb[qq].bitcast(fx.BFloat16)
                else:
                    kk = wave * KCH + 8 * g + 32 * qq
                    b = fx.Vector(
                        fx.ptr_load(
                            xl + (t * X8ROW + kk) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    ).bitcast(fx.BFloat16)
                if const_expr(S3_ML):
                    aop = q_ops[n][qq]
                else:
                    aop = bf_mma_operand(q_ops[n], qq, lane)
                acc = mfma_bf16(aop.bitcast(fx.BFloat16), b, acc)
            fx.ptr_store(acc, red + ((wave * NQT + n) * 64 + lane) * 4)
        gpu.barrier()
        SO = S // 8 if S3_FUSE else 1  # FUSE: tokens a owner (t = k8 + 8 j)
        for it_ in range_constexpr((NQT * 8 * S + THREADS - 1) // THREADS):
            e = tid + THREADS * it_
            if e < NQT * 8 * S:
                n = e // (8 * S)
                rp = (e // S) % 8
                tt = e % S
                v = []
                for h2 in range_constexpr(2):
                    rr = 2 * rp + h2
                    s_ = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        s_ = s_ + fx.ptr_load(
                            red
                            + (((w * NQT + n) * 64 + 16 * (rr // 4) + tt) * 4 + rr % 4)
                        )
                    v.append(s_)
                if const_expr(S3_FUSE):
                    if (tt % 8) == k8:
                        for h2 in range_constexpr(2):
                            fx.ptr_store(
                                v[h2], own + ((n * 16 + 2 * rp + h2) * SO + tt // 8)
                            )
                    else:
                        for h2 in range_constexpr(2):
                            mb_put(
                                tag,
                                scratch + fx.Int64(lay["qx"][0]),
                                (((gq * 8 + k8) * NQT + n) * 16 + 2 * rp + h2) * S + tt,
                                v[h2],
                            )
                elif rp == k8:
                    fx.ptr_store(v[0], own + ((n * 2) * S + tt))
                    fx.ptr_store(v[1], own + ((n * 2 + 1) * S + tt))
                else:
                    for h2 in range_constexpr(2):
                        mb_put(
                            tag,
                            scratch + fx.Int64(lay["qx"][0]),
                            (((gq * 8 + k8) * NQT + n) * 16 + 2 * rp + h2) * S + tt,
                            v[h2],
                        )
        gpu.barrier()
        if const_expr(not S3_FUSE):
            if tid < NQT * S:
                n = tid // S
                tt = tid % S
                specs, oks = [], []
                for src in range_constexpr(8):
                    for h2 in range_constexpr(2):
                        specs.append(
                            (
                                scratch + fx.Int64(lay["qx"][0]),
                                (((gq * 8 + src) * NQT + n) * 16 + 2 * k8 + h2) * S
                                + tt,
                                1,
                            )
                        )
                        oks.append(fx.Int32(src) != k8)
                got = mb_poll(tag, health, specs, REGION_IDS["qx"], oks=oks)
                tot = []
                for h2 in range_constexpr(2):
                    s_ = fx.Float32(0.0)
                    for src in range_constexpr(8):
                        mine = fx.ptr_load(own + ((n * 2 + h2) * S + tt))
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
                        tt * (QKV_ROWS // 2) + row // 2,
                        w2,
                    )
                else:
                    st_if(
                        row < G.QKVG_ROWS, w2, g_out, (tt * QROWS + row - QKV_ROWS) // 2
                    )
        else:
            # owner wave j: token tt = k8 + 8 j; lane p < 40 = (tile n = p / 8, row pair p % 8); a head's 32 pairs
            # are lanes 0 .. 31 (pair p of the head = lane p, as in head_finish)
            if (tid < 64 * SO) & (lane < NQT * 8):
                j = tid // 64
                p = lane
                n = p // 8
                rp = p % 8
                tt = k8 + 8 * j
                specs, oks = [], []
                for src in range_constexpr(8):
                    for h2 in range_constexpr(2):
                        specs.append(
                            (
                                scratch + fx.Int64(lay["qx"][0]),
                                (((gq * 8 + src) * NQT + n) * 16 + 2 * rp + h2) * S
                                + tt,
                                1,
                            )
                        )
                        oks.append(fx.Int32(src) != k8)
                got = mb_poll(tag, health, specs, REGION_IDS["qx"], oks=oks)
                tot = []
                for h2 in range_constexpr(2):
                    s_ = fx.Float32(0.0)
                    for src in range_constexpr(8):
                        mine = fx.ptr_load(own + ((n * 16 + 2 * rp + h2) * SO + j))
                        s_ = s_ + (fx.Int32(src) == k8).select(
                            mine, got[2 * src + h2][0].bitcast(fx.Float32)
                        )
                    tot.append(s_)
                head = gq < NH
                tl = head.select(4 * gq + n, NQKVT + NH + NQT * (gq - NH) + n)
                tl = (head & (n == 4)).select(NQKVT + gq, tl)
                row = tl * 16 + 2 * rp
                w2 = bf16_pair(tot[0], tot[1])
                if row < QKV_ROWS:
                    mb_put(
                        tag,
                        scratch + fx.Int64(lay["qkv"][0]),
                        tt * (QKV_ROWS // 2) + row // 2,
                        w2,
                    )
                else:
                    st_if(
                        row < G.QKVG_ROWS, w2, g_out, (tt * QROWS + row - QKV_ROWS) // 2
                    )
                if const_expr(VQK or VKV):
                    if head & (p < 32):
                        vllm_finish(
                            w2, gq, tt, p, q_out, k_cache, v_cache, slots, cos_sin, pos
                        )
                elif head & (p < 32):
                    hh = gq
                    x0 = bf_lo(w2)
                    x1 = bf_hi(w2)
                    ss = butterfly(x0 * x0 + x1 * x1, (1, 2, 4, 8, 16))
                    rs = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS))
                    isv = hh >= HQ + HKV
                    isq = hh < HQ
                    n0 = bf16_round(x0 * rs)
                    n1 = bf16_round(x1 * rs)
                    n0 = isq.select(bf16_round(n0 * fx.Float32(QK_SCALE)), n0)
                    n1 = isq.select(bf16_round(n1 * fx.Float32(QK_SCALE)), n1)
                    if const_expr(key.next_rope):
                        ps = ld_i32(pos, tt * 2)  # int64 positions: low word
                        c = bf16_round(ld_f32(cos_sin, ps * HD + p))
                        sn = bf16_round(ld_f32(cos_sin, ps * HD + HD // 2 + p))
                        r0 = bf16_round(bf16_round(n0 * c) - bf16_round(n1 * sn))
                        r1 = bf16_round(bf16_round(n1 * c) + bf16_round(n0 * sn))
                        n0 = r0
                        n1 = r1
                    o0 = isv.select(x0, n0)
                    o1 = isv.select(x1, n1)
                    row0 = (hh * HD) // 2 + p
                    if isq:
                        bo.buffer_store(
                            bf16_pair(o0, o1), rsrc(q_out), (tt * QROWS) // 2 + row0
                        )
                    else:
                        w8 = fp8x2(o0, o1)
                        w8n = xshfl(w8, 1)
                        kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
                        slot = ld_i32(slots, tt * 2)
                        if (p % 2) == 0:
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
        lane,
    ):
        """CTA hh < 40 finishes head hh of the next layer (32 q, 4 k, 4 v): qk-norm (SGLang's
        _qk_norm_kernel rounding), x qk_scale (q), RoPE (GPT-J pairs, bf16 per op like
        sgl_kernel.rotary_embedding), q -> q_out bf16, k / v -> fp8 E4M3 into the paged cache
        [block][16][4][64] at slot = slots[t]."""
        tid = fresh(tid)  # no cross-stage CSE of tid-derived offsets (S=16 spills)
        hh = bid
        if (bid < HQ + 2 * HKV) & (tid < 32 * S):
            p = tid % 32  # pair 2p, 2p+1 of the head
            tt = tid // 32
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
            n0 = bf16_round(x0 * rs)
            n1 = bf16_round(x1 * rs)
            n0 = isq.select(bf16_round(n0 * fx.Float32(QK_SCALE)), n0)
            n1 = isq.select(bf16_round(n1 * fx.Float32(QK_SCALE)), n1)
            if const_expr(key.next_rope):
                ps = ld_i32(pos, tt * 2)  # int64 positions: low word
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
                # 2 fp8 bytes a lane; even lanes store the dword of lanes p, p + 1
                w8 = fp8x2(o0, o1)
                w8n = xshfl(w8, 1)
                kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
                slot = ld_i32(slots, tt * 2)
                if (p % 2) == 0:
                    off = (slot * (HKV * HD) + kv * HD + 2 * p) // 4
                    dw = w8 | (w8n << 16)
                    if isv:
                        bo.buffer_store(dw, rsrc(v_cache), off)
                    else:
                        bo.buffer_store(dw, rsrc(k_cache), off)

    # ------------------------------------------------------------ kernel
    @traced
    def k3_body(
        attn_o,
        attn_g,
        stream_pre,
        w_o,
        alpha_a,
        beta_a,
        w_pffn,
        w_r,
        bias,
        w_pre,
        w_pren,
        w13,
        w13s,
        w2,
        w2s,
        w_post,
        w_pp,
        w_pff,
        alpha,
        beta,
        w_in,
        w_qkvg,
        stream_out,
        xn_out,
        part_out,
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
        g_stride,
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
        misc = alloc.allocate(Misc).peek()
        rl = alloc.allocate(RouteLds).peek()
        sl = alloc.allocate(StageLds)
        sa, sc_, sd, se, sf = (
            sl.a_.peek(),
            sl.c_.peek(),
            sl.d_.peek(),
            sl.e_.peek(),
            sl.f_.peek(),
        )
        so, sq = sl.o_.peek(), sl.q_.peek()
        tls = misc.tls.ptr
        rt = {
            "rid": rl.rid.ptr,
            "rw": rl.rw.ptr,
            "uexp": rl.uexp.ptr,
            "kof": rl.kof.ptr,
            "nu": rl.nu.ptr,
            "sel": rl.sel.ptr,
            "plist": rl.plist.ptr,
            "pidx": rl.pidx.ptr,
            "np_": rl.np_.ptr,
            "rstd": rl.rstd.ptr,
            "rank": rank,
            "uslot": rl.uslot.ptr,
            "ssacc": rl.ssacc.ptr,
            "upk": rl.upk.ptr,
            "upn": rl.upn.ptr,
            "wi": rl.wi.ptr,
            "gpk": rl.gpk.ptr,
            "flist": rl.flist.ptr,
            "w13r": w13r,
            "w13sr": w13sr,
            "w2r": w2r,
            "w2sr": w2sr,
            "eptab": eptab,
        }
        tok = rl.tok.ptr
        pl = list(peers)[:W]
        own_peer = pl[0]
        for rr_ in range_constexpr(1, W):
            own_peer = (rank == rr_).select(pl[rr_], own_peer)
        stamp(TL, tls, tid, 0)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        if const_expr(on("O")):
            if const_expr(FA is not None and not key.attn_olate):
                o_ops = o_loads(
                    w_o, bid, lane, wave
                )  # #10: the o_proj weights land during the attention
            if const_expr(
                FA is not None
            ):  # S4 #8: attn_o is the fragment's descriptor; the real pointer comes back
                attn_o = FA.run(
                    attn_o,
                    tag,
                    health,
                    tid,
                    bid,
                    lane,
                    wave,
                    sl.fa_.peek(),
                    True,
                    layer,
                )
                if const_expr(
                    FA.AOT
                ):  # stage O polls the tagged rows (no all-rows wait)
                    attn_o, ao_t = attn_o
            if const_expr(FA is None or key.attn_olate):
                o_ops = o_loads(w_o, bid, lane, wave)
            stage_o(
                o_ops,
                attn_o,
                attn_g,
                g_stride,
                pl,
                tag,
                tid,
                bid,
                lane,
                wave,
                rank,
                so.y.ptr,
                so.red.ptr,
                (ao_t, health) if (FA is not None and FA.AOT) else None,
            )
        stamp(TL, tls, tid, 12)
        if const_expr(on("AB")):
            stage_ab(
                own_peer,
                scratch,
                stream_pre,
                alpha_a,
                beta_a,
                w_pffn,
                tag,
                health,
                tid,
                bid,
                lane,
                rl.rs.ptr,
                tls,
                pl,
                rank,
            )
        stamp(TL, tls, tid, 13)
        if const_expr(on("A")):
            bias_v = stage_a(
                0,
                w_r,
                bias,
                w_pre,
                scratch,
                own_peer,
                pl,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                rank,
                sa.x.ptr,
                sa.red.ptr,
                sa.own.ptr,
                tok,
                tls,
            )
        else:
            bias_v = [
                ld_f32(bias, lane + 64 * i) for i in range(4)
            ]  # stage A's bias read (route needs it)
        q, hv, wv, t0, t1, last = front2(
            own_peer,
            bias_v,
            w13,
            w13s,
            w_pren,
            scratch,
            tag,
            health,
            rank,
            tid,
            lane,
            wave,
            rt,
            bid,
            tls,
        )
        if const_expr(on("h")):
            load_h_norm(hv, wv, tid, lane, wave, sc_.h.ptr, tok)
        stamp(TL, tls, tid, 4)
        if const_expr(on("gate_up")):
            task = t0
            it = fx.Int32(0)
            if const_expr(UNR):
                # ring: three tasks an iteration in fixed slots (q[0], q[1], qc): two tasks' loads stay in
                # flight during each compute, no rotation copy waits for its data (the tail: the loop below)
                while task + 2 < t1:
                    qc = gu_loads(w13, w13s, rt, task + 2, lane, wave, (task + 2) < t1)
                    gu_compute(
                        q[0],
                        task,
                        it,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sc_.h.ptr,
                        sc_.red.ptr,
                    )
                    qa = gu_loads(w13, w13s, rt, task + 3, lane, wave, (task + 3) < t1)
                    gu_compute(
                        q[1],
                        task + 1,
                        it + 1,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sc_.h.ptr,
                        sc_.red.ptr,
                    )
                    qb = gu_loads(w13, w13s, rt, task + 4, lane, wave, (task + 4) < t1)
                    gu_compute(
                        qc,
                        task + 2,
                        it + 2,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sc_.h.ptr,
                        sc_.red.ptr,
                    )
                    q = [qa, qb]
                    task = task + 3
                    it = it + 3
            while task < t1:
                nxt = gu_loads(
                    w13, w13s, rt, task + PF_GU, lane, wave, (task + PF_GU) < t1
                )
                gu_compute(
                    q[0],
                    task,
                    it,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    sc_.h.ptr,
                    sc_.red.ptr,
                )
                q = q[1:] + [nxt]
                task = task + 1
                it = it + 1
        stamp(TL, tls, tid, 5)
        if const_expr(on("down")):
            dq = [dn_loads(w2, w2s, w_post, rt, t0, lane, wave, t0 < t1)]
            gpu.barrier()
            u0 = t0 >> 8
            u1 = last >> 8
            dn_build_picks(rt, u0, u1, tid)
            load_ac(scratch, tag, health, rt, False, sd.a.ptr, tid)
            dq = dq + [
                dn_loads(w2, w2s, w_post, rt, t0 + d, lane, wave, (t0 + d) < t1)
                for d in range(1, PF_DN)
            ]
            if tid < 2 * S:
                fx.ptr_store(fx.Float32(0.0), rt["ssacc"] + tid)
            gpu.barrier()
            stamp(TL, tls, tid, 6)
            task = t0
            it = fx.Int32(0)
            tsplit = fx.min(t1, (u0 + 1) << 8)
            if const_expr(UNRD):
                while task + 2 < tsplit:
                    dc = dn_loads(
                        w2, w2s, w_post, rt, task + 2, lane, wave, (task + 2) < t1
                    )
                    dn_compute_c(
                        dq[0],
                        task,
                        it,
                        u0,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sd.a.ptr,
                        sd.red.ptr,
                    )
                    da = dn_loads(
                        w2, w2s, w_post, rt, task + 3, lane, wave, (task + 3) < t1
                    )
                    dn_compute_c(
                        dq[1],
                        task + 1,
                        it + 1,
                        u0,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sd.a.ptr,
                        sd.red.ptr,
                    )
                    db = dn_loads(
                        w2, w2s, w_post, rt, task + 4, lane, wave, (task + 4) < t1
                    )
                    dn_compute_c(
                        dc,
                        task + 2,
                        it + 2,
                        u0,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sd.a.ptr,
                        sd.red.ptr,
                    )
                    dq = [da, db]
                    task = task + 3
                    it = it + 3
            while task < tsplit:
                nxt = dn_loads(
                    w2, w2s, w_post, rt, task + PF_DN, lane, wave, (task + PF_DN) < t1
                )
                dn_compute_c(
                    dq[0],
                    task,
                    it,
                    u0,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    sd.a.ptr,
                    sd.red.ptr,
                )
                dq = dq[1:] + [nxt]
                task = task + 1
                it = it + 1
            dn_switch(scratch, tag, health, rt, tsplit, u0, u1, sd.a.ptr, tid)
            if const_expr(UNRD):
                while task + 2 < t1:
                    dc = dn_loads(
                        w2, w2s, w_post, rt, task + 2, lane, wave, (task + 2) < t1
                    )
                    dn_compute_c(
                        dq[0],
                        task,
                        it,
                        u0,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sd.a.ptr,
                        sd.red.ptr,
                    )
                    da = dn_loads(
                        w2, w2s, w_post, rt, task + 3, lane, wave, (task + 3) < t1
                    )
                    dn_compute_c(
                        dq[1],
                        task + 1,
                        it + 1,
                        u0,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sd.a.ptr,
                        sd.red.ptr,
                    )
                    db = dn_loads(
                        w2, w2s, w_post, rt, task + 4, lane, wave, (task + 4) < t1
                    )
                    dn_compute_c(
                        dc,
                        task + 2,
                        it + 2,
                        u0,
                        scratch,
                        tag,
                        rt,
                        tid,
                        lane,
                        wave,
                        sd.a.ptr,
                        sd.red.ptr,
                    )
                    dq = [da, db]
                    task = task + 3
                    it = it + 3
            while task < t1:
                nxt = dn_loads(
                    w2, w2s, w_post, rt, task + PF_DN, lane, wave, (task + PF_DN) < t1
                )
                dn_compute_c(
                    dq[0],
                    task,
                    it,
                    u0,
                    scratch,
                    tag,
                    rt,
                    tid,
                    lane,
                    wave,
                    sd.a.ptr,
                    sd.red.ptr,
                )
                dq = dq[1:] + [nxt]
                task = task + 1
                it = it + 1
            gpu.barrier()
            if tid < 2 * S:
                h2 = tid // S
                tt = tid % S
                u = u0 + h2
                k = fx.ptr_load(rt["kof"] + (fx.min(u, NU - 1) * S + tt))
                if ((h2 == 0) | (u1 != u0)) & (k >= 0):
                    cf = (u * 256) // (t1 - t0)
                    mb_put(
                        tag,
                        scratch + fx.Int64(lay["ss2"][0]),
                        (tt * TOPK + k) * 256 + (bid - cf),
                        fx.ptr_load(rt["ssacc"] + tid),
                    )
        stamp(TL, tls, tid, 7)
        # post_expert_proj weights in flight before the all-reduce wait (pp_early: before
        # the final stage's polls too)
        if const_expr(key.pp_early and on("post")):
            ppo = pp_loads(w_pp, bid, lane, wave)
        gpu.barrier()
        if const_expr(on("final")):
            stage_final(scratch, part_out, tag, health, rt, tid, bid, lane, se.z.ptr)
            final_push(scratch, part_out, pl, tag, rt, tid, bid, lane, se.z.ptr, rank)
        if const_expr((not key.pp_early) and on("post")):
            ppo = pp_loads(w_pp, bid, lane, wave)
        stamp(TL, tls, tid, 8)
        if const_expr(on("ar")):
            ar_reduce(scratch, own_peer, tag, health, tid, bid, pl, rank)
        stamp(TL, tls, tid, 9)
        if const_expr(on("post")):
            post_proj(
                ppo,
                scratch,
                own_peer,
                pl,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                rank,
                sf.mm.ptr,
                sf.red.ptr,
            )
        stamp(TL, tls, tid, 10)
        if const_expr(on("QKV")):
            if const_expr(S3Q):
                qo = q8_loads(w_qkvg, bid, lane, wave, 0, QPF)
            else:
                qo = q_loads(w_qkvg, bid, lane, wave, 0, QPF)
        if const_expr(on("G")):
            stage_g(
                own_peer,
                scratch,
                0,
                w_pff,
                alpha,
                beta,
                w_in,
                stream_out,
                xn_out,
                tag,
                health,
                tid,
                bid,
                lane,
                rl.rs.ptr,
                tls,
            )
        stamp(TL, tls, tid, 11)
        if const_expr(on("QKV") and S3Q):
            qo = qo + q8_loads(w_qkvg, bid, lane, wave, QPF, NQT)
            stage_qkv8(
                qo,
                scratch,
                q_out,
                g_out,
                k_cache,
                v_cache,
                slots,
                cos_sin,
                pos,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                sq.xq.ptr,
                sq.red.ptr,
                sq.own.ptr,
            )
        elif const_expr(on("QKV")):
            qo = qo + q_loads(w_qkvg, bid, lane, wave, QPF, NQT)
            stage_qkv(
                qo,
                scratch,
                g_out,
                tag,
                health,
                tid,
                bid,
                lane,
                wave,
                sq.xq.ptr,
                sq.red.ptr,
                sq.own.ptr,
            )
        stamp(TL, tls, tid, 14)
        if const_expr(on("finish") and not S3_FUSE):
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
                lane,
            )
        stamp(TL, tls, tid, 15)
        if const_expr(on("route")):
            if (bid == 0) & (tid < P):
                bo.buffer_store(
                    fx.ptr_load(rt["rid"] + tid),
                    rsrc(scratch + fx.Int64(lay["route"][0])),
                    tid,
                )
                bo.buffer_store(
                    fx.ptr_load(rt["rw"] + tid),
                    rsrc(scratch + fx.Int64(lay["route"][0])),
                    P + tid,
                )
        stamp_flush(TL, tls, tl, tid, bid, TL_POINTS)

    @traced
    def front2(
        own_peer,
        bias_v,
        w13,
        w13s,
        w_pren,
        scratch,
        tag,
        health,
        rank,
        tid,
        lane,
        wave,
        rt,
        bid,
        tls,
    ):
        ept = None
        if const_expr(RB > 0 and on("route")):
            ept = [ld_i32(rt["eptab"], lane + 64 * hh) for hh in range(E // 64)]
        if const_expr(on("route")):
            lgs_r = route_poll(own_peer, tag, health, lane, wave)
            stage_route(
                scratch, bias_v, lgs_r, tag, health, rank, tid, lane, wave, rt, ept=ept
            )
        stamp(TL, tls, tid, 3)
        nu = uniform(fx.ptr_load(rt["nu"] + 0))
        t0 = bid * nu
        t1 = t0 + nu
        last = t1 - 1
        q, hv, wv = [], None, None
        if const_expr(on("gate_up")):
            q = [gu_loads(w13, w13s, rt, t0, lane, wave, t0 < t1)]
        if const_expr(on("h")):
            hv, wv = h_wait_issue(own_peer, w_pren, tag, health, tid, rank)
        if const_expr(on("gate_up")):
            q = q + [
                gu_loads(w13, w13s, rt, t0 + d, lane, wave, (t0 + d) < t1)
                for d in range(1, PF_GU)
            ]
        return q, hv, wv, t0, t1, last

    # the kernel and its launcher: peer0 .. peer3 at W <= 4 (the TP4 signature, unchanged), peer0 .. peer7 at W = 8
    if W <= 4:

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def k3(
            attn_o: Int64,
            attn_g: Int64,
            stream_pre: Int64,
            w_o: Int64,
            alpha_a: Int64,
            beta_a: Int64,
            w_pffn: Int64,
            w_r: Int64,
            bias: Int64,
            w_pre: Int64,
            w_pren: Int64,
            w13: Int64,
            w13s: Int64,
            w2: Int64,
            w2s: Int64,
            w_post: Int64,
            w_pp: Int64,
            w_pff: Int64,
            alpha: Int64,
            beta: Int64,
            w_in: Int64,
            w_qkvg: Int64,
            stream_out: Int64,
            xn_out: Int64,
            part_out: Int64,
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
            g_stride: Int32,
            w13r: Int64,
            w13sr: Int64,
            w2r: Int64,
            w2sr: Int64,
            eptab: Int64,
        ):
            k3_body(
                attn_o,
                attn_g,
                stream_pre,
                w_o,
                alpha_a,
                beta_a,
                w_pffn,
                w_r,
                bias,
                w_pre,
                w_pren,
                w13,
                w13s,
                w2,
                w2s,
                w_post,
                w_pp,
                w_pff,
                alpha,
                beta,
                w_in,
                w_qkvg,
                stream_out,
                xn_out,
                part_out,
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
                (peer0, peer1, peer2, peer3),
                layer,
                rank,
                g_stride,
                w13r,
                w13sr,
                w2r,
                w2sr,
                eptab,
            )

        @flyc.jit
        def launch(
            attn_o: Int64,
            attn_g: Int64,
            stream_pre: Int64,
            w_o: Int64,
            alpha_a: Int64,
            beta_a: Int64,
            w_pffn: Int64,
            w_r: Int64,
            bias: Int64,
            w_pre: Int64,
            w_pren: Int64,
            w13: Int64,
            w13s: Int64,
            w2: Int64,
            w2s: Int64,
            w_post: Int64,
            w_pp: Int64,
            w_pff: Int64,
            alpha: Int64,
            beta: Int64,
            w_in: Int64,
            w_qkvg: Int64,
            stream_out: Int64,
            xn_out: Int64,
            part_out: Int64,
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
            g_stride: Int32,
            w13r: Int64,
            w13sr: Int64,
            w2r: Int64,
            w2sr: Int64,
            eptab: Int64,
            stream: fx.Stream = fx.Stream(None),
        ):
            _ = _KEYSTR
            k3(
                attn_o,
                attn_g,
                stream_pre,
                w_o,
                alpha_a,
                beta_a,
                w_pffn,
                w_r,
                bias,
                w_pre,
                w_pren,
                w13,
                w13s,
                w2,
                w2s,
                w_post,
                w_pp,
                w_pff,
                alpha,
                beta,
                w_in,
                w_qkvg,
                stream_out,
                xn_out,
                part_out,
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
                g_stride,
                w13r,
                w13sr,
                w2r,
                w2sr,
                eptab,
            ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)
    else:

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def k3(
            attn_o: Int64,
            attn_g: Int64,
            stream_pre: Int64,
            w_o: Int64,
            alpha_a: Int64,
            beta_a: Int64,
            w_pffn: Int64,
            w_r: Int64,
            bias: Int64,
            w_pre: Int64,
            w_pren: Int64,
            w13: Int64,
            w13s: Int64,
            w2: Int64,
            w2s: Int64,
            w_post: Int64,
            w_pp: Int64,
            w_pff: Int64,
            alpha: Int64,
            beta: Int64,
            w_in: Int64,
            w_qkvg: Int64,
            stream_out: Int64,
            xn_out: Int64,
            part_out: Int64,
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
            g_stride: Int32,
            w13r: Int64,
            w13sr: Int64,
            w2r: Int64,
            w2sr: Int64,
            eptab: Int64,
        ):
            k3_body(
                attn_o,
                attn_g,
                stream_pre,
                w_o,
                alpha_a,
                beta_a,
                w_pffn,
                w_r,
                bias,
                w_pre,
                w_pren,
                w13,
                w13s,
                w2,
                w2s,
                w_post,
                w_pp,
                w_pff,
                alpha,
                beta,
                w_in,
                w_qkvg,
                stream_out,
                xn_out,
                part_out,
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
                (peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7),
                layer,
                rank,
                g_stride,
                w13r,
                w13sr,
                w2r,
                w2sr,
                eptab,
            )

        @flyc.jit
        def launch(
            attn_o: Int64,
            attn_g: Int64,
            stream_pre: Int64,
            w_o: Int64,
            alpha_a: Int64,
            beta_a: Int64,
            w_pffn: Int64,
            w_r: Int64,
            bias: Int64,
            w_pre: Int64,
            w_pren: Int64,
            w13: Int64,
            w13s: Int64,
            w2: Int64,
            w2s: Int64,
            w_post: Int64,
            w_pp: Int64,
            w_pff: Int64,
            alpha: Int64,
            beta: Int64,
            w_in: Int64,
            w_qkvg: Int64,
            stream_out: Int64,
            xn_out: Int64,
            part_out: Int64,
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
            g_stride: Int32,
            w13r: Int64,
            w13sr: Int64,
            w2r: Int64,
            w2sr: Int64,
            eptab: Int64,
            stream: fx.Stream = fx.Stream(None),
        ):
            _ = _KEYSTR
            k3(
                attn_o,
                attn_g,
                stream_pre,
                w_o,
                alpha_a,
                beta_a,
                w_pffn,
                w_r,
                bias,
                w_pre,
                w_pren,
                w13,
                w13s,
                w2,
                w2s,
                w_post,
                w_pp,
                w_pff,
                alpha,
                beta,
                w_in,
                w_qkvg,
                stream_out,
                xn_out,
                part_out,
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
                g_stride,
                w13r,
                w13sr,
                w2r,
                w2sr,
                eptab,
            ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


def _eplb_for(S):
    from .eplb import eplb_for

    return eplb_for(S)


def build_from_env(S=8, W=4, **over):
    # TEP8 (W = 8) defaults of the S3 front: MK_S3Q=19 (QKV: MFMA-layout weights 16 + x in registers 2 + head
    # finish in the K-split owners 1), MK_QKV_PF=4, MK_S3O=1 (O: MFMA-layout weights); 0 = the previous kernels
    S3Q_W8 = int(os.environ.get("MK_S3Q", "19"))
    kw = dict(
        S=S,
        W=W,
        shard=os.environ.get("MK_SHARD", "1") == "1",
        pf_gu=int(os.environ.get("MK_PF_GU", "2" if S <= 8 else "1")),
        pf_dn=int(os.environ.get("MK_PF_DN", "2")),
        gusplit=os.environ.get("MK_GUSPLIT", "1" if S > 8 else "0") == "1",
        ac_cap=int(os.environ.get("MK_AC_CAP", "0")),
        pp_early=os.environ.get("MK_PP_EARLY", "1" if over.get("shard", True) else "0")
        == "1",
        next_rope=over.pop("next_rope", True),
        qkv_pf=int(os.environ.get("MK_QKV_PF", "4" if W == 8 and S3Q_W8 else "3")),
        timeline=os.environ.get("MK_TIMELINE", "0") == "1",
        stop=int(os.environ.get("MK_STOP", "99")),
        onepass=os.environ.get("MK_ONEPASS", "0") == "1",
        eplb=_eplb_for(S),
        xrs=os.environ.get("MK_XRS", "0") == "1",
        unr=int(os.environ.get("MK_UNR", "-1")),
        only=os.environ.get("MK_ONLY", ""),
        p12=os.environ.get("MK_P12", "1") == "1"
        and W == 8
        and os.environ.get("MK_XRS", "0") == "1"
        and os.environ.get("MK_ONEPASS", "0") == "1",
        s3q=S3Q_W8 if W == 8 else 0,
        s3o=int(os.environ.get("MK_S3O", "1")) if W == 8 else 0,
        shuffled_w=os.environ.get("MK_SHUFFLED_W", "0") == "1",
        shuffled_s=os.environ.get("MK_SHUFFLED_S", "0") == "1",
    )
    if (
        os.environ.get("MK_FUSED_ATTN", "0") == "1" and W == 8
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


def build_vllm(S=8, W=8, next_rope=True, **over):
    """The K3 build that vLLM main launches (TEP8, c = 1 / 2). No env reads: the shipped TEP8 knobs
    (onepass, xrs, p12, s3q = 19, s3o = 1, no EPLB) plus the vLLM flags (mult1, vqk, kvl = "lbhnc") and the
    shuffled-weight / scale flags when this tree has them (vLLM's shuffle_weight((16, 16)) / e8m0_shuffle in place)."""
    import dataclasses

    assert W == 8 and S in (8, 16), (S, W)
    kw = dict(
        S=S,
        W=W,
        shard=True,
        pf_gu=2 if S <= 8 else 1,
        pf_dn=2,
        gusplit=S > 8,
        ac_cap=0,
        pp_early=True,
        next_rope=bool(next_rope),
        qkv_pf=4,
        timeline=False,
        stop=99,
        onepass=True,
        eplb=0,
        xrs=True,
        unr=-1,
        only="",
        p12=True,
        s3q=19,
        s3o=1,
        attn="",
        attn_olate=False,
        mult1=True,
        vqk=True,
        kvl="lbhnc",
    )
    names = {f.name for f in dataclasses.fields(Build)}
    for f in ("shuffled_w", "shuffled_s"):
        if f in names:
            kw[f] = True
    kw.update(over)
    return Build(**kw)
