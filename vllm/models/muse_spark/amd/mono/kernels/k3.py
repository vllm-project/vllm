# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008, E501

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
"""K3: one Muse Spark decoder layer for S = 8 / 16 rows as one persistent launch.

One rank of a TP = EP = 8 group (rank r owns experts [32 r, 32 r + 32)), 256 CTAs x 512
threads, everything between two attention cores:

    [attention]  optional: this layer's verify attention (attn_frag.py) at the head
    O      o-norm x sigmoid(gate) -> o_proj rows 32 b .. over this rank's K slice; the bf16
           partial goes to the owner rank of the column block (b // 32): reduce-scatter
    AB     the owner sums the 8 partials in rank order and all-gathers the reduced o and its
           norm partials; every CTA: post-attention norm, gated residual on the fp32 stream,
           pre-FFN norm -> ffn_in (one token-reduction round for both norms)
    A      router logits (fp32) + pre_expert_proj of this rank's shard, K in 8 parts over
           8-CTA groups; logit partials and h_raw go to every rank's peer buffer
    route  every CTA routes every token (identical top-8), the local experts' picks
    C, D   gate_up / down of the local experts: MXFP4 weights in vLLM's shuffled layouts,
           decoded in registers, bf16 MFMA
    E      per-pick post-expert norm and weighted sum -> the EP partial, reduce-scattered to
           the block owner, summed in rank order, all-gathered
    F      post_expert_proj (this rank's 1024 rows, K in 4 parts) -> every rank
    G      post-FFN norm, gated residual, the next layer's input norm -> stream (fp32) and
           the next attention input (bf16)
    QKV    the next layer's qkv + gate GEMV (K in eighths); the owners finish whole heads:
           qk-norm, RoPE (next_rope), q out, bf16 K / V into vLLM's LBHNC KV cache

Peer regions are double buffered by tag parity; tag = epoch * 64 + layer + 1 with the
epoch bumped in lockstep by every rank once per forward.
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
    permlane_swap,
    popcount,
    readlane,
    rsrc,
    st_if,
    traced,
    uniform,
    wave_incl_scan,
    wave_sum,
)

HID = 8192
DM = 4096
INTER = 4096
E = 256
TOPK = 8
W = 8  # ranks in the TP = EP group
EL = E // W  # local experts
ROUTE_EPS = 1e-15
PRE_EPS = 1e-5
POST_EPS = 1e-8
XPAD = 8
HROW = DM + XPAD
AROW = INTER + XPAD
KCH = 128
FCH = 256
W2_BYTES = EL * DM * (INTER // 2)
W2S_BYTES = EL * DM * (INTER // 32)
KQ = 8  # stage-A K split (CTAs a group)
NG = BLOCKS // KQ  # stage-A groups
XQ = HID // KQ  # stage-A K a CTA
XQROW = XQ + XPAD
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
}
HEALTH_WORDS = 8 + 8 * BLOCKS
HQ, HKV, HD = 128 // W, 16 // W, 64  # q / kv heads a rank
QROWS, KROWS = HQ * HD, HKV * HD
QKV_ROWS = QROWS + 2 * KROWS  # qkvg rows a rank: q 1024 | k 128 | v 128 | gate 1024
QKVG_ROWS = QKV_ROWS + QROWS
OK_K = HID // W  # o_proj K a rank
NQTILES = QKVG_ROWS // 16  # qkvg 16-row tiles a rank
# qkvg tiles a CTA group (32 groups x 8 K eighths; the last group padded)
NQT = -(-NQTILES // 32)


def bf_tile_load_mma(wbase, row16, ldk, k, lane, cm=CM_NT):
    """16 rows (row16 ..) x 128 K (k ..) of a row-major bf16 [*, ldk] weight, loaded straight in the
    16x16x32 bf16 MFMA A-operand layout: element qq (of 4) is the operand of MFMA step qq (lane l = row
    l % 16, K 32 qq + 8 (l / 16) .. + 8), 16 rows x 64 B an instruction. The same operand values as
    bf_tile_load + bf_mma_operand (bit-identical results) without the 8 ds_bpermute a step."""
    rowm = row16 + lane % 16
    kk = k + 8 * (lane // 16)
    return [ld_vec(wbase, (rowm * ldk + kk + 32 * qq) // 2, 4, cm) for qq in range(4)]


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
                ld_vec(wbase, (rowc * ldk + k + 64 * h + 8 * (lane % 8)) // 2, 4, CM_NT)
            )
    return out


def okey(v):
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@traced
def route_token(tkns, bias_v, lgss, tid, lane, rt):
    """Tokens ``tkns`` (this wave's, all valid), interleaved for ILP: sigmoid +
    bias, radix top-8, rank, renorm -> rid / rw."""
    T_ = len(tkns)
    keys, sigs = [], []
    for x_ in range_constexpr(T_):
        kx, sx = [], []
        for i in range_constexpr(4):
            lg = lgss[x_][i].bitcast(fx.Float32)
            sg = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bias_v[i]
            live = sc == sc  # NaN ranks last
            kx.append(okey(live.select(sc, fx.Float32(-3.0e38))))
            sx.append((sg == sg).select(sg, fx.Float32(0.0)))
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
            sel = (keys[x_][i] > th[x_]) | ((keys[x_][i] == th[x_]) & (eq_rank < need))
            selb = ballot(sel)
            pos = sel_before + lanes_below(selb, lane)
            if sel:
                fx.ptr_store(keys[x_][i].bitcast(fx.Int32), sel_base + pos * 3)
                fx.ptr_store(fx.Int32(lane + 64 * i), sel_base + (pos * 3 + 1))
                fx.ptr_store(sigs[x_][i].bitcast(fx.Int32), sel_base + (pos * 3 + 2))
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


def sel_peer(pl, d):
    p = pl[0]
    for r_ in range_constexpr(1, W):
        p = (d == r_).select(pl[r_], p)
    return p


def fp4_mma(ops, base, bl, brow, lane, acc, c, jpar, rows, row0=0):
    """Chunk c (256 K) of a 16-row MXFP4 group: 8 MFMA steps, B column n = LDS row row0 + n (at
    most rows - 1: columns past the expert's picks read some row, their outputs are never stored).
    ``jpar``: the task's 16-row group parity (j & 1), for the shuffled scales."""
    g = lane // 16
    t = fx.min(row0 + lane % 16, rows - 1)
    sw = ops[base + 2]
    # a = the dword of the lane's lower-half partner (K block % 4 = 2 (g % 2)), b = of the upper-half one
    # (2 (g % 2) + 1); the byte = 2 (g // 2) (K block + 4) + the row group parity (rows + 16)
    sa_, sb_ = permlane_swap(32, sw, sw)
    shs = (g // 2) * 16 + jpar * 8
    s0 = e8m0_f32((sa_ >> shs) & 0xFF)
    s1 = e8m0_f32((sb_ >> shs) & 0xFF)
    # loaded (shuffled): vector q lane 16 p + r' = row r', piece 4 q + p; MFMA lane 16 g + r wants row r
    # pieces 2 g, 2 g + 1: X0 = [V0 p0, V0 p2, V1 p0, V1 p2], X1 = [V0 p1, V0 p3, V1 p1, V1 p3] (16-lane
    # rows) = permlane32_swap then permlane16_swap, no bpermute
    x0, x1 = [], []
    for jj in range_constexpr(4):
        a_, b_ = permlane_swap(32, ops[base][jj], ops[base + 1][jj])
        p_, q_ = permlane_swap(16, a_, b_)
        x0.append(p_)
        x1.append(q_)
    xs = [x0, x1]
    for d in range_constexpr(8):
        word = xs[d // 4][d % 4]
        a = fp4x8_bf16(word, s0 if d < 4 else s1)
        kk = FCH * c + 64 * g + 8 * d
        b = fx.Vector(
            fx.ptr_load(
                bl + (t * brow + kk) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(a, b, acc)
    return acc


def mx_chunk(ok, w, ws, rbase, k, c, lane, g, r, nbytes, sbytes):
    """Rows rbase .. rbase + 16 x K chunk c (256) of a shuffled MXFP4 [*, k] weight w and its e8m0
    scales ws: 2 dwordx4 + 1 scale dword. vLLM shuffle_weight((16, 16)): lane L reads the 16 B at
    16 L of the 1 KB block (16 rows x K bytes 64 (2 c + q) ..): one instruction = 1 KB contiguous;
    fp4_mma's permlane swaps map the pieces to the MFMA lanes. e8m0_shuffle: lane 16 g + r reads
    the dword (rbase // 32, c, K block % 4 = 2 (g % 2) + g // 2, row r): rows r, r + 16 x K blocks
    of 4 apart; 64 lanes = 256 B contiguous (kernel_s_dword)."""
    wb = (rbase * (k // 2) + (2 * c) * 1024 + lane * 16) // 4
    ops = [
        ld_vec_if(ok, w, wb, 4, nbytes, CM_NT),
        ld_vec_if(ok, w, wb + 256, 4, nbytes, CM_NT),
    ]
    sdw = (rbase >> 5) * (8 * (k // 32)) + 64 * c + 16 * (2 * (g % 2) + g // 2) + r
    return ops + [ld_i32_if(ok, ws, sdw, sbytes)]


def dn_loads(w2, w2s, w_post, etab, task, lane, wave, ok, nv=None):
    """Task (u, j)'s w2 operands for this wave (expert etab[u], u clamped below nv)."""
    r = lane % 16
    g = lane // 16
    task = ok.select(task, fx.Int32(0))
    u = task >> 8
    j = task & 255
    e = uniform(fx.ptr_load(etab + (u if nv is None else fx.min(u, nv - 1))))
    rbase = e * DM + j * 16
    ops = []
    for i in range_constexpr(2):
        c = 2 * wave + i
        ops += mx_chunk(ok, w2, w2s, rbase, INTER, c, lane, g, r, W2_BYTES, W2S_BYTES)
    # the epilogue's w_post pair (rows 16j + 2 (lane % 8), +1): loaded with the weights,
    # not in the epilogue, where it would wait behind the whole prefetch queue
    ops.append(ld_i32_if(ok, w_post, 8 * j + lane % 8, DM * 2))
    return ops


@traced
def task_ring(load, compute, q, task, it, end, unr, *args):
    """Tasks task .. end - 1: compute(q[0], task, it, *args) with the next len(q) tasks' loads
    (load(task)) in flight. ``unr`` (len(q) = 2): three tasks an iteration in fixed slots (q[0],
    q[1], qc): two tasks' loads stay in flight during each compute, no rotation copy waits for its
    data (the tail: the second loop)."""
    if const_expr(unr):
        while task + 2 < end:
            qc = load(task + 2)
            compute(q[0], task, it, *args)
            qa = load(task + 3)
            compute(q[1], task + 1, it + 1, *args)
            qb = load(task + 4)
            compute(qc, task + 2, it + 2, *args)
            q = [qa, qb]
            task = task + 3
            it = it + 3
    while task < end:
        nxt = load(task + len(q))
        compute(q[0], task, it, *args)
        q = q[1:] + [nxt]
        task = task + 1
        it = it + 1
    return q, task, it


REGION_IDS.update({"orr": 24, "mrr": 25, "ssa4p": 26})
REGION_IDS.update(
    {"op": 13, "xf": 16, "xnf": 17, "qx": 18, "qkv": 19, "ssa4": 20, "ssg4": 21}
)


@dataclass(frozen=True)
class Build:
    S: int = 8  # rows (8 or 16)
    next_rope: bool = True  # the next layer has RoPE (SWA) or not (full, NoPE)
    attn: str = ""  # this layer's verify attention fused at the head ("swa" / "full"); "" = run before the launch


def _lay(items):
    out, off = {}, 0
    for name, n in items:
        off = (off + 255) // 256 * 256
        out[name] = (off, n)
        off += n
    return out, (off + 4095) // 4096 * 4096


def scratch_layout(S, world=W):
    """This rank's scratch regions. ss, ss1, ss2g, ssa1 and ssa2 are unused; they keep the
    offsets of the regions after them."""
    assert world == W, world
    P = S * TOPK
    return _lay(
        [
            ("px", NG * KQ * 64 * S * 8),  # [group][K part][row < RT <= 64][t]
            ("a", P * (INTER // 2) * 8),
            ("z", P * (DM // 2) * 8),
            ("ss", P * 256 * 8),
            ("ss2", P * 256 * 8),
            ("m", S * DM * 2),  # all-reduced MoE output (bf16, plain)
            ("mf", BLOCKS * 8),  # m tile flags
            ("py", 512 * 16 * S * 8),  # post-proj K-part partials
            ("ss1", (S * BLOCKS + S) * 8),
            ("ss2g", (S * BLOCKS + S) * 8),
            ("route", P * 8),
            ("xin", S * HID * 2),  # ffn_in (bf16 plain), CTA b's 32 columns each
            ("xf", BLOCKS * 8),  # its flags
            ("smid", S * HID * 4),  # stream after the attention residual (fp32 plain)
            ("ssa1", (S * BLOCKS + S) * 8),
            ("ssa2", (S * BLOCKS + S) * 8),
            ("xn", S * HID * 2),  # next layer's attention input (bf16 plain)
            ("xnf", BLOCKS * 8),
            # AB: 4 partials [t][CTA][4], then (r1, r2)[t]
            ("ssa4", (4 * S * BLOCKS + 2 * S) * 8),
            ("ssg4", (4 * S * BLOCKS + 2 * S) * 8),  # G
            # qkvg K-eighth partials [group][src][tile row][t]
            ("qx", 32 * 8 * NQT * 16 * S * 8),
            # q / k / v rows (tagged bf16 pairs)
            ("qkv", S * (QKV_ROWS // 2) * 8),
        ]
    )


def peer_layout(S, world=W):
    """Every rank's peer buffer (IPC-shared), [parity] first."""
    assert world == W, world
    return _lay(
        [
            ("logit", 2 * KQ * S * E * 8),  # [par][kq][t][e] tagged
            ("hraw", 2 * S * DM * 2),  # [par][t][col] bf16 plain
            ("hf", 2 * W * BLOCKS * 8),  # [par][src][CTA] tagged flags
            ("moep", 2 * W * S * (DM // 2) * 8),  # [par][src][t][pair] tagged
            ("y", 2 * S * (HID // 2) * 8),  # [par][t][pair] tagged
            # o_proj partials [par][src][t][pair] tagged
            ("op", 2 * W * S * (HID // 2) * 8),
            # the owner's reduced o_proj [par][t][pair] tagged (all-gather)
            ("orr", 2 * S * (HID // 2) * 8),
            # the owner's reduced EP sum [par][t][pair] tagged (all-gather)
            ("mrr", 2 * S * (DM // 2) * 8),
            # the AB owners' norm partials [par][t][CTA][4] tagged (all-gather)
            ("ssa4p", 2 * 4 * S * BLOCKS * 8),
        ]
    )


_BUILDS = {}


def build(key: Build):
    if key in _BUILDS:
        return _BUILDS[key]
    # FlyDSL keys its disk cache on sources + *scalar* closure values; the Build
    # dataclass is not a scalar, so every knob read as key.x would share one binary.
    # Referencing this string from the launcher puts the whole key into the cache key.
    _KEYSTR = repr(key)
    S = key.S
    assert S in (8, 16), S
    XQROW = XQ + XPAD
    CA = XQ // KCH // WAVES  # stage-A pre-proj chunks a wave
    CO = OK_K // KCH // WAVES  # o_proj chunks a wave
    QPF = 4  # qkvg tiles issued before stage G (of NQT)
    P = S * TOPK
    RROUNDS = (S + WAVES - 1) // WAVES
    lay, _ = scratch_layout(S)
    play, _ = peer_layout(S)
    PF_GU = 2 if S == 8 else 1  # gate_up tasks in flight
    PF_DN = 2  # down tasks in flight
    # gate_up: the two 16-row halves of a task in separate MFMA passes
    GUSPLIT = S == 16
    W13_BYTES = EL * 2 * INTER * (DM // 2)
    W13S_BYTES = EL * 2 * INTER * (DM // 32)
    # stage A: a KQ-CTA group (K parts) covers RT pre-proj rows and EG router experts
    RS_PRE = DM // W
    RT = RS_PRE // NG  # 512 / 32 = 16
    NT_A = RT // 16  # MFMA tiles a CTA
    OWN = RT // KQ  # rows a CTA owns after the K-part exchange
    EG = (E // W) // NG  # router experts a group
    TPE = THREADS // EG  # threads an expert
    KPT = XQ // TPE  # router K a thread
    assert (NT_A, OWN, EG, KPT) == (1, 2, 1, 2), (NT_A, OWN, EG, KPT)
    # stage F: post_expert_proj rows RS_POST a rank, 16-row tiles
    RS_POST = HID // W
    NTL = RS_POST // 16
    TPC = 1  # tiles a CTA
    KSP = BLOCKS // NTL  # K split of a tile
    assert KSP == 4, KSP
    KR = DM // KSP  # K a CTA reads
    CPW = KR // KCH // WAVES  # chunks a wave a tile
    MROW = KR + XPAD
    tag_par = lambda tag: tag & 1

    @fx.struct
    class RouteLds:
        rid: fx.Array[fx.Int32, S * TOPK, 16]
        rw: fx.Array[fx.Float32, S * TOPK, 16]
        uexp: fx.Array[fx.Int32, EL, 16]
        kof: fx.Array[fx.Int32, EL * S, 16]
        nu: fx.Array[fx.Int32, 4, 16]
        sel: fx.Array[fx.Int32, max(S * TOPK * 3, EL), 16]
        plist: fx.Array[fx.Int32, P, 16]
        pidx: fx.Array[fx.Int32, P, 16]
        np_: fx.Array[fx.Int32, 4, 16]
        rstd: fx.Array[fx.Float32, P, 16]
        uslot: fx.Array[fx.Int32, EL, 16]
        unused: fx.Array[fx.Int32, 3 * E, 16]  # keeps the tuned LDS offsets
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
        red: fx.Array[fx.Float32, 2 * (1 if GUSPLIT else 2) * WAVES * 64 * 4, 16]

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
    assert CA >= 1 and CO >= 1 and NQT * 32 * 16 >= QKVG_ROWS
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
    if key.attn:  # the fused attention fragment (attn_frag.py)
        from . import attn_frag

        assert key.attn in ("swa", "full"), key
        FA = attn_frag.make(S // 8, key.attn == "swa")
        FaLds = FA.Lds
        # the fragment's source hash: the disk cache does not follow FA.run
        _KEYSTR = _KEYSTR + FA.key

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
    class Reserved:  # 128 B at the LDS base, unused: keeps the tuned LDS offsets
        pad: fx.Array[fx.Int64, TL_POINTS, 16]

    UNR = S == 8  # gate_up / down task loops as a ring (unroll 3, fixed slots)
    name = f"muse_k3_s{S}_r{int(key.next_rope)}" + (
        f"_fa{key.attn}" if key.attn else ""
    )
    CPRX = BLOCKS // W  # CTA blocks a rank owns (owner of block b = b // CPRX)

    # ------------------------------------------------------------ stage A (sharded)
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
        wr = ld_vec(w_r, (EG * rb + ei) * HID + k0 + kr, 2)  # KPT = 2 router weights
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
        wf = [wr[0].bitcast(fx.Float32), wr[1].bitcast(fx.Float32)]
        for t in range_constexpr(S):
            pr = fx.Float32(0.0)
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
            eg = rank * (E // W) + EG * rb + e2
            idx = ((par * KQ + kq) * S + t) * E + eg
            for pb in pl:
                mb_put_sys(tag, pb + fx.Int64(play["logit"][0]), idx, v)
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
            col = (rank * RS_PRE) + rb * RT + kq * OWN + 2 * rp
            hw = bf16_pair(tot[0], tot[1])
            idx = (par * S * DM + tt * DM + col) // 2
            for pb in pl:
                bo.buffer_store(
                    hw, rsrc(pb + fx.Int64(play["hraw"][0])), idx, cache_modifier=CM_SYS
                )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            fidx = (par * W + rank) * BLOCKS + bid
            for pb in pl:
                mb_put_sys(tag, pb + fx.Int64(play["hf"][0]), fidx, fx.Int32(1))
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
        NSRC = W
        nf = NSRC * BLOCKS
        specs, oks = [], []
        for i in range_constexpr((nf + THREADS - 1) // THREADS):
            f = fx.min(tid + THREADS * i, nf - 1)
            src = f // BLOCKS
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

    @traced
    def stage_route(scratch, bias_v, lgs_r, tag, health, rank, tid, lane, wave, rt):
        """Every token's routing (wave w: tokens w, w + 8, ...), then the local union."""
        if const_expr(S <= WAVES):
            if wave < S:
                route_token([wave], bias_v, [lgs_r[0]], tid, lane, rt)
        else:
            # S a multiple of 8: wave w routes tokens w, w + 8, .. interleaved
            route_token(
                [wave + WAVES * rr for rr in range(RROUNDS)],
                bias_v,
                lgs_r,
                tid,
                lane,
                rt,
            )
        gpu.barrier()
        # the union U of local experts (sorted), kof[u][t]; picks list
        e0 = rank * EL
        # parallel: picks flag their experts, a scan numbers U, picks scatter kof
        for ii in range_constexpr((EL * S + THREADS - 1) // THREADS):
            ix = tid + THREADS * ii
            if ix < EL * S:
                fx.ptr_store(fx.Int32(-1), rt["kof"] + ix)
        if tid < EL:
            fx.ptr_store(fx.Int32(0), rt["sel"] + tid)
        gpu.barrier()
        for ii in range_constexpr((P + THREADS - 1) // THREADS):
            pq = tid + THREADS * ii
            if pq < P:
                eid = slot_of(fx.ptr_load(rt["rid"] + pq), e0)
                if eid >= 0:
                    fx.ptr_store(fx.Int32(1), rt["sel"] + eid)
        gpu.barrier()
        if wave == 0:
            ucarry = fx.Int32(0)
            for hh in range_constexpr((EL + 63) // 64):
                el_ = lane + 64 * hh
                f = fx.ptr_load(rt["sel"] + fx.min(el_, EL - 1))
                f = (el_ < EL).select(f, fx.Int32(0))
                incl = wave_incl_scan(f, lane) + ucarry
                slot = incl - f
                if f != 0:
                    fx.ptr_store(el_, rt["uexp"] + slot)
                if el_ < EL:
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
                eid = slot_of(fx.ptr_load(rt["rid"] + pp), e0)
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
        for ii in range_constexpr((P + THREADS - 1) // THREADS):
            pq = tid + THREADS * ii
            if pq < P:
                eid = slot_of(fx.ptr_load(rt["rid"] + pq), e0)
                if eid >= 0:
                    slot = fx.ptr_load(rt["uslot"] + eid)
                    fx.ptr_store(
                        fx.Int32(pq % TOPK), rt["kof"] + (slot * S + pq // TOPK)
                    )
        gpu.barrier()

    def slot_of(e, e0):
        """Local slot of global expert e on this rank (-1: not computed here)."""
        el_ = e - e0
        return ((el_ >= 0) & (el_ < EL)).select(el_, fx.Int32(-1))

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
        ops = []
        if const_expr(GUSPLIT):
            # waves 0-3 the gate rows, 4-7 the up rows; wave w the chunks 4 (w % 4) + i
            rgs = [wave // 4]
            cs = [4 * (wave % 4) + i for i in range(4)]
        else:
            rgs = list(range(2))
            cs = [2 * wave + i for i in range(2)]
        for rg in rgs:
            rbase = e * (2 * INTER) + rg * INTER + j * 16
            for c in cs:
                ops += mx_chunk(
                    ok, w13, w13s, rbase, DM, c, lane, g, r, W13_BYTES, W13S_BYTES
                )
        return ops

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
        if const_expr(GUSPLIT):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(4):
                acc = fp4_mma(
                    ops, 3 * i, hl, HROW, lane, acc, 4 * (wave % 4) + i, task & 1, S
                )
            par = (it & 1) * (WAVES * 64 * 4)
            fx.ptr_store(acc, red + (par + (wave * 64 + lane) * 4))
        else:
            acc_g = fx.Vector.filled(4, 0.0, fx.Float32)
            acc_u = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(2):
                acc_g = fp4_mma(
                    ops, 3 * i, hl, HROW, lane, acc_g, 2 * wave + i, task & 1, S
                )
                acc_u = fp4_mma(
                    ops, 6 + 3 * i, hl, HROW, lane, acc_u, 2 * wave + i, task & 1, S
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
                if const_expr(GUSPLIT):
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

    @traced
    def dn_build_picks(rt, u0, u1, tid):
        """upk[h][n]: the picks (t*8 + k, token order) of expert slot u0 + h; upn[h]."""
        if tid < 2:
            u = fx.min(u0 + tid, EL - 1)
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
        fit = (n0 + n1) <= S
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
        if (u1 != u0) & ((n0 + n1) > S):
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
        fit = (n0 + n1) <= S
        rowb = (h == 0).select(fx.Int32(0), fit.select(n0, fx.Int32(0)))
        nv = (h == 0).select(n0, n1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(2):
            acc = fp4_mma(
                ops, 3 * i, al, AROW, lane, acc, 2 * wave + i, task & 1, S, rowb
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
        # the CTA-aggregated partials of the pick's expert: CTAs cf .. cl
        nu_ = uniform(fx.ptr_load(rt["nu"] + 0))
        el_ = slot_of(fx.ptr_load(rt["rid"] + pick), rt["rank"] * EL)
        u = fx.ptr_load(rt["uslot"] + fx.min(fx.max(el_, 0), EL - 1))
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
        (local) and slot [parity][rank] of the block owner's MOEP."""
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
                    isl = slot_of(eid, rank * EL) >= 0
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
            pd = sel_peer(pl, bid // CPRX)
            mb_put_sys(
                tag,
                pd + fx.Int64(play["moep"][0]),
                ((par * W + rank) * S + tt) * (DM // 2) + bid * 8 + rp,
                w,
            )

    @traced
    def ar_reduce(scratch, own_peer, tag, health, tid, bid, pl=None, rank=None):
        """The W ranks' slots of this CTA's columns, summed in rank order (fp32) ->
        bf16 m (local, plain) -> drain -> this CTA's MF flag."""
        par = tag_par(tag)
        # the block owner: W slots in rank order -> every rank
        if (tid < 8 * S) & ((bid // CPRX) == rank):
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
            bo.buffer_store(
                bf16_pair(lo, hi),
                rsrc(scratch + fx.Int64(lay["m"][0])),
                (tt * DM + bid * 16 + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["mf"][0]), bid, fx.Int32(1))

    # ------------------------------------------------------------ stage F: post_expert_proj
    def pp_tile(bid, i):
        return bid // KSP

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
        # LDS addresses here, holding them across the whole kernel (spills)
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
            tile = bid // KSP
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
                col = (rank * RS_POST) + tile * 16 + 2 * rp
                idx = (par * S + tt) * (HID // 2) + col // 2
                for pb in pl:
                    mb_put_sys(tag, pb + fx.Int64(play["y"][0]), idx, bf16_pair(y0, y1))

    # ------------------------------------------------------------ stage G: post-FFN norm, residual, next norm
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
        # stage AB: the partials come from the own peer buffer (the block owners)
        prb = rbase if pbase is None else pbase
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
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
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
                ops.append(
                    bf_tile_load_mma(w_o, bid * 32 + 16 * i, OK_K, c * KCH, lane)
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
        """Y = bf16(o rms(o) sigmoid(g)) per 64-wide head (the gated o-norm: fp32,
        one rounding) into LDS for every CTA, then o_proj rows 32b.. (this rank's K slice),
        the bf16 partial pushed into slot [par][rank] of the block owner's OP."""
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
            else:  # the fused attention: the merged rows as tagged pairs
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
                    aop = o_ops[i * CO + ci][qq]
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
            pd = sel_peer(pl, bid // CPRX)
            mb_put_sys(
                tag,
                pd + fx.Int64(play["op"][0]),
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
        # the block owner: W partials in rank order -> bf16 -> every rank
        if live & ((bid // CPRX) == rank):
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
                            [p1.bitcast(fx.Int32), tag, p2.bitcast(fx.Int32), tag],
                            fx.Int32,
                        ),
                        rsrc(base),
                        i0 * 2,
                        cache_modifier=CM_SYS,
                    )
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [p3.bitcast(fx.Int32), tag, p4.bitcast(fx.Int32), tag],
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
        a0 = al[0].bitcast(fx.Float32) * st[0].bitcast(fx.Float32)
        a1 = al[1].bitcast(fx.Float32) * st[1].bitcast(fx.Float32)
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
        r2 = fx.ptr_load(rsl + (S + tt))
        if live:
            bo.buffer_store(
                bf16_pair((s0 * r2) * bf_lo(wf), (s1 * r2) * bf_hi(wf)),
                rsrc(scratch + fx.Int64(lay["xin"][0])),
                (tt * HID + c) // 2,
                cache_modifier=CM_DEV,
            )
        rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(tag, scratch + fx.Int64(lay["xf"][0]), bid, fx.Int32(1))

    # ------------------------------------------------------------ stage QKV: next layer's qkv+gate
    # Head-aligned tile map: group gq < NH = HQ + 2 HKV holds the 4 tiles of head gq (q heads, then k,
    # then v) + gate tile NQKVT + gq; groups NH .. hold the other gate tiles (5 a group, the tail padded).
    # After the K-split partial sums, CTA k8 of the group owns the tokens t = k8 (mod 8) for all its rows,
    # so it holds whole heads and finishes them (qk-norm, RoPE, q / KV stores). The qkv rows also go to
    # the scratch region. The x eighth is the MFMA B operand straight from registers; the weights load in
    # the MFMA A-operand layout (bf_tile_load_mma).
    NH = HQ + 2 * HKV
    NQKVT = QKV_ROWS // 16
    assert NQT == 5 and NH <= 32 and S % 8 == 0, (NQT, NH, S)

    def q8_tile(gq, n):
        """Tile index (traced) of chunk-tile n of group gq (the head-aligned map)."""
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
            row16 = fx.min(tl, NQTILES - 1) * 16
            out.append(
                bf_tile_load_mma(w_qkvg, row16, HID, k8 * XQ8 + wave * KCH, lane)
            )
        return out

    @traced
    def vllm_finish(w2, hh, tt, p, q_out, k_cache, v_cache, slots, cos_sin, pos):
        """The owner's head finish: head hh (uniform), token tt (uniform in the wave), lane p = pair
        (2p, 2p + 1) of the head; w2 = the bf16 pair of the qkv GEMV.
        q / k as vLLM's fused_muse_spark_qknorm_rope_kv_insert: weightless qk-norm, then GPT-J RoPE in
        fp32 with the fp32 cos / sin, one RNE bf16 rounding (bf16_pair); q without qk_scale. V passes
        through (the bf16 GEMV). K / V pairs go as one bf16 dword into the LBHNC cache, element
        ((slot >> 4) * bstride + (kv * 16 + (slot & 15)) * 128 + 64 isv + 2p), bstride = v_cache (the
        block stride in elements, a value, not a pointer); slot < 0 (padded rows) skipped. The block
        base is 64-bit and wave-uniform (one buffer resource a wave)."""
        x0 = bf_lo(w2)
        x1 = bf_hi(w2)
        ss = butterfly(x0 * x0 + x1 * x1, (1, 2, 4, 8, 16))
        rs = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(PRE_EPS))
        isv = hh >= HQ + HKV
        isq = hh < HQ
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
        o0 = isv.select(x0, n0)
        o1 = isv.select(x1, n1)
        row0 = (hh * HD) // 2 + p
        if isq:
            bo.buffer_store(bf16_pair(o0, o1), rsrc(q_out), (tt * QROWS) // 2 + row0)
        else:
            kv = hh - HQ - isv.select(fx.Int32(HKV), fx.Int32(0))
            # int64 slot_mapping: low word (-1 = padding)
            slot = uniform(ld_i32(slots, tt * 2))
            if slot >= 0:
                eoff = fx.Int64(slot >> 4) * v_cache + fx.Int64(
                    (kv * 16 + (slot & 15)) * 128
                    + isv.select(fx.Int32(HD), fx.Int32(0))
                )
                bo.buffer_store(
                    bf16_pair(o0, o1), rsrc(k_cache + eoff * fx.Int64(2)), p
                )

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
        rocdl.sched_barrier(0)  # keep the x loads issued here, before the MFMA loop
        for n in range_constexpr(NQT):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for qq in range_constexpr(4):
                b = xb[qq].bitcast(fx.BFloat16)
                aop = q_ops[n][qq]
                acc = mfma_bf16(aop.bitcast(fx.BFloat16), b, acc)
            fx.ptr_store(acc, red + ((wave * NQT + n) * 64 + lane) * 4)
        gpu.barrier()
        SO = S // 8  # tokens an owner (t = k8 + 8 j)
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
        gpu.barrier()
        # owner wave j: token tt = k8 + 8 j; lane p < 40 = (tile n = p / 8, row pair p % 8); a head's 32 pairs
        # are lanes 0 .. 31 (pair p of the head = lane p)
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
                            (((gq * 8 + src) * NQT + n) * 16 + 2 * rp + h2) * S + tt,
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
                st_if(row < QKVG_ROWS, w2, g_out, (tt * QROWS + row - QKV_ROWS) // 2)
            if head & (p < 32):
                vllm_finish(w2, gq, tt, p, q_out, k_cache, v_cache, slots, cos_sin, pos)

    # ------------------------------------------------------------ kernel
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
    ):
        lgs_r = route_poll(own_peer, tag, health, lane, wave)
        stage_route(scratch, bias_v, lgs_r, tag, health, rank, tid, lane, wave, rt)
        nu = uniform(fx.ptr_load(rt["nu"] + 0))
        t0 = bid * nu
        t1 = t0 + nu
        last = t1 - 1
        q = [gu_loads(w13, w13s, rt, t0, lane, wave, t0 < t1)]
        hv, wv = h_wait_issue(own_peer, w_pren, tag, health, tid, rank)
        q = q + [
            gu_loads(w13, w13s, rt, t0 + d, lane, wave, (t0 + d) < t1)
            for d in range(1, PF_GU)
        ]
        return q, hv, wv, t0, t1, last

    # The kernel and its launcher. tl (a timeline buffer) and w13r .. eptab (expert replicas) are
    # part of the launch signature the decoder passes; this build does not read them.

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
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        alloc = fx.SharedAllocator()
        alloc.allocate(Reserved)
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
        }
        tok = rl.tok.ptr
        pl = [peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7]
        own_peer = sel_peer(pl, rank)
        ep = uniform(ld_i32(epoch, 0, CM_DEV))
        tag = (ep & 0x1FFFFFF) * 64 + layer + 1
        ao = None
        if const_expr(FA is not None):
            # attn_o is the fragment's descriptor; stage O polls the merged rows (tagged pairs)
            attn_o, ao_t = FA.run(
                attn_o, tag, health, tid, bid, lane, wave, sl.fa_.peek()
            )
            ao = (ao_t, health)
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
            ao,
        )
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
            pl,
            rank,
        )
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
        )
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
        )
        load_h_norm(hv, wv, tid, lane, wave, sc_.h.ptr, tok)
        gu_ld = lambda t: gu_loads(w13, w13s, rt, t, lane, wave, t < t1)
        gu_c = lambda ops, t, i: gu_compute(
            ops, t, i, scratch, tag, rt, tid, lane, wave, sc_.h.ptr, sc_.red.ptr
        )
        task_ring(gu_ld, gu_c, q, t0, fx.Int32(0), t1, UNR)
        dn_ld = lambda t: dn_loads(w2, w2s, w_post, rt["uexp"], t, lane, wave, t < t1)
        dq = [dn_loads(w2, w2s, w_post, rt["uexp"], t0, lane, wave, t0 < t1)]
        gpu.barrier()
        u0 = t0 >> 8
        u1 = last >> 8
        dn_build_picks(rt, u0, u1, tid)
        load_ac(scratch, tag, health, rt, False, sd.a.ptr, tid)
        dq = dq + [dn_ld(t0 + d) for d in range(1, PF_DN)]
        if tid < 2 * S:
            fx.ptr_store(fx.Float32(0.0), rt["ssacc"] + tid)
        gpu.barrier()
        tsplit = fx.min(t1, (u0 + 1) << 8)
        dn_c = lambda ops, t, i: dn_compute_c(
            ops, t, i, u0, scratch, tag, rt, tid, lane, wave, sd.a.ptr, sd.red.ptr
        )
        dq, task, it = task_ring(dn_ld, dn_c, dq, t0, fx.Int32(0), tsplit, UNR)
        dn_switch(scratch, tag, health, rt, tsplit, u0, u1, sd.a.ptr, tid)
        task_ring(dn_ld, dn_c, dq, task, it, t1, UNR)
        gpu.barrier()
        if tid < 2 * S:
            h2 = tid // S
            tt = tid % S
            u = u0 + h2
            k = fx.ptr_load(rt["kof"] + (fx.min(u, EL - 1) * S + tt))
            if ((h2 == 0) | (u1 != u0)) & (k >= 0):
                cf = (u * 256) // (t1 - t0)
                mb_put(
                    tag,
                    scratch + fx.Int64(lay["ss2"][0]),
                    (tt * TOPK + k) * 256 + (bid - cf),
                    fx.ptr_load(rt["ssacc"] + tid),
                )
        # post_expert_proj weights in flight before the all-reduce wait (pp_early: before
        # the final stage's polls too)
        ppo = pp_loads(w_pp, bid, lane, wave)
        gpu.barrier()
        stage_final(scratch, part_out, tag, health, rt, tid, bid, lane, se.z.ptr)
        final_push(scratch, part_out, pl, tag, rt, tid, bid, lane, se.z.ptr, rank)
        ar_reduce(scratch, own_peer, tag, health, tid, bid, pl, rank)
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
        qo = q8_loads(w_qkvg, bid, lane, wave, 0, QPF)
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
        )
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


def build_vllm(S=8, W=8, next_rope=True, attn=""):
    """The K3 build vLLM launches for S rows (8 / 16) of a TP = EP = 8 group; next_rope: the next
    layer has RoPE; attn = "swa" / "full": this layer's verify attention fused at the head."""
    assert W == 8 and S in (8, 16), (S, W)
    return Build(S=S, next_rope=bool(next_rope), attn=attn)
