# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gated delta net of a GDN layer (vLLM's ``QwenGatedDeltaNetAttention``
decode on one rank) as grid phases over every row of the step.

- ``proj``: ``[q | k | v | z] = x @ W_qkvz.T`` (MXFP4) in two K slices a
  64-row group, folded by the group's last arriving unit, whose epilogue runs
  the causal conv on ``q | k | v`` with its state update (after FlyTape's
  ``gdn_proj``): ``silu(w0 s0 + w1 s1 + w2 s2 + w3 x)``, new state ``[s1, s2,
  x]``, ``x`` the projection rounded to bf16 (the stock GEMM's output); ``z``
  is stored as is. The ``[b | a]`` rows (bf16 ``W_ba``) are one-chunk units
  whose partials the recurrence sums.
- ``recur``: a unit is (row, key head), a CTA: the key head's three value
  heads in turn, each the gated delta rule on its 128 x 128 fp32 state in
  place, then the gated RMSNorm of the output (``RMSNormGated`` with the
  model's sigmoid output gate: ``norm(o) w sigmoid(z)``). A row whose state slot is
  the null block (<= 0, CUDA graph padding) skips its state and outputs zeros,
  as stock does.
- ``out``: ``core @ W_out.T`` (MXFP4; bf16 for a QSA layer's ``o_proj``, the
  output gate applied as its activations are staged) in six K slices a
  64-column group, folded by the group's last arriver, which pushes the group
  to the TP exchange.
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import TraceInvariant
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_NT,
    _permlane_swap,
    _xshfl,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    exp,
    gld,
    gst,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    pack_bf16x8,
    rsqrt,
    sigmoid,
    silu,
    softplus,
    st,
    traced,
    uniform,
    unpack_bf16x8,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BA,
    BLOCKS,
    CONV_DIM,
    GDN_DK,
    GDN_DV,
    HIDDEN,
    KEY_DIM,
    NK,
    NV,
    QKVZ,
    VALUE_DIM,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import tiled

PROJ_SPLITS = 2  # 128 row groups x 2 = 256 units
BA_SPLITS = HIDDEN // tiled.CHUNK  # one chunk a unit, summed by the recurrence
OUT_SPLITS = 6  # 40 column groups x 6 = 240 units
HEADS_PER_K = NV // NK  # value heads a key head serves
HALF = GDN_DK // 2  # key elements a thread holds
L2_EPS = 1e-6  # the in-kernel q / k L2 norm's


def proj_plan():
    return tiled.SplitK(QKVZ, HIDDEN, PROJ_SPLITS)


def ba_plan():
    return tiled.SplitK(BA, HIDDEN, BA_SPLITS)


def out_plan(k: int = VALUE_DIM):
    return tiled.SplitK(HIDDEN, k, OUT_SPLITS)


def i32(v):
    return fx.Int32(v)


def _halves(word):
    return bf_lo(word), bf_hi(word)


class Bufs(TraceInvariant):
    """Named addresses (Int64) and trace-time constants of a launch."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(
                self, k, v if isinstance(v, (bool, int, float, str)) else fx.Int64(v)
            )


# --------------------------------------------------------------------------
# input projection + conv
# --------------------------------------------------------------------------


def proj(g, gemm, w, b, rows: int, fold0: int):
    """The input projection and its conv. ``w``: ``qkvz`` / ``qkvz_s`` (MXFP4,
    ``shuffled`` its layout), ``ba`` (bf16), ``conv`` ([CONV_DIM, 4] bf16);
    ``b``: ``x``, ``qkv``, ``z``, ``ba`` (partials), ``part``, ``st_idx``,
    ``conv_state`` (a slot [3, CONV_DIM] bf16, slots ``conv_stride`` elements
    apart). Fold counters ``fold0 ..``."""
    plan = proj_plan()
    weights = tiled.Fp4Weights(w.qkvz, w.qkvz_s, QKVZ, HIDDEN, w.shuffled)
    source = tiled.Source(b.x, HIDDEN)

    def finish(group, first, lane, acc):
        conv_rows(w, b, rows, first + lane // 16 * 4, lane, acc, gemm.tiles)

    epilogue, after = tiled.fold(g, gemm, plan, b.part, fold0, finish)
    tiled.gemm_units(g.bid, BLOCKS, gemm, source, weights, plan, epilogue, after)
    ba_units(g, gemm, w, b, plan.units % BLOCKS)


@traced
def ba_units(g, gemm, w, b, first_unit: int):
    """The [b | a] rows: one chunk a unit, placed after the projection's
    units (on CTAs ``first_unit ..`` of the grid)."""
    bplan = ba_plan()
    bw = tiled.Bf16Weights(w.ba, BA, HIDDEN)
    source = tiled.Source(b.x, HIDDEN)
    epi = tiled.partial_epilogue(gemm, bplan, b.ba)
    for unit in range(g.bid - first_unit, bplan.units, BLOCKS):
        if unit >= 0:
            _, split, firsts, rws = tiled.unit_rows(bplan, unit, gemm.tid)
            acc = gemm.run(source, bw, rws, split * i32(bplan.per), bplan.per)
            epi(unit, split, firsts[0], gemm.lane, acc[0])


@traced
def conv_rows(w, b, rows: int, row, lane, acc, tiles: int):
    """Projection rows ``row .. row + 3`` of every token (a fold's summed
    accumulators): conv + silu with the state update for q | k | v, the plain
    store for z. A group of four never straddles the two. Every token tile's
    slot and state loads are issued before the first use (a padding token's
    lane reads the last token's and stores nothing)."""
    tokens = [fx.min(i32(t * 16) + lane % 16, i32(rows - 1)) for t in range(tiles)]
    valid = [i32(t * 16) + lane % 16 < rows for t in range(tiles)]
    if row < CONV_DIM:
        words = [
            ld_i32x4(w.conv, row * 2),
            ld_i32x4(w.conv, row * 2 + 4),
        ]  # the rows' 16 taps
        slots = [ld(b.st_idx, tk, fx.Int32, cm=CM_DEV) for tk in tokens]
        live = [sl > 0 for sl in slots]
        # a slot's [3, CONV_DIM] state, slots ``conv_stride`` elements apart
        sbs = [
            b.conv_state
            + (
                fx.Int64(live[t].select(slots[t], i32(0))) * b.conv_stride
                + fx.Int64(row)
            )
            * fx.Int64(2)
            for t in range(tiles)
        ]
        states = [
            [gld(sbs[t] + fx.Int64(jj * CONV_DIM * 2), fx.Int32, 2) for jj in range(3)]
            for t in range(tiles)
        ]
        tap = []
        for e in range_constexpr(4):
            w01 = _halves(words[e // 2][2 * (e % 2)])
            w23 = _halves(words[e // 2][2 * (e % 2) + 1])
            tap.append([w01[0], w01[1], w23[0], w23[1]])
        for t in range_constexpr(tiles):
            s = states[t]
            xs = [bf16_round(acc[t][e]) for e in range(4)]
            out = []
            for e in range_constexpr(4):
                sv = [
                    live[t].select(_halves(s[jj][e // 2])[e % 2], fx.Float32(0.0))
                    for jj in range(3)
                ]
                out.append(
                    silu(
                        tap[e][0] * sv[0]
                        + tap[e][1] * sv[1]
                        + tap[e][2] * sv[2]
                        + tap[e][3] * xs[e]
                    )
                )
            if valid[t]:
                st(
                    b.qkv,
                    (tokens[t] * CONV_DIM + row) // 2,
                    fx.Vector.from_elements(out, fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32),
                    CM_DEV,
                )
                if live[t]:
                    gst(sbs[t], s[1])
                    gst(sbs[t] + fx.Int64(CONV_DIM * 2), s[2])
                    gst(
                        sbs[t] + fx.Int64(2 * CONV_DIM * 2),
                        fx.Vector.from_elements(xs, fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32),
                    )
    else:
        for t in range_constexpr(tiles):
            if valid[t]:
                v = fx.Vector.from_elements([acc[t][e] for e in range(4)], fx.Float32)
                st(
                    b.z,
                    (tokens[t] * VALUE_DIM + row - CONV_DIM) // 2,
                    v.to(fx.BFloat16).bitcast(fx.Int32),
                    CM_DEV,
                )


# --------------------------------------------------------------------------
# recurrence + gated RMSNorm
# --------------------------------------------------------------------------

# The state of a value head is [DV rows (value), DK cols (key)] fp32, key
# contiguous. Lane ``l`` of wave ``w`` holds key quad ``l % 32`` of rows ``8 i
# + 2 w + l // 32`` (i = 0..15): every access is 16 bytes a lane and a wave
# instruction covers two whole rows (1 KiB). Dots over the key dimension
# reduce over the 32 lanes of a row.
ROWS_PER_THREAD = GDN_DV // 8


def _row_sum(x):
    return butterfly(x, (16, 8, 4, 2, 1))


@traced
def recur(g, w, b, lds, rows: int, eps: float):
    """``w``: ``a_log`` (f32), ``dt_bias`` (bf16), ``norm`` ([GDN_DV] bf16);
    ``b``: ``qkv``, ``z``, ``ba`` (partials), ``st_idx``, ``ssm`` (a slot [NV,
    DV, DK] f32, slots ``ssm_stride`` apart), ``core`` [rows, VALUE_DIM] bf16; ``lds``:
    ``recur_lds_words()`` f32 words. A unit (row, key head) is a CTA; the
    state loads of the next head (the next unit's first, after the last) are
    in flight while a head computes."""
    kq = g.lane % 32
    k0 = kq * 4
    rp = g.lane // 32
    at = [(i32(8 * i) + g.wave * 2 + rp) * GDN_DK + k0 for i in range(ROWS_PER_THREAD)]
    units = rows * NK
    # the norm weight of the one row each lane stores (``head``)
    vmine = (kq // 2) * 8 + g.wave * 2 + rp
    nw = bf_lo(ld(w.norm, vmine // 2, fx.Int32) >> ((vmine % 2) * 16))
    h = load_heads(g, b, g.bid, at, 0)
    for unit in range(g.bid, units, BLOCKS):
        row, kh = unit // NK, unit % NK
        slot = uniform(ld(b.st_idx, row, fx.Int32, cm=CM_DEV))
        stage_unit(g, w, b, lds, rows, row, kh)
        q = lds_ld(lds + Q_AT + k0, 4, fx.Float32)
        k = lds_ld(lds + K_AT + k0, 4, fx.Float32)
        hb = state_base(b, slot, kh)
        for i in range_constexpr(HEADS_PER_K):
            if const_expr(i + 1 < HEADS_PER_K):
                nxt = load_heads(g, b, unit, at, i + 1)
            else:
                nxt = load_heads(
                    g,
                    b,
                    (i32(unit) + BLOCKS < units).select(unit + BLOCKS, unit),
                    at,
                    0,
                )
            if slot > 0:
                head(
                    g,
                    b,
                    nw,
                    h,
                    hb + fx.Int64(i * HEAD_BYTES),
                    at,
                    q,
                    k,
                    lds,
                    i,
                    row,
                    kh * HEADS_PER_K + i,
                    rp,
                    kq,
                    eps,
                )
            else:
                if g.tid < GDN_DV // 2:
                    st(
                        b.core,
                        (row * VALUE_DIM + (kh * HEADS_PER_K + i) * GDN_DV) // 2
                        + g.tid,
                        fx.Int32(0),
                        CM_DEV,
                    )
            h = nxt
        fx.barrier()


HEAD_BYTES = GDN_DV * GDN_DK * 4
V_AT, Z_AT = 0, 3 * GDN_DV  # LDS f32 offsets: v, z of the unit's 3 heads
Q_AT, K_AT = 6 * GDN_DV, 6 * GDN_DV + GDN_DK
G_AT = 6 * GDN_DV + 2 * GDN_DK  # [beta x 3, decay x 3]
N_AT = G_AT + 8  # the norm's wave partials
P_AT = N_AT + 8  # the gates' split partials [6][splits]
B_AT = P_AT + 6 * BA_SPLITS  # k . h of each wave's rows, [wave][half][16]


def recur_lds_words() -> int:
    return B_AT + 4 * 32


def state_base(b, slot, kh):
    """Byte address of value head ``3 kh``'s state in slot ``slot`` (clamped to
    the null block 0 for a padding row, whose state is never written); slots
    are ``ssm_stride`` f32 apart, a slot's heads dense."""
    s = (slot > 0).select(slot, i32(0))
    return (
        b.ssm
        + fx.Int64(s) * b.ssm_stride * fx.Int64(4)
        + fx.Int64(kh * HEADS_PER_K) * fx.Int64(HEAD_BYTES)
    )


def load_heads(g, b, unit, at, i: int):
    """This thread's 16 state quads of head ``i`` of ``unit``."""
    row, kh = unit // NK, unit % NK
    slot = uniform(ld(b.st_idx, row, fx.Int32, cm=CM_DEV))
    hb = state_base(b, slot, kh) + fx.Int64(i * HEAD_BYTES)
    return [ld(hb, a, fx.Float32, 4, cm=CM_NT) for a in at]


@traced
def stage_unit(g, w, b, lds, rows: int, row, kh):
    """q, k (L2-normalized, q scaled), v and z of the unit's 3 heads, and their
    gates (beta, decay: the [b | a] split partials summed in order, rounded to
    bf16) into LDS."""
    bplan = ba_plan()
    base = row * CONV_DIM
    scale = fx.Float32(1.0 / math.sqrt(GDN_DK))
    if g.tid < 64:  # q | k, four elements a thread
        src = (g.tid < 32).select(
            base + kh * GDN_DK + g.tid * 4,
            base + KEY_DIM + kh * GDN_DK + (g.tid - 32) * 4,
        )
        wv = ld(b.qkv, src // 2, fx.Int32, 2, cm=CM_DEV)
        x = [bf_lo(wv[0]), bf_hi(wv[0]), bf_lo(wv[1]), bf_hi(wv[1])]
        ss = _row_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        r = rsqrt(ss + fx.Float32(L2_EPS)) * (g.tid < 32).select(scale, fx.Float32(1.0))
        lds_st(
            lds + Q_AT + g.tid * 4,
            fx.Vector.from_elements([x[e] * r for e in range(4)], fx.Float32),
        )
    else:
        if g.tid < 64 + 96:  # v of 3 heads, four a thread
            t = g.tid - 64
            wv = ld(
                b.qkv,
                (base + 2 * KEY_DIM + kh * 3 * GDN_DV + t * 4) // 2,
                fx.Int32,
                2,
                cm=CM_DEV,
            )
            lds_st(
                lds + V_AT + t * 4,
                fx.Vector.from_elements(
                    [bf_lo(wv[0]), bf_hi(wv[0]), bf_lo(wv[1]), bf_hi(wv[1])], fx.Float32
                ),
            )
        else:  # z of 3 heads
            t = g.tid - 160
            wv = ld(
                b.z,
                (row * VALUE_DIM + kh * 3 * GDN_DV + t * 4) // 2,
                fx.Int32,
                2,
                cm=CM_DEV,
            )
            lds_st(
                lds + Z_AT + t * 4,
                fx.Vector.from_elements(
                    [bf_lo(wv[0]), bf_hi(wv[0]), bf_lo(wv[1]), bf_hi(wv[1])], fx.Float32
                ),
            )
    # the gates' partials, one a thread: gate j (beta 0..2, decay 3..5), split s
    if g.tid < 6 * bplan.splits:
        j, s = g.tid // bplan.splits, g.tid % bplan.splits
        r = (j < 3).select(kh * HEADS_PER_K + j, NV + kh * HEADS_PER_K + j - 3)
        lds_st(
            lds + P_AT + g.tid,
            ld(b.ba, (s * rows + row) * bplan.padded + r, fx.Float32, cm=CM_DEV),
        )
    fx.barrier()
    if g.tid < 6:
        tot = lds_ld(lds + P_AT + g.tid * bplan.splits)
        for s in range_constexpr(1, bplan.splits):
            tot = tot + lds_ld(lds + P_AT + g.tid * bplan.splits + s)
        tot = bf16_round(tot)
        vh = kh * HEADS_PER_K + g.tid % 3
        a_log = ld(w.a_log, vh, fx.Float32)
        dt = bf_lo(ld(w.dt_bias, vh // 2, fx.Int32) >> ((vh % 2) * 16))
        gate = (g.tid < 3).select(sigmoid(tot), exp(-exp(a_log) * softplus(tot + dt)))
        lds_st(lds + G_AT + g.tid, gate)
    fx.barrier()


def _halve(v, lane):
    """Sums over the 32 lanes of a row half (lane bits 0-4) of the 16 values a
    lane holds, one a state row, by halving: at each level a lane keeps half
    its values and adds its partner's copy of that half (FlyTape's
    ``wave_sums``), so 16 rows cost 16 exchanges, not 80. Returns the sum of
    row ``(lane % 32) // 2``, in both lanes of each pair."""
    # level 1 (lane ^ 16): v_permlane16_swap of (low half, high half) leaves
    # the row-0 lanes' low and the row-1 lanes' high partial pairs in place
    out = []
    for j in range(8):
        a, b = _permlane16(v[j], v[j + 8])
        out.append(a + b)
    v = out
    for level, n in ((8, 4), (4, 2), (2, 1)):
        hi = (lane & level) != 0
        out = []
        for j in range(n):
            keep = hi.select(v[j + n], v[j])
            send = hi.select(v[j], v[j + n])
            out.append(keep + _xshfl(send, level))
        v = out
    return v[0] + _xshfl(v[0], 1)


def _permlane16(x, y):
    a, b = _permlane_swap(16, x.bitcast(fx.Int32), y.bitcast(fx.Int32))
    return a.bitcast(fx.Float32), b.bitcast(fx.Float32)


@traced
def head(g, b, nw, h, hb, at, q, k, lds, i_head: int, row, vh, rp, kq, eps):
    """One value head of one row: the delta rule on the state (``h``, this
    thread's 16 rows of 4 keys, loaded), stored back in place; then the gated
    RMSNorm of the head's output into ``core``. ``k . h`` of each row is
    reduced by halving (``_halve``) and broadcast to the row's lanes through
    the wave's LDS words (``B_AT``); ``q . h`` stays in the lanes that hold it."""
    beta = lds_ld(lds + G_AT + i_head)
    decay = lds_ld(lds + G_AT + 3 + i_head)
    vl, zl, nl = lds + V_AT + i_head * GDN_DV, lds + Z_AT + i_head * GDN_DV, lds + N_AT
    bl = lds + B_AT + g.wave * 32 + rp * 16  # this half-wave's 16 rows
    x = [[h[i][e] * decay for e in range(4)] for i in range(ROWS_PER_THREAD)]
    kv = _halve(
        [
            x[i][0] * k[0] + x[i][1] * k[1] + x[i][2] * k[2] + x[i][3] * k[3]
            for i in range(ROWS_PER_THREAD)
        ],
        g.lane,
    )
    mine = kq // 2  # the row whose sum this lane holds
    if kq % 2 == 0:
        lds_st(bl + mine, kv)
    rows_kv = [lds_ld(bl + 4 * c, 4, fx.Float32) for c in range(4)]
    outs = []
    for i in range_constexpr(ROWS_PER_THREAD):
        v = 8 * i + g.wave * 2 + rp
        dv = (lds_ld(vl + v) - rows_kv[i // 4][i % 4]) * beta
        x[i] = [x[i][e] + dv * k[e] for e in range(4)]
        st(hb, at[i], fx.Vector.from_elements(x[i], fx.Float32), cm=CM_NT)
        outs.append(x[i][0] * q[0] + x[i][1] * q[1] + x[i][2] * q[2] + x[i][3] * q[3])
    o = bf16_round(
        _halve(outs, g.lane)
    )  # row ``mine``'s output, rounded as stock's core
    # sum of squares over the head's 128 outputs: one lane of each pair counts
    sq = (kq % 2 == 0).select(o * o, fx.Float32(0.0))
    sq = butterfly(sq, (32, 16, 8, 4, 2, 1))
    if g.lane == 0:
        lds_st(nl + g.wave, sq)
    fx.barrier()
    tot = lds_ld(nl) + lds_ld(nl + 1) + lds_ld(nl + 2) + lds_ld(nl + 3)
    fx.barrier()
    rr = rsqrt(tot / fx.Float32(GDN_DV) + fx.Float32(eps))
    v = 8 * mine + g.wave * 2 + rp
    if kq % 2 == 0:
        y = o * rr * nw * sigmoid(lds_ld(zl + v))
        st(
            b.core,
            row * VALUE_DIM + vh * GDN_DV + v,
            fx.Float32(y).to(fx.BFloat16),
            CM_DEV,
        )


# --------------------------------------------------------------------------
# output projection -> TP exchange
# --------------------------------------------------------------------------


class GatedCore(tiled.Source):
    """A QSA layer's o_proj input: ``bf16(attn * bf16(sigmoid(gate)))``, as the
    stock path's two bf16 tensors multiply."""

    def __init__(self, attn, gate, ld_elems: int):
        self.attn, self.gate, self.ld = fx.Int64(attn), fx.Int64(gate), ld_elems

    def issue(self, token, k):
        at = (token * i32(self.ld) + k) // i32(2)
        return [ld_i32x4(self.attn, at), ld_i32x4(self.gate, at)]

    def convert(self, raw):
        a, gt = unpack_bf16x8(raw[0]), unpack_bf16x8(raw[1])
        return pack_bf16x8([a[e] * bf16_round(sigmoid(gt[e])) for e in range(8)])


def out(g, gemm, weights, source, part, tile, xch, k: int, fold0: int):
    """``source @ weights.T`` (K = ``k``) to the exchange ``xch``: each
    64-column group's last arriving unit sums its six slices into the LDS
    ``tile`` and pushes the group's lines."""
    plan = out_plan(k)

    def finish(group, first, lane, acc):
        tile.put_acc(first - group * ex_.COLS, lane, acc, gemm.tiles)

    def done(group):
        fx.barrier()
        ex_.push_tile(g, xch, tile, group)
        fx.barrier()

    epilogue, after = tiled.fold(g, gemm, plan, part, fold0, finish, done)
    tiled.gemm_units(g.bid, BLOCKS, gemm, source, weights, plan, epilogue, after)
