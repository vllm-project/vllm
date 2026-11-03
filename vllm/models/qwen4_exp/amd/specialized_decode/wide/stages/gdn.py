# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gated delta net of a GDN layer (vLLM's ``QwenGatedDeltaNetAttention``
decode on one rank) as grid phases over every row of the step.

- ``proj``: ``[q | k | v | z] = x @ W_qkvz.T`` (MXFP4) in two K slices a
  64-row group, folded by the group's last arriving unit, whose epilogue runs
  the causal conv on ``q | k | v`` with its state update:
  ``silu(w0 s0 + w1 s1 + w2 s2 + w3 x)``, new state ``[s1, s2, x]``, ``x``
  the projection rounded to bf16 (the stock GEMM's output); ``z`` is stored
  as is. The ``[b | a]`` rows (bf16 ``W_ba``) are one-chunk units
  whose partials the recurrence sums, run by the first arrivers of the
  first groups, so no last arriver's conv waits behind one.
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

Up to ``tiled.RESIDENT_MAX`` rows the three are one phase: a recurrence unit
is (row, value head), and the projection's outputs and the recurrence's
reach their consumers as tagged pairs (``tagged``) instead of through grid
barriers; the ``[b | a]`` chunks ride on the projection units' stages.
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
    pack_bf16,
    pack_bf16x8,
    rsqrt,
    sigmoid,
    silu,
    softplus,
    st,
    traced,
    uniform,
    unpack_bf16x8,
    zeros,
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
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import tagged, tiled

PROJ_SPLITS = 2  # 128 row groups x 2 = 256 units
BA_SPLITS = HIDDEN // tiled.CHUNK  # one chunk a unit, summed by the recurrence
OUT_SPLITS = 6  # 40 column groups x 6 = 240 units
HEADS_PER_K = NV // NK  # value heads a key head serves
HALF = GDN_DK // 2  # key elements a thread holds
L2_EPS = 1e-6  # the in-kernel q / k L2 norm's
# widths past tiled.RESIDENT_MAX up to which a projection group's conv loads
# are issued by its last arriver alone (beat every unit issuing them at 32
# rows, not from 64)
CONV_LATE_ROWS = 48


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


def handoff_words(rows: int) -> tuple[int, int, int]:
    """i32 words of the ``qkv``, ``ba`` and ``core`` buffers: plain, or at the
    resident widths tagged pairs (``[rows, QKVZ / 2]`` bf16 pairs, the
    partials' f32 values, ``[rows, VALUE_DIM / 2]`` bf16 pairs)."""
    if tiled.resident(rows):
        return rows * QKVZ, 2 * ba_plan().partial_words(rows), rows * VALUE_DIM
    return rows * CONV_DIM // 2, ba_plan().partial_words(rows), rows * VALUE_DIM // 2


def proj_weights(w):
    return tiled.Fp4Weights(w.qkvz, w.qkvz_s, QKVZ, HIDDEN, w.shuffled)


def ba_weights(w):
    return tiled.Bf16Weights(w.ba, BA, HIDDEN)


def proj(g, gemm, w, b, rows: int, fold0: int, pending=None, tag=None):
    """The input projection and its conv. ``w``: ``qkvz`` / ``qkvz_s`` (MXFP4,
    ``shuffled`` its layout), ``ba`` (bf16), ``conv`` ([CONV_DIM, 4] bf16);
    ``b``: ``x``, ``qkv``, ``z``, ``ba`` (partials), ``part``, ``st_idx``,
    ``conv_state`` (a slot [3, CONV_DIM] bf16, slots ``conv_stride`` elements
    apart). Fold counters ``fold0 ..``. ``pending``: the first unit's weights,
    at the resident widths ``proj_issue``'s, else the CTA's
    ``tiled.unit_preload`` of ``proj_weights``.

    At the resident widths the [b | a] chunk ``c`` is multiplied by the unit
    of split ``c // per`` and group ``c % per``, whose stage holds it, before
    the unit's fold; wider steps run it as one-chunk units on the first
    arrivers of the projection's groups (``_proj_streamed``). With ``tag``
    (resident widths) the outputs are tagged pairs (``tagged``) the
    recurrence polls, with no grid barrier between: ``qkv`` holds ``[rows,
    QKVZ / 2]`` (bf16 pair, tag) pairs, z included, ``ba`` the partials'
    (value, tag) pairs."""
    plan = proj_plan()
    source = tiled.Source(b.x, HIDDEN)
    if not gemm.resident:
        _proj_streamed(g, gemm, w, b, rows, fold0, pending, plan, source)
        return

    def finish(group, first, lane, acc):
        conv_rows(w, b, rows, first + lane // 16 * 4, lane, acc, gemm.tiles, tag)

    epilogue, after = tiled.fold(g, gemm, plan, b.part, fold0, finish)
    if plan.units > BLOCKS or plan.per * plan.splits != BA_SPLITS:
        raise ValueError("the [b | a] chunks ride on one projection unit a CTA")
    loads, ba_loads = (None, None) if pending is None else pending
    bw = ba_weights(w)
    bplan = ba_plan()
    if tag is None:
        ba_epi = tiled.partial_epilogue(gemm, bplan, b.ba)
    else:
        ba_epi = _tagged_partials(gemm, bplan, b.ba, tag)

    def ba_after(group, split):
        if group < plan.per:
            c = split * plan.per + group
            got = ba_loads
            if const_expr(got is None):
                got = _ba_issue(g, bw, c)
            acc = gemm.mma_tiles(
                [[zeros(4) for _ in range(gemm.tiles)]],
                [bw.operands(got)],
                lambda t: gemm.operands_resident(group, plan.per, t, bw.koff),
            )
            ba_epi(c, c, g.wave * 16, gemm.lane, acc[0])
        after(group, split)

    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        source,
        proj_weights(w),
        plan,
        epilogue,
        traced(ba_after),
        pending=loads,
    )


def _proj_streamed(g, gemm, w, b, rows: int, fold0: int, pending, plan, source):
    """``proj`` past ``tiled.RESIDENT_MAX``. The conv's inputs do not depend on
    the projection, so they leave the group's fold chain: the slots load
    before the GEMM. Up to ``CONV_LATE_ROWS`` the last arriver issues the taps
    and states with its partial loads (``tiled.fold``'s ``begin``), so their
    round trips overlap; above, every unit issues them after its partials,
    before the fold arrival, which then drains them. The ``[b | a]`` units run
    on the first arrivers of groups ``0 ..``, which have nothing else to do in
    the phase."""
    tokens, _ = _conv_tokens(rows, gemm.lane, gemm.tiles)
    slots = [ld(b.st_idx, tk, fx.Int32, cm=CM_DEV) for tk in tokens]
    late = rows <= CONV_LATE_ROWS
    staged: list = []

    def stage(first, lane):
        # a z row reads the last conv rows' and uses nothing
        row = fx.min(first + lane // 16 * 4, i32(CONV_DIM - 4))
        staged[:] = [conv_loads(w, b, row, tokens, slots)]

    def begin(group, first, lane):
        stage(first, lane)

    def finish(group, first, lane, acc):
        row = first + lane // 16 * 4
        conv_rows(w, b, rows, row, lane, acc, gemm.tiles, loads=staged[0])

    bplan = ba_plan()
    bw = ba_weights(w)
    ba_epilogue = tiled.partial_epilogue(gemm, bplan, b.ba)

    @traced
    def ba_unit(group):
        """The [b | a] unit ``group`` (one chunk), on the group's first arriver."""
        if group < i32(bplan.units):
            _, split, firsts, rws = tiled.unit_rows(bplan, group, gemm.tid)
            acc = gemm.run(source, bw, rws, split * i32(bplan.per), bplan.per)
            ba_epilogue(group, split, firsts[0], gemm.lane, acc[0])

    epilogue, after = tiled.fold(
        g,
        gemm,
        plan,
        b.part,
        fold0,
        finish,
        other=ba_unit,
        begin=begin if late else None,
    )

    def staging_epilogue(unit, split, first_row, lane, acc):
        epilogue(unit, split, first_row, lane, acc)
        stage(first_row, lane)

    # the staged loads are the CTA's one unit's (its one weight tile)
    assert late or (plan.units == BLOCKS and plan.mtiles == 1), plan.units
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        source,
        proj_weights(w),
        plan,
        epilogue if late else staging_epilogue,
        after,
        pending,
    )


def _tagged_partials(gemm, plan, part, tag):
    """``tiled.partial_epilogue`` storing (value, tag) pairs."""

    @traced
    def epilogue(unit, split, first_row, lane, acc):
        for t in range_constexpr(gemm.tiles):
            token = i32(t * 16) + lane % 16
            if token < gemm.batch:
                at = (split * gemm.batch + token) * plan.padded + first_row
                at = at + lane // 16 * 4
                tagged.put(part, at, [acc[t][e] for e in range(4)], tag)

    return epilogue


def proj_issue(g, w):
    """The first unit's weights of ``proj`` at the resident widths, in flight:
    its input projection chunks and its [b | a] chunk (a unit without one
    loads the last chunk's and drops them)."""
    plan = proj_plan()
    unit = (g.bid < i32(plan.units)).select(g.bid, i32(plan.units - 1))
    c = unit // plan.groups * plan.per + unit % plan.groups
    c = (c < BA_SPLITS).select(c, i32(BA_SPLITS - 1))
    return (
        tiled.issue_unit(g.bid, g.tid, proj_weights(w), plan),
        _ba_issue(g, ba_weights(w), c),
    )


def _ba_issue(g, bw, c):
    return bw.issue(g.wave * 16 + g.lane % 16, c, g.lane // 16)


def _conv_tokens(rows: int, lane, tiles: int):
    """A lane's token of each tile (a padding lane takes the last token) and
    whether it is a real one."""
    tokens = [fx.min(i32(t * 16) + lane % 16, i32(rows - 1)) for t in range(tiles)]
    valid = [i32(t * 16) + lane % 16 < rows for t in range(tiles)]
    return tokens, valid


def conv_loads(w, b, row, tokens, slots=None):
    """The conv's loads for projection rows ``row .. row + 3`` (conv rows):
    their taps and each token's state (``slots``: the ``tokens``' state slots,
    loaded here if None)."""
    words = [
        ld_i32x4(w.conv, row * 2),
        ld_i32x4(w.conv, row * 2 + 4),
    ]  # the rows' 16 taps
    if slots is None:
        slots = [ld(b.st_idx, tk, fx.Int32, cm=CM_DEV) for tk in tokens]
    live = [sl > 0 for sl in slots]
    # a slot's [3, CONV_DIM] state, slots ``conv_stride`` elements apart
    sbs = [
        b.conv_state
        + (fx.Int64(live[t].select(slots[t], i32(0))) * b.conv_stride + fx.Int64(row))
        * fx.Int64(2)
        for t in range(len(tokens))
    ]
    states = [
        [gld(sbs[t] + fx.Int64(jj * CONV_DIM * 2), fx.Int32, 2) for jj in range(3)]
        for t in range(len(tokens))
    ]
    return words, live, sbs, states


@traced
def conv_rows(w, b, rows: int, row, lane, acc, tiles: int, tag=None, loads=None):
    """Projection rows ``row .. row + 3`` of every token (a fold's summed
    accumulators): conv + silu with the state update for q | k | v, the plain
    store for z. A group of four never straddles the two. ``loads``: the rows'
    ``conv_loads``, issued before the summed partials were loaded; without
    them every token tile's slot and state loads are issued here before the
    first use. A padding token's lane reads the last token's and stores
    nothing. With ``tag`` q | k | v and z are stored as tagged pairs
    (``proj``)."""
    tokens, valid = _conv_tokens(rows, lane, tiles)
    if row < CONV_DIM:
        if const_expr(loads is None):
            words, live, sbs, states = conv_loads(w, b, row, tokens)
        else:
            words, live, sbs, states = loads
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
                packed = (
                    fx.Vector.from_elements(out, fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)
                )
                if const_expr(tag is None):
                    st(b.qkv, (tokens[t] * CONV_DIM + row) // 2, packed, CM_DEV)
                else:
                    at = (tokens[t] * QKVZ + row) // 2
                    tagged.put(b.qkv, at, [packed[0], packed[1]], tag)
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
                packed = v.to(fx.BFloat16).bitcast(fx.Int32)
                if const_expr(tag is None):
                    at = (tokens[t] * VALUE_DIM + row - CONV_DIM) // 2
                    st(b.z, at, packed, CM_DEV)
                else:
                    at = (tokens[t] * QKVZ + row) // 2
                    tagged.put(b.qkv, at, [packed[0], packed[1]], tag)


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


def _state_offsets(g):
    """Element offsets of this thread's 16 state quads in a head."""
    kq, rp = g.lane % 32, g.lane // 32
    return [
        (i32(8 * i) + g.wave * 2 + rp) * GDN_DK + kq * 4 for i in range(ROWS_PER_THREAD)
    ]


def recur(g, w, b, lds, rows: int, eps: float, first=None, tag=None):
    """``w``: ``a_log`` (f32), ``dt_bias`` (bf16), ``norm`` ([GDN_DV] bf16);
    ``b``: ``qkv``, ``z``, ``ba`` (partials), ``st_idx``, ``ssm`` (a slot [NV,
    DV, DK] f32, slots ``ssm_stride`` apart), ``core`` [rows, VALUE_DIM] bf16; ``lds``:
    ``recur_lds_words()`` f32 words. A unit is a CTA: (row, key head), its
    three value heads in turn, or at the resident widths (row, value head),
    three times the units; the next head's state loads are in flight while a
    head computes. At the resident widths (``tag`` required) the projection's
    outputs are tagged pairs (``proj``), ``core`` is published as (bf16 pair,
    tag) pairs, and ``first`` may be the first unit's state loads
    (``recur_first``)."""
    if tiled.resident(rows):
        if tag is None:
            raise ValueError("the resident widths' recurrence takes tagged pairs")
        return _recur_heads(g, w, b, lds, rows, eps, first, tag)
    return _recur(g, w, b, lds, rows, eps)


def recur_first(g, b, rows: int):
    """The state loads of the CTA's first unit at the resident widths. No phase
    of the launch writes them before the recurrence, so they may be issued
    ahead (inside the barrier before it)."""
    units = rows * NV
    unit = (g.bid < i32(units)).select(g.bid, i32(units - 1))
    return load_head(b, unit, _state_offsets(g))


def load_head(b, unit, at):
    """This thread's 16 state quads of value head ``unit % NV`` of row ``unit //
    NV``."""
    row, vh = unit // NV, unit % NV
    slot = uniform(ld(b.st_idx, row, fx.Int32, cm=CM_DEV))
    hb = state_base(b, slot, vh // HEADS_PER_K) + fx.Int64(vh % HEADS_PER_K) * fx.Int64(
        HEAD_BYTES
    )
    return [ld(hb, a, fx.Float32, 4, cm=CM_NT) for a in at]


@traced
def _recur_heads(g, w, b, lds, rows: int, eps: float, h, tag):
    kq = g.lane % 32
    k0 = kq * 4
    rp = g.lane // 32
    at = _state_offsets(g)
    units = rows * NV
    vmine = (kq // 2) * 8 + g.wave * 2 + rp
    nw = bf_lo(ld(w.norm, vmine // 2, fx.Int32) >> ((vmine % 2) * 16))
    if const_expr(h is None):
        h = recur_first(g, b, rows)
    for unit in range(g.bid, units, BLOCKS):
        row, vh = unit // NV, unit % NV
        kh, i = vh // HEADS_PER_K, vh % HEADS_PER_K
        slot = uniform(ld(b.st_idx, row, fx.Int32, cm=CM_DEV))
        stage_tagged(g, w, b, lds, rows, row, kh, tag)
        q = lds_ld(lds + Q_AT + k0, 4, fx.Float32)
        k = lds_ld(lds + K_AT + k0, 4, fx.Float32)
        nxt = load_head(b, (i32(unit) + BLOCKS < units).select(unit + BLOCKS, unit), at)
        if slot > 0:
            hb = state_base(b, slot, kh) + fx.Int64(i) * fx.Int64(HEAD_BYTES)
            head(g, b, nw, h, hb, at, q, k, lds, i, row, vh, rp, kq, eps, tag)
        else:
            if g.tid < GDN_DV // 2:
                at0 = (row * VALUE_DIM + vh * GDN_DV) // 2 + g.tid
                tagged.put(b.core, at0, [i32(0)], tag)
        h = nxt
        fx.barrier()


@traced
def _recur(g, w, b, lds, rows: int, eps: float):
    kq = g.lane % 32
    k0 = kq * 4
    rp = g.lane // 32
    at = _state_offsets(g)
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


Y_AT = B_AT + 4 * 32  # a head's gated-norm outputs (``head`` with a tag)


def recur_lds_words() -> int:
    return Y_AT + GDN_DV


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
        lds_st(
            lds + P_AT + g.tid,
            ld(b.ba, _gate_partial(g, rows, row, kh), fx.Float32, cm=CM_DEV),
        )
    _gates(g, w, lds, kh)


def _gate_partial(g, rows: int, row, kh):
    """The [b | a] partial thread ``tid < 6 splits`` stages: gate ``j = tid /
    splits`` (beta 0..2, decay 3..5) of the key head's value heads, split
    ``tid % splits``."""
    bplan = ba_plan()
    j, s = g.tid // bplan.splits, g.tid % bplan.splits
    r = (j < 3).select(kh * HEADS_PER_K + j, NV + kh * HEADS_PER_K + j - 3)
    return (s * rows + row) * bplan.padded + r


@traced
def _gates(g, w, lds, kh):
    """The gates from their staged partials (summed in split order, rounded to
    bf16) into LDS, between barriers."""
    bplan = ba_plan()
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


@traced
def stage_tagged(g, w, b, lds, rows: int, row, kh, tag):
    """``stage_unit`` from the projection's tagged pairs (``proj``): every
    thread polls its four q | k, v or z values and its gate partial together."""
    bplan = ba_plan()
    t = g.tid
    scale = fx.Float32(1.0 / math.sqrt(GDN_DK))
    qrow = (t < 32).select(
        kh * GDN_DK + t * 4,
        (t < 64).select(
            KEY_DIM + kh * GDN_DK + (t - 32) * 4,
            (t < 160).select(
                2 * KEY_DIM + kh * 3 * GDN_DV + (t - 64) * 4,
                CONV_DIM + kh * 3 * GDN_DV + (t - 160) * 4,
            ),
        ),
    )
    live = t < 6 * bplan.splits
    gate_at = live.select(_gate_partial(g, rows, row, kh), i32(0))
    got = tagged.poll(
        g, [(b.qkv, (row * QKVZ + qrow) // 2, 2), (b.ba, gate_at, 1)], tag
    )
    wv = got[0]
    x = [bf_lo(wv[0]), bf_hi(wv[0]), bf_lo(wv[1]), bf_hi(wv[1])]
    if t < 64:  # q | k: L2-normalized, q scaled
        ss = _row_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        r = rsqrt(ss + fx.Float32(L2_EPS)) * (t < 32).select(scale, fx.Float32(1.0))
        lds_st(
            lds + Q_AT + t * 4,
            fx.Vector.from_elements([x[e] * r for e in range(4)], fx.Float32),
        )
    else:
        at = (t < 160).select(V_AT + (t - 64) * 4, Z_AT + (t - 160) * 4)
        lds_st(lds + at, fx.Vector.from_elements(x, fx.Float32))
    if live:
        lds_st(lds + P_AT + t, got[1][0].bitcast(fx.Float32))
    _gates(g, w, lds, kh)


def _halve(v, lane):
    """Sums over the 32 lanes of a row half (lane bits 0-4) of the 16 values a
    lane holds, one a state row, by halving: at each level a lane keeps half
    its values and adds its partner's copy of that half, so 16 rows cost
    16 exchanges, not 80. Returns the sum of
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
def head(g, b, nw, h, hb, at, q, k, lds, i_head, row, vh, rp, kq, eps, tag=None):
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
    if const_expr(tag is None):
        if kq % 2 == 0:
            y = o * rr * nw * sigmoid(lds_ld(zl + v))
            st(
                b.core,
                row * VALUE_DIM + vh * GDN_DV + v,
                fx.Float32(y).to(fx.BFloat16),
                CM_DEV,
            )
    else:
        # the head's outputs gathered, then published as tagged bf16 pairs
        if kq % 2 == 0:
            lds_st(lds + Y_AT + v, o * rr * nw * sigmoid(lds_ld(zl + v)))
        fx.barrier()
        if g.tid < GDN_DV // 4:
            ys = lds_ld(lds + Y_AT + g.tid * 4, 4, fx.Float32)
            tagged.put(
                b.core,
                (row * VALUE_DIM + vh * GDN_DV) // 2 + g.tid * 2,
                [pack_bf16(ys[0], ys[1]), pack_bf16(ys[2], ys[3])],
                tag,
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


def out(g, gemm, weights, source, part, tile, xch, k: int, fold0: int, pending=None):
    """``source @ weights.T`` (K = ``k``) to the exchange ``xch``: each
    64-column group's last arriving unit sums its six slices into the LDS
    ``tile`` and pushes the group's lines. ``pending``: the first unit's
    weights, at the resident widths ``tiled.issue_unit``'s of ``weights`` and
    ``out_plan(k)``, else the CTA's ``tiled.unit_preload`` of them."""
    plan = out_plan(k)

    def finish(group, first, lane, acc):
        tile.put_acc(first - group * ex_.COLS, lane, acc, gemm.tiles)

    def done(group):
        fx.barrier()
        ex_.push_tile(g, xch, tile, group)
        fx.barrier()

    epilogue, after = tiled.fold(g, gemm, plan, part, fold0, finish, done)
    tiled.gemm_units(
        g.bid, BLOCKS, gemm, source, weights, plan, epilogue, after, pending=pending
    )
