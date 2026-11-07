# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gated residual (``GatedResidual`` of the AMD Qwen4Exp model) as grid
phases over every row of the step, with the stock path's bf16 roundings.

The multi-stream state ``[rows, HC * HIDDEN]`` (bf16) is materialized at every
kernel boundary: the block that consumes a pending combine writes it, with the
reciprocal RMS of each (row, stream), ``rrms`` (``combine``); a kernel whose
input state came from vLLM computes ``rrms`` first (``stats``). A mix is then

- ``mix_down``: ``[down | inject] = xn @ W_di.T`` (K = HC * HIDDEN, split K),
  ``xn = bf16(x rrms (1 + w))`` built as the activations are staged;
- ``mix_reduce``: the splits summed in order, rounded to bf16 (the GEMM's
  output); ``down = bf16(silu(down / HC))``, the injection scale ``2
  sigmoid(inject / HC)`` for the block's combine;
- ``mix_up``: ``gate = bf16(down @ W_up.T)`` and the gated mean ``x =
  bf16(sum_s sigmoid(gate_s) xn_s / HC)``: a unit is 16 columns of every
  stream, wave ``s`` stream ``s``, the streams summed in order through LDS.

Up to ``tiled.RESIDENT_MAX`` rows the three are one phase (``mix_resident``):
the down GEMM's K-wave units hand their sums to mix_up through tagged pairs.

``combine`` takes the block's output ``y`` (the TP ranks' partials, summed in
rank order through the exchange) into the state: ``bf16(x + y inj_s)``.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    bf16_round,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    pack_bf16,
    pack_bf16x8,
    rsqrt,
    sigmoid,
    silu,
    st,
    traced,
    unpack_bf16x8,
    wave_sum,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BLOCKS,
    DOWN_ROWS,
    HC,
    HC_HIDDEN,
    HIDDEN,
    LOWRANK,
    MAX_TOKENS,
    THREADS,
    WAVES,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import tagged, tiled

DOWN_SPLITS = 40  # K slices of the down GEMM: 6 row groups x 40 = 240 units
DOWN_KW_SPLITS = 5  # resident widths: 21 K-wave row tiles x 5 = 105 units
DOWN_KW_COLD = 10  # and when none of their weights were issued ahead: 210
UP_COLS = 16  # columns of every stream a mix_up unit covers
MIXED = LOWRANK + HC  # the down GEMM's rows that are used
UP_CHUNKS = -(-LOWRANK // tiled.CHUNK)  # the up GEMM's K chunks
UP_UNITS = HIDDEN // UP_COLS
assert UP_UNITS < BLOCKS  # ``inject`` runs on a CTA without a mix_up unit


def down_plan(rows: int = MAX_TOKENS, cold: bool = False):
    """The down GEMM's split: at the resident widths five K slices of each
    16-row tile, the four waves sharing a slice (``mix_resident``), or ten
    when the weights load in the phase (``cold``: twice the CTAs share them);
    else 40 slices of each 64-row group, summed by ``mix_reduce``."""
    if tiled.resident(rows):
        splits = DOWN_KW_COLD if cold else DOWN_KW_SPLITS
        return tiled.SplitK(DOWN_ROWS, HC_HIDDEN, splits, kwaves=True)
    return tiled.SplitK(DOWN_ROWS, HC_HIDDEN, DOWN_SPLITS)


def up_lds_words(rows: int) -> int:
    """f32 words of ``mix_up``'s LDS: a wave's gated stream, every token."""
    return WAVES * rows * UP_COLS


def i32(v):
    return fx.Int32(v)


@traced
def stats(g, state, rrms, rows: int, eps: float):
    """``rrms[row, s]`` of a state that came from outside the kernels: a wave
    a (row, stream)."""
    for unit in range(g.bid * WAVES + g.wave, rows * HC, BLOCKS * WAVES):
        base = unit * HIDDEN  # (row, stream) is the state's unit-th HIDDEN run
        sq = fx.Float32(0.0)
        for i in range_constexpr(HIDDEN // 512):
            v = unpack_bf16x8(
                ld_i32x4(state, (base + (i * 64 + g.lane) * 8) // 2, CM_DEV)
            )
            for e in range_constexpr(8):
                sq = sq + v[e] * v[e]
        tot = wave_sum(sq)
        if g.lane == 0:
            st(rrms, unit, rsqrt(tot / fx.Float32(HIDDEN) + fx.Float32(eps)), CM_DEV)


class NormedState(tiled.Source):
    """The normalized state ``xn = bf16(x rrms (1 + w))`` as GEMM activations:
    the loads of x, w and rrms are issued a K step ahead, the arithmetic after.
    The affine is Gemma's ``y + y w`` (the stock kernel's FMA form)."""

    def __init__(self, state, rrms, norm_w):
        self.state, self.rrms, self.norm_w = (
            fx.Int64(state),
            fx.Int64(rrms),
            fx.Int64(norm_w),
        )

    def issue(self, token, k):
        x = ld_i32x4(self.state, (token * i32(HC_HIDDEN) + k) // i32(2), CM_DEV)
        w = ld_i32x4(self.norm_w, k // i32(2))
        r = ld(self.rrms, token * i32(HC) + k // i32(HIDDEN), fx.Float32, cm=CM_DEV)
        return [x, w, r]

    def convert(self, raw):
        x, w, r = unpack_bf16x8(raw[0]), unpack_bf16x8(raw[1]), raw[2]
        out = []
        for e in range(8):
            y = x[e] * r
            out.append(y + y * w[e])
        return pack_bf16x8(out)


def down_weights(w_di):
    return tiled.Bf16Weights(w_di, DOWN_ROWS, HC_HIDDEN)


def mix_down(g, gemm, state, rrms, norm_w, w_di, part, pending=None):
    """``[down | inject]`` partials of every split to ``part``; ``pending``:
    the CTA's ``tiled.unit_preload`` of ``down_weights`` (past
    ``tiled.RESIDENT_MAX``)."""
    plan = down_plan(gemm.batch)
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        NormedState(state, rrms, norm_w),
        down_weights(w_di),
        plan,
        tiled.partial_epilogue(gemm, plan, part),
        pending=pending,
    )


# --------------------------------------------------------------------------
# the resident widths' mix: one phase, tagged hand-offs
# --------------------------------------------------------------------------
#
# At the resident widths the down GEMM runs as K-wave units (``down_plan``)
# and mix_up after it, on CTAs from ``_mix_ctas``; between them no grid
# barrier. Every hand-off is a tagged pair (``tagged``, the tag the mix's): a
# unit of split s > 0 publishes its partial sums, the unit of split 0 of the
# same row tile (the tile's owner) polls them, adds them in split order and
# publishes the tile's low-rank rows as (bf16 pair, tag) pairs (the injection
# scales, read after the next grid barrier, as plain words), and the mix_up
# units poll those as their activations.


def mix_words(rows: int) -> tuple[int, int]:
    """i32 words of the resident mix's tagged partials and tagged low-rank
    rows (0, 0 at the other widths)."""
    if not tiled.resident(rows):
        return 0, 0
    return 2 * down_plan(rows, cold=True).partial_words(rows), rows * LOWRANK


def _mix_ctas(rows: int):
    """The first CTA of the resident mix's down units and of its mix_up units.
    CTAs ``0 .. rows`` take none: they finish a block's combine (a wave a
    (row, stream)), the last to reach the barrier before a mix, where a CTA's
    issued weights would hold it."""
    return rows, BLOCKS - UP_UNITS


def mix_issue(g, w_di, w_up, rows: int):
    """The resident mix's first weights, in flight: the CTA's down unit's (its
    wave's K share) and its mix_up unit's (a CTA without one loads another's
    and drops them)."""
    d0, _ = _mix_ctas(rows)
    plan = down_plan(rows)
    down = tiled.Bf16Weights(w_di, DOWN_ROWS, HC_HIDDEN)
    d = fx.min(fx.max(g.bid - d0, i32(0)), i32(plan.units - 1))
    return tiled.issue_unit(d, g.tid, down, plan), _up_issue(g, w_up, rows)


def _up_issue(g, w_up, rows: int):
    up = tiled.Bf16Weights(w_up, HC_HIDDEN, LOWRANK)
    u = fx.max(g.bid - _mix_ctas(rows)[1], i32(0))
    group = g.lane // i32(tiled.TILE)
    return tiled.issue_chunks(up, [_up_row(g, u)], i32(0), UP_CHUNKS, group)


@traced
def mix_resident(
    g,
    gemm,
    up_gemm,
    red,
    state,
    rrms,
    norm_w,
    w_di,
    w_up,
    x_out,
    inj,
    bufs,
    tag,
    rows: int,
    pending,
):
    """The mix of a resident width as one phase: the down units, then the
    mix_up units (on CTAs ``_mix_ctas`` on).
    ``bufs``: the tagged partials and low-rank rows (``mix_words``);
    ``pending``: ``mix_issue``'s loads, or None (``down_plan``'s ``cold``)."""
    plan = down_plan(rows, cold=pending is None)
    d0, u0 = _mix_ctas(rows)
    part, low = bufs
    down_w, up_w = (None, None) if pending is None else pending
    if (g.bid >= d0) & (g.bid < d0 + plan.units):
        _down_unit(
            g,
            gemm,
            NormedState(state, rrms, norm_w),
            tiled.Bf16Weights(w_di, DOWN_ROWS, HC_HIDDEN),
            plan,
            part,
            low,
            inj,
            tag,
            rows,
            g.bid - d0,
            down_w,
        )
    if const_expr(pending is None):
        up_w = _up_issue(g, w_up, rows)
    if g.bid >= u0:
        _up_unit(
            g,
            up_gemm,
            red,
            tagged.Pairs(g, low, LOWRANK, tag),
            tiled.Bf16Weights(w_up, HC_HIDDEN, LOWRANK),
            state,
            rrms,
            norm_w,
            x_out,
            rows,
            g.bid - u0,
            up_w,
        )


@traced
def _down_unit(
    g, gemm, source, weights, plan, part, low, inj, tag, rows, unit, pending
):
    """A K-wave down unit; then wave 0 publishes its sums (split > 0) or, as
    the tile's owner (split 0), adds the other splits' and publishes the low-
    rank rows (or the injection scales)."""
    _, split, firsts, rws = tiled.unit_rows(plan, unit, g.tid)
    acc = gemm.run_kwaves(source, weights, rws[0], split * plan.per, plan.per, pending)
    if g.wave == 0:
        row = firsts[0] + g.lane // 16 * 4
        stride = i32(rows * plan.padded)  # a split's partials
        for t in range_constexpr(gemm.tiles):
            mine = i32(t * 16) + g.lane % 16
            token = fx.min(mine, i32(rows - 1))
            first = token * i32(plan.padded) + row  # split 0's pair index
            own = [acc[t][e] for e in range(4)]
            if split > 0:
                if mine < rows:
                    tagged.put(part, first + split * stride, own, tag)
            else:
                specs = [
                    (part, first + i32(s) * stride, 4) for s in range(1, plan.splits)
                ]
                got = tagged.poll(g, specs, tag)
                v = own
                for s in range_constexpr(plan.splits - 1):
                    v = [v[e] + got[s][e].bitcast(fx.Float32) for e in range(4)]
                v = [bf16_round(x) / fx.Float32(HC) for x in v]
                if mine < rows:
                    if row < LOWRANK:
                        words = [pack_bf16(silu(v[0]), silu(v[1]))]
                        words += [pack_bf16(silu(v[2]), silu(v[3]))]
                        tagged.put(low, (token * LOWRANK + row) // 2, words, tag)
                    if row == LOWRANK:
                        scale = [2.0 * sigmoid(x) for x in v]
                        st(
                            inj,
                            token * HC,
                            fx.Vector.from_elements(scale, fx.Float32),
                            CM_DEV,
                        )


@traced
def mix_reduce(g, part, down, inj, rows: int, inject: bool):
    """Sum the splits in order (bf16, the GEMM's output), then ``down =
    bf16(silu(v / HC))`` and, with ``inject``, ``inj = 2 sigmoid(v / HC)``."""
    plan = down_plan(rows)
    used = MIXED if inject else LOWRANK
    pairs = used // 2
    items = rows * pairs
    # runs of 64 items a wave, mirrored (``Grid.mirror``): off the CTAs whose
    # mix_up loads are in flight (``UpLoads``)
    for run in range(g.mirror, -(-items // 64), BLOCKS * WAVES):
        idx = run * 64 + g.lane
        if idx < items:
            b, r = idx // pairs, idx % pairs * 2
            loads = [
                ld(
                    part,
                    (i32(s * rows) + b) * i32(plan.padded) + r,
                    fx.Float32,
                    2,
                    cm=CM_DEV,
                )
                for s in range_constexpr(plan.splits)
            ]
            lo, hi = loads[0][0], loads[0][1]
            for s in range_constexpr(1, plan.splits):
                lo, hi = lo + loads[s][0], hi + loads[s][1]
            lo = bf16_round(lo) / fx.Float32(HC)
            hi = bf16_round(hi) / fx.Float32(HC)
            if r < LOWRANK:
                st(
                    down,
                    b * (LOWRANK // 2) + r // 2,
                    pack_bf16(silu(lo), silu(hi)),
                    CM_DEV,
                )
            else:
                v = fx.Vector.from_elements(
                    [2.0 * sigmoid(lo), 2.0 * sigmoid(hi)], fx.Float32
                )
                st(inj, b * HC + r - LOWRANK, v, CM_DEV)


class UpLoads(TraceInvariant):
    """The loads of the CTA's mix_up unit (a CTA has at most one) that do not
    depend on ``down``: the first chunks of its W_up rows (``Gemm.preload``)
    and its epilogue's norm weight, state and rrms (final before mix_down,
    which reads them too). ``mix_up`` waits only for ``down``, so they are
    issued early: up to ``tiled.PRELOAD_ROWS`` before mix_down, landing during
    its GEMM, and above that as mix_reduce starts, landing while the grid waits
    for the reduce. A CTA with no unit issues no request. Not at the resident
    widths, whose mix is one phase (``mix_resident``)."""

    def __init__(self, g, gemm, w_up, state, rrms, norm_w, rows: int):
        live = g.bid < i32(UP_UNITS)
        col = g.bid * UP_COLS + g.lane // 16 * 4
        self.weights = gemm.preload(
            tiled.Bf16Weights(w_up, HC_HIDDEN, LOWRANK),
            [_up_row(g, g.bid)],
            i32(0),
            UP_CHUNKS,
            live,
        )
        self.tokens = [
            fx.min(i32(t * 16) + g.lane % 16, i32(rows - 1)) for t in range(gemm.tiles)
        ]
        at = g.wave * HIDDEN + col
        self.norm = ld(norm_w, at // 2, fx.Int32, 2, nbytes=HC_HIDDEN * 2, mask=live)
        self.state = [
            ld(
                state,
                (tk * HC_HIDDEN + at) // 2,
                fx.Int32,
                2,
                cm=CM_DEV,
                nbytes=rows * HC_HIDDEN * 2,
                mask=live,
            )
            for tk in self.tokens
        ]
        self.rrms = [
            ld(
                rrms,
                tk * HC + g.wave,
                fx.Float32,
                cm=CM_DEV,
                nbytes=rows * HC * 4,
                mask=live,
            )
            for tk in self.tokens
        ]


@traced
def mix_up(g, gemm, red, down, w_up, x_out, rows: int, pre):
    """``x = bf16(sum_s sigmoid(bf16(down @ W_up[s].T)) xn_s / HC)``: unit
    ``u`` is columns ``16 u ..`` of every stream, wave ``s`` stream ``s``'s
    rows of W_up; ``red`` (LDS, ``up_lds_words`` f32) adds the streams in
    order. ``pre``: the CTA's ``UpLoads``."""
    weights = tiled.Bf16Weights(w_up, HC_HIDDEN, LOWRANK)
    src = tiled.Source(down, LOWRANK)
    for unit in range(g.bid, UP_UNITS, BLOCKS):
        _up_unit(
            g, gemm, red, src, weights, None, None, None, x_out, rows, unit, None, pre
        )


def _up_row(g, unit):
    """The lane's W_up row of ``unit``: stream ``wave``'s column ``16 unit +
    lane % 16``."""
    return g.wave * HIDDEN + unit * UP_COLS + g.lane % 16


@traced
def _up_unit(
    g,
    gemm,
    red,
    source,
    weights,
    state,
    rrms,
    norm_w,
    x_out,
    rows,
    unit,
    pending,
    pre=None,
):
    """One mix_up unit. ``pending``: its W_up chunks in flight (resident
    widths); ``pre`` (``mix_up``): the CTA's ``UpLoads``, its epilogue's loads
    and first W_up chunks already issued (``state``, ``rrms`` and ``norm_w``
    then unused)."""
    lane = g.lane
    c0 = unit * UP_COLS
    rows_w = [_up_row(g, unit)]
    if const_expr(pre is None):
        col = c0 + lane // 16 * 4
        # the epilogue's state, norm weight and rrms loads go out before the
        # GEMM, so their latency hides behind it
        tks = [
            fx.min(i32(t * 16) + lane % 16, i32(rows - 1)) for t in range(gemm.tiles)
        ]
        ww = ld(norm_w, (g.wave * HIDDEN + col) // 2, fx.Int32, 2)
        xws = [
            ld(
                state,
                (tk * HC_HIDDEN + g.wave * HIDDEN + col) // 2,
                fx.Int32,
                2,
                cm=CM_DEV,
            )
            for tk in tks
        ]
        rs = [ld(rrms, tk * HC + g.wave, fx.Float32, cm=CM_DEV) for tk in tks]
    else:
        tks, ww, xws, rs = pre.tokens, pre.norm, pre.state, pre.rrms
        pending = pre.weights
    acc = gemm.run(source, weights, rows_w, i32(0), UP_CHUNKS, pending)[0]
    ws = [
        fx.Int32(ww[0]) << 16,
        fx.Int32(ww[0]) & i32(-65536),
        fx.Int32(ww[1]) << 16,
        fx.Int32(ww[1]) & i32(-65536),
    ]
    for t in range_constexpr(gemm.tiles):
        tk, xw, r = tks[t], xws[t], rs[t]
        xs = [
            fx.Int32(xw[0]) << 16,
            fx.Int32(xw[0]) & i32(-65536),
            fx.Int32(xw[1]) << 16,
            fx.Int32(xw[1]) & i32(-65536),
        ]
        vals = []
        for e in range_constexpr(4):
            y = xs[e].bitcast(fx.Float32) * r
            xn = bf16_round(y + y * ws[e].bitcast(fx.Float32))
            vals.append(sigmoid(bf16_round(acc[t][e])) * xn)
        lds_st(
            red + (g.wave * rows + tk) * UP_COLS + lane // 16 * 4,
            fx.Vector.from_elements(vals, fx.Float32),
        )
    fx.barrier()
    for idx in range(g.tid, rows * (UP_COLS // 2), THREADS):
        token, c = idx // (UP_COLS // 2), idx % (UP_COLS // 2) * 2
        lo = fx.Float32(0.0)
        hi = fx.Float32(0.0)
        for s in range_constexpr(HC):
            v = lds_ld(red + (s * rows + token) * UP_COLS + c, 2, fx.Float32)
            lo, hi = lo + v[0], hi + v[1]
        st(
            x_out,
            (token * HIDDEN + c0 + c) // 2,
            pack_bf16(lo / fx.Float32(HC), hi / fx.Float32(HC)),
            CM_DEV,
        )
    fx.barrier()


@traced
def combine(g, xch, state, inj, rrms, rows: int, eps: float):
    """``state[b, s] = bf16(state + y[b] inj[b, s])`` and its ``rrms``, a wave a
    (row, stream); ``y`` is the ranks' partials from the exchange ``xch``,
    added in rank order and rounded to bf16. A lane's pieces are disjoint, so
    every piece's loads (the ranks' lines, then the state) are issued before
    the first piece's store, rather than a round trip a piece."""
    lpg = ex_.lines_per_group(rows)
    reps = HIDDEN // 512  # a lane's 8-column pieces
    for unit in range(g.bid * WAVES + g.wave, rows * HC, BLOCKS * WAVES):
        b = unit // HC
        groups = [i32(i * 8) + g.lane // 8 for i in range(reps)]
        lines = [grp * lpg + b // ex_.TOKENS for grp in groups]
        gates = ex_.wait_lines(g, xch, lines)
        scale = ld(inj, unit, fx.Float32, cm=CM_DEV)
        seg = g.lane % 8
        ats = [unit * HIDDEN + (i * 64 + g.lane) * 8 for i in range(reps)]
        loaded = [
            (
                [
                    xch.get(src, groups[i], b, seg, gates[i * xch.tp + src])
                    for src in range(xch.tp)
                ],
                ld_i32x4(state, ats[i] // 2, CM_DEV),
            )
            for i in range(reps)
        ]
        sq = fx.Float32(0.0)
        for i in range_constexpr(reps):
            ys, xw = loaded[i]
            y = ys[0]
            for src in range_constexpr(1, xch.tp):
                y = [y[e] + ys[src][e] for e in range(8)]
            x = unpack_bf16x8(xw)
            out = [bf16_round(x[e] + bf16_round(y[e]) * scale) for e in range(8)]
            for e in range_constexpr(8):
                sq = sq + out[e] * out[e]
            st(state, ats[i] // 2, pack_bf16x8(out), CM_DEV)
        tot = wave_sum(sq)
        if g.lane == 0:
            st(rrms, unit, rsqrt(tot / fx.Float32(HIDDEN) + fx.Float32(eps)), CM_DEV)
