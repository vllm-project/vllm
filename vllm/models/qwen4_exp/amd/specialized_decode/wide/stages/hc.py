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

``combine`` takes the block's output ``y`` (the TP ranks' partials, summed in
rank order through the exchange) into the state: ``bf16(x + y inj_s)``.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr

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
    THREADS,
    WAVES,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import tiled

DOWN_SPLITS = 40  # K slices of the down GEMM: 6 row groups x 40 = 240 units
UP_COLS = 16  # columns of every stream a mix_up unit covers
MIXED = LOWRANK + HC  # the down GEMM's rows that are used


def down_plan():
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


def mix_down(g, gemm, state, rrms, norm_w, w_di, part):
    """``[down | inject]`` partials of every split to ``part``."""
    plan = down_plan()
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        NormedState(state, rrms, norm_w),
        tiled.Bf16Weights(w_di, DOWN_ROWS, HC_HIDDEN),
        plan,
        tiled.partial_epilogue(gemm, plan, part),
    )


@traced
def mix_reduce(g, part, down, inj, rows: int, inject: bool):
    """Sum the splits in order (bf16, the GEMM's output), then ``down =
    bf16(silu(v / HC))`` and, with ``inject``, ``inj = 2 sigmoid(v / HC)``."""
    plan = down_plan()
    used = MIXED if inject else LOWRANK
    pairs = used // 2
    for idx in range(g.bid * THREADS + g.tid, rows * pairs, BLOCKS * THREADS):
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
            st(down, b * (LOWRANK // 2) + r // 2, pack_bf16(silu(lo), silu(hi)), CM_DEV)
        else:
            v = fx.Vector.from_elements(
                [2.0 * sigmoid(lo), 2.0 * sigmoid(hi)], fx.Float32
            )
            st(inj, b * HC + r - LOWRANK, v, CM_DEV)


@traced
def mix_up(g, gemm, red, down, w_up, state, rrms, norm_w, x_out, rows: int):
    """``x = bf16(sum_s sigmoid(bf16(down @ W_up[s].T)) xn_s / HC)``: unit
    ``u`` is columns ``16 u ..`` of every stream, wave ``s`` stream ``s``'s
    rows of W_up; ``red`` (LDS, ``up_lds_words`` f32) adds the streams in
    order."""
    weights = tiled.Bf16Weights(w_up, HC_HIDDEN, LOWRANK)
    source = tiled.Source(down, LOWRANK)
    chunks = -(-LOWRANK // tiled.CHUNK)
    lane = g.lane
    for unit in range(g.bid, HIDDEN // UP_COLS, BLOCKS):
        c0 = unit * UP_COLS
        rows_w = [g.wave * HIDDEN + c0 + lane % 16]
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
        acc = gemm.run(source, weights, rows_w, i32(0), chunks)[0]
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
    added in rank order and rounded to bf16."""
    lpg = ex_.lines_per_group(rows)
    reps = HIDDEN // 512  # a lane's 8-column pieces
    for unit in range(g.bid * WAVES + g.wave, rows * HC, BLOCKS * WAVES):
        b = unit // HC
        groups = [i32(i * 8) + g.lane // 8 for i in range(reps)]
        lines = [grp * lpg + b // ex_.TOKENS for grp in groups]
        gates = ex_.wait_lines(g, xch, lines)
        scale = ld(inj, unit, fx.Float32, cm=CM_DEV)
        sq = fx.Float32(0.0)
        for i in range_constexpr(reps):
            dword = ex_.line_dword(rows, groups[i], b, g.lane % 8)
            y = unpack_bf16x8(xch.load4(0, dword, gates[i * xch.tp]))
            for src in range_constexpr(1, xch.tp):
                more = unpack_bf16x8(xch.load4(src, dword, gates[i * xch.tp + src]))
                y = [y[e] + more[e] for e in range(8)]
            at = unit * HIDDEN + (i * 64 + g.lane) * 8
            x = unpack_bf16x8(ld_i32x4(state, at // 2, CM_DEV))
            out = [bf16_round(x[e] + bf16_round(y[e]) * scale) for e in range(8)]
            for e in range_constexpr(8):
                sq = sq + out[e] * out[e]
            st(state, at // 2, pack_bf16x8(out), CM_DEV)
        tot = wave_sum(sq)
        if g.lane == 0:
            st(rrms, unit, rsqrt(tot / fx.Float32(HIDDEN) + fx.Float32(eps)), CM_DEV)
