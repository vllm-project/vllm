# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dense phases of a Muse layer on one TP rank: the o-path's gated norm
and quant, the residual boundaries, and the finishes of the split-K GEMMs
(o_proj, router, pooler, unpooler, QKVG).

`o_gate_quant` (a wave a token): `rms_norm_gated` of the rank's 16 heads
(a weightless RMSNorm a 64-wide head times `sigmoid(og)`, rounded to bf16),
then the per-token FP8 quant of o_proj's input (`DENSE_QUANT`).

`boundary_token` (a CTA a token): `fused_norm_residual_norm_fwd`: the
branch read from a gather's lines, its post-norm (weightless, or input-scaled
`x * (w + 1)` first), `new = alpha * stream + beta * y` in f32 (the
stream updated in place), the next pre-norm `new * rstd * (w + 1)` rounded
to bf16, stored as bf16 or quantized per token to FP8.

The GEMM finishes run in the row group's last arriving split-K unit
(`tiled.fold`) and either push the group's lines (`push_finish`: o_proj
scattered, pooler and unpooler gathered) or store rows (`store_finish`:
QKVG). The router's fp32 weights are three bf16 planes (`w = hi + mid +
lo`) so its GEMM runs on the bf16 MFMA with fp32 products: rows `0 ..` the
32 local experts' `hi`, `32 ..` their `mid`, `64 ..` their `lo`,
`96 .. 127` zero; the finishes push `hi + mid` and `lo`, the route
phase adds them.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    bf16_round,
    butterfly,
    fabs,
    fp8_quant8,
    ld,
    lds_ld,
    lds_st,
    pack_bf16x8,
    quant_factor,
    quant_scale,
    rsqrt,
    sigmoid,
    st,
    traced,
    unpack_bf16x8,
    wave_max,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
    BLOCKS,
    DENSE_QUANT,
    HEAD_DIM,
    Q_ROWS,
    RMS_EPS,
    THREADS,
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.stages.exchange import (
    COLS,
    GroupTile,
    push_tile,
    wait_lines,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.stages.tiled import TILE

LANES = 64


def i32(v) -> fx.Int32:
    return fx.Int32(v)


def _dense_codes(vals, amax):
    """`vals` (8 f32 a lane) as FP8 codes with the token's scale; a zero row
    gets zero codes and scale 0."""
    scale = quant_scale(amax, DENSE_QUANT)
    f = (scale > fx.Float32(0.0)).select(
        quant_factor(scale, DENSE_QUANT), fx.Float32(0.0)
    )
    return fp8_quant8(vals, f, DENSE_QUANT), scale


# --------------------------------------------------------------------------
# o-path: gated norm and quant
# --------------------------------------------------------------------------


@traced
def o_gate_token(g, attn, qkv, qkv_ld: int, og_at: int, xo, xs, t):
    """Token `t`: lane `l` holds 16 columns `16 l ..` of the rank's
    1024 (head `l // 4`). `attn`: bf16 `[rows, 1024]`; `og`: columns
    `og_at ..` of the bf16 QKVG output `qkv` (`qkv_ld` columns a row).
    Stores `xo` (FP8 `[rows, 1024]`) and `xs` (f32 `[rows]`)."""
    lane = g.lane
    col = lane * 16
    a = unpack_bf16x8(
        ld(attn, (t * Q_ROWS + col) // 2, fx.Int32, 4, cm=CM_DEV)
    ) + unpack_bf16x8(ld(attn, (t * Q_ROWS + col + 8) // 2, fx.Int32, 4, cm=CM_DEV))
    at = (t * qkv_ld + og_at + col) // 2
    og = unpack_bf16x8(ld(qkv, at, fx.Int32, 4, cm=CM_DEV)) + unpack_bf16x8(
        ld(qkv, at + 4, fx.Int32, 4, cm=CM_DEV)
    )
    ss = fx.Float32(0.0)
    for e in range_constexpr(16):
        ss = ss + a[e] * a[e]
    # a head is 4 lanes
    ss = butterfly(ss, (1, 2))
    rstd = rsqrt(ss / fx.Float32(HEAD_DIM) + fx.Float32(RMS_EPS))
    y = [bf16_round(a[e] * rstd * sigmoid(og[e])) for e in range(16)]
    m = fx.Float32(0.0)
    for e in range_constexpr(16):
        m = fx.max(m, fabs(y[e]))
    amax = wave_max(m)
    for h in range_constexpr(2):
        codes, scale = _dense_codes(y[8 * h : 8 * h + 8], amax)
        st(
            xo,
            (t * Q_ROWS + col + 8 * h) // 4,
            fx.Vector.from_elements(codes, fx.Int32),
            cm=CM_DEV,
        )
    if lane == 0:
        st(xs, t, scale, cm=CM_DEV)


@traced
def o_gate_quant(g, rows: int, attn, qkv, qkv_ld: int, og_at: int, xo, xs):
    waves = i32(BLOCKS * WAVES)
    for t in range(g.spread, i32(rows), waves):
        o_gate_token(g, attn, qkv, qkv_ld, og_at, xo, xs, t)


# --------------------------------------------------------------------------
# residual boundary
# --------------------------------------------------------------------------


class Boundary(TraceInvariant):
    """One boundary's vectors (f32 `[8192]` byte addresses): `post_w1`
    (`w + 1`; None: weightless), `alpha`, `beta`, `pre_w1`; its
    `post_eps`. `quant`: store the output as FP8 + scale, else bf16."""

    def __init__(self, post_w1, alpha, beta, pre_w1, post_eps: float, quant: bool):
        self.post_w1 = None if post_w1 is None else fx.Int64(post_w1)
        self.alpha, self.beta, self.pre_w1 = (
            fx.Int64(alpha),
            fx.Int64(beta),
            fx.Int64(pre_w1),
        )
        self.post_eps, self.quant = post_eps, quant


@traced
def _block_sum(g, red, v):
    """The CTA's sum of `v` (waves in order) on every thread; `red`: 4 LDS
    words, not rewritten before every thread is past the next barrier (the
    boundary alternates two sets)."""
    s = wave_sum(v)
    if g.lane == 0:
        lds_st(red + g.wave, s.bitcast(fx.Int32))
    fx.barrier()
    total = lds_ld(red, 4).bitcast(fx.Float32)
    return ((total[0] + total[1]) + total[2]) + total[3]


@traced
def boundary_token(g, bd, ex, hidden: int, stream, out, out_s, red, t):
    """Token `t` (the CTA): thread `v` holds items `v + 256 i` (8
    columns each) of the `hidden` columns; the branch from `ex` (a
    gather), the f32 `stream` row updated in place, the output to `out`
    (bf16 or FP8 `[rows, hidden]`) and `out_s` (f32 scales)."""
    items = hidden // 8 // THREADS
    groups = hidden // COLS
    for r in range_constexpr(-(-groups // THREADS)):
        grp = g.tid + i32(r * THREADS)
        if grp < groups:
            wait_lines(g, ex, [(grp, grp * i32(ex.lpg) + t // 8)])
    fx.barrier()
    xs, cols = [], []
    for i in range_constexpr(items):
        item = g.tid + i32(i * THREADS)
        grp = item // 8
        xs.append(ex.get(ex.owner(grp), grp, t, item % 8, i32(0)))
        cols.append(item * 8)

    def vec8(addr, i, cm=0):
        return list(ld(addr, i, fx.Float32, 4, cm=cm)) + list(
            ld(addr, i + 4, fx.Float32, 4, cm=cm)
        )

    if bd.post_w1 is not None:
        xs = [
            [x[e] * w[e] for e in range(8)]
            for x, w in zip(xs, [vec8(bd.post_w1, c) for c in cols])
        ]
    ss = fx.Float32(0.0)
    for x in xs:
        for e in range_constexpr(8):
            ss = ss + x[e] * x[e]
    rstd = rsqrt(_block_sum(g, red, ss) / fx.Float32(hidden) + fx.Float32(bd.post_eps))
    news = []
    for x, c in zip(xs, cols):
        base = t * hidden + c
        s = vec8(stream, base, CM_DEV)
        al, be = vec8(bd.alpha, c), vec8(bd.beta, c)
        new = [al[e] * s[e] + be[e] * (x[e] * rstd) for e in range(8)]
        st(stream, base, fx.Vector.from_elements(new[0:4], fx.Float32), cm=CM_DEV)
        st(stream, base + 4, fx.Vector.from_elements(new[4:8], fx.Float32), cm=CM_DEV)
        news.append(new)
    ss = fx.Float32(0.0)
    for new in news:
        for e in range_constexpr(8):
            ss = ss + new[e] * new[e]
    rstd = rsqrt(_block_sum(g, red + 4, ss) / fx.Float32(hidden) + fx.Float32(RMS_EPS))
    zs = []
    for new, c in zip(news, cols):
        w = vec8(bd.pre_w1, c)
        zs.append([bf16_round(new[e] * rstd * w[e]) for e in range(8)])
    if bd.quant:
        m = fx.Float32(0.0)
        for z in zs:
            for e in range_constexpr(8):
                m = fx.max(m, fabs(z[e]))
        amax = _block_max(g, red, m)
        for z, c in zip(zs, cols):
            codes, scale = _dense_codes(z, amax)
            st(
                out,
                (t * hidden + c) // 4,
                fx.Vector.from_elements(codes, fx.Int32),
                cm=CM_DEV,
            )
        if g.tid == 0:
            st(out_s, t, scale, cm=CM_DEV)
    else:
        for z, c in zip(zs, cols):
            st(out, (t * hidden + c) // 2, pack_bf16x8(z), cm=CM_DEV)
    fx.barrier()


@traced
def _block_max(g, red, v):
    s = wave_max(v)
    if g.lane == 0:
        lds_st(red + g.wave, s.bitcast(fx.Int32))
    fx.barrier()
    m = lds_ld(red, 4).bitcast(fx.Float32)
    return fx.max(fx.max(m[0], m[1]), fx.max(m[2], m[3]))


@traced
def boundary_tokens(g, rows: int, bd, ex, hidden: int, stream, out, out_s, red):
    """Every token's boundary, a CTA each over the grid."""
    for t in range(g.bid, i32(rows), i32(BLOCKS)):
        boundary_token(g, bd, ex, hidden, stream, out, out_s, red, t)


# --------------------------------------------------------------------------
# split-K finishes
# --------------------------------------------------------------------------


def dequant(xs, ws_addr, ws_at):
    """The FP8 GEMM epilogue's `acc * xs[token] * ws[row]`: `scale(token,
    col)` for `GroupTile.put_acc`, `ws_at(col)` the weight row."""

    def scale(token, col):
        x = ld(xs, token, fx.Float32, cm=CM_DEV)
        w = ld(ws_addr, ws_at(col), fx.Float32, 4)
        return [x * w[e] for e in range(4)]

    return scale


def push_finish(g, gemm, tile, xch, group_of, scale_of=None):
    """`fold`'s (finish, done) pair: the group's accumulators (times
    `scale_of(group)(token, col)` when given) rounded to bf16 into
    `tile`, then pushed as the lines of exchange group `group_of(group)`."""

    @traced
    def finish(group, first_row, lane, acc):
        col0 = first_row - group * i32(COLS)
        tile.put_acc(
            col0, lane, acc, gemm.tiles, None if scale_of is None else scale_of(group)
        )

    @traced
    def done(group):
        fx.barrier()
        push_tile(g, xch, tile, group_of(group))
        fx.barrier()

    return finish, done


def store_finish(gemm, out, out_ld: int, xs, ws):
    """`fold`'s finish for the QKVG: `bf16(acc * xs[token] * ws[row])` to
    the row-major bf16 `out` (`out_ld` columns a row)."""

    @traced
    def finish(group, first_row, lane, acc):
        row = first_row + lane // i32(TILE) * 4
        w = ld(ws, row, fx.Float32, 4)
        for t in range_constexpr(gemm.tiles):
            token = i32(t * TILE) + lane % i32(TILE)
            if token < i32(gemm.batch):
                x = ld(xs, token, fx.Float32, cm=CM_DEV)
                v = [acc[t][e] * x * w[e] for e in range(4)]
                st(
                    out,
                    (token * out_ld + row) // 2,
                    fx.Vector.from_elements(v, fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32),
                    cm=CM_DEV,
                )

    return finish


ROUTER_ROWS = 128  # hi, mid, lo planes of 32 experts, then zeros
ROUTER_GROUPS = 2  # fold groups: (hi, mid), (lo, zeros)


def router_finish(g, gemm, tile, mid_page, xch, rank):
    """`fold`'s (finish, done) for the router (two 64-row groups, a wave a
    tile: waves 0, 1 the first plane of 16 experts each, waves 2, 3 the
    second): waves 2, 3 leave theirs in `mid_page`, waves 0, 1 add it, so
    group 0 finishes `hi + mid` and group 1 `lo` (+ exact zeros); the f32
    words go out as group `2 rank + group` of the logits gather."""
    stride = 16  # words a token of a half

    @traced
    def finish(group, first_row, lane, acc):
        wave = g.wave
        if wave >= 2:
            for t in range_constexpr(gemm.tiles):
                token = i32(t * TILE) + lane % i32(TILE)
                at = ((wave - 2) * i32(gemm.tiles * TILE) + token) * stride
                lds_st(mid_page + at + lane // i32(TILE) * 4, acc[t].bitcast(fx.Int32))
        fx.barrier()
        if wave < 2:
            for t in range_constexpr(gemm.tiles):
                token = i32(t * TILE) + lane % i32(TILE)
                at = (wave * i32(gemm.tiles * TILE) + token) * stride
                mid = lds_ld(mid_page + at + lane // i32(TILE) * 4, 4).bitcast(
                    fx.Float32
                )
                v = [acc[t][e] + mid[e] for e in range(4)]
                tk = fx.min(token, i32(tile.rows - 1))
                col = wave * 16 + lane // i32(TILE) * 4
                lds_st(
                    tile.ptr + tk * GroupTile.STRIDE + col,
                    fx.Vector.from_elements(v, fx.Float32).bitcast(fx.Int32),
                )

    @traced
    def done(group):
        fx.barrier()
        push_tile(g, xch, tile, rank * ROUTER_GROUPS + group)
        fx.barrier()

    return finish, done
