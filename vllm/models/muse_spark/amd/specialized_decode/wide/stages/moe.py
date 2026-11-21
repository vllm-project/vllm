# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed experts of a Muse layer on one EP rank (32 local experts), from
the gathered router logits and pooled activations to the scattered lines of
the expert sum.

`route_tokens` (a wave a token): the token's 256 logits and its pooled row
from the exchange lines; sigmoid scores, the top 8 of score + bias (ties to
the lowest id), the route weights `score * (route_scale / (sum + 1e-15))`;
the pre-expert norm of the pooled row, rounded to bf16 and quantized per
token to FP8 (`MOE_IN_QUANT`). It zeroes the routes' activation maxima.

`schedule` (every CTA, from the route ids): each local expert's routes in
assignment order, cut into blocks of at most `cap` routes. The up and down
phases run units over the blocks; a block's routes are the MFMA's N
dimension, so an expert with more than `cap` routes streams its weights
once a block.

`up_units`: unit = (block, piece `p`): gate rows `32 p ..` and up rows
`4096 + 32 p ..` of the expert's w13 over the full K, so its epilogue holds
both halves: `bf16(bf16(silu(g)) * u)` (`g`, `u` the bf16 GEMM outputs)
to `act[assignment]`, its absolute max into `amax[assignment]` (atomic).

`down_units`: unit = (block, owner `o`): rows `64 o ..` of w2 over the
full K, the activation quantized per route while it is staged
(`MOE_MID_QUANT`). The epilogue stores `y` (bf16) and the owner's sum of
`(y * (w + 1))^2` in the route's slot `o`.

`finalize_lines` (a wave a line of the 4096-column sum): each local route's
`rsqrt(sum of its 64 slots / 4096 + 1e-8) * weight`, the routes' scaled
`y * (w + 1)` summed in slot order, pushed to the line's owner (the scatter
of the expert all-reduce). Non-local routes contribute nothing.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    acquire_device,
    atomic_max_pos,
    bf16_round,
    butterfly,
    fabs,
    fp8_quant8,
    ld,
    ld_i32x4,
    lds_atomic_or,
    lds_ld,
    lds_st,
    quant_factor,
    quant_scale,
    rocdl,
    rsqrt,
    sigmoid,
    silu,
    st,
    traced,
    uniform,
    unpack_bf16x8,
    wave_max,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
    BLOCKS,
    E_LOCAL,
    EXPERTS,
    INTER,
    MOE_IN_QUANT,
    MOE_MID_QUANT,
    OWNER,
    PIECE,
    POOLED,
    POST_EPS,
    RMS_EPS,
    ROUTE_EPS,
    THREADS,
    TOPK,
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.stages.exchange import (
    put_line,
    wait_lines,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.stages.tiled import (
    TILE,
    Gemm,
    Source,
    fp8_weights,
    stage_words,
    token_tiles,
)

CAP = 16  # routes of a block at most
UP_PIECES = INTER // PIECE  # 64
DOWN_OWNERS = POOLED // OWNER  # 32
MT = PIECE // 32  # weight tiles a wave a unit (two waves a half of a piece)
assert OWNER == 64 * MT
LINE_GROUPS = POOLED // 64  # the expert sum's 64-column exchange groups
EXPERT_CM = 0  # plain loads: nt streams the experts ~25% slower on MI300
EXPERT_CHUNK = 512  # K bytes a step: 256-512 B a row stream fastest on MI300
UP_CHUNKS = POOLED // EXPERT_CHUNK  # FP8 K
DOWN_CHUNKS = INTER // EXPERT_CHUNK
TILE_STRIDE = PIECE + 4  # words a token row of the up epilogue's u tile


def i32(v) -> fx.Int32:
    return fx.Int32(v)


class MoEShape:
    """The trace-time sizes of one step width, and the LDS words of the
    schedule and the expert GEMMs (offsets into the MoE's LDS page)."""

    def __init__(self, rows: int):
        self.rows = rows
        self.assignments = rows * TOPK
        self.cap = min(rows, CAP)
        self.tiles = token_tiles(self.cap)
        self.mask_words = -(-rows // 32)
        # sum over experts of ceil(count / cap)
        self.max_blocks = min(E_LOCAL, self.assignments) + self.assignments // self.cap
        at = 0

        def take(n):
            nonlocal at
            base, at = at, at + (n + 3) // 4 * 4
            return base

        self.masks = take(E_LOCAL * self.mask_words)
        self.count = take(E_LOCAL)
        self.start = take(E_LOCAL)
        self.bstart = take(E_LOCAL)
        self.nblocks = take(1)
        self.list = take(self.assignments)
        self.blocks = take(self.max_blocks)
        self.share = take(WAVES * 64)
        self.utile = take(self.tiles * TILE * TILE_STRIDE)
        self.stage = take(stage_words(self.cap, EXPERT_CHUNK * EXPERT_DEPTH))
        self.words = at


class MoEBufs(TraceInvariant):
    """A rank's MoE buffers (byte addresses, wave-uniform Int64): the route
    (`ids` i32, `wts` f32 `[rows, 8]`), the quantized expert input
    (`hq` FP8 `[rows, 4096]`, `hs` f32 `[rows]`), per assignment the
    activation (`act` bf16 `[rows * 8, 4096]`) and its max (`amax`), the
    expert output (`y` bf16 `[rows * 8, 4096]`) and its slots (`sq` f32
    `[rows * 8, 64]`); the weights: `w13` FP8 `[32, 8192, 4096]` (gate
    rows then up rows), `w13s` f32 `[32, 8192]`, `w2` FP8 `[32, 4096,
    4096]`, `w2s` f32 `[32, 4096]`, the router bias f32 `[256]`, the
    pre- and post-expert norm gains `w + 1` (f32 `[4096]`)."""

    def __init__(
        self,
        ids,
        wts,
        hq,
        hs,
        act,
        amax,
        y,
        sq,
        w13,
        w13s,
        w2,
        w2s,
        bias,
        pre_w1,
        post_w1,
        route_scale: float,
        w13_ld: int = POOLED,
        w2_ld: int = INTER,
    ):
        (
            self.ids,
            self.wts,
            self.hq,
            self.hs,
            self.act,
            self.amax,
            self.y,
            self.sq,
            self.w13,
            self.w13s,
            self.w2,
            self.w2s,
            self.bias,
            self.pre_w1,
            self.post_w1,
        ) = (
            fx.Int64(a)
            for a in (
                ids,
                wts,
                hq,
                hs,
                act,
                amax,
                y,
                sq,
                w13,
                w13s,
                w2,
                w2s,
                bias,
                pre_w1,
                post_w1,
            )
        )
        self.route_scale = route_scale
        # the expert matrices' row strides in bytes (FP8), padding included
        self.w13_ld, self.w2_ld = w13_ld, w2_ld


def _ballot(pred):
    return fx.Int64(rocdl.ballot(T.i64, pred.ir_value()))


def _wave_min(v):
    return butterfly(v, (1, 2, 4, 8, 16, 32), fx.min)


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------


@traced
def route_token(g, sh, b, logits, pooled, t):
    """Token `t`'s route and expert input (one wave); see the module doc.
    `logits`: the gather of the f32 logits as raw words (groups `2 r`,
    `2 r + 1`: rank `r`'s 32 experts' `hi + mid` and `lo` planes);
    `pooled`: the gather of the pooled row (64
    groups, rank `r` owns `8 r ..`)."""
    lane = g.lane
    lpg = pooled.lpg
    gp = wait_lines(g, pooled, [(lane, lane * i32(lpg) + t // 8)])
    lg = lane % 16
    gl = wait_lines(g, logits, [(lg, lg * i32(logits.lpg) + t // 8)])
    # zero once every lane's polls ended: the payload loads depend on all
    late = _ballot((gp[0] != 0) | (gl[0] != 0)) != fx.Int64(0)
    gate = late.select(i32(1), i32(0))

    # scores of experts 32 r + 4 (l % 8) + i, r = l // 8: rank r's groups
    # 2 r (hi + mid) and 2 r + 1 (lo)
    r = lane // 8
    hm = logits.get_words(r, 2 * r, t, lane % 8, gate)
    lo = logits.get_words(r, 2 * r + 1, t, lane % 8, gate)
    first = r * 32 + lane % 8 * 4
    bias = ld(b.bias, first, fx.Float32, 4)
    score = [
        sigmoid(hm[i].bitcast(fx.Float32) + lo[i].bitcast(fx.Float32)) for i in range(4)
    ]
    biased = [score[i] + bias[i] for i in range(4)]
    ids = [first + i32(i) for i in range(4)]
    ninf = fx.Float32(float("-inf"))
    sel, sel_score = [], []
    for _ in range_constexpr(TOPK):
        v, e = biased[0], ids[0]
        for i in range_constexpr(1, 4):
            better = biased[i] > v
            v, e = better.select(biased[i], v), better.select(ids[i], e)
        m = wave_max(v)
        k = _wave_min((v == m).select(e, i32(EXPERTS)))
        hit = [ids[i] == k for i in range(4)]
        s = fx.Float32(0.0)
        for i in range_constexpr(4):
            s = hit[i].select(score[i], s)
        s = wave_sum(s)
        biased = [hit[i].select(ninf, biased[i]) for i in range(4)]
        sel.append(k)
        sel_score.append(s)
    total = fx.Float32(0.0)
    for k in range_constexpr(TOPK):
        total = total + sel_score[k]
    inv = fx.Float32(b.route_scale) / (total + fx.Float32(ROUTE_EPS))
    my_id, my_w = sel[0], sel_score[0] * inv
    for k in range_constexpr(1, TOPK):
        here = lane == k
        my_id = here.select(sel[k], my_id)
        my_w = here.select(sel_score[k] * inv, my_w)
    if lane < TOPK:
        a = t * TOPK + lane
        st(b.ids, a, my_id, cm=CM_DEV)
        st(b.wts, a, my_w, cm=CM_DEV)
        st(b.amax, a, fx.Float32(0.0), cm=CM_DEV)

    # the pooled row: item i is group l // 8 + 8 i (rank i's), segment l % 8
    xs = []
    for i in range_constexpr(8):
        grp_i = lane // 8 + i32(8 * i)
        xs.append(pooled.get(i32(i), grp_i, t, lane % 8, gate))
    ss = fx.Float32(0.0)
    for i in range_constexpr(8):
        for e in range_constexpr(8):
            ss = ss + xs[i][e] * xs[i][e]
    rstd = rsqrt(wave_sum(ss) / fx.Float32(POOLED) + fx.Float32(RMS_EPS))
    ys, amax = [], fx.Float32(0.0)
    for i in range_constexpr(8):
        col = (lane // 8 + i32(8 * i)) * 64 + lane % 8 * 8
        w = ld(b.pre_w1, col, fx.Float32, 4), ld(b.pre_w1, col + 4, fx.Float32, 4)
        y = [bf16_round(xs[i][e] * rstd * w[e // 4][e % 4]) for e in range(8)]
        for e in range_constexpr(8):
            amax = fx.max(amax, fabs(y[e]))
        ys.append((col, y))
    scale = quant_scale(wave_max(amax), MOE_IN_QUANT)
    f = quant_factor(scale, MOE_IN_QUANT)
    for col, y in ys:
        codes = fp8_quant8(y, f, MOE_IN_QUANT)
        st(
            b.hq,
            (t * POOLED + col) // 4,
            fx.Vector.from_elements(codes, fx.Int32),
            cm=CM_DEV,
        )
    if lane == 0:
        st(b.hs, t, scale, cm=CM_DEV)


@traced
def route_tokens(g, sh, b, logits, pooled):
    """Every token's `route_token`, a wave each over the grid."""
    waves = i32(BLOCKS * WAVES)
    for t in range(g.spread, i32(sh.rows), waves):
        route_token(g, sh, b, logits, pooled, t)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


@traced
def schedule(g, sh, lp, b, rank):
    """The rank's blocks into LDS page `lp` (`MoEShape` offsets): per
    local expert a token mask, its count, its first list slot and first
    block; the list of its assignments in order; `blocks[B] = expert | b <<
    8`; `nblocks`. Ends with a barrier."""
    mw = sh.mask_words
    for r in range_constexpr(-(-E_LOCAL * mw // THREADS)):
        i = g.tid + i32(r * THREADS)
        if i < E_LOCAL * mw:
            lds_st(lp + sh.masks + i, i32(0))
    fx.barrier()
    base = rank * i32(E_LOCAL)
    for r in range_constexpr(-(-sh.assignments // THREADS)):
        a = g.tid + i32(r * THREADS)
        if a < sh.assignments:
            j = ld(b.ids, a, cm=CM_DEV) - base
            if fx.Uint32(j) < fx.Uint32(E_LOCAL):
                t = a // TOPK
                lds_atomic_or(lp + sh.masks + j * mw + t // 32, i32(1) << (t % 32))
    fx.barrier()
    if g.tid < E_LOCAL:
        n = i32(0)
        for w in range_constexpr(mw):
            n = n + fx.Int32(fx.ctpop(lds_ld(lp + sh.masks + g.tid * mw + w)))
        lds_st(lp + sh.count + g.tid, n)
    fx.barrier()
    if g.tid < E_LOCAL:
        start, bstart, nb = i32(0), i32(0), i32(0)
        for e in range_constexpr(E_LOCAL):
            n = lds_ld(lp + sh.count + e)
            blocks = (n + (sh.cap - 1)) // sh.cap
            below = i32(e) < g.tid
            start = start + below.select(n, i32(0))
            bstart = bstart + below.select(blocks, i32(0))
            nb = nb + blocks
        lds_st(lp + sh.start + g.tid, start)
        lds_st(lp + sh.bstart + g.tid, bstart)
        if g.tid == 0:
            lds_st(lp + sh.nblocks, nb)
    fx.barrier()
    for r in range_constexpr(-(-sh.assignments // THREADS)):
        a = g.tid + i32(r * THREADS)
        if a < sh.assignments:
            j = ld(b.ids, a, cm=CM_DEV) - base
            if fx.Uint32(j) < fx.Uint32(E_LOCAL):
                t = a // TOPK
                pos = lds_ld(lp + sh.start + j)
                for w in range_constexpr(mw):
                    word = lds_ld(lp + sh.masks + j * mw + w)
                    below = (i32(w) < t // 32).select(
                        word,
                        (i32(w) == t // 32).select(
                            word & ((i32(1) << (t % 32)) - 1), i32(0)
                        ),
                    )
                    pos = pos + fx.Int32(fx.ctpop(below))
                lds_st(lp + sh.list + pos, a)
    if g.tid < E_LOCAL:
        n = lds_ld(lp + sh.count + g.tid)
        first = lds_ld(lp + sh.bstart + g.tid)
        for k in range(i32(0), (n + (sh.cap - 1)) // sh.cap, i32(1)):
            lds_st(lp + sh.blocks + first + k, g.tid | (k << 8))
    fx.barrier()


class Block(TraceInvariant):
    """Block `B`'s expert, its first list slot and its route count
    (wave-uniform)."""

    def __init__(self, sh, lp, B):
        e = uniform(lds_ld(lp + sh.blocks + B))
        self.expert = e & 255
        k = e >> 8
        self.first = uniform(lds_ld(lp + sh.start + self.expert)) + k * sh.cap
        count = uniform(lds_ld(lp + sh.count + self.expert))
        self.n = fx.min(count - k * sh.cap, i32(sh.cap))
        self.sh, self.lp = sh, lp

    def assignment(self, i):
        """Route `i` of the block (the last one past `n`)."""
        return lds_ld(self.lp + self.sh.list + self.first + fx.min(i, self.n - 1))


def nblocks(sh, lp):
    return uniform(lds_ld(lp + sh.nblocks))


# --------------------------------------------------------------------------
# up: gate and up rows, silu * mul
# --------------------------------------------------------------------------


def up_weights(g, sh, lp, b, u):
    """Unit `u`'s block, w13 and the wave's first row of each weight tile:
    waves 0-1 the piece's gate rows, 2-3 its up rows, 16-row tiles `wave % 2
    + 2 mt`."""
    blk = Block(sh, lp, u // UP_PIECES)
    w = fp8_weights(
        b.w13 + fx.Int64(blk.expert) * fx.Int64(2 * INTER * b.w13_ld),
        2 * INTER,
        POOLED,
        cm=EXPERT_CM,
        row_bytes=b.w13_ld,
    )
    base = g.wave // 2 * i32(INTER) + u % i32(UP_PIECES) * i32(PIECE)
    firsts = [base + (g.wave % 2 + i32(2 * mt)) * i32(TILE) for mt in range(MT)]
    return blk, w, firsts


def lane_rows(g, firsts):
    return [f + g.lane % TILE for f in firsts]


def ahead(g, units, u, weights):
    """`Gemm.run`'s `ahead` for the CTA's unit after `u`."""
    nu = u + i32(BLOCKS)
    _, w, firsts = weights(fx.min(nu, units - 1))
    return w, lane_rows(g, firsts), nu < units


def first_weights(g, gemm, units, weights, count):
    """The first group's weight loads of the CTA's first unit (masked)."""
    _, w, firsts = weights(fx.min(g.bid, units - 1))
    return gemm.first_group(w, lane_rows(g, firsts), count, g.bid, g.bid < units)


@traced
def up_unit(g, sh, lp, b, gemm, unit, units, pending):
    lane, wave = g.lane, g.wave
    blk, w, firsts = up_weights(g, sh, lp, b, unit)
    p = unit % i32(UP_PIECES)
    src = Source(
        b.hq,
        POOLED,
        0,
        rows=lambda i: blk.assignment(i) // TOPK,
        live=lambda i: i < blk.n,
        nbytes=sh.rows * POOLED,
    )
    half = wave // 2  # 0: gate, 1: up
    # the epilogue's loads before the K loop: waiting for them after it would
    # also wait for the next unit's weights (vmcnt is in order)
    cols = [
        (wave % 2 + i32(2 * mt)) * i32(TILE) + lane // TILE * 4 for mt in range(MT)
    ]  # within the piece
    ws = [
        ld(b.w13s, blk.expert * (2 * INTER) + f + lane // TILE * 4, fx.Float32, 4)
        for f in firsts
    ]
    routes, hss = [], []
    for t in range_constexpr(sh.tiles):
        i = i32(t * TILE) + lane % TILE
        a = blk.assignment(i)
        hss.append(ld(b.hs, a // TOPK, fx.Float32, cm=CM_DEV))
        routes.append((i, a))
    acc, pending = gemm.run(
        src,
        w,
        lane_rows(g, firsts),
        i32(0),
        UP_CHUNKS,
        pending=pending,
        rot=g.bid,
        ahead=ahead(g, units, unit, lambda v: up_weights(g, sh, lp, b, v)),
    )

    vals = [
        [
            [bf16_round(acc[mt][t][e] * hss[t] * ws[mt][e]) for e in range(4)]
            for t in range(sh.tiles)
        ]
        for mt in range(MT)
    ]
    ut = lp + sh.utile
    if half == 1:
        for mt in range_constexpr(MT):
            for t in range_constexpr(sh.tiles):
                i, _ = routes[t]
                lds_st(
                    ut + i * TILE_STRIDE + cols[mt],
                    fx.Vector.from_elements(vals[mt][t], fx.Float32).bitcast(fx.Int32),
                )
    fx.barrier()
    if half == 0:
        for t in range_constexpr(sh.tiles):
            i, a = routes[t]
            acts, m = [], fx.Float32(0.0)
            for mt in range_constexpr(MT):
                u = lds_ld(ut + i * TILE_STRIDE + cols[mt], 4).bitcast(fx.Float32)
                act = [
                    bf16_round(bf16_round(silu(vals[mt][t][e])) * u[e])
                    for e in range(4)
                ]
                for e in range_constexpr(4):
                    m = fx.max(m, fabs(act[e]))
                acts.append(act)
            m = butterfly(m, (16, 32), fx.max)
            if i < blk.n:
                for mt in range_constexpr(MT):
                    st(
                        b.act,
                        (a * INTER + p * PIECE + cols[mt]) // 2,
                        fx.Vector.from_elements(acts[mt], fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32),
                        cm=CM_DEV,
                    )
                if lane < TILE:
                    atomic_max_pos(b.amax + fx.Int64(a) * fx.Int64(4), m)
    fx.barrier()
    return pending


@traced
def up_units(g, sh, lp, b, gemm):
    """Every (block, piece) unit over the grid: CTA `bid` takes `bid, bid
    + BLOCKS, ..`."""
    acquire_device()  # hq: cached loads, shared in the XCD's L2
    units = nblocks(sh, lp) * i32(UP_PIECES)
    pending = first_weights(
        g, gemm, units, lambda v: up_weights(g, sh, lp, b, v), UP_CHUNKS
    )
    for u in range(g.bid, units, i32(BLOCKS)):
        pending = up_unit(g, sh, lp, b, gemm, u, units, pending)


# --------------------------------------------------------------------------
# down: quantized activations, y and its slots
# --------------------------------------------------------------------------


class QuantSource(TraceInvariant):
    """The down GEMM's activations: a route's bf16 `act` row quantized to
    FP8 with its scale while it is staged (`tiled.Source` interface)."""

    def __init__(self, b, blk, nbytes):
        self.b, self.blk, self.nbytes = b, blk, nbytes

    def prepare(self, token):
        """The route's assignment, its quant factor and whether it is one of
        the block's (padding tokens load nothing)."""
        a = self.blk.assignment(token)
        amax = ld(self.b.amax, a, fx.Float32, cm=CM_DEV)
        f = quant_factor(quant_scale(amax, MOE_MID_QUANT), MOE_MID_QUANT)
        return a, f, token < self.blk.n

    def issue(self, prep, byte):
        at = (prep[0] * i32(2 * INTER) + byte * 2) // 4
        return [
            ld_i32x4(self.b.act, at + 4 * h, nbytes=self.nbytes, mask=prep[2])
            for h in range(2)
        ]

    def convert(self, raw, prep):
        f = prep[1]
        codes = fp8_quant8(unpack_bf16x8(raw[0]), f, MOE_MID_QUANT) + fp8_quant8(
            unpack_bf16x8(raw[1]), f, MOE_MID_QUANT
        )
        return fx.Vector.from_elements(codes, fx.Int32)


def down_weights(g, sh, lp, b, u):
    """Unit `u`'s block, w2 and the wave's first row of each weight tile of
    the owner's `OWNER` rows."""
    blk = Block(sh, lp, u // DOWN_OWNERS)
    w = fp8_weights(
        b.w2 + fx.Int64(blk.expert) * fx.Int64(POOLED * b.w2_ld),
        POOLED,
        INTER,
        cm=EXPERT_CM,
        row_bytes=b.w2_ld,
    )
    base = u % i32(DOWN_OWNERS) * i32(OWNER) + g.wave * i32(TILE)
    return blk, w, [base + i32(64 * mt) for mt in range(MT)]


@traced
def down_unit(g, sh, lp, b, gemm, unit, units, pending):
    lane, wave = g.lane, g.wave
    blk, w, firsts = down_weights(g, sh, lp, b, unit)
    o = unit % i32(DOWN_OWNERS)
    # the epilogue's loads before the K loop (see up_unit)
    rows = [f + lane // TILE * 4 for f in firsts]
    ws = [ld(b.w2s, blk.expert * POOLED + r, fx.Float32, 4) for r in rows]
    gain = [ld(b.post_w1, r, fx.Float32, 4) for r in rows]
    routes, amaxes = [], []
    for t in range_constexpr(sh.tiles):
        i = i32(t * TILE) + lane % TILE
        a = blk.assignment(i)
        amaxes.append(ld(b.amax, a, fx.Float32, cm=CM_DEV))
        routes.append((i, a))
    acc, pending = gemm.run(
        QuantSource(b, blk, sh.assignments * INTER * 2),
        w,
        lane_rows(g, firsts),
        i32(0),
        DOWN_CHUNKS,
        pending=pending,
        rot=g.bid,
        ahead=ahead(g, units, unit, lambda v: down_weights(g, sh, lp, b, v)),
    )

    share = lp + sh.share
    for t in range_constexpr(sh.tiles):
        i, a = routes[t]
        s2 = quant_scale(amaxes[t], MOE_MID_QUANT)
        q = fx.Float32(0.0)
        for mt in range_constexpr(MT):
            y = [bf16_round(acc[mt][t][e] * s2 * ws[mt][e]) for e in range(4)]
            if i < blk.n:
                st(
                    b.y,
                    (a * POOLED + rows[mt]) // 2,
                    fx.Vector.from_elements(y, fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32),
                    cm=CM_DEV,
                )
            for e in range_constexpr(4):
                x = y[e] * gain[mt][e]
                q = q + x * x
        q = butterfly(q, (16, 32))
        if lane < TILE:
            lds_st(share + wave * (sh.tiles * TILE) + i, q.bitcast(fx.Int32))
    fx.barrier()
    if g.tid < sh.tiles * TILE:
        i = g.tid
        total = lds_ld(share + i).bitcast(fx.Float32)
        for w_ in range_constexpr(1, WAVES):
            total = total + lds_ld(share + w_ * (sh.tiles * TILE) + i).bitcast(
                fx.Float32
            )
        if i < blk.n:
            st(b.sq, blk.assignment(i) * DOWN_OWNERS + o, total, cm=CM_DEV)
    fx.barrier()
    return pending


@traced
def down_units(g, sh, lp, b, gemm):
    acquire_device()  # act: cached loads, shared in the XCD's L2
    units = nblocks(sh, lp) * i32(DOWN_OWNERS)
    pending = first_weights(
        g, gemm, units, lambda v: down_weights(g, sh, lp, b, v), DOWN_CHUNKS
    )
    for u in range(g.bid, units, i32(BLOCKS)):
        pending = down_unit(g, sh, lp, b, gemm, u, units, pending)


EXPERT_DEPTH = 1  # chunks a group: deeper groups (more loads in flight) stream slower


def expert_gemm(sh, lp, tid):
    return Gemm(
        lp + sh.stage, sh.cap, tid, 1, MT, EXPERT_DEPTH, EXPERT_CHUNK, grouped=True
    )


# --------------------------------------------------------------------------
# finalize: the expert sum's lines
# --------------------------------------------------------------------------


@traced
def finalize_line(g, sh, lp, b, sc, rank, group, line):
    """Line `line` of column group `group` of the rank's expert sum,
    pushed through the scatter `sc`."""
    lane = g.lane
    token = line * 8 + lane // 8
    tk = fx.min(token, i32(sh.rows - 1))
    # lane l: route l % 8 of its own token
    a = tk * TOPK + lane % 8
    j = ld(b.ids, a, cm=CM_DEV) - rank * i32(E_LOCAL)
    local = fx.Uint32(j) < fx.Uint32(E_LOCAL)
    nbytes = sh.assignments * DOWN_OWNERS * 4
    slots = [
        ld(
            b.sq,
            a * DOWN_OWNERS + i32(4 * s),
            fx.Float32,
            4,
            cm=CM_DEV,
            nbytes=nbytes,
            mask=local,
        )
        for s in range(DOWN_OWNERS // 4)
    ]
    ss = slots[0][0]
    for s in range_constexpr(DOWN_OWNERS // 4):
        for e in range_constexpr(4):
            if s + e > 0:
                ss = ss + slots[s][e]
    tw = ld(b.wts, a, fx.Float32, cm=CM_DEV)
    scale = rsqrt(ss / fx.Float32(POOLED) + fx.Float32(POST_EPS)) * tw
    locals_ = _ballot(local)
    share = lp + sh.share + g.wave * 64
    lds_st(share + lane, scale.bitcast(fx.Int32))
    own = lane // 8 * 8
    sc4 = [
        lds_ld(share + own, 4).bitcast(fx.Float32),
        lds_ld(share + own + 4, 4).bitcast(fx.Float32),
    ]

    col = group * 64 + lane % 8 * 8
    gain = [ld(b.post_w1, col, fx.Float32, 4), ld(b.post_w1, col + 4, fx.Float32, 4)]
    ybytes = sh.assignments * POOLED * 2
    acc = [fx.Float32(0.0)] * 8
    for k in range_constexpr(TOPK):
        here = ((locals_ >> fx.Int64(own + k)) & fx.Int64(1)) != fx.Int64(0)
        yw = ld(
            b.y,
            ((tk * TOPK + k) * POOLED + col) // 2,
            fx.Int32,
            4,
            cm=CM_DEV,
            nbytes=ybytes,
            mask=here,
        )
        y = unpack_bf16x8(yw)
        s = sc4[k // 4][k % 4]
        for e in range_constexpr(8):
            acc[e] = here.select(acc[e] + y[e] * gain[e // 4][e % 4] * s, acc[e])
    put_line(g, sc, group, line, acc)


@traced
def finalize_lines(g, sh, lp, b, sc, rank):
    """Every line of the `LINE_GROUPS` column groups, a wave each over the grid."""
    lpg = sc.lpg
    waves = i32(BLOCKS * WAVES)
    for i in range(g.spread, i32(LINE_GROUPS * lpg), waves):
        finalize_line(g, sh, lp, b, sc, rank, i // lpg, i % lpg)
