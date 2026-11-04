# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse MoE block (``Qwen4ExpSparseMoeBlock`` with the shared expert
fused as routed expert ``EXPERTS``) as grid phases: an expert-major W4A4
MoE whose units stream one expert's weights each.

- ``router``: ``logits = x @ [W_gate; w_shared_gate].T`` (bf16) in K slices.
- ``route``: a wave a row sums the slices in order and rounds to bf16 (the
  stock GEMM's output), takes the top-k (renormalized softmax weights) and
  the shared expert's ``sigmoid`` weight, into ``route`` (``[rows, SLOTS]``
  expert ids, then the weights' f32 bits); the other threads quantize ``x``
  to MXFP4 (``x4`` codes, ``x4s`` scale codes), AITER's activation scaling.
- ``experts``: every CTA builds the same (expert, <= 16 tokens) unit list
  (a counting sort of the ``rows * SLOTS`` assignments in LDS); a CTA runs a
  whole unit: the tokens' ``x4`` rows into LDS, gate/up on the scaled fp4
  MFMA, ``silu(gate) * up`` requantized to MXFP4 in LDS, down, and one f32
  partial a (row, slot) to ``part``. The experts are vLLM's tensors in place
  (``AITER_MXFP4_MXFP4``: ``shuffle_weight(16, 16)`` codes, ``e8m0_shuffle``
  scales; each of gate and up padded from ``RI`` to ``rip`` rows, the down
  projection's K likewise); the padded rows are skipped.
- ``combine``: a wave a TP exchange line, ``y = sum_j w_j part_j`` in slot
  order, rounded to bf16 and pushed to every rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    fold_arrive,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    bf16_round,
    butterfly,
    drain_stores,
    exp,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    mfma_bf16,
    mfma_f4,
    sched_barrier,
    sigmoid,
    silu,
    st,
    traced,
    uniform,
    wave_max,
    zeros,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BLOCKS,
    E_ALL,
    EXPERTS,
    HIDDEN,
    RI,
    SLOTS,
    THREADS,
    TOPK,
    WAVES,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import mx, tiled

ROUTER_SPLITS = 20  # 9 row groups x 20 = 180 units
TILE = 16
NEG_INF = float("-inf")
UP_TILES = RI // (WAVES * TILE)  # a wave's 16-row tiles of each of gate and up
DOWN_TILES = HIDDEN // (WAVES * TILE)  # a wave's 16-row output tiles
DOWN_BLOCK = 4  # output tiles a down block shares its activation operands over
STAGE_STRIDE = 4 * TILE + 4  # f32 words a staged output row: 64 columns, 4 pad
DEPTH = 3  # K steps of loads in flight at the MFMAs (3 beat 6 at 128 rows)
XGROUPS = HIDDEN // mx.GROUP  # 80 scale groups an input row
AGROUPS = RI // mx.GROUP  # 10 an intermediate row
PER = EXPERTS // 64  # a lane's experts in ``route``


def router_plan():
    return tiled.SplitK(E_ALL, HIDDEN, ROUTER_SPLITS)


def i32(v):
    return fx.Int32(v)


class Shape:
    """A step's MoE sizes: ``rows`` tokens, ``rip`` the experts' padded
    intermediate rows a rank (the weights' layout)."""

    def __init__(self, rows: int, rip: int):
        self.rows, self.rip = rows, rip
        self.assignments = rows * SLOTS
        self.mask_words = -(-rows // 32)
        self.max_units = min(E_ALL, self.assignments) + self.assignments // TILE
        # LDS words: counts, starts, unit count + pad, token masks, sorted ids
        self.mask_base = 2 * E_ALL + 8
        self.token_base = self.mask_base + self.mask_words * E_ALL
        self.index_words = self.token_base + self.assignments
        self.w13_row_bytes = HIDDEN // 2
        self.w13_bytes = 2 * rip * self.w13_row_bytes  # an expert's gate+up codes
        self.w13_scale_cols = (HIDDEN // 32 + 7) // 8 * 8
        self.w13_scale_bytes = 2 * rip * self.w13_scale_cols
        self.w2_row_bytes = rip // 2
        self.w2_bytes = HIDDEN * self.w2_row_bytes
        self.w2_scale_cols = (rip // 32 + 7) // 8 * 8
        self.w2_scale_bytes = HIDDEN * self.w2_scale_cols
        self.down_steps = -(
            -RI // 128
        )  # K steps of the down projection (the rest is padding)


@dataclass(frozen=True)
class LdsWords:
    """i32 LDS words of each region of the expert phase."""

    index: int
    units: int
    scan: int
    x: int  # a unit's gathered e2m1 rows; then the f32 bridge panel
    xs: int  # their scale codes, a word each
    act: int
    acts: int
    stage: int


def lds_words(shape: Shape) -> LdsWords:
    return LdsWords(
        index=shape.index_words,
        units=3 * shape.max_units,
        scan=8,
        x=TILE * HIDDEN // 8,
        xs=TILE * XGROUPS,
        act=TILE * RI // 8,
        acts=TILE * AGROUPS,
        stage=WAVES * TILE * STAGE_STRIDE,
    )


# --------------------------------------------------------------------------
# router and routing
# --------------------------------------------------------------------------


def router_weights(w_gate):
    return tiled.Bf16Weights(w_gate, E_ALL, HIDDEN)


def router(g, gemm, w_gate, x, part, pending=None):
    """The router logits' K-slice partials to ``part``. ``w_gate``: the 513 x
    HIDDEN bf16 gate with the shared expert's gate as its last row.
    ``pending``: the CTA's ``tiled.unit_preload`` of ``router_weights``."""
    plan = router_plan()
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        tiled.Source(x, HIDDEN),
        router_weights(w_gate),
        plan,
        tiled.partial_epilogue(gemm, plan, part),
        pending=pending,
    )


@traced
def quantize_rows(g, x, x4, x4s, rows: int):
    """``x`` [rows, HIDDEN] bf16 as MXFP4: a thread a 32-value group, four code
    words to ``x4`` and the scale code to ``x4s`` (a word each)."""
    groups = rows * XGROUPS
    # runs of 64 groups a wave, spread (``Grid.spread``)
    for run in range(g.spread, -(-groups // 64), BLOCKS * WAVES):
        unit = run * 64 + g.lane
        if unit < groups:
            at = unit * (mx.GROUP // 2)  # i32 words: two bf16 a word
            words = [ld_i32x4(x, at + 4 * m, CM_DEV) for m in range(4)]
            pairs = [words[m][j] for m in range(4) for j in range(4)]
            vals = []
            for p in pairs:
                vals += [
                    (fx.Int32(p) << 16).bitcast(fx.Float32),
                    (fx.Int32(p) & i32(-65536)).bitcast(fx.Float32),
                ]
            code = mx.scale_code(mx.amax(vals))
            st(
                x4,
                unit * 4,
                fx.Vector.from_elements(mx.quant_bf16_group(pairs, code), fx.Int32),
                CM_DEV,
            )
            st(x4s, unit, code, CM_DEV)


@traced
def route(g, part, route_buf, rows: int):
    """A wave a row: the top-k of the summed router partials (rounded to bf16)
    and their renormalized softmax weights; slot TOPK is the shared expert
    with its sigmoid weight. ``route_buf``: [rows * SLOTS] ids, then [rows *
    SLOTS] f32 weights."""
    plan = router_plan()
    first = g.lane * PER
    index = [first + i32(p) for p in range(PER)]
    # mirrored (``Grid.mirror``): off quantize_rows' waves in the same phase
    for row in range(g.mirror, rows, BLOCKS * WAVES):
        vec = [
            [
                ld(
                    part,
                    (i32(s * rows) + row) * i32(plan.padded) + first + i32(4 * v),
                    fx.Float32,
                    4,
                    cm=CM_DEV,
                )
                for v in range(PER // 4)
            ]
            for s in range(plan.splits)
        ]
        shared = [
            ld(
                part,
                (i32(s * rows) + row) * i32(plan.padded) + EXPERTS,
                fx.Float32,
                cm=CM_DEV,
            )
            for s in range(plan.splits)
        ]
        values = []
        for p in range_constexpr(PER):
            tot = vec[0][p // 4][p % 4]
            for s in range_constexpr(1, plan.splits):
                tot = tot + vec[s][p // 4][p % 4]
            values.append(bf16_round(tot))
        sg = shared[0]
        for s in range_constexpr(1, plan.splits):
            sg = sg + shared[s]
        picks = []
        for j in range_constexpr(TOPK):
            best_v, best_i = values[0], index[0]
            for p in range_constexpr(1, PER):
                take = values[p] > best_v
                best_v = take.select(values[p], best_v)
                best_i = take.select(index[p], best_i)
            win_v = wave_max(best_v)
            cand = (best_v == win_v).select(best_i, i32(1 << 30))
            win_i = -wave_max(-cand)  # ties: the lowest expert
            picks.append((win_v, win_i))
            values = [
                (index[p] == win_i).select(fx.Float32(NEG_INF), values[p])
                for p in range(PER)
            ]
        ex = [exp(v - picks[0][0]) for v, _ in picks]
        norm = ex[0]
        for j in range_constexpr(1, TOPK):
            norm = norm + ex[j]
        scale = fx.Float32(1.0) / norm
        if g.lane == 0:
            base = row * SLOTS
            for j in range_constexpr(TOPK):
                st(route_buf, base + j, picks[j][1], CM_DEV)
                st(
                    route_buf,
                    rows * SLOTS + base + j,
                    (ex[j] * scale).bitcast(fx.Int32),
                    CM_DEV,
                )
            st(route_buf, base + TOPK, i32(EXPERTS), CM_DEV)
            st(
                route_buf,
                rows * SLOTS + base + TOPK,
                sigmoid(bf16_round(sg)).bitcast(fx.Int32),
                CM_DEV,
            )


# --------------------------------------------------------------------------
# expert schedule
# --------------------------------------------------------------------------


@traced
def _block_scan(tid, scan, value):
    """(exclusive prefix of ``value`` over the CTA in thread order, total)."""
    lane, wave = tid % 64, tid // 64
    x = value
    for step in range_constexpr(6):
        d = 1 << step
        x = x + (lane >= d).select(fx.shuffle_up(x, i32(d), i32(64)), i32(0))
    fx.barrier()
    if lane == 63:
        lds_st(scan + wave, x)
    fx.barrier()
    before = i32(0)
    total = i32(0)
    for w in range_constexpr(WAVES):
        t = lds_ld(scan + w)
        before = before + (i32(w) < wave).select(t, i32(0))
        total = total + t
    return before + x - value, total


@traced
def build_schedule(tid, s: Shape, idx, units, scan, route_buf):
    """Every CTA the same (expert, <= 16 token) units from ``route_buf``'s ids:
    per-expert token bitmasks, counts, starts, a flat unit list ``units``
    ([n][3]: expert, start, count) and the sorted assignments; bitmask ranks
    give every CTA the same order. The unit count is ``idx[2 E_ALL]``."""
    E, A = E_ALL, s.assignments
    rounds = -(-E // THREADS)
    for r in range_constexpr(-(-(E * s.mask_words) // THREADS)):
        at = tid + i32(r * THREADS)
        if at < E * s.mask_words:
            lds_st(idx + s.mask_base + at, i32(0))
    ids = []
    for r in range_constexpr(-(-A // THREADS)):
        a = tid + i32(r * THREADS)
        ids.append(ld(route_buf, (a < A).select(a, i32(0)), fx.Int32, cm=CM_DEV))
    fx.barrier()
    for r in range_constexpr(-(-A // THREADS)):
        a = tid + i32(r * THREADS)
        if a < A:
            token = a // SLOTS
            word = ids[r] * s.mask_words + token // 32
            fx.llvm.atomic_or(
                idx + s.mask_base + word, i32(1) << (token % 32), syncscope="workgroup"
            )
    fx.barrier()
    counts = []
    for r in range_constexpr(rounds):
        e = tid * rounds + r
        safe = (e < E).select(e, i32(0))
        count = i32(0)
        for word in range_constexpr(s.mask_words):
            count = count + fx.Int32(
                fx.ctpop(lds_ld(idx + s.mask_base + safe * s.mask_words + word))
            )
        counts.append((e < E).select(count, i32(0)))
    blocks = [(c + TILE - 1) // TILE for c in counts]
    count_sum, block_sum = counts[0], blocks[0]
    for r in range_constexpr(1, rounds):
        count_sum = count_sum + counts[r]
        block_sum = block_sum + blocks[r]
    start, _ = _block_scan(tid, scan, count_sum)
    unit_start, total = _block_scan(tid, scan, block_sum)
    for r in range_constexpr(rounds):
        e = tid * rounds + r
        if e < E:
            lds_st(idx + e, counts[r])
            lds_st(idx + E + e, start)
            for blk in range(i32(0), blocks[r], i32(1)):
                u = unit_start + blk
                lds_st(units + u * 3, e)
                lds_st(units + u * 3 + 1, start + blk * TILE)
                lds_st(units + u * 3 + 2, fx.min(counts[r] - blk * TILE, i32(TILE)))
        start = start + counts[r]
        unit_start = unit_start + blocks[r]
    if tid == 0:
        lds_st(idx + 2 * E, total)
    fx.barrier()
    for r in range_constexpr(-(-A // THREADS)):
        a = tid + i32(r * THREADS)
        if a < A:
            e = ids[r]
            token = a // SLOTS
            position = i32(0)
            for word in range_constexpr(s.mask_words):
                mask = lds_ld(idx + s.mask_base + e * s.mask_words + word)
                earlier = token // 32 > word
                same = token // 32 == word
                low = (i32(1) << (token % 32)) - 1
                bits = earlier.select(mask, same.select(mask & low, i32(0)))
                position = position + fx.Int32(fx.ctpop(bits))
            lds_st(idx + s.token_base + lds_ld(idx + E + e) + position, a)
    fx.barrier()


# --------------------------------------------------------------------------
# the fused expert unit
# --------------------------------------------------------------------------


class Lds(TraceInvariant):
    """The expert phase's LDS regions (i32 pointers)."""

    def __init__(self, **regions):
        self.__dict__.update(regions)


def _assignment(idx, s, start, count, token):
    n = (token < count).select(token, count - 1)
    return lds_ld(idx + s.token_base + start + n)


XCDS = 8  # CTAs are dispatched round-robin over the XCDs (L2 domains)
XCD_RUN = 8  # consecutive units placed on one XCD


def unit_slot(bid):
    """CTA ``bid``'s place in unit order. Runs of ``XCD_RUN`` consecutive
    units go to one XCD, so the units of one expert's rows (consecutive in
    the schedule) read its weights through the same L2; the runs rotate over
    the XCDs, so a partial round still spreads over all of them."""
    i = bid // XCDS
    return (i // XCD_RUN) * (XCDS * XCD_RUN) + (bid % XCDS) * XCD_RUN + i % XCD_RUN


# A half piece: half the intermediate rows of a unit (gate, up and the down
# projection's K), ``HALF_TILES`` 16-row tiles of each of gate and up over
# the waves as (first, count) = (0, 3), (3, 3), (6, 2), (8, 2).
HALF_TILES = UP_TILES * WAVES // 2
HALF_GROUPS = AGROUPS // 2


def pieces(n):
    """``(whole, split)`` units of a step's ``n``: a last round at most half
    full is run as two half pieces a unit, so it ends after about half a
    unit's time (a unit's time is one CU's weight stream, so a short round
    otherwise costs as much as a full one)."""
    r = n % i32(BLOCKS)
    split = ((r > 0) & (r <= BLOCKS // 2)).select(r, i32(0))
    return n - split, split


@traced
def experts(
    g, s: Shape, lds, route_buf, x4, x4s, w13, w13s, w2, w2s, part, part2, flags, tag
):
    """Build the schedule, then every piece of it (CTA ``bid``: pieces ``p, p
    + BLOCKS, ..`` from its slot ``p``): the whole units, then the split
    units' halves. A second half writes ``part2`` and marks its (row, slot)
    with ``tag`` in ``flags``."""
    build_schedule(g.tid, s, lds.index, lds.units, lds.scan, route_buf)
    n = lds_ld(lds.index + 2 * E_ALL)
    whole, split = pieces(n)
    for p in range(unit_slot(g.bid), whole + 2 * split, BLOCKS):
        late = p >= whole
        u = late.select(whole + (p - whole) // 2, p)
        half = late.select(1 + (p - whole) % 2, i32(0))
        expert = uniform(lds_ld(lds.units + u * 3))
        start = uniform(lds_ld(lds.units + u * 3 + 1))
        count = uniform(lds_ld(lds.units + u * 3 + 2))
        ub = fx.Int64(w13) + fx.Int64(expert) * fx.Int64(s.w13_bytes)
        us = fx.Int64(w13s) + fx.Int64(expert) * fx.Int64(s.w13_scale_bytes)
        db = fx.Int64(w2) + fx.Int64(expert) * fx.Int64(s.w2_bytes)
        ds = fx.Int64(w2s) + fx.Int64(expert) * fx.Int64(s.w2_scale_bytes)
        unit(
            g,
            s,
            lds,
            x4,
            x4s,
            ub,
            us,
            db,
            ds,
            start,
            count,
            part,
            part2,
            flags,
            tag,
            uniform(half),
        )


@traced
def gather_rows(g, s, lds, x4, x4s, start, count):
    """The unit's 16 token rows of ``x4`` (1280 bytes each) and their scale
    codes into LDS, every load issued before the first store; a padding token
    repeats the unit's last."""
    row_words = HIDDEN // 8
    rounds = TILE * row_words // 4 // THREADS  # 16-byte pieces: 5 rounds
    srounds = -(-(TILE * XGROUPS) // THREADS)
    pieces, codes = [], []
    for r in range_constexpr(rounds):
        piece = g.tid + i32(r * THREADS)
        token, p = piece // (row_words // 4), piece % (row_words // 4)
        src = _assignment(lds.index, s, start, count, token) // SLOTS
        pieces.append(ld_i32x4(x4, src * row_words + p * 4, CM_DEV))
    for r in range_constexpr(srounds):
        at = g.tid + i32(r * THREADS)
        tk = fx.min(at // XGROUPS, i32(TILE - 1))
        src = _assignment(lds.index, s, start, count, tk) // SLOTS
        codes.append(ld(x4s, src * XGROUPS + at % XGROUPS, fx.Int32, cm=CM_DEV))
    for r in range_constexpr(rounds):
        piece = g.tid + i32(r * THREADS)
        lds_st(lds.x + piece * 4, pieces[r])
    for r in range_constexpr(srounds):
        at = g.tid + i32(r * THREADS)
        if at < TILE * XGROUPS:
            lds_st(lds.xs + at, codes[r])


def pipelined(steps: int, depth: int, issue, compute):
    """Unrolled software pipeline: step ``k + depth - 1``'s loads are issued
    before step ``k`` computes."""
    ring = [issue(k) for k in range(min(depth - 1, steps))]
    for k in range(steps):
        if k + depth - 1 < steps:
            ring.append(issue(k + depth - 1))
            sched_barrier()
        compute(k, ring.pop(0))


@traced
def unit(
    g, s, lds, x4, x4s, ub, us, db, ds, start, count, part, part2, flags, tag, half
):
    """One (expert, <= 16 tokens) unit, or one half of it (``half`` 1, 2; 0 the
    whole unit): gate/up -> silu * up -> MXFP4 -> down -> f32 partials of
    each (row, slot), a second half's into ``part2``."""
    lane, wave = g.lane, g.wave
    whole = half == 0
    hh = half - 1
    # the wave's gate/up tiles: 5 of the unit's 20, or 3/3/2/2 of a half's 10
    first = (wave < 2).select(wave * 3, wave * 2 + 2)
    tiles = whole.select(i32(UP_TILES), (wave < 2).select(i32(3), i32(2)))

    def group_live(q):
        return whole | (q // HALF_GROUPS == hh)

    row, group = lane % TILE, lane // TILE
    gather_rows(g, s, lds, x4, x4s, start, count)
    fx.barrier()
    rip_tiles = s.rip // TILE
    scale_tiles = s.w13_scale_cols // 8
    base_tile = whole.select(wave * UP_TILES, hh * HALF_TILES + first)
    even_base = (base_tile % 2) == 0
    accs = [zeros(4) for _ in range(2 * UP_TILES)]
    scales = {}

    def tile_of(t):
        return base_tile + (t % UP_TILES) + (t // UP_TILES) * rip_tiles

    def issue(k):
        weights = []
        for t in range_constexpr(2 * UP_TILES):
            at = (tile_of(t) * (TILE * s.w13_row_bytes) + 1024 * k + 16 * lane) // 4
            on = i32(t % UP_TILES) < tiles
            weights.append(
                ld(ub, at, fx.Int32, 4, cm=CM_NT, nbytes=s.w13_bytes, mask=on)
            )
        if const_expr(k % 2 == 0):
            for proj in range_constexpr(2):
                for pair in range_constexpr((UP_TILES + 1) // 2):
                    tp = base_tile // 2 + proj * (rip_tiles // 2) + pair
                    scales[proj, pair, k // 2] = ld(
                        us, (tp * scale_tiles + k // 2) * 64 + lane, fx.Int32, cm=CM_NT
                    )
        q = 4 * k + group
        xb = lds_ld(lds.x + row * (HIDDEN // 8) + q * 4, 4)
        xc = lds_ld(lds.xs + row * XGROUPS + q) & 0xFF
        return weights, xb, xc

    def compute(k, loaded):
        weights, xb, xc = loaded
        for t in range_constexpr(2 * UP_TILES):
            odd = tile_of(t) % 2
            local = t % UP_TILES
            even_code = scales[t // UP_TILES, local // 2, k // 2]
            odd_code = scales[t // UP_TILES, (local + 1) // 2, k // 2]
            code = even_base.select(even_code, odd_code)
            sa = (code >> (odd * 8)) & 0x00FF00FF
            accs[t] = mfma_f4(weights[t], xb, accs[t], sa, xc, opsel_a=2 * (k % 2))

    pipelined(HIDDEN // 128, DEPTH, issue, compute)
    fx.barrier()
    # the f32 bridge panel [16 tokens][RI] aliases the gathered rows
    for t in range_constexpr(UP_TILES):
        if i32(t) < tiles:
            col = (base_tile + t) * TILE + group * 4
            vals = [silu(accs[t][e]) * accs[UP_TILES + t][e] for e in range(4)]
            lds_st(
                lds.x + row * RI + col,
                fx.Vector.from_elements(vals, fx.Float32).bitcast(fx.Int32),
            )
    fx.barrier()
    for r in range_constexpr(-(-(TILE * AGROUPS) // THREADS)):
        at = g.tid + i32(r * THREADS)
        if (at < TILE * AGROUPS) & group_live(at % AGROUPS):
            token, q = at // AGROUPS, at % AGROUPS
            vals = []
            for m in range_constexpr(8):
                v4 = lds_ld(lds.x + token * RI + q * 32 + 4 * m, 4).bitcast(fx.Float32)
                vals += [v4[e] for e in range(4)]
            code = mx.scale_code(mx.amax(vals))
            lds_st(
                lds.act + token * (RI // 8) + q * 4,
                fx.Vector.from_elements(mx.quant_f32_group(vals, code), fx.Int32),
            )
            lds_st(lds.acts + token * AGROUPS + q, code)
    fx.barrier()
    xb, xc = [], []
    for k in range_constexpr(s.down_steps):
        q = 4 * k + group
        live = (q < AGROUPS) & group_live(q)
        qq = live.select(q, i32(0))
        raw = lds_ld(lds.act + row * (RI // 8) + qq * 4, 4)
        xb.append(live.select(raw, zeros(4, fx.Int32)))
        xc.append(live.select(lds_ld(lds.acts + row * AGROUPS + qq) & 0xFF, i32(127)))
    dest = []
    for qd in range_constexpr(4):
        r = i32(4 * qd) + group
        dest.append((r < count, _assignment(lds.index, s, start, count, r)))
    out = (half == 2).select(fx.Int64(part2), fx.Int64(part))
    for blk in range(i32(0), i32(DOWN_TILES // DOWN_BLOCK), i32(1)):
        down_block(g, s, lds, db, ds, out, wave, lane, dest, blk, xb, xc, group_live)
    fx.barrier()
    if (half == 2) & (g.tid < count):
        st(flags, _assignment(lds.index, s, start, count, g.tid), tag, CM_DEV)


@traced
def down_block(g, s, lds, db, ds, part, wave, lane, dest, blk, xb, xc, group_live):
    first = wave * DOWN_TILES + blk * DOWN_BLOCK
    partials = [zeros(4) for _ in range(DOWN_BLOCK)]
    scale_tiles = s.w2_scale_cols // 8

    def issue(k):
        at = [
            ((first + t) * (TILE * s.w2_row_bytes) + 1024 * k + 16 * lane) // 4
            for t in range(DOWN_BLOCK)
        ]
        # a lane's K group past RI (the padding) or outside the half is not read
        q = 4 * k + lane // TILE
        live = (q < AGROUPS) & group_live(q)
        weights = [
            ld(db, a, fx.Int32, 4, cm=CM_NT, nbytes=s.w2_bytes, mask=live) for a in at
        ]
        return (
            weights,
            [
                ld(
                    ds,
                    ((first // 2 + p) * scale_tiles + k // 2) * 64 + lane,
                    fx.Int32,
                    cm=CM_NT,
                )
                for p in range(DOWN_BLOCK // 2)
            ],
        )

    def compute(k, loaded):
        weights, scales = loaded
        for t in range_constexpr(DOWN_BLOCK):
            partials[t] = mfma_f4(
                weights[t],
                xb[k],
                partials[t],
                scales[t // 2],
                xc[k],
                opsel_a=2 * (k % 2) + t % 2,
            )

    pipelined(s.down_steps, DEPTH, issue, compute)
    # the block's 16 tokens x 64 columns through the wave's stage slot, stored
    # as 4 rows x 256 bytes an instruction
    row, group = lane % TILE, lane // TILE
    st_base = lds.stage + wave * (TILE * STAGE_STRIDE)
    for t in range_constexpr(DOWN_BLOCK):
        lds_st(
            st_base + row * STAGE_STRIDE + (4 * t + group) * 4,
            partials[t].bitcast(fx.Int32),
        )
    column = first * TILE + row * 4
    for qd in range_constexpr(4):
        valid, a = dest[qd]
        v = lds_ld(st_base + (4 * qd + group) * STAGE_STRIDE + row * 4, 4)
        if valid:
            st(part, a * HIDDEN + column, v, CM_DEV)


# --------------------------------------------------------------------------
# combine -> TP exchange
# --------------------------------------------------------------------------


@traced
def combine(g, xch, route_buf, part, part2, flags, rows: int):
    """``y[b] = sum_j w[b, j] part[b, j]`` in slot order, a wave an exchange
    line: lane ``l`` sums 8 columns of token ``l // 8`` and pushes them to
    every rank (a partial line's lanes past the rows repeat the last row and
    push nothing), then lane 0 the line's flag."""
    lpg = ex_.lines_per_group(rows)
    lines = ex_.GROUPS * lpg
    row = xch._row(xch.rank)
    for line in range(g.spread, lines, BLOCKS * WAVES):  # ``Grid.spread``
        grp, blk = line // lpg, line % lpg
        token = blk * ex_.TOKENS + g.lane // 8
        tk = (token < rows).select(token, i32(rows - 1))
        col = grp * ex_.COLS + (g.lane % 8) * 8
        ws = [
            ld(route_buf, rows * SLOTS + tk * SLOTS + j, fx.Float32, cm=CM_DEV)
            for j in range(SLOTS)
        ]
        ps = [
            [
                ld(
                    part,
                    (tk * SLOTS + j) * HIDDEN + col + 4 * h,
                    fx.Float32,
                    4,
                    cm=CM_DEV,
                )
                for h in range(2)
            ]
            for j in range(SLOTS)
        ]
        # a split unit's second half of (row, slot), marked with this use's tag
        halves = [
            ld(flags, tk * SLOTS + j, fx.Int32, cm=CM_DEV) == xch.tag
            for j in range(SLOTS)
        ]
        ps2 = [
            [
                ld(
                    part2,
                    (tk * SLOTS + j) * HIDDEN + col + 4 * h,
                    fx.Float32,
                    4,
                    cm=CM_DEV,
                    nbytes=rows * SLOTS * HIDDEN * 4,
                    mask=halves[j],
                )
                for h in range(2)
            ]
            for j in range(SLOTS)
        ]
        sums = [fx.Float32(0.0)] * 8
        for j in range_constexpr(SLOTS):
            sums = [
                sums[e] + ws[j] * (ps[j][e // 4][e % 4] + ps2[j][e // 4][e % 4])
                for e in range(8)
            ]
        xch.put(grp, token, g.lane % 8, sums)
        drain_stores()
        if g.lane == 0:
            for base in xch.bases:
                st(base, row + i32(ex_.FLAGS_AT) + line, xch.tag, cm=CM_SYS)


# --------------------------------------------------------------------------
# the small-batch path (rows <= SMALL_MAX)
# --------------------------------------------------------------------------
#
# A step of a few rows routes ~10 experts a row onto 256 CUs: whole units leave
# most CUs idle while each streams a whole expert. Here the work is cut by
# weight bytes instead, over three phases:
#
# - ``router``: whole bf16 logits, a CTA two or three gate rows over all of K
#   (no split-K partials to fold or sum).
# - ``route+experts_up``: every CTA routes every row itself from the logits
#   (no route phase) and builds the unit schedule (the shared expert is one
#   more unit); pieces (unit, 32 intermediate rows) run over the grid: the
#   unit's token rows of ``x`` quantized to MXFP4 in LDS, a wave a 16-row gate
#   or up tile over all of K (every load issued up front), ``silu(gate) * up``
#   requantized to the tokens' MXFP4 activation rows ``act4``.
# - ``experts_down+allreduce``: CTA ``t`` owns output rows ``16 t ..`` of
#   every unit's down projection (its waves split the units), the (row, slot)
#   results summed in slot order with the routing weights to ``y``; the last
#   of a line group's four owners pushes its exchange lines.
#
# The arithmetic is the whole unit's: the same router sums, top-k, MXFP4
# quantization and MFMA K order, so a row's result matches the expert-major
# path's unsplit units.

SMALL_MAX = 12  # widest step the small-batch path runs (16: mixed, measured)
# the router barrier runs ``stage_rows`` in its wait (``mlp``), and the layer
# kernels' ``end_phase`` runs such a hook at the resident widths only
assert SMALL_MAX <= tiled.RESIDENT_MAX
PIECES = AGROUPS  # gate/up pieces a unit: 32 intermediate rows (a scale group) each
UP_STEPS = HIDDEN // 128  # K steps of gate/up
OWNERS = HIDDEN // TILE  # down owners: 16 output rows each
LINE_OWNERS = ex_.COLS // TILE  # owners a line group
DOWN_AHEAD = 4  # units a down wave loads ahead (deeper measured slower)
MIN_KEY = -(1 << 31)
# a descending sorting network of 8 (19 compare-exchanges)
_SORT8 = (
    (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1),
    (2, 3), (4, 5), (6, 7), (2, 4), (3, 5), (1, 4), (3, 6), (1, 2), (3, 4), (5, 6),
)  # fmt: skip


def small(rows: int) -> bool:
    return rows <= SMALL_MAX


class Weights(TraceInvariant):
    """A layer's MoE weights (byte addresses): the router gate, the experts'
    gate/up and down codes and scales (the shared expert is expert EXPERTS)."""

    def __init__(self, w_gate, w13, w13s, w2, w2s):
        self.w_gate, self.w13, self.w13s = (
            fx.Int64(w_gate),
            fx.Int64(w13),
            fx.Int64(w13s),
        )
        self.w2, self.w2s = fx.Int64(w2), fx.Int64(w2s)


class Bufs(TraceInvariant):
    """The MoE's scratch (byte addresses): router partials, the route, MXFP4
    rows of ``x``, the f32 partials and their split marks. The small-batch
    path keeps its activation rows and schedule in ``mpart``, ``y`` in
    ``mpart2``."""

    def __init__(self, part, route, x4, x4s, mpart, mpart2, split):
        self.part, self.route = fx.Int64(part), fx.Int64(route)
        self.x4, self.x4s = fx.Int64(x4), fx.Int64(x4s)
        self.mpart, self.mpart2, self.split = (
            fx.Int64(mpart),
            fx.Int64(mpart2),
            fx.Int64(split),
        )


class SmallScratch(TraceInvariant):
    """The small-batch path's global regions: ``act4`` / ``act4s`` (an
    assignment's MXFP4 activation row and scale codes, a word each), ``y``
    ([rows, HIDDEN] f32)."""

    def __init__(self, s: Shape, b: Bufs):
        self.act4 = b.mpart
        self.act4s = b.mpart + fx.Int64(s.assignments * (RI // 8) * 4)
        self.y = b.mpart2


def phase_names(rows: int) -> list[str]:
    """The MoE's phases, in order; ``mlp`` ends all but the last, whose end
    is the caller's (the MLP exchange's readers follow it)."""
    if small(rows):
        return ["router", "route+experts_up", "experts_down+allreduce+combine"]
    return ["router", "route+quant", "experts", "moe_combine+allreduce+combine"]


def _lds_struct(name: str, words: dict[str, int]):
    """An ``fx.struct`` of i32 arrays (built here: this module's annotations
    are strings)."""
    fields = {k: fx.Array[fx.Int32, n, 16] for k, n in words.items()}
    return fx.struct(type(name, (), {"__annotations__": fields}))


ROUTER_STRIDE = (
    HIDDEN // 2 + 4
)  # i32 words a staged bf16 row: 16 pad bytes stagger banks


def small_work(s: Shape) -> dict[str, int]:
    """Word offsets in the small path's LDS work area, aliased by phase: every
    row's e2m1 codes ``x`` and scale codes ``xs`` (written in the router phase,
    read in the gate/up phase); the router's staged gate rows ``rw``, rows of
    ``x`` ``rx`` and wave sums ``red``, then a piece's f32 gate | up
    ``panel``; every assignment's act4 | act4s rows ``acts`` and the down
    results ``contrib``. ``words``: its size."""
    A = s.assignments
    acts = -(-A * (RI // 8 + AGROUPS) // 4) * 4
    x4 = s.rows * (HIDDEN // 8) + s.rows * XGROUPS
    staged = (ROUTER_ROWS + s.rows) * ROUTER_STRIDE
    return dict(
        x=0,
        xs=s.rows * (HIDDEN // 8),
        rw=x4,
        rx=x4 + ROUTER_ROWS * ROUTER_STRIDE,
        red=x4 + staged,
        panel=x4,
        acts=0,
        contrib=acts,
        words=max(x4 + staged + WAVES * TILE * 4, x4 + TILE * 64, acts + A * TILE),
    )


def lds_struct(s: Shape):
    """The MoE's LDS struct for ``s``'s width."""
    w = lds_words(s)
    if small(s.rows):
        return _lds_struct(
            "SmallMoeLds",
            dict(
                ids=s.assignments,
                wts=s.assignments,
                index=w.index,
                units=w.units,
                scan=w.scan,
                work=small_work(s)["words"],
            ),
        )
    return _lds_struct(
        "MoeLds",
        dict(
            index=w.index,
            units=w.units,
            scan=w.scan,
            x=w.x,
            xs=w.xs,
            act=w.act,
            acts=w.acts,
            stage=w.stage,
        ),
    )


def mlp(g, s: Shape, gemm, ml, x, w: Weights, b: Bufs, xch, end_phase, pre=None):
    """The MoE of ``x`` ([rows, HIDDEN] bf16) into exchange ``xch``: every phase
    of ``phase_names``, ``end_phase(name, between=None)`` after each but the
    last (the phase's grid barrier, ``between()`` run in its wait, as
    ``grid_sync``'s: on the small path the router's quantizes ``x`` there,
    ``stage_rows``). ``ml``: the ``lds_struct`` view; ``pre``: the router's
    weights in flight, on the small path ``prefetch``'s (issued here if
    None), else the CTA's ``tiled.unit_preload`` of ``router_weights`` (or
    None)."""
    names = phase_names(s.rows)
    if const_expr(small(s.rows)):
        work = {k: ml.work.ptr + v for k, v in small_work(s).items() if k != "words"}
        lds = Lds(
            ids=ml.ids.ptr,
            wts=ml.wts.ptr,
            index=ml.index.ptr,
            units=ml.units.ptr,
            scan=ml.scan.ptr,
            **work,
        )
        sc = SmallScratch(s, b)
        router_gemv(
            g, lds, prefetch(g, s, w) if pre is None else pre, x, b.part, s.rows
        )
        end_phase(names[0], lambda: stage_rows(g, s.rows, lds))
        small_up(g, s, lds, x, w, b, sc)
        end_phase(names[1])
        small_down(g, s, lds, w, b, sc, xch)
        return
    router(g, gemm, w.w_gate, x, b.part, pre)
    end_phase(names[0])
    quantize_rows(g, x, b.x4, b.x4s, s.rows)
    route(g, b.part, b.route, s.rows)
    end_phase(names[1])
    lds = Lds(
        index=ml.index.ptr,
        units=ml.units.ptr,
        scan=ml.scan.ptr,
        x=ml.x.ptr,
        xs=ml.xs.ptr,
        act=ml.act.ptr,
        acts=ml.acts.ptr,
        stage=ml.stage.ptr,
    )
    experts(
        g,
        s,
        lds,
        b.route,
        b.x4,
        b.x4s,
        w.w13,
        w.w13s,
        w.w2,
        w.w2s,
        b.mpart,
        b.mpart2,
        b.split,
        xch.tag,
    )
    end_phase(names[2])
    combine(g, xch, b.route, b.mpart, b.mpart2, b.split, s.rows)


# -- phase 1: router logits -------------------------------------------------

ROUTER_ROWS = -(-E_ALL // BLOCKS)  # gate rows a CTA: c, c + BLOCKS, ..
ROUTER_STEPS = HIDDEN // (WAVES * 32)  # a wave's K quarter in 32-wide MFMA steps


def prefetch(g, s: Shape, w: Weights):
    """Loads ``mlp`` can issue before the MLP mix (its input does not depend
    on them): the small path's router gate rows of this CTA, whole 16-byte
    loads (16 VGPRs a thread); None for the expert-major path."""
    if not small(s.rows):
        return None
    chunks = HIDDEN // 8  # 16-byte pieces of a bf16 row
    out = []
    for r in range(-(-(ROUTER_ROWS * chunks) // THREADS)):
        at = g.tid + i32(r * THREADS)
        expert = g.bid + BLOCKS * (at // chunks)
        out.append(
            ld(
                w.w_gate,
                expert * (HIDDEN // 2) + (at % chunks) * 4,
                fx.Int32,
                4,
                cm=CM_NT,
                nbytes=E_ALL * HIDDEN * 2,
                mask=(at < ROUTER_ROWS * chunks) & (expert < E_ALL),
            )
        )
    return out


@traced
def router_gemv(g, lds, wl, x, part, rows: int):
    """The router logits whole: CTA ``c`` the gate rows ``c, c + BLOCKS, ..``
    (``wl``: ``prefetch``'s loads) and every row of ``x``, staged in LDS by
    whole 16-byte loads; the rows the A operand of one 16 x 16 MFMA tile, the
    tokens its columns, wave ``w`` K quarter ``w``; the quarters summed in wave
    order and rounded to bf16 into ``part[token * padded + e]`` (``route``'s
    logits)."""
    plan = router_plan()
    lane, wave = g.lane, g.wave
    chunks = HIDDEN // 8  # 16-byte pieces of a bf16 row
    xl = []
    for r in range_constexpr(-(-(rows * chunks) // THREADS)):
        at = g.tid + i32(r * THREADS)
        xl.append(
            ld(
                x,
                at * 4,
                fx.Int32,
                4,
                cm=CM_DEV,
                nbytes=rows * HIDDEN * 2,
                mask=at < rows * chunks,
            )
        )
    for r in range_constexpr(len(wl)):
        at = g.tid + i32(r * THREADS)
        if at < ROUTER_ROWS * chunks:
            lds_st(lds.rw + (at // chunks) * ROUTER_STRIDE + (at % chunks) * 4, wl[r])
    for r in range_constexpr(len(xl)):
        at = g.tid + i32(r * THREADS)
        if at < rows * chunks:
            lds_st(lds.rx + (at // chunks) * ROUTER_STRIDE + (at % chunks) * 4, xl[r])
    fx.barrier()
    r, grp = lane % TILE, lane // TILE
    w_on = r < ROUTER_ROWS
    x_on = r < rows
    wr = fx.min(r, i32(ROUTER_ROWS - 1))
    xr = fx.min(r, i32(rows - 1))
    k0 = wave * (HIDDEN // WAVES) + 8 * grp
    acc = zeros(4)
    for k in range_constexpr(ROUTER_STEPS):
        at = (k0 + 32 * k) // 2
        a = lds_ld(lds.rw + wr * ROUTER_STRIDE + at, 4)
        b = lds_ld(lds.rx + xr * ROUTER_STRIDE + at, 4)
        a = w_on.select(a, zeros(4, fx.Int32))
        b = x_on.select(b, zeros(4, fx.Int32))
        acc = mfma_bf16(a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), acc)
    # lanes 0..15 hold rows 0..3 of token ``lane``
    if grp == 0:
        lds_st(lds.red + (wave * TILE + r) * 4, acc.bitcast(fx.Int32))
    fx.barrier()
    if (wave == 0) & (grp == 0) & (r < rows):
        sums = lds_ld(lds.red + r * 4, 4).bitcast(fx.Float32)
        tot = [sums[e] for e in range(ROUTER_ROWS)]
        for w in range_constexpr(1, WAVES):
            more = lds_ld(lds.red + (w * TILE + r) * 4, 4).bitcast(fx.Float32)
            tot = [tot[e] + more[e] for e in range(ROUTER_ROWS)]
        for e in range_constexpr(ROUTER_ROWS):
            if g.bid + BLOCKS * e < E_ALL:
                st(
                    part,
                    r * plan.padded + g.bid + BLOCKS * e,
                    bf16_round(tot[e]),
                    CM_DEV,
                )


# -- phase 2: local routing, the schedule, the gate/up pieces ----------------


def _route_key(v, expert):
    """An order-preserving i32 key of a bf16-valued logit over ``511 -
    expert``: distinct keys, equal logits ranked by the lower expert; NaN
    last."""
    v = (v == v).select(v, fx.Float32(-3.0e38))
    bits = v.bitcast(fx.Int32)
    k = (bits < 0).select(bits ^ i32(0x7FFFFFFF), bits)
    return (k & i32(-65536)) | (i32(EXPERTS - 1) - expert)


def _key_logit(key):
    hi = key & i32(-65536)
    return (hi < 0).select(hi ^ i32(0x7FFF0000), hi).bitcast(fx.Float32)


def _route_lanes(rows: int) -> int:
    """Lanes a row's routing takes: a wave a row up to WAVES rows, else a
    32-lane half a row (two rows a wave, 16 experts a lane; past 2 WAVES rows
    in passes of 2 WAVES)."""
    return 64 if rows <= WAVES else 32


def _route_loads(g, logits, rows: int):
    """Each pass's lane's row (pass ``k``, wave ``w``'s half ``h``: row ``8 k +
    w + 4 h``): its logits of the lane's ``EXPERTS / lanes`` experts, and the
    shared expert's gate logit."""
    plan = router_plan()
    lanes = _route_lanes(rows)
    per = EXPERTS // lanes
    out = []
    for k in range(-(-rows // (WAVES * 64 // lanes))):
        r = fx.min(
            i32(k * WAVES * 64 // lanes) + g.wave + WAVES * (g.lane // lanes),
            i32(rows - 1),
        )
        first = r * plan.padded + (g.lane % lanes) * per
        v = [
            ld(logits, first + 4 * h, fx.Float32, 4, cm=CM_DEV) for h in range(per // 4)
        ]
        out.append((v, ld(logits, r * plan.padded + EXPERTS, fx.Float32, cm=CM_DEV)))
    return out


def _sort_desc(ks):
    """8 or 16 keys descending: a sorting network of 8 on each 8, then a
    bitonic merge of the two."""
    ks = list(ks)
    for base in range(0, len(ks), 8):
        for a, b in _SORT8:
            i, j = base + a, base + b
            ks[i], ks[j] = fx.max(ks[i], ks[j]), fx.min(ks[i], ks[j])
    if len(ks) == 16:
        ks = ks[:8] + ks[8:][::-1]
        d = 8
        while d >= 1:
            for i in range(16):
                if i & d == 0:
                    ks[i], ks[i + d] = (
                        fx.max(ks[i], ks[i + d]),
                        fx.min(ks[i], ks[i + d]),
                    )
            d //= 2
    return ks


@traced
def route_rows(g, loaded, ids, wts, rows: int):
    """``route`` of the lane's rows from their logits (``_route_loads``, a
    pass each) into LDS (``ids``, ``wts``: [rows * SLOTS]), the passes
    interleaved: the lane's keys sorted, a pick one max of keys over the row's
    lanes."""
    lanes = _route_lanes(rows)
    per = EXPERTS // lanes
    offsets = (32, 16, 8, 4, 2, 1)[6 - (lanes.bit_length() - 1) :]
    pl = g.lane % lanes
    passes = len(loaded)
    keys = [
        _sort_desc([_route_key(v[p // 4][p % 4], pl * per + p) for p in range(per)])
        for v, _ in loaded
    ]
    mine = [i32(MIN_KEY)] * passes
    top = [fx.Float32(0.0)] * passes
    total = [fx.Float32(0.0)] * passes
    mex = [fx.Float32(0.0)] * passes
    for j in range_constexpr(TOPK):
        for k in range_constexpr(passes):
            ks = keys[k]
            m = butterfly(ks[0], offsets, fx.max)
            hit = ks[0] == m
            keys[k] = [hit.select(ks[p + 1], ks[p]) for p in range(per - 1)] + [
                hit.select(i32(MIN_KEY), ks[per - 1])
            ]
            lg = _key_logit(m)
            if const_expr(j == 0):
                top[k] = lg
            ex = exp(lg - top[k])
            total[k] = ex if j == 0 else total[k] + ex
            mine[k] = (pl == j).select(m, mine[k])
            mex[k] = (pl == j).select(ex, mex[k])
    for k in range_constexpr(passes):
        r = k * WAVES * 64 // lanes + g.wave + WAVES * (g.lane // lanes)
        scale = fx.Float32(1.0) / total[k]
        if (r < rows) & (pl < TOPK):
            lds_st(ids + r * SLOTS + pl, i32(EXPERTS - 1) - (mine[k] & 0xFFFF))
            lds_st(wts + r * SLOTS + pl, (mex[k] * scale).bitcast(fx.Int32))
        if (r < rows) & (pl == TOPK):
            lds_st(ids + r * SLOTS + TOPK, i32(EXPERTS))
            lds_st(wts + r * SLOTS + TOPK, sigmoid(loaded[k][1]).bitcast(fx.Int32))


@traced
def stage_rows(g, rows: int, lds):
    """``quantize_rows`` from the router's staged bf16 rows (``lds.rx``) into
    LDS: the e2m1 rows ``lds.x`` ([rows][HIDDEN / 8] words), their scale codes
    ``lds.xs`` ([rows][XGROUPS]), for the gate/up phase; run in the router
    barrier's wait (no global access)."""
    for r in range_constexpr(-(-(rows * XGROUPS) // THREADS)):
        unit = g.tid + i32(r * THREADS)
        u = fx.min(unit, i32(rows * XGROUPS - 1))
        at = lds.rx + (u // XGROUPS) * ROUTER_STRIDE + (u % XGROUPS) * (mx.GROUP // 2)
        words = [lds_ld(at + 4 * m, 4) for m in range(4)]
        pairs = [words[m][j] for m in range(4) for j in range(4)]
        vals = []
        for p in pairs:
            vals += [
                (fx.Int32(p) << 16).bitcast(fx.Float32),
                (fx.Int32(p) & i32(-65536)).bitcast(fx.Float32),
            ]
        code = mx.scale_code(mx.amax(vals))
        if unit < rows * XGROUPS:
            lds_st(
                lds.x + unit * 4,
                fx.Vector.from_elements(mx.quant_bf16_group(pairs, code), fx.Int32),
            )
            lds_st(lds.xs + unit, code)


def _piece_loads(g, s, lds, w, p, total):
    """Piece ``p``'s gate/up operands for this wave: 16-row tile ``w % 2`` of
    gate (w < 2) or up of the piece's 32 rows, all of K, and their scale
    words; none past ``total`` (no memory access)."""
    live = p < total
    pp = live.select(p, i32(0))
    u, blk = pp // PIECES, pp % PIECES
    expert = uniform(lds_ld(lds.units + u * 3))
    ub = w.w13 + fx.Int64(expert) * fx.Int64(s.w13_bytes)
    us = w.w13s + fx.Int64(expert) * fx.Int64(s.w13_scale_bytes)
    proj, half = g.wave // 2, g.wave % 2
    rip_tiles = s.rip // TILE
    tile = proj * rip_tiles + blk * 2 + half
    pair = proj * (rip_tiles // 2) + blk
    scale_tiles = s.w13_scale_cols // 8
    weights = [
        ld(
            ub,
            (tile * (TILE * s.w13_row_bytes) + 1024 * k + 16 * g.lane) // 4,
            fx.Int32,
            4,
            cm=CM_NT,
            nbytes=s.w13_bytes,
            mask=live,
        )
        for k in range(UP_STEPS)
    ]
    codes = [
        ld(
            us,
            (pair * scale_tiles + k) * 64 + g.lane,
            fx.Int32,
            cm=CM_NT,
            nbytes=s.w13_scale_bytes,
            mask=live,
        )
        for k in range(UP_STEPS // 2)
    ]
    return weights + codes


@traced
def up_piece(g, s, lds, sc, p, loaded):
    """Piece ``p`` (unit ``p // PIECES``, intermediate rows ``32 (p % PIECES)
    ..``) on its operands (``_piece_loads``) against its tokens' rows of the
    staged ``x``: ``silu(gate) * up`` through MXFP4 to the tokens' ``act4``
    rows."""
    lane, wave = g.lane, g.wave
    row, group = lane % TILE, lane // TILE
    proj, half = wave // 2, wave % 2
    u, blk = p // PIECES, p % PIECES
    start = uniform(lds_ld(lds.units + u * 3 + 1))
    count = uniform(lds_ld(lds.units + u * 3 + 2))
    weights, codes = loaded[:UP_STEPS], loaded[UP_STEPS:]
    xr = lds_ld(lds.index + s.token_base + start + fx.min(row, count - 1)) // SLOTS
    acc = zeros(4)
    for k in range_constexpr(UP_STEPS):
        q = 4 * k + group
        xb = lds_ld(lds.x + xr * (HIDDEN // 8) + q * 4, 4)
        xc = lds_ld(lds.xs + xr * XGROUPS + q) & 0xFF
        sa = (codes[k // 2] >> (half * 8)) & 0x00FF00FF
        acc = mfma_f4(weights[k], xb, acc, sa, xc, opsel_a=2 * (k % 2))
    # the f32 panel [16 tokens][gate 32 | up 32]
    lds_st(
        lds.panel + row * 64 + proj * 32 + half * 16 + group * 4, acc.bitcast(fx.Int32)
    )
    fx.barrier()
    # wave 0: lane (token lane % 16, quarter lane / 16) requantizes 8 of the
    # token's 32 products, its quarter's code word; the scale from the four
    # quarters' amax (the group's 32 values, mx.quant_f32_group's words)
    if wave == 0:
        at = lds.panel + row * 64 + group * 8
        gv = [lds_ld(at + 4 * h, 4).bitcast(fx.Float32) for h in range(2)]
        uv = [lds_ld(at + 32 + 4 * h, 4).bitcast(fx.Float32) for h in range(2)]
        vals = [silu(gv[e // 4][e % 4]) * uv[e // 4][e % 4] for e in range(8)]
        code = mx.scale_code(butterfly(mx.amax(vals), (32, 16), fx.max))
        word = mx.quant_f32_word(vals, mx.scale_value(code))
        if row < count:
            a = lds_ld(lds.index + s.token_base + start + row)
            st(sc.act4, a * (RI // 8) + blk * 4 + group, word, CM_DEV)
            if group == 0:
                st(sc.act4s, a * AGROUPS + blk, code, CM_DEV)
    fx.barrier()


@traced
def schedule_units(g, s, lds):
    """The small path's unit list from the route in LDS (rows <= 32, so an
    expert's tokens are one mask word and at most one unit): token masks,
    then one block scan of (units, tokens) packed in a word; ``units`` ([n][3]:
    expert, start, count), each expert's token start, ``n`` at ``index[2
    E_ALL]``. The masks must be zero."""
    E, idx = E_ALL, lds.index
    masks = idx + s.mask_base
    if g.tid < s.assignments:
        fx.llvm.atomic_or(
            masks + lds_ld(lds.ids + g.tid),
            i32(1) << (g.tid // SLOTS),
            syncscope="workgroup",
        )
    fx.barrier()
    experts = [g.tid + i32(r * THREADS) for r in range(-(-E // THREADS))]
    ms = [
        (e < E).select(lds_ld(masks + fx.min(e, i32(E - 1))), i32(0)) for e in experts
    ]
    cnts = [fx.Int32(fx.ctpop(m)) for m in ms]
    packed = i32(0)
    for r in range_constexpr(len(ms)):
        packed = packed + (ms[r] != 0).select(i32(1), i32(0)) + (cnts[r] << 16)
    before, total = _block_scan(g.tid, lds.scan, packed)
    unit, start = before & 0xFFFF, before >> 16
    for r in range_constexpr(len(ms)):
        if ms[r] != 0:
            lds_st(lds.units + unit * 3, experts[r])
            lds_st(lds.units + unit * 3 + 1, start)
            lds_st(lds.units + unit * 3 + 2, cnts[r])
            lds_st(idx + E + experts[r], start)
        unit = unit + (ms[r] != 0).select(i32(1), i32(0))
        start = start + cnts[r]
    if g.tid == 0:
        lds_st(idx + 2 * E, total & 0xFFFF)
    fx.barrier()


@traced
def schedule_tokens(g, s, lds):
    """Each unit's assignments in row order at its start (``schedule_units``)."""
    idx = lds.index
    if g.tid < s.assignments:
        e = lds_ld(lds.ids + g.tid)
        below = lds_ld(idx + s.mask_base + e) & ((i32(1) << (g.tid // SLOTS)) - 1)
        at = lds_ld(idx + E_ALL + e) + fx.Int32(fx.ctpop(below))
        lds_st(idx + s.token_base + at, g.tid)
    fx.barrier()


@traced
def small_up(g, s, lds, x, w, b, sc):
    """Route every row, list the units, then the pieces ``bid, bid + BLOCKS,
    ..`` against the rows of ``x`` the router phase left in LDS as MXFP4: past
    a piece a CTA, two pieces' operands loaded together (carrying the next
    piece's through the loop measured slower). CTA 0 also stores the route
    (``route``'s layout)."""
    A = s.assignments
    assert s.rows <= 32 and A <= THREADS
    rl = _route_loads(g, b.part, s.rows)
    for r in range_constexpr(-(-E_ALL // THREADS)):
        at = g.tid + i32(r * THREADS)
        if at < E_ALL:
            lds_st(lds.index + s.mask_base + at, i32(0))
    route_rows(g, rl, lds.ids, lds.wts, s.rows)
    fx.barrier()
    if (g.bid == 0) & (g.tid < A):
        st(b.route, g.tid, lds_ld(lds.ids + g.tid), CM_DEV)
        st(b.route, A + g.tid, lds_ld(lds.wts + g.tid), CM_DEV)
    schedule_units(g, s, lds)
    total = uniform(lds_ld(lds.index + 2 * E_ALL)) * PIECES
    schedule_tokens(g, s, lds)
    if const_expr(s.rows * TOPK * PIECES <= BLOCKS):
        # at most a piece a CTA with distinct experts
        for p in range(g.bid, total, BLOCKS):
            up_piece(g, s, lds, sc, p, _piece_loads(g, s, lds, w, p, total))
    else:
        # two pieces' operands in flight together, none carried through the loop
        for p in range(g.bid, total, 2 * BLOCKS):
            first = _piece_loads(g, s, lds, w, p, total)
            second = _piece_loads(g, s, lds, w, p + BLOCKS, total)
            up_piece(g, s, lds, sc, p, first)
            if p + BLOCKS < total:
                up_piece(g, s, lds, sc, p + BLOCKS, second)


# -- phase 3: the down projection by output owners, the exchange -------------


@traced
def small_down(g, s, lds, w, b, sc, xch):
    """CTA ``t`` < OWNERS, with the route and schedule it built in LDS in the
    previous phase: output rows ``16 t ..`` of every unit into LDS by
    (row, slot), their slot-order sum
    with the routing weights to ``y``; the line group's last owner pushes its
    lines. The first units' down weights depend only on the schedule: they
    are issued with the activation rows' loads (``stage_acts``)."""
    A, rows = s.assignments, s.rows
    assert A <= THREADS, "a thread an assignment"
    tile = g.bid
    if tile < OWNERS:
        pending = stage_acts(g, s, lds, sc, lambda: down_issue(g, s, lds, w))
        fx.barrier()
        down_units(g, s, lds, w, pending)
        fx.barrier()
        if g.tid < rows * TILE:
            r, c = g.tid // TILE, g.tid % TILE
            y = fx.Float32(0.0)
            for j in range_constexpr(SLOTS):
                wj = lds_ld(lds.wts + r * SLOTS + j).bitcast(fx.Float32)
                cj = lds_ld(lds.contrib + (r * SLOTS + j) * TILE + c).bitcast(
                    fx.Float32
                )
                y = y + wj * cj
            st(sc.y, r * HIDDEN + tile * TILE + c, y, CM_DEV)
        grp = tile // LINE_OWNERS
        if fold_arrive(g, grp, LINE_OWNERS):
            push_group(g, xch, sc.y, grp, rows)


@traced
def stage_acts(g, s, lds, sc, between):
    """Every assignment's MXFP4 activation row (``act4``, ``act4s``) into
    LDS (``lds.acts``: codes then scale words), whole 16-byte loads first;
    ``between()`` runs after them and before the stores, held there (the
    stores wait for these loads only, and its loads are in flight under
    them), its result returned."""
    quads = -(-s.assignments * (RI // 8 + AGROUPS) // 4)
    rounds = -(-quads // THREADS)
    loads = [
        ld(
            sc.act4,
            (g.tid + i32(r * THREADS)) * 4,
            fx.Int32,
            4,
            cm=CM_DEV,
            nbytes=quads * 16,
            mask=g.tid + i32(r * THREADS) < quads,
        )
        for r in range(rounds)
    ]
    sched_barrier()
    out = between()
    sched_barrier()
    for r in range_constexpr(rounds):
        at = g.tid + i32(r * THREADS)
        if at < quads:
            lds_st(lds.acts + at * 4, loads[r])
    return out


def _down_unit(g, s, lds, i):
    """The wave's ``i``-th unit (``w + 4 i``): (live, expert, the lane's
    token's assignment, the unit's token count; 0 past the unit count)."""
    n = uniform(lds_ld(lds.index + 2 * E_ALL))
    u = g.wave + i32(WAVES * i)
    live = u < n
    uu = live.select(u, i32(0))
    expert = uniform(lds_ld(lds.units + uu * 3))
    start = lds_ld(lds.units + uu * 3 + 1)
    count = live.select(lds_ld(lds.units + uu * 3 + 2), i32(0))
    token = fx.min(g.lane % TILE, fx.max(count - 1, i32(0)))
    return live, expert, lds_ld(lds.index + s.token_base + start + token), count


def _down_weights(g, s, lds, w, i):
    """Unit ``i``'s down weights of output tile ``bid`` (every K step; K past
    RI is padding) and their scale words."""
    tile, lane = g.bid, g.lane
    live, expert, _, _ = _down_unit(g, s, lds, i)
    db = w.w2 + fx.Int64(expert) * fx.Int64(s.w2_bytes)
    ds = w.w2s + fx.Int64(expert) * fx.Int64(s.w2_scale_bytes)
    weights = []
    for k in range_constexpr(s.down_steps):
        on = live & (4 * k + lane // TILE < AGROUPS)
        at = (tile * (TILE * s.w2_row_bytes) + 1024 * k + 16 * lane) // 4
        weights.append(ld(db, at, fx.Int32, 4, cm=CM_NT, nbytes=s.w2_bytes, mask=on))
    scale_tiles = s.w2_scale_cols // 8
    codes = [
        ld(
            ds,
            ((tile // 2) * scale_tiles + k) * 64 + lane,
            fx.Int32,
            cm=CM_NT,
            nbytes=s.w2_scale_bytes,
            mask=live,
        )
        for k in range(-(-s.down_steps // 2))
    ]
    return weights + codes


def _down_acts(g, s, lds, i):
    """Unit ``i``'s activation operands from LDS: (codes and scale codes a K
    step, the lane's token's assignment, the token count)."""
    live, _, a, count = _down_unit(g, s, lds, i)
    group = g.lane // TILE
    acts = s.assignments * (RI // 8)  # the scale words follow the codes in LDS
    xb, xc = [], []
    for k in range_constexpr(s.down_steps):
        q = 4 * k + group
        on = live & (q < AGROUPS)
        qq = on.select(q, i32(0))
        xb.append(lds_ld(lds.acts + a * (RI // 8) + qq * 4, 4))
        xc.append(
            on.select(lds_ld(lds.acts + acts + a * AGROUPS + qq) & 0xFF, i32(127))
        )
    return xb, xc, a, count


def _wave_units(s) -> int:
    """A down wave's units, of the step's <= rows * TOPK + 1."""
    return -(-(s.rows * TOPK + 1) // WAVES)


def down_issue(g, s, lds, w):
    """The down weights of the wave's first ``DOWN_AHEAD`` units of output
    tile ``bid``, for ``down_units``."""
    return [
        _down_weights(g, s, lds, w, i) for i in range(min(DOWN_AHEAD, _wave_units(s)))
    ]


@traced
def down_units(g, s, lds, w, pending):
    """Wave ``w``: units ``w, w + 4, ..`` of output tile ``bid``, each a K
    chain over its tokens' activation rows in LDS, ``DOWN_AHEAD`` units'
    operands loaded ahead (``pending``: the first units' weights, issued by
    ``down_issue``); a token's 16 results to ``lds.contrib[(row, slot)]``."""
    lane = g.lane
    group = lane // TILE
    odd = g.bid % 2
    units = _wave_units(s)

    def issue(i):
        return _down_weights(g, s, lds, w, i), _down_acts(g, s, lds, i)

    ring = [(p, _down_acts(g, s, lds, i)) for i, p in enumerate(pending)]
    for i in range_constexpr(units):
        if const_expr(i + DOWN_AHEAD < units):
            ring.append(issue(i + DOWN_AHEAD))
            sched_barrier()
        loaded, (xb, xc, a, count) = ring.pop(0)
        weights, codes = loaded[: s.down_steps], loaded[s.down_steps :]
        acc = zeros(4)
        for k in range_constexpr(s.down_steps):
            sa = (codes[k // 2] >> (odd * 8)) & 0x00FF00FF
            acc = mfma_f4(weights[k], xb[k], acc, sa, xc[k], opsel_a=2 * (k % 2))
        if lane % TILE < count:
            lds_st(lds.contrib + a * TILE + group * 4, acc.bitcast(fx.Int32))


@traced
def push_group(g, xch, y, grp, rows: int):
    """Line group ``grp``'s exchange lines from ``y`` ([rows, HIDDEN] f32), a
    wave a line (``combine``'s lanes: token ``l // 8``, columns ``8 (l % 8)
    ..``), then each line's flag in every rank's mailbox."""
    lpg = ex_.lines_per_group(rows)
    row = xch._row(xch.rank)
    for blk in range(g.wave, i32(lpg), i32(WAVES)):
        line = grp * lpg + blk
        token = blk * ex_.TOKENS + g.lane // 8
        tk = (token < rows).select(token, i32(rows - 1))
        col = grp * ex_.COLS + (g.lane % 8) * 8
        v = [
            ld(y, tk * HIDDEN + col + 4 * h, fx.Float32, 4, cm=CM_DEV) for h in range(2)
        ]
        xch.put(grp, token, g.lane % 8, [v[e // 4][e % 4] for e in range(8)])
        drain_stores()
        if g.lane == 0:
            for base in xch.bases:
                st(base, row + i32(ex_.FLAGS_AT) + line, xch.tag, cm=CM_SYS)
