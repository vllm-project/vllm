# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse MoE block (``Qwen4ExpSparseMoeBlock`` with the shared expert
fused as routed expert ``EXPERTS``) as grid phases, after FlyTape's
expert-major W4A4 MoE (``blocks/moe``: ``ExpertSchedule``, ``FusedExpert``).

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

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import TraceInvariant
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    bf16_round,
    drain_stores,
    exp,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    mfma_f4,
    pack_bf16x8,
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
DEPTH = 3  # K steps of loads in flight at the MFMAs (FlyTape: 3 beats 6 at B128)
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


def router(g, gemm, w_gate, x, part):
    """The router logits' K-slice partials to ``part``. ``w_gate``: the 513 x
    HIDDEN bf16 gate with the shared expert's gate as its last row."""
    plan = router_plan()
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        tiled.Source(x, HIDDEN),
        tiled.Bf16Weights(w_gate, E_ALL, HIDDEN),
        plan,
        tiled.partial_epilogue(gemm, plan, part),
    )


@traced
def quantize_rows(g, x, x4, x4s, rows: int):
    """``x`` [rows, HIDDEN] bf16 as MXFP4: a thread a 32-value group, four code
    words to ``x4`` and the scale code to ``x4s`` (a word each)."""
    for unit in range(g.bid * THREADS + g.tid, rows * XGROUPS, BLOCKS * THREADS):
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
    for row in range(g.bid * WAVES + g.wave, rows, BLOCKS * WAVES):
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


@traced
def experts(g, s: Shape, lds, route_buf, x4, x4s, w13, w13s, w2, w2s, part):
    """Build the schedule, then every unit of it (CTA ``bid``: units ``bid, bid
    + BLOCKS, ..``)."""
    build_schedule(g.tid, s, lds.index, lds.units, lds.scan, route_buf)
    n = lds_ld(lds.index + 2 * E_ALL)
    for u in range(g.bid, n, BLOCKS):
        expert = uniform(lds_ld(lds.units + u * 3))
        start = uniform(lds_ld(lds.units + u * 3 + 1))
        count = uniform(lds_ld(lds.units + u * 3 + 2))
        ub = fx.Int64(w13) + fx.Int64(expert) * fx.Int64(s.w13_bytes)
        us = fx.Int64(w13s) + fx.Int64(expert) * fx.Int64(s.w13_scale_bytes)
        db = fx.Int64(w2) + fx.Int64(expert) * fx.Int64(s.w2_bytes)
        ds = fx.Int64(w2s) + fx.Int64(expert) * fx.Int64(s.w2_scale_bytes)
        unit(g, s, lds, x4, x4s, ub, us, db, ds, start, count, part)


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
def unit(g, s, lds, x4, x4s, ub, us, db, ds, start, count, part):
    """One (expert, <= 16 tokens) unit: gate/up -> silu * up -> MXFP4 -> down ->
    f32 partials of each (row, slot)."""
    lane, wave = g.lane, g.wave
    row, group = lane % TILE, lane // TILE
    gather_rows(g, s, lds, x4, x4s, start, count)
    fx.barrier()
    rip_tiles = s.rip // TILE
    scale_tiles = s.w13_scale_cols // 8
    base_tile = wave * UP_TILES
    even_base = (base_tile % 2) == 0
    accs = [zeros(4) for _ in range(2 * UP_TILES)]
    scales = {}

    def tile_of(t):
        return base_tile + (t % UP_TILES) + (t // UP_TILES) * rip_tiles

    def issue(k):
        weights = []
        for t in range_constexpr(2 * UP_TILES):
            at = (tile_of(t) * (TILE * s.w13_row_bytes) + 1024 * k + 16 * lane) // 4
            weights.append(ld_i32x4(ub, at, CM_NT))
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
        col = (base_tile + t) * TILE + group * 4
        vals = [silu(accs[t][e]) * accs[UP_TILES + t][e] for e in range(4)]
        lds_st(
            lds.x + row * RI + col,
            fx.Vector.from_elements(vals, fx.Float32).bitcast(fx.Int32),
        )
    fx.barrier()
    for r in range_constexpr(-(-(TILE * AGROUPS) // THREADS)):
        at = g.tid + i32(r * THREADS)
        if at < TILE * AGROUPS:
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
        live = q < AGROUPS
        qq = live.select(q, i32(0))
        raw = lds_ld(lds.act + row * (RI // 8) + qq * 4, 4)
        xb.append(live.select(raw, zeros(4, fx.Int32)))
        xc.append(live.select(lds_ld(lds.acts + row * AGROUPS + qq) & 0xFF, i32(127)))
    dest = []
    for qd in range_constexpr(4):
        r = i32(4 * qd) + group
        dest.append((r < count, _assignment(lds.index, s, start, count, r)))
    for blk in range(i32(0), i32(DOWN_TILES // DOWN_BLOCK), i32(1)):
        down_block(g, s, lds, db, ds, part, wave, lane, dest, blk, xb, xc)
    fx.barrier()


@traced
def down_block(g, s, lds, db, ds, part, wave, lane, dest, blk, xb, xc):
    first = wave * DOWN_TILES + blk * DOWN_BLOCK
    partials = [zeros(4) for _ in range(DOWN_BLOCK)]
    scale_tiles = s.w2_scale_cols // 8

    def issue(k):
        at = [
            ((first + t) * (TILE * s.w2_row_bytes) + 1024 * k + 16 * lane) // 4
            for t in range(DOWN_BLOCK)
        ]
        if const_expr((k + 1) * 128 > RI):
            # the K step reaches into the padding past RI: those lanes' (zero)
            # weights are not read
            live = (lane // TILE) * mx.GROUP + 128 * k < RI
            weights = [
                ld(db, a, fx.Int32, 4, cm=CM_NT, nbytes=s.w2_bytes, mask=live)
                for a in at
            ]
        else:
            weights = [ld_i32x4(db, a, CM_NT) for a in at]
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
def combine(g, xch, route_buf, part, rows: int):
    """``y[b] = sum_j w[b, j] part[b, j]`` in slot order, a wave an exchange
    line: lane ``l`` sums 8 columns of token ``l // 8`` and pushes them as four
    packed bf16 dwords to every rank, then lane 0 the line's flag."""
    lpg = ex_.lines_per_group(rows)
    lines = ex_.GROUPS * lpg
    row = xch._row(xch.rank)
    for line in range(g.bid * WAVES + g.wave, lines, BLOCKS * WAVES):
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
        sums = [fx.Float32(0.0)] * 8
        for j in range_constexpr(SLOTS):
            sums = [sums[e] + ws[j] * ps[j][e // 4][e % 4] for e in range(8)]
        dword = ex_.line_dword(rows, grp, token, g.lane % 8)
        for base in xch.bases:
            st(base, row + dword, pack_bf16x8(sums), cm=CM_SYS)
        drain_stores()
        if g.lane == 0:
            for base in xch.bases:
                st(base, row + i32(xch.payload) + line, xch.tag, cm=CM_SYS)
