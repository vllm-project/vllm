# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse MoE block (`Qwen3NextSparseMoeBlock` with the shared expert
fused as routed expert `EXPERTS`, AITER's W4A4 MXFP4 experts) as grid
phases, for the steps past `layout.SPLIT_MAX` rows:

- `router`: `logits = x @ [W_gate; w_shared_gate].T` (bf16, vLLM's
  fused gate) in K slices of 64-row groups, their partials to `part`.
- `route`: a wave a row, the top-k of the row's summed logits rounded to
  bf16 (the stock GEMM's output; renormalized softmax weights) and the
  shared expert's `sigmoid` weight in slot `TOPK`.
- schedule: every CTA builds the same (expert, <= 16 tokens) unit list in
  LDS (a counting sort of the `rows * SLOTS` assignments).
- `experts`: a CTA a unit (expert-major): gate/up -> `silu(gate) * up`
  requantized to MXFP4 (the stock MoE's inter-stage quantization) -> down,
  every (row, slot)'s f32 result to `mpart`.
- `major_combine`: the slot-order sum with the routing weights into the
  TP exchange.

The experts are vLLM's tensors in place: `shuffle_weight(16, 16)` codes and
`e8m0_shuffle` scales; gate and up each padded from `RI` to `rip` rows,
the down projection's K likewise.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.layout import (
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
from vllm.models.qwen3_5.amd.specialized_decode.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    drain_stores,
    exp,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
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
from vllm.models.qwen3_5.amd.specialized_decode.wide.grid import TraceInvariant
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import mx, tiled

ROUTER_SPLITS = 28  # 9 row groups x 28 = 252 units
TILE = 16
NEG_INF = float("-inf")
XGROUPS = HIDDEN // mx.GROUP  # scale groups an input row
AGROUPS = RI // mx.GROUP  # scale groups an intermediate row
UP_STEPS = HIDDEN // 128  # K steps of gate/up
PER = EXPERTS // 64  # a lane's experts in ``route``


def i32(v):
    return fx.Int32(v)


def router_plan():
    return tiled.SplitK(E_ALL, HIDDEN, ROUTER_SPLITS)


class Shape:
    """A step's MoE sizes: `rows` tokens, `rip` the experts' padded
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
        self.down_steps = -(-RI // 128)  # K steps of the down projection


class Weights(TraceInvariant):
    """A layer's MoE weights (byte addresses): the fused router gate, the
    experts' gate/up and down codes and scales (the shared expert is expert
    EXPERTS)."""

    def __init__(self, w_gate, w13, w13s, w2, w2s):
        self.w_gate, self.w13, self.w13s = (
            fx.Int64(w_gate),
            fx.Int64(w13),
            fx.Int64(w13s),
        )
        self.w2, self.w2s = fx.Int64(w2), fx.Int64(w2s)


class Bufs(TraceInvariant):
    """The MoE's scratch (byte addresses): `part` the router partials,
    `route` ([rows * SLOTS] ids, then their f32 weights), `x4` / `x4s`
    the MXFP4 rows of `x`, `mpart` / `mpart2` the (row, slot) results
    ([rows * SLOTS, HIDDEN] f32; a split unit's second half), `split`
    the second halves' tags."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, fx.Int64(v))

    def __getattr__(self, name: str) -> fx.Int64:
        # the attributes are the width's buffer names, set in __init__
        raise AttributeError(name)


def scratch_words(s: Shape) -> dict[str, int]:
    """i32 words of each MoE scratch buffer at `s`'s width."""
    return {
        "route": 2 * s.assignments,
        "mpart": s.assignments * HIDDEN,
        "mpart2": s.assignments * HIDDEN,
        "split": s.assignments,
    }


# the MoE's phases, in order; `mlp` ends all but the last, whose end is the
# caller's (the MLP exchange's readers follow it)
PHASES = ["router", "route", "experts", "moe_combine"]


def _lds_struct(name: str, words: dict[str, int]):
    """An `fx.struct` of i32 arrays (built here: this module's annotations
    are strings)."""
    fields = {k: fx.Array[fx.Int32, max(n, 4), 16] for k, n in words.items()}
    return fx.struct(type(name, (), {"__annotations__": fields}))


def lds_words(s: Shape) -> dict[str, int]:
    return dict(
        index=s.index_words,
        units=3 * s.max_units,
        scan=8,
        x=TILE * XROW,
        xs=TILE * XGROUPS,
        act=TILE * RI // 8,
        acts=TILE * AGROUPS,
        stage=WAVES * TILE * STAGE_STRIDE,
    )


def lds_struct(s: Shape):
    """The MoE's LDS struct for `s`'s width."""
    return _lds_struct(f"MoeLds{s.rows}", lds_words(s))


class Lds(TraceInvariant):
    """The MoE's LDS regions (i32 pointers)."""

    def __init__(self, ml, names):
        for k in names:
            setattr(self, k, getattr(ml, k).ptr)


# --------------------------------------------------------------------------
# router and routing
# --------------------------------------------------------------------------


def router_weights(w_gate):
    return tiled.Bf16Weights(w_gate, E_ALL, HIDDEN)


def router(g, gemm, w_gate, x, b):
    """The router logits' K-slice partials to `b.part`."""
    plan = router_plan()
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        tiled.Source(x, HIDDEN),
        router_weights(w_gate),
        plan,
        tiled.partial_epilogue(gemm, plan, b.part),
    )


@traced
def route(g, part, route_buf, rows: int):
    """A wave a row: the top-k of the summed router partials (rounded to bf16)
    and their renormalized softmax weights; slot TOPK is the shared expert
    with its sigmoid weight. `route_buf`: [rows * SLOTS] ids, then [rows *
    SLOTS] f32 weights."""
    plan = router_plan()
    first = g.lane * PER
    index = [first + i32(p) for p in range(PER)]
    for row in range(g.spread, rows, BLOCKS * WAVES):
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
        for _ in range_constexpr(TOPK):
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
# the expert schedule
# --------------------------------------------------------------------------


@traced
def _block_scan(tid, scan, value):
    """(exclusive prefix of `value` over the CTA in thread order, total)."""
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
def build_schedule(tid, s: Shape, lds, load_id):
    """Every CTA the same (expert, <= 16 token) units from the assignments'
    expert ids (`load_id(a)`): per-expert token bitmasks, counts, starts, a
    flat unit list `lds.units` (`[n][3]`: expert, start, count) and the
    sorted assignments; bitmask ranks give every CTA the same order. The
    unit count is `index[2 E_ALL]`."""
    E, A = E_ALL, s.assignments
    idx = lds.index
    rounds = -(-E // THREADS)
    for r in range_constexpr(-(-(E * s.mask_words) // THREADS)):
        at = tid + i32(r * THREADS)
        if at < E * s.mask_words:
            lds_st(idx + s.mask_base + at, i32(0))
    ids = []
    for r in range_constexpr(-(-A // THREADS)):
        a = tid + i32(r * THREADS)
        ids.append(load_id((a < A).select(a, i32(0))))
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
    start, _ = _block_scan(tid, lds.scan, count_sum)
    unit_start, total = _block_scan(tid, lds.scan, block_sum)
    for r in range_constexpr(rounds):
        e = tid * rounds + r
        if e < E:
            lds_st(idx + e, counts[r])
            lds_st(idx + E + e, start)
            for blk in range(i32(0), blocks[r], i32(1)):
                u = unit_start + blk
                lds_st(lds.units + u * 3, e)
                lds_st(lds.units + u * 3 + 1, start + blk * TILE)
                lds_st(lds.units + u * 3 + 2, fx.min(counts[r] - blk * TILE, i32(TILE)))
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


def _unit(lds, u):
    """(expert, start, count) of unit `u` (CTA-uniform)."""
    return (
        uniform(lds_ld(lds.units + u * 3)),
        uniform(lds_ld(lds.units + u * 3 + 1)),
        uniform(lds_ld(lds.units + u * 3 + 2)),
    )


def _assignment(lds, s, start, count, token):
    n = (token < count).select(token, count - 1)
    return lds_ld(lds.index + s.token_base + start + n)


def unit_count(lds):
    return uniform(lds_ld(lds.index + 2 * E_ALL))


# --------------------------------------------------------------------------
# the experts, expert-major
# --------------------------------------------------------------------------
#
# A CTA runs whole units (the Qwen3.8-Flash-Next decode's expert-major MoE):
# the unit's token rows of ``x4`` gathered into LDS once, gate/up streaming
# the expert's codes with the activations read from LDS, ``silu(gate) * up``
# requantized in LDS, then the down projection streaming the rest of the
# expert, every (row, slot)'s f32 result to ``mpart``; ``major_combine`` sums
# them with the routing weights in slot order into the exchange. A wave's
# registers in flight are then all weights.

UP_TILES = RI // (WAVES * TILE)  # a wave's 16-row tiles of each of gate and up
DOWN_TILES = HIDDEN // (WAVES * TILE)  # a wave's 16-row output tiles
DOWN_BLOCK = 4  # output tiles a block of the down stream
STAGE_STRIDE = 4 * TILE + 4  # f32 words a staged output row: 64 columns, 4 pad
XROW = HIDDEN // 8 + 4  # i32 words a gathered row: 16 pad bytes stagger banks
MAJOR_DEPTH = 3  # gate/up K steps of loads in flight at the MFMAs
DOWN_DEPTH = 6  # down (block, K step)s of loads in flight
XCDS = 8  # CTAs are dispatched round-robin over the XCDs (L2 domains)
XCD_RUN = 8  # consecutive units placed on one XCD
# a half piece: half the intermediate rows (gate, up and the down K)
HALF_TILES = UP_TILES * WAVES // 2
HALF_GROUPS = AGROUPS // 2
assert UP_TILES % 2 == 0


def unit_slot(bid):
    """CTA `bid`'s place in unit order: runs of `XCD_RUN` consecutive
    units on one XCD (an expert's units are consecutive, so a hot expert's
    weights come through one L2), the runs rotating over the XCDs."""
    i = bid // XCDS
    return (i // XCD_RUN) * (XCDS * XCD_RUN) + (bid % XCDS) * XCD_RUN + i % XCD_RUN


def pieces(n):
    """`(whole, split)` units of the step's `n`: a last round at most half
    full runs as two half pieces a unit, ending after about half a unit's
    time (a unit's time is one CU's weight stream, whatever its rows)."""
    r = n % i32(BLOCKS)
    split = ((r > 0) & (r <= BLOCKS // 2)).select(r, i32(0))
    return n - split, split


@traced
def gather_rows(g, s, lds, b, start, count):
    """The unit's 16 token rows of `x4` and their scale codes into LDS,
    every load issued before the first store; a padding token repeats the
    unit's last."""
    row_words = HIDDEN // 8
    rounds = TILE * row_words // 4 // THREADS
    srounds = -(-(TILE * XGROUPS) // THREADS)
    vecs, codes = [], []
    for r in range_constexpr(rounds):
        piece = g.tid + i32(r * THREADS)
        token, p = piece // (row_words // 4), piece % (row_words // 4)
        src = _assignment(lds, s, start, count, token) // SLOTS
        vecs.append(ld_i32x4(b.x4, src * row_words + p * 4, CM_DEV))
    for r in range_constexpr(srounds):
        at = g.tid + i32(r * THREADS)
        tk = fx.min(at // XGROUPS, i32(TILE - 1))
        src = _assignment(lds, s, start, count, tk) // SLOTS
        codes.append(ld(b.x4s, src * XGROUPS + at % XGROUPS, fx.Int32, cm=CM_DEV))
    for r in range_constexpr(rounds):
        piece = g.tid + i32(r * THREADS)
        token, p = piece // (row_words // 4), piece % (row_words // 4)
        lds_st(lds.x + token * XROW + p * 4, vecs[r])
    for r in range_constexpr(srounds):
        at = g.tid + i32(r * THREADS)
        if at < TILE * XGROUPS:
            lds_st(lds.xs + at, codes[r])


def pipelined(steps: int, depth: int, issue, compute):
    """Unrolled software pipeline: step `k + depth - 1`'s loads are issued
    before step `k` computes."""
    ring = [issue(k) for k in range(min(depth - 1, steps))]
    for k in range(steps):
        if k + depth - 1 < steps:
            ring.append(issue(k + depth - 1))
            sched_barrier()
        compute(k, ring.pop(0))


@traced
def experts(g, s, lds, w, b, tag):
    """Every piece of the schedule in LDS (CTA `bid`: pieces `p, p +
    BLOCKS, ..` from its slot `p`): the whole units, then the split units'
    halves. A second half writes `mpart2` and marks its (row, slot) with
    `tag` in `split`."""
    whole, split = pieces(unit_count(lds))
    for p in range(unit_slot(g.bid), whole + 2 * split, BLOCKS):
        late = p >= whole
        u = late.select(whole + (p - whole) // 2, p)
        half = late.select(1 + (p - whole) % 2, i32(0))
        expert, start, count = _unit(lds, u)
        e = fx.Int64(expert)
        _expert_unit(
            g,
            s,
            lds,
            b,
            w.w13 + e * fx.Int64(s.w13_bytes),
            w.w13s + e * fx.Int64(s.w13_scale_bytes),
            w.w2 + e * fx.Int64(s.w2_bytes),
            w.w2s + e * fx.Int64(s.w2_scale_bytes),
            start,
            count,
            tag,
            uniform(half),
        )


@traced
def _expert_unit(g, s, lds, b, ub, us, db, ds, start, count, tag, half):
    """One (expert, <= 16 tokens) unit, or one half of it (`half` 1, 2; 0 the
    whole unit): gate/up -> silu * up rounded to bf16 -> MXFP4 -> down -> f32
    results of each (row, slot), a second half's into `mpart2`."""
    lane, wave = g.lane, g.wave
    whole = half == 0
    hh = half - 1
    tiles = whole.select(i32(UP_TILES), i32(UP_TILES // 2))

    def group_live(q):
        return whole | (q // HALF_GROUPS == hh)

    row, group = lane % TILE, lane // TILE
    gather_rows(g, s, lds, b, start, count)
    fx.barrier()
    rip_tiles = s.rip // TILE
    scale_tiles = s.w13_scale_cols // 8
    # even: a tile pair shares a scale word, its odd tile the high bytes
    base_tile = whole.select(wave * UP_TILES, hh * HALF_TILES + wave * (UP_TILES // 2))
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
                for pair in range_constexpr(UP_TILES // 2):
                    tp = base_tile // 2 + proj * (rip_tiles // 2) + pair
                    scales[proj, pair, k // 2] = ld(
                        us,
                        (tp * scale_tiles + k // 2) * 64 + lane,
                        fx.Int32,
                        cm=CM_NT,
                        nbytes=s.w13_scale_bytes,
                        mask=i32(2 * pair) < tiles,
                    )
        q = 4 * k + group
        xb = lds_ld(lds.x + row * XROW + q * 4, 4)
        xc = lds_ld(lds.xs + row * XGROUPS + q) & 0xFF
        return weights, xb, xc

    def compute(k, loaded):
        weights, xb, xc = loaded
        for t in range_constexpr(2 * UP_TILES):
            local = t % UP_TILES
            code = scales[t // UP_TILES, local // 2, k // 2]
            sa = (code >> ((local % 2) * 8)) & 0x00FF00FF
            accs[t] = mfma_f4(weights[t], xb, accs[t], sa, xc, opsel_a=2 * (k % 2))

    pipelined(UP_STEPS, MAJOR_DEPTH, issue, compute)
    fx.barrier()
    _bridge(g, lds, accs, base_tile, tiles, group_live)
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
        dest.append((r < count, _assignment(lds, s, start, count, r)))
    out = (half == 2).select(b.mpart2, b.mpart)
    _down_stream(g, s, lds, db, ds, out, dest, xb, xc, group_live)
    fx.barrier()
    if (half == 2) & (g.tid < count):
        st(b.split, _assignment(lds, s, start, count, g.tid), tag, CM_DEV)


@traced
def _bridge(g, lds, accs, base_tile, tiles, group_live):
    """`bf16(silu(gate) * up)` of the wave's tiles to the f32 panel
    `[16 tokens][RI]` (aliasing the gathered rows), then a thread a (token, scale
    group) requantizes it to the MXFP4 rows `act` / `acts`."""
    row, group = g.lane % TILE, g.lane // TILE
    for t in range_constexpr(UP_TILES):
        if i32(t) < tiles:
            col = (base_tile + t) * TILE + group * 4
            vals = [
                bf16_round(silu(accs[t][e]) * accs[UP_TILES + t][e]) for e in range(4)
            ]
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


@traced
def _down_stream(g, s, lds, db, ds, out, dest, xb, xc, group_live):
    """The wave's `DOWN_TILES` output tiles in blocks of `DOWN_BLOCK`, one
    pipeline over every (block, K step); a block's 16 tokens x 64 columns go
    through the wave's stage slot to `out`'s (row, slot) rows."""
    lane, wave = g.lane, g.wave
    steps = s.down_steps
    scale_tiles = s.w2_scale_cols // 8
    q = [4 * k + lane // TILE for k in range(steps)]
    live = [(q[k] < AGROUPS) & group_live(q[k]) for k in range(steps)]
    acc = [None] * DOWN_BLOCK

    def first_tile(blk):
        return wave * DOWN_TILES + blk * DOWN_BLOCK

    def issue(i):
        blk, k = divmod(i, steps)
        first = first_tile(blk)
        weights = [
            ld(
                db,
                ((first + t) * (TILE * s.w2_row_bytes) + 1024 * k + 16 * lane) // 4,
                fx.Int32,
                4,
                cm=CM_NT,
                nbytes=s.w2_bytes,
                mask=live[k],
            )
            for t in range(DOWN_BLOCK)
        ]
        codes = [
            ld(
                ds,
                ((first // 2 + p) * scale_tiles + k // 2) * 64 + lane,
                fx.Int32,
                cm=CM_NT,
            )
            for p in range(DOWN_BLOCK // 2)
        ]
        return weights, codes

    def compute(i, loaded):
        blk, k = divmod(i, steps)
        weights, codes = loaded
        for t in range_constexpr(DOWN_BLOCK):
            acc[t] = mfma_f4(
                weights[t],
                xb[k],
                zeros(4) if k == 0 else acc[t],
                codes[t // 2],
                xc[k],
                opsel_a=2 * (k % 2) + t % 2,
            )
        if const_expr(k == steps - 1):
            _down_store(g, lds, out, dest, first_tile(blk), acc)

    pipelined(DOWN_TILES // DOWN_BLOCK * steps, DOWN_DEPTH, issue, compute)


@traced
def _down_store(g, lds, out, dest, first, acc):
    row, group = g.lane % TILE, g.lane // TILE
    base = lds.stage + g.wave * (TILE * STAGE_STRIDE)
    for t in range_constexpr(DOWN_BLOCK):
        lds_st(
            base + row * STAGE_STRIDE + (4 * t + group) * 4, acc[t].bitcast(fx.Int32)
        )
    column = first * TILE + row * 4
    for qd in range_constexpr(4):
        valid, a = dest[qd]
        v = lds_ld(base + (4 * qd + group) * STAGE_STRIDE + row * 4, 4)
        if valid:
            st(out, a * HIDDEN + column, v, CM_DEV)


@traced
def major_combine(g, xch, b, rows: int):
    """`y[r] = sum_j w[r, j] part[r, j]` in slot order (a split unit's two
    halves added first), a wave an exchange line: lane `l` sums 8 columns
    of token `l // 8` and pushes them, then lane 0 the line's flag."""
    lpg = ex_.lines_per_group(rows)
    for line in range(g.spread, ex_.GROUPS * lpg, BLOCKS * WAVES):
        grp, blk = line // lpg, line % lpg
        token = blk * ex_.TOKENS + g.lane // 8
        tk = (token < rows).select(token, i32(rows - 1))
        col = grp * ex_.COLS + (g.lane % 8) * 8
        ws = [
            ld(b.route, rows * SLOTS + tk * SLOTS + j, fx.Float32, cm=CM_DEV)
            for j in range(SLOTS)
        ]
        halves = [
            ld(b.split, tk * SLOTS + j, fx.Int32, cm=CM_DEV) == xch.tag
            for j in range(SLOTS)
        ]
        ps, ps2 = [], []
        for j in range_constexpr(SLOTS):
            at = (tk * SLOTS + j) * HIDDEN + col
            ps.append(
                [ld(b.mpart, at + 4 * h, fx.Float32, 4, cm=CM_DEV) for h in range(2)]
            )
            ps2.append(
                [
                    ld(
                        b.mpart2,
                        at + 4 * h,
                        fx.Float32,
                        4,
                        cm=CM_DEV,
                        nbytes=rows * SLOTS * HIDDEN * 4,
                        mask=halves[j],
                    )
                    for h in range(2)
                ]
            )
        sums = [fx.Float32(0.0)] * 8
        for j in range_constexpr(SLOTS):
            sums = [
                sums[e] + ws[j] * (ps[j][e // 4][e % 4] + ps2[j][e // 4][e % 4])
                for e in range(8)
            ]
        xch.put(grp, token, g.lane % 8, sums)
        drain_stores()
        if g.lane == 0:
            xch.flag(grp, blk)


# --------------------------------------------------------------------------
# the MLP half
# --------------------------------------------------------------------------


@traced
def mlp(g, s: Shape, gemm, ml, x, w: Weights, b: Bufs, xch, end_phase):
    """The MoE of `x` ([rows, HIDDEN] bf16, its MXFP4 rows in `b.x4` /
    `b.x4s`) into exchange `xch`: every phase of `PHASES`,
    `end_phase(name)` after each but the last. `ml`: the `lds_struct`
    view."""
    lds = Lds(ml, lds_words(s))
    router(g, gemm, w.w_gate, x, b)
    end_phase(PHASES[0])
    route(g, b.part, b.route, s.rows)
    end_phase(PHASES[1])
    build_schedule(g.tid, s, lds, lambda a: ld(b.route, a, fx.Int32, cm=CM_DEV))
    experts(g, s, lds, w, b, xch.tag)
    end_phase(PHASES[2])
    major_combine(g, xch, b, s.rows)
