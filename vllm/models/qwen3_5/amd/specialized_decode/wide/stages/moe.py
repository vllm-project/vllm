# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse MoE block (``Qwen3NextSparseMoeBlock`` with the shared expert
fused as routed expert ``EXPERTS``, AITER's W4A4 MXFP4 experts) as grid
phases, cut by weight bytes at every width:

- ``router``: ``logits = x @ [W_gate; w_shared_gate].T`` (bf16, vLLM's
  fused gate) in K slices of 64-row groups. Up to ``LOCAL_ROUTE_MAX`` rows a
  group's last arriving unit sums its slices and rounds them to bf16 (the
  stock GEMM's output) into ``logits``; wider, the route phase sums them.
- route: the top-k of a row's logits (renormalized softmax weights) and the
  shared expert's ``sigmoid`` weight in slot ``TOPK``. Up to
  ``LOCAL_ROUTE_MAX`` rows every CTA routes every row itself, at the start
  of the gate/up phase; wider, a phase of a wave a row into ``route``.
- schedule: every CTA builds the same (expert, <= 16 tokens) unit list in
  LDS (a counting sort of the ``rows * SLOTS`` assignments), kept there for
  the down phase.
- ``up``: pieces (unit, 32 intermediate rows: one scale group), at the
  narrowest widths also cut along K (``up_splits``), over the grid: wave
  ``w`` a 16-row tile of gate (w < 2) or up of the piece against the unit's
  tokens' MXFP4 rows of ``x`` (``x4``), the K steps streamed with the next
  piece's first steps in flight under the last ones; ``silu(gate) * up``
  requantized to MXFP4 (the stock MoE's inter-stage quantization) to the
  assignments' rows of ``act4``.
- ``down``: CTA ``c`` owns output rows ``32 c ..`` (two 16-row tiles) of
  every unit's down projection: the (row, slot) results into LDS, their
  slot-order sum with the routing weights to ``y`` (f32); the last of a
  line group's two owners pushes its exchange lines.

The experts are vLLM's tensors in place: ``shuffle_weight(16, 16)`` codes and
``e8m0_shuffle`` scales; gate and up each padded from ``RI`` to ``rip`` rows,
the down projection's K likewise.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    fold_arrive,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    drain_stores,
    exp,
    ld,
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
from vllm.models.qwen3_5.amd.specialized_decode.wide.layout import (
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
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import mx, tiled

ROUTER_SPLITS = 28  # 9 row groups x 28 = 252 units
LOCAL_ROUTE_MAX = 16  # widest step whose CTAs route every row themselves
TILE = 16
NEG_INF = float("-inf")
XGROUPS = HIDDEN // mx.GROUP  # scale groups an input row
AGROUPS = RI // mx.GROUP  # scale groups an intermediate row
PIECES = AGROUPS  # gate/up pieces a unit: 32 intermediate rows each
UP_STEPS = HIDDEN // 128  # K steps of gate/up
UP_DEPTH = 12  # K steps of a wave's loads in flight
OWN_ROWS = HIDDEN // BLOCKS  # output rows a CTA owns in the down projection
OWN_TILES = OWN_ROWS // TILE
LINE_OWNERS = ex_.COLS // OWN_ROWS  # owners a line group
DOWN_AHEAD = 4  # units a down wave loads ahead
PER = EXPERTS // 64  # a lane's experts in ``route``
MIN_KEY = -(1 << 31)
CONTRIB_BYTES = 96 * 1024  # LDS for the down results of a pass
# a descending sorting network of 8 (19 compare-exchanges)
_SORT8 = (
    (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1),
    (2, 3), (4, 5), (6, 7), (2, 4), (3, 5), (1, 4), (3, 6), (1, 2), (3, 4), (5, 6),
)  # fmt: skip
assert OWN_ROWS * BLOCKS == HIDDEN and ex_.COLS % OWN_ROWS == 0


def i32(v):
    return fx.Int32(v)


def local_route(rows: int) -> bool:
    return rows <= LOCAL_ROUTE_MAX


def up_splits(rows: int) -> int:
    """K slices of a gate/up piece: a step of a few rows has ~11 units a row,
    8 pieces a unit, too few for 256 CUs when each streams 256 KiB."""
    if rows <= 1:
        return 4
    if rows <= 2:
        return 2
    return 1


def router_plan():
    return tiled.SplitK(E_ALL, HIDDEN, ROUTER_SPLITS)


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
        self.down_steps = -(-RI // 128)  # K steps of the down projection
        self.ks = up_splits(rows)
        self.max_pieces = self.max_units * PIECES
        # down tiles a pass: both while their results fit the LDS budget
        both = self.assignments * TILE * OWN_TILES * 4 <= CONTRIB_BYTES
        self.pass_tiles = OWN_TILES if both else 1


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
    """The MoE's scratch (byte addresses): ``part`` the router partials,
    ``logits`` ([rows, padded] f32 of bf16 values), ``route`` ([rows * SLOTS]
    ids, then their f32 weights), ``x4`` / ``x4s`` the MXFP4 rows of ``x``,
    ``act4`` / ``act4s`` an assignment's MXFP4 intermediate row and scale
    words, ``upart`` the K-split pieces' partials, ``y`` ([rows, HIDDEN]
    f32)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, fx.Int64(v))


def scratch_words(s: Shape) -> dict[str, int]:
    """i32 words of each MoE scratch buffer at ``s``'s width."""
    plan = router_plan()
    return {
        "logits": s.rows * plan.padded,
        "route": 2 * s.assignments,
        "act4": s.assignments * (RI // 8),
        "act4s": s.assignments * AGROUPS,
        "upart": (s.max_pieces * s.ks * TILE * 64) if s.ks > 1 else 0,
        "y": s.rows * HIDDEN,
    }


def phase_names(rows: int) -> list[str]:
    """The MoE's phases, in order; ``mlp`` ends all but the last, whose end
    is the caller's (the MLP exchange's readers follow it)."""
    if local_route(rows):
        return ["router", "route+experts_up", "experts_down"]
    return ["router", "route", "experts_up", "experts_down"]


def _lds_struct(name: str, words: dict[str, int]):
    """An ``fx.struct`` of i32 arrays (built here: this module's annotations
    are strings)."""
    fields = {k: fx.Array[fx.Int32, max(n, 4), 16] for k, n in words.items()}
    return fx.struct(type(name, (), {"__annotations__": fields}))


def lds_words(s: Shape) -> dict[str, int]:
    return dict(
        ids=s.assignments if local_route(s.rows) else 0,
        wts=s.assignments if local_route(s.rows) else 0,
        index=s.index_words,
        units=3 * s.max_units,
        scan=8,
        panel=TILE * 64,
        contrib=s.assignments * TILE * s.pass_tiles,
    )


def lds_struct(s: Shape):
    """The MoE's LDS struct for ``s``'s width."""
    return _lds_struct(f"MoeLds{s.rows}", lds_words(s))


class Lds(TraceInvariant):
    """The MoE's LDS regions (i32 pointers)."""

    def __init__(self, ml):
        for k in ("ids", "wts", "index", "units", "scan", "panel", "contrib"):
            setattr(self, k, getattr(ml, k).ptr)


# --------------------------------------------------------------------------
# router and routing
# --------------------------------------------------------------------------


def router_weights(w_gate):
    return tiled.Bf16Weights(w_gate, E_ALL, HIDDEN)


def router(g, gemm, w_gate, x, b, rows: int, fold0: int, pending=None):
    """The router logits' K-slice partials to ``b.part``; up to
    ``LOCAL_ROUTE_MAX`` rows each group's last arriver sums them into
    ``b.logits`` (rounded to bf16). ``pending``: the CTA's first unit's
    weights (``tiled.issue_unit`` / ``tiled.unit_preload``)."""
    plan = router_plan()
    if const_expr(local_route(rows)):

        def finish(group, first, lane, acc):
            row = first + lane // TILE * 4
            for t in range_constexpr(gemm.tiles):
                token = i32(t * TILE) + lane % TILE
                if token < rows:
                    vals = [bf16_round(acc[t][e]) for e in range(4)]
                    st(
                        b.logits,
                        token * plan.padded + row,
                        fx.Vector.from_elements(vals, fx.Float32),
                        CM_DEV,
                    )

        epilogue, after = tiled.fold(g, gemm, plan, b.part, fold0, finish)
    else:
        epilogue, after = tiled.partial_epilogue(gemm, plan, b.part), None
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        tiled.Source(x, HIDDEN),
        router_weights(w_gate),
        plan,
        epilogue,
        after,
        pending=pending,
    )


@traced
def route(g, part, route_buf, rows: int):
    """A wave a row: the top-k of the summed router partials (rounded to bf16)
    and their renormalized softmax weights; slot TOPK is the shared expert
    with its sigmoid weight. ``route_buf``: [rows * SLOTS] ids, then [rows *
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


# --------------------------------------------------------------------------
# the expert schedule
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
def build_schedule(tid, s: Shape, lds, load_id):
    """Every CTA the same (expert, <= 16 token) units from the assignments'
    expert ids (``load_id(a)``): per-expert token bitmasks, counts, starts, a
    flat unit list ``lds.units`` ([n][3]: expert, start, count) and the
    sorted assignments; bitmask ranks give every CTA the same order. The
    unit count is ``index[2 E_ALL]``."""
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
                lds_st(
                    lds.units + u * 3 + 2, fx.min(counts[r] - blk * TILE, i32(TILE))
                )
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
    """(expert, start, count) of unit ``u`` (CTA-uniform)."""
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
# gate / up pieces
# --------------------------------------------------------------------------


class _Piece(TraceInvariant):
    """Piece ``p``'s CTA-uniform coordinates and this lane's operands' bases:
    unit ``u``, intermediate block ``blk`` (32 rows), K slice ``ks``; the
    wave's weight tile and scale pair; the lane's token row of ``x``. A
    piece past ``total`` is clamped to piece 0 and issues no request."""

    def __init__(self, g, s, lds, w, p, total):
        self.live = p < total
        pp = self.live.select(p, i32(0))
        per_unit = PIECES * s.ks
        self.u, rest = pp // per_unit, pp % per_unit
        self.blk, self.ks = rest // s.ks, rest % s.ks
        expert, self.start, self.count = _unit(lds, self.u)
        self.ub = w.w13 + fx.Int64(expert) * fx.Int64(s.w13_bytes)
        self.us = w.w13s + fx.Int64(expert) * fx.Int64(s.w13_scale_bytes)
        proj, half = g.wave // 2, g.wave % 2
        rip_tiles = s.rip // TILE
        self.tile = proj * rip_tiles + self.blk * 2 + half
        self.pair = proj * (rip_tiles // 2) + self.blk
        self.half = half
        token = g.lane % TILE
        self.xlive = self.live & (token < self.count)
        self.xrow = _assignment(lds, s, self.start, self.count, token) // SLOTS


def _step_loads(g, s, pc, b, k: int):
    """Step ``k`` (of the piece's K slice) of this wave: its weight tile's 16
    bytes, the lane's token's x codes and scale code, and on even steps the
    weight scale word of the step pair."""
    steps = UP_STEPS // s.ks
    kk = pc.ks * i32(steps) + i32(k)
    group = g.lane // TILE
    weights = ld(
        pc.ub,
        (pc.tile * (TILE * s.w13_row_bytes) + kk * 1024 + 16 * g.lane) // 4,
        fx.Int32,
        4,
        cm=CM_NT,
        nbytes=s.w13_bytes,
        mask=pc.live,
    )
    q = kk * 4 + group
    xb = ld(
        b.x4,
        pc.xrow * (HIDDEN // 8) + q * 4,
        fx.Int32,
        4,
        cm=CM_DEV,
        nbytes=s.rows * HIDDEN // 2,
        mask=pc.xlive,
    )
    xc = ld(
        b.x4s,
        pc.xrow * XGROUPS + q,
        fx.Int32,
        cm=CM_DEV,
        nbytes=s.rows * XGROUPS * 4,
        mask=pc.xlive,
    )
    out = [weights, xb, xc]
    if const_expr(k % 2 == 0):
        scale_tiles = s.w13_scale_cols // 8
        out.append(
            ld(
                pc.us,
                (pc.pair * scale_tiles + kk // 2) * 64 + g.lane,
                fx.Int32,
                cm=CM_NT,
                nbytes=s.w13_scale_bytes,
                mask=pc.live,
            )
        )
    return out


@traced
def up(g, s, lds, w, b, fold0: int):
    """The gate/up pieces ``bid, bid + BLOCKS, ..`` of the schedule in LDS, a
    wave's K steps streamed ``UP_DEPTH`` ahead across pieces."""
    steps = UP_STEPS // s.ks
    depth = min(UP_DEPTH, steps)
    total = unit_count(lds) * i32(PIECES * s.ks)
    first = _Piece(g, s, lds, w, g.bid, total)
    ring = [_step_loads(g, s, first, b, k) for k in range(depth)]
    for p in range(g.bid, total, BLOCKS):
        pc = _Piece(g, s, lds, w, p, total)
        nx = _Piece(g, s, lds, w, p + BLOCKS, total)
        cur = list(ring)
        ahead = []
        acc = zeros(4)
        code = i32(0)
        for k in range_constexpr(steps):
            if const_expr(k + depth < steps):
                cur.append(_step_loads(g, s, pc, b, k + depth))
            else:
                ahead.append(_step_loads(g, s, nx, b, k + depth - steps))
            sched_barrier()
            loaded = cur[k]
            if const_expr(k % 2 == 0):
                code = loaded[3]
            sa = (code >> (pc.half * 8)) & 0x00FF00FF
            acc = mfma_f4(
                loaded[0], loaded[1], acc, sa, loaded[2] & 0xFF, opsel_a=2 * (k % 2)
            )
        ring = ahead
        _up_epilogue(g, s, lds, b, pc, acc, fold0)


@traced
def _up_epilogue(g, s, lds, b, pc, acc, fold0: int):
    """A piece's wave results to the f32 panel ([16 tokens][gate 32 | up 32]),
    or for a K slice to its partials, the piece's last slice summing them in
    slice order; then wave 0 requantizes ``silu(gate) * up``: lane (token
    ``l % 16``, quarter ``l / 16``) 8 of the token's 32 products, the scale
    from the four quarters' amax."""
    lane, wave = g.lane, g.wave
    row, group = lane % TILE, lane // TILE
    proj, half = wave // 2, wave % 2
    col = proj * 32 + half * 16 + group * 4
    if const_expr(s.ks == 1):
        lds_st(lds.panel + row * 64 + col, acc.bitcast(fx.Int32))
        fx.barrier()
        last = True
    else:
        piece = pc.u * PIECES + pc.blk
        st(b.upart, ((piece * s.ks + pc.ks) * TILE + row) * 64 + col, acc, CM_DEV)
        last = fold_arrive(g, i32(fold0) + piece, s.ks)
    if last:
        if wave == 0:
            if const_expr(s.ks == 1):
                at = lds.panel + row * 64 + group * 8
                gv = [lds_ld(at + 4 * h, 4).bitcast(fx.Float32) for h in range(2)]
                uv = [lds_ld(at + 32 + 4 * h, 4).bitcast(fx.Float32) for h in range(2)]
                gs = [gv[e // 4][e % 4] for e in range(8)]
                us = [uv[e // 4][e % 4] for e in range(8)]
            else:
                piece = pc.u * PIECES + pc.blk
                gs = [fx.Float32(0.0)] * 8
                us = [fx.Float32(0.0)] * 8
                for k in range_constexpr(s.ks):
                    at = ((piece * s.ks + k) * TILE + row) * 64 + group * 8
                    gk = [ld(b.upart, at + 4 * h, fx.Float32, 4, cm=CM_DEV) for h in range(2)]
                    uk = [
                        ld(b.upart, at + 32 + 4 * h, fx.Float32, 4, cm=CM_DEV)
                        for h in range(2)
                    ]
                    gs = [gs[e] + gk[e // 4][e % 4] for e in range(8)]
                    us = [us[e] + uk[e // 4][e % 4] for e in range(8)]
            vals = [silu(gs[e]) * us[e] for e in range(8)]
            code = mx.scale_code(butterfly(mx.amax(vals), (32, 16), fx.max))
            word = mx.quant_f32_word(vals, mx.scale_value(code))
            if row < pc.count:
                a = lds_ld(lds.index + s.token_base + pc.start + row)
                st(b.act4, a * (RI // 8) + pc.blk * 4 + group, word, CM_DEV)
                if group == 0:
                    st(b.act4s, a * AGROUPS + pc.blk, code, CM_DEV)
    fx.barrier()


# --------------------------------------------------------------------------
# the down projection by output owners
# --------------------------------------------------------------------------


def _down_loads(g, s, lds, w, b, i, tiles):
    """Wave unit ``i`` (unit ``w + 4 i``): its down weights of the owned
    ``tiles`` (every K step), the tiles' scale word, the lane's token's
    activation codes and scale codes, its assignment and the token count
    (0 past the unit count: no request)."""
    n = unit_count(lds)
    u = g.wave + WAVES * i
    live = u < n
    uu = live.select(u, i32(0))
    expert = uniform(lds_ld(lds.units + uu * 3))
    start = lds_ld(lds.units + uu * 3 + 1)
    count = live.select(lds_ld(lds.units + uu * 3 + 2), i32(0))
    token = fx.min(g.lane % TILE, fx.max(count - 1, i32(0)))
    a = lds_ld(lds.index + s.token_base + start + token)
    db = w.w2 + fx.Int64(expert) * fx.Int64(s.w2_bytes)
    ds = w.w2s + fx.Int64(expert) * fx.Int64(s.w2_scale_bytes)
    group = g.lane // TILE
    weights = [
        [
            ld(
                db,
                (tile * (TILE * s.w2_row_bytes) + 1024 * k + 16 * g.lane) // 4,
                fx.Int32,
                4,
                cm=CM_NT,
                nbytes=s.w2_bytes,
                mask=live & (4 * k + group < AGROUPS),
            )
            for k in range(s.down_steps)
        ]
        for tile in tiles
    ]
    scale_tiles = s.w2_scale_cols // 8
    code = ld(
        ds,
        ((tiles[0] // 2) * scale_tiles) * 64 + g.lane,
        fx.Int32,
        cm=CM_NT,
        nbytes=s.w2_scale_bytes,
        mask=live,
    )
    xb, xc = [], []
    for k in range_constexpr(s.down_steps):
        q = 4 * k + group
        on = live & (q < AGROUPS)
        qq = on.select(q, i32(0))
        xb.append(
            ld(
                b.act4,
                a * (RI // 8) + qq * 4,
                fx.Int32,
                4,
                cm=CM_DEV,
                nbytes=s.assignments * RI // 2,
                mask=on,
            )
        )
        xc.append(
            on.select(
                ld(b.act4s, a * AGROUPS + qq, fx.Int32, cm=CM_DEV) & 0xFF, i32(127)
            )
        )
    return weights, code, xb, xc, a, count


@traced
def down(g, s, lds, w, b, xch, local: bool):
    """CTA ``c``'s output rows ``OWN_ROWS c ..`` of every unit, in passes of
    ``s.pass_tiles`` tiles: wave ``w`` takes units ``w, w + 4, ..``, a K
    chain each over its tokens' activation rows, ``DOWN_AHEAD`` units'
    operands in flight; a token's results to ``lds.contrib[(row, slot)]``;
    then their slot-order sum with the routing weights (``lds.wts`` with
    ``local`` routing, else ``b.route``'s) to ``y``. The line group's last
    owner pushes its lines to ``xch``."""
    rows = s.rows
    n = unit_count(lds)
    per_wave = (n + i32(WAVES - 1)) // i32(WAVES)
    for ps in range_constexpr(OWN_TILES // s.pass_tiles):
        tiles = [
            g.bid * OWN_TILES + i32(ps * s.pass_tiles + t) for t in range(s.pass_tiles)
        ]
        odd = [(ps * s.pass_tiles + t) % 2 for t in range(s.pass_tiles)]
        ring = [_down_loads(g, s, lds, w, b, i32(i), tiles) for i in range(DOWN_AHEAD)]
        for i in range(i32(0), per_wave, i32(1)):
            cur = list(ring)
            cur.append(_down_loads(g, s, lds, w, b, i + DOWN_AHEAD, tiles))
            sched_barrier()
            weights, code, xb, xc, a, count = cur[0]
            for t in range_constexpr(s.pass_tiles):
                sa = (code >> (odd[t] * 8)) & 0x00FF00FF
                acc = zeros(4)
                for k in range_constexpr(s.down_steps):
                    acc = mfma_f4(
                        weights[t][k], xb[k], acc, sa, xc[k], opsel_a=2 * (k % 2)
                    )
                if g.lane % TILE < count:
                    lds_st(
                        lds.contrib + a * (TILE * s.pass_tiles) + t * TILE
                        + (g.lane // TILE) * 4,
                        acc.bitcast(fx.Int32),
                    )
            ring = cur[1:]
        fx.barrier()
        cols = TILE * s.pass_tiles
        for r in range_constexpr(-(-(rows * cols) // THREADS)):
            at = g.tid + i32(r * THREADS)
            if at < rows * cols:
                row, c = at // cols, at % cols
                y = fx.Float32(0.0)
                for j in range_constexpr(SLOTS):
                    if const_expr(local):
                        wj = lds_ld(lds.wts + row * SLOTS + j).bitcast(fx.Float32)
                    else:
                        wj = ld(
                            b.route, rows * SLOTS + row * SLOTS + j, fx.Float32, cm=CM_DEV
                        )
                    cj = lds_ld(lds.contrib + (row * SLOTS + j) * cols + c).bitcast(
                        fx.Float32
                    )
                    y = y + wj * cj
                col = g.bid * OWN_ROWS + ps * cols + c
                st(b.y, row * HIDDEN + col, y, CM_DEV)
        fx.barrier()
    grp = g.bid // LINE_OWNERS
    if fold_arrive(g, grp, LINE_OWNERS):
        push_group(g, xch, b.y, grp, rows)


@traced
def push_group(g, xch, y, grp, rows: int):
    """Line group ``grp``'s exchange lines from ``y`` ([rows, HIDDEN] f32), a
    wave a line (lane ``l``: token ``l // 8``, columns ``8 (l % 8) ..``),
    drained, then each line's flag from lane 0."""
    lpg = ex_.lines_per_group(rows)
    for blk in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + i32(blk * WAVES)
        if line < lpg:
            token = line * ex_.TOKENS + g.lane // 8
            tk = (token < rows).select(token, i32(rows - 1))
            col = grp * ex_.COLS + (g.lane % 8) * 8
            v = [
                ld(y, tk * HIDDEN + col + 4 * h, fx.Float32, 4, cm=CM_DEV)
                for h in range(2)
            ]
            xch.put(grp, token, g.lane % 8, [v[e // 4][e % 4] for e in range(8)])
    drain_stores()
    for blk in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + i32(blk * WAVES)
        if (line < lpg) & (g.lane == 0):
            xch.flag(grp, line)


# --------------------------------------------------------------------------
# the MLP half
# --------------------------------------------------------------------------


def mlp(g, s: Shape, gemm, ml, x, w: Weights, b: Bufs, xch, end_phase, folds, pre=None):
    """The MoE of ``x`` ([rows, HIDDEN] bf16, its MXFP4 rows in ``b.x4`` /
    ``b.x4s``) into exchange ``xch``: every phase of ``phase_names``,
    ``end_phase(name)`` after each but the last. ``ml``: the
    ``lds_struct`` view; ``folds``: two fold counter bases (router, pieces);
    ``pre``: the router's first weights in flight, or None."""
    names = phase_names(s.rows)
    lds = Lds(ml)
    router(g, gemm, w.w_gate, x, b, s.rows, folds[0], pre)
    end_phase(names[0])
    local = local_route(s.rows)
    if const_expr(local):
        assert s.assignments <= THREADS
        rl = _route_loads(g, b.logits, s.rows)
        route_rows(g, rl, lds.ids, lds.wts, s.rows)
        fx.barrier()
        if (g.bid == 0) & (g.tid < s.assignments):
            st(b.route, g.tid, lds_ld(lds.ids + g.tid), CM_DEV)
            st(b.route, s.assignments + g.tid, lds_ld(lds.wts + g.tid), CM_DEV)
        build_schedule(g.tid, s, lds, lambda a: lds_ld(lds.ids + a))
    else:
        route(g, b.part, b.route, s.rows)
        end_phase(names[1])
        build_schedule(
            g.tid, s, lds, lambda a: ld(b.route, a, fx.Int32, cm=CM_DEV)
        )
    up(g, s, lds, w, b, folds[1])
    end_phase(names[-2])
    down(g, s, lds, w, b, xch, local)
