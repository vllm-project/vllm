# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tensor-parallel all-reduce of a block's output `y_r [rows, HIDDEN]`
over peer mailboxes, in lines: every rank ends with the same bf16 sum of the
ranks' partials, added in rank order.

Lines: a partial is stored as lines ordered by (column group, token block):
group `g` is columns `64 g .. 64 g + 63` (one row group of the producing
GEMM, whose last arriving unit pushes it), a line eight tokens of one group,
256 dwords of bf16. Lane `l` of the wave that writes a line holds token
`l // 8` of the block, columns `8 (l % 8) ..`. A step whose rows are not
a multiple of eight ends each group with a partial line: only its rows'
tokens are stored and read, at the same addresses as in a whole line.

An exchange's `mode`:

- `sum` (one-shot, up to `ONE_SHOT_MAX` rows): every rank pushes every
  group to every rank; a reader adds the ranks' copies in rank order. Every
  link carries the whole tensor, one hop.
- `scatter` then `gather` (two-shot, wider steps): group `g`'s owner
  is rank `g // OWNED`; every rank pushes `g` to its owner only, the
  owner adds the copies in rank order (`reduce_lines`) and pushes the sum,
  rounded to bf16, to every rank. A link carries an eighth of the tensor
  each way, two hops.

After its payload a wave drains its stores and sets the line's flag to the
use's tag in each destination's mailbox; a reader polls the flags of the lines
it needs and gates its payload loads on them.

A rank's mailbox (uncached, peer-mapped): a region per (exchange kind of a
layer, mode, layer parity), a row per source rank in each, sized for the
widest step the mode serves: payload, then one flag a line at a fixed offset,
so a flag word only ever holds a tag. A tag carries the step's epoch, so a
flag of an earlier step, or of another width's layout, never passes. A rank
can be at most one layer ahead of a peer, so a region is never rewritten
before every reader of its previous use is done.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.wide.abi import SITE_PEER, WAIT_TICKS
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    failed,
    give_up,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.ops import (
    CM_SYS,
    drain_stores,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    memrealtime,
    pack_bf16x8,
    st,
    traced,
    unpack_bf16x8,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.layout import (
    HIDDEN,
    MAX_TOKENS,
    TP,
    WAVES,
)

COLS = 64  # columns of a line group
TOKENS = 8  # tokens of a line
LINE_DWORDS = 256  # a bf16 line
GROUPS = HIDDEN // COLS  # 128
OWNED = GROUPS // TP  # groups a rank owns in a two-shot exchange: 16
ONE_SHOT_MAX = 16  # widest step whose exchanges are one-shot
MODES = ("sum", "scatter", "gather")
KINDS = 2  # a layer's exchanges: the attention / GDN output, the MLP's
PARITIES = 2


def lines_per_group(rows: int) -> int:
    return -(-rows // TOKENS)


def one_shot(rows: int) -> bool:
    return rows <= ONE_SHOT_MAX


def _area_groups(mode: str) -> int:
    """Groups of a source row: every group (sum), the owned ones (two-shot)."""
    return GROUPS if mode == "sum" else OWNED


def _area_rows(mode: str) -> int:
    return ONE_SHOT_MAX if mode == "sum" else MAX_TOKENS


def payload_words(mode: str) -> int:
    return _area_groups(mode) * lines_per_group(_area_rows(mode)) * LINE_DWORDS


def flag_words(mode: str) -> int:
    return _area_groups(mode) * lines_per_group(_area_rows(mode))


def row_words(mode: str) -> int:
    """A source rank's row of a region: payload, flags, padded to 64 dwords."""
    return (payload_words(mode) + flag_words(mode) + 63) // 64 * 64


def _region_words(mode: str, tp: int) -> int:
    return tp * row_words(mode)


def region_base(kind: int, mode: str, parity, tp: int = TP):
    """Word offset of region (kind, mode, parity) in a mailbox; `parity` may
    be traced."""
    base = 0
    for k in range(KINDS):
        for m in MODES:
            if k == kind and m == mode:
                return fx.Int32(base) + fx.Int32(parity) * fx.Int32(
                    _region_words(m, tp)
                )
            base += PARITIES * _region_words(m, tp)
    raise ValueError((kind, mode))


def mailbox_words(tp: int = TP) -> int:
    return KINDS * PARITIES * sum(_region_words(m, tp) for m in MODES)


def mailbox_bytes(tp: int = TP) -> int:
    return mailbox_words(tp) * 4


def owner(group):
    return fx.Int32(group) // fx.Int32(OWNED)


class Exchange(TraceInvariant):
    """One use of a mailbox region: exchange `kind` of the layer in `mode`,
    this rank `rank` of `tp`, layer parity `parity`, tagged `tag`.
    `bases`: every rank's mailbox base (wave-uniform Int64, rank order),
    `own`: this rank's."""

    def __init__(
        self, bases, own, rank, tp: int, kind: int, mode: str, parity, tag, rows
    ):
        if mode not in MODES:
            raise ValueError(f"exchange mode {mode!r}: one of {MODES}")
        if rows > _area_rows(mode):
            raise ValueError(f"{rows} rows past the {mode} region's")
        self.bases, self.own, self.rank, self.tp = (
            bases,
            fx.Int64(own),
            fx.Int32(rank),
            tp,
        )
        self.kind, self.mode, self.rows = kind, mode, rows
        self.region = region_base(kind, mode, parity, tp)
        self.tag = fx.Int32(tag)
        self.lpg = lines_per_group(rows)
        self.partial = rows % TOKENS != 0
        self.row_words = row_words(mode)
        self.flags_at = payload_words(mode)

    def _row(self, src):
        return self.region + fx.Int32(src) * fx.Int32(self.row_words)

    def line(self, group, blk):
        """The (group, token block) line's index in a source row."""
        local = group if self.mode == "sum" else fx.Int32(group) % fx.Int32(OWNED)
        return fx.Int32(local) * fx.Int32(self.lpg) + blk

    def flag_index(self, src, group, blk):
        return self._row(src) + fx.Int32(self.flags_at) + self.line(group, blk)

    def _word(self, group, token):
        blk = fx.Int32(token) // fx.Int32(TOKENS)
        return self.line(group, blk) * fx.Int32(LINE_DWORDS) + (
            fx.Int32(token) % fx.Int32(TOKENS)
        ) * fx.Int32(32)

    def dests(self, group):
        """`(base, live)` of every rank this rank pushes `group` to (live
        None: always)."""
        if const_expr(self.mode == "scatter"):
            own = owner(group)
            return [(b, own == fx.Int32(r)) for r, b in enumerate(self.bases)]
        return [(b, None) for b in self.bases]

    def sources(self, group):
        """The ranks a reader takes `group` from, in rank order."""
        if const_expr(self.mode == "gather"):
            return [owner(group)]
        return [fx.Int32(r) for r in range(self.tp)]

    def put_words(self, group, token, seg, words):
        """The lane's four packed bf16 dwords of `token`, columns `64 group +
        8 seg ..`, into its destinations' rows of this rank; a token past the
        step's rows stores nothing."""
        at = self._row(self.rank) + self._word(group, token) + fx.Int32(seg) * 4
        live = fx.Int32(token) < fx.Int32(self.rows) if self.partial else None
        for base, to in self.dests(group):
            ok = live if to is None else (to if live is None else to & live)
            if const_expr(ok is None):
                st(base, at, words, cm=CM_SYS)
            else:
                _st_if(base, at, words, ok)

    def put(self, group, token, seg, vals):
        self.put_words(group, token, seg, pack_bf16x8(vals))

    def flag(self, group, blk):
        """Lane 0's flag of line `blk` of `group`, after the drain."""
        at = self._row(self.rank) + fx.Int32(self.flags_at) + self.line(group, blk)
        for base, to in self.dests(group):
            if const_expr(to is None):
                st(base, at, self.tag, cm=CM_SYS)
            else:
                _st_if(base, at, self.tag, to)

    def get_words(self, src, group, token, seg, gate):
        """`src`'s four dwords of `token` (segment `seg` of its group's
        line) after the poll whose difference `gate` (zero) covers it."""
        at = self._row(src) + gate + self._word(group, token) + fx.Int32(seg) * 4
        return ld_i32x4(self.own, at, cm=CM_SYS)

    def get(self, src, group, token, seg, gate):
        return unpack_bf16x8(self.get_words(src, group, token, seg, gate))


@traced
def _st_if(base, at, value, ok):
    if ok:
        st(base, at, value, cm=CM_SYS)


@traced
def wait_lines(g, ex, lines):
    """Poll, from every lane, the flags of `lines` (a list of (group, token
    block) pairs, the lane's own) of every source `ex.sources(group)` until
    all carry the tag, bounded. Returns the differences from the tag (zero
    once the poll ends; 0 or 1, so a timed-out poll's loads stay in bounds),
    one a (line, source) in that order, to add to the payload loads."""
    idx = [
        ex.flag_index(src, group, blk)
        for group, blk in lines
        for src in ex.sources(group)
    ]
    t0 = memrealtime()
    seen = [ld(ex.own, i, fx.Int32, cm=CM_SYS) for i in idx]
    pend = seen[0] != ex.tag
    for v in seen[1:]:
        pend = pend | (v != ex.tag)
    why = failed(g).select(fx.Int32(2), fx.Int32(0))
    while pend & (why == 0):
        seen = [ld(ex.own, i, fx.Int32, cm=CM_SYS) for i in idx]
        pend = seen[0] != ex.tag
        for v in seen[1:]:
            pend = pend | (v != ex.tag)
        late = memrealtime() - t0 > fx.Int64(WAIT_TICKS)
        why = late.select(fx.Int32(1), fx.Int32(0))
    if why == 1:
        give_up(g, SITE_PEER, seen[0], ex.tag)
    return [(v != ex.tag).select(fx.Int32(1), fx.Int32(0)) for v in seen]


class GroupTile(TraceInvariant):
    """An LDS `[rows, 64]` bf16 tile (i32 words, a 16-byte pad a row) that a
    producer gathers one column group of every token in before it pushes the
    group's lines (`push_tile`)."""

    STRIDE = COLS // 2 + 4

    def __init__(self, ptr, rows: int):
        self.ptr, self.rows = ptr, rows

    @staticmethod
    def words(rows: int) -> int:
        return rows * GroupTile.STRIDE

    def put_vals(self, token, col, vals4):
        """Four f32 of `token` at columns `col ..` of the group, rounded to
        bf16."""
        w = fx.Vector.from_elements(list(vals4), fx.Float32)
        lds_st(
            self.ptr + token * self.STRIDE + col // 2,
            w.to(fx.BFloat16).bitcast(fx.Int32),
        )

    def put_acc(self, col0, lane, acc, tiles: int, scale=None):
        """A wave's GEMM accumulators (16 columns from `col0` within the
        group, the epilogue layout: lane `l` holds columns `col0 + 4 (l //
        16) ..` of token `16 t + l % 16`), each times `scale(token, col)`
        (4 f32) when given, rounded to bf16."""
        for t in range(tiles):
            token = fx.Int32(t * 16) + lane % 16
            tk = (token < self.rows).select(token, fx.Int32(self.rows - 1))
            col = col0 + lane // 16 * 4
            vals = [acc[t][e] for e in range(4)]
            if scale is not None:
                s = scale(tk, col)
                vals = [vals[e] * s[e] for e in range(4)]
            self.put_vals(tk, col, vals)


@traced
def push_tile(g, xch, tile, group):
    """The tile (column group `group` of every token) as exchange lines to
    the group's destinations: wave `w` pushes lines `w, w + 4, ..`, the
    payload first, one drain, then each line's flag from lane 0."""
    lpg = lines_per_group(tile.rows)
    seg = g.lane % 8
    for k in range_constexpr(-(-lpg // WAVES)):
        blk = g.wave + fx.Int32(k * WAVES)
        if blk < lpg:
            token = blk * TOKENS + g.lane // 8
            tk = token
            if const_expr(xch.partial):
                tk = fx.min(token, fx.Int32(tile.rows - 1))  # a partial line's tail
            words = lds_ld(tile.ptr + tk * GroupTile.STRIDE + seg * 4, 4)
            xch.put_words(group, token, seg, words)
    drain_stores()
    for k in range_constexpr(-(-lpg // WAVES)):
        blk = g.wave + fx.Int32(k * WAVES)
        if (blk < lpg) & (g.lane == 0):
            xch.flag(group, blk)


@traced
def push_line(g, xch, group, blk, vals):
    """A wave's line `blk` of `group` (lane `l`: token `8 blk + l //
    8`, columns `64 group + 8 (l % 8) ..`, 8 f32 rounded to bf16), drained,
    then flagged from lane 0."""
    token = blk * fx.Int32(TOKENS) + g.lane // 8
    xch.put(group, token, g.lane % 8, vals)
    drain_stores()
    if g.lane == 0:
        xch.flag(group, blk)


@traced
def reduce_lines(g, sc, ga, wave_id, waves: int):
    """The owner half of a two-shot all-reduce: wave `wave_id` (of `waves`
    over the grid) takes lines `wave_id, wave_id + waves, ..` of this rank's
    owned groups, adds the scatter `sc`'s copies in rank order and pushes
    the bf16 sum to every rank through the gather `ga`."""
    lines = OWNED * sc.lpg
    for i in range(wave_id, fx.Int32(lines), waves):
        group = sc.rank * fx.Int32(OWNED) + i // fx.Int32(sc.lpg)
        blk = i % fx.Int32(sc.lpg)
        token = blk * fx.Int32(TOKENS) + g.lane // 8
        tk = fx.min(token, fx.Int32(sc.rows - 1))
        gates = wait_lines(g, sc, [(group, blk)])
        loads = [
            sc.get(fx.Int32(s), group, tk, g.lane % 8, gates[s]) for s in range(sc.tp)
        ]
        acc = list(loads[0])
        for s in range_constexpr(1, sc.tp):
            acc = [acc[e] + loads[s][e] for e in range(8)]
        push_line(g, ga, group, blk, acc)


@traced
def wait_row(g, xch, row, group):
    """The flags of `row`'s line of `group` from its sources, polled by the
    calling thread; returns the per-source gates (`wait_lines`)."""
    return wait_lines(g, xch, [(group, fx.Int32(row) // fx.Int32(TOKENS))])


def row_sum(xch, row, group, seg, gates):
    """`row`'s 8 values at segment `seg` of `group` summed over the
    sources in rank order (f32), after `wait_row`'s `gates`."""
    srcs = xch.sources(group)
    loads = [xch.get(s, group, row, seg, gates[i]) for i, s in enumerate(srcs)]
    acc = list(loads[0])
    for i in range_constexpr(1, len(srcs)):
        acc = [acc[e] + loads[i][e] for e in range(8)]
    return acc
