# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the Qwen3.8-Flash-Next wide decode's ``wide/stages/exchange.py``
# (TP2, one width) for TP8 and the Muse layer's exchanges.
"""The tensor-parallel exchanges of a layer over peer mailboxes, in lines.

Lines: a ``[rows, cols]`` tensor is stored as lines ordered by (column group,
token block): group ``g`` is columns ``64 g .. 64 g + 63`` (one 64-row group of
the producing GEMM, whose last arriving unit pushes it), a line eight tokens
of one group, 256 dwords. Lane ``l`` of the wave that writes a line holds
token ``l // 8`` of the block and four dwords at ``32 (l // 8) + 4 (l % 8)``:
eight bf16 of columns ``8 (l % 8) ..`` (``put`` / ``get``) or four raw words
(``put_words`` / ``get_words``, the f32 router logits). A step whose rows are
not a multiple of eight ends each group with a partial line: only its rows'
tokens are stored and read, at the same addresses as in a whole line.

A use's ``mode``:

- ``sum``: every rank pushes every group to every rank; a reader sums the
  ranks' copies in rank order, so every rank holds the same bf16 sum
  (one-shot all-reduce: every link carries the whole tensor).
- ``gather``: rank ``r`` pushes the groups it owns (``owner(g)``) to every
  rank; a reader takes group ``g`` from its owner (all-gather).
- ``scatter``: every rank pushes group ``g`` to its owner only, which sums the
  copies in rank order (the reduce-scatter half of a two-shot all-reduce,
  followed by a ``gather`` of the sums).

After its payload a wave drains its stores and sets the line's flag to the
use's tag in each destination's mailbox; a reader polls the flags of the lines
it needs and gates its payload loads on them.

A rank's mailbox (uncached, peer-mapped) has a region a ``Kind`` (a layer's
exchanges), two slots a region (the layer's parity), a row a slot per source
rank sized for ``MAX_TOKENS`` rows: payload, then one flag a line at a fixed
offset, so a flag word only ever holds a tag. A rank can be at most one layer
ahead of its peer, so a slot is never rewritten before every reader of its
previous use is done.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.wide.abi import (
    SITE_PEER,
    WAIT_TICKS,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    failed,
    give_up,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.ops import (
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
from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
    MAX_TOKENS,
    TP,
    WAVES,
)

COLS = 64  # bf16 columns of a line group
TOKENS = 8  # tokens of a line
LINE_DWORDS = 256
MODES = ("sum", "gather", "scatter")
PARITIES = 2


def lines_per_group(rows: int) -> int:
    return -(-rows // TOKENS)


@dataclass(frozen=True)
class Kind:
    """One of a layer's exchanges: ``groups`` line groups of the whole tensor."""

    name: str
    groups: int

    def payload_words(self, rows: int = MAX_TOKENS) -> int:
        return self.groups * lines_per_group(rows) * LINE_DWORDS

    def flag_words(self, rows: int = MAX_TOKENS) -> int:
        return self.groups * lines_per_group(rows)

    def row_words(self) -> int:
        """A source rank's row of a slot: payload, flags, padded to 64 dwords."""
        return (self.payload_words() + self.flag_words() + 63) // 64 * 64


class Mailbox:
    """The regions of a rank's mailbox, a ``Kind`` each, in ``kinds`` order."""

    def __init__(self, kinds: tuple[Kind, ...], tp: int = TP):
        self.kinds, self.tp = kinds, tp
        self.base: dict[str, int] = {}
        at = 0
        for k in kinds:
            self.base[k.name] = at
            at += PARITIES * tp * k.row_words()
        self.words = at

    def bytes(self) -> int:
        return self.words * 4


class Exchange(TraceInvariant):
    """One use of a ``Kind`` in ``mode``: this rank ``rank`` of ``tp``, slot
    ``parity``, tagged ``tag``. ``bases``: every rank's mailbox base (wave-
    uniform Int64, rank order), ``own``: this rank's; ``owner(g)``: the rank
    owning group ``g`` (gather / scatter modes), a trace-time function of an
    Int32."""

    def __init__(
        self,
        mailbox: Mailbox,
        kind: Kind,
        mode: str,
        bases,
        own,
        rank,
        parity,
        tag,
        rows: int,
        owner=None,
    ):
        if mode not in MODES:
            raise ValueError(f"exchange mode {mode!r}: one of {MODES}")
        if mode != "sum" and owner is None:
            raise ValueError(f"{mode} needs the groups' owner")
        self.kind, self.mode, self.tp = kind, mode, mailbox.tp
        self.region = fx.Int32(mailbox.base[kind.name])
        self.bases, self.own, self.rank = bases, fx.Int64(own), fx.Int32(rank)
        self.parity, self.tag, self.rows = fx.Int32(parity), fx.Int32(tag), rows
        self.owner = owner
        self.lpg = lines_per_group(rows)
        self.partial = rows % TOKENS != 0
        self.flags_at = kind.payload_words()

    def _row(self, src):
        return self.region + (self.parity * fx.Int32(self.tp) + src) * fx.Int32(
            self.kind.row_words()
        )

    def line(self, group, token):
        return group * fx.Int32(self.lpg) + token // fx.Int32(TOKENS)

    def flag_index(self, src, line):
        return self._row(src) + fx.Int32(self.flags_at) + line

    def _word(self, group, token, seg):
        return (
            self.line(group, token) * fx.Int32(LINE_DWORDS)
            + (token % fx.Int32(TOKENS)) * fx.Int32(32)
            + seg * fx.Int32(4)
        )

    def dests(self, group):
        """``(base, live)`` of every rank this rank pushes ``group`` to."""
        if const_expr(self.mode == "scatter"):
            own = self.owner(group)
            return [(b, own == fx.Int32(r)) for r, b in enumerate(self.bases)]
        return [(b, None) for b in self.bases]

    def put_words(self, group, token, seg, words):
        """The lane's four dwords of ``token`` into its destinations' rows of
        this rank; a token past the step's rows stores nothing."""
        at = self._row(self.rank) + self._word(group, token, seg)
        live = token < fx.Int32(self.rows) if self.partial else None
        for base, to in self.dests(group):
            ok = live if to is None else (to if live is None else to & live)
            if ok is None:
                st(base, at, words, cm=CM_SYS)
            else:
                _st_if(base, at, words, ok)

    def put(self, group, token, seg, vals):
        """The lane's 8 values (f32) of ``token``, columns ``64 group + 8 seg
        ..``, rounded to bf16."""
        self.put_words(group, token, seg, pack_bf16x8(vals))

    def flag(self, group, line):
        """Lane 0's flag of ``line`` of ``group``'s lines, after the drain."""
        at = self._row(self.rank) + fx.Int32(self.flags_at) + line
        for base, to in self.dests(group):
            if to is None:
                st(base, at, self.tag, cm=CM_SYS)
            else:
                _st_if(base, at, self.tag, to)

    def get_words(self, src, group, token, seg, gate):
        """``src``'s four dwords of ``token`` after the poll whose difference
        ``gate`` (zero) covers their line."""
        at = self._row(src) + gate + self._word(group, token, seg)
        return ld_i32x4(self.own, at, cm=CM_SYS)

    def get(self, src, group, token, seg, gate):
        return unpack_bf16x8(self.get_words(src, group, token, seg, gate))

    def sources(self, group):
        """The ranks a reader takes ``group`` from: every rank (sum, scatter at
        the owner) or the owner (gather) -- as Int32 values."""
        if const_expr(self.mode == "gather"):
            return [self.owner(group)]
        return [fx.Int32(r) for r in range(self.tp)]


@traced
def _st_if(base, at, value, ok):
    if ok:
        st(base, at, value, cm=CM_SYS)


@traced
def wait_lines(g, ex, lines):
    """Poll, from every lane, the flags of ``lines`` (a list of (group, line)
    pairs, the lane's own) of every source ``ex.sources(group)`` until all
    carry the tag, bounded. Returns the differences from the tag (zero once
    the poll ends: 0 or 1, so a timed-out poll's loads stay in bounds), one
    a (line, source), to add to the payload loads."""
    idx = [
        ex.flag_index(src, line) for group, line in lines for src in ex.sources(group)
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


@traced
def sum_line(g, ex, group, token, seg):
    """The lane's 8 values of ``token``, columns ``64 group + 8 seg ..``,
    summed over ``ex.sources(group)`` in rank order after their flags (f32;
    the caller rounds)."""
    lines = [(group, ex.line(group, token))]
    gates = wait_lines(g, ex, lines)
    srcs = ex.sources(group)
    loads = [ex.get(s, group, token, seg, gates[i]) for i, s in enumerate(srcs)]
    acc = list(loads[0])
    for i in range_constexpr(1, len(srcs)):
        acc = [acc[e] + loads[i][e] for e in range(8)]
    return acc


@traced
def put_line(g, ex, group, line, vals):
    """A wave's line ``line`` of ``group`` (lane ``l``: token ``8 line + l //
    8``, columns ``64 group + 8 (l % 8) ..``, 8 f32 rounded to bf16), drained,
    then flagged from lane 0."""
    token = line * fx.Int32(TOKENS) + g.lane // 8
    ex.put(group, token, g.lane % 8, vals)
    drain_stores()
    if g.lane == 0:
        ex.flag(group, group * fx.Int32(ex.lpg) + line)


@traced
def reduce_lines(g, sc, ga, owned: int, wave_id, waves):
    """The owner half of a two-shot all-reduce: wave ``wave_id`` (of ``waves``
    over the grid) takes lines ``wave_id, wave_id + waves, ..`` of this rank's
    ``owned`` groups (``rank * owned ..``), sums the scatter ``sc``'s copies in
    rank order and pushes the sum to every rank through the gather ``ga``."""
    lines = owned * sc.lpg
    for i in range(wave_id, fx.Int32(lines), waves):
        group = sc.rank * fx.Int32(owned) + i // fx.Int32(sc.lpg)
        line = i % fx.Int32(sc.lpg)
        token = line * fx.Int32(TOKENS) + g.lane // 8
        vals = sum_line(g, sc, group, token, g.lane % 8)
        put_line(g, ga, group, line, vals)


@traced
def wait_groups(g, ex, first, count: int):
    """Every line of groups ``first .. first + count`` from its source(s),
    polled by the CTA's lanes together, then a CTA barrier: after it any
    thread may read them (``LineSource``)."""
    pairs = count * ex.lpg
    for r in range_constexpr(-(-pairs // (WAVES * 64))):
        i = g.tid + fx.Int32(r * WAVES * 64)
        if i < pairs:
            group = first + i // fx.Int32(ex.lpg)
            wait_lines(
                g, ex, [(group, group * fx.Int32(ex.lpg) + i % fx.Int32(ex.lpg))]
            )
    fx.barrier()


class LineSource(TraceInvariant):
    """A GEMM's bf16 activations read from a gather's lines in this rank's
    mailbox (the ``tiled.Source`` interface): K element ``k`` of a token is
    column ``k`` of the exchanged tensor, a 16-byte item one lane's segment of
    a line. The caller has waited for the lines (``wait_groups``)."""

    def __init__(self, ex):
        assert ex.mode == "gather"
        self.ex = ex

    def prepare(self, token):
        return token

    def issue(self, token, byte):
        k = byte // fx.Int32(2)
        group = k // fx.Int32(COLS)
        seg = k % fx.Int32(COLS) // fx.Int32(8)
        ex = self.ex
        at = ex._row(ex.owner(group)) + ex._word(group, token, seg)
        return [ld_i32x4(ex.own, at, cm=CM_SYS)]

    def convert(self, raw, token):
        return raw[0]


class GroupTile(TraceInvariant):
    """An LDS ``[rows, 64]`` bf16 tile (i32 words, a 16-byte pad a row) that a
    producer gathers one column group of every token in before it pushes the
    group's lines (``push_tile``)."""

    STRIDE = COLS // 2 + 4

    def __init__(self, ptr, rows: int):
        self.ptr, self.rows = ptr, rows

    @staticmethod
    def words(rows: int) -> int:
        return rows * GroupTile.STRIDE

    def put_vals(self, token, col, vals4):
        """Four f32 of ``token`` at columns ``col ..`` of the group, rounded to
        bf16."""
        w = fx.Vector.from_elements(list(vals4), fx.Float32)
        lds_st(
            self.ptr + token * self.STRIDE + col // 2,
            w.to(fx.BFloat16).bitcast(fx.Int32),
        )

    def put_acc(self, col0, lane, acc, tiles: int, scale=None):
        """A wave's GEMM accumulators (16 columns from ``col0`` within the
        group, the epilogue layout: lane ``l`` holds columns ``col0 + 4 (l //
        16) ..`` of token ``16 t + l % 16``), each times ``scale(token, col)``
        (a 4-vector of f32) when given, rounded to bf16."""
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
    """The tile (column group ``group`` of every token) as exchange lines to
    the group's destinations: wave ``w`` pushes lines ``w, w + 4, ..``, the
    payload first, one drain, then each line's flag from lane 0."""
    lpg = lines_per_group(tile.rows)
    seg = g.lane % 8
    for k in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + fx.Int32(k * WAVES)
        if line < lpg:
            token = line * TOKENS + g.lane // 8
            tk = token
            if const_expr(xch.partial):
                tk = fx.min(token, fx.Int32(tile.rows - 1))
            words = lds_ld(tile.ptr + tk * GroupTile.STRIDE + seg * 4, 4)
            xch.put_words(group, token, seg, words)
    drain_stores()
    for k in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + fx.Int32(k * WAVES)
        if (line < lpg) & (g.lane == 0):
            xch.flag(group, group * lpg + line)
