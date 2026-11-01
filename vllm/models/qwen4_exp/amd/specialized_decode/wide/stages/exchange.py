# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tensor-parallel exchange of a block's output: every rank's partial
``y_r [rows, HIDDEN]`` (bf16) lands in every rank's mailbox, and a reader sums
the ranks' copies in rank order, so every rank holds the same bf16 sum.

After FlyTape's line exchange: a partial is stored as lines of 256 dwords (1
KiB), ordered by (column group, token block): group ``g`` is columns ``64 g ..
64 g + 63`` (one row group of the output GEMM, whose last arriving unit pushes
it), a line eight tokens of one group. Lane ``l`` of the wave that writes a
line holds token ``l // 8`` of the block, columns ``8 (l % 8) ..``: four packed
dwords, one 16-byte store. After its payload a wave drains its stores and sets
the line's flag to the use's tag, in every rank's mailbox; a reader polls the
flags of the lines it needs and gates its payload loads on them. Payload and
flags carry 99.6 % payload on the link, where tagged pairs would carry half.

A rank's mailbox (uncached, peer-mapped): ``SLOTS`` slots, a slot a row per
source rank of ``payload_words`` dwords then one flag a line. A layer's two
exchanges (attention, MLP) and the layer's parity pick the slot: a rank can be
at most one layer ahead of its peer, so a slot is never rewritten before every
reader of its previous use is done.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import SITE_PEER, WAIT_TICKS
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    failed,
    give_up,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_SYS,
    drain_stores,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    memrealtime,
    st,
    traced,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    HIDDEN,
    MAX_TOKENS,
    WAVES,
)

COLS = 64  # columns of a line group
TOKENS = 8  # tokens of a line
LINE_DWORDS = 256
GROUPS = HIDDEN // COLS
SLOTS = 4


def lines_per_group(rows: int) -> int:
    return -(-rows // TOKENS)


def payload_words(rows: int = MAX_TOKENS) -> int:
    return GROUPS * lines_per_group(rows) * LINE_DWORDS


def flag_words(rows: int = MAX_TOKENS) -> int:
    return GROUPS * lines_per_group(rows)


def row_words(rows: int = MAX_TOKENS) -> int:
    """A source rank's row of a slot: payload, flags, padded to 64 dwords."""
    return (payload_words(rows) + flag_words(rows) + 63) // 64 * 64


def mailbox_bytes(tp: int) -> int:
    return SLOTS * tp * row_words() * 4


def line_dword(rows: int, group, token, seg):
    """Dword of the four packed pairs of columns ``64 group + 8 seg .. + 7`` of
    ``token``."""
    line = group * fx.Int32(lines_per_group(rows)) + token // fx.Int32(TOKENS)
    return (
        line * fx.Int32(LINE_DWORDS)
        + (token % fx.Int32(TOKENS)) * fx.Int32(32)
        + seg * fx.Int32(4)
    )


class Exchange(TraceInvariant):
    """One use of the exchange: slot ``slot`` of every rank's mailbox, this
    rank ``rank`` of ``tp``, tagged ``tag``. ``bases``: every rank's mailbox
    base (wave-uniform Int64, rank order), ``own``: this rank's."""

    def __init__(self, bases, own, rank, tp: int, slot, tag, rows: int):
        self.bases, self.own, self.rank, self.tp = (
            bases,
            fx.Int64(own),
            fx.Int32(rank),
            tp,
        )
        self.slot, self.tag, self.rows = fx.Int32(slot), fx.Int32(tag), rows
        self.payload = payload_words(rows)

    def _row(self, src):
        return (self.slot * fx.Int32(self.tp) + src) * fx.Int32(row_words())

    def flag_index(self, src, line):
        return self._row(src) + fx.Int32(self.payload) + line

    def load4(self, src, dword, gate):
        """Four payload dwords of ``src``'s row at ``dword`` after the poll whose
        difference ``gate`` (zero) covers their line."""
        return ld_i32x4(self.own, self._row(fx.Int32(src)) + dword + gate, cm=CM_SYS)


@traced
def wait_lines(g, ex, lines):
    """Poll, from every lane, the flags of ``lines`` (a list of Int32 line
    indices, the lane's own) of every source rank until all carry the tag,
    bounded. Returns the per (line, source) differences from the tag (zero
    once the poll ends), to add to the lines' payload loads."""
    idx = [
        ex.flag_index(fx.Int32(src), line)
        for line in lines
        for src in range_constexpr(ex.tp)
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
    return [v - ex.tag for v in seen]


class GroupTile(TraceInvariant):
    """An LDS ``[rows, 64]`` bf16 tile (i32 words, a 16-byte pad a row) that a
    block's producer gathers one column group of every token in before it
    pushes the group's lines (``push_tile``)."""

    STRIDE = COLS // 2 + 4

    def __init__(self, ptr, rows: int):
        self.ptr, self.rows = ptr, rows

    @staticmethod
    def words(rows: int) -> int:
        return rows * GroupTile.STRIDE

    def put_acc(self, col0, lane, acc, tiles: int):
        """A wave's GEMM accumulators (16 columns from ``col0`` within the
        group, the epilogue layout: lane ``l`` holds columns ``col0 + 4 (l //
        16) ..`` of token ``16 t + l % 16``) rounded to bf16."""
        for t in range(tiles):
            token = fx.Int32(t * 16) + lane % 16
            tk = (token < self.rows).select(token, fx.Int32(self.rows - 1))
            w = fx.Vector.from_elements([acc[t][e] for e in range(4)], fx.Float32)
            lds_st(
                self.ptr + tk * self.STRIDE + (col0 + lane // 16 * 4) // 2,
                w.to(fx.BFloat16).bitcast(fx.Int32),
            )


@traced
def push_tile(g, xch, tile, group):
    """The tile (column group ``group`` of every token) as exchange lines to
    every rank: wave ``w`` pushes lines ``w, w + 4, ..``, the payload first,
    one drain, then each line's flag from lane 0."""
    lpg = lines_per_group(tile.rows)
    seg = g.lane % 8
    row = xch._row(xch.rank)
    for k in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + fx.Int32(k * WAVES)
        if line < lpg:
            token = line * TOKENS + g.lane // 8
            words = lds_ld(tile.ptr + token * GroupTile.STRIDE + seg * 4, 4)
            for base in xch.bases:
                st(
                    base,
                    row + line_dword(tile.rows, group, token, seg),
                    words,
                    cm=CM_SYS,
                )
    drain_stores()
    for k in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + fx.Int32(k * WAVES)
        if (line < lpg) & (g.lane == 0):
            for base in xch.bases:
                st(
                    base,
                    row + fx.Int32(xch.payload) + group * lpg + line,
                    xch.tag,
                    cm=CM_SYS,
                )
