# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tensor-parallel exchange of a block's output: every rank's partial
``y_r [rows, HIDDEN]`` (bf16) lands in every rank's mailbox, and a reader sums
the ranks' copies in rank order, so every rank holds the same bf16 sum.

Lines: a partial is stored as lines ordered by
(column group, token block): group ``g`` is columns ``64 g .. 64 g + 63`` (one
row group of the output GEMM, whose last arriving unit pushes it), a line
eight tokens of one group. Lane ``l`` of the wave that writes a line holds
token ``l // 8`` of the block, columns ``8 (l % 8) ..``. A step whose rows are
not a multiple of eight ends each group with a partial line: only its rows'
tokens are stored and read (a one-row step moves 128 B a line), at the same
addresses as in a whole line. A line's ``codec``:

- ``bf16`` (exact): 256 dwords (1 KiB), a lane's four packed bf16 dwords;
- ``int4`` (QuickReduce's INT4 codec, opt-in): 80 dwords (320 B), a lane's
  eight 4-bit codes in one dword, then each token's four group scales as bf16.
  A group is 16 columns (a lane pair); its scale maps the group's signed
  extremum to -8 (codes ``round(x / scale)`` clamped to [-8, 7]). Every rank
  reads every rank's copy, its own included, so the ranks' sums agree.

After its payload a wave drains its stores and sets the line's flag to the
use's tag, in every rank's mailbox; a reader polls the flags of the lines it
needs and gates its payload loads on them. Payload and flags carry 99.6 %
payload on the link (bf16 lines), where tagged pairs would carry half.

A rank's mailbox (uncached, peer-mapped): ``SLOTS`` slots, a slot a row per
source rank of the widest step's payload, then one flag a line at a fixed
offset (``FLAGS_AT``): a flag word is never a narrower or wider step's
payload, so it only ever holds a tag. A layer's two
exchanges (attention, MLP) and the layer's parity pick the slot: a rank can be
at most one layer ahead of its peer, so a slot is never rewritten before every
reader of its previous use is done.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath

from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import SITE_PEER, WAIT_TICKS
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    failed,
    give_up,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_SYS,
    _xshfl,
    bf16_round,
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
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    HIDDEN,
    MAX_TOKENS,
    WAVES,
)

COLS = 64  # columns of a line group
TOKENS = 8  # tokens of a line
LINE_DWORDS = 256  # a bf16 line
INT4_LINE_DWORDS = 80  # an int4 line: 64 code dwords, then 2 scale dwords a token
CODECS = ("bf16", "int4")
GROUPS = HIDDEN // COLS
SLOTS = 4


def line_words(codec: str) -> int:
    return LINE_DWORDS if codec == "bf16" else INT4_LINE_DWORDS


def lines_per_group(rows: int) -> int:
    return -(-rows // TOKENS)


def payload_words(rows: int = MAX_TOKENS, codec: str = "bf16") -> int:
    return GROUPS * lines_per_group(rows) * line_words(codec)


def flag_words(rows: int = MAX_TOKENS) -> int:
    return GROUPS * lines_per_group(rows)


# a source rank's flags: past the widest payload (the bf16 lines, the larger)
FLAGS_AT = payload_words()


def row_words() -> int:
    """A source rank's row of a slot: payload, flags, padded to 64 dwords."""
    return (FLAGS_AT + flag_words() + 63) // 64 * 64


def mailbox_bytes(tp: int) -> int:
    return SLOTS * tp * row_words() * 4


class Exchange(TraceInvariant):
    """One use of the exchange: slot ``slot`` of every rank's mailbox, this
    rank ``rank`` of ``tp``, tagged ``tag``. ``bases``: every rank's mailbox
    base (wave-uniform Int64, rank order), ``own``: this rank's."""

    def __init__(
        self, bases, own, rank, tp: int, slot, tag, rows: int, codec: str = "bf16"
    ):
        if codec not in CODECS:
            raise ValueError(f"exchange codec {codec!r}: one of {CODECS}")
        self.bases, self.own, self.rank, self.tp = (
            bases,
            fx.Int64(own),
            fx.Int32(rank),
            tp,
        )
        self.slot, self.tag, self.rows = fx.Int32(slot), fx.Int32(tag), rows
        self.codec, self.words = codec, line_words(codec)
        self.lpg = lines_per_group(rows)
        self.partial = rows % TOKENS != 0  # each group's last line is partial

    def _row(self, src):
        return (self.slot * fx.Int32(self.tp) + src) * fx.Int32(row_words())

    def flag_index(self, src, line):
        return self._row(src) + fx.Int32(FLAGS_AT) + line

    def _line_at(self, group, token):
        line = group * fx.Int32(self.lpg) + token // fx.Int32(TOKENS)
        return line * fx.Int32(self.words), token % fx.Int32(TOKENS)

    def put(self, group, token, seg, vals):
        """The lane's 8 values (f32) of ``token``, columns ``64 group + 8 seg
        ..``, into every rank's row of this rank. Lane ``l`` of the wave holds
        token ``l // 8`` of the line, ``seg = l % 8``; every lane of the wave
        takes part (the int4 groups pair lanes), and a token past the step's
        rows (a partial line's) stores nothing."""
        row = self._row(self.rank)
        at, t8 = self._line_at(group, token)
        if const_expr(self.partial):
            _put_partial(self, row + at, t8, seg, vals, token < fx.Int32(self.rows))
        elif const_expr(self.codec == "bf16"):
            words = pack_bf16x8(vals)
            for base in self.bases:
                st(base, row + at + t8 * 32 + seg * 4, words, cm=CM_SYS)
        else:
            _put_int4(self, row + at + t8 * 8 + seg, row + at + 64 + t8 * 2, seg, vals)

    def get(self, src: int, group, token, seg, gate):
        """``src``'s 8 values (f32) of ``token``, columns ``64 group + 8 seg
        ..``, after the poll whose difference ``gate`` (zero) covers their
        line."""
        row = self._row(fx.Int32(src)) + gate
        at, t8 = self._line_at(group, token)
        if const_expr(self.codec == "bf16"):
            return unpack_bf16x8(
                ld_i32x4(self.own, row + at + t8 * 32 + seg * 4, cm=CM_SYS)
            )
        codes = ld(self.own, row + at + t8 * 8 + seg, fx.Int32, cm=CM_SYS)
        pair = ld(self.own, row + at + 64 + t8 * 2 + seg // 4, fx.Int32, cm=CM_SYS)
        scale = ((pair >> (((seg // 2) % 2) * 16)) << 16).bitcast(fx.Float32)
        return [fx.Float32(((codes >> (4 * e)) & 15) - 8) * scale for e in range(8)]


@traced
def _put_int4(xch, code_at, scale_at, seg, vals):
    codes, scales = _int4_encode(vals)
    for base in xch.bases:
        st(base, code_at, codes, cm=CM_SYS)
    if seg % 4 == 0:
        for base in xch.bases:
            st(base, scale_at + seg // 4, scales, cm=CM_SYS)


@traced
def _put_partial(xch, line_at, t8, seg, vals, live):
    """``put`` into a partial line (``line_at``: its first word in this rank's
    row): every lane encodes (the int4 groups pair lanes), only the lanes of
    the step's rows (``live``) store."""
    if const_expr(xch.codec == "bf16"):
        words = pack_bf16x8(vals)
        if live:
            for base in xch.bases:
                st(base, line_at + t8 * 32 + seg * 4, words, cm=CM_SYS)
    else:
        codes, scales = _int4_encode(vals)
        if live:
            for base in xch.bases:
                st(base, line_at + t8 * 8 + seg, codes, cm=CM_SYS)
        if live & (seg % 4 == 0):
            for base in xch.bases:
                st(base, line_at + 64 + t8 * 2 + seg // 4, scales, cm=CM_SYS)


def _int4_encode(vals):
    """A lane's 8 values as 4-bit codes (one dword, value ``e`` in bits ``4 e
    ..``) under its 16-column group's scale (the lane and its ``lane ^ 1``
    partner): ``scale = bf16(extremum / -8)``. Returns the codes and, for the
    lanes ``seg % 4 == 0``, the dword of its two groups' bf16 scales."""
    hi, lo = vals[0], vals[0]
    for v in vals[1:]:
        hi = (v > hi).select(v, hi)
        lo = (v < lo).select(v, lo)
    other = _xshfl(hi, 1)
    hi = (other > hi).select(other, hi)
    other = _xshfl(lo, 1)
    lo = (other < lo).select(other, lo)
    ext = (hi > -lo).select(hi, lo)
    scale = bf16_round(ext * fx.Float32(-0.125))
    inv = (scale == fx.Float32(0.0)).select(fx.Float32(0.0), fx.Float32(1.0) / scale)
    codes = fx.Int32(0)
    for e in range_constexpr(8):
        q = fmath.roundeven(vals[e] * inv)
        q = (q < -8.0).select(fx.Float32(-8.0), (q > 7.0).select(fx.Float32(7.0), q))
        codes = codes | ((fx.Int32(q) + 8) << (4 * e))
    bits = fx.Int32(scale.bitcast(fx.Int32)).shrui(fx.Int32(16))
    scales = bits | (fx.Int32(_xshfl(bits, 2)) << 16)
    return codes, scales


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
            tk = token
            if const_expr(xch.partial):
                tk = fx.min(token, fx.Int32(tile.rows - 1))  # a partial line's tail
            words = lds_ld(tile.ptr + tk * GroupTile.STRIDE + seg * 4, 4)
            xch.put(group, token, seg, unpack_bf16x8(words))
    drain_stores()
    for k in range_constexpr(-(-lpg // WAVES)):
        line = g.wave + fx.Int32(k * WAVES)
        if (line < lpg) & (g.lane == 0):
            for base in xch.bases:
                st(
                    base,
                    row + fx.Int32(FLAGS_AT) + group * lpg + line,
                    xch.tag,
                    cm=CM_SYS,
                )
