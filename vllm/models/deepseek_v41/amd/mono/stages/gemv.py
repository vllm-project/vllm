# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/attn_pre.py (its GEMV and LDS helpers).
"""The FP8 GEMV tiles of the seam and MoE stages and the helpers they share:
token tiles, LDS rows of MXFP8 words, scalar loads and the mailbox-to-LDS
copy."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.mx import (
    UNIT_SCALE,
    mfma_scaled,
)
from vllm.models.deepseek_v41.amd.mono.common.ops import (
    CM_NT,
    fresh,
    rsrc,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import WAVES
from vllm.models.deepseek_v41.amd.mono.common.sync import POLL_MAX

ROWS = 16  # GEMV rows a task
TILE = 16  # an MFMA's N: the tokens a GEMV pass takes (``token_tiles``)


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def token_tiles(s):
    """A step's tokens in GEMV passes: [(first token, tokens)]. A GEMV loads its
    weights once and runs every pass on them, an x tile in LDS at a time."""
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    """The LDS rows an x tile holds."""
    return min(s, TILE)


# a token's LDS row of MXFP8 words / E8M0 codes, 16 of which a MFMA B operand
# gathers, is padded by 16 B: at a multiple of 64 words they share a bank group
LDS_PAD = 4


def lds_row(words):
    return words + LDS_PAD


def lds_at(u, row):
    """Element u of a span of rows ``row`` wide, at the ``lds_row`` stride."""
    return u + u // row * LDS_PAD


def lds_tail(k):
    """Words a K = ``k`` row's last K step reads past its ``lds_row`` (selected
    out): the last row's buffer must hold them."""
    return max(0, 32 * -(-k // 128) - lds_row(k // 4))


def plus(t0, v):
    """Token ``v`` of the tile from ``t0`` (a first tile adds nothing; ``t0``
    traced: a tile a CTA picks at run time)."""
    return v if isinstance(t0, int) and t0 == 0 else v + t0


# ---------------------------------------------------------------- GEMV
def _fp8_step_load(c, r_w, r_ws, k, rg, st, row_major=False):
    """K step st's global operands of rows 16 rg ..: the two 64-column weight
    chunks (the upper clamped into the row) and the raw scale word. The (16, 16)
    shuffle lays a row group out in 32-column blocks of 16 rows, a chunk two of
    them (lanes 0-31, 32-63); ``k`` % 64 = 32 (a last quarter step): the lanes
    of a block past the row read its last block. ``row_major`` (vLLM's dense
    layout, k % 64 == 0): the same registers from a row-major [N, K] weight --
    lane l row 16 rg + l % 16, K bytes 128 st + 16 (l // 16) .. and + 64."""
    lane = c["lane"]
    j = lane // 16
    kg = k // 32
    if const_expr(row_major):
        assert k % 64 == 0
        row = rg * 16 + lane % 16
        lo_at = row * (k // 4) + st * 32 + j * 4
        hi_at = row * (k // 4) + fx.min(2 * st + 1, k // 64 - 1) * 16 + j * 4
        lo = fx.Vector(
            bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
        )
        hi = fx.Vector(
            bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
        )
        wsi = (rg // 2) * kg + fx.min(st * 4 + j, kg - 1)
        word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
        return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF
    if const_expr(k % 64 == 0):
        lo_at = (rg * (k // 64) + 2 * st) * 256 + lane * 4
    else:
        lo_at = rg * (k * 4) + fx.min(4 * st + lane // 32, kg - 1) * 128 + lane % 32 * 4
    lo = fx.Vector(
        bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    if const_expr(k % 64 == 0):
        hi_at = (rg * (k // 64) + fx.min(2 * st + 1, k // 64 - 1)) * 256 + lane * 4
    else:
        hi_at = rg * (k * 4) + fx.min(2 * st + 1, k // 64 - 1) * 256 + lane * 4
    hi = fx.Vector(
        bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    wsi = (rg // 2) * kg + fx.min(st * 4 + j, kg - 1)
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def _fp8_step_mfma(c, k, xl, xsl, st, ops, acc, live=None, rows=None):
    """``acc`` through K step st's scaled MFMA: ``_fp8_step_load``'s operands
    against every token's MXFP8 row in LDS. ``live`` false: a step past the end
    (its operands those of the last step), zero at unit scale. ``k`` % 64 = 32:
    a lane's lower block past the row is zero on both sides."""
    s, lane = c["S"] if rows is None else rows, c["lane"]
    lo, hi, sa = ops
    kg = k // 32
    x_row, s_row = lds_row(k // 4), lds_row(kg)
    j = lane // 16
    col = fx.min(lane % 16, s - 1)
    if const_expr(live is not None):
        st = fx.min(st, (k + 127) // 128 - 1)
        lo = [live.select(lo[d], fx.Int32(0)) for d in range(4)]
    past = (2 * st + 1) * 64 >= k
    if const_expr(live is not None):
        past = past | ~live
    av = fx.Vector.from_elements(
        [lo[d] for d in range(4)] + [past.select(fx.Int32(0), hi[d]) for d in range(4)],
        fx.Int32,
    )
    grp = st * 4 + j
    xlo = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xhi = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + 16 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    if const_expr(k % 64 != 0):
        over = 4 * st + j // 2 >= kg
        av = fx.Vector.from_elements(
            [over.select(fx.Int32(0), av[d]) for d in range(4)]
            + [av[4 + d] for d in range(4)],
            fx.Int32,
        )
        xlo = [over.select(fx.Int32(0), xlo[d]) for d in range(4)]
    xv = fx.Vector.from_elements(
        [xlo[d] for d in range(4)]
        + [past.select(fx.Int32(0), xhi[d]) for d in range(4)],
        fx.Int32,
    )
    sb = fx.ptr_load(xsl + (col * s_row + fx.min(grp, kg - 1)))
    unit = grp >= kg
    if const_expr(live is not None):
        unit = unit | ~live
    sa = unit.select(fx.Int32(UNIT_SCALE), sa)
    sb = unit.select(fx.Int32(UNIT_SCALE), sb)
    return mfma_scaled(av, xv, acc, sa, sb)


def _gemv_steps(c, k, split, tiled=False):
    """This wave's K steps (a step past the end marked, ``split`` waves past
    split repeat the last one's; ``tiled``: wave w takes part w % split of tile
    w / split): [(step, live)]."""
    wave = c["wave"]
    steps = (k + 127) // 128
    if const_expr(split is None):
        per = (steps + WAVES - 1) // WAVES
        ragged = steps % WAVES != 0
        return [
            (wave + WAVES * i, (wave + WAVES * i < steps) if ragged else None)
            for i in range(per)
        ]
    assert split <= WAVES and steps % split == 0
    per = steps // split
    part = wave % split if tiled else fx.min(wave, split - 1)
    return [(part * per + i, None) for i in range(per)]


def gemv_fp8_loads(c, w, ws, k, rg, split=None, tiled=False, row_major=False):
    """The weight operands of ``gemv_fp8``'s rows for this wave, every one in
    flight at once (a step's latency each would put them in series): loads
    only. A step past the end loads the last step. ``tiled`` (split x tiles =
    WAVES): row groups rg .., one a group of ``split`` waves."""
    steps = (k + 127) // 128
    r_w, r_ws = rsrc(w), rsrc(ws)
    if const_expr(tiled):
        assert WAVES % split == 0
        rg = rg + c["wave"] // split
    return [
        _fp8_step_load(
            c,
            r_w,
            r_ws,
            k,
            rg,
            st if live is None else fx.min(st, steps - 1),
            row_major,
        )
        for st, live in _gemv_steps(c, k, split, tiled)
    ]


def gemv_fp8_mfmas(c, k, xl, xsl, red, ops, split=None, tiled=False, rows=None):
    """``gemv_fp8_loads``' operands against the rows in LDS (``rows``: a token
    tile's, every token's by default) ->
    ``red[wave][lane]``; the scaled 16x16x128 MFMA, chunk kc in dwords 0-3 and kc +
    1 in 4-7. ``k`` a multiple of 32: a last half step has its upper chunk zeroed
    on both sides at unit scale (a NaN byte past the row must not reach the
    MFMA), its loads clamped into the row: the last row group's would read past
    the tensor; a last quarter step its lower chunk's upper block too. A step
    past the end adds zero at unit scale.

    The K order is the original kernel's, which ``row_sum`` completes:
    wave w takes steps w, w + 8, ... (``split`` None: aiter's preshuffled group32
    GEMM, P0.5 microbenchmark 1), or waves 0 .. split - 1 one contiguous K range
    each, the other waves zero (aiter's split-K bmm, whose last split sums the
    partials in split order)."""
    lane, wave = c["lane"], c["wave"]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i, (st, live) in enumerate(_gemv_steps(c, k, split, tiled)):
        acc = _fp8_step_mfma(c, k, xl, xsl, st, ops[i], acc, live, rows)
    if const_expr(split is not None and not tiled):
        acc = fx.Vector.from_elements(
            [(wave < split).select(acc[e], fx.Float32(0.0)) for e in range(4)],
            fx.Float32,
        )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)


# ---------------------------------------------------------------- mailbox -> LDS
@traced
def poll_copy(c, idx, width, spans, live=None):
    """Mailbox spans [(region, pair_of, n, dst, row)] into LDS: element u (pair
    ``pair_of(u)``) to dst + u, or ``lds_at(u, row)`` for a span of rows
    ``row`` words wide. Thread ``idx`` of ``width`` takes u = idx, idx +
    width, ..., POLL_MAX a round trip, each batch stored before the next is
    polled (holding them all spills past S = 16); a thread past a span's end
    re-reads its last element and stores nothing.

    ``live``: a span's elements the reader needs (traced, <= n), a span each;
    a batch past them is not polled and an element past them neither polled
    nor written (one past a step's rows is never written)."""
    idx = fresh(idx)  # its elements' indices not hoisted out of a task loop
    if const_expr(live is None):
        _poll_store_batches(
            c,
            [
                (region, pair_of, n, (dst, row), idx + width * i)
                for region, pair_of, n, dst, row in spans
                for i in range((n + width - 1) // width)
            ],
        )
        return
    for (region, pair_of, n, dst, row), need in zip(spans, live):
        chunks = [
            (region, pair_of, need, (dst, row), idx + width * i)
            for i in range((n + width - 1) // width)
        ]
        for b0 in range_constexpr(0, len(chunks), POLL_MAX):
            if chunks[b0][4] < need:
                _poll_store_batches(c, chunks[b0 : b0 + POLL_MAX])


@traced
def _poll_store_batches(c, chunks):
    """``poll_copy``'s chunks [(region, pair_of, n, (dst, row), u)], POLL_MAX a
    round trip, each batch stored before the next is polled."""
    for b0 in range_constexpr(0, len(chunks), POLL_MAX):
        batch = chunks[b0 : b0 + POLL_MAX]
        got = c["poll"]([(r, p(fx.min(u, n - 1)), 1) for r, p, n, _, u in batch])
        for (_, _, n, (dst, row), u), v in zip(batch, got):
            at = u if row is None else lds_at(u, row)
            if u < n:
                fx.ptr_store(v[0], dst + at)
