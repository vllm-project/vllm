# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the Qwen3.8-Flash-Next wide decode's ``wide/stages/tiled.py``
# (gfx950, bf16 16x16x32) for gfx942's FP8 16x16x32 and bf16 16x16x16 MFMAs.
"""Batch-tiled GEMMs on the matrix unit, FP8 (e4m3fnuz) or bf16.

A CTA computes ``C[token, row] = sum_k X[token, k] * W[row, k]`` for 64 x
``mtiles`` weight rows (16-row tiles, one a wave per ``mtiles`` step) over a
slice of K and every row of the step. Activations stream through LDS in
``CHUNK_BYTES``-wide K steps shared by the four waves, double buffered;
weights go from global memory straight to the A operand registers, ``DEPTH``
chunks ahead. The step's rows are the MFMA's N dimension: ``ceil(S / 16)``
token tiles reuse each weight load, and the staged activations never grow
past two chunks (37 KB at 128 rows, of gfx942's 64 KB).

Everything is counted in bytes: a chunk is ``chunk`` bytes of a row (128 by
default: 128 FP8 K or 64 bf16 K), ``chunk / 32`` MFMA K slices of 8 bytes a
lane (``v_mfma_f32_16x16x32_fp8_fp8``: 8 codes; ``v_mfma_f32_16x16x16bf16_1k``:
4 bf16). Lane ``l`` of a wave holds
weight row ``l % 16`` and, for slice ``m`` of a chunk, the 8 bytes at
``koff(m, l // 16) = 64 (m // 2) + 16 (l // 16) + 8 (m % 2)``: one 16-byte load
a lane feeds two slices, and a load instruction's four lanes of a row read 64
contiguous bytes. The B operand is read from LDS at the same ``koff``. Its
accumulator of token tile ``t`` holds rows ``4 (l // 16) + e`` (e = 0..3) of
token ``16 t + l % 16``. The sum over K runs in a different order than
hipBLASLt's or Triton's (fp32 accumulation order only).

A split-K plan (``SplitK``) cuts a GEMM too narrow to fill the grid along K;
its partials ``[split, token, row]`` (f32) are summed in split order by a
later phase or by the row group's last arriving unit (``fold``).
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    fold_arrive,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_NT,
    i64_of,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    mfma_bf16,
    mfma_fp8,
    sched_barrier,
    st,
    traced,
    zeros,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
    THREADS,
    WAVES,
)

LANES = 64
TILE = 16
CHUNK_BYTES = 128  # bytes of a row a staged step (default)
ITEM_BYTES = 16  # bytes a staging thread moves at once
ROWS = WAVES * TILE  # weight rows a CTA covers per weight tile step
DEPTH = 2  # chunks of loads in flight
UNROLL = 4  # K chunks up to which the chunk loop is unrolled


@traced
def _lds_st_if(ptr, value, ok):
    if ok:
        lds_st(ptr, value)


def token_tiles(rows: int) -> int:
    return -(-rows // TILE)


def stage_words(rows: int, chunk: int = CHUNK_BYTES) -> int:
    """i32 words of a GEMM's LDS stage: two chunk buffers of every padded row."""
    return 2 * token_tiles(rows) * TILE * (chunk // 4 + 4)


def i32(v) -> fx.Int32:
    return fx.Int32(v)


def koff(m, g):
    """Byte offset in a chunk of a lane's slice ``m`` operand (``g = lane // 16``)."""
    return i32(64 * (m // 2) + 8 * (m % 2)) + g * i32(16)


# --------------------------------------------------------------------------
# weight sources (the A operand)
# --------------------------------------------------------------------------


class Weights(TraceInvariant):
    """A row-major ``[rows, cols]`` matrix of ``elem`` bytes an element (1:
    e4m3fnuz, 2: bf16) at byte address ``addr``. ``issue`` returns a lane's two
    16-byte loads of a chunk; ``operand`` slice ``m``'s 8 bytes of them."""

    def __init__(
        self,
        addr,
        rows: int,
        cols: int,
        elem: int,
        cm: int = CM_NT,
        row_bytes: int | None = None,
    ):
        if cols * elem % CHUNK_BYTES:
            raise ValueError(f"K {cols} is not whole {CHUNK_BYTES}-byte chunks")
        self.addr, self.rows, self.cols, self.elem, self.cm = (
            fx.Int64(addr),
            rows,
            cols,
            elem,
            cm,
        )
        # a padded matrix's row stride (bytes), e.g. ROCm's MoE weight padding
        self.row_bytes = cols * elem if row_bytes is None else row_bytes
        assert self.row_bytes % 16 == 0 and self.row_bytes >= cols * elem
        self.nbytes = rows * self.row_bytes

    def issue(self, row, chunk, g, mask=None, chunk_bytes=CHUNK_BYTES):
        """With ``mask`` false, no request (zeros); a row past ``rows`` reads the
        last row."""
        row = (row < i32(self.rows)).select(row, i32(self.rows - 1))
        out = []
        for j in range(chunk_bytes // 64):
            at = (
                row * i32(self.row_bytes) + chunk * i32(chunk_bytes) + koff(2 * j, g)
            ) // i32(4)
            if mask is None:
                out.append(ld_i32x4(self.addr, at, self.cm))
            else:
                out.append(
                    ld(
                        self.addr,
                        at,
                        fx.Int32,
                        4,
                        cm=self.cm,
                        nbytes=self.nbytes,
                        mask=mask,
                    )
                )
        return out

    @staticmethod
    def operand(loaded, m):
        w = loaded[m // 2]
        return (w[2 * (m % 2)], w[2 * (m % 2) + 1])


def fp8_weights(addr, rows, cols, cm=CM_NT, row_bytes=None):
    return Weights(addr, rows, cols, 1, cm, row_bytes)


def bf16_weights(addr, rows, cols, cm=CM_NT):
    return Weights(addr, rows, cols, 2, cm)


def mma(elem: int, a, b, acc):
    """One slice: ``a``, ``b`` the (lo, hi) words of the lane's 8 bytes."""
    if elem == 1:
        return mfma_fp8(i64_of(*a), i64_of(*b), acc)
    va = fx.Vector.from_elements([fx.Int32(a[0]), fx.Int32(a[1])], fx.Int32)
    vb = fx.Vector.from_elements([fx.Int32(b[0]), fx.Int32(b[1])], fx.Int32)
    return mfma_bf16(va.bitcast(fx.Int16), vb.bitcast(fx.Int16), acc)


# --------------------------------------------------------------------------
# activation sources (the B operand, staged through LDS)
# --------------------------------------------------------------------------


class Source(TraceInvariant):
    """Activations ``X[token, k]`` already in the GEMM's element format, a
    row-major ``[tokens, ld]`` matrix at ``addr`` (``ld`` in bytes).
    ``prepare(token)``: what a staged token's loads need over the whole K
    (computed once, before the K loop); ``issue(prep, byte)`` returns the raw
    loads of 16 bytes, ``convert(raw, prep)`` them as four i32 words. ``fill``
    issues a chunk K steps ahead and ``put`` converts after a step's MFMAs, so
    the wait for the loads lands behind the matrix work. ``rows(token)`` maps
    a staged token to its source row (the MoE's gathered routes)."""

    def __init__(self, addr, ld_bytes: int, cm: int = CM_DEV, rows=None):
        self.addr, self.ld, self.cm, self.rows = fx.Int64(addr), ld_bytes, cm, rows

    def prepare(self, token):
        return token if self.rows is None else self.rows(token)

    def issue(self, row, byte):
        return [ld_i32x4(self.addr, (row * i32(self.ld) + byte) // i32(4), self.cm)]

    def convert(self, raw, row):
        return raw[0]


class Gemm(TraceInvariant):
    """The staged-activation K loop for one step of ``rows`` tokens and ``elem``
    -byte elements, ``chunk`` bytes of K a step. ``page``: the LDS stage
    (``stage_words(rows, chunk)`` i32 words). ``grouped`` (a long K): the
    stage holds ``depth`` chunks, one group, staged once a group with one
    barrier (``stage_words(rows, depth * chunk)``), not once a chunk."""

    def __init__(
        self,
        page,
        rows: int,
        tid,
        elem: int,
        mtiles: int = 1,
        depth: int = DEPTH,
        chunk: int = CHUNK_BYTES,
        grouped: bool = False,
    ):
        self.grouped = grouped
        self.page = page
        self.batch = rows
        self.tiles = token_tiles(rows)
        self.tid = i32(tid)
        self.lane = self.tid % i32(LANES)
        self.group = self.lane // i32(TILE)
        self.elem = elem
        self.mtiles = mtiles
        self.depth = depth
        self.chunk = chunk
        self.slices = chunk // 32
        # the stage's K bytes: a chunk, or a group of ``depth`` chunks
        self.xchunk = chunk * depth if grouped else chunk
        self.items = self.xchunk // ITEM_BYTES
        self.xstride = self.xchunk // 4 + 4

    # staging: thread v of a token tile round holds token v // items, bytes
    # 16 (v % items) .. + 15 of the chunk
    def _item(self, t):
        v = i32(t * THREADS) + self.tid
        token = v // i32(self.items)
        if (t + 1) * THREADS // self.items > self.batch:
            token = (token < i32(self.batch)).select(token, i32(self.batch - 1))
        return v, token

    def _rounds(self):
        return -(-self.tiles * TILE * self.items // THREADS)

    def prepare(self, source):
        """Each staging round's ``source.prepare`` of its token (a thread
        stages the same token at every chunk)."""
        return [source.prepare(self._item(t)[1]) for t in range(self._rounds())]

    def fill(self, source, chunk, prep):
        out = []
        for t in range(self._rounds()):
            v, _ = self._item(t)
            byte = chunk * i32(self.xchunk) + (v % i32(self.items)) * i32(ITEM_BYTES)
            out.append(source.issue(prep[t], byte))
        return out

    def put(self, buf, items, source, prep):
        """A thread whose item is past the tiles' rows (fewer than 16 rows:
        128 items of 256 threads) stores nothing: its row would be the other
        buffer's, which the MFMAs may still read."""
        items_per_buf = self.tiles * TILE * self.items
        for t in range(self._rounds()):
            v, _ = self._item(t)
            at = (buf * i32(self.tiles * TILE) + v // i32(self.items)) * i32(
                self.xstride
            ) + (v % i32(self.items)) * i32(ITEM_BYTES // 4)
            value = fx.Vector(source.convert(items[t], prep[t]))
            if (t + 1) * THREADS > items_per_buf:
                _lds_st_if(self.page + at, value, v < i32(items_per_buf))
            else:
                lds_st(self.page + at, value)

    def operands(self, buf, t, sub: int = 0):
        """Token tile ``t``'s slice operands of LDS buffer ``buf`` (chunk ``sub``
        of a group)."""
        base = (
            (buf * i32(self.tiles) + i32(t)) * i32(TILE) + self.lane % i32(TILE)
        ) * i32(self.xstride) + i32(sub * self.chunk // 4)
        out = []
        for m in range(self.slices):
            w = lds_ld(self.page + base + koff(m, self.group) // i32(4), 2)
            out.append((w[0], w[1]))
        return out

    def mma_tiles(self, acc, a, operands):
        """``acc`` plus the weight operands ``a`` ([mtile] of loaded) times the B
        operands ``operands(t)`` of each token tile; tile ``t + 1``'s are read
        while tile ``t``'s MFMAs run."""
        acc = [list(r) for r in acc]
        ops = operands(0)
        for t in range(self.tiles):
            ahead = operands(t + 1) if t + 1 < self.tiles else None
            for m in range(self.slices):
                for mt in range(self.mtiles):
                    acc[mt][t] = mma(
                        self.elem, Weights.operand(a[mt], m), ops[m], acc[mt][t]
                    )
            ops = ahead
        return acc

    def multiply(self, buf, a, acc):
        return self.mma_tiles(acc, a, lambda t: self.operands(buf, t))

    def stage(
        self, source, weights, rows, first, count: int, i, prep, pre=None, rot=None
    ):
        """The activation and weight loads of chunk ``first + min(i, count - 1)``
        (rotated by ``rot`` mod ``count``); ``pre``: its weight loads, already
        issued (``preload``)."""
        k = (i < i32(count)).select(i, i32(count - 1))
        if rot is not None:
            k = (k + rot) % i32(count)
        chunk = first + k
        # the fill first: its LDS store waits for it, and vmcnt is in order
        xs = self.fill(source, chunk, prep)
        if pre is None:
            pre = [weights.issue(r, chunk, self.group, None, self.chunk) for r in rows]
        return xs, pre

    def preload(self, weights, rows, first, count: int, mask=None):
        """The weight loads ``run`` stages outside its loop (every chunk when
        ``count`` <= UNROLL, else the first ``depth``), to issue before the
        phase and pass to ``run`` as ``pending``."""
        n = count if count <= UNROLL else min(self.depth, count)
        return [
            [
                weights.issue(r, first + i32(i), self.group, mask, self.chunk)
                for r in rows
            ]
            for i in range(n)
        ]

    def first_group(self, weights, rows, count: int, rot, mask):
        """A grouped ``run``'s first group's weight loads (``rot`` as ``run``'s),
        to pass as its ``pending``."""
        g0 = rot % i32(count // self.depth)
        return [
            [
                weights.issue(
                    r, g0 * i32(self.depth) + i32(j), self.group, mask, self.chunk
                )
                for r in rows
            ]
            for j in range(self.depth)
        ]

    def run(
        self,
        source,
        weights,
        rows,
        first,
        count: int,
        pending=None,
        rot=None,
        ahead=None,
    ):
        """Accumulate chunks ``first .. first + count`` (a trace-time count) into
        one accumulator per (weight tile, token tile); ``rows``: the lane's
        weight row of each weight tile. ``pending``: ``preload``'s loads.
        ``rot`` (a long K only): start at chunk ``rot`` and wrap, so that CTAs
        streaming many rows in step do not all read the same column bytes of
        their rows at once. Grouped only: ``ahead = (weights, rows, live)`` of the
        CTA's next unit, whose first group's weights the last group issues
        (masked by ``live``); returns ``(acc, those loads)``, the next ``run``'s
        ``pending``."""
        pre = list(pending or ())
        pre += [None] * (count - len(pre))
        if const_expr(self.grouped):
            assert count % self.depth == 0, "grouped: whole groups"
            return _grouped_loop(
                self, source, weights, rows, first, count, rot, pending, ahead
            )
        prep = self.prepare(source)
        if count <= UNROLL:
            return self._unrolled(source, weights, rows, first, count, pre, prep)
        return _gemm_loop(self, source, weights, rows, first, count, pre, prep, rot)

    def _unrolled(self, source, weights, rows, first, count: int, pre, prep):
        depth = min(self.depth, count)
        acc = [[zeros(4) for _ in range(self.tiles)] for _ in range(self.mtiles)]
        staged = [
            self.stage(source, weights, rows, first, count, i32(i), prep, pre[i])
            for i in range(depth)
        ]
        self.put(i32(0), staged[0][0], source, prep)
        fx.barrier()
        for c in range(count):
            if c + depth < count:
                nxt = c + depth
                staged.append(
                    self.stage(
                        source, weights, rows, first, count, i32(nxt), prep, pre[nxt]
                    )
                )
            acc = self.multiply(i32(c % 2), staged[c][1], acc)
            if c + 1 < count:
                self.put(i32(1 - c % 2), staged[c + 1][0], source, prep)
            fx.barrier()
        return acc


@traced
def _gemm_loop(gemm, source, weights, rows, first, count, pre, prep, rot=None):
    """``Gemm.run`` for a long K, ``depth`` chunks an iteration: ring slot
    ``j`` stays one set of registers, chunk ``c + j + depth`` issued into it
    once chunk ``c + j``'s MFMAs have read it (a ring rotated through a rolled
    loop's registers moves values whose loads are in flight, which waits for
    them: one chunk ahead whatever the depth). The last ``depth`` chunks
    issued re-read the last chunk; a remainder of ``count % depth`` chunks
    runs after the loop without further loads."""
    depth = gemm.depth
    assert depth % 2 == 0, depth
    whole = count - count % depth
    acc = [[zeros(4) for _ in range(gemm.tiles)] for _ in range(gemm.mtiles)]
    assert rot is None or all(p is None for p in pre), "preloads are unrotated"
    slots = [
        gemm.stage(source, weights, rows, first, count, i32(i), prep, pre[i], rot)
        for i in range(depth)
    ]
    gemm.put(i32(0), slots[0][0], source, prep)
    fx.barrier()
    for c in range(i32(0), i32(whole), i32(depth)):
        nxt = []
        for j in range_constexpr(depth):
            acc = gemm.multiply(i32(j % 2), slots[j][1], acc)
            nxt.append(
                gemm.stage(
                    source,
                    weights,
                    rows,
                    first,
                    count,
                    c + i32(j + depth),
                    prep,
                    rot=rot,
                )
            )
            xs = slots[j + 1][0] if j + 1 < depth else nxt[0][0]
            gemm.put(i32(1 - j % 2), xs, source, prep)
            fx.barrier()
        slots = nxt
    for j in range_constexpr(count - whole):
        acc = gemm.multiply(i32(j % 2), slots[j][1], acc)
        if j + 1 < count - whole:
            gemm.put(i32(1 - j % 2), slots[j + 1][0], source, prep)
        fx.barrier()
    return acc


@traced
def _grouped_loop(
    gemm, source, weights, rows, first, count, rot=None, pending=None, ahead=None
):
    """``Gemm.run(grouped)``: an iteration multiplies one group of ``depth``
    chunks from one LDS buffer while it issues the next group's activations
    (first: the barrier's wait for them is then not behind its weights) and
    weights, each chunk's into its ring slot; then one store and one barrier.
    ``rot`` rotates whole groups; the last group issues ``ahead``'s first."""
    depth = gemm.depth
    groups = count // depth
    prep = gemm.prepare(source)

    def group_at(gi):
        gi = (gi < i32(groups)).select(gi, i32(groups - 1))
        if rot is not None:
            gi = (gi + rot) % i32(groups)
        return gi

    def wload(w, rs, gi, j, mask=None):
        chunk = first + gi * i32(depth) + i32(j)
        return [w.issue(r, chunk, gemm.group, mask, gemm.chunk) for r in rs]

    def multiply(acc, slots, buf):
        for j in range_constexpr(depth):
            acc = gemm.mma_tiles(
                acc, slots[j], lambda t, j=j, buf=buf: gemm.operands(buf, t, j)
            )
        return acc

    g0 = group_at(i32(0))
    xs = gemm.fill(source, first // i32(depth) + g0, prep)
    slots = (
        list(pending)
        if pending
        else [wload(weights, rows, g0, j) for j in range(depth)]
    )
    gemm.put(i32(0), xs, source, prep)
    fx.barrier()
    acc = [[zeros(4) for _ in range(gemm.tiles)] for _ in range(gemm.mtiles)]
    for gi in range(i32(0), i32(groups - 1), i32(1)):
        buf = gi % i32(2)
        gn = group_at(gi + i32(1))
        xs = gemm.fill(source, first // i32(depth) + gn, prep)
        nxt = []
        for j in range_constexpr(depth):
            acc = gemm.mma_tiles(
                acc, slots[j], lambda t, j=j, buf=buf: gemm.operands(buf, t, j)
            )
            nxt.append(wload(weights, rows, gn, j))
            sched_barrier()
        gemm.put(i32(1) - buf, xs, source, prep)
        fx.barrier()
        slots = nxt
    buf = i32((groups - 1) % 2)
    if const_expr(ahead is None):
        return multiply(acc, slots, buf)
    w2, rows2, live = ahead
    nxt = []
    for j in range_constexpr(depth):
        acc = gemm.mma_tiles(
            acc, slots[j], lambda t, j=j, buf=buf: gemm.operands(buf, t, j)
        )
        nxt.append(wload(w2, rows2, g0, j, live))
        sched_barrier()
    return acc, nxt


# --------------------------------------------------------------------------
# split-K plans and their epilogues
# --------------------------------------------------------------------------


class SplitK(TraceInvariant):
    """A GEMM of ``rows`` weight rows over ``k`` columns of ``elem`` bytes cut
    into ``splits`` K slices, one unit per (row group, slice); a row group is
    ``ROWS * mtiles`` rows. Partials index rows from 0 with stride ``padded``."""

    def __init__(self, rows: int, k: int, elem: int, splits: int, mtiles: int = 1):
        chunks = k * elem // CHUNK_BYTES
        if k * elem % CHUNK_BYTES or chunks % splits:
            raise ValueError(f"{k} K of {elem} bytes do not split {splits} ways")
        self.rows, self.k, self.elem, self.splits, self.mtiles = (
            rows,
            k,
            elem,
            splits,
            mtiles,
        )
        self.unit_rows = ROWS * mtiles
        self.groups = -(-rows // self.unit_rows)
        self.per = chunks // splits
        self.padded = self.groups * self.unit_rows

    @property
    def units(self) -> int:
        return self.groups * self.splits

    def partial_words(self, tokens: int) -> int:
        return self.splits * tokens * self.padded


def unit_rows(plan, unit, tid):
    """(group, split, the wave's first row a weight tile, the lane's row a
    weight tile) of ``unit``."""
    wave, lane = tid // i32(LANES), tid % i32(LANES)
    group, split = unit % i32(plan.groups), unit // i32(plan.groups)
    firsts = [
        group * i32(plan.unit_rows) + (i32(mt * WAVES) + wave) * i32(TILE)
        for mt in range(plan.mtiles)
    ]
    return group, split, firsts, [f + lane % i32(TILE) for f in firsts]


def unit_preload(gemm, weights, plan, bid):
    """The weight loads of ``run``'s first chunks of CTA ``bid``'s first unit
    of ``plan``, for ``gemm_units(pending=...)``: issued in an earlier phase
    (weights do not depend on it), so they land while the grid waits. A CTA
    without a unit issues no request."""
    live = bid < i32(plan.units)
    _, split, _, rows = unit_rows(plan, live.select(bid, i32(0)), gemm.tid)
    return gemm.preload(weights, rows, split * i32(plan.per), plan.per, live)


@traced
def _gemm_unit(
    gemm, source, weights, plan, epilogue, after, unit, pending, before=None
):
    group, split, firsts, rows = unit_rows(plan, unit, gemm.tid)
    if const_expr(before is not None):
        before(group, split)
    acc = gemm.run(source, weights, rows, split * i32(plan.per), plan.per, pending)
    for mt in range_constexpr(plan.mtiles):
        epilogue(unit, split, firsts[mt], gemm.lane, acc[mt])
    if const_expr(after is not None):
        after(group, split)


@traced
def gemm_units(
    bid,
    workers,
    gemm,
    source,
    weights,
    plan,
    epilogue,
    after=None,
    pending=None,
    before=None,
):
    """Run every unit of ``plan`` over the grid: CTA ``bid`` takes units
    ``bid, bid + workers, ...``. ``epilogue(unit, split, first_row, lane, acc)``
    finishes a weight tile's accumulators; ``after(group, split)``, when given,
    runs once a unit behind all waves' epilogues (CTA-uniform); ``before(group,
    split)`` ahead of a unit's loads (CTA-uniform, ending in a barrier).
    ``pending``: ``unit_preload``'s loads of the first unit."""
    start = bid
    if const_expr(pending is not None):
        if bid < i32(plan.units):
            _gemm_unit(
                gemm, source, weights, plan, epilogue, after, bid, pending, before
            )
        start = bid + workers
    for unit in range(start, i32(plan.units), workers):
        _gemm_unit(gemm, source, weights, plan, epilogue, after, unit, None, before)


def partial_epilogue(gemm, plan, part):
    """Each unit's accumulators to ``part[(split, token, row)]`` (f32, device
    scope)."""

    @traced
    def epilogue(unit, split, first_row, lane, acc):
        for t in range_constexpr(gemm.tiles):
            token = i32(t * TILE) + lane % i32(TILE)
            row = first_row + lane // i32(TILE) * i32(4)
            if token < i32(gemm.batch):
                st(
                    part,
                    (split * i32(gemm.batch) + token) * i32(plan.padded) + row,
                    acc[t],
                    cm=CM_DEV,
                )

    return epilogue


SUM_BATCH = 8  # splits' loads in flight while summing partials
# 16-byte partial loads a fold keeps in flight: several token tiles of a
# shallow split, ``SUM_BATCH`` splits of one tile of a deep one
FOLD_LOADS = 16


def sum_partials4(part, plan, tokens: int, token, row):
    """Rows ``row .. row + 3`` of ``token`` summed over the splits in split
    order; ``SUM_BATCH`` splits' 16-byte loads are issued before their adds."""
    total = None
    for s0 in range(0, plan.splits, SUM_BATCH):
        loads = [
            ld(
                part,
                (i32(s * tokens) + token) * i32(plan.padded) + row,
                fx.Float32,
                4,
                cm=CM_DEV,
            )
            for s in range(s0, min(s0 + SUM_BATCH, plan.splits))
        ]
        for v in loads:
            total = (
                [v[e] for e in range(4)]
                if total is None
                else [total[e] + v[e] for e in range(4)]
            )
    return total


def fold(g, gemm, plan, part, counter0, finish, done=None):
    """Split-K units whose row group's last arriving unit sums the group's
    partials and finishes them: ``finish(group, first_row, lane, acc)`` with the
    summed accumulators of each weight tile, in the epilogue's layout, then
    ``done(group)`` (CTA-uniform, when given). Returns the (epilogue, after)
    pair of ``gemm_units``; ``counter0`` is the first of the plan's ``groups``
    fold counters."""
    epilogue = partial_epilogue(gemm, plan, part)

    span = max(1, FOLD_LOADS // plan.splits)

    @traced
    def last(group):
        wave, lane = gemm.tid // i32(LANES), gemm.tid % i32(LANES)
        for mt in range_constexpr(plan.mtiles):
            first = group * i32(plan.unit_rows) + (i32(mt * WAVES) + wave) * i32(TILE)
            row = first + lane // i32(TILE) * i32(4)
            tokens = [i32(t * TILE) + lane % i32(TILE) for t in range(gemm.tiles)]
            tokens = [
                (token < i32(gemm.batch)).select(token, i32(gemm.batch - 1))
                for token in tokens
            ]
            acc = []
            if const_expr(plan.splits > FOLD_LOADS):
                for t in range_constexpr(gemm.tiles):
                    acc.append(
                        fx.Vector.from_elements(
                            sum_partials4(part, plan, gemm.batch, tokens[t], row),
                            fx.Float32,
                        )
                    )
            else:
                for t0 in range_constexpr(0, gemm.tiles, span):
                    tiles = range(t0, min(t0 + span, gemm.tiles))
                    loads = [
                        [
                            ld(
                                part,
                                (i32(s * gemm.batch) + tokens[t]) * i32(plan.padded)
                                + row,
                                fx.Float32,
                                4,
                                cm=CM_DEV,
                            )
                            for t in tiles
                        ]
                        for s in range(plan.splits)
                    ]
                    for i in range_constexpr(len(tiles)):
                        total = [loads[0][i][e] for e in range(4)]
                        for s in range_constexpr(1, plan.splits):
                            total = [total[e] + loads[s][i][e] for e in range(4)]
                        acc.append(fx.Vector.from_elements(total, fx.Float32))
            finish(group, first, lane, acc)
        if const_expr(done is not None):
            done(group)

    @traced
    def after(group, split):
        if fold_arrive(g, counter0 + group, plan.splits):
            last(group)

    return epilogue, after
