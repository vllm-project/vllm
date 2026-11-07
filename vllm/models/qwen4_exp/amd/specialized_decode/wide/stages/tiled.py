# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-tiled GEMMs on the bf16 matrix unit.

A CTA computes ``C[token, row] = sum_k X[token, k] * W[row, k]`` for 64 x
``mtiles`` weight rows (16-row tiles, one a wave per ``mtiles`` step) over a
slice of K and every row of the step. Activations stream through LDS in
``CHUNK``-wide K steps shared by the four waves, double buffered; weights go
from global memory straight to the A operand registers, ``DEPTH`` chunks
ahead. The step's rows are the N dimension of ``v_mfma_f32_16x16x32_bf16``:
``ceil(S / 16)`` token tiles reuse each weight load, and the staged
activations never grow past two chunks. Up to ``RESIDENT_MAX`` rows a unit
instead stages its whole K slice of the real rows once and keeps every
chunk's weights in flight, with no barrier per chunk (``Gemm._resident``);
its first weights may be issued before the phase's grid wait (``pending``,
``issue_unit``), and a 16-row tile may share its K slice among the four
waves (``SplitK(kwaves=True)``, ``Gemm.run_kwaves``).

The MFMA operand ABI (explicit, it is the instruction's): lane ``l`` of a wave
holds weight row ``l % 16`` and, for K slice ``m`` (0..3) of a chunk, the 8
K elements at ``koff(m, l // 16)``; its accumulator of token tile ``t`` holds
rows ``4 (l // 16) + e`` (e = 0..3) of token ``16 t + l % 16``. A weight
source picks ``koff`` so a lane's loads are contiguous in its format: bf16
rows take ``32 m + 8 g`` (a load instruction's four lanes of a row read 64
contiguous bytes), MXFP4 rows ``32 g + 8 m`` (a lane's one 16-byte load is
its 32 K values). The B operand is read from LDS at the same ``koff``.

A split-K plan (``SplitK``) cuts a GEMM too narrow to fill the grid along K;
its partials ``[split, token, row]`` (f32) are summed in split order by a
later phase or by the row group's last arriving unit (``fold``).
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    TraceInvariant,
    fold_arrive,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_NT,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    mfma_bf16,
    st,
    traced,
    zeros,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BLOCKS,
    THREADS,
    WAVES,
)

LANES = 64
TILE = 16
SLICES = 4  # MFMA K slices of a chunk
CHUNK = 128  # K elements a staged step
ROWS = WAVES * TILE  # weight rows a CTA covers per weight tile step
XSTRIDE = CHUNK // 2 + 4  # LDS row stride in i32 words: 16 pad bytes stagger the banks
DEPTH = 2  # chunks of loads in flight (2 beat 1 and 3 at 128 rows)
UNROLL = 4  # K chunks up to which the chunk loop is unrolled
RESIDENT_MAX = 16  # widths whose GEMM units stage their whole K slice once
# widths past RESIDENT_MAX up to which a phase's weights are loaded in an
# earlier phase (``Gemm.preload``): wider, the held registers and the drain of
# the loads at that phase's barrier cost more than the preload hides (measured
# at 128 rows)
PRELOAD_ROWS = 96


def token_tiles(rows: int) -> int:
    return -(-rows // TILE)


def resident(rows: int) -> bool:
    return rows <= RESIDENT_MAX


def resident_stride(chunks: int) -> int:
    """i32 words of a resident stage row of ``chunks`` K chunks; 16 pad bytes
    stagger the rows over the banks."""
    return chunks * CHUNK // 2 + 4


def stage_words(rows: int, chunks: int = 1) -> int:
    """i32 words of a GEMM's LDS stage: two chunk buffers of every padded row,
    or, at a resident width, ``chunks`` K chunks of every real row if larger."""
    words = 2 * token_tiles(rows) * TILE * XSTRIDE
    if resident(rows):
        words = max(words, rows * resident_stride(chunks))
    return words


def i32(v) -> fx.Int32:
    return fx.Int32(v)


# --------------------------------------------------------------------------
# weight sources (the A operand)
# --------------------------------------------------------------------------


class Bf16Weights(TraceInvariant):
    """A bf16 ``[rows, cols]`` row-major matrix at byte address ``addr``."""

    def __init__(self, addr, rows: int, cols: int, cm: int = CM_NT):
        self.addr, self.rows, self.cols, self.cm = fx.Int64(addr), rows, cols, cm

    @staticmethod
    def koff(m, g):
        return i32(32 * m) + g * i32(8)

    def issue(self, row, chunk, g, mask=None):
        """The lane's four 16-byte loads of chunk ``chunk``; a row past ``rows``
        reads the last row, a column past ``cols`` loads as zero. With
        ``mask`` false, no request (zeros)."""
        row = (row < i32(self.rows)).select(row, i32(self.rows - 1))
        out = []
        for m in range(SLICES):
            k = chunk * i32(CHUNK) + self.koff(m, g)
            at = (row * i32(self.cols) + k) // i32(2)  # i32 words
            if self.cols % CHUNK:
                live = k < i32(self.cols)
                w = self._ld(live.select(at, i32(0)), mask)
                out.append(live.select(w, zeros(4, fx.Int32)))
            else:
                out.append(self._ld(at, mask))
        return out

    def _ld(self, at, mask):
        if mask is None:
            return ld_i32x4(self.addr, at, self.cm)
        nbytes = self.rows * self.cols * 2
        return ld(self.addr, at, fx.Int32, 4, cm=self.cm, nbytes=nbytes, mask=mask)

    @staticmethod
    def operands(loaded):
        return [fx.Vector(w).bitcast(fx.BFloat16) for w in loaded]


class Fp4Weights(TraceInvariant):
    """An MXFP4 ``[rows, cols]`` matrix (e2m1, one E8M0 scale per 32 K) decoded to
    bf16 operands in registers. ``shuffled``: AITER's ASM-GEMM layout (data
    through ``shuffle_weight(layout=(16, 16))``, scales through
    ``e8m0_shuffle``); else the Triton FP4 GEMM's (data row-major ``[rows,
    cols / 2]``, scales transposed ``[cols / 32, rows]``)."""

    def __init__(
        self, addr, scale, rows: int, cols: int, shuffled: bool, cm: int = CM_NT
    ):
        if cols % CHUNK:
            raise ValueError(f"MXFP4 K {cols} is not whole {CHUNK}-column steps")
        self.addr, self.scale = fx.Int64(addr), fx.Int64(scale)
        self.rows, self.cols, self.shuffled, self.cm = rows, cols, shuffled, cm
        self.scale_cols = (cols // 32 + 7) // 8 * 8  # e8m0_shuffle pads K / 32 to 8

    @staticmethod
    def koff(m, g):
        return g * i32(32) + i32(8 * m)

    def _data_word(self, row, chunk, g):
        if self.shuffled:
            # a 16-row group's 16-byte K pieces as [piece][row % 16][16 B]
            at = (
                (row // i32(TILE)) * i32(TILE * self.cols // 2)
                + (chunk * i32(4) + g) * i32(256)
                + (row % i32(TILE)) * i32(16)
            )
            return at // i32(4)
        return (
            row * i32(self.cols // 2) + chunk * i32(CHUNK // 2) + g * i32(16)
        ) // i32(4)

    def _scale_byte(self, row, kb):
        if self.shuffled:
            r32, a, b = row // i32(32), (row // i32(16)) % i32(2), row % i32(16)
            c8, d, f = kb // i32(8), (kb // i32(4)) % i32(2), kb % i32(4)
            return (
                (((r32 * i32(self.scale_cols // 8) + c8) * i32(4) + f) * i32(16) + b)
                * i32(2)
                + d
            ) * i32(2) + a
        return kb * i32(self.rows) + row

    def issue(self, row, chunk, g, mask=None):
        """The lane's 16-byte data load and its scale byte of chunk ``chunk``;
        with ``mask`` false, no request (zeros)."""
        row = (row < i32(self.rows)).select(row, i32(self.rows - 1))
        at = self._data_word(row, chunk, g)
        if mask is None:
            data, nbytes = ld_i32x4(self.addr, at, self.cm), None
        else:
            dbytes = self.rows * self.cols // 2
            data = ld(self.addr, at, fx.Int32, 4, cm=self.cm, nbytes=dbytes, mask=mask)
            nbytes = self.rows * (self.scale_cols if self.shuffled else self.cols // 32)
        kb = self._scale_byte(row, chunk * i32(4) + g)
        code = ld(self.scale, kb, fx.Uint8, cm=self.cm, nbytes=nbytes, mask=mask)
        return [data, code]

    @staticmethod
    def operands(loaded):
        data, code = loaded
        scale = (fx.Int32(code) << i32(23)).bitcast(fx.Float32)
        out = []
        for m in range(SLICES):
            pairs = []
            for byte in range(4):
                p = rocdl.cvt_scalef32_pk_bf16_fp4(
                    T.vec(2, T.bf16),
                    fx.Int32(data[m]).ir_value(),
                    scale.ir_value(),
                    byte,
                )
                pairs.append(fx.Vector(p).bitcast(fx.Int32)[0])
            out.append(fx.Vector.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16))
        return out


# --------------------------------------------------------------------------
# activation sources (the B operand, staged through LDS)
# --------------------------------------------------------------------------


class Source(TraceInvariant):
    """Activations ``X[token, k]``: ``issue(token, k)`` returns the raw loads of 8
    elements (as stored, no arithmetic), ``convert(raw)`` their 8 bf16 as four
    packed i32 words. ``fill`` issues a chunk a K step ahead and ``put``
    converts after the step's MFMAs, so the wait for the loads lands behind the
    matrix work. Default: a bf16 ``[tokens, ld]`` matrix at ``addr``."""

    def __init__(self, addr=None, ld_elems: int = 0, cm: int = CM_DEV):
        self.addr, self.ld, self.cm = (
            (None if addr is None else fx.Int64(addr)),
            ld_elems,
            cm,
        )

    def issue(self, token, k):
        return [ld_i32x4(self.addr, (token * i32(self.ld) + k) // i32(2), self.cm)]

    def issue_batch(self, coords):
        """``issue`` of every (token, k) of ``coords``: a resident stage's
        loads of one thread."""
        return [self.issue(token, k) for token, k in coords]

    def convert(self, raw):
        return raw[0]


class Gemm(TraceInvariant):
    """The staged-activation K loop for one step width ``rows`` (the tokens).
    ``page``: the LDS stage (``stage_words(rows, chunks)`` i32 words),
    ``k_total``: K when it is not a multiple of ``CHUNK`` (activations past it
    stage as 0). At a resident width a unit's K slice (at most ``chunks``
    chunks) of the real rows is staged once (``_resident``)."""

    def __init__(
        self,
        page,
        rows: int,
        tid,
        k_total=None,
        mtiles: int = 1,
        depth: int = DEPTH,
        chunks: int = 1,
    ):
        self.page = page
        self.batch = rows
        self.tiles = token_tiles(rows)
        self.tid = i32(tid)
        self.lane = self.tid % i32(LANES)
        self.group = self.lane // i32(TILE)
        self.mtiles = mtiles
        self.depth = depth
        self.k_total = k_total if k_total is not None and k_total % CHUNK else None
        self.resident = resident(rows)
        self.chunks = chunks

    # staging: thread v of a token tile round holds token v // 16, columns
    # 8 (v % 16) .. + 7 of the chunk
    def fill(self, source, chunk):
        out = []
        for t in range(self.tiles):
            v = i32(t * THREADS) + self.tid
            token = v // i32(CHUNK // 8)
            if (t + 1) * THREADS // (CHUNK // 8) > self.batch:
                token = (token < i32(self.batch)).select(token, i32(self.batch - 1))
            k = chunk * i32(CHUNK) + (v % i32(CHUNK // 8)) * i32(8)
            if self.k_total is None:
                out.append(source.issue(token, k))
            else:
                live = k < i32(self.k_total)
                out.append(list(source.issue(token, live.select(k, i32(0)))) + [live])
        return out

    def put(self, buf, items, source):
        for t in range(self.tiles):
            v = i32(t * THREADS) + self.tid
            at = (buf * i32(self.tiles * TILE) + v // i32(CHUNK // 8)) * i32(
                XSTRIDE
            ) + (v % i32(CHUNK // 8)) * i32(4)
            item = items[t]
            if self.k_total is None:
                words = source.convert(item)
            else:
                words = item[-1].select(
                    fx.Vector(source.convert(item[:-1])), zeros(4, fx.Int32)
                )
            lds_st(self.page + at, fx.Vector(words))

    def operand(self, buf, t, m, koff):
        at = (
            (buf * i32(self.tiles) + i32(t)) * i32(TILE) + self.lane % i32(TILE)
        ) * i32(XSTRIDE) + koff(m, self.group) // i32(2)
        return lds_ld(self.page + at, 4).bitcast(fx.BFloat16)

    def operands(self, buf, t, koff):
        return [self.operand(buf, t, m, koff) for m in range(SLICES)]

    def run_kwaves(self, source, weights, row, first, count: int, pending=None):
        """One 16-row weight tile (the lane's row ``row``) over chunks ``first
        .. first + count`` of a resident stage, wave ``w`` multiplying chunks
        ``w count / 4 ..`` of them; the waves' accumulators are then added in
        wave order through the stage. Returns the sums, one a token tile, the
        same in every wave. ``pending``: the wave's weights (``issue_unit``)."""
        if not self.resident or self.mtiles != 1:
            raise ValueError("K waves need a resident stage of one weight tile")
        if count > self.chunks:
            raise ValueError(f"{count} K chunks past the stage's {self.chunks}")
        per = count // WAVES
        wave = self.tid // i32(LANES)
        mine = wave * i32(per)
        items = self.fill_resident(source, first, count)
        loaded = pending
        if loaded is None:
            loaded = issue_chunks(weights, [row], first + mine, per, self.group)
        self.put_resident(items, source, count)
        fx.barrier()
        acc = [[zeros(4) for _ in range(self.tiles)]]
        for c in range(per):
            a = [weights.operands(w) for w in loaded[c]]
            acc = self.mma_tiles(
                acc,
                a,
                lambda t, c=c: self.operands_resident(mine + c, count, t, weights.koff),
            )
        fx.barrier()
        # [wave][token tile][lane] 16-byte accumulators over the stage
        slot = lambda w, t: (i32(w * self.tiles + t) * i32(LANES) + self.lane) * i32(4)  # noqa: E731
        for t in range(self.tiles):
            lds_st(self.page + slot(wave, t), acc[0][t])
        fx.barrier()
        out = []
        for t in range(self.tiles):
            parts = [
                lds_ld(self.page + slot(w, t), 4, fx.Float32) for w in range(WAVES)
            ]
            total = [parts[0][e] for e in range(4)]
            for w in range(1, WAVES):
                total = [total[e] + parts[w][e] for e in range(4)]
            out.append(fx.Vector.from_elements(total, fx.Float32))
        fx.barrier()
        return out

    def multiply(self, buf, a, acc, weights):
        """``acc`` plus the chunk in LDS buffer ``buf`` times the weight
        operands ``a`` ([mtile][slice])."""
        return self.mma_tiles(acc, a, lambda t: self.operands(buf, t, weights.koff))

    def stage(self, source, weights, rows, first, count: int, i, pre=None):
        """The activation and weight loads of chunk ``first + min(i, count - 1)``;
        ``pre``: its weight loads, already issued (``preload``)."""
        chunk = first + (i < i32(count)).select(i, i32(count - 1))
        if pre is None:
            pre = [weights.issue(r, chunk, self.group) for r in rows]
        return self.fill(source, chunk), pre

    def preload(self, weights, rows, first, count: int, mask=None):
        """The weight loads ``run`` stages outside its loop (every chunk when
        ``count`` <= UNROLL, else the first ``depth``), to issue before the
        phase (weights do not depend on the phase before) and pass to ``run``
        as ``pending``."""
        n = count if count <= UNROLL else min(self.depth, count)
        return [
            [weights.issue(r, first + i32(i), self.group, mask) for r in rows]
            for i in range(n)
        ]

    def run(self, source, weights, rows, first, count: int, pending=None):
        """Accumulate chunks ``first .. first + count`` (a trace-time count) into
        one accumulator per (weight tile, token tile); ``rows``: the lane's
        weight row of each weight tile. ``pending``: the unit's weights already
        in flight, at the resident widths every chunk's (``issue_chunks``),
        else ``preload``'s of the first chunks."""
        if self.resident:
            return self._resident(source, weights, rows, first, count, pending)
        pre = list(pending or ())
        pre += [None] * (count - len(pre))
        if count <= UNROLL:
            return self._unrolled(source, weights, rows, first, count, pre)
        return _gemm_loop(self, source, weights, rows, first, count, pre)

    # resident stage: row ``token`` of the real rows at ``token *
    # resident_stride(count)`` words, its K slice contiguous
    def _resident(self, source, weights, rows, first, count: int, pending):
        """The unit's activations first, then the weights of every chunk, then
        the activations to LDS behind one barrier, then each chunk's MFMAs,
        waiting only for its own weights (``vmcnt`` counts in issue order: the
        activations land first, the weights stream in under the MFMAs). A
        closing barrier frees the stage for the next unit."""
        if count > self.chunks:
            raise ValueError(f"{count} K chunks past the stage's {self.chunks}")
        items = self.fill_resident(source, first, count)
        loaded = pending
        if loaded is None:
            loaded = issue_chunks(weights, rows, first, count, self.group)
        self.put_resident(items, source, count)
        fx.barrier()
        acc = [[zeros(4) for _ in range(self.tiles)] for _ in range(self.mtiles)]
        for c in range(count):
            a = [weights.operands(w) for w in loaded[c]]
            acc = self.mma_tiles(
                acc, a, lambda t, c=c: self.operands_resident(c, count, t, weights.koff)
            )
        fx.barrier()
        return acc

    def _resident_vectors(self, count: int):
        """This thread's 8-element vectors of a resident stage: ``tid, tid +
        THREADS, ...``; past the last one a thread repeats it (the same value to
        the same address)."""
        n = self.batch * count * (CHUNK // 8)
        out = []
        for r in range(-(-n // THREADS)):
            v = i32(r * THREADS) + self.tid
            if (r + 1) * THREADS > n:
                v = (v < i32(n)).select(v, i32(n - 1))
            out.append(v)
        return out

    def fill_resident(self, source, first, count: int):
        per_row = count * (CHUNK // 8)
        coords, lives = [], []
        for v in self._resident_vectors(count):
            token = v // i32(per_row)
            k = first * i32(CHUNK) + v % i32(per_row) * i32(8)
            if self.k_total is not None:
                lives.append(k < i32(self.k_total))
                k = lives[-1].select(k, i32(0))
            coords.append((token, k))
        raws = source.issue_batch(coords)
        if self.k_total is None:
            return raws
        return [list(raw) + [live] for raw, live in zip(raws, lives)]

    def put_resident(self, items, source, count: int):
        per_row = count * (CHUNK // 8)
        for v, item in zip(self._resident_vectors(count), items):
            at = v // i32(per_row) * i32(resident_stride(count)) + v % i32(
                per_row
            ) * i32(4)
            if self.k_total is None:
                words = source.convert(item)
            else:
                words = item[-1].select(
                    fx.Vector(source.convert(item[:-1])), zeros(4, fx.Int32)
                )
            lds_st(self.page + at, fx.Vector(words))

    def operands_resident(self, c, count: int, t: int, koff):
        """Token tile ``t``'s four slice operands of chunk ``c`` of a resident
        stage of ``count`` chunks (a token past the step reads the last row)."""
        token = i32(t * TILE) + self.lane % i32(TILE)
        if (t + 1) * TILE > self.batch:
            token = (token < i32(self.batch)).select(token, i32(self.batch - 1))
        at = token * i32(resident_stride(count)) + i32(c) * i32(CHUNK // 2)
        return [
            lds_ld(self.page + at + koff(m, self.group) // i32(2), 4).bitcast(
                fx.BFloat16
            )
            for m in range(SLICES)
        ]

    def mma_tiles(self, acc, a, operands):
        """``acc`` plus the weight operands ``a`` ([mtile][slice]) times the B
        operands ``operands(t)`` of each token tile; tile ``t + 1``'s are read
        while tile ``t``'s MFMAs run."""
        acc = [list(r) for r in acc]
        ops = operands(0)
        for t in range(self.tiles):
            ahead = operands(t + 1) if t + 1 < self.tiles else None
            for m in range(SLICES):
                for mt in range(self.mtiles):
                    acc[mt][t] = mfma_bf16(a[mt][m], ops[m], acc[mt][t])
            ops = ahead
        return acc

    def _unrolled(self, source, weights, rows, first, count: int, pre):
        depth = min(self.depth, count)
        acc = [[zeros(4) for _ in range(self.tiles)] for _ in range(self.mtiles)]
        staged = [
            self.stage(source, weights, rows, first, count, i32(i), pre[i])
            for i in range(depth)
        ]
        self.put(i32(0), staged[0][0], source)
        fx.barrier()
        for c in range(count):
            if c + depth < count:
                nxt = c + depth
                staged.append(
                    self.stage(source, weights, rows, first, count, i32(nxt), pre[nxt])
                )
            a = [weights.operands(w) for w in staged[c][1]]
            acc = self.multiply(i32(c % 2), a, acc, weights)
            if c + 1 < count:
                self.put(i32(1 - c % 2), staged[c + 1][0], source)
            fx.barrier()
        return acc


@traced
def _gemm_loop(gemm, source, weights, rows, first, count, pre):
    """``Gemm.run`` for a long K: chunk ``c + depth`` issued while chunk ``c``
    multiplies; the last ``depth`` iterations re-issue the last chunk."""
    depth = gemm.depth
    acc = [[zeros(4) for _ in range(gemm.tiles)] for _ in range(gemm.mtiles)]
    staged = [
        gemm.stage(source, weights, rows, first, count, i32(i), pre[i])
        for i in range(depth)
    ]
    a = staged[0][1]
    gemm.put(i32(0), staged[0][0], source)
    ring_xs = [s[0] for s in staged[1:]]
    ring_a = [s[1] for s in staged[1:]]
    fx.barrier()
    for c in range(i32(0), i32(count), i32(1)):
        new_xs, new_a = gemm.stage(source, weights, rows, first, count, c + i32(depth))
        buf = c % i32(2)
        acc = gemm.multiply(buf, [weights.operands(w) for w in a], acc, weights)
        gemm.put(i32(1) - buf, (ring_xs + [new_xs])[0], source)
        fx.barrier()
        a = (ring_a + [new_a])[0]
        ring_xs = (ring_xs + [new_xs])[1:]
        ring_a = (ring_a + [new_a])[1:]
    return acc


# --------------------------------------------------------------------------
# split-K plans and their epilogues
# --------------------------------------------------------------------------


class SplitK(TraceInvariant):
    """A GEMM of ``rows`` weight rows over ``k`` columns cut into ``splits`` K
    slices, one unit per (row group, slice); a row group is ``ROWS * mtiles``
    rows, or with ``kwaves`` one 16-row tile whose K slice the four waves
    share (``Gemm.run_kwaves``). Partials index rows from 0 with stride
    ``padded``."""

    def __init__(
        self, rows: int, k: int, splits: int, mtiles: int = 1, kwaves: bool = False
    ):
        chunks = -(-k // CHUNK)
        if chunks % splits:
            raise ValueError(f"{chunks} K chunks do not split {splits} ways")
        if kwaves and (mtiles != 1 or chunks // splits % WAVES):
            raise ValueError(f"{chunks // splits} K chunks a unit over {WAVES} waves")
        self.rows, self.k, self.splits, self.mtiles = rows, k, splits, mtiles
        self.kwaves = kwaves
        self.unit_rows = TILE if kwaves else ROWS * mtiles
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
    if plan.kwaves:
        wave = i32(0)
    firsts = [
        group * i32(plan.unit_rows) + (i32(mt * WAVES) + wave) * i32(TILE)
        for mt in range(plan.mtiles)
    ]
    return group, split, firsts, [f + lane % i32(TILE) for f in firsts]


def issue_chunks(weights, rows, first, count: int, group):
    """The weight loads of chunks ``first .. first + count`` of a lane's
    ``rows`` (one a weight tile), ``[chunk][weight tile]``."""
    return [
        [weights.issue(r, first + i32(c), group) for r in rows] for c in range(count)
    ]


def issue_unit(bid, tid, weights, plan):
    """The weights of CTA ``bid``'s first unit of ``plan`` (``gemm_units``'s
    ``pending``), every chunk in flight (with ``kwaves`` the wave's share);
    a CTA past the last unit loads the last unit's and drops them. Weights
    are immutable, so they may be issued before the phase's grid wait."""
    unit = (i32(bid) < i32(plan.units)).select(i32(bid), i32(plan.units - 1))
    _, split, _, rows = unit_rows(plan, unit, i32(tid))
    group = i32(tid) % i32(LANES) // i32(TILE)
    first = split * i32(plan.per)
    if plan.kwaves:
        per = plan.per // WAVES
        return issue_chunks(weights, rows, first + i32(tid) // LANES * per, per, group)
    return issue_chunks(weights, rows, first, plan.per, group)


def unit_preload(gemm, weights, plan, bid):
    """Past ``RESIDENT_MAX``, the weight loads of ``run``'s first chunks
    (``Gemm.preload``) of CTA ``bid``'s unit of ``plan``, for
    ``gemm_units(pending=...)``: issued in an earlier phase (weights do not
    depend on it), so they land while the grid waits. ``plan`` has at most one
    unit a CTA; a CTA without one issues no request."""
    assert plan.units <= BLOCKS, (plan.units, BLOCKS)
    live = bid < i32(plan.units)
    _, split, _, rows = unit_rows(plan, live.select(bid, i32(0)), gemm.tid)
    return gemm.preload(weights, rows, split * i32(plan.per), plan.per, live)


@traced
def _gemm_unit(gemm, source, weights, plan, epilogue, after, unit, pending):
    group, split, firsts, rows = unit_rows(plan, unit, gemm.tid)
    acc = gemm.run(source, weights, rows, split * i32(plan.per), plan.per, pending)
    for mt in range_constexpr(plan.mtiles):
        epilogue(unit, split, firsts[mt], gemm.lane, acc[mt])
    if const_expr(after is not None):
        after(group, split)


@traced
def gemm_units(
    bid, workers, gemm, source, weights, plan, epilogue, after=None, pending=None
):
    """Run every unit of ``plan`` over the grid: CTA ``bid`` takes units
    ``bid, bid + workers, ...``. ``epilogue(unit, split, first_row, lane, acc)``
    finishes a weight tile's accumulators; ``after(group, split)``, when given,
    runs once a unit behind all waves' epilogues (CTA-uniform). ``pending``:
    the first unit's weights, ``issue_unit``'s at the resident widths,
    ``unit_preload``'s at the others."""
    start = bid
    if const_expr(pending is not None):
        if bid < i32(plan.units):
            _gemm_unit(gemm, source, weights, plan, epilogue, after, bid, pending)
        start = bid + workers
    for unit in range(start, i32(plan.units), workers):
        _gemm_unit(gemm, source, weights, plan, epilogue, after, unit, None)


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


def sum_partials4(part, plan, tokens: int, token, row):
    """Rows ``row .. row + 3`` of ``token`` summed over the splits in split
    order; every split's 16-byte load is issued before the first add."""
    loads = [
        ld(
            part,
            (i32(s * tokens) + token) * i32(plan.padded) + row,
            fx.Float32,
            4,
            cm=CM_DEV,
        )
        for s in range(plan.splits)
    ]
    total = [loads[0][e] for e in range(4)]
    for s in range(1, plan.splits):
        total = [total[e] + loads[s][e] for e in range(4)]
    return total


def fold(g, gemm, plan, part, counter0, finish, done=None, other=None, begin=None):
    """Split-K units whose row group's last arriving unit sums the group's
    partials and finishes them: ``finish(group, first_row, lane, acc)`` with the
    summed accumulators of each weight tile, in the epilogue's layout, then
    ``done(group)`` (CTA-uniform, when given). ``begin(group, first_row,
    lane)`` (when given) runs on the last arriver before each tile's partial
    loads, to issue loads ``finish`` needs alongside them. With two splits,
    ``other(group)`` (CTA-uniform, when given) runs on the group's first
    arriver instead. Returns the (epilogue, after) pair of ``gemm_units``;
    ``counter0`` is the first of the plan's ``groups`` fold counters."""
    assert other is None or plan.splits == 2, plan.splits
    epilogue = partial_epilogue(gemm, plan, part)

    @traced
    def last(group):
        wave, lane = gemm.tid // i32(LANES), gemm.tid % i32(LANES)
        for mt in range_constexpr(plan.mtiles):
            first = group * i32(plan.unit_rows) + (i32(mt * WAVES) + wave) * i32(TILE)
            row = first + lane // i32(TILE) * i32(4)
            if const_expr(begin is not None):
                begin(group, first, lane)
            acc = []
            for t in range_constexpr(gemm.tiles):
                token = i32(t * TILE) + lane % i32(TILE)
                token = (token < i32(gemm.batch)).select(token, i32(gemm.batch - 1))
                acc.append(
                    fx.Vector.from_elements(
                        sum_partials4(part, plan, gemm.batch, token, row),
                        fx.Float32,
                    )
                )
            finish(group, first, lane, acc)
        if const_expr(done is not None):
            done(group)

    @traced
    def after(group, split):
        if fold_arrive(g, counter0 + group, plan.splits):
            last(group)

    @traced
    def after_other(group, split):
        if fold_arrive(g, counter0 + group, plan.splits):
            last(group)
        else:
            other(group)

    return epilogue, (after if other is None else after_other)
