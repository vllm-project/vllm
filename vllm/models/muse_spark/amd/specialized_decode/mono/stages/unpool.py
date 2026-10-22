# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 4b ``unpool`` / ``gather``: the unpooler (``F.linear(moe,
w_unpool)``, [S, 4096] x [8192, 4096]^T) split by output rows over the ranks,
then the branch all-gather -> ``branch_out``.

Rank r computes the branch columns 1024 r .. (``w_unpool`` rows 1024 r ..,
the whole K = 4096, the stock per-element math: one fp32
dot over K and one bf16 rounding). CTA b of every rank takes 16 rows ``g = b //
4`` and a K quarter ``q = b % 4``:

    unpool  the ``moe`` rows' K quarter (polled: the owners' all-gather) ->
            LDS; a 16-row bf16 GEMV (``gemv``); q > 0 pushes its fp32 row sums
            to ``upart``; q = 0 adds the three in order (((q0 + q1) + q2) + q3),
            rounds once to bf16 and pushes the row pair to every rank's
            ``branch``
    gather  CTA j < 64: columns 128 j .. of every row from ``branch`` (each
            rank's 1024 columns) -> ``branch_out``, pad rows 0

The weight rows are issued (``unpool_loads``) before the stage waits on any
``moe`` pair: a static weight, independent of the step.
"""

import functools

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    ld_pad,
    row_sum,
    rsrc,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    COLS,
    HIDDEN,
    MOE_HIDDEN,
    ROWS,
    TP,
    U_K,
    U_PARTS,
    branch_pair,
    moe_pair,
    upart_pair,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import gemv

U_N = HIDDEN // TP  # 1024 branch columns a rank computes
U_XROW = U_K + gemv.LDS_PAD  # bf16 a token row in LDS (padded)
GATHER_LANES = COLS // 2  # 64 threads a row in the gather: 2 columns each
assert U_K // 2 == gemv.THREADS  # a moe row's K quarter: a pair a thread


@functools.cache
def lds_struct(s: int):
    """``xl``: the moe rows' K quarter (bf16 [s, U_K + 8], as i32 words)."""

    @fx.struct
    class UnpoolLds:
        xl: fx.Array[fx.Int32, s * U_XROW // 2, 16]

    return UnpoolLds


# Index helpers: plain integer arithmetic on Python ints and traced values
# alike (the CPU tests check the protocol with the same formulas).


def task(b):
    """CTA b's unpooler task: (row group g, K quarter q)."""
    return b // U_PARTS, b % U_PARTS


def row0(rank, g):
    """The first ``w_unpool`` row (= branch column) of rank ``rank``'s group g."""
    return rank * U_N + g * ROWS


def branch_elem(t, col):
    """``branch`` element index (``put_bf``) of row t, column col: 2
    ``spec.branch_pair`` for an even col."""
    return t * HIDDEN + col


def gather_col(j, lane):
    """The first of the two columns gather CTA j's lane ``lane`` moves."""
    return j * COLS + 2 * lane


def unpool_loads(c, b):
    """CTA b's weight rows: 16 rows of ``w_unpool`` from ``row0``, K quarter q
    (one 128-K chunk a wave, 4 dwordx4)."""
    g, q = task(b)
    return gemv.loads(
        c["lane"],
        c["wave"],
        c["args"]["w_unpool"],
        row0(c["rank"], g),
        U_K,
        ldk=MOE_HIDDEN,
        k0=U_K * q,
    )


@traced
def stage_unpool(c, b, ops):
    """CTA b: the moe rows' K quarter q -> LDS, the GEMV against ``ops``
    (``unpool_loads``), then q > 0 -> ``upart``, q = 0 -> the four parts in
    order, bf16 -> every rank's ``branch``. Ends with a barrier."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    xl = c["lds"]["unpool"]["xl"]
    g, q = task(fx.Int32(b))
    n0 = row0(c["rank"], g)
    # the moe rows (every owner rank's slice, as one region): K pair tid of
    # each row, one poll batch a thread
    got = c["poll"](
        [(c["peer"]["moe_own"], moe_pair(t, U_K * q) + tid, 1) for t in range(s)]
    )
    for t in range_constexpr(s):
        fx.ptr_store(got[t][0], xl + (t * (U_XROW // 2) + tid))
    gpu.barrier()
    acc = gemv.mfmas(lane, wave, xl, U_XROW, U_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # q > 0: the fp32 row sums of this K quarter -> the group's q = 0 CTA
    if (q > 0) & (tid < s * ROWS):
        tt = tid // ROWS
        r = tid % ROWS
        c["put"](c["scr"]["upart"], upart_pair(g, q, tt, r), row_sum(red, r, tt))
    # q = 0: + the three other quarters in order, one bf16 rounding
    if (q == 0) & (tid < s * (ROWS // 2)):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        parts = c["poll"](
            [
                (c["scr"]["upart"], upart_pair(g, qq, tt, 2 * rp + h), 1)
                for h in range(2)
                for qq in range(1, U_PARTS)
            ]
        )
        vs = []
        for h in range_constexpr(2):
            v = row_sum(red, 2 * rp + h, tt)
            for qq in range_constexpr(U_PARTS - 1):
                v = v + parts[h * (U_PARTS - 1) + qq][0].bitcast(fx.Float32)
            vs.append(v)
        pad_t = ld_pad(c["args"]["pad"], tt)
        v0 = pad_t.select(fx.Float32(0.0), vs[0])
        v1 = pad_t.select(fx.Float32(0.0), vs[1])
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["branch"][p], branch_elem(tt, n0 + 2 * rp), [v0, v1])
    gpu.barrier()  # the LDS hand-over (xl, red)


@traced
def stage_gather(c, j):
    """Gather CTA j < 64: columns 128 j .. of every row from this rank's
    ``branch`` (the ranks' unpooler pushes) -> ``branch_out`` (bf16, plain
    stores), pad rows 0. Touches no LDS."""
    s, tid = c["S"], c["tid"]
    # thread (row t = tid // 64 < S, lane l): the first S waves; the others
    # poll nothing
    if tid < s * GATHER_LANES:
        t = tid // GATHER_LANES
        col = gather_col(j, tid % GATHER_LANES)
        w = c["poll"]([(c["peer"]["branch_own"], branch_pair(t, col), 1)])[0][0]
        pad_t = ld_pad(c["args"]["pad"], t)
        w = fx.Int32(pad_t.select(fx.Int32(0), w))
        bo.buffer_store(w, rsrc(c["args"]["branch_out"]), branch_pair(t, col))
