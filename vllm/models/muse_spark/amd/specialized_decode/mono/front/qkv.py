# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's qkv GEMV (DESIGN_BC 5.5, Phase B1) on the GEMV CTAs b in [NA, BLOCKS),
g = b - NA: this rank's 2304 qkv rows (``qkv_proj``, column-parallel: q 1024
| k 128 | v 128 | og 1024) against the boundary's ``attn_in`` rows.

Stock reference: ``wvSplitK`` (S <= 5) / hipBLASLt (S = 8) / ``F.linear``:
an fp32 dot over K = 8192, rounded to bf16 once. Here the K1 stage hands
fp32 partial rows to the epilogue, which adds the two contributors and
rounds once (``front/epilogue.py``)::

    tasks    tau = 16 T + kp: tile T (16 rows 16 T ..) x K part kp (512
             columns, 4 chunks of 128); CTA g runs tau = 12 g + i, i < 12,
             12 distinct K parts (consecutive mod 16, no wrap inside a tile)
    rounds   r = 0 .. 5: waves 0-3 task 2 r, waves 4-7 task 2 r + 1; wave w
             chunk w % 4 of its task (``gemv.loads``: 4 dwordx4 a lane,
             nontemporal); rounds 0 .. EARLY - 1 issued at entry, the rest
             right after the ``arow`` loads
    B rows   ``arow`` (bf16 [MT, 8192], the boundary's, device scope) of the
             CTA's 12 K parts, S rows each, staged into LDS ``xa`` in one
             batched pass (every load, the late weight rounds, then every
             ds_write, one barrier) behind the ``ardy`` flags of the 48
             slices they cover
    sums     a task's value: its 4 waves' accumulators in wave order
             (``row_sum``); a tile's partial in this CTA: its tasks in K part
             order -> ``qpart`` ((T 2 + c) MT + t) 16 + r, contributor c = g
             - ``contributors(T)[0]`` (0: the lower K parts)

Every tile has exactly ``K1_MAXC`` = 2 contributors (``spec.k1_contributor0``),
each puts its partial of every row of every token, zeros
included, so the epilogue's waits depend on nothing but the GEMV CTAs. The
order (waves, then K parts, then contributors) is fixed and the same on
every rank: ``qkvo`` and the cache are rank-bit-identical.

Rule 8 (DESIGN_BC 10.4): a GEMV CTA polls ``ardy`` holding only its own
weights in flight; it publishes nothing. ``vmcnt`` retires in order, so the
poll's ``vmcnt(0)`` (``_spin``) and the staging pass's ds_writes wait for
every weight load issued before them: only the EARLY rounds (128 of the
CTA's 192 KiB) are, the last rounds stream behind the ``arow`` loads, so the
ds_writes wait on a counted ``vmcnt`` (the late rounds' loads still in
flight) and the poll and staging round trips hide under the stream's tail.

``stages/gemv.py`` is K2's (compiled into every K2 build) and is not edited:
the staging pass and the task sums are here.
"""

import functools

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    ld_dev,
    mfma_bf16,
    row_sum,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    THREADS,
    cdiv,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    AROW,
    COLS,
    HIDDEN,
    K1_MAXC,
    K1_ROUNDS,
    K1_TPC,
    KPART,
    KPARTS,
    ROWS,
    WAVES,
    ardy_pair,
    k1_contributor0,
    qpart_pair,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import gemv

STAGE = "qkv"
XA_ROW = KPART + gemv.LDS_PAD  # bf16 a padded LDS K part row (520)
X4 = KPART // 8  # dwordx4 a K part row (64)
CHUNKS = KPART // gemv.KCH  # 4 chunks a K part: one a wave of a half
SLICES = KPART // COLS  # boundary slices (``ardy`` flags) a K part covers
EARLY = 4  # weight rounds issued at entry; the rest after the arow loads
assert CHUNKS * 2 == WAVES and K1_ROUNDS * 2 == K1_TPC
assert SLICES * K1_TPC <= THREADS  # a flag a polling thread
assert 0 < EARLY < K1_ROUNDS


def tau_of(g, i):
    """Task i of GEMV CTA g."""
    return K1_TPC * g + i


@functools.cache
def lds_struct(s: int):
    """The stage's LDS: the CTA's 12 K parts of rows 0 .. s-1 of ``arow``,
    task i's row t at ``(i s + t) XA_ROW`` bf16 (99,840 B at s = 8). Cached:
    one struct type a width."""

    @fx.struct
    class QkvLds:
        xa: fx.Array[fx.Int32, K1_TPC * s * XA_ROW // 2, 16]

    return QkvLds


def contributors(tile):
    """(the first GEMV CTA g whose tasks cover ``tile``, the contributors a
    tile has): the tile's K parts are split between g (the lower ones) and
    g + 1. Python ints (host) or traced Int32 (in-kernel) alike."""
    return k1_contributor0(tile), K1_MAXC


def task(g, i):
    """GEMV CTA g's task i: (tile T, K part kp)."""
    tau = tau_of(g, i)
    return tau // KPARTS, tau % KPARTS


def last_of_tile(g, i):
    """Whether task i is the last of its tile in CTA g (its partial is
    complete): i == K1_TPC - 1, or task i + 1 starts the next tile. True (a
    Python bool) for i == K1_TPC - 1, else a traced Boolean (Python bool for
    an int g)."""
    if i == K1_TPC - 1:
        return True
    return (tau_of(g, i) + 1) % KPARTS == 0


def qkv_loads(c, g, rounds) -> list:
    """GEMV CTA g's weights of ``rounds``: round r's task 2 r + h (h = wave
    // 4), chunk wave % 4 of its K part, 4 dwordx4 a lane (nontemporal),
    every round in flight (6: 96 VGPRs). The EARLY rounds are the CTA's first
    loads, the rest are issued by ``stage_rows``. A scheduling barrier after
    them keeps every load where it is issued (LLVM would sink the late rounds
    to their MFMAs)."""
    lane, wave = c["lane"], c["wave"]
    w4 = wave % CHUNKS
    half = wave // CHUNKS
    w = c["args"]["w_qkv"]
    ops = []
    for r in rounds:
        tile, kp = task(g, 2 * r + half)
        ops.append(
            gemv.loads(lane, w4, w, tile * ROWS, KPART, ldk=HIDDEN, k0=kp * KPART)[0]
        )
    rocdl.sched_barrier(0)
    return ops


def _stage_ij(v, s):
    """A staging element v -> (task i, token t, dwordx4 x of the K part)."""
    return v // (s * X4), v // X4 % s, v % X4


def stage_rows(c, g) -> list:
    """The one staging pass: the ``arow`` rows (device scope: the boundary
    CTAs of this launch wrote them) of the CTA's 12 K parts -> LDS ``xa``, 16
    B a thread a round (12 S 64 dwordx4: 12 rounds at S = 8), every round's
    load issued before the first ds_write, then one barrier. The K parts are
    (12 g + i) % 16, derived from g in-kernel. Between the loads and the
    ds_writes it issues the late weight rounds (EARLY ..) and returns them:
    younger than the ``arow`` loads, they stay in flight through the
    ds_writes and the barrier.

    Not ``ops.batched_rounds``: its last round's store is predicated (v <
    n), so LLVM sinks that round's load into the branch, behind the earlier
    rounds' stores (two round trips at S = 1). Here a round past the end
    loads and stores element n - 1 again (the same bytes to the same LDS
    word), so nothing is predicated and every load stays ahead."""
    s, tid = c["S"], c["tid"]
    arow = c["scr"][AROW]
    xa = c["lds"][STAGE]["xa"]
    total = K1_TPC * s * X4
    vs = [fx.min(tid + THREADS * i, total - 1) for i in range(cdiv(total, THREADS))]
    got = []
    for v in vs:
        i, t, x = _stage_ij(v, s)
        kp = tau_of(g, i) % KPARTS
        word = (t * HIDDEN + kp * KPART + 8 * x) // 2
        got.append(fx.Vector(ld_dev(arow, word, 4)))
    rocdl.sched_barrier(0)
    late = qkv_loads(c, g, range(EARLY, K1_ROUNDS))
    for v, w in zip(vs, got):
        i, t, x = _stage_ij(v, s)
        fx.ptr_store(w, xa + ((i * s + t) * XA_ROW + 8 * x) // 2)
    gpu.barrier()
    return late


@traced
def _poll_ardy(c, g):
    """Thread i < 48: the ``ardy`` flag of slice 4 kp + i % 4
    of the CTA's task i // 4's K part kp; then a barrier."""
    tid = c["tid"]
    if tid < SLICES * K1_TPC:
        kp = tau_of(g, tid // SLICES) % KPARTS
        c["poll"]([(c["scr"]["ardy"], ardy_pair(SLICES * kp + tid % SLICES), 1)])
    gpu.barrier()


def _mfmas(c, r, ops):
    """Round r: this wave's 16x16 accumulator of its task's chunk (4 MFMAs of
    32 K against the staged rows of tokens 0 .. S-1)."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    xa = c["lds"][STAGE]["xa"]
    w4 = wave % CHUNKS
    half = wave // CHUNKS
    t = fx.min(lane % ROWS, s - 1)
    row = (2 * r + half) * s + t
    kk = w4 * gemv.KCH + lane // ROWS * 8
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        xb = fx.Vector(
            fx.ptr_load(
                xa + (row * XA_ROW + kk + 32 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        acc = mfma_bf16(ops[q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    return acc


@traced
def stage_qkv(c, g, ops) -> None:
    """GEMV CTA g: the ``ardy`` polls (its EARLY weight rounds ``ops`` in
    flight) -> the staging pass (which issues the late rounds) -> the rounds
    of MFMAs; thread (t, r) = (tid // 16, tid % 16), t < S, adds each task's
    row r of token t to the tile's partial and puts it to ``qpart`` when the
    CTA's last task of the tile is in.
    Ends with a barrier (``red`` and ``xa`` handed over)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    qpart = c["scr"]["qpart"]
    _poll_ardy(c, g)
    ops = ops + stage_rows(c, g)
    r16 = tid % ROWS
    t = tid // ROWS
    tc = fx.min(t, s - 1)  # threads past S read token S - 1's (in bounds)
    live = t < s
    zero = fx.Float32(0.0)
    part = zero
    for r in range_constexpr(K1_ROUNDS):
        acc = _mfmas(c, r, ops[r])
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        for h in range_constexpr(2):
            i = 2 * r + h
            part = part + row_sum(red, r16, tc, range(CHUNKS * h, CHUNKS * (h + 1)))
            tile, _ = task(g, i)
            last = last_of_tile(g, i)
            put = live if last is True else live & last
            if put:
                pair = qpart_pair(tile, g - contributors(tile)[0], t, r16)
                c["put"](qpart, pair, part)
            part = zero if last is True else fx.Float32(last.select(zero, part))
        gpu.barrier()
