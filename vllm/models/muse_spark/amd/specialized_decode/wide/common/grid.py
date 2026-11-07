# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the Qwen3.8-Flash-Next wide decode's ``wide/common/grid.py``
# (256 CTAs) for the 304 CUs of an MI300X.
"""A launch's identity and its synchronization: the grid barrier between
phases and the split-K fold's last-arriver count, both behind a bounded wait.

Grid barrier (striped): a CTA adds one to its arrival stripe, one of
``STRIPES`` counters on separate 256-byte lines. The 38 CTAs of an XCD
(``bid % 8``) share its 8 stripes, by ``(bid / 8) % 8``: stripes 0-5 of an XCD
take 5 CTAs, 6-7 take 4 (``stripe_ctas``). Wave 0's 64 lanes poll every stripe
at once, one load each, against the lane's own count, and a ballot says when
all reached it. The counters are a launch slot's: the k-th barrier of a launch
waits for ``k * stripe_ctas`` on every stripe, so each launch must start from
zeroed counters. Launches alternate between two slots, and each zeroes the slot
the next one uses (the launch before it is done by then): ``sync`` is this
launch's slot, ``sync_next`` the next one's. Every CTA must take every barrier
of a launch: the kernels run one CTA per CU, all co-resident.

Every wait gives up after ``WAIT_TICKS``: the CTA records the site in its
health record, raises its rank's flag on the output rank (``alarm``) and sets
the rank's FAILED word, so its later waits, and every other CTA's, exit at
once and the launch runs to its end. The host reads FAILED after the step.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.wide.abi import (
    FAILED,
    RECORD_WORDS,
    SITE_FOLD,
    SITE_GRID,
    WAIT_TICKS,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    CM_SYS,
    atomic_add,
    drain_stores,
    ld,
    memrealtime,
    st,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
    BLOCKS,
    CUS_PER_XCD,
    WAVES,
    XCDS,
)

STRIPES_PER_XCD = 8
STRIPES = XCDS * STRIPES_PER_XCD  # 64: wave 0's lanes poll them all at once
STRIPE_WORDS = 64  # one 256-byte line a counter
SLOT_WORDS = STRIPES * STRIPE_WORDS  # a launch slot's barrier counters
FOLD_BASE = 2 * SLOT_WORDS  # the fold counters follow both slots' stripes
FOLD_STRIDE = 16  # a fold counter's words: one 64-byte line
FOLD_COUNTERS = 4096
assert STRIPES <= 64


def stripe_ctas(s: int) -> int:
    """CTAs that arrive on stripe ``s``."""
    j = s % STRIPES_PER_XCD
    return len(range(j, CUS_PER_XCD, STRIPES_PER_XCD))


assert sum(stripe_ctas(s) for s in range(STRIPES)) == BLOCKS


def sync_words() -> int:
    """i32 words of a rank's sync buffer: two launch slots, then the fold
    counters."""
    return FOLD_BASE + FOLD_COUNTERS * FOLD_STRIDE


class TraceInvariant:
    """A handle never reassigned inside traced control flow: FlyDSL carries any
    local that is subscripted or has a method called in a dynamic loop or
    branch, and these reference only values defined before it, so they carry
    no IR values and rebuild as themselves."""

    def __extract_to_ir_values__(self):
        return []

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar):
        return exemplar


class Grid(TraceInvariant):
    """The CTA's identity and its synchronization words.

    ``sync`` / ``sync_next``: this launch's barrier slot and the next launch's
    (byte addresses); ``folds``: the fold counters; ``health``: the rank's
    health words; ``alarm``: this rank's flag word on the output rank (0:
    none); ``scratch``: an LDS word the CTA's threads share verdicts in."""

    def __init__(self, tid, bid, sync, sync_next, folds, health, alarm, scratch):
        self.tid = fx.Int32(tid)
        self.bid = fx.Int32(bid)
        self.lane = self.tid % fx.Int32(64)
        self.wave = self.tid // fx.Int32(64)
        self.sync = fx.Int64(sync)
        self.sync_next = fx.Int64(sync_next)
        self.folds = fx.Int64(folds)
        self.health = fx.Int64(health)
        self.alarm = fx.Int64(alarm)
        self.scratch = scratch
        self.barriers = 0  # trace-time count of the launch's grid barriers
        self.xcd = self.bid % fx.Int32(XCDS)
        self.stripe = self.xcd * fx.Int32(STRIPES_PER_XCD) + (
            self.bid // fx.Int32(XCDS)
        ) % fx.Int32(STRIPES_PER_XCD)
        # a phase of few independent items takes item i on CTA i % BLOCKS,
        # wave i // BLOCKS: one wave a CU first, so a CU's memory pipe queues
        # one wave's loads where four waves of a quarter of the CUs did.
        # ``mirror``: CTA BLOCKS - 1 - i % BLOCKS, wave WAVES - 1 - i // BLOCKS,
        # for a phase that shares the CTAs with another. Loop
        # ``range(g.spread, items, BLOCKS * WAVES)``.
        self.spread = self.wave * fx.Int32(BLOCKS) + self.bid
        self.mirror = (fx.Int32(WAVES - 1) - self.wave) * fx.Int32(BLOCKS) + (
            fx.Int32(BLOCKS - 1) - self.bid
        )


@traced
def give_up(g, site, seen, target):
    """A wait that timed out: this CTA's record (once), the alarm, FAILED."""
    rec = (g.bid + 1) * RECORD_WORDS
    if ld(g.health, rec, fx.Int32, cm=CM_SYS) == 0:
        at = fx.Int32(memrealtime() & fx.Int64(0x7FFFFFFF))
        words = [fx.Int32(1), fx.Int32(site), fx.Int32(seen), fx.Int32(target)]
        st(g.health, rec, fx.Vector.from_elements(words, fx.Int32), cm=CM_SYS)
        tail = [g.bid, at, fx.Int32(0), fx.Int32(0)]
        st(g.health, rec + 4, fx.Vector.from_elements(tail, fx.Int32), cm=CM_SYS)
    if g.alarm != fx.Int64(0):
        st(g.alarm, 0, fx.Int32(1), cm=CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    st(g.health, FAILED, fx.Int32(1), cm=CM_SYS)


def failed(g):
    return ld(g.health, FAILED, fx.Int32, cm=CM_SYS) != fx.Int32(0)


@traced
def prologue(g):
    """The launch's first act: CTA 0 zeroes the next launch's barrier slot."""
    if (g.bid == 0) & (g.tid < STRIPES):
        st(g.sync_next, g.tid * STRIPE_WORDS, fx.Int32(0), cm=CM_DEV)


def _lane_ctas(lane):
    """``stripe_ctas`` of the stripe lane ``lane`` polls."""
    j = lane % fx.Int32(STRIPES_PER_XCD)
    full = CUS_PER_XCD % STRIPES_PER_XCD  # stripes with one CTA more
    base = CUS_PER_XCD // STRIPES_PER_XCD
    return (j < fx.Int32(full)).select(fx.Int32(base + 1), fx.Int32(base))


@traced
def _poll_stripes(g, k: int):
    """Wave 0: poll every stripe until all reached ``k`` arrivals of each of
    their CTAs, bounded. A rank that failed before this barrier does not wait;
    one that fails while this CTA waits lets it time out on its own (the poll
    loop stays one load)."""
    lane_word = g.lane * STRIPE_WORDS  # the descriptor base stays wave-uniform
    target = fx.Int32(k) * _lane_ctas(g.lane)
    t0 = memrealtime()
    seen = ld(g.sync, lane_word, fx.Int32, cm=CM_DEV)
    behind = fx.Int64(rocdl.ballot(T.i64, (seen < target).ir_value()))
    why = failed(g).select(fx.Int32(2), fx.Int32(0))
    while (behind != fx.Int64(0)) & (why == 0):
        seen = ld(g.sync, lane_word, fx.Int32, cm=CM_DEV)
        behind = fx.Int64(rocdl.ballot(T.i64, (seen < target).ir_value()))
        late = memrealtime() - t0 > fx.Int64(WAIT_TICKS)
        why = late.select(fx.Int32(1), fx.Int32(0))
    if (why == 1) & (g.lane == 0):
        give_up(g, SITE_GRID, seen, target)


def grid_sync(g, between=None):
    """Every CTA's stores so far visible to every CTA, then all CTAs past this
    point: drain, CTA barrier, one arrival on the CTA's stripe, wave 0 polls
    all stripes for this barrier's count. ``between()``, when given, runs
    after the arrival and before the wait: work that reads no other CTA's
    stores and stores nothing another CTA reads, such as loads of data no CTA
    writes (the next phase's weights) or LDS work of the CTA's own; its
    result is returned. Wave 0's poll waits for its loads (``vmcnt`` counts
    in issue order)."""
    g.barriers += 1
    _arrive(g)
    out = None if between is None else between()
    _wait(g, g.barriers)
    return out


@traced
def _arrive(g):
    drain_stores()
    fx.barrier()
    if g.tid == 0:
        atomic_add(g.sync + fx.Int64(g.stripe * (STRIPE_WORDS * 4)), 1)


@traced
def _wait(g, k: int):
    if uniform(g.wave) == 0:
        _poll_stripes(g, k)
    fx.barrier()
    fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)


@traced
def fold_arrive(g, counter, arrivals):
    """A split-K unit's arrival at fold counter ``counter`` after its partials:
    True on the CTA that arrives last of ``arrivals``, which resets the counter
    for the next launch."""
    drain_stores()
    fx.barrier()
    if g.tid == 0:
        addr = g.folds + fx.Int64(counter * (FOLD_STRIDE * 4))
        old = atomic_add(addr, 1)
        last = old == fx.Int32(arrivals - 1)
        if last:
            st(addr, 0, fx.Int32(0), cm=CM_DEV)
        fx.ptr_store(last.select(fx.Int32(1), fx.Int32(0)), g.scratch)
    fx.barrier()
    verdict = fx.ptr_load(g.scratch) != fx.Int32(0)
    fx.barrier()
    return verdict


@traced
def wait_count(g, addr, target):
    """Wave 0 waits until the i32 at ``addr`` reaches ``target`` (a counter other
    CTAs bump), bounded; then the CTA."""
    if uniform(g.wave) == 0:
        t0 = memrealtime()
        seen = ld(addr, 0, fx.Int32, cm=CM_DEV)
        why = failed(g).select(fx.Int32(2), fx.Int32(0))
        while (seen < target) & (why == 0):
            seen = ld(addr, 0, fx.Int32, cm=CM_DEV)
            late = memrealtime() - t0 > fx.Int64(WAIT_TICKS)
            why = late.select(fx.Int32(1), fx.Int32(0))
        if (why == 1) & (g.lane == 0):
            give_up(g, SITE_FOLD, seen, target)
    fx.barrier()
    fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)
