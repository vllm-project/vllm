# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/mono/device/select.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A CTA's exact top-k for a small k, by counting: entry e of an LDS array is a
signed 64-bit sort key (a 32-bit key high, the entry's id low: key, then id,
descending), and an entry's rank is the count of entries that beat it -- no
order of arrival enters, so every CTA and rank agrees.

Counting every entry against every other is VALU-bound (one 64-bit compare an
entry pair). Past a few hundred entries, ``group_floor`` and ``compact`` first
narrow the field to the entries that can still be among the top k, and only
those are counted against each other.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from ..plan.execution import (
    WAVES,
)
from .ops import (
    butterfly,
    traced,
)

KEY_LEAST = -(2**31)  # the 32-bit key of a dead entry: below every live one
BATCH = 16  # entries every lane reads a round of ``count_beaten``


def put_key(keys, e, key, idx, live):
    """Entry e of ``keys`` (an Int32 LDS pointer, two words an entry: id, key);
    a dead one the least, beating nothing. -> this entry's 64-bit sort key."""
    fx.ptr_store(live.select(idx, fx.Int32(0)), keys + 2 * e)
    fx.ptr_store(live.select(key, fx.Int32(KEY_LEAST)), keys + (2 * e + 1))
    return (fx.Int64(key) << 32) | fx.Int64(idx)


@traced
def count_beaten(keys, mines, m):
    """For each of up to two 64-bit sort keys ``mines``, how many of the first
    m entries of ``keys`` beat it. Entries up to m rounded up to BATCH must
    hold ``put_key``'s (a dead one beats nothing). Every lane reads the same
    entry (an LDS broadcast), BATCH a round so the reads overlap, and a
    compare is one instruction; the counts share one register, 16 bits each,
    so the dynamic loop carries one value."""
    assert len(mines) <= 2, "two 16-bit counts a register"
    v4i = fx.Vector.make_type(4, fx.Int32)
    packed = fx.Int32(0)
    for j0 in range(0, m, BATCH):
        j0 = fx.Int32(j0)
        pairs = [
            fx.Vector(fx.ptr_load(keys + (2 * j0 + 4 * q), result_type=v4i)).bitcast(
                fx.Int64
            )
            for q in range(BATCH // 2)
        ]
        for q in range_constexpr(BATCH // 2):
            for e in range_constexpr(2):
                other = fx.Int64(pairs[q][e])
                for i, mine in enumerate(mines):
                    packed = packed + (other > mine).select(1 << (16 * i), 0)
    return [(packed >> (16 * i)) & 0xFFFF for i in range(len(mines))]


@traced
def group_floor(keys32, k, red, lane, wave):
    """A lower bound of the k-th largest of the CTA's 32-bit keys (each
    thread's ``keys32``, dead ones ``KEY_LEAST``): k disjoint groups -- the
    entries of the lanes with one ``lane % k`` -- each group's largest key, the
    least of those; k distinct entries are at least it. Live entries packed
    from entry 0 (k of them at least) leave no group empty: a group of
    contiguous lanes would hold only dead ones past the live, and its least key
    makes the floor no floor. ``red``: WAVES * k + 1 words of LDS. Barriers
    included."""
    assert k in (1, 2, 4, 8, 16, 32, 64), k
    top = keys32[0]
    for v in keys32[1:]:
        top = fx.max(top, v)
    top = butterfly(top, tuple(o for o in (32, 16, 8, 4, 2, 1) if o >= k), fx.max)
    if lane < k:
        fx.ptr_store(top, red + (wave * k + lane))
    gpu.barrier()
    if (wave == 0) & (lane < k):
        g = fx.ptr_load(red + lane)
        for w in range_constexpr(1, WAVES):
            g = fx.max(g, fx.ptr_load(red + (w * k + lane)))
        g = butterfly(g, tuple(o for o in (32, 16, 8, 4, 2, 1) if o < k), fx.min)
        if lane == 0:
            fx.ptr_store(g, red + WAVES * k)
    gpu.barrier()
    return fx.ptr_load(red + WAVES * k)


@traced
def compact(cand, counter, cap, entries, floor):
    """The entries of ``entries`` ((key32, id, live) a thread) that can be in
    the top k -- live, key at least ``floor`` (ties kept: a superset) -- to
    ``cand`` as ``put_key`` entries, an LDS slot each from ``counter`` (zeroed,
    and ``cand`` filled with dead entries, before the barrier ``group_floor``
    holds). Slots past ``cap`` are dropped; the caller reads the count after a
    barrier and counts every entry instead when it is past ``cap``."""
    for key, idx, live in entries:
        if live & (key >= floor):
            slot = fx.llvm.atomic_add(counter, fx.Int32(1), syncscope="workgroup")
            if slot < cap:
                put_key(cand, slot, key, idx, live)
