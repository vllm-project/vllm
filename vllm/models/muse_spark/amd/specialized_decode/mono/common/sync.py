# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
"""The tagged mailbox: hand-offs between the CTAs of one GPU (device scope) and
between GPUs (system scope), every address a named region (``Addr``).

Muse Spark: a launch's tag is ``step_tag(e, slot)`` of the 22-bit step epoch,
and every wait's budget is the ``WAIT_BUDGET`` health word (seconds), read once
a launch (``load_budget``), instead of the constant ``WAIT_TICKS``."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    FAILED,
    RECORD_WORDS,
    WAIT_BUDGET,
    WAIT_SECONDS,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    ld_i32,
    ld_s32,
    memrealtime,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import Space
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    EPOCH_MASK,
    TAG_SLOTS,
)

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)

TICKS_PER_SECOND = 100_000_000  # s_memrealtime runs at 100 MHz
WAIT_TICKS = WAIT_SECONDS * TICKS_PER_SECOND


def step_tag(e, slot):
    """A launch's mailbox tag: the step epoch ``e`` (the runner's device
    counter, bumped once a step in the graph) masked to 22 bits, and the
    launch's ``slot``, so no step's or other launch's pair ever passes for this
    one's -- nothing is zeroed. ``&``, not a signed ``%``: a wrapped (negative)
    int32 epoch still gives a valid positive tag. Never 0."""
    return (e & EPOCH_MASK) * TAG_SLOTS + slot + 1


def load_epoch(epoch):
    """The step epoch word at ``epoch``, a scalar load (``ld_s32``): the runner
    bumps it before the launch. ``step_epoch`` masks it."""
    return ld_s32(epoch, 0)


def step_epoch(w):
    """The step epoch of its word ``w`` (``load_epoch``), masked to 22 bits
    (its parity ``e & 1`` alternates across the mask's wrap: 2^22 is even)."""
    return w & EPOCH_MASK


def load_budget(health):
    """This rank's ``WAIT_BUDGET`` health word, read once a launch, a scalar
    load (``ld_s32``): the runner sets it between launches, no MonoKernel
    writes it. ``wait_budget`` reads it."""
    return ld_s32(health, WAIT_BUDGET)


def wait_budget(w):
    """An in-kernel wait's budget in seconds from its word ``w``
    (``load_budget``). A zeroed header (health words reset without a budget)
    falls back to ``WAIT_SECONDS`` instead of failing every wait that is not
    satisfied at its first load."""
    return fx.Int32((w > 0).select(w, fx.Int32(WAIT_SECONDS)))


@traced
def _spin(load_all, pending, health, alarm, region_id, index, tag, ticks):
    """Re-issue ``load_all`` until ``pending`` clears, for at most ``ticks``
    ``s_memrealtime`` ticks (the launch's ``WAIT_BUDGET``), and not at all once
    this rank has failed (``health`` word FAILED), so a failed step runs to its
    end instead of hanging. A side-effecting op in the loop keeps the loads from
    being hoisted.

    A CTA's first give-up is recorded in its ``health`` record; it then raises the
    output rank's flag for this rank (``alarm``) and drains that store before it
    sets FAILED, so no later store of this rank's can overtake the flag."""
    t0 = memrealtime()
    v = load_all()
    why = fx.Int32(0)  # 1: timed out, 2: the rank had failed
    while pending(v) & (why == 0):
        rocdl.s_sleep(1)
        v = load_all()
        failed = ld_health(health, FAILED) != 0
        late = memrealtime() - t0 > ticks
        why = failed.select(fx.Int32(2), late.select(fx.Int32(1), fx.Int32(0)))
    if why == 1:
        bid = fx.block_idx.x
        rec = (bid + 1) * RECORD_WORDS
        if ld_health(health, rec) == 0:
            at = fx.Int32(memrealtime() & 0x7FFFFFFF)
            record = [1, region_id, index, tag, v[1], bid, at, 0]
            for h in range_constexpr(2):  # 16 B a store
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in record[4 * h : 4 * h + 4]], fx.Int32
                    ),
                    rsrc(health),
                    rec + 4 * h,
                    cache_modifier=CM_SYS,
                )
        bo.buffer_store(fx.Int32(1), rsrc(alarm), 0, cache_modifier=CM_SYS)
        rocdl.s_waitcnt(vmcnt=0)
        bo.buffer_store(fx.Int32(1), rsrc(health), FAILED, cache_modifier=CM_SYS)
    return v


@traced
def stall(fault_rank, fault_epoch, fault_slot, ticks, rank, epoch, slot):
    """The fault hook (tests only): CTA 0 of rank ``fault_rank`` stalls ``ticks``
    once, in the launch for slot ``fault_slot`` of the step at epoch
    ``fault_epoch`` (the raw epoch word)."""
    if (
        (rank == fault_rank)
        & (uniform(ld_i32(epoch, 0)) == fault_epoch)
        & (slot == fault_slot)
        & (fx.block_idx.x == 0)
    ):
        t0 = memrealtime()
        now = memrealtime()
        while now - t0 < ticks:
            rocdl.s_sleep(127)
            now = memrealtime()


def ld_health(health, i):
    return fx.Int32(
        bo.buffer_load(rsrc(health), i, vec_width=1, dtype=T.i32, cache_modifier=CM_SYS)
    )


@dataclass(frozen=True)
class Addr:
    """A mailbox region's address: the region's name and space, and the traced
    address. The space sets the hand-off's scope: device for SCRATCH, system
    for PEER."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def preg(base, offset, region):
    """Peer-symmetric region ``region`` at byte ``offset`` of a rank's buffer
    ``base``."""
    return Addr(region, Space.PEER, base + fx.Int64(offset))


def shift(a, delta):
    """``a`` moved by ``delta`` bytes (a traced or Python integer), same region."""
    return Addr(a.region, a.space, a.value + delta)


def uniform_addr(a):
    """``a`` with its address re-read from the first lane (``v_readfirstlane``):
    a wave-uniform base that is a NEW value. Poll a region through it right
    after a lane-predicated put to the same region (``if lane == 0: put``):
    otherwise LLVM merges the put's buffer resource with the poll's (GVN-PRE),
    phis it at the divergent join into VGPRs, and every poll load becomes a
    readfirstlane waterfall loop. Only for a base that is already wave-uniform
    (a scratch region, this rank's own peer region): lane 0's value is kept."""
    v = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(a.value).ir_value()))
    return Addr(a.region, a.space, v)


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


class Mailbox:
    """Tagged-pair hand-off between CTAs (device scope) and GPUs (system scope).

    Every 32-bit value is stored next to this launch's tag; a consumer
    polls the payload until all tags match, so a hand-off costs one round trip
    (no store drain, no separate flag). The tag is ``tag + 1``, the launch's
    ``step_tag(e, slot)`` (``Mailbox(step_tag(e, slot) - 1, ...)``): it carries
    the step epoch the runner bumps once a step and the launch's slot, so a pair
    from an earlier step or another launch never carries a live tag, and no
    mailbox is ever zeroed. Every address is an ``Addr``: its space picks the
    scope.
    """

    def __init__(self, tag, health, alarm, region_ids, budget):
        """``tag``: this launch's tag - 1; ``health``: this rank's health words
        (FAILED, then a wait record a CTA); ``alarm``: this rank's flag word on
        the output rank; ``region_ids``: region name -> the id a record names it
        by; ``budget``: a wait's budget in seconds (``wait_budget``)."""
        self.tag = tag + 1
        self.health = health
        self.alarm = alarm
        self.region_ids = region_ids
        self.ticks = fx.Int64(budget) * TICKS_PER_SECOND

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_bf(self, a, i, vs):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs, one
        store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(a.value),
            i,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_words(self, a, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag). Integer payloads
        (fp8 words) must come this way: through an f32 value a NaN bit pattern is
        not kept."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def poll(self, specs, batch=POLL_MAX):
        """Batched poll of pairs ``specs`` = [(Addr, pair index, npairs in {1, 2})],
        all of one space.

        All pairs are re-loaded together while any tag is stale, so a batch costs
        one round trip after its last producer lands. Returns the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        cm = _SCOPE_CM[spaces.pop()]
        bases = [a.value for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, (_, i, n) in zip(bases, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        rid = self.region_ids.get(specs[0][0].region, -1)
        v = _spin(
            load_all,
            pending,
            self.health,
            self.alarm,
            rid,
            specs[0][1],
            tag,
            self.ticks,
        )
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


@traced
def publish(put, a, i, v, who):
    """This CTA's plain stores made readable by every CTA, then (thread ``who``)
    the mailbox pair (``a``, i) := v. The stores are drained first: a flag may
    not overtake the data it announces, and a workgroup-scope release fence does
    not wait for them (LLVM's AMDGPU memory model omits vmcnt outside tgsplit
    mode: the workgroup's waves share one CU, the readers here do not)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(a, i, v)
