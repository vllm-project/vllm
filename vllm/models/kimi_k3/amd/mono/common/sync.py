# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tickets, counters and flags between the launch's workgroups.

Control-word updates branch on the wave-uniform ``wave == 0``, never on
``tid == 0``: LLVM tail-merges identical divergent blocks across the barriers
that follow them, which deadlocks the workgroup. Only lane 0 issues an update
(~0.4 us on gfx950; all 64 lanes adding 1 or 0 costs ~2.5 us), so each call
site passes its own LDS slot and no two of those divergent blocks are identical.

A release either fences (L2 writeback) or, when everything it guards was
written through to memory (``ops.st_wt``), only drains the stores. An acquire
either fences or, for words that guard written-through data nothing in this
launch read before, only drops the CU's L1.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import l1_invalidate, lds_i32, st_wt
from vllm.models.kimi_k3.amd.mono.common.plan import SPIN_SLEEP


@comm_ops.traced
def spin_ge(addr_i64, val, sleep, eq=False):
    """Spin until the word reaches val (eq: equals val)."""
    cur = fx.Int32(comm_ops.load_i32_global_agent(addr_i64))
    if const_expr(eq):
        while cur != fx.Int32(val):
            rocdl.s_sleep(sleep)
            cur = fx.Int32(comm_ops.load_i32_global_agent(addr_i64))
    else:
        while cur < fx.Int32(val):
            rocdl.s_sleep(sleep)
            cur = fx.Int32(comm_ops.load_i32_global_agent(addr_i64))
    return cur


@comm_ops.traced
def _fetch_add(lane, addr, slot):
    """Lane 0 adds one to a control word and leaves the old value in slot[0]."""
    if lane == fx.Int32(0):
        slot[0] = fx.Int32(comm_ops.atomic_add_agent(addr, fx.Int32(1)))


@comm_ops.traced
def count(wave, lane, lds_base, addr, slot_word, release=True):
    """Release, add one; every thread gets the old value."""
    slot = lds_i32(lds_base, slot_word)
    if const_expr(not release):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if wave == fx.Int32(0):
        if const_expr(release):
            comm_ops.fence_agent_release()
        _fetch_add(lane, addr, slot)
    gpu.barrier()
    return rocdl.readfirstlane(T.i32, fx.Int32(slot[0]))


@comm_ops.traced
def grab(wave, lane, lds_base, a_ticket, slot_word):
    """Next ticket for the whole workgroup."""
    slot = lds_i32(lds_base, slot_word)
    gpu.barrier()
    if wave == fx.Int32(0):
        _fetch_add(lane, a_ticket, slot)
    gpu.barrier()
    return rocdl.readfirstlane(T.i32, fx.Int32(slot[0]))


@comm_ops.traced
def bump(wave, lane, addr, release=True):
    """Release, then add one to a control word."""
    if const_expr(not release):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if wave == fx.Int32(0):
        if const_expr(release):
            comm_ops.fence_agent_release()
        if lane == fx.Int32(0):
            comm_ops.atomic_add_agent(addr, fx.Int32(1))


@comm_ops.traced
def raise_flag(wave, addr, val, release=True):
    """Release, then store val to a control word."""
    if const_expr(not release):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if wave == fx.Int32(0):
        if const_expr(release):
            comm_ops.fence_agent_release()
        st_wt(addr, fx.Int32(0), fx.Int32(val), 4)


@comm_ops.traced
def wait_ge(wave, addr, val, l1_only=False, sleep=SPIN_SLEEP, eq=False):
    """Spin until a control word reaches val (eq: equals val), then acquire.

    l1_only drops only this CU's L1. That is enough when everything the word
    guards was written through before it was raised and nothing in this launch
    read those lines earlier: the dispatch already invalidated L2.
    """
    if wave == fx.Int32(0):
        spin_ge(addr, val, sleep, eq)
        if const_expr(l1_only):
            rocdl.s_waitcnt(vmcnt=0)
            l1_invalidate()
        else:
            comm_ops.fence_agent_acquire()
    gpu.barrier()
