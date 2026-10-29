# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A step's beginning (``DESIGN_v3`` I1), one launch: every mailbox buffer
zeroed, then the peer fence -- no rank pushes a step's first pair into a peer's
buffer before that peer has zeroed it.

A TP collective between the zeroing and the first push is no fence: aiter's
custom all-reduce skips the communication in a graph capture's warmup forward.
So each rank writes its step epoch into its slot at every rank's buffer tail
(never zeroed; the epoch only grows) and waits until all of its own slots reach
it. The waits are FlyDSL's: Triton 3.7 compiled this poll into a loop whose one
reloading lane was masked off, a fence that could spin for ever.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.mistral_large4.amd.specialized_decode.mono.common.device.ops import (
    CM_DEV,
    CM_SYS,
    ballot,
    kernel_symbol,
    launcher,
    load_ptr64,
    memrealtime,
    rsrc,
    traced,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.device.sync import (
    DIAG_TICKS,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.plan.build_key import (
    key_tuple,
    source_digest,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.plan.execution import (
    BLOCKS,
    THREADS,
)

_CURRENT_STREAM = fx.Stream(None)

# int64 epoch slots at a peer buffer's tail, one a source rank
FENCE_SLOTS = 8
FENCE_BYTES = FENCE_SLOTS * 8
# a runner's scratch buffers a step zeroes, at most (its peer data besides)
MAX_SCRATCH = 2
# a lane zeroes 16 B a store: every buffer's size is a multiple
ZERO_ALIGN = 16
# the step begin's own int64 words (``state``): its epoch, a debug build's
# missing ranks, then an int32 flag a CTA (the epoch it zeroed its peer data at)
EPOCH, MISSING = 0, 1
ARRIVED = MISSING + FENCE_SLOTS
STATE_WORDS = ARRIVED + BLOCKS // 2
# a wave reads every CTA's flag in one load
assert BLOCKS == 4 * 64


def _ld64(buf, i, cm=0):
    """Int64 ``i`` of buffer resource ``buf``."""
    w = fx.Vector(
        bo.buffer_load(buf, i * 2, vec_width=2, dtype=T.i32, cache_modifier=cm)
    )
    return (fx.Int64(w[1]) << 32) | fx.Int64(fx.Uint32(w[0]))


def _words(v):
    return fx.Vector.from_elements(
        [fx.Int32(v & 0xFFFFFFFF), fx.Int32(v >> 32)], fx.Int32
    )


@traced
def _zero(base, nbytes, first, stride, cm=0):
    """``nbytes`` at ``base`` zeroed, 16 B a store, the grid's lanes from
    ``first`` by ``stride``."""
    zero = fx.Vector.from_elements([fx.Int32(0)] * 4, fx.Int32)
    i = first
    n = fx.Int32(nbytes >> 4)
    while i < n:
        bo.buffer_store(zero, rsrc(base), i * 4, cache_modifier=cm)
        i = i + stride


@traced
def _wait_arrived(state, e):
    """Until every other CTA flagged epoch ``e`` (lane l: CTAs 4 l ..)."""
    lane = fx.thread_idx.x
    want = fx.Int32(e & 0xFFFFFFFF)

    def late():
        w = fx.Vector(
            bo.buffer_load(
                rsrc(state), ARRIVED * 2 + lane * 4, vec_width=4, dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        bad = fx.Boolean(False)
        for q in range_constexpr(4):
            cta = lane * 4 + q
            bad = bad | ((cta != 0) & (w[q] != want))
        return ballot(bad) != 0

    while late():
        rocdl.s_sleep(1)


@traced
def _fence(peers, fence_off, rank, state, e, tp, debug):
    """Epoch ``e`` into this rank's slot at every rank, then this rank's slots
    polled (lane p: source rank p's) until every one reaches it. A debug build
    gives up after DIAG_TICKS, a rank that never arrived leaving its last epoch
    seen + 1 in its MISSING word."""
    tid = fx.thread_idx.x
    if tid == 0:
        bo.buffer_store(_words(e), rsrc(state), EPOCH * 2)
        for p in range_constexpr(tp):
            tail = load_ptr64(peers, p) + fence_off
            bo.buffer_store(_words(e), rsrc(tail), rank * 2, cache_modifier=CM_SYS)
    own = rsrc(load_ptr64(peers, rank) + fence_off)
    src = fx.min(tid, tp - 1)

    def seen_now():
        return _ld64(own, src, CM_SYS)

    seen = seen_now()
    if const_expr(not debug):
        while ballot(seen < e) != 0:
            rocdl.s_sleep(1)
            seen = seen_now()
    else:
        t0 = memrealtime()
        late = fx.Int32(0)
        while (ballot(seen < e) != 0) & (late == 0):
            rocdl.s_sleep(1)
            seen = seen_now()
            late = (memrealtime() - t0 > DIAG_TICKS).select(fx.Int32(1), fx.Int32(0))
        if (tid < tp) & (seen < e):
            bo.buffer_store(_words(seen + 1), rsrc(state), (MISSING + tid) * 2)


@dataclass(frozen=True)
class StepBeginBuild:
    tp: int
    debug: bool


@functools.cache
def _build(key: StepBeginBuild):
    tp, debug = key.tp, key.debug
    assert 1 <= tp <= FENCE_SLOTS
    keyed = key_tuple(key, source_digest("mono"))

    name = kernel_symbol("mono_step_begin", tp=tp, debug=int(debug))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def step_begin(
        s0: Int64, n0: Int64, s1: Int64, n1: Int64,
        peers: Int64, fence_off: Int64, rank: Int32, state: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane0, stride = bid * THREADS + tid, BLOCKS * THREADS
        # this step's epoch: CTA 0 stores it only once every CTA flagged
        e = _ld64(rsrc(state), EPOCH) + 1
        # the peer data first, at system scope: in memory before the epoch
        # announces it to a peer
        _zero(load_ptr64(peers, rank), fence_off, lane0, stride, CM_SYS)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if (bid != 0) & (tid == 0):
            bo.buffer_store(
                fx.Int32(e & 0xFFFFFFFF), rsrc(state), ARRIVED * 2 + bid,
                cache_modifier=CM_DEV,
            )  # fmt: skip
        # CTA 0's wave 0 fences (every lane polling: wave-uniform loops) while
        # the others zero the scratch, this rank's alone
        if (bid == 0) & (tid < 64):
            _wait_arrived(state, e)
            _fence(peers, fence_off, rank, state, e, tp, debug)
        _zero(s0, n0, lane0, stride)
        _zero(s1, n1, lane0, stride)

    @launcher(name)
    def launch(
        s0: Int64, n0: Int64, s1: Int64, n1: Int64,
        peers: Int64, fence_off: Int64, rank: Int32, state: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        step_begin(s0, n0, s1, n1, peers, fence_off, rank, state).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


def step_begin(scratch, peers, state: torch.Tensor, debug: bool) -> None:
    """Zero ``scratch`` (at most MAX_SCRATCH tensors) and the data region of
    ``peers`` (a ``PeerBuffer``), then the fence: one launch on the current
    stream (a graph records it). ``state``: STATE_WORDS int64, zero at first."""
    assert len(scratch) <= MAX_SCRATCH
    # the peer data's size is its fence slots' offset
    sizes = [peers.bytes.nbytes] + [b.nbytes for b in scratch]
    assert all(n % ZERO_ALIGN == 0 for n in sizes), sizes
    args = []
    for b in scratch:
        args += [b.data_ptr(), b.nbytes]
    args += [0, 0] * (MAX_SCRATCH - len(scratch))
    _build(StepBeginBuild(peers.addresses.numel(), debug))(
        *args,
        peers.addresses.data_ptr(),
        peers.bytes.nbytes,
        peers.rank,
        state.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
