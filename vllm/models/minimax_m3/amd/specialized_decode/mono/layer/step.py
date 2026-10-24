# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-row metadata, mailbox clearing, and a bounded step-start TP fence."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.triton_utils import tl, triton

from .support.device.ops import ballot, load_ptr64, memrealtime

ERROR_WORDS = 4
ERROR_BYTES = 256
FENCE_BYTES = 64
WAIT_TICKS = 1_000_000_000
_STREAM = fx.Stream(None)


def _ptr(addr, i, bits=64):
    """The global pointer to element i (of ``bits``) of address ``addr``."""
    at = addr + fx.Int64(i) * (bits // 8)
    return fx.inttoptr(
        fx.PointerType.get(
            T.i64 if bits == 64 else T.i32, fx.AddressSpace.Global, bits // 8
        ),
        at,
    )


def load64(addr, i, scope):
    return fx.generic_load(
        _ptr(addr, i),
        dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=scope,
    )


def store32(addr, i, word, scope):
    fx.generic_store(
        _ptr(addr, i, 32),
        fx.Int32(word),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=scope,
    )


def store64(addr, i, word, scope):
    fx.generic_store(
        _ptr(addr, i), word, memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope
    )


def release(scope):
    fx.memory_fence(syncscope=scope, ordering=fx.AtomicOrdering.Release)


@fx.struct
class _FenceLds:
    word: fx.Array[fx.Int64, 1, 8]


@triton.jit
def expand_rows(
    main_table,
    index_table,
    seq_lens,
    main_slots,
    index_slots,
    out_main,
    out_index,
    out_seq,
    out_slots,
    out_index_slots,
    batch_ids,
    MAIN_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    Q: tl.constexpr,
    REQUESTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    req = tl.minimum(row // Q, REQUESTS - 1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    main = tl.load(main_table + req * MAIN_STRIDE + col, col < MAIN_STRIDE, other=0)
    index = tl.load(index_table + req * INDEX_STRIDE + col, col < INDEX_STRIDE, other=0)
    tl.store(out_main + row * WIDTH + col, main, col < WIDTH)
    tl.store(out_index + row * WIDTH + col, index, col < WIDTH)
    if block == 0:
        slot = tl.load(main_slots + row)
        index_slot = tl.load(index_slots + row)
        length = tl.load(seq_lens + req) - Q + 1 + row % Q
        live = slot >= 0
        tl.store(out_seq + row, tl.where(live, tl.maximum(length, 1), 1))
        rebased = slot // 128 * 256 + slot % 128
        tl.store(out_slots + row, tl.where(live, rebased, -1))
        tl.store(out_index_slots + row, index_slot)
        tl.store(batch_ids + row, tl.where(live, req, -1))


@flyc.kernel(known_block_size=[64, 1, 1])
def _begin_fence(
    peers: Int64,
    rank: Int32,
    epoch: Int64,
    offset: Int64,
    tp: fx.Constexpr[int],
    mode: Int32,
):
    if mode != 0:
        lane = fx.thread_idx.x
        shared = fx.SharedAllocator().allocate(_FenceLds).peek().word.ptr
        if lane == 0:
            e = load64(epoch, 0, "agent") + 1
            store64(epoch, 0, e, "agent")
            fx.ptr_store(e, shared)
        gpu.barrier()
        want = fx.ptr_load(shared)
        own = load_ptr64(peers, rank)
        if lane == 0:
            release(None)
            for peer in range_constexpr(tp):
                store64(load_ptr64(peers, peer) + offset, rank, want, None)
        src = fx.min(lane, tp - 1)
        seen = load64(own + offset, src, None)
        start = memrealtime()
        late = fx.Int32(0)
        while (ballot(seen < want) != 0) & (late == 0):
            rocdl.s_sleep(1)
            seen = load64(own + offset, src, None)
            late = (memrealtime() - start > WAIT_TICKS).select(fx.Int32(1), fx.Int32(0))
        if (lane == 0) & (late != 0):
            for peer in range_constexpr(tp):
                error = load_ptr64(peers, peer) - ERROR_BYTES
                for i, word in enumerate([1, rank, fx.Int32(want), 1]):
                    store32(error, i, fx.Int32(word), None)
        fx.memory_fence(syncscope=None, ordering=fx.AtomicOrdering.Acquire)


@flyc.jit
def begin_fence(
    peers: Int64,
    rank: Int32,
    epoch: Int64,
    offset: Int64,
    tp: fx.Constexpr[int],
    mode: Int32 = 1,
    stream: fx.Stream = _STREAM,
):
    _begin_fence(peers, rank, epoch, offset, tp, mode).launch(
        grid=(1,), block=(64,), stream=stream
    )


@triton.jit
def check_diagnostics(
    diag, peers, rank, epoch, TP: tl.constexpr, ERROR_OFFSET: tl.constexpr = 256
):
    cta = tl.arange(0, 256)
    failed = tl.max(tl.load(diag + cta * 8)) != 0
    if failed:
        step = tl.load(epoch)
        words = tl.arange(0, 4)
        values = tl.where(
            words == 0,
            1,
            tl.where(words == 1, rank, tl.where(words == 2, step.to(tl.int32), 2)),
        )
        for peer in tl.static_range(TP):
            address = tl.load(peers + peer) - ERROR_OFFSET
            target = address.to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(target + words, values, sem="release", scope="sys")
