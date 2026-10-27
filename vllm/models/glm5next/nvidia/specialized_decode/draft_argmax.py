# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft tokens from vocab-parallel logits in one kernel.

One CTA per token scans this rank's logits shard for its first maximum, packs it
into a 64-bit key (order-preserving value bits, then the complement of the
global index, so the larger key is the larger value and, on ties, the lower
index) and multicasts the key into every rank's mailbox. Each rank then polls
the TP keys and keeps the largest: the token ``argmax`` of the all-gathered
logits picks, without gathering them.

A mailbox word is never zero while published (the low word is the index
complement), so zero is the empty sentinel; each half is written once per use
and reset by its reader, and two generations per token slot keep a rank one
call ahead from overwriting a key its peers have not read.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from vllm.distributed import get_tp_group
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.primitives import (
    WARP_SIZE,
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    float_bits,
    load_bf16x8,
    load_words,
    multicast_words,
    redux,
    store_words,
)

THREADS = 1024
VEC = 8
GENERATIONS = 2


@cute.jit
def _ordered(value):
    """Unsigned bits whose order is the float order."""
    bits = Uint32(float_bits(value))
    flip = Uint32(0x80000000)
    if bits >= Uint32(0x80000000):
        flip = Uint32(0xFFFFFFFF)
    return bits ^ flip


@cute.jit
def _warp_max(high, low):
    """Lexicographic warp maximum of (high, low)."""
    top = redux(high, "max")
    candidate = Uint32(0)
    if high == top:
        candidate = low
    return top, redux(candidate, "max")


class _ArgmaxKernel:
    def __init__(self, *, columns: int, valid: int, offset: int, tp: int, rank: int,
                 capacity: int):  # fmt: skip
        if columns % VEC:
            raise ValueError("the logits shard must hold whole 8-column vectors")
        self.columns, self.valid, self.offset = columns, valid, offset
        self.tp, self.rank, self.capacity = tp, rank, capacity
        self.warps = THREADS // WARP_SIZE

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        mailbox: cute.Tensor,
        multicast: Int64,
        epoch: cute.Tensor,
        tokens: cute.Tensor,
        m: Int32,
        stream,
    ) -> None:
        self.kernel(logits, mailbox, multicast, epoch, tokens).launch(
            grid=(m, 1, 1),
            block=(THREADS, 1, 1),
            smem=self.warps * 8,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        mailbox: cute.Tensor,
        multicast: Int64,
        epoch: cute.Tensor,
        tokens: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        row = logits.iterator + Int64(token) * self.columns

        best = Uint32(0)
        index = Int32(0)
        vectors = self.columns // VEC
        rounds = (vectors + THREADS - 1) // THREADS
        # A thread issues all of its loads before the compares; the last round's
        # out-of-range vectors reread the last one and fail the column mask.
        loaded = []
        for r in cutlass.range_constexpr(rounds):
            vector = tidx + r * THREADS
            if r == rounds - 1 and vectors % THREADS:
                vector = cutlass.min(vector, vectors - 1)
            loaded.append(load_bf16x8(row + vector * VEC))
        for r in cutlass.range_constexpr(rounds):
            for k in cutlass.range_constexpr(VEC):
                key = _ordered(loaded[r][k])
                column = (tidx + r * THREADS) * VEC + k
                if key > best:  # noqa: SIM102 (one dynamic test per if)
                    if column < self.valid:
                        best = key
                        index = column
        high, low = _warp_max(best, Uint32(0xFFFFFFFF) - Uint32(index + self.offset))

        smem = cutlass.utils.SmemAllocator()
        highs = smem.allocate_array(Uint32, self.warps)
        lows = smem.allocate_array(Uint32, self.warps)
        if lane == 0:
            cute.arch.store((highs + warp).llvm_ptr, high)
            cute.arch.store((lows + warp).llvm_ptr, low)
        cute.arch.barrier()
        if warp == 0:
            high = cute.arch.load((highs + lane).llvm_ptr, Uint32)
            low = cute.arch.load((lows + lane).llvm_ptr, Uint32)
            high, low = _warp_max(high, low)
            if lane == 0:
                use = epoch[token]
                generation = Int64(use % GENERATIONS)
                own = ((generation * self.tp + self.rank) * self.capacity + token) * 2
                multicast_words(multicast + own * 4, (high, low), kind="f32")
                for source in cutlass.range_constexpr(self.tp):
                    element = (
                        (generation * self.tp + source) * self.capacity + Int64(token)
                    ) * 2
                    slot = mailbox.iterator + element
                    peer_high = Uint32(0)
                    peer_low = Uint32(0)
                    empty = True
                    while empty:
                        peer_high, peer_low = load_words(
                            slot, count=2, volatile=True, word_type="u32"
                        )
                        empty = peer_high == Uint32(0)
                        if peer_low == Uint32(0):
                            empty = True
                    store_words(slot, (Uint32(0), Uint32(0)))
                    if peer_high > high:
                        high = peer_high
                        low = peer_low
                    if peer_high == high:  # noqa: SIM102 (one dynamic test per if)
                        if peer_low > low:
                            low = peer_low
                epoch[token] = use + 1
                tokens[token] = Int64(Uint32(0xFFFFFFFF) - low)


class DraftArgmax:
    """``argmax(all_gather(logits))`` of [M, vocab shard] BF16 logits, M <= capacity.

    Construction is collective over the TP group.
    """

    def __init__(self, *, columns: int, valid: int, offset: int, capacity: int,
                 device: torch.device):  # fmt: skip
        group = get_tp_group().device_group
        tp = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self.capacity = capacity
        device = torch.device(device)
        with torch.accelerator.device_index(device.index):
            self._kernel = cute.compile(
                _ArgmaxKernel(
                    columns=columns,
                    valid=valid,
                    offset=offset,
                    tp=tp,
                    rank=rank,
                    capacity=capacity,
                ),  # fmt: skip
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=columns
                ),
                make_fake_compact_tensor(
                    Int32, (GENERATIONS * tp * capacity * 2,), assumed_align=16
                ),
                Int64(0),
                make_fake_compact_tensor(Int32, (capacity,), assumed_align=4),
                make_fake_dynamic_compact_tensor(Int64, alignment=8, divisibility=1),
                Int32(capacity),
                current_cu_stream(),
            )
            self._mailbox = symm_mem.empty(
                (GENERATIONS, tp, capacity, 2), dtype=torch.int32, device=device
            )
            handle = symm_mem.rendezvous(self._mailbox, group)
            multicast = int(handle.multicast_ptr or 0)
            if not multicast or multicast % 16:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._handle = handle
            self._multicast = multicast
            self._mailbox.zero_()
            self._epoch = torch.zeros(capacity, dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def supports(self, tokens: int) -> bool:
        return 1 <= tokens <= self.capacity

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        m = logits.shape[0]
        tokens = torch.empty(m, dtype=torch.int64, device=logits.device)
        self._kernel(
            to_cute_dynamic(logits.flatten(), 16, divisibility=logits.shape[1]),
            to_cute(self._mailbox.flatten(), 16),
            Int64(self._multicast),
            to_cute(self._epoch, 4),
            to_cute_dynamic(tokens, 8, divisibility=1),
            Int32(m),
            current_cu_stream(),
        )
        return tokens
