# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-gather of the vocab-parallel LM head logits straight into [M, vocab].

Stock gathers the [M, vocab / tp] shards with NCCL into [tp, M, vocab / tp] and
permutes that into [M, vocab]: ~62 us of all-gather and ~19 us of copy at 64 rows
(the verify step of MTP3 at 16 requests, or 64 requests without MTP). Here each
rank stores its shard into column block ``rank`` of every rank's symmetric
[M, vocab] buffer over NVLink, then every CTA meets its counterparts on the other
ranks at a flag barrier, and the caller reads the local buffer. Unicast stores
to the peers beat one NVLS multicast store per vector (34 vs 39 us at 64 rows,
destination out of L2).

Two buffers (and flag sets) alternate: a rank starting gather k + 2 has passed
gather k + 1's barrier, which its peers reach only after their streams finished
with gather k's logits.
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
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
    load_vector,
    store_words,
)

THREADS = 512
VEC = 8
UNROLL = 4
BUFFERS = 2


@asm()
def _arrive(address):
    """Publishes this CTA's stores and counts it in every rank's flag."""
    code = "fence.acq_rel.sys; multimem.red.release.sys.global.add.u32 [$0], 1;"
    return code, "l,~{memory}", (Int64(address),)


@asm(Uint32)
def _load_acquire(pointer):
    return "ld.acquire.sys.global.u32 $0, [$1];", "=r,l,~{memory}", (pointer,)


class _GatherKernel:
    def __init__(self, *, columns: int, rank: int, capacity: int, ctas: int,
                 buffers: tuple[int, ...]):  # fmt: skip
        if columns % VEC:
            raise ValueError("the logits shard must hold whole 8-column vectors")
        self.columns, self.rank = columns, rank
        self.capacity, self.ctas = capacity, ctas
        # Every rank's buffer, mapped into this rank's address space.
        self.buffers = buffers
        self.tp = len(buffers)
        self.vectors = columns // VEC

    @cute.jit
    def __call__(
        self,
        shard: cute.Tensor,
        flags: cute.Tensor,
        flags_multicast: Int64,
        parity: Int32,
        m: Int32,
        stream,
    ) -> None:
        self.kernel(shard, flags, flags_multicast, parity, m).launch(
            grid=(self.ctas, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        shard: cute.Tensor,
        flags: cute.Tensor,
        flags_multicast: Int64,
        parity: Int32,
        m: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        stride = self.ctas * THREADS
        total = m * self.vectors
        width = Int64(self.columns * self.tp)
        base = (Int64(parity) * self.capacity * width + self.rank * self.columns) * 2
        cute.arch.griddepcontrol_wait()
        for first in cutlass.range(block * THREADS + tidx, total, stride * UNROLL):
            # Every load of a round is issued before its stores; out-of-range
            # vectors reread the last one and skip the store.
            loaded = []
            for u in cutlass.range_constexpr(UNROLL):
                vector = cutlass.min(first + u * stride, total - 1)
                loaded.append(load_vector(shard.iterator + Int64(vector) * VEC))
            for u in cutlass.range_constexpr(UNROLL):
                vector = first + u * stride
                if vector < total:
                    row = vector // self.vectors
                    column = vector - row * self.vectors
                    offset = base + (Int64(row) * width + Int64(column) * VEC) * 2
                    for peer in cutlass.range_constexpr(self.tp):
                        store_words(Int64(self.buffers[peer]) + offset, loaded[u])
        cute.arch.barrier()
        if tidx == 0:
            flag = parity * self.ctas + block
            _arrive(flags_multicast + Int64(flag) * 4)
            count = Uint32(0)
            while count != Uint32(self.tp):
                count = _load_acquire(flags.iterator + flag)
            store_words(flags.iterator + flag, (Uint32(0),))


class LogitsGather:
    """``tensor_model_parallel_all_gather`` of [M, vocab shard] BF16 logits for
    M <= capacity, returned as a view of a buffer that the next gather but one
    overwrites. Construction is collective over the TP group."""

    def __init__(self, *, columns: int, capacity: int, device: torch.device):
        group = get_tp_group().device_group
        rank = dist.get_rank(group)
        tp = dist.get_world_size(group)
        device = torch.device(device)
        self.columns, self.capacity = columns, capacity
        ctas = torch.cuda.get_device_properties(device).multi_processor_count
        with torch.accelerator.device_index(device.index):
            self._buffer = symm_mem.empty(
                (BUFFERS, capacity, tp * columns), dtype=torch.bfloat16, device=device
            )
            self._flags = symm_mem.empty(
                BUFFERS * ctas, dtype=torch.int32, device=device
            )
            self._flags.zero_()
            buffer = symm_mem.rendezvous(self._buffer, group)
            flags = symm_mem.rendezvous(self._flags, group)
            self._flags_multicast = int(flags.multicast_ptr or 0)
            if not self._flags_multicast:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._handles = (buffer, flags)
            self._kernel = cute.compile(
                _GatherKernel(
                    columns=columns,
                    rank=rank,
                    capacity=capacity,
                    ctas=ctas,
                    buffers=tuple(int(p) for p in buffer.buffer_ptrs),
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=columns
                ),
                make_fake_compact_tensor(Int32, (BUFFERS * ctas,), assumed_align=4),
                Int64(0),
                Int32(0),
                Int32(capacity),
                current_cu_stream(),
            )
            torch.accelerator.synchronize()
        dist.barrier(group=group)
        self._parity = 0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Eager calls only: the host alternates the buffers."""
        m = logits.shape[0]
        assert 1 <= m <= self.capacity and logits.shape[1] == self.columns
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous()
        parity, self._parity = self._parity, self._parity ^ 1
        self._kernel(
            to_cute_dynamic(logits.flatten(), 16, divisibility=self.columns),
            to_cute(self._flags, 4),
            Int64(self._flags_multicast),
            Int32(parity),
            Int32(m),
            current_cu_stream(),
        )
        return self._buffer[parity, :m]
