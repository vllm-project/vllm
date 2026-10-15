# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded TP4 peer all-reduce prototype and sticky health ABI."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    CM_SYS,
    peer_bases,
    rsrc,
)
from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer

TP = 4
THREADS = 256
ELEMENTS = 256
POLL_LIMIT = 1 << 18
DATA_BYTES = TP * ELEMENTS * 4
TAG_OFFSET = DATA_BYTES
ERROR_OFFSET = TAG_OFFSET + TP * 4
PEER_BYTES = ERROR_OFFSET + 4 * 4
_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class MonoHealth:
    failed: bool
    rank: int
    epoch: int
    waiter_rank: int


def build_bounded_allreduce():
    @flyc.kernel(
        name="minimax_m3_bounded_peer_allreduce",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        partial: Int64,
        output: Int64,
        own: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int32,
        fault_rank: Int32,
    ):
        tid = fx.thread_idx.x
        bases = peer_bases(peers, TP)

        if rank != fault_rank:
            value = bo.buffer_load(rsrc(partial), tid, vec_width=1, dtype=T.f32)
            for peer in range_constexpr(TP):
                bo.buffer_store(
                    value,
                    rsrc(bases[peer]),
                    rank * ELEMENTS + tid,
                    cache_modifier=CM_SYS,
                )
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                for peer in range_constexpr(TP):
                    bo.buffer_store(
                        epoch,
                        rsrc(bases[peer] + fx.Int64(TAG_OFFSET)),
                        rank,
                        cache_modifier=CM_SYS,
                    )

        gpu.barrier()
        if tid == 0:
            count = fx.Int32(0)
            tag = bo.buffer_load(
                rsrc(own + fx.Int64(TAG_OFFSET)),
                0,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
            ready = tag == epoch
            for peer in range_constexpr(1, TP):
                tag = bo.buffer_load(
                    rsrc(own + fx.Int64(TAG_OFFSET)),
                    peer,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
                ready = ready & (tag == epoch)
            failed = bo.buffer_load(
                rsrc(own + fx.Int64(ERROR_OFFSET)),
                0,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
            while (~ready) & (failed == 0) & (count < POLL_LIMIT):
                rocdl.s_sleep(1)
                tag = bo.buffer_load(
                    rsrc(own + fx.Int64(TAG_OFFSET)),
                    0,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
                ready = tag == epoch
                for peer in range_constexpr(1, TP):
                    tag = bo.buffer_load(
                        rsrc(own + fx.Int64(TAG_OFFSET)),
                        peer,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                    ready = ready & (tag == epoch)
                failed = bo.buffer_load(
                    rsrc(own + fx.Int64(ERROR_OFFSET)),
                    0,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
                count = count + 1
            if (~ready) & (failed == 0):
                bad_rank = (fault_rank >= 0).select(fault_rank, rank)
                error = fx.Vector.from_elements(
                    [fx.Int32(1), bad_rank, epoch, rank], fx.Int32
                )
                for peer in range_constexpr(TP):
                    bo.buffer_store(
                        error,
                        rsrc(bases[peer] + fx.Int64(ERROR_OFFSET)),
                        0,
                        cache_modifier=CM_SYS,
                    )
                rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()

        failed = bo.buffer_load(
            rsrc(own + fx.Int64(ERROR_OFFSET)),
            0,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_SYS,
        )
        if failed == 0:
            acc = fx.Float32(0.0)
            for peer in range_constexpr(TP):
                value = bo.buffer_load(
                    rsrc(own),
                    peer * ELEMENTS + tid,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_SYS,
                )
                acc = acc + value
            bo.buffer_store(acc, rsrc(output), tid)

    @flyc.jit
    def launch(
        partial: Int64,
        output: Int64,
        own: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int32,
        fault_rank: Int32,
        stream: fx.Stream = _STREAM,
    ):
        kernel(
            partial,
            output,
            own,
            peers,
            rank,
            epoch,
            fault_rank,
        ).launch(grid=(1,), block=(THREADS,), stream=stream)

    return launch


class BoundedAllReducePrototype:
    def __init__(self, rank: int, group, device: torch.device) -> None:
        self.rank = rank
        self.peer = PeerBuffer(PEER_BYTES, group, rank, TP, device)
        self.peer.bytes.zero_()
        self.output = torch.empty(ELEMENTS, dtype=torch.float32, device=device)
        self._launch = build_bounded_allreduce()

    def run(
        self, partial: torch.Tensor, epoch: int, fault_rank: int = -1
    ) -> torch.Tensor:
        self._launch(
            partial.data_ptr(),
            self.output.data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            epoch,
            fault_rank,
            stream=torch.cuda.current_stream(),
        )
        return self.output

    def health(self) -> MonoHealth:
        words = self.peer.bytes[ERROR_OFFSET : ERROR_OFFSET + 16].view(torch.int32)
        failed, rank, epoch, waiter = words.cpu().tolist()
        return MonoHealth(bool(failed), rank, epoch, waiter)

    def close(self) -> None:
        self.peer.close()
