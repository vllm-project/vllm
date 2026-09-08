# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL kernels dedicated to two-rank RDNA4 all-reduce.

This family contains the small-message mapped-memory kernels and the
graph-only direct-peer kernel. The launcher names identify the transport.
"""

from __future__ import annotations

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64, Stream

from .common import load_pack_128b as _load_pack
from .common import store_pack_128b as _store_pack

_SG_START_OFFSET = 0
_SG_END_OFFSET = 80 * 8 * 4
_SG_FLAG_OFFSET = 80 * 8 * 4 * 2


def _load_i64_acquire(addr_i64):
    return fx.rocdl.global_load(
        addr_i64,
        fx.Int64,
        alignment=8,
        memory_order=fx.rocdl.MemoryOrder.Acquire,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i64_release(addr_i64, value):
    fx.rocdl.global_store(
        addr_i64,
        value,
        alignment=8,
        memory_order=fx.rocdl.MemoryOrder.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _sleep_one():
    fx.rocdl.sleep(1)


def _add_bf16_pack(lhs_raw, rhs_raw):
    lhs = lhs_raw.bitcast(fx.BFloat16).to(fx.Float32)
    rhs = rhs_raw.bitcast(fx.BFloat16).to(fx.Float32)
    return (lhs + rhs).to(fx.BFloat16).bitcast(fx.Int32)


def _load_i32_acquire(addr_i32):
    return fx.rocdl.global_load(
        addr_i32,
        fx.Int32,
        alignment=4,
        memory_order=fx.rocdl.MemoryOrder.Acquire,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i32_release(addr_i32, value):
    fx.rocdl.global_store(
        addr_i32,
        value,
        alignment=4,
        memory_order=fx.rocdl.MemoryOrder.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _load_pointer(array_addr, index):
    return fx.rocdl.global_load(
        array_addr + fx.Int64(index) * fx.Int64(8),
        fx.Int64,
        alignment=8,
    )


@cache
def make_p2p_tp2_one_shot_launcher(*, blocks: int, threads: int):
    if not 0 < blocks <= 80:
        raise ValueError(f"one-shot blocks must be in [1, 80], got {blocks}")
    if threads not in (256, 512):
        raise ValueError(f"one-shot threads must be 256 or 512, got {threads}")

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def tp2_allreduce_bf16_p2p_one_shot(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_addr: Int64,
        input_ptrs_addr: Int64,
        input_addr: Int64,
        output_addr: Int64,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        peer_rank = fx.Int32(1) - rank
        flag_addr = (
            self_signal + fx.Int64(_SG_FLAG_OFFSET) + fx.Int64(bid) * fx.Int64(4)
        )
        ticket = fx.Int32(_load_i32_acquire(flag_addr)) + fx.Int32(1)
        peer_input = _load_pointer(input_ptrs_addr, 1)
        pack_count = numel // fx.Int32(8)
        block_packs = (pack_count + fx.Int32(blocks) - fx.Int32(1)) // fx.Int32(blocks)
        block_begin = bid * block_packs
        candidate_block_end = block_begin + block_packs
        block_end = (candidate_block_end < pack_count).select(
            candidate_block_end, pack_count
        )
        peer_signal = _load_pointer(signal_ptrs_addr, peer_rank)
        block_slot = bid * fx.Int32(8)
        peer_ready_addr = (
            peer_signal
            + fx.Int64(_SG_START_OFFSET)
            + fx.Int64(block_slot + rank) * fx.Int64(4)
        )
        local_ready_addr = (
            self_signal
            + fx.Int64(_SG_START_OFFSET)
            + fx.Int64(block_slot + peer_rank) * fx.Int64(4)
        )
        ticket_epoch = ticket & fx.Int32(0x7FFFF)
        progress = (ticket_epoch << fx.Int32(12)) | fx.Int32(1)
        if tid == fx.Int32(0):
            _store_i32_release(peer_ready_addr, progress)
            observed = fx.Int32(_load_i32_acquire(local_ready_addr))
            observed_epoch = observed >> fx.Int32(12)
            while observed_epoch != ticket_epoch:
                _sleep_one()
                observed = fx.Int32(_load_i32_acquire(local_ready_addr))
                observed_epoch = observed >> fx.Int32(12)
        gpu.barrier()

        for pack in range(block_begin + tid, block_end, fx.Int32(threads)):
            _store_pack(
                output_addr,
                pack,
                _add_bf16_pack(
                    _load_pack(input_addr, pack),
                    _load_pack(peer_input, pack),
                ),
            )
        gpu.barrier()

        peer_done_addr = (
            peer_signal
            + fx.Int64(_SG_END_OFFSET)
            + fx.Int64(block_slot + rank) * fx.Int64(4)
        )
        local_done_addr = (
            self_signal
            + fx.Int64(_SG_END_OFFSET)
            + fx.Int64(block_slot + peer_rank) * fx.Int64(4)
        )
        if tid == fx.Int32(0):
            _store_i32_release(peer_done_addr, ticket_epoch)
            observed_done = fx.Int32(_load_i32_acquire(local_done_addr))
            while observed_done < ticket_epoch:
                _sleep_one()
                observed_done = fx.Int32(_load_i32_acquire(local_done_addr))
        gpu.barrier()

        if tid == fx.Int32(0):
            _store_i32_release(flag_addr, ticket)

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_p2p_one_shot(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_addr: Int64,
        input_ptrs_addr: Int64,
        input_addr: Int64,
        output_addr: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        tp2_allreduce_bf16_p2p_one_shot(
            rank,
            self_signal,
            signal_ptrs_addr,
            input_ptrs_addr,
            input_addr,
            output_addr,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_p2p_one_shot.func.__name__ = (
        f"launch_tp2_bf16_p2p_one_shot_b{blocks}_t{threads}_directinput"
    )
    return launch_p2p_one_shot


@cache
def make_mapped_tp2_full_launcher(*, blocks: int, threads: int):
    if blocks not in (1, 2, 4):
        raise ValueError(f"full kernel blocks must be 1, 2, or 4, got {blocks}")

    SharedStorage = fx.struct(
        type(
            "FullSharedStorage",
            (),
            {"__annotations__": {"ticket": fx.Array[fx.Int64, 1, 8]}},
        )
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def tp2_allreduce_bf16_full(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_ready_addr: Int64,
        peer_ready_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_ptr = lds.ticket.ptr
        ticket = fx.Int64(0)

        if tid == fx.Int32(0):
            if const_expr(blocks == 1):
                ticket = fx.Int64(_load_i64_acquire(local_ready_addr)) + fx.Int64(1)
            else:
                block_ready_addr = local_ready_addr + fx.Int64(bid) * fx.Int64(8)
                previous = fx.Int64(_load_i64_acquire(block_ready_addr))
                if bid == fx.Int32(0):
                    ticket = previous + fx.Int64(1)
                    _store_i64_release(local_launch_addr, ticket)
                else:
                    ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
                    while ticket <= previous:
                        _sleep_one()
                        ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
            fx.ptr_store(fx.Vector.from_elements([ticket], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket = fx.Vector(
            fx.ptr_load(
                ticket_ptr,
                result_type=fx.Vector.make_type(1, fx.Int64),
            )
        )[0]

        parity_bytes = (ticket & fx.Int64(1)) * slot_bytes
        local_slot_addr = local_slot_addr + parity_bytes
        peer_slot_addr = peer_slot_addr + parity_bytes
        pack_count = numel // fx.Int32(8)
        global_thread = bid * fx.Int32(threads) + tid
        global_stride = fx.Int32(blocks * threads)

        for pack in range(global_thread, pack_count, global_stride):
            _store_pack(local_slot_addr, pack, _load_pack(input_addr, pack))
        gpu.barrier()

        if tid == fx.Int32(0):
            if const_expr(blocks == 1):
                _store_i64_release(local_ready_addr, ticket)
                peer_ticket = fx.Int64(_load_i64_acquire(peer_ready_addr))
                while peer_ticket < ticket:
                    _sleep_one()
                    peer_ticket = fx.Int64(_load_i64_acquire(peer_ready_addr))
            else:
                local_block_addr = local_ready_addr + fx.Int64(bid) * fx.Int64(8)
                peer_block_addr = peer_ready_addr + fx.Int64(bid) * fx.Int64(8)
                _store_i64_release(local_block_addr, ticket)
                peer_ticket = fx.Int64(_load_i64_acquire(peer_block_addr))
                while peer_ticket < ticket:
                    _sleep_one()
                    peer_ticket = fx.Int64(_load_i64_acquire(peer_block_addr))
        gpu.barrier()

        for pack in range(global_thread, pack_count, global_stride):
            _store_pack(
                output_addr,
                pack,
                _add_bf16_pack(
                    _load_pack(input_addr, pack),
                    _load_pack(peer_slot_addr, pack),
                ),
            )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_full(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_ready_addr: Int64,
        peer_ready_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        tp2_allreduce_bf16_full(
            input_addr,
            output_addr,
            local_slot_addr,
            peer_slot_addr,
            slot_bytes,
            local_ready_addr,
            peer_ready_addr,
            local_launch_addr,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_full.func.__name__ = f"launch_tp2_bf16_full_b{blocks}_t{threads}"
    return launch_full


@cache
def make_mapped_tp2_pipeline_launcher(*, blocks: int, threads: int, chunk_packs: int):
    if blocks < 6 or blocks > 16:
        raise ValueError(f"pipeline blocks must be in [6, 16], got {blocks}")
    if chunk_packs < threads or chunk_packs % threads:
        raise ValueError("chunk_packs must be a positive multiple of threads")

    SharedStorage = fx.struct(
        type(
            "PipelineSharedStorage",
            (),
            {"__annotations__": {"ticket": fx.Array[fx.Int64, 1, 8]}},
        )
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def tp2_allreduce_bf16_pipeline(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_progress_addr: Int64,
        peer_progress_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_ptr = lds.ticket.ptr
        ticket = fx.Int64(0)

        if tid == fx.Int32(0):
            block_progress_addr = local_progress_addr + fx.Int64(bid) * fx.Int64(8)
            previous_progress = fx.Int64(_load_i64_acquire(block_progress_addr))
            previous_ticket = previous_progress >> fx.Int64(32)
            if bid == fx.Int32(0):
                ticket = fx.Int64(_load_i64_acquire(local_launch_addr)) + fx.Int64(1)
                _store_i64_release(local_launch_addr, ticket)
            else:
                ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
                while ticket <= previous_ticket:
                    _sleep_one()
                    ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
            fx.ptr_store(fx.Vector.from_elements([ticket], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket = fx.Vector(
            fx.ptr_load(
                ticket_ptr,
                result_type=fx.Vector.make_type(1, fx.Int64),
            )
        )[0]

        parity_bytes = (ticket & fx.Int64(1)) * slot_bytes
        local_slot_addr = local_slot_addr + parity_bytes
        peer_slot_addr = peer_slot_addr + parity_bytes
        pack_count = numel // fx.Int32(8)
        first_pack = bid * fx.Int32(chunk_packs)
        round_stride = fx.Int32(blocks * chunk_packs)
        rounds = (pack_count - first_pack + round_stride - fx.Int32(1)) // round_stride

        for round_index in range(fx.Int32(0), rounds, fx.Int32(1)):
            chunk_begin = first_pack + round_index * round_stride
            candidate_end = chunk_begin + fx.Int32(chunk_packs)
            chunk_end = (candidate_end < pack_count).select(candidate_end, pack_count)

            for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                _store_pack(local_slot_addr, pack, _load_pack(input_addr, pack))
            gpu.barrier()

            if tid == fx.Int32(0):
                progress = (ticket << fx.Int64(32)) | fx.Int64(
                    round_index + fx.Int32(1)
                )
                local_block_addr = local_progress_addr + fx.Int64(bid) * fx.Int64(8)
                peer_block_addr = peer_progress_addr + fx.Int64(bid) * fx.Int64(8)
                _store_i64_release(local_block_addr, progress)
                peer_value = fx.Int64(_load_i64_acquire(peer_block_addr))
                while peer_value < progress:
                    _sleep_one()
                    peer_value = fx.Int64(_load_i64_acquire(peer_block_addr))
            gpu.barrier()

            for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                _store_pack(
                    output_addr,
                    pack,
                    _add_bf16_pack(
                        _load_pack(input_addr, pack),
                        _load_pack(peer_slot_addr, pack),
                    ),
                )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_pipeline(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_progress_addr: Int64,
        peer_progress_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        tp2_allreduce_bf16_pipeline(
            input_addr,
            output_addr,
            local_slot_addr,
            peer_slot_addr,
            slot_bytes,
            local_progress_addr,
            peer_progress_addr,
            local_launch_addr,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_pipeline.func.__name__ = (
        f"launch_tp2_bf16_pipeline_b{blocks}_t{threads}_p{chunk_packs}"
    )
    return launch_pipeline


__all__ = [
    "make_mapped_tp2_full_launcher",
    "make_mapped_tp2_pipeline_launcher",
    "make_p2p_tp2_one_shot_launcher",
]
