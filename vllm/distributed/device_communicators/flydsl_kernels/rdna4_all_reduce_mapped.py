# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL kernels for TP4/TP8 HIP-mapped host-memory all-reduce."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, Stream

from .common import load_pack_128b as _load_pack
from .common import store_pack_128b as _store_pack


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


def _bf16_pack_to_f32(raw):
    return raw.bitcast(fx.BFloat16).to(fx.Float32)


@cache
def make_mapped_full_launcher(*, world_size: int, threads: int = 1024):
    if world_size not in (4, 8):
        raise ValueError(f"mapped-host all-reduce requires TP4/TP8, got TP{world_size}")
    if threads not in (256, 512, 1024):
        raise ValueError(f"unsupported thread count: {threads}")

    SharedStorage = fx.struct(
        type(
            "HostAllReduceSharedStorage",
            (),
            {"__annotations__": {"ticket": fx.Array[fx.Int64, 1, 8]}},
        )
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def mapped_allreduce_bf16(
        input_addr: Int64,
        output_addr: Int64,
        shared_addr: Int64,
        data_offset: Int64,
        slot_bytes: Int64,
        rank: Int32,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        rank_i64 = fx.Int64(rank)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_ptr = lds.ticket.ptr
        local_ready_addr = shared_addr + rank_i64 * fx.Int64(8)
        ticket = fx.Int64(0)

        if tid == fx.Int32(0):
            ticket = fx.Int64(_load_i64_acquire(local_ready_addr)) + fx.Int64(1)
            fx.ptr_store(fx.Vector.from_elements([ticket], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket = fx.Vector(
            fx.ptr_load(
                ticket_ptr,
                result_type=fx.Vector.make_type(1, fx.Int64),
            )
        )[0]

        parity = ticket & fx.Int64(1)
        local_slot_addr = (
            shared_addr + data_offset + (rank_i64 * fx.Int64(2) + parity) * slot_bytes
        )
        pack_count = numel // fx.Int32(8)

        for pack in range(tid, pack_count, fx.Int32(threads)):
            _store_pack(local_slot_addr, pack, _load_pack(input_addr, pack))
        gpu.barrier()

        if tid == fx.Int32(0):
            _store_i64_release(local_ready_addr, ticket)
            for peer_offset in range_constexpr(1, world_size):
                peer_rank = (rank + fx.Int32(peer_offset)) & fx.Int32(world_size - 1)
                peer_ready_addr = shared_addr + fx.Int64(peer_rank) * fx.Int64(8)
                peer_ticket = fx.Int64(_load_i64_acquire(peer_ready_addr))
                while peer_ticket < ticket:
                    _sleep_one()
                    peer_ticket = fx.Int64(_load_i64_acquire(peer_ready_addr))
        gpu.barrier()

        for pack in range(tid, pack_count, fx.Int32(threads)):
            acc = _bf16_pack_to_f32(_load_pack(input_addr, pack))
            for peer_offset in range_constexpr(1, world_size):
                peer_rank = (rank + fx.Int32(peer_offset)) & fx.Int32(world_size - 1)
                peer_slot_addr = (
                    shared_addr
                    + data_offset
                    + (fx.Int64(peer_rank) * fx.Int64(2) + parity) * slot_bytes
                )
                acc = acc + _bf16_pack_to_f32(_load_pack(peer_slot_addr, pack))
            _store_pack(
                output_addr,
                pack,
                acc.to(fx.BFloat16).bitcast(fx.Int32),
            )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_mapped_allreduce(
        input_addr: Int64,
        output_addr: Int64,
        shared_addr: Int64,
        data_offset: Int64,
        slot_bytes: Int64,
        rank: Int32,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        mapped_allreduce_bf16(
            input_addr,
            output_addr,
            shared_addr,
            data_offset,
            slot_bytes,
            rank,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(1, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_mapped_allreduce.func.__name__ = (
        f"launch_mapped_allreduce_bf16_ws{world_size}_t{threads}"
    )
    return launch_mapped_allreduce


@cache
def make_mapped_rsag_launcher(
    *,
    world_size: int,
    blocks: int,
    threads: int,
    chunk_packs: int,
):
    if world_size not in (4, 8):
        raise ValueError(f"mapped-host all-reduce requires TP4/TP8, got TP{world_size}")
    if blocks < world_size or blocks > 16 or blocks % world_size:
        raise ValueError("RSAG blocks must be a world-size multiple in [TP, 16]")
    if threads not in (256, 512, 1024):
        raise ValueError(f"unsupported thread count: {threads}")
    if chunk_packs < threads or chunk_packs % threads:
        raise ValueError("chunk_packs must be a positive multiple of threads")

    SharedStorage = fx.struct(
        type(
            "HostRsagSharedStorage",
            (),
            {"__annotations__": {"ticket": fx.Array[fx.Int64, 1, 8]}},
        )
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def mapped_allreduce_bf16_rsag(
        input_addr: Int64,
        output_addr: Int64,
        shared_addr: Int64,
        data_offset: Int64,
        slot_bytes: Int64,
        launch_offset: Int64,
        progress_offset: Int64,
        rank: Int32,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        rank_i64 = fx.Int64(rank)
        owner = bid & fx.Int32(world_size - 1)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_ptr = lds.ticket.ptr
        local_launch_addr = shared_addr + launch_offset + rank_i64 * fx.Int64(8)
        local_progress_addr = (
            shared_addr
            + progress_offset
            + (rank_i64 * fx.Int64(blocks) + fx.Int64(bid)) * fx.Int64(8)
        )
        ticket = fx.Int64(0)

        if tid == fx.Int32(0):
            previous_progress = fx.Int64(_load_i64_acquire(local_progress_addr))
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

        parity = ticket & fx.Int64(1)
        local_slot_addr = (
            shared_addr + data_offset + (rank_i64 * fx.Int64(2) + parity) * slot_bytes
        )
        owner_slot_addr = (
            shared_addr
            + data_offset
            + (fx.Int64(owner) * fx.Int64(2) + parity) * slot_bytes
        )
        pack_count = numel // fx.Int32(8)
        first_pack = bid * fx.Int32(chunk_packs)
        round_stride = fx.Int32(blocks * chunk_packs)
        rounds = (pack_count - first_pack + round_stride - fx.Int32(1)) // round_stride

        for round_index in range(fx.Int32(0), rounds, fx.Int32(1)):
            chunk_begin = first_pack + round_index * round_stride
            candidate_end = chunk_begin + fx.Int32(chunk_packs)
            chunk_end = (candidate_end < pack_count).select(candidate_end, pack_count)
            input_ready = (ticket << fx.Int64(32)) | fx.Int64(
                round_index * fx.Int32(2) + fx.Int32(1)
            )
            result_ready = input_ready + fx.Int64(1)

            for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                _store_pack(local_slot_addr, pack, _load_pack(input_addr, pack))
            gpu.barrier()

            if tid == fx.Int32(0):
                _store_i64_release(local_progress_addr, input_ready)
                for peer_offset in range_constexpr(1, world_size):
                    peer_rank = (rank + fx.Int32(peer_offset)) & fx.Int32(
                        world_size - 1
                    )
                    peer_progress_addr = (
                        shared_addr
                        + progress_offset
                        + (fx.Int64(peer_rank) * fx.Int64(blocks) + fx.Int64(bid))
                        * fx.Int64(8)
                    )
                    peer_progress = fx.Int64(_load_i64_acquire(peer_progress_addr))
                    while peer_progress < input_ready:
                        _sleep_one()
                        peer_progress = fx.Int64(_load_i64_acquire(peer_progress_addr))
            gpu.barrier()

            if rank == owner:
                for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                    acc = _bf16_pack_to_f32(_load_pack(input_addr, pack))
                    for peer_offset in range_constexpr(1, world_size):
                        peer_rank = (rank + fx.Int32(peer_offset)) & fx.Int32(
                            world_size - 1
                        )
                        peer_slot_addr = (
                            shared_addr
                            + data_offset
                            + (fx.Int64(peer_rank) * fx.Int64(2) + parity) * slot_bytes
                        )
                        acc = acc + _bf16_pack_to_f32(_load_pack(peer_slot_addr, pack))
                    reduced = acc.to(fx.BFloat16).bitcast(fx.Int32)
                    _store_pack(output_addr, pack, reduced)
                    _store_pack(local_slot_addr, pack, reduced)
            gpu.barrier()

            if tid == fx.Int32(0):
                if rank == owner:
                    _store_i64_release(local_progress_addr, result_ready)
                else:
                    owner_progress_addr = (
                        shared_addr
                        + progress_offset
                        + (fx.Int64(owner) * fx.Int64(blocks) + fx.Int64(bid))
                        * fx.Int64(8)
                    )
                    owner_progress = fx.Int64(_load_i64_acquire(owner_progress_addr))
                    while owner_progress < result_ready:
                        _sleep_one()
                        owner_progress = fx.Int64(
                            _load_i64_acquire(owner_progress_addr)
                        )
            gpu.barrier()

            if rank != owner:
                for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                    _store_pack(
                        output_addr,
                        pack,
                        _load_pack(owner_slot_addr, pack),
                    )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_mapped_rsag(
        input_addr: Int64,
        output_addr: Int64,
        shared_addr: Int64,
        data_offset: Int64,
        slot_bytes: Int64,
        launch_offset: Int64,
        progress_offset: Int64,
        rank: Int32,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        mapped_allreduce_bf16_rsag(
            input_addr,
            output_addr,
            shared_addr,
            data_offset,
            slot_bytes,
            launch_offset,
            progress_offset,
            rank,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_mapped_rsag.func.__name__ = (
        "launch_mapped_allreduce_bf16_rsag_"
        f"ws{world_size}_b{blocks}_t{threads}_p{chunk_packs}"
    )
    return launch_mapped_rsag


__all__ = [
    "make_mapped_full_launcher",
    "make_mapped_rsag_launcher",
]
