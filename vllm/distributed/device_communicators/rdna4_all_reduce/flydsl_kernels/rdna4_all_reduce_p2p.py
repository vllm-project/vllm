# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Direct-peer FlyDSL kernel families for TP4 and TP8 RDNA4 all-reduce."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, Stream

from .common import MAX_BLOCKS, global_pointer
from .common import load_pack_128b as _load_pack
from .common import store_pack_128b as _store_pack

_PACK_BYTES = 16
_PACK_ELEMENTS = 8
_SG_START_OFFSET = 0
_SG_END_OFFSET = MAX_BLOCKS * 8 * 4
_SG_FLAG_OFFSET = MAX_BLOCKS * 8 * 4 * 2


def _load_i32_acquire(address):
    return fx.generic_load(
        global_pointer(address, fx.Int32, 4),
        memory_order=fx.AtomicOrdering.Acquire,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i32_release(address, value):
    fx.generic_store(
        global_pointer(address, fx.Int32, 4),
        value,
        memory_order=fx.AtomicOrdering.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _load_pointer(array_address, index):
    address = array_address + fx.Int64(index * 8)
    return fx.generic_load(global_pointer(address, fx.Int64, 8))


@flyc.jit
def _sync(
    *,
    signal_ptrs,
    self_signal,
    rank,
    block,
    thread,
    world_size: int,
    table_offset: int,
):
    flag_address = (
        self_signal + fx.Int64(_SG_FLAG_OFFSET) + fx.Int64(block) * fx.Int64(4)
    )
    ticket = fx.Int32(_load_i32_acquire(flag_address)) + fx.Int32(1)
    block_slot = block * fx.Int32(8)

    for peer in range_constexpr(world_size):
        if thread == fx.Int32(peer):
            peer_address = (
                signal_ptrs[peer]
                + fx.Int64(table_offset)
                + fx.Int64(block_slot + rank) * fx.Int64(4)
            )
            _store_i32_release(peer_address, ticket)
            local_address = (
                self_signal
                + fx.Int64(table_offset)
                + fx.Int64(block_slot + fx.Int32(peer)) * fx.Int64(4)
            )
            observed = fx.Int32(_load_i32_acquire(local_address))
            while observed < ticket:
                observed = fx.Int32(_load_i32_acquire(local_address))

    gpu.barrier()
    if thread == fx.Int32(0):
        _store_i32_release(flag_address, ticket)
    gpu.barrier()


@flyc.jit
def _publish_progress(
    *,
    signal_ptrs,
    rank,
    block,
    thread,
    ticket,
    table_offset: int,
):
    block_slot = block * fx.Int32(8)
    for peer in range_constexpr(4):
        if thread == fx.Int32(peer):
            peer_address = (
                signal_ptrs[peer]
                + fx.Int64(table_offset)
                + fx.Int64(block_slot + rank) * fx.Int64(4)
            )
            _store_i32_release(peer_address, ticket)


@flyc.jit
def _wait_progress(
    *,
    self_signal,
    block,
    thread,
    ticket,
    table_offset: int,
):
    block_slot = block * fx.Int32(8)
    for peer in range_constexpr(4):
        if thread == fx.Int32(peer):
            local_address = (
                self_signal
                + fx.Int64(table_offset)
                + fx.Int64(block_slot + fx.Int32(peer)) * fx.Int64(4)
            )
            observed = fx.Int32(_load_i32_acquire(local_address))
            while observed < ticket:
                observed = fx.Int32(_load_i32_acquire(local_address))
    gpu.barrier()


@flyc.jit
def _sync_strided_group(
    *,
    signal_ptrs_address,
    self_signal,
    rank,
    block,
    thread,
    group_base,
    group_size: int,
    rank_stride: int,
    table_offset: int,
):
    """Synchronize a compile-time-sized, strided subset of absolute ranks."""
    flag_address = (
        self_signal + fx.Int64(_SG_FLAG_OFFSET) + fx.Int64(block) * fx.Int64(4)
    )
    ticket = fx.Int32(_load_i32_acquire(flag_address)) + fx.Int32(1)
    block_slot = block * fx.Int32(8)

    for group_index in range_constexpr(group_size):
        if thread == fx.Int32(group_index):
            peer_rank = group_base + fx.Int32(group_index * rank_stride)
            peer_signal = _load_pointer(signal_ptrs_address, peer_rank)
            peer_address = (
                peer_signal
                + fx.Int64(table_offset)
                + fx.Int64(block_slot + rank) * fx.Int64(4)
            )
            _store_i32_release(peer_address, ticket)
            local_address = (
                self_signal
                + fx.Int64(table_offset)
                + fx.Int64(block_slot + peer_rank) * fx.Int64(4)
            )
            observed = fx.Int32(_load_i32_acquire(local_address))
            while observed < ticket:
                observed = fx.Int32(_load_i32_acquire(local_address))

    gpu.barrier()
    if thread == fx.Int32(0):
        _store_i32_release(flag_address, ticket)
    gpu.barrier()


@cache
def make_p2p_tp4_push_rsag_launcher(
    *,
    blocks: int,
    threads: int = 512,
    copy_load_nontemporal: bool = False,
):
    """Build the graph-only, direct-output TP4 BF16 push-RSAG launcher."""
    world_size = 4
    if not 0 < blocks <= MAX_BLOCKS:
        raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
    if threads not in (256, 512, 1024):
        raise ValueError(f"threads must be 256, 512, or 1024, got {threads}")
    if threads % world_size:
        raise ValueError("threads must be divisible by world_size")
    threads_per_peer = threads // world_size

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def p2p_tp4_push_rsag_bf16(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_address: Int64,
        receive_ptrs_address: Int64,
        tmp_ptrs_address: Int64,
        input_address: Int64,
        numel: Int32,
    ):
        thread = fx.thread_idx.x
        block = fx.block_idx.x
        signal_ptrs = [
            _load_pointer(signal_ptrs_address, peer) for peer in range(world_size)
        ]
        receive_ptrs = [
            _load_pointer(receive_ptrs_address, peer) for peer in range(world_size)
        ]
        tmp_ptrs = [_load_pointer(tmp_ptrs_address, peer) for peer in range(world_size)]
        pack_count = numel // fx.Int32(_PACK_ELEMENTS)
        part_packs = pack_count // fx.Int32(world_size)
        peer_group = thread // fx.Int32(threads_per_peer)
        peer_lane = thread & fx.Int32(threads_per_peer - 1)
        relative_start = block * fx.Int32(threads_per_peer) + peer_lane
        stride = fx.Int32(blocks * threads_per_peer)
        rank_slot_start = rank * part_packs

        # Push each local input partition into its owner's scratch allocation.
        # The destination offset is keyed by sender rank, so every owner sees
        # all contributions in contiguous, rank-ordered slots in local HBM.
        for owner_offset in range_constexpr(world_size):
            if peer_group == fx.Int32(owner_offset):
                owner = (rank + fx.Int32(owner_offset)) & fx.Int32(world_size - 1)
                owner_input_start = owner * part_packs
                owner_tmp = tmp_ptrs[owner_offset]
                for relative_pack in range(relative_start, part_packs, stride):
                    value = _load_pack(
                        input_address,
                        owner_input_start + relative_pack,
                        nontemporal=copy_load_nontemporal,
                    )
                    _store_pack(
                        owner_tmp,
                        rank_slot_start + relative_pack,
                        value,
                        nontemporal=owner_offset != 0,
                    )

        gpu.barrier()
        _sync(
            signal_ptrs=signal_ptrs,
            self_signal=self_signal,
            rank=rank,
            block=block,
            thread=thread,
            world_size=world_size,
            table_offset=_SG_START_OFFSET,
        )

        # Split the destinations across thread groups. Each group independently
        # reduces the rank-owned partition from local HBM and writes directly
        # to one destination. Re-reading the local contributions is cheaper on
        # this PCIe topology than serialising all peer stores behind one group.
        local_tmp = tmp_ptrs[0]
        local_partition_start = rank * part_packs
        for peer in range_constexpr(world_size):
            if peer_group == fx.Int32(peer):
                for relative_pack in range(relative_start, part_packs, stride):
                    accumulator = (
                        _load_pack(local_tmp, relative_pack)
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    for sender in range_constexpr(1, world_size):
                        value = (
                            _load_pack(
                                local_tmp,
                                fx.Int32(sender) * part_packs + relative_pack,
                            )
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        accumulator = accumulator + value
                    reduced = accumulator.to(fx.BFloat16).bitcast(fx.Int32)
                    _store_pack(
                        receive_ptrs[peer],
                        local_partition_start + relative_pack,
                        reduced,
                        nontemporal=peer != 0,
                    )

        gpu.barrier()
        _sync(
            signal_ptrs=signal_ptrs,
            self_signal=self_signal,
            rank=rank,
            block=block,
            thread=thread,
            world_size=world_size,
            table_offset=_SG_END_OFFSET,
        )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_p2p_tp4_push_rsag(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_address: Int64,
        receive_ptrs_address: Int64,
        tmp_ptrs_address: Int64,
        input_address: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        p2p_tp4_push_rsag_bf16(
            rank,
            self_signal,
            signal_ptrs_address,
            receive_ptrs_address,
            tmp_ptrs_address,
            input_address,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    load_suffix = "_loadnt" if copy_load_nontemporal else ""
    launch_p2p_tp4_push_rsag.func.__name__ = (
        f"launch_p2p_tp4_push_rsag_bf16_b{blocks}_t{threads}{load_suffix}_direct"
    )
    return launch_p2p_tp4_push_rsag


@cache
def make_p2p_tp4_pull_launcher(
    *,
    blocks: int,
    threads: int = 512,
    chunk_bytes: int = 32 * 1024 * 1024,
    tail_safe: bool = False,
):
    """Build the graph-only TP4 BF16 two-shot peer-pull launcher."""
    world_size = 4
    if not 0 < blocks <= MAX_BLOCKS:
        raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
    if threads not in (256, 512, 1024):
        raise ValueError(f"threads must be 256, 512, or 1024, got {threads}")
    chunk_packs = chunk_bytes // _PACK_BYTES
    if chunk_bytes % _PACK_BYTES or chunk_packs % (blocks * threads):
        raise ValueError("chunk_bytes must be a multiple of blocks * threads * 16")
    if chunk_packs % world_size:
        raise ValueError("two-shot chunks must split evenly across ranks")

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def p2p_tp4_pull_bf16(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_address: Int64,
        output_ptrs_address: Int64,
        input_ptrs_address: Int64,
        output_address: Int64,
        numel: Int32,
    ):
        thread = fx.thread_idx.x
        block = fx.block_idx.x
        signal_ptrs = [
            _load_pointer(signal_ptrs_address, peer) for peer in range(world_size)
        ]
        pack_count = numel // fx.Int32(_PACK_ELEMENTS)
        relative_pack_start = block * fx.Int32(threads) + thread
        stride = fx.Int32(blocks * threads)
        chunk_count = (pack_count + fx.Int32(chunk_packs - 1)) // fx.Int32(chunk_packs)
        flag_address = (
            self_signal + fx.Int64(_SG_FLAG_OFFSET) + fx.Int64(block) * fx.Int64(4)
        )
        base_ticket = fx.Int32(_load_i32_acquire(flag_address))

        # Captured direct inputs are IPC-registered after graph capture.
        # Publishing entry proves that peer streams completed their producers.
        _publish_progress(
            signal_ptrs=signal_ptrs,
            rank=rank,
            block=block,
            thread=thread,
            ticket=base_ticket + chunk_count,
            table_offset=_SG_START_OFFSET,
        )

        for chunk in range(chunk_count):
            _wait_progress(
                self_signal=self_signal,
                block=block,
                thread=thread,
                ticket=base_ticket + chunk + fx.Int32(1),
                table_offset=_SG_START_OFFSET,
            )

            # Rotate rank-relative pointers back to absolute rank order. Loading
            # them after staging keeps four 64-bit values out of that live set.
            rank0 = _load_pointer(
                input_ptrs_address,
                (fx.Int32(0) - rank) & fx.Int32(world_size - 1),
            )
            rank1 = _load_pointer(
                input_ptrs_address,
                (fx.Int32(1) - rank) & fx.Int32(world_size - 1),
            )
            rank2 = _load_pointer(
                input_ptrs_address,
                (fx.Int32(2) - rank) & fx.Int32(world_size - 1),
            )
            rank3 = _load_pointer(
                input_ptrs_address,
                (fx.Int32(3) - rank) & fx.Int32(world_size - 1),
            )
            chunk_start = chunk * fx.Int32(chunk_packs)
            chunk_limit = chunk_start + fx.Int32(chunk_packs)
            chunk_end = (pack_count < chunk_limit).select(pack_count, chunk_limit)
            if const_expr(tail_safe):
                wave_packs = fx.Int32(32)
                part_packs = (
                    (chunk_end - chunk_start) // fx.Int32(world_size * 32) * wave_packs
                )
                remainder_packs = (
                    chunk_end - chunk_start - part_packs * fx.Int32(world_size)
                )
            else:
                part_packs = (chunk_end - chunk_start) // fx.Int32(world_size)
            owner_start = chunk_start + rank * part_packs
            if const_expr(tail_safe):
                owner_packs = (rank == fx.Int32(world_size - 1)).select(
                    part_packs + remainder_packs,
                    part_packs,
                )
            else:
                owner_packs = part_packs
            for relative_pack in range(
                relative_pack_start,
                owner_packs,
                stride,
            ):
                pack = owner_start + relative_pack
                value0 = _load_pack(rank0, pack)
                value1 = _load_pack(rank1, pack)
                value2 = _load_pack(rank2, pack)
                value3 = _load_pack(rank3, pack)
                accumulator = value0.bitcast(fx.BFloat16).to(fx.Float32)
                accumulator = accumulator + value1.bitcast(fx.BFloat16).to(fx.Float32)
                accumulator = accumulator + value2.bitcast(fx.BFloat16).to(fx.Float32)
                accumulator = accumulator + value3.bitcast(fx.BFloat16).to(fx.Float32)
                _store_pack(
                    output_address,
                    pack,
                    accumulator.to(fx.BFloat16).bitcast(fx.Int32),
                )

            gpu.barrier()
            _publish_progress(
                signal_ptrs=signal_ptrs,
                rank=rank,
                block=block,
                thread=thread,
                ticket=base_ticket + chunk + fx.Int32(1),
                table_offset=_SG_END_OFFSET,
            )
            _wait_progress(
                self_signal=self_signal,
                block=block,
                thread=thread,
                ticket=base_ticket + chunk + fx.Int32(1),
                table_offset=_SG_END_OFFSET,
            )

            peer1_output = _load_pointer(output_ptrs_address, 1)
            peer2_output = _load_pointer(output_ptrs_address, 2)
            peer3_output = _load_pointer(output_ptrs_address, 3)
            owner1_start = (
                chunk_start
                + ((rank + fx.Int32(1)) & fx.Int32(world_size - 1)) * part_packs
            )
            owner2_start = (
                chunk_start
                + ((rank + fx.Int32(2)) & fx.Int32(world_size - 1)) * part_packs
            )
            owner3_start = (
                chunk_start
                + ((rank + fx.Int32(3)) & fx.Int32(world_size - 1)) * part_packs
            )
            for relative_pack in range(
                relative_pack_start,
                part_packs,
                stride,
            ):
                value1 = _load_pack(peer1_output, owner1_start + relative_pack)
                value2 = _load_pack(peer2_output, owner2_start + relative_pack)
                value3 = _load_pack(peer3_output, owner3_start + relative_pack)
                _store_pack(output_address, owner1_start + relative_pack, value1)
                _store_pack(output_address, owner2_start + relative_pack, value2)
                _store_pack(output_address, owner3_start + relative_pack, value3)

            if const_expr(tail_safe):
                remainder_start = chunk_start + part_packs * fx.Int32(world_size)
                rank3_output = _load_pointer(
                    output_ptrs_address,
                    (fx.Int32(world_size - 1) - rank) & fx.Int32(world_size - 1),
                )
                for relative_pack in range(
                    relative_pack_start,
                    remainder_packs,
                    stride,
                ):
                    value = _load_pack(
                        rank3_output,
                        remainder_start + relative_pack,
                    )
                    _store_pack(
                        output_address,
                        remainder_start + relative_pack,
                        value,
                    )

        gpu.barrier()
        completion_ticket = base_ticket + chunk_count + fx.Int32(1)
        _publish_progress(
            signal_ptrs=signal_ptrs,
            rank=rank,
            block=block,
            thread=thread,
            ticket=completion_ticket,
            table_offset=_SG_END_OFFSET,
        )
        _wait_progress(
            self_signal=self_signal,
            block=block,
            thread=thread,
            ticket=completion_ticket,
            table_offset=_SG_END_OFFSET,
        )
        if thread == fx.Int32(0):
            _store_i32_release(flag_address, completion_ticket)

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_p2p_tp4_pull(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_address: Int64,
        output_ptrs_address: Int64,
        input_ptrs_address: Int64,
        output_address: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        p2p_tp4_pull_bf16(
            rank,
            self_signal,
            signal_ptrs_address,
            output_ptrs_address,
            input_ptrs_address,
            output_address,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    tail_suffix = "_tail" if tail_safe else ""
    launch_p2p_tp4_pull.func.__name__ = (
        f"launch_p2p_tp4_pull_bf16_b{blocks}_t{threads}_c{chunk_bytes}"
        f"_twoshot{tail_suffix}_directinput"
    )
    return launch_p2p_tp4_pull


@cache
def make_p2p_hierarchical_tp8_launcher(
    *,
    blocks: int,
    threads: int = 512,
    local_copy_threads: int = 0,
):
    """Build a topology-aware direct-output TP8 BF16 all-reduce launcher."""
    world_size = 8
    local_size = 4
    if not 0 < blocks <= MAX_BLOCKS:
        raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
    if threads not in (256, 512, 1024) or threads % local_size:
        raise ValueError("threads must be 256, 512, or 1024")
    if not local_copy_threads:
        local_copy_threads = threads
    if (
        local_copy_threads not in (256, 512, 1024)
        or local_copy_threads > threads
        or local_copy_threads % local_size
    ):
        raise ValueError("local_copy_threads must be 256, 512, or 1024 and <= threads")
    threads_per_peer = local_copy_threads // local_size

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def p2p_hierarchical_tp8_bf16(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_address: Int64,
        input_ptrs_address: Int64,
        tmp_ptrs_address: Int64,
        input_address: Int64,
        output_ptrs_address: Int64,
        numel: Int32,
    ):
        thread = fx.thread_idx.x
        block = fx.block_idx.x
        pack_count = numel // fx.Int32(_PACK_ELEMENTS)
        part_packs = pack_count // fx.Int32(local_size)
        local_rank = rank & fx.Int32(local_size - 1)
        half_base = rank - local_rank
        peer_group = thread // fx.Int32(threads_per_peer)
        peer_lane = thread & fx.Int32(threads_per_peer - 1)
        relative_start = block * fx.Int32(threads_per_peer) + peer_lane
        grouped_stride = fx.Int32(blocks * threads_per_peer)

        # Reduce-scatter staging stays within each four-GPU NUMA half.
        sender_slot_start = local_rank * part_packs
        for owner_step in range_constexpr(local_size):
            if thread < fx.Int32(local_copy_threads):  # noqa: SIM102
                if peer_group == fx.Int32(owner_step):
                    owner_index = (local_rank + fx.Int32(owner_step)) & fx.Int32(
                        local_size - 1
                    )
                    owner_rank = half_base + owner_index
                    owner_delta = owner_rank - rank
                    owner_offset = (owner_delta < fx.Int32(0)).select(
                        owner_delta + fx.Int32(world_size),
                        owner_delta,
                    )
                    owner_tmp = _load_pointer(tmp_ptrs_address, owner_offset)
                    owner_input_start = owner_index * part_packs
                    for relative_pack in range(
                        relative_start, part_packs, grouped_stride
                    ):
                        value = _load_pack(
                            input_address,
                            owner_input_start + relative_pack,
                        )
                        _store_pack(
                            owner_tmp,
                            sender_slot_start + relative_pack,
                            value,
                            nontemporal=owner_step != 0,
                        )
        gpu.barrier()
        _sync_strided_group(
            signal_ptrs_address=signal_ptrs_address,
            self_signal=self_signal,
            rank=rank,
            block=block,
            thread=thread,
            group_base=half_base,
            group_size=local_size,
            rank_stride=1,
            table_offset=_SG_START_OFFSET,
        )

        # Each owner reduces one quarter locally, then exchanges only that
        # partial with the corresponding owner in the other NUMA half.
        local_tmp = _load_pointer(tmp_ptrs_address, 0)
        local_cross = _load_pointer(input_ptrs_address, 0)
        remote_cross = _load_pointer(input_ptrs_address, 4)
        all_start = block * fx.Int32(threads) + thread
        all_stride = fx.Int32(blocks * threads)
        for relative_pack in range(all_start, part_packs, all_stride):
            accumulator = (
                _load_pack(local_tmp, relative_pack).bitcast(fx.BFloat16).to(fx.Float32)
            )
            for sender in range_constexpr(1, local_size):
                value = (
                    _load_pack(
                        local_tmp,
                        fx.Int32(sender) * part_packs + relative_pack,
                    )
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                accumulator = accumulator + value
            partial = accumulator.to(fx.BFloat16).bitcast(fx.Int32)
            _store_pack(local_cross, relative_pack, partial)
            _store_pack(
                remote_cross,
                part_packs + relative_pack,
                partial,
                nontemporal=True,
            )

        gpu.barrier()
        _sync_strided_group(
            signal_ptrs_address=signal_ptrs_address,
            self_signal=self_signal,
            rank=rank,
            block=block,
            thread=thread,
            group_base=local_rank,
            group_size=2,
            rank_stride=local_size,
            table_offset=_SG_END_OFFSET,
        )

        # Both paired owners now have the two half reductions. Combine and
        # push their quarter directly to all four outputs in their local half.
        partition_start = local_rank * part_packs
        for target_step in range_constexpr(local_size):
            if thread < fx.Int32(local_copy_threads):  # noqa: SIM102
                if peer_group == fx.Int32(target_step):
                    target_index = (local_rank + fx.Int32(target_step)) & fx.Int32(
                        local_size - 1
                    )
                    target_rank = half_base + target_index
                    target_delta = target_rank - rank
                    target_offset = (target_delta < fx.Int32(0)).select(
                        target_delta + fx.Int32(world_size),
                        target_delta,
                    )
                    target_output = _load_pointer(output_ptrs_address, target_offset)
                    for relative_pack in range(
                        relative_start, part_packs, grouped_stride
                    ):
                        first = (
                            _load_pack(local_cross, relative_pack)
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        second = (
                            _load_pack(local_cross, part_packs + relative_pack)
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        reduced = (first + second).to(fx.BFloat16).bitcast(fx.Int32)
                        _store_pack(
                            target_output,
                            partition_start + relative_pack,
                            reduced,
                            nontemporal=target_step != 0,
                        )

        gpu.barrier()
        _sync_strided_group(
            signal_ptrs_address=signal_ptrs_address,
            self_signal=self_signal,
            rank=rank,
            block=block,
            thread=thread,
            group_base=half_base,
            group_size=local_size,
            rank_stride=1,
            table_offset=_SG_START_OFFSET,
        )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_p2p_hierarchical_tp8(
        rank: Int32,
        self_signal: Int64,
        signal_ptrs_address: Int64,
        input_ptrs_address: Int64,
        tmp_ptrs_address: Int64,
        input_address: Int64,
        output_ptrs_address: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        p2p_hierarchical_tp8_bf16(
            rank,
            self_signal,
            signal_ptrs_address,
            input_ptrs_address,
            tmp_ptrs_address,
            input_address,
            output_ptrs_address,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    local_copy_suffix = (
        f"_copythreads{local_copy_threads}" if local_copy_threads != threads else ""
    )
    launch_p2p_hierarchical_tp8.func.__name__ = (
        f"launch_p2p_hierarchical_tp8_b{blocks}_t{threads}{local_copy_suffix}"
    )
    return launch_p2p_hierarchical_tp8


__all__ = [
    "make_p2p_hierarchical_tp8_launcher",
    "make_p2p_tp4_pull_launcher",
    "make_p2p_tp4_push_rsag_launcher",
]
