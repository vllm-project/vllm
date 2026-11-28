# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-independent CPU offloading worker.

Copy planning, transfer queues and local host-memory allocation live here.
Copies, streams/events and host registration go through the `CopyEngine` and
`DeviceRuntime` contracts; device selection, synchronization and pinned host
allocation use `torch` directly.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    DevicePointers,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.engines.base import CopyBatch, CopyEngine, CopyStream
from vllm.v1.kv_offload.cpu.platforms.base import DeviceRuntime
from vllm.v1.kv_offload.cpu.platforms.factory import get_device_runtime
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.pointer_utils import iter_pointer_groups

logger = init_logger(__name__)


class TensorRows(NamedTuple):
    """Where the rows of a 2D int8 KV tensor are: row i starts at
    `base + i * stride` and holds `size` bytes. Copy plans need nothing
    more, so they can address memory that this process has not allocated."""

    base: int
    stride: int
    size: int
    num_rows: int

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "TensorRows":
        num_rows, size = tensor.shape
        return cls(tensor.data_ptr(), tensor.stride(0), size, num_rows)


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    rows: TensorRows,
    skip_count: int = 0,
):
    """Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = rows.size // blocks_per_chunk (device page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        rows: the rows of the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.

    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = rows.base
    row_stride = rows.stride

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert rows.size % blocks_per_chunk == 0
    block_page_size = rows.size // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


class CopyPlan(NamedTuple):
    """Precomputed fragment-copy template for one data ref under the canonical
    CPU layout, unrolled from the ref's mapped runs. Offsets are relative to
    the per-block base pointers on each side."""

    frag_offsets_src: np.ndarray
    frag_offsets_dst: np.ndarray
    frag_sizes: np.ndarray
    total_bytes: int

    @property
    def num_frags(self) -> int:
        return len(self.frag_sizes)


def _build_copy_plan(ref: CanonicalKVCacheRef, device_to_cpu: bool) -> CopyPlan:
    """Unroll one data ref's mapped runs into a per-fragment CopyPlan."""
    mapping = ref.mapping
    assert mapping is not None
    local: list[int] = []
    canonical: list[int] = []
    sizes: list[int] = []
    for run in mapping.runs:
        for i in range(run.num_fragments):
            local.append(run.local_offset + i * run.local_stride)
            canonical.append(run.canonical_offset + i * run.canonical_stride)
            sizes.append(run.fragment_size)
    src, dst = (local, canonical) if device_to_cpu else (canonical, local)
    return CopyPlan(
        frag_offsets_src=np.asarray(src, dtype=np.uint64),
        frag_offsets_dst=np.asarray(dst, dtype=np.uint64),
        frag_sizes=np.asarray(sizes, dtype=np.int64),
        total_bytes=sum(sizes),
    )


def _canonical_page_ids(
    block_ids: np.ndarray, blocks_per_chunk: int, count: int, skip_count: int
) -> np.ndarray:
    """Global canonical page ids matching compute_sub_block_ptrs' enumeration.
    These identify canonical pages consistently across ranks, so they key
    CanonicalPageMapping.is_writer rotation."""
    if blocks_per_chunk == 1:
        return block_ids[:count]
    flat = (
        block_ids[:, np.newaxis] * blocks_per_chunk + np.arange(blocks_per_chunk)
    ).ravel()
    return flat[skip_count : skip_count + count]


def _canonical_block_sizes(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]], num_tensors: int
) -> list[int]:
    """Canonical CPU bytes per device block for each tensor, taken from the refs'
    mappings. Requires every ref to carry a mapping."""
    canonical_bytes_per_block = [0] * num_tensors
    for layer_refs in layer_refs_per_group:
        for ref in layer_refs:
            assert ref.mapping is not None
            canonical_bytes_per_block[ref.tensor_idx] = max(
                canonical_bytes_per_block[ref.tensor_idx],
                ref.mapping.canonical_page_size_bytes,
            )
    assert all(size > 0 for size in canonical_bytes_per_block)
    return canonical_bytes_per_block


# Bound registration size to avoid driver limits on large host allocations.
MAX_HOST_REGISTER_CHUNK_BYTES = 64 * 1024**3
# Amortize per-call overhead with the bytes not registered in each row.
MIN_HOST_REGISTER_SAVED_BYTES = 3 * 1024**2


def pin_mmap_region(
    region: SharedOffloadRegion,
    worker_slot_only: bool = False,
    should_stop: Callable[[], bool] | None = None,
    max_chunk_bytes: int | None = None,
    pace: float = 0.0,
) -> None:
    """Register row-aligned chunks or this worker's slots, rolling back on failure
    or when should_stop() turns true.

    pace > 0 sleeps that multiple of each registration's duration after it:
    back-to-back registrations hold a CUDA driver lock that starves other
    threads' CUDA calls, which stalls serving while this runs in the background.
    """
    registrar = region.registrar
    rank = region.rank
    base_ptr = region._base.data_ptr()
    total_size = region.total_size_bytes
    # Chunks end on block-row boundaries, which are page aligned, so neither the
    # driver's page rounding nor any single block transfer straddles two
    # registrations.
    max_chunk_bytes = max_chunk_bytes or MAX_HOST_REGISTER_CHUNK_BYTES
    rows_per_chunk = max(max_chunk_bytes // region._row_stride, 1)
    chunk_size = rows_per_chunk * region._row_stride
    ranges: Iterator[tuple[int, int]] = (
        (offset, min(chunk_size, total_size - offset))
        for offset in range(0, total_size, chunk_size)
    )
    stride, slot, page = region._row_stride, region.cpu_page_size, region.page_size
    min_savings = max(page, MIN_HOST_REGISTER_SAVED_BYTES)
    if worker_slot_only and region.rank is not None and stride - slot >= min_savings:
        # Round each slot out to pages; the worker's next slot is a page away.
        ranges = region.iter_worker_slot_ranges()
    else:
        reason = (
            "whole-region layout requested"
            if not worker_slot_only
            else "no worker rank"
            if region.rank is None
            else f"less than {min_savings} bytes saved per row"
        )
        logger.info(
            "Registering full CPU offload region rank=%s (row=%d, slot=%d bytes): %s",
            rank,
            stride,
            slot,
            reason,
        )

    # Register, drain and roll back through the same helper, so a failed
    # chunk leaves neither a pending error nor a partly pinned region.
    addresses: list[int] = []
    registered_bytes = 0
    for offset, size in ranges:
        address = base_ptr + offset
        t0 = time.monotonic()
        stopped = should_stop is not None and should_stop()
        if not stopped and registrar.register_host(address, size):
            addresses.append(address)
            registered_bytes += size
            if pace > 0:
                time.sleep((time.monotonic() - t0) * pace)
            continue
        if not stopped:
            logger.warning(
                "host_register failed for rank=%d at %.2f of %.2f GB; "
                "the offload region stays pageable",
                rank,
                offset / 1e9,
                total_size / 1e9,
            )
        for registered in reversed(addresses):
            registrar.unregister_host(registered)
        return

    region.pinned_addresses.extend(addresses)
    region.is_pinned = True
    logger.debug(
        "Host-registered mmap region rank=%d %.2f GB in %d chunk(s)",
        rank,
        registered_bytes / 1e9,
        len(addresses),
    )


@dataclass(slots=True)
class Transfer:
    job_id: int
    stream: CopyStream
    num_bytes: int
    batch: CopyBatch
    failed: bool = False


class CopyPlanner:
    """Computes the copy descriptors of transfers in one direction, either
    CPU->device or device->CPU, from the resolved device pointers and where the
    rows of each CPU KV tensor are."""

    def __init__(
        self,
        cpu_rows: list[TensorRows],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        device_to_cpu: bool,
        canonical_layout: bool = False,
    ):
        """Initialize a CopyPlanner.

        Args:
            cpu_rows: rows of each CPU KV cache tensor, one CPU chunk each.
            blocks_per_chunk: number of device blocks per CPU chunk.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            device_to_cpu: if True, transfer from device to CPU; otherwise CPU
                to device.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.

        """
        assert len(cpu_rows) > 0

        self.cpu_rows = cpu_rows
        self.device_to_cpu: bool = device_to_cpu
        self.layer_refs_per_group = layer_refs_per_group
        self.blocks_per_chunk = blocks_per_chunk

        # Per (group, ref) static copy plans for the canonical layout
        self._canonical_copy_plans: list[list[CopyPlan]] | None = (
            [
                [_build_copy_plan(ref, device_to_cpu) for ref in layer_refs]
                for layer_refs in layer_refs_per_group
            ]
            if canonical_layout
            else None
        )
        self._fill_group_ops = (
            self._fill_canonical_ops if canonical_layout else self._fill_direct_ops
        )
        # Reusable per-block base-pointer scratch for the canonical fill,
        # sized to the largest possible group (grown on demand)
        num_scratch_blocks = cpu_rows[0].num_rows if canonical_layout else 0
        self._scratch_bases_cpu = np.empty(num_scratch_blocks, dtype=np.uint64)

    def _estimate_max_copy_ops(self, group_sizes: Sequence[int]) -> int:
        """Upper bound on the number of copy descriptors for a transfer.

        Exact for the direct layout. The canonical path may fill fewer:
        writer rotation later drops the blocks this rank does not write."""
        num_copy_ops = 0
        for g_idx, (group_size, layer_refs) in enumerate(
            zip(group_sizes, self.layer_refs_per_group)
        ):
            if self._canonical_copy_plans is None:
                num_copy_ops += group_size * len(layer_refs)
            else:
                num_copy_ops += group_size * sum(
                    plan.num_frags for plan in self._canonical_copy_plans[g_idx]
                )
        return num_copy_ops

    def _fill_direct_ops(
        self,
        g_idx: int,
        device_ptrs: DevicePointers,
        dev_ptr_offset: int,
        cpu_block_ids: np.ndarray,
        group_size: int,
        cpu_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the direct (worker-private)
        layout: one whole-page copy per (block, ref).

        Returns (op_idx past the filled descriptors, bytes added)."""
        num_bytes = 0
        for d, data_ref in enumerate(self.layer_refs_per_group[g_idx]):
            t_idx = data_ref.tensor_idx
            end_idx = op_idx + group_size

            ptr_start = dev_ptr_offset + d * group_size
            dev_ptrs = device_ptrs.ptrs[ptr_start : ptr_start + group_size]
            dev_out, cpu_out = (
                (all_src, all_dst) if self.device_to_cpu else (all_dst, all_src)
            )
            dev_out[op_idx:end_idx] = dev_ptrs
            compute_sub_block_ptrs(
                cpu_block_ids,
                self.blocks_per_chunk,
                cpu_out[op_idx:end_idx],
                self.cpu_rows[t_idx],
                skip_count=cpu_skip_count,
            )
            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _fill_canonical_ops(
        self,
        g_idx: int,
        device_ptrs: DevicePointers,
        dev_ptr_offset: int,
        cpu_block_ids: np.ndarray,
        group_size: int,
        cpu_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the canonical layout:
        scatter each block through the ref's precomputed CopyPlan, keeping
        only the blocks this rank writes.

        Returns (op_idx past the filled descriptors, bytes added)."""
        assert self._canonical_copy_plans is not None
        if group_size > len(self._scratch_bases_cpu):
            self._scratch_bases_cpu = np.empty(group_size, dtype=np.uint64)

        num_bytes = 0
        for d, (plan, data_ref) in enumerate(
            zip(
                self._canonical_copy_plans[g_idx],
                self.layer_refs_per_group[g_idx],
            )
        ):
            if plan.num_frags == 0:
                continue
            t_idx = data_ref.tensor_idx

            # 1. Base byte pointer of every block on each side
            ptr_start = dev_ptr_offset + d * group_size
            dev_bases = device_ptrs.ptrs[ptr_start : ptr_start + group_size]
            cpu_bases = self._scratch_bases_cpu[:group_size]
            compute_sub_block_ptrs(
                cpu_block_ids,
                self.blocks_per_chunk,
                cpu_bases,
                self.cpu_rows[t_idx],
                skip_count=cpu_skip_count,
            )

            if self.device_to_cpu:
                block_bases_src = dev_bases
                block_bases_dst = cpu_bases
            else:
                block_bases_src = cpu_bases
                block_bases_dst = dev_bases

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.device_to_cpu and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    cpu_block_ids,
                    group_size,
                    cpu_skip_count,
                )
            num_active_blocks = len(block_bases_src)

            # 3. Expand (block base + fragment offset) into one descriptor
            #    per (block, fragment), writing straight into the descriptor
            #    buffers: reshaping a contiguous 1D slice is a view, so the
            #    broadcasts below allocate nothing
            end_idx = op_idx + num_active_blocks * plan.num_frags
            np.add(
                block_bases_src[:, None],
                plan.frag_offsets_src[None, :],
                out=all_src[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags),
            )
            np.add(
                block_bases_dst[:, None],
                plan.frag_offsets_dst[None, :],
                out=all_dst[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags),
            )
            all_sizes[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags)[:] = (
                plan.frag_sizes
            )
            num_bytes += num_active_blocks * plan.total_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _filter_writer_blocks(
        self,
        block_bases_src: np.ndarray,
        block_bases_dst: np.ndarray,
        mapping: CanonicalPageMapping,
        cpu_block_ids: np.ndarray,
        group_size: int,
        cpu_skip_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the blocks this rank writes: replicated ranks take turns
        writing shared canonical pages, keyed by the rank-consistent CPU-side
        canonical page id."""
        cpu_page_ids = _canonical_page_ids(
            cpu_block_ids,
            self.blocks_per_chunk,
            group_size,
            cpu_skip_count,
        )
        writer_mask = mapping.is_writer(cpu_page_ids)
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def _fill_ops(
        self,
        device_ptrs: DevicePointers,
        cpu_spec: BlockIDsLoadStoreSpec,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
    ) -> tuple[int, int]:
        """Fill the copy descriptors of one transfer.

        Returns (number of descriptors filled, bytes they copy)."""
        cpu_blocks = cpu_spec.block_ids
        assert cpu_blocks.ndim == 1
        num_cpu_blocks = len(cpu_blocks)

        cpu_offset = 0
        op_idx = 0
        num_transfer_bytes = 0
        for g in iter_pointer_groups(device_ptrs, self.blocks_per_chunk):
            cpu_end_offset = cpu_offset + g.n_chunks
            assert cpu_end_offset <= num_cpu_blocks

            op_idx, group_bytes = self._fill_group_ops(
                g.group_idx,
                device_ptrs=device_ptrs,
                dev_ptr_offset=g.dev_ptr_offset,
                cpu_block_ids=cpu_blocks[cpu_offset:cpu_end_offset],
                group_size=g.group_size,
                cpu_skip_count=g.skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes
            cpu_offset = cpu_end_offset

        assert cpu_offset == num_cpu_blocks
        assert op_idx <= len(all_sizes)
        return op_idx, num_transfer_bytes


class SingleDirectionOffloadingHandler(CopyPlanner):
    """Handles transfers for a single direction, either CPU->device or
    device->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        device_to_cpu: bool,
        canonical_layout: bool = False,
        *,
        engine: CopyEngine,
    ):
        """Initialize a SingleDirectionOffloadingHandler.

        Args:
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
            blocks_per_chunk: number of device blocks per CPU chunk.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            device_to_cpu: if True, transfer from device to CPU; otherwise CPU
                to device.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
            engine: copy engine for this direction.

        """
        for cpu_tensor in cpu_tensors:
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
        super().__init__(
            [TensorRows.of(t) for t in cpu_tensors],
            blocks_per_chunk,
            layer_refs_per_group,
            device_to_cpu,
            canonical_layout,
        )
        # Kept alive while their rows are addressed.
        self.cpu_tensors = cpu_tensors
        self._engine = engine

        # job_id -> stream
        self._job_streams: dict[int, CopyStream] = {}
        # queue of transfers
        self._transfers: deque[Transfer] = deque()
        # streams available for re-use
        self._stream_pool: list[CopyStream] = []
        # descriptor batches available for re-use
        self._buffer_pool: list[CopyBatch] = []

    def transfer_async(
        self,
        job_id: int,
        device_ptrs: DevicePointers,
        cpu_spec: BlockIDsLoadStoreSpec,
    ) -> bool:
        num_copy_ops = self._estimate_max_copy_ops(device_ptrs.group_block_counts)

        # reuse a pooled batch, growing it if this transfer needs more room
        batch = self._buffer_pool.pop() if self._buffer_pool else None
        if batch is None or batch.capacity < num_copy_ops:
            batch = self._engine.new_batch(num_copy_ops)

        op_idx, num_transfer_bytes = self._fill_ops(
            device_ptrs, cpu_spec, batch.src_ptrs, batch.dst_ptrs, batch.sizes
        )
        # Writer rotation may skip non-writer blocks, leaving op_idx below
        # the sized upper bound
        assert op_idx <= batch.capacity

        stream = (
            self._stream_pool.pop() if self._stream_pool else self._engine.new_stream()
        )

        # Stores must wait for the model to finish writing the KV they read.
        # Loads must wait for pending writes (including zeroing) to their
        # destination blocks: the scheduler leaves a block unzeroed only in
        # the step that allocates it for the load and cannot retract zeroing
        # shipped in an earlier step for a since-reallocated block; with async
        # scheduling nothing else orders that zeroing against this copy.
        stream.wait_for_compute()
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            # assure job will start only after the previous one completes
            stream.wait_for(last_transfer.stream)

        failed = False
        try:
            stream.submit(batch, op_idx)
        except RuntimeError:
            logger.exception(
                "KV offload %s transfer failed for job %d",
                "device->CPU" if self.device_to_cpu else "CPU->device",
                job_id,
            )
            stream.abort()
            failed = True

        self._job_streams[job_id] = stream
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                num_bytes=0 if failed else num_transfer_bytes,
                batch=batch,
                failed=failed,
            )
        )

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers:
            transfer = self._transfers[0]
            try:
                if not transfer.stream.done():
                    break
                transfer_time = (
                    None if transfer.failed else transfer.stream.elapsed_seconds()
                )
            except RuntimeError:
                logger.exception(
                    "KV offload %s transfer failed for job %d while polling "
                    "for completion",
                    "device->CPU" if self.device_to_cpu else "CPU->device",
                    transfer.job_id,
                )
                self._transfers.popleft()
                self._job_streams.pop(transfer.job_id, None)
                results.append(TransferResult(job_id=transfer.job_id, success=False))
                continue

            self._transfers.popleft()
            self._job_streams.pop(transfer.job_id, None)
            if transfer.failed:
                results.append(TransferResult(job_id=transfer.job_id, success=False))
                continue

            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_time=transfer_time,
                )
            )
            self._stream_pool.append(transfer.stream)
            self._buffer_pool.append(transfer.batch)
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            stream = self._job_streams.get(job_id)
            if stream is not None:
                try:
                    stream.synchronize()
                except RuntimeError:
                    logger.exception(
                        "KV offload transfer for job %d failed while waiting "
                        "for it to complete",
                        job_id,
                    )

    def shutdown(self) -> None:
        """Drain this direction and release its transfer-side resources."""
        sync_error: Exception | None = None
        while self._transfers:
            transfer = self._transfers[0]
            try:
                transfer.stream.synchronize()
            except Exception as e:
                logger.exception(
                    "Failed to synchronize transfer end event; "
                    "skipping %d remaining transfers",
                    len(self._transfers) - 1,
                )
                self._transfers.clear()
                sync_error = e
                break
            self._transfers.popleft()

        self._job_streams.clear()
        self._stream_pool.clear()
        self._buffer_pool.clear()
        self.cpu_tensors.clear()
        if sync_error is not None:
            raise sync_error


# Pre-fault step, so shutdown can stop a deferred setup between steps.
_PREFAULT_STEP_BYTES = 1 << 30
# Deferred pinning registers small chunks (never less than one block row or
# worker slot) to shorten each hold of the CUDA driver lock while serving.
_DEFERRED_PIN_CHUNK_BYTES = 8 << 20


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    _setup_thread: threading.Thread | None = None

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
        defer_host_setup: bool = False,
        prefault_shard: tuple[int, int] | None = None,
        pin_pace: float = 2.0,
        runtime: DeviceRuntime | None = None,
    ):
        """defer_host_setup leaves pre-faulting and pinning mmap_region to
        start_deferred_setup(); prefault_shard is then the share this worker
        pre-faults first, and pin_pace paces its pinning (see pin_mmap_region)."""
        assert not canonical_layout or mmap_region is not None
        self._runtime = runtime if runtime is not None else get_device_runtime()
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        self._pin_memory = pin_memory
        self._worker_slot_only = not canonical_layout
        self._defer_host_setup = defer_host_setup and mmap_region is not None
        self._prefault_shard = prefault_shard
        self._pin_pace = pin_pace
        self._setup_stop = threading.Event()
        self._ready = threading.Event()
        if self._defer_host_setup:
            self._device = torch.accelerator.current_device_index()
        else:
            self._ready.set()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory and not self._defer_host_setup:
            pin_mmap_region(mmap_region, worker_slot_only=not canonical_layout)
        # A deferred region is pinned before is_ready(), or never used.
        host_memory_is_pinned = pin_memory and (
            mmap_region is None or mmap_region.is_pinned or self._defer_host_setup
        )

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            device_page_size_bytes = kv_cache_tensor.page_size_bytes
            cpu_page_size_bytes = device_page_size_bytes * blocks_per_chunk

            if canonical_bytes_per_block is not None:
                assert mmap_region is not None
                cpu_tensor = mmap_region.create_next_canonical_view(
                    canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            elif mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_chunks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_chunks,
                    cpu_page_size_bytes,
                    num_cpu_chunks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            cpu_tensors.append(cpu_tensor)

        page_sizes = [
            r.page_size_bytes for refs in kv_caches.group_data_refs for r in refs
        ]

        def make_handler(device_to_cpu: bool) -> SingleDirectionOffloadingHandler:
            return SingleDirectionOffloadingHandler(
                cpu_tensors=cpu_tensors,
                blocks_per_chunk=blocks_per_chunk,
                layer_refs_per_group=kv_caches.group_data_refs,
                device_to_cpu=device_to_cpu,
                canonical_layout=canonical_layout,
                engine=self._runtime.create_copy_engine(
                    device_to_cpu=device_to_cpu,
                    host_memory_is_pinned=host_memory_is_pinned,
                    page_sizes=page_sizes,
                ),
            )

        self._store_handler = make_handler(device_to_cpu=True)
        self._load_handler = make_handler(device_to_cpu=False)
        # Own stream, so expedited loads do not queue behind regular loads.
        self._expedited_load_handler = make_handler(device_to_cpu=False)

    def start_deferred_setup(self) -> None:
        if self._defer_host_setup and self._setup_thread is None:
            self._setup_thread = threading.Thread(
                target=self._host_setup, name="vllm_offload_host_setup", daemon=True
            )
            self._setup_thread.start()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def _host_setup(self) -> None:
        region = self._mmap_region
        assert region is not None
        t0 = time.monotonic()
        try:
            torch.accelerator.set_device_index(self._device)
            self._runtime.relax_stream_capture()
            for offset, length in region.prefault_ranges(self._prefault_shard):
                for start in range(offset, offset + length, _PREFAULT_STEP_BYTES):
                    if self._setup_stop.is_set():
                        return
                    end = min(start + _PREFAULT_STEP_BYTES, offset + length)
                    region.prefault(start, end - start)
            t1 = time.monotonic()
            if self._pin_memory:
                pin_mmap_region(
                    region,
                    worker_slot_only=self._worker_slot_only,
                    should_stop=self._setup_stop.is_set,
                    max_chunk_bytes=_DEFERRED_PIN_CHUNK_BYTES,
                    pace=self._pin_pace,
                )
                if not region.is_pinned:
                    if not self._setup_stop.is_set():
                        logger.error(
                            "Pinning the CPU offload region failed; "
                            "offloading stays off"
                        )
                    return
        except Exception:
            logger.exception("CPU offload setup failed; offloading stays off")
            return
        logger.info(
            "CPU offload region set up in the background: pre-fault %.1f s, pin %.1f s",
            t1 - t0,
            time.monotonic() - t1,
        )
        self._ready.set()

    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async device -> CPU."""
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        return self._store_handler.transfer_async(job_id, device_ptrs, dst_spec)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        device_ptrs: DevicePointers,
        expedited: bool = False,
    ) -> bool:
        """Async CPU -> device."""
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        handler = self._expedited_load_handler if expedited else self._load_handler
        return handler.transfer_async(job_id, device_ptrs, src_spec)

    def get_finished(self) -> list[TransferResult]:
        return (
            self._store_handler.get_finished()
            + self._load_handler.get_finished()
            + self._expedited_load_handler.get_finished()
        )

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)
        self._expedited_load_handler.wait(job_ids)

    def shutdown(self) -> None:
        # The region must outlive the setup thread that touches it.
        if self._setup_thread is not None:
            self._setup_stop.set()
            self._setup_thread.join()
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        for handler in (self._load_handler, self._expedited_load_handler):
            try:
                handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down load offloading handler")
                handler_failed = True

        if self._mmap_region is not None:
            if handler_failed:
                try:
                    torch.accelerator.synchronize()
                except Exception:
                    logger.warning(
                        "Device sync before mmap cleanup failed; "
                        "proceeding with cleanup anyway",
                        exc_info=True,
                    )
            self._mmap_region.cleanup()
            self._mmap_region = None
