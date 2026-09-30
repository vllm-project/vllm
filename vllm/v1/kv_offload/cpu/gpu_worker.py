# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    CopyRun,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister
from vllm.v1.kv_offload.cpu.copy_backend import (
    CopyBackendAdapter,
    CopyRunDescriptor,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.

    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def _build_run_plans(ref: CanonicalKVCacheRef) -> tuple[CopyRun, ...]:
    """Build structured copy runs for one data ref."""
    mapping = ref.mapping
    if mapping is None:
        page_size = ref.page_size_bytes
        return (CopyRun(0, 0, page_size, 1, page_size, page_size),)

    for run in mapping.runs:
        assert run.num_fragments >= 0
    return mapping.runs


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
    """Canonical CPU bytes per GPU block for each tensor, taken from the refs'
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


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register row-aligned chunks, rolling back on failure."""
    rank = region.rank
    base_ptr = region._base.data_ptr()
    total_size = region.total_size_bytes
    # Chunks end on block-row boundaries, which are page aligned, so neither the
    # driver's page rounding nor any single block transfer straddles two
    # registrations.
    rows_per_chunk = max(MAX_HOST_REGISTER_CHUNK_BYTES // region._row_stride, 1)
    chunk_size = rows_per_chunk * region._row_stride

    # Register, drain and roll back through the same helper, so a failed
    # chunk leaves neither a pending error nor a partly pinned region.
    addresses: list[int] = []
    for offset in range(0, total_size, chunk_size):
        address = base_ptr + offset
        size = min(chunk_size, total_size - offset)
        if host_register(address, size):
            addresses.append(address)
            continue
        logger.warning(
            "host_register failed for rank=%d at %.2f of %.2f GB; "
            "the offload region stays pageable",
            rank,
            offset / 1e9,
            total_size / 1e9,
        )
        for registered in reversed(addresses):
            host_unregister(registered)
        return

    region.pinned_addresses.extend(addresses)
    region.is_pinned = True
    logger.debug(
        "Host-registered mmap region rank=%d %.2f GB in %d chunk(s)",
        rank,
        total_size / 1e9,
        len(addresses),
    )


class SingleDirectionOffloadingHandler:
    """Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        canonical_layout: bool = False,
        host_memory_is_pinned: bool = True,
    ):
        """Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_chunks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            blocks_per_chunk: number of blocks transferred per chunk.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
            host_memory_is_pinned: whether the CPU tensors are pinned, so GPU
                kernels may dereference them directly.

        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        canonical_bytes_per_block = (
            _canonical_block_sizes(layer_refs_per_group, len(gpu_tensors))
            if canonical_layout
            else None
        )

        # assert input tensors are as expected
        for t_idx, (gpu_tensor, cpu_tensor) in enumerate(zip(gpu_tensors, cpu_tensors)):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            if canonical_bytes_per_block is not None:
                assert (
                    cpu_page_size == canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            else:
                assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.layer_refs_per_group = layer_refs_per_group

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

        # Keep canonical runs structured until the backend consumes them.
        # Non-canonical refs use one whole-page run per layer.
        self._copy_runs: list[list[tuple[CopyRun, ...]]] = [
            [_build_run_plans(ref) for ref in layer_refs]
            for layer_refs in layer_refs_per_group
        ]
        self._backend = CopyBackendAdapter(
            layer_refs_per_group=layer_refs_per_group,
            gpu_to_cpu=gpu_to_cpu,
            host_memory_is_pinned=host_memory_is_pinned,
            copy_runs=self._copy_runs,
        )
        # Reusable per-block base-pointer scratch for structured run filling.
        num_scratch_blocks = gpu_tensors[0].shape[0]
        self._scratch_bases_src = np.empty(num_scratch_blocks, dtype=np.uint64)
        self._scratch_bases_dst = np.empty(num_scratch_blocks, dtype=np.uint64)

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []

    def _fill_run_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
    ) -> tuple[list[CopyRunDescriptor], int]:
        """Build logical runs and defer tensor materialization to the adapter."""
        if group_size > len(self._scratch_bases_src):
            self._scratch_bases_src = np.empty(group_size, dtype=np.uint64)
            self._scratch_bases_dst = np.empty(group_size, dtype=np.uint64)

        run_descs: list[CopyRunDescriptor] = []
        num_bytes = 0
        for runs, data_ref in zip(
            self._copy_runs[g_idx], self.layer_refs_per_group[g_idx]
        ):
            if not runs:
                continue
            t_idx = data_ref.tensor_idx
            block_bases_src = self._scratch_bases_src[:group_size]
            block_bases_dst = self._scratch_bases_dst[:group_size]
            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                block_bases_src,
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                block_bases_dst,
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            mapping = data_ref.mapping
            if self.gpu_to_cpu and mapping is not None and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    group_dst,
                    group_size,
                    dst_skip_count,
                )
            num_active_blocks = len(block_bases_src)

            for run in runs:
                if self.gpu_to_cpu:
                    src_offset = run.local_offset
                    dst_offset = run.canonical_offset
                    src_stride = run.local_stride
                    dst_stride = run.canonical_stride
                else:
                    src_offset = run.canonical_offset
                    dst_offset = run.local_offset
                    src_stride = run.canonical_stride
                    dst_stride = run.local_stride
                run_descs.extend(
                    CopyRunDescriptor(
                        src_base=int(src_base) + src_offset,
                        dst_base=int(dst_base) + dst_offset,
                        fragment_size=run.fragment_size,
                        num_fragments=run.num_fragments,
                        src_stride=src_stride,
                        dst_stride=dst_stride,
                    )
                    for src_base, dst_base in zip(block_bases_src, block_bases_dst)
                )
                num_bytes += num_active_blocks * run.fragment_size * run.num_fragments
        return run_descs, num_bytes

    def _filter_writer_blocks(
        self,
        block_bases_src: np.ndarray,
        block_bases_dst: np.ndarray,
        mapping: CanonicalPageMapping,
        group_dst: np.ndarray,
        group_size: int,
        dst_skip_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the blocks this rank writes: replicated ranks take turns
        writing shared canonical pages, keyed by the rank-consistent CPU-side
        canonical page id."""
        cpu_page_ids = _canonical_page_ids(
            group_dst,
            self.dst_blocks_per_chunk,
            group_size,
            dst_skip_count,
        )
        writer_mask = cpu_page_ids % mapping.num_writers == mapping.writer_index
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU chunks, EXCEPT MAYBE for the first and last chunk.
        # i.e. the first and last CPU chunks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU chunk.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU chunk per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU chunk.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.layer_refs_per_group)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.layer_refs_per_group)

        src_offset = 0
        dst_offset = 0
        run_descs: list[CopyRunDescriptor] = []
        # count total number of bytes copied
        num_transfer_bytes = 0
        for g_idx, (group_size, block_idx) in enumerate(
            zip(group_sizes, block_indices)
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            group_run_descs, group_bytes = self._fill_run_ops(
                g_idx,
                group_src=src_blocks[src_offset:src_end_offset],
                group_dst=dst_blocks[dst_offset:dst_end_offset],
                group_size=group_size,
                src_skip_count=src_logical_blocks_to_skip,
                dst_skip_count=dst_logical_blocks_to_skip,
            )
            run_descs.extend(group_run_descs)
            num_transfer_bytes += group_bytes

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        # Stores must wait for the model to finish writing the KV they read.
        # Loads must wait for pending writes (including zeroing) to their
        # destination blocks: the scheduler leaves a block unzeroed only in
        # the step that allocates it for the load and cannot retract zeroing
        # shipped in an earlier step for a since-reallocated block; with async
        # scheduling nothing else orders that zeroing against this copy.
        stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        with current_platform.stream(stream):
            start_event.record(stream)
            self._backend.submit(
                job_id,
                run_descs,
                is_src_access_order_any=is_src_access_order_any,
            )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
            )
        )

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._backend.finish_transfer(transfer.job_id)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        """Drain this direction and release its transfer-side resources."""
        sync_error: Exception | None = None
        while self._transfers:
            transfer = self._transfers[0]
            try:
                transfer.end_event.synchronize()
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

        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._backend.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if sync_error is not None:
            raise sync_error


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)
        host_memory_is_pinned = pin_memory and (
            mmap_region is None or mmap_region.is_pinned
        )

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

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

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        try:
            self._load_handler.shutdown()
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
