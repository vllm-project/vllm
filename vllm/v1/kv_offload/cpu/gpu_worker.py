# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
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
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)


def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm host mappings) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu() or current_platform.is_rocm():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return ops.swap_blocks_batch
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

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


def _build_copy_plan(ref: CanonicalKVCacheRef, gpu_to_cpu: bool) -> CopyPlan:
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
    src, dst = (local, canonical) if gpu_to_cpu else (canonical, local)
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


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    rank = region.rank

    base_ptr = region._base.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """
    Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        canonical_layout: bool = False,
    ):
        """Initialize a SingleDirectionOffloadingHandler.

        Args:
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
        """
        assert len(cpu_tensors) > 0

        for cpu_tensor in cpu_tensors:
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"

        self.cpu_tensors = cpu_tensors
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.layer_refs_per_group = layer_refs_per_group
        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu
        )
        self.blocks_per_chunk = blocks_per_chunk

        # Per (group, ref) static copy plans for the canonical layout
        self._canonical_copy_plans: list[list[CopyPlan]] | None = (
            [
                [_build_copy_plan(ref, gpu_to_cpu) for ref in layer_refs]
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
        num_scratch_blocks = cpu_tensors[0].shape[0] if canonical_layout else 0
        self._scratch_bases_cpu = np.empty(num_scratch_blocks, dtype=np.uint64)

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

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
            gpu_ptrs = device_ptrs.ptrs[ptr_start : ptr_start + group_size]
            if self.gpu_to_cpu:
                all_src[op_idx:end_idx] = gpu_ptrs
                compute_sub_block_ptrs(
                    cpu_block_ids,
                    self.blocks_per_chunk,
                    all_dst[op_idx:end_idx],
                    self.cpu_tensors[t_idx],
                    skip_count=cpu_skip_count,
                )
            else:
                all_dst[op_idx:end_idx] = gpu_ptrs
                compute_sub_block_ptrs(
                    cpu_block_ids,
                    self.blocks_per_chunk,
                    all_src[op_idx:end_idx],
                    self.cpu_tensors[t_idx],
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
        # Zero-copy reinterpretation for pointer arithmetic: uint64 and the
        # buffers' int64 are bit-equivalent for addresses
        all_src_u64 = all_src.view(np.uint64)
        all_dst_u64 = all_dst.view(np.uint64)
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
            gpu_bases = device_ptrs.ptrs[ptr_start : ptr_start + group_size]
            cpu_bases = self._scratch_bases_cpu[:group_size]
            compute_sub_block_ptrs(
                cpu_block_ids,
                self.blocks_per_chunk,
                cpu_bases,
                self.cpu_tensors[t_idx],
                skip_count=cpu_skip_count,
            )

            if self.gpu_to_cpu:
                block_bases_src = gpu_bases
                block_bases_dst = cpu_bases
            else:
                block_bases_src = cpu_bases
                block_bases_dst = gpu_bases

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.gpu_to_cpu and mapping.num_writers > 1:
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
                out=all_src_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            np.add(
                block_bases_dst[:, None],
                plan.frag_offsets_dst[None, :],
                out=all_dst_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
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
        writer_mask = cpu_page_ids % mapping.num_writers == mapping.writer_index
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def transfer_async(
        self,
        job_id: int,
        device_ptrs: DevicePointers,
        cpu_spec: BlockIDsLoadStoreSpec,
    ) -> bool:
        cpu_blocks = cpu_spec.block_ids
        assert cpu_blocks.ndim == 1
        num_cpu_blocks = len(cpu_blocks)

        group_sizes = device_ptrs.group_block_counts
        block_indices = device_ptrs.block_indices
        assert len(group_sizes) == len(self.layer_refs_per_group)
        assert len(block_indices) == len(self.layer_refs_per_group)

        num_copy_ops = self._estimate_max_copy_ops(group_sizes)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        cpu_offset = 0
        dev_ptr_offset = 0
        op_idx = 0
        num_transfer_bytes = 0
        for g_idx, (group_size, block_idx) in enumerate(
            zip(group_sizes, block_indices)
        ):
            n_data_refs = len(self.layer_refs_per_group[g_idx])
            if group_size == 0:
                continue

            cpu_skip = block_idx % self.blocks_per_chunk
            cpu_logical_count = group_size + cpu_skip
            cpu_blocks_count = cdiv(cpu_logical_count, self.blocks_per_chunk)
            cpu_end_offset = cpu_offset + cpu_blocks_count
            assert cpu_end_offset <= num_cpu_blocks

            op_idx, group_bytes = self._fill_group_ops(
                g_idx,
                device_ptrs=device_ptrs,
                dev_ptr_offset=dev_ptr_offset,
                cpu_block_ids=cpu_blocks[cpu_offset:cpu_end_offset],
                group_size=group_size,
                cpu_skip_count=cpu_skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes
            cpu_offset = cpu_end_offset
            dev_ptr_offset += group_size * n_data_refs

        assert cpu_offset == num_cpu_blocks
        # Writer rotation may skip non-writer blocks, leaving op_idx below
        # the sized upper bound
        assert op_idx <= num_copy_ops
        src = src[:op_idx]
        dst = dst[:op_idx]
        sizes = sizes[:op_idx]

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
        # destination blocks; otherwise an earlier transfer can be overwritten
        # by compute-stream work that was already queued when the load began.
        stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host pinned memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        with current_platform.stream(stream):
            start_event.record(stream)
            if op_idx > 0:
                self._swap_blocks_batch(
                    src,
                    dst,
                    sizes,
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
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
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
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
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
        self._buffer_pool.clear()
        self.cpu_tensors.clear()
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
        num_cpu_blocks: int,
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
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
        )

    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        return self._store_handler.transfer_async(job_id, device_ptrs, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, device_ptrs: DevicePointers
    ) -> bool:
        """Async CPU -> GPU."""
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        return self._load_handler.transfer_async(job_id, device_ptrs, src_spec)

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
