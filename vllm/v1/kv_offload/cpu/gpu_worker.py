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
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.config import CompactGroupSliceConfig
from vllm.v1.kv_offload.cpu.common import CompactCPULoadStoreSpec
from vllm.v1.kv_offload.cpu.compact_transfer import (
    CompactTransferPlan,
    plan_compact_transfer,
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


def _fill_compact_descriptor_buffers(
    gpu_to_cpu: bool,
    plan: CompactTransferPlan,
    batch_src: torch.Tensor,
    batch_dst: torch.Tensor,
    batch_sizes: torch.Tensor,
    num_copy_ops: int,
) -> None:
    """Fill PyTorch descriptor buffers from a compact transfer plan.

    Pure NumPy helper (no CUDA) that swaps GPU/CPU pointer roles for
    store vs. load.  ``batch_src``, ``batch_dst``, ``batch_sizes`` are
    pre-allocated pinned PyTorch int64/uint64 tensors; only the first
    ``num_copy_ops`` entries are filled.

    Extracted from ``_transfer_compact`` so that bounded tests can capture
    exact descriptors without CUDA, CUPTI, stream synchronization, or real
    GPU/CPU tensor allocations.
    """
    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=gpu_to_cpu,
        plan=plan,
        src_arr=batch_src.numpy()[:num_copy_ops],
        dst_arr=batch_dst.numpy()[:num_copy_ops],
        sizes_arr=batch_sizes.numpy()[:num_copy_ops],
    )


def _fill_compact_descriptor_buffers_numpy(
    gpu_to_cpu: bool,
    plan: CompactTransferPlan,
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    sizes_arr: np.ndarray,
) -> None:
    """Fill numpy descriptor arrays from a compact transfer plan.

    Pure NumPy (no torch, no CUDA) -- usable in bounded tests that prove
    exact descriptor construction without importing the full vllm module
    chain.
    """
    if gpu_to_cpu:
        src_arr[:] = np.asarray(plan.gpu_ptrs)
        dst_arr[:] = np.asarray(plan.cpu_ptrs)
    else:
        src_arr[:] = np.asarray(plan.cpu_ptrs)
        dst_arr[:] = np.asarray(plan.gpu_ptrs)
    sizes_arr[:] = np.asarray(plan.sizes)


class SingleDirectionOffloadingHandler:
    """
    Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
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
        mmap_region: SharedOffloadRegion | None = None,
        compact_region: torch.Tensor | None = None,
        compact_group_slice_configs: tuple[CompactGroupSliceConfig, ...] | None = None,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            blocks_per_chunk: number of GPU blocks per CPU block.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
            compact_region: contiguous pinned CPU region for compact layout.
            compact_group_slice_configs: slice accounting for compact transfer
                planning, matching the transported compact_slice_accounting.
        """
        compact_mode = compact_region is not None
        if compact_mode != (compact_group_slice_configs is not None):
            raise ValueError(
                "compact_region and compact_group_slice_configs "
                "must be provided together"
            )
        if compact_mode and mmap_region is not None:
            raise ValueError("compact CPU layout does not support mmap storage")
        if not compact_mode and len(gpu_tensors) != len(cpu_tensors):
            raise ValueError("legacy GPU and CPU tensor counts must match")
        if not gpu_tensors:
            raise ValueError("at least one GPU tensor is required")

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
        self._compact_region = compact_region
        self._compact_group_slice_configs = compact_group_slice_configs
        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

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
        num_scratch_blocks = gpu_tensors[0].shape[0] if canonical_layout else 0
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
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the direct (worker-private)
        layout: one whole-page copy per (block, ref).

        Returns (op_idx past the filled descriptors, bytes added)."""
        num_bytes = 0
        for data_ref in self.layer_refs_per_group[g_idx]:
            t_idx = data_ref.tensor_idx
            end_idx = op_idx + group_size

            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                all_src[op_idx:end_idx],
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                all_dst[op_idx:end_idx],
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _fill_canonical_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
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
        if group_size > len(self._scratch_bases_src):
            self._scratch_bases_src = np.empty(group_size, dtype=np.uint64)
            self._scratch_bases_dst = np.empty(group_size, dtype=np.uint64)

        num_bytes = 0
        for plan, data_ref in zip(
            self._canonical_copy_plans[g_idx], self.layer_refs_per_group[g_idx]
        ):
            if plan.num_frags == 0:
                continue
            t_idx = data_ref.tensor_idx

            # 1. Base byte pointer of every block on each side
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

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.gpu_to_cpu and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    group_dst,
                    group_size,
                    dst_skip_count,
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

    def _submit_descriptors(
        self,
        *,
        job_id: int,
        batch_src: torch.Tensor,
        batch_dst: torch.Tensor,
        batch_sizes: torch.Tensor,
        num_copy_ops: int,
        num_transfer_bytes: int,
        use_batch_api: bool = True,
    ) -> bool:
        """Submit descriptors through the one canonical async lifecycle path.

        Extracted for reuse by both the legacy block-id path and the compact
        packed-slice path so that stream/wait/event plumbing stays in one place.
        """
        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
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
        # (upstream dropped the gpu_to_cpu guard here; this helper is the
        # extracted form of that block and needs the same fix.)
        stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)
        is_src_access_order_any = not self.gpu_to_cpu
        with current_platform.stream(stream):
            start_event.record(stream)
            if num_copy_ops > 0:
                self._swap_blocks_batch(
                    src,
                    dst,
                    sizes,
                    is_src_access_order_any=is_src_access_order_any,
                    use_batch_api=use_batch_api,
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
        return True

    def _transfer_compact(
        self,
        job_id: int,
        cpu_spec: CompactCPULoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
    ) -> bool:
        """Transfer compact packed-slice data through plan_compact_transfer.

        Uses the direction-neutral planner from commit 3: for store (GPU->CPU)
        the GPU block IDs are src and compact addresses are dst; for load
        (CPU->GPU) the roles are reversed.
        """
        region = self._compact_region
        slice_configs = self._compact_group_slice_configs
        assert region is not None
        assert slice_configs is not None

        # Select the one canonical packed GPU tensor and derive row geometry.
        gpu_tensor = self.src_tensors[0] if self.gpu_to_cpu else self.dst_tensors[0]
        gpu_base_ptr = gpu_tensor.data_ptr()
        gpu_row_stride = gpu_tensor.stride(0)
        cpu_base_ptr = region.data_ptr()
        cpu_region_size = region.numel() * region.element_size()

        plan = plan_compact_transfer(
            gpu_base_ptr=gpu_base_ptr,
            gpu_row_stride=gpu_row_stride,
            cpu_base_ptr=cpu_base_ptr,
            cpu_region_size=cpu_region_size,
            gpu_block_ids=gpu_spec.block_ids,
            group_sizes=gpu_spec.group_sizes,
            block_indices=gpu_spec.block_indices,
            compact_addresses=cpu_spec.compact_addresses,
            group_slice_configs=slice_configs,
            block_size_factor=(
                self.dst_blocks_per_chunk
                if self.gpu_to_cpu
                else self.src_blocks_per_chunk
            ),
        )

        num_copy_ops = len(plan.sizes)
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        # Fill descriptor buffers using the extracted pure helper.
        _fill_compact_descriptor_buffers(
            gpu_to_cpu=self.gpu_to_cpu,
            plan=plan,
            batch_src=batch_src,
            batch_dst=batch_dst,
            batch_sizes=batch_sizes,
            num_copy_ops=num_copy_ops,
        )

        return self._submit_descriptors(
            job_id=job_id,
            batch_src=batch_src,
            batch_dst=batch_dst,
            batch_sizes=batch_sizes,
            num_copy_ops=num_copy_ops,
            num_transfer_bytes=plan.num_bytes,
            use_batch_api=False,
        )

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        # Compact path: detect CompactCPULoadStoreSpec on the CPU side.
        if self.gpu_to_cpu and isinstance(dst_spec, CompactCPULoadStoreSpec):
            assert isinstance(src_spec, GPULoadStoreSpec)
            return self._transfer_compact(job_id, dst_spec, src_spec)
        if not self.gpu_to_cpu and isinstance(src_spec, CompactCPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            return self._transfer_compact(job_id, src_spec, dst_spec)

        # Legacy block-id path.
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
        # transfers are also to CPU blocks, EXCEPT MAYBE for the first and last block.
        # i.e. the first and last CPU blocks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU block.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU block per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU block.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.layer_refs_per_group)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
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

        src_offset = 0
        dst_offset = 0
        op_idx = 0
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

            op_idx, group_bytes = self._fill_group_ops(
                g_idx,
                group_src=src_blocks[src_offset:src_end_offset],
                group_dst=dst_blocks[dst_offset:dst_end_offset],
                group_size=group_size,
                src_skip_count=src_logical_blocks_to_skip,
                dst_skip_count=dst_logical_blocks_to_skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        # Writer rotation may skip non-writer blocks, so op_idx is the number
        # of descriptors actually written; num_copy_ops is only the sized upper
        # bound. Submitting the bound would hand the DMA the untouched tail of
        # buffers that come from torch.empty and are recycled through
        # _buffer_pool -- i.e. uninitialized or stale device pointers.
        assert op_idx <= num_copy_ops

        return self._submit_descriptors(
            job_id=job_id,
            batch_src=batch_src,
            batch_dst=batch_dst,
            batch_sizes=batch_sizes,
            num_copy_ops=op_idx,
            num_transfer_bytes=num_transfer_bytes,
        )

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
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if sync_error is not None:
            raise sync_error
        self._compact_region = None
        self._compact_group_slice_configs = None


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.

    When *compact_slice_accounting* is provided, the worker allocates a
    single contiguous pinned CPU region instead of per-tensor block rows.
    Both handlers share the same compact region and slice configs for
    packed-slice transfer planning.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
        compact_slice_accounting: tuple[CompactGroupSliceConfig, ...] | None = None,
        compact_cpu_budget_bytes_per_rank: int | None = None,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        compact_mode = compact_slice_accounting is not None
        if compact_mode != (compact_cpu_budget_bytes_per_rank is not None):
            raise ValueError(
                "compact slice accounting and per-rank budget must be provided together"
            )
        if compact_mode and mmap_region is not None:
            raise ValueError("compact CPU layout does not support mmap storage")
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )
        compact_region: torch.Tensor | None = None
        if compact_mode:
            assert compact_cpu_budget_bytes_per_rank is not None
            if compact_cpu_budget_bytes_per_rank <= 0:
                raise ValueError("compact per-rank CPU budget must be positive")
            compact_region = torch.empty(
                compact_cpu_budget_bytes_per_rank,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pin_memory,
            )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            # Compact and canonical are mutually exclusive CPU page layouts;
            # compact short-circuits because it shares one contiguous region
            # across both handlers rather than carving per-tensor views.
            assert not (compact_mode and canonical_bytes_per_block is not None), (
                "compact and canonical CPU layouts cannot be combined"
            )
            if compact_mode:
                # Compact mode uses one contiguous CPU region shared by both
                # handlers. The gpu_tensor is still needed for row geometry.
                gpu_tensors.append(gpu_tensor)
                continue
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
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
            mmap_region=mmap_region,
            compact_region=compact_region,
            compact_group_slice_configs=compact_slice_accounting,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
            compact_region=compact_region,
            compact_group_slice_configs=compact_slice_accounting,
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
