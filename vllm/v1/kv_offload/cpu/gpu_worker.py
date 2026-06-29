# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GroupTransfer,
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
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm builds without Triton) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in kv_cache_groups_data_refs for r in g]
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
    block_size_factor: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains block_size_factor sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * sub_block_size

    where sub_block_size = tensor.shape[1] // block_size_factor (gpu page size).

    This handles tensors where row_stride != block_size_factor * sub_block_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        block_size_factor: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for block_size_factor > 1.
    assert tensor.shape[1] % block_size_factor == 0
    sub_block_size = tensor.shape[1] // block_size_factor
    sub_offsets = np.arange(block_size_factor, dtype=np.uint64) * sub_block_size
    # (num_blocks, 1) + (1, block_size_factor) -> (num_blocks, block_size_factor)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


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
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
        pin_thread: threading.Thread | None = None,
        manually_pinned_tensors: list[torch.Tensor] | None = None,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            kv_cache_groups_data_refs: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        # assert input tensors are as expected
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            assert cpu_page_size == gpu_page_size * block_size_factor

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self._swap_blocks_batch = _select_swap_blocks_fn(
            kv_cache_groups_data_refs, gpu_to_cpu
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * block_size_factor.
        self.src_block_size_factor = 1 if self.gpu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if self.gpu_to_cpu else 1

        # mmap_region to clean up on shutdown (gpu_to_cpu handler owns it)
        self._mmap_region = mmap_region
        self._pin_thread = pin_thread
        self._manually_pinned_tensors = manually_pinned_tensors
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

    def transfer_async(self, job_id: int, groups: Sequence[GroupTransfer]) -> bool:
        # There are 2 types of transfers: GPU->CPU (store) and CPU->GPU (load).
        # The offloaded (CPU) side may have larger blocks than the GPU side
        # (block_size_factor > 1). The first offloaded block of each group
        # may be unaligned: offload_spec.gpu_block_offset % block_size_factor
        # sub-blocks at the start are skipped.

        # Pre-compute total copy operations for buffer sizing.
        num_copy_ops = 0
        for group, group_data_refs in zip(groups, self.kv_cache_groups_data_refs):
            num_copy_ops += len(group.gpu_spec.block_ids) * len(group_data_refs)

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

        op_idx = 0
        num_transfer_bytes = 0
        # GPU side always has block_size_factor=1; the offloaded side has
        # dst_block_size_factor (store) or src_block_size_factor (load).
        offload_bsf = (
            self.dst_block_size_factor
            if self.gpu_to_cpu
            else self.src_block_size_factor
        )

        for group, group_data_refs in zip(groups, self.kv_cache_groups_data_refs):
            gpu_spec = group.gpu_spec
            offload_spec = group.offload_spec
            group_size = len(gpu_spec.block_ids)
            if group_size == 0:
                continue

            offload_skip = offload_spec.gpu_block_offset % offload_bsf

            assert isinstance(offload_spec, BlockIDsLoadStoreSpec)
            src_bsf = self.src_block_size_factor
            dst_bsf = self.dst_block_size_factor
            if self.gpu_to_cpu:
                src_blocks = gpu_spec.block_ids
                dst_blocks = offload_spec.block_ids
                src_skip, dst_skip = 0, offload_skip
            else:
                src_blocks = offload_spec.block_ids
                dst_blocks = gpu_spec.block_ids
                src_skip, dst_skip = offload_skip, 0

            for data_ref in group_data_refs:
                t_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size

                compute_sub_block_ptrs(
                    src_blocks,
                    src_bsf,
                    all_src[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    skip_count=src_skip,
                )
                compute_sub_block_ptrs(
                    dst_blocks,
                    dst_bsf,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[t_idx],
                    skip_count=dst_skip,
                )

                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

        assert op_idx == num_copy_ops

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

        if self.gpu_to_cpu:
            # wait for model computation to finish before offloading
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
            if num_copy_ops > 0:
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
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()

        if self._pin_thread is not None:
            self._pin_thread.join()
            self._pin_thread = None

        if self._manually_pinned_tensors is not None:
            for tensor in self._manually_pinned_tensors:
                result = torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for CPU tensor (code=%d)",
                        result.value,
                    )

        self.src_tensors.clear()
        self.dst_tensors.clear()

        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        pin_memory = PIN_MEMORY
        self.pin_thread: threading.Thread | None = None
        self._manually_pinned_tensors: list[torch.Tensor] = []

        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self._mmap_region = mmap_region

        gpu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor

            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    # CUDA/ROCm memory is registered asynchronously below.
                    # Pinning here would block worker initialization; other
                    # hardware need PyTorch allocation-time pinning.
                    pin_memory=PIN_MEMORY and not current_platform.is_cuda_alike(),
                )
                logger.debug(
                    "torch.zeros tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            self.cpu_tensors.append(cpu_tensor)

        if pin_memory:
            if not current_platform.is_cuda_alike():
                logger.info(
                    "Skipping host registration on %s; cudaHostRegister is only "
                    "available on CUDA/ROCm.",
                    current_platform.device_name,
                )
            else:
                self.pin_thread = threading.Thread(
                    target=self._pin_cpu_tensors,
                    name="CPUTensorPinThread",
                )
                self.pin_thread.start()
                logger.info("Starting to pin memory in background...")

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=self.cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            mmap_region=mmap_region,
            pin_thread=self.pin_thread,
            manually_pinned_tensors=self._manually_pinned_tensors,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=self.cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
        )

    def _pin_cpu_tensors(self) -> None:
        """Register the CPU offload memory as CUDA pinned memory."""

        t0 = time.monotonic()
        tensors_to_pin = (
            [self._mmap_region._base]
            if self._mmap_region is not None
            else self.cpu_tensors
        )
        num_pinned = 0
        for tensor in tensors_to_pin:
            total_size_bytes = tensor.numel() * tensor.element_size()
            result = torch.cuda.cudart().cudaHostRegister(
                tensor.data_ptr(), total_size_bytes, 0
            )
            if result.value != 0:
                logger.warning(
                    "cudaHostRegister failed for host tensor (code=%d) "
                    "- transfers will still work but may be slower (unpinned DMA)",
                    result.value,
                )
                continue
            if self._mmap_region is not None:
                self._mmap_region.is_pinned = True
            else:
                self._manually_pinned_tensors.append(tensor)
            num_pinned += 1

            logger.debug(
                "cudaHostRegister pin %.2f GB",
                total_size_bytes / 1e9,
            )

        logger.info(
            "Completed CPU memory pinning: %d tensors pinned in %.3f s",
            num_pinned,
            time.monotonic() - t0,
        )

    def submit_store(self, job_id: int, groups: Sequence[GroupTransfer]) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, groups)

    def submit_load(self, job_id: int, groups: Sequence[GroupTransfer]) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, groups)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()
