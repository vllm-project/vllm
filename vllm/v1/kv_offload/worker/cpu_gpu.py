# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCacheRef, CanonicalKVCaches
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

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
        output: pre-allocated int64 array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for block_size_factor > 1.
    assert tensor.shape[1] % block_size_factor == 0
    sub_block_size = tensor.shape[1] // block_size_factor
    sub_offsets = np.arange(block_size_factor, dtype=np.int64) * sub_block_size
    # (num_blocks, 1) + (1, block_size_factor) -> (num_blocks, block_size_factor)
    all_ptrs = (
        base_ptr + block_ids.astype(np.int64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
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


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """
    SingleDirectionOffloadingHandler handles transfers for a single direction,
    either CPU->GPU or GPU->CPU.
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

        # assert a single KV group until transfer_async supports multiple groups
        assert len(kv_cache_groups_data_refs) == 1

        # assert input tensors are as expected
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda
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

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * block_size_factor.
        self.src_block_size_factor = 1 if self.gpu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if self.gpu_to_cpu else 1

        # per-tensor block size in byte
        self.tensor_block_size_in_bytes = [
            gpu_tensor.shape[1] for gpu_tensor in gpu_tensors
        ]

        # per-group block size in bytes
        self.group_block_size_in_bytes = []
        for kv_cache_group_data_refs in kv_cache_groups_data_refs:
            group_block_size_in_bytes = 0
            for kv_cache_data_ref in kv_cache_group_data_refs:
                # TODO(orozery): use kv_cache_data_ref.page_size_bytes
                # once swap_blocks support it
                group_block_size_in_bytes += self.tensor_block_size_in_bytes[
                    kv_cache_data_ref.tensor_idx
                ]
            self.group_block_size_in_bytes.append(group_block_size_in_bytes)

        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []

        # Pre-compute block sizes for batch copies.
        self._block_size_in_bytes_arr = np.array(
            self.tensor_block_size_in_bytes, dtype=np.int64
        )

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        num_pairs = dst_sub_block_count
        num_tensors = len(self.src_tensors)
        total = num_pairs * num_tensors

        all_src = np.empty(total, dtype=np.int64)
        all_dst = np.empty(total, dtype=np.int64)
        all_sizes = np.empty(total, dtype=np.int64)

        for t_idx, bsz in enumerate(self._block_size_in_bytes_arr):
            start = t_idx * num_pairs
            end = start + num_pairs
            compute_sub_block_ptrs(
                block_ids=src_blocks,
                block_size_factor=self.src_block_size_factor,
                output=all_src[start:end],
                tensor=self.src_tensors[t_idx],
                skip_count=src_sub_blocks_to_skip,
            )
            compute_sub_block_ptrs(
                block_ids=dst_blocks,
                block_size_factor=self.dst_block_size_factor,
                output=all_dst[start:end],
                tensor=self.dst_tensors[t_idx],
            )
            all_sizes[start:end] = bsz

        batch_src = torch.from_numpy(all_src)
        batch_dst = torch.from_numpy(all_dst)
        batch_sizes = torch.from_numpy(all_sizes)

        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
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
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            if total > 0:
                ops.swap_blocks_batch(batch_src, batch_dst, batch_sizes)
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=dst_sub_block_count * self.group_block_size_in_bytes[0],
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
                transfer_type=self.transfer_type,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
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
        self.src_tensors.clear()
        self.dst_tensors.clear()


class CpuGpuOffloadingHandlers:
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self._mmap_region = mmap_region
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
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

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
        )
