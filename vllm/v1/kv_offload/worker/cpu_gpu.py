# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec, GPULoadStoreSpec
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


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
    block_offsets: np.ndarray | None = None,
    block_counts: np.ndarray | None = None,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor
    if block_offsets is not None or block_counts is not None:
        assert block_offsets is not None and block_counts is not None
        assert len(block_offsets) == len(block_ids)
        assert len(block_counts) == len(block_ids)

        output_idx = 0
        for block_id, block_offset, block_count in zip(
            block_ids, block_offsets, block_counts
        ):
            assert block_offset >= 0
            assert block_count >= 0
            assert block_offset + block_count <= block_size_factor
            base_block_id = block_id * block_size_factor
            output_end_idx = output_idx + block_count
            output[output_idx:output_end_idx] = base_block_id + np.arange(
                block_offset, block_offset + block_count
            )
            output_idx = output_end_idx
        assert output_idx == len(output)
        return

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


def build_transfer_indices(
    src_spec: BlockIDsLoadStoreSpec,
    dst_spec: BlockIDsLoadStoreSpec,
    src_block_size_factor: int,
    dst_block_size_factor: int,
) -> np.ndarray:
    src_blocks = src_spec.block_ids
    dst_blocks = dst_spec.block_ids
    assert src_blocks.ndim == 1
    assert dst_blocks.ndim == 1

    src_block_offsets = (
        src_spec.block_offsets if isinstance(src_spec, GPULoadStoreSpec) else None
    )
    src_block_counts = (
        src_spec.block_counts if isinstance(src_spec, GPULoadStoreSpec) else None
    )
    dst_block_offsets = (
        dst_spec.block_offsets if isinstance(dst_spec, GPULoadStoreSpec) else None
    )
    dst_block_counts = (
        dst_spec.block_counts if isinstance(dst_spec, GPULoadStoreSpec) else None
    )

    src_sub_block_count = (
        int(np.sum(src_block_counts))
        if src_block_counts is not None
        else src_blocks.size * src_block_size_factor
    )
    dst_sub_block_count = (
        int(np.sum(dst_block_counts))
        if dst_block_counts is not None
        else dst_blocks.size * dst_block_size_factor
    )

    src_sub_blocks_to_skip = 0
    if src_block_counts is None and dst_block_counts is None:
        src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor
        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip
    else:
        assert dst_sub_block_count == src_sub_block_count

    src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
    expand_block_ids(
        src_blocks,
        src_block_size_factor,
        src_to_dst[:, 0],
        skip_count=src_sub_blocks_to_skip,
        block_offsets=src_block_offsets,
        block_counts=src_block_counts,
    )
    expand_block_ids(
        dst_blocks,
        dst_block_size_factor,
        src_to_dst[:, 1],
        block_offsets=dst_block_offsets,
        block_counts=dst_block_counts,
    )
    return src_to_dst


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

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_to_dst = build_transfer_indices(
            src_spec,
            dst_spec,
            src_block_size_factor=self.src_block_size_factor,
            dst_block_size_factor=self.dst_block_size_factor,
        )
        dst_sub_block_count = src_to_dst.shape[0]
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

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
            for src_tensor, dst_tensor, block_size_in_bytes in zip(
                self.src_tensors,
                self.dst_tensors,
                self.tensor_block_size_in_bytes,
            ):
                ops.swap_blocks(
                    src_tensor,
                    dst_tensor,
                    block_size_in_bytes,
                    src_to_dst_tensor,
                )
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


class CpuGpuOffloadingHandlers:
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
    ):
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor
            cpu_tensor = torch.zeros(
                (num_cpu_blocks, cpu_page_size_bytes),
                dtype=torch.int8,
                device="cpu",
                pin_memory=pin_memory,
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
