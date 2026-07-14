# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


@dataclass
class NPUTransfer:
    job_id: int
    start_event: Any
    end_event: Any
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


def _get_torch_npu() -> Any:
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "Ascend KV offloading requires torch_npu to expose torch.npu."
        )
    return npu


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=PIN_MEMORY),
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=PIN_MEMORY),
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=PIN_MEMORY),
    )


class AscendSingleDirectionOffloadingHandler:
    """Handle one NPU/CPU transfer direction with Ascend batch copy."""

    def __init__(
        self,
        npu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        npu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
    ) -> None:
        assert len(npu_tensors) == len(cpu_tensors)
        assert npu_tensors

        for npu_tensor, cpu_tensor in zip(npu_tensors, cpu_tensors):
            assert npu_tensor.dtype == torch.int8
            assert npu_tensor.ndim == 2
            assert npu_tensor.device.type == "npu"
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            assert cpu_tensor.shape[1] == npu_tensor.shape[1] * block_size_factor

        self.src_tensors = npu_tensors if npu_to_cpu else cpu_tensors
        self.dst_tensors = cpu_tensors if npu_to_cpu else npu_tensors
        self.npu_to_cpu = npu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self.src_block_size_factor = 1 if npu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if npu_to_cpu else 1
        self._mmap_region = mmap_region
        self._transfers: deque[NPUTransfer] = deque()
        self._event_pool: list[Any] = []
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._stream = _get_torch_npu().Stream()

    def _new_event(self) -> Any:
        if self._event_pool:
            return self._event_pool.pop()
        return _get_torch_npu().Event(enable_timing=True)

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        npu_spec = src_spec if self.npu_to_cpu else dst_spec
        assert isinstance(npu_spec, GPULoadStoreSpec)
        group_sizes = npu_spec.group_sizes
        block_indices = npu_spec.block_indices
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = sum(
            group_size * len(group_data_refs)
            for group_size, group_data_refs in zip(
                group_sizes, self.kv_cache_groups_data_refs
            )
        )
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
        num_transfer_bytes = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue

            src_skip = block_idx % self.src_block_size_factor
            dst_skip = block_idx % self.dst_block_size_factor
            src_count = (
                group_size + src_skip + self.src_block_size_factor - 1
            ) // self.src_block_size_factor
            dst_count = (
                group_size + dst_skip + self.dst_block_size_factor - 1
            ) // self.dst_block_size_factor
            src_end = src_offset + src_count
            dst_end = dst_offset + dst_count
            assert src_end <= len(src_blocks)
            assert dst_end <= len(dst_blocks)

            group_src = src_blocks[src_offset:src_end]
            group_dst = dst_blocks[dst_offset:dst_end]
            for data_ref in group_data_refs:
                end_idx = op_idx + group_size
                compute_sub_block_ptrs(
                    group_src,
                    self.src_block_size_factor,
                    all_src[op_idx:end_idx],
                    self.src_tensors[data_ref.tensor_idx],
                    skip_count=src_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_block_size_factor,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[data_ref.tensor_idx],
                    skip_count=dst_skip,
                )
                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end
            dst_offset = dst_end

        assert src_offset == len(src_blocks)
        assert dst_offset == len(dst_blocks)
        assert op_idx == num_copy_ops

        npu = _get_torch_npu()
        start_event = self._new_event()
        end_event = self._new_event()
        if self.npu_to_cpu:
            self._stream.wait_stream(npu.current_stream())
        if self._transfers:
            self._stream.wait_event(self._transfers[-1].end_event)

        with npu.stream(self._stream):
            start_event.record(self._stream)
            if num_copy_ops:
                direction = 1 if self.npu_to_cpu else 0
                torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, direction)
            end_event.record(self._stream)

        self._transfers.append(
            NPUTransfer(
                job_id=job_id,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
            )
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_time=(
                        transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
                    ),
                )
            )
            self._event_pool.extend((transfer.end_event, transfer.start_event))
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        for transfer in self._transfers:
            if transfer.job_id in job_ids:
                transfer.end_event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            self._transfers.popleft().end_event.synchronize()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class AscendCPUOffloadingWorker(OffloadingWorker):
    """Offloading worker for Ascend NPU <-> CPU KV transfers."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ) -> None:
        _get_torch_npu()
        logger.info(
            "Allocating %d CPU tensors for Ascend KV offload", len(kv_caches.tensors)
        )

        npu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            npu_page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, npu_page_size_bytes)
            )
            cpu_page_size_bytes = npu_page_size_bytes * block_size_factor
            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
            npu_tensors.append(npu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = AscendSingleDirectionOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=True,
            mmap_region=mmap_region,
        )
        self._load_handler = AscendSingleDirectionOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=False,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()
