# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from collections import deque
from ctypes.util import find_library
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.shared_mmap_region import SharedMmapRegion
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

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


def pin_mmap_region(region: SharedMmapRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    tp_rank = get_tensor_model_parallel_rank()
    libcudart = find_library("cudart") or "libcudart.so"
    cuda = ctypes.CDLL(libcudart)
    cuda.cudaHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cuda.cudaHostRegister.restype = ctypes.c_int

    mv = memoryview(region.mmap_obj)
    base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mv))

    result = cuda.cudaHostRegister(
        ctypes.c_void_p(base_ptr),
        ctypes.c_size_t(region.total_size_bytes),
        ctypes.c_uint(0),
    )
    if result != 0:
        logger.warning(
            "cudaHostRegister failed for tp_rank=%d (code=%d) — "
            "falling back to standard pin_memory allocation",
            tp_rank,
            result,
        )
    else:
        logger.info(
            "Pinned mmap region for tp_rank=%d: %.2f GB",
            tp_rank,
            region.total_size_bytes / 1e9,
        )


class RemappedBlockHandler(OffloadingHandler):
    """
    Wraps a SingleDirectionOffloadingHandler and remaps logical CPU block IDs
    to physical block IDs for the interleaved mmap layout:

        physical = logical * num_workers + rank

    For GPU->CPU transfers the CPU side is dst; for CPU->GPU it is src.
    """

    def __init__(
        self,
        inner: "SingleDirectionOffloadingHandler",
        num_workers: int,
        rank: int,
        remap_src: bool,
    ) -> None:
        self._inner = inner
        self._num_workers = num_workers
        self._rank = rank
        self._remap_src = remap_src

    def _remap(self, spec: BlockIDsLoadStoreSpec) -> BlockIDsLoadStoreSpec:
        spec.block_ids = spec.block_ids * self._num_workers + self._rank
        return spec

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        if self._remap_src:
            src_spec = self._remap(src_spec)
        else:
            dst_spec = self._remap(dst_spec)
        return self._inner.transfer_async(job_id, (src_spec, dst_spec))

    def get_finished(self) -> list[TransferResult]:
        return self._inner.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        return self._inner.wait(job_ids)


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
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            src_tensors: list of KV cache tensors to copy from.
            dst_tensors: list of KV cache tensors to copy to.
                Order should match src_tensors.
            src_block_size_factor: The number of kernel blocks
                per KV block in a source tensor.
            dst_block_size_factor: The number of kernel blocks
                per KV block in a destination tensor.
        """
        assert len(src_tensors) == len(dst_tensors)

        self.src_tensors: list[torch.Tensor] = src_tensors
        self.dst_tensors: list[torch.Tensor] = dst_tensors
        min_block_size_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor: int = src_block_size_factor // min_block_size_factor
        self.dst_block_size_factor: int = dst_block_size_factor // min_block_size_factor

        self.block_size_in_bytes = [
            tensor.element_size() * tensor.stride(0) * min_block_size_factor
            for tensor in src_tensors
        ]
        self.total_block_size_in_bytes = sum(self.block_size_in_bytes)

        assert len(src_tensors) > 0
        self.gpu_to_cpu: bool = self.src_tensors[0].is_cuda
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

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
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
                self.block_size_in_bytes,
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
                num_bytes=dst_sub_block_count * self.total_block_size_in_bytes,
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
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        mmap_region: SharedMmapRegion | None = None,
        num_workers: int = 1,
    ):
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0

        # find kernel block size and determine layout per each gpu tensor
        kernel_block_size: int | None = None
        # list of (gpu_tensor, split_k_and_v)
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            if has_layers_dim:
                # in the cross layers case, the registered kv cache tensor
                # shape matches the physical layout, whereas test_shape
                # is the logical layout.
                # To match them, we need to permute test_shape
                try:
                    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=has_layers_dim
                    )
                    assert len(kv_cache_stride_order) == len(gpu_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(gpu_shape)))

                test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        # Total kernel blocks in the CPU tensor covers all workers' data
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor * num_workers

        # allocate cpu tensors
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))

        self._mmap_region = mmap_region
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            if mmap_region is not None:
                cpu_tensor = mmap_region.alloc_tensor(
                    shape=tuple(cpu_shape), dtype=gpu_tensor.dtype
                )
            else:
                cpu_tensor = torch.zeros(
                    cpu_shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )

            gpu_tensors.extend(gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor])
            cpu_tensors.extend(cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor])

        inner_gpu_to_cpu = SingleDirectionOffloadingHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )
        inner_cpu_to_gpu = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )

        if num_workers > 1:
            rank = get_tensor_model_parallel_rank()
            self.gpu_to_cpu_handler: OffloadingHandler = RemappedBlockHandler(
                inner=inner_gpu_to_cpu,
                num_workers=num_workers,
                rank=rank,
                remap_src=False,  # CPU is dst
            )
            self.cpu_to_gpu_handler: OffloadingHandler = RemappedBlockHandler(
                inner=inner_cpu_to_gpu,
                num_workers=num_workers,
                rank=rank,
                remap_src=True,  # CPU is src
            )
        else:
            self.gpu_to_cpu_handler = inner_gpu_to_cpu
            self.cpu_to_gpu_handler = inner_cpu_to_gpu
