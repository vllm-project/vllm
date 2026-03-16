# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import ctypes
import math
import mmap
import os
import time
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


def _page_align(size: int, page_size: int) -> int:
    return ((size + page_size - 1) // page_size) * page_size


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class _SharedMmapRegion:
    """
    Single mmap-backed memory region shared across all TP workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file and pins its own slice
    with cudaHostRegister for fast GPU<->CPU DMA.

    File path: /dev/shm/vllm_offload_{instance_id}.mmap
    Layout:
        [ worker-0 region | worker-1 region | ... | worker-(N-1) region ]
    """

    def __init__(
        self,
        instance_id: str,
        total_size_bytes: int,
        tp_world_size: int,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        self.total_size_bytes = _page_align(total_size_bytes, self.page_size)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_world_size = tp_world_size
        self.mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
        self._alloc_offset = 0  # bytes consumed so far within this worker's region
        self._creator = False  # set True only if this worker creates the file

        try:
            # Exclusive create — only one worker succeeds
            self.fd = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_TRUNC, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.info(
                "Worker tp_rank=%d created mmap file %s (%.2f GB)",
                self.tp_rank,
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info(
                "Worker tp_rank=%d opened existing mmap file %s",
                self.tp_rank,
                self.mmap_path,
            )

        self.mmap_obj = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.mmap_obj.madvise(mmap.MADV_WILLNEED)
        atexit.register(self.cleanup)

    def worker_region(self) -> tuple[int, int]:
        """Return (offset_bytes, size_bytes) for this worker's slice."""
        per_worker = _page_align(
            self.total_size_bytes // self.tp_world_size, self.page_size
        )
        return self.tp_rank * per_worker, per_worker

    def alloc_tensor(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Allocate the next tensor sequentially within this worker's region."""
        region_offset, region_size = self.worker_region()
        num_elements = math.prod(shape)
        tensor_bytes = num_elements * torch.tensor([], dtype=dtype).element_size()
        assert self._alloc_offset + tensor_bytes <= region_size, (
            f"mmap worker region exhausted: need {tensor_bytes} more bytes "
            f"but only {region_size - self._alloc_offset} remain"
        )
        start = region_offset + self._alloc_offset
        mv = memoryview(self.mmap_obj)[start : start + tensor_bytes]
        tensor = torch.frombuffer(mv, dtype=dtype, count=num_elements).reshape(shape)
        self._alloc_offset += tensor_bytes
        return tensor

    def pin_region(self) -> None:
        """Register this worker's mmap slice as CUDA pinned memory."""
        libcudart = find_library("cudart") or "libcudart.so"
        cuda = ctypes.CDLL(libcudart)
        cuda.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cuda.cudaHostRegister.restype = ctypes.c_int

        # Obtain base pointer of the mmap buffer
        mv = memoryview(self.mmap_obj)
        base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        offset, size = self.worker_region()
        ptr = base_ptr + offset

        result = cuda.cudaHostRegister(
            ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_uint(0)
        )
        if result != 0:
            logger.warning(
                "cudaHostRegister failed for tp_rank=%d (code=%d) — "
                "falling back to standard pin_memory allocation",
                self.tp_rank,
                result,
            )
        else:
            logger.info(
                "Pinned mmap region for tp_rank=%d: %.2f GB",
                self.tp_rank,
                size / 1e9,
            )

    def cleanup(self) -> None:
        if getattr(self, "mmap_obj", None) is not None:
            with contextlib.suppress(Exception):
                self.mmap_obj.close()
        if getattr(self, "fd", None) is not None:
            with contextlib.suppress(Exception):
                os.close(self.fd)
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                pass


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
        instance_id: str | None = None,
        tp_world_size: int | None = None,
        total_cpu_bytes: int | None = None,
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
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        # allocate cpu tensors
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))

        # Set up shared mmap region when all required params are provided
        self._mmap_region: _SharedMmapRegion | None = None
        if (
            instance_id is not None
            and tp_world_size is not None
            and total_cpu_bytes is not None
            and pin_memory
        ):
            self._mmap_region = _SharedMmapRegion(
                instance_id=instance_id,
                total_size_bytes=total_cpu_bytes,
                tp_world_size=tp_world_size,
            )
            self._mmap_region.pin_region()
        mmap_region = self._mmap_region

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

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )
