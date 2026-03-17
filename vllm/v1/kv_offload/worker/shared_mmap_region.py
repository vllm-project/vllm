# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import math
import mmap
import os
import time

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


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


class SharedMmapRegion:
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
            flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
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
