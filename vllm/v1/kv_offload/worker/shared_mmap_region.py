# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import mmap
import os
import time

import torch

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
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{instance_id}.mmap
    """

    def __init__(
        self,
        instance_id: str,
        total_size_bytes: int,
        num_blocks: int,
        rank: int,
        num_workers: int,
        cpu_page_size: int,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        self.total_size_bytes = _page_align(total_size_bytes, self.page_size)
        self.mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.num_blocks = num_blocks
        # interleaved-layout stride: one row = all workers' data for one block
        self._row_stride = cpu_page_size * num_workers
        # byte offset to this worker's first slot within each block row
        self._worker_offset = rank * cpu_page_size
        try:
            # Exclusive create — only one worker succeeds
            self.fd = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_TRUNC, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing mmap file %s", self.mmap_path)

        self.mmap_obj = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        # int8 base — one element == one byte, so strides equal byte offsets
        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        atexit.register(self.cleanup)

    def alloc_tensor(self, tensor_size: int, block_size_factor: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_size: Bytes per block for this canonical tensor
                         (= CanonicalKVCacheTensor.page_size_bytes * block_size_factor).
        """
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks * block_size_factor, tensor_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset += tensor_size
        return worker_layer_view

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
