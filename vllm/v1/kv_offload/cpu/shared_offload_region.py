# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import os
import time

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu.memory import (
    CPUOffloadMemoryBackend,
    CPUOffloadMemoryConfig,
    _wait_for_file_size,
    create_shared_memory_allocation,
)

logger = init_logger(__name__)

__all__ = ["SharedOffloadRegion", "_wait_for_file_size"]


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    Default file path: /dev/shm/vllm_offload_{instance_id}.mmap
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        instance_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
        memory_config: CPUOffloadMemoryConfig | None = None,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.memory_config = memory_config or CPUOffloadMemoryConfig()
        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride
        self.mapped_size_bytes = self.memory_config.mapped_size(self.total_size_bytes)

        self.mmap_path = self.memory_config.mmap_path(instance_id)
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        self.fd: int | None = None
        self.mmap_obj: mmap.mmap | None = None
        self._base: torch.Tensor | None = None
        self._views: list[torch.Tensor] = []
        self.is_pinned: bool = False
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        try:
            allocation = create_shared_memory_allocation(
                instance_id=instance_id,
                logical_size_bytes=self.total_size_bytes,
                memory_config=self.memory_config,
            )
            self.mmap_path = allocation.path
            self.fd = allocation.fd
            self._creator = allocation.creator

            self.mmap_obj = mmap.mmap(
                self.fd,
                self.mapped_size_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except Exception:
            self.cleanup()
            raise

        # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
        _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                if not self._madvise(
                    _MADV_POPULATE_WRITE, aligned_offset, aligned_length
                ):
                    break
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            self._madvise(_MADV_POPULATE_WRITE, 0, self.mapped_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(
            self.mmap_obj, dtype=torch.int8, count=self.total_size_bytes
        )

    def _madvise(self, advice: int, offset: int, length: int) -> bool:
        assert self.mmap_obj is not None
        try:
            self.mmap_obj.madvise(advice, offset, length)
            return True
        except (AttributeError, OSError, ValueError):
            if (
                self.memory_config.effective_backend
                == CPUOffloadMemoryBackend.HUGETLBFS
            ):
                logger.warning(
                    "MADV_POPULATE_WRITE failed for hugetlbfs mmap %s; "
                    "continuing without eager population",
                    self.mmap_path,
                    exc_info=True,
                )
                return False
            raise

    def create_next_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        assert self._base is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        assert self._base is not None
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike():
                base_ptr = self._base.data_ptr()
                result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        result,
                    )
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
