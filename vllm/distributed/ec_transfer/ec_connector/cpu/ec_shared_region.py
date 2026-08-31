# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight mmap-backed shared memory region for encoder cache (EC) data.

Modeled after SharedOffloadRegion (vllm/v1/kv_offload/cpu/) but simplified
for EC: flat shared layout, no multi-tensor cursor, no block_size_factor.
"""

import errno
import mmap
import os
from collections.abc import Callable

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.shm_utils import open_region_file, reap_orphaned_region_files

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written EC data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for EC mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class ECSharedRegion:
    """Flat mmap-backed memory region shared across TP workers for
    encoder cache blocks.

    Layout: (num_blocks, block_size_bytes) — contiguous, no per-worker
    interleaving. All workers map the same file and see identical data.

    File path: /dev/shm/vllm_ec_{engine_id}.mmap

    This class owns only the shared memory substrate (mmap lifecycle, the
    `blocks` view, CUDA host registration). Block allocation and eviction
    are tracked by `EmbeddingCache` in the scheduler process.
    """

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        block_size_bytes: int,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes

        total_size_bytes = num_blocks * block_size_bytes
        # Path in /dev/shm (tmpfs); unique per engine instance.
        self._mmap_path = f"/dev/shm/vllm_ec_{engine_id}.mmap"
        # True for the process that created the file (responsible for unlink).
        self._is_creator = False
        # True after successful cudaHostRegister (cleanup must unregister).
        self._is_pinned = False

        reap_orphaned_region_files("/dev/shm/vllm_ec_*.mmap")
        self._fd: int | None = None
        self._fd, self._is_creator = open_region_file(self._mmap_path, total_size_bytes)

        # MAP_SHARED mmap over _fd; all processes see the same pages.
        self._mmap_obj: mmap.mmap | None = mmap.mmap(
            self._fd,
            total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        if self._is_creator:
            populate_write_fn = _get_populate_write_fn(self._mmap_obj)
            populate_write_fn(self._mmap_obj, 0, total_size_bytes)

        # (num_blocks, block_size_bytes) int8 tensor over the mmap buffer.
        self.blocks: torch.Tensor = torch.frombuffer(
            memoryview(self._mmap_obj), dtype=torch.int8
        ).view(num_blocks, block_size_bytes)
        # Cached for cudaHostRegister/Unregister and pointer math.
        self._blocks_ptr: int = self.blocks.data_ptr()
        self._blocks_nbytes: int = self.blocks.nbytes

    def pin_memory(self) -> None:
        """Register the entire mmap as CUDA pinned memory for fast DMA.

        Each TP worker process owns its own virtual address for the same
        shared physical pages, so every process must register independently.
        No-op when CUDA is not available or already pinned.
        """
        if self._is_pinned or not torch.cuda.is_available():
            return
        result = torch.cuda.cudart().cudaHostRegister(
            self._blocks_ptr, self._blocks_nbytes, 0
        )
        if result.value != 0:
            logger.warning(
                "cudaHostRegister failed (code=%d) — "
                "transfers will still work but may be slower (unpinned DMA)",
                result.value,
            )
        else:
            logger.debug("cudaHostRegister %.2f MB", self._blocks_nbytes / 1e6)
            self._is_pinned = True

    def cleanup(self) -> None:
        """Tear down the region. Lifecycle method; no concurrent access."""
        logger.info("Starting ECSharedRegion cleanup...")
        if self._is_creator:
            try:
                os.unlink(self._mmap_path)
                logger.info("Removed EC mmap file %s", self._mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self._mmap_path, exc_info=True
                )
            self._is_creator = False
        if self._is_pinned:
            result = torch.cuda.cudart().cudaHostUnregister(self._blocks_ptr)
            if result.value != 0:
                logger.warning("cudaHostUnregister failed (code=%d)", result)
            self._is_pinned = False
        if hasattr(self, "blocks"):
            del self.blocks
        if self._mmap_obj:
            try:
                self._mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self._mmap_obj = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                logger.warning("Failed to close fd %s", self._fd, exc_info=True)
            self._fd = None
