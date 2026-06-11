# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight mmap-backed shared memory region for encoder cache (EC) data.

Modeled after SharedOffloadRegion (vllm/v1/kv_offload/cpu/) but simplified
for EC: flat shared layout, no multi-tensor cursor, no block_size_factor.
"""

import mmap
import os
import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class AllocationError(RuntimeError):
    """Raised when the shared region cannot satisfy an allocation."""


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0):
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for EC mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class ECSharedRegion:
    """Flat mmap-backed memory region shared across TP workers for
    encoder cache blocks.

    Layout: (num_blocks, block_size_bytes) — contiguous, no per-worker
    interleaving. All workers map the same file and see identical data.

    File path: /dev/shm/vllm_ec_{instance_id}.mmap

    Thread-safety
    -------------
    Producer-side, the scheduler's main thread allocates / frees blocks
    while a ZMQ listener thread pins / unpins them around in-flight
    NIXL WRITEs. A single internal `threading.Lock` guards both the
    free pool and the ref counts so that the two state machines stay
    consistent.
    """

    def __init__(
        self,
        instance_id: str,
        num_blocks: int,
        block_size_bytes: int,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes

        self.total_size_bytes = num_blocks * block_size_bytes
        self.mmap_path = f"/dev/shm/vllm_ec_{instance_id}.mmap"
        self._is_creator = False

        try:
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._is_creator = True
            logger.info(
                "Created EC mmap file %s (%.2f MB)",
                self.mmap_path,
                self.total_size_bytes / 1e6,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            try:
                _wait_for_file_size(self.fd, self.total_size_bytes)
            except Exception:
                os.close(self.fd)
                self.fd = None
                raise
            logger.info("Opened existing EC mmap file %s", self.mmap_path)

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        if self._is_creator:
            _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)

        self._base: torch.Tensor | None = torch.frombuffer(
            memoryview(self.mmap_obj), dtype=torch.int8
        )
        self.is_pinned: bool = False

        self.blocks: torch.Tensor = self._base.view(num_blocks, block_size_bytes)

        self._lock = threading.Lock()
        # `_free` is the free pool used as a LIFO.
        # eviction order lives in the scheduler.
        self._free: list[int] = list(range(num_blocks))
        # `_ref_count[idx]` is the number of outstanding references
        # holding the block in place (incremented by `pin`, decremented
        # by `unpin`). Missing keys mean zero. A non-zero ref count
        # blocks `try_free` from reclaiming.
        self._ref_count: dict[int, int] = {}

    @property
    def base_ptr(self) -> int:
        """Raw address of the mmap region; used by the NIXL registration
        path on the scheduler."""
        assert self._base is not None
        return self._base.data_ptr()

    def alloc(self, n_blocks: int) -> list[int]:
        """Reserve `n_blocks` free block indices.

        Raises `AllocationError` if fewer than `n_blocks` are free. The
        caller is responsible for making room (e.g. evicting reclaimable
        entries via `try_free`) and retrying.
        """
        with self._lock:
            if len(self._free) < n_blocks:
                raise AllocationError(
                    f"ECSharedRegion: requested {n_blocks} blocks, "
                    f"only {len(self._free)} free"
                )
            return [self._free.pop() for _ in range(n_blocks)]

    def try_free(self, indices: list[int]) -> bool:
        """Atomically free `indices` iff none of them have a non-zero
        ref count.

        Returns True if the blocks were freed, False if any of them
        were pinned (in which case the region is unchanged). This is
        the primitive the scheduler's LRU eviction loop should use.
        """
        with self._lock:
            if any(idx in self._ref_count for idx in indices):
                return False
            self._free.extend(indices)
            return True

    def free(self, indices: list[int]) -> None:
        """Unconditionally free `indices`.

        Pinned blocks must never reach this path — use `try_free` on
        the eviction side. Hitting the assert means a caller is freeing
        blocks that are still in use by an in-flight transfer, which
        would corrupt the transfer.
        """
        with self._lock:
            for idx in indices:
                assert idx not in self._ref_count, (
                    f"ECSharedRegion: refusing to free pinned block {idx}"
                )
            self._free.extend(indices)

    def pin(self, indices: list[int]) -> None:
        """Increment the ref count on each block in `indices`.

        Pins nest: two `pin()` calls require two matching `unpin()`
        calls. A non-zero ref count blocks `try_free` from reclaiming
        the block, which protects in-flight NIXL transfers from reuse.
        """
        with self._lock:
            for idx in indices:
                self._ref_count[idx] = self._ref_count.get(idx, 0) + 1

    def unpin(self, indices: list[int]) -> None:
        """Decrement the ref count on each block in `indices`.

        Asserts that every block was pinned — unpinning an un-pinned
        block is always a caller bug.
        """
        with self._lock:
            for idx in indices:
                count = self._ref_count.get(idx, 0)
                assert count > 0, f"ECSharedRegion: unpin of un-pinned block {idx}"
                if count == 1:
                    del self._ref_count[idx]
                else:
                    self._ref_count[idx] = count - 1

    def pin_memory(self) -> None:
        """Register the entire mmap as CUDA pinned memory for fast DMA.

        Lifecycle method, called once per process at startup before any
        concurrent access. Not under the lock. Each TP worker process owns
        its own virtual address for the same shared physical pages, so every
        process must register independently.
        """
        if self._base is None or self.is_pinned:
            return
        result = torch.cuda.cudart().cudaHostRegister(
            self.base_ptr, self.total_size_bytes, 0
        )
        if result.value != 0:
            logger.warning(
                "cudaHostRegister failed (code=%d) — "
                "transfers will still work but may be slower (unpinned DMA)",
                result,
            )
        else:
            logger.debug("cudaHostRegister %.2f MB", self.total_size_bytes / 1e6)
            self.is_pinned = True

    def cleanup(self) -> None:
        """Tear down the region. Lifecycle method; no concurrent access."""
        if self.is_pinned and self._base is not None:
            result = torch.cuda.cudart().cudaHostUnregister(self.base_ptr)
            if result.value != 0:
                logger.warning("cudaHostUnregister failed (code=%d)", result)
            self.is_pinned = False
        self.blocks = None  # type: ignore[assignment]
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
        if self._is_creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed EC mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._is_creator = False
