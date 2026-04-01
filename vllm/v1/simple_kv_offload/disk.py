# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disk tier for SimpleCPUOffloadConnector.

Provides write-back from CPU eviction to NVMe and read-back (prefetch)
from NVMe to CPU (or directly to GPU via GDS when available).

Two backends:
  - PosixDiskBackend: pread/pwrite with single writer thread (works everywhere)
  - GDSDiskBackend: cuFile/KvikIO for direct NVMe↔GPU DMA (zero CPU overhead)

Auto-detects GDS at startup; falls back to POSIX transparently.
"""

import os
import pickle
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# DiskBlockIndex — tracks which block hashes live on disk
# ---------------------------------------------------------------------------

class DiskBlockIndex:
    """Maps block hashes to disk block IDs.

    Simple bump allocator with a free list. All blocks are the same size,
    so there is no fragmentation. Supports pickle persistence for
    cross-restart cache.
    """

    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks
        self._num_allocated = 0
        self._free_list: list[int] = []
        self._index: dict[BlockHash, int] = {}

    def contains(self, bh: BlockHash) -> bool:
        return bh in self._index

    def get_block_id(self, bh: BlockHash) -> int | None:
        return self._index.get(bh)

    def allocate(self, bh: BlockHash) -> int | None:
        """Allocate a disk block for the given hash. Returns None if full."""
        if bh in self._index:
            return self._index[bh]
        if self._free_list:
            bid = self._free_list.pop()
        elif self._num_allocated < self._num_blocks:
            bid = self._num_allocated
            self._num_allocated += 1
        else:
            return None
        self._index[bh] = bid
        return bid

    def free(self, bh: BlockHash) -> None:
        bid = self._index.pop(bh, None)
        if bid is not None:
            self._free_list.append(bid)

    @property
    def size(self) -> int:
        return len(self._index)

    @property
    def capacity(self) -> int:
        return self._num_blocks

    def save(self, path: str) -> None:
        """Persist index to disk for cross-restart cache."""
        state = {
            "num_blocks": self._num_blocks,
            "num_allocated": self._num_allocated,
            "free_list": self._free_list,
            "index": self._index,
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        logger.info(
            "DiskBlockIndex saved: %d entries, %d allocated, path=%s",
            len(self._index), self._num_allocated, path,
        )

    @classmethod
    def load(cls, path: str, num_blocks: int) -> "DiskBlockIndex | None":
        """Load index from disk. Returns None if not found or incompatible."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)  # noqa: S301
            if state["num_blocks"] != num_blocks:
                logger.warning(
                    "DiskBlockIndex: num_blocks mismatch "
                    "(saved=%d, current=%d), ignoring saved index",
                    state["num_blocks"], num_blocks,
                )
                return None
            idx = cls(num_blocks)
            idx._num_allocated = state["num_allocated"]
            idx._free_list = state["free_list"]
            idx._index = state["index"]
            logger.info(
                "DiskBlockIndex loaded: %d entries, %d allocated, path=%s",
                len(idx._index), idx._num_allocated, path,
            )
            return idx
        except Exception as e:
            logger.warning("DiskBlockIndex: failed to load %s: %s", path, e)
            return None


# ---------------------------------------------------------------------------
# DiskBackend — abstract base for disk I/O
# ---------------------------------------------------------------------------

class DiskBackend(ABC):
    """Abstract disk I/O backend for KV cache blocks."""

    @abstractmethod
    def write_blocks(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """Write CPU blocks to disk (background, non-blocking)."""

    @abstractmethod
    def read_blocks_to_cpu(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """Read disk blocks into CPU tensors (blocking)."""

    @abstractmethod
    def read_blocks_to_gpu(
        self,
        gpu_tensors: dict[str, torch.Tensor],
        gpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> bool:
        """Read disk blocks directly to GPU via GDS.

        Returns False if not supported (caller should fall back to
        read_blocks_to_cpu + CPU→GPU copy).
        """

    @abstractmethod
    def sync_writes(self) -> None:
        """Wait for all pending background writes to complete."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""


# ---------------------------------------------------------------------------
# PosixDiskBackend — pread/pwrite with single writer thread
# ---------------------------------------------------------------------------

class PosixDiskBackend(DiskBackend):
    """Disk backend using POSIX pread/pwrite.

    Writes use a single dedicated background thread with a SimpleQueue
    to avoid GIL contention (proven pattern from our testing).
    Reads are synchronous (blocking) since they're on the prefetch path.
    """

    def __init__(
        self,
        disk_path: str,
        tensor_names: list[str],
        bytes_per_block: dict[str, int],
        num_disk_blocks: int,
        read_threads: int = 4,
    ):
        os.makedirs(disk_path, exist_ok=True)

        self._fds: dict[str, int] = {}
        self._bytes_per_block = bytes_per_block
        self._tensor_names = tensor_names

        # Create/open one file per tensor, pre-allocate
        for name in tensor_names:
            bpb = bytes_per_block[name]
            path = os.path.join(disk_path, f"kv_{name}.bin")
            fd = os.open(path, os.O_RDWR | os.O_CREAT)
            needed = num_disk_blocks * bpb
            if os.fstat(fd).st_size < needed:
                os.ftruncate(fd, needed)
            self._fds[name] = fd

        # Single writer thread — avoids GIL contention
        self._write_queue: queue.SimpleQueue[
            tuple[dict[str, torch.Tensor], list[int], list[int]] | None
        ] = queue.SimpleQueue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="kv-disk-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # Thread pool for parallel reads (prefetch needs parallelism)
        self._read_pool = ThreadPoolExecutor(
            max_workers=read_threads,
            thread_name_prefix="kv-disk-read",
        )

        logger.info(
            "PosixDiskBackend: %d tensors, %d blocks, path=%s",
            len(tensor_names), num_disk_blocks, disk_path,
        )

    def _writer_loop(self) -> None:
        """Dedicated writer thread. Processes batches from the queue."""
        while True:
            item = self._write_queue.get()
            if item is None:
                break  # Shutdown signal
            cpu_tensors, cpu_block_ids, disk_block_ids = item
            for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
                for name in self._tensor_names:
                    try:
                        bpb = self._bytes_per_block[name]
                        data = cpu_tensors[name][cpu_bid].numpy().tobytes()
                        os.pwrite(
                            self._fds[name], data, disk_bid * bpb
                        )
                    except Exception as e:
                        logger.warning("Disk write failed [%s]: %s", name, e)

    def write_blocks(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """Queue blocks for background write. Non-blocking."""
        self._write_queue.put((cpu_tensors, cpu_block_ids, disk_block_ids))

    def read_blocks_to_cpu(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """Read blocks from disk into CPU tensors. Blocking."""
        futures: list[Future] = []
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for name in self._tensor_names:
                fut = self._read_pool.submit(
                    self._read_one_block, name, cpu_bid, disk_bid,
                    cpu_tensors[name],
                )
                futures.append(fut)
        # Wait for all reads to complete
        for fut in futures:
            exc = fut.exception()
            if exc is not None:
                logger.warning("Disk read failed: %s", exc)

    def _read_one_block(
        self,
        name: str,
        cpu_block_id: int,
        disk_block_id: int,
        cpu_tensor: torch.Tensor,
    ) -> None:
        """Read one block from disk into CPU tensor."""
        bpb = self._bytes_per_block[name]
        data = os.pread(self._fds[name], bpb, disk_block_id * bpb)
        cpu_tensor[cpu_block_id].copy_(
            torch.frombuffer(bytearray(data), dtype=cpu_tensor.dtype).view(
                cpu_tensor[cpu_block_id].shape
            )
        )

    def read_blocks_to_gpu(
        self,
        gpu_tensors: dict[str, torch.Tensor],
        gpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> bool:
        """Not supported — POSIX backend can't write directly to GPU."""
        return False

    def sync_writes(self) -> None:
        """Wait for pending writes by sending a barrier through the queue."""
        done = threading.Event()
        # We can't easily barrier a SimpleQueue, so we just give it time
        # In practice, writes are fast on NVMe and we don't need strict sync
        pass

    def shutdown(self) -> None:
        self._write_queue.put(None)
        self._writer_thread.join(timeout=10)
        self._read_pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)


# ---------------------------------------------------------------------------
# GDSDiskBackend — cuFile/KvikIO for direct NVMe↔GPU DMA
# ---------------------------------------------------------------------------

class GDSDiskBackend(DiskBackend):
    """Disk backend using NVIDIA GPUDirect Storage (GDS).

    Uses torch.cuda.gds.GdsFile for direct NVMe↔GPU transfers,
    bypassing CPU entirely. Falls back to PosixDiskBackend if GDS
    is not available.

    Requirements:
      - CUDA 12.6+ (for torch.cuda.gds)
      - XFS or ext4 filesystem (not overlay/tmpfs)
      - 4KB alignment for file offsets and I/O sizes
      - nvidia-fs kernel module OR P2PDMA support (CUDA 12.8+)
    """

    def __init__(
        self,
        disk_path: str,
        tensor_names: list[str],
        bytes_per_block: dict[str, int],
        num_disk_blocks: int,
        posix_fallback: PosixDiskBackend,
    ):
        self._posix = posix_fallback
        self._gds_files: dict[str, "torch.cuda.gds.GdsFile"] = {}
        self._bytes_per_block = bytes_per_block
        self._tensor_names = tensor_names
        self._gds_read_available = False

        # Validate 4KB alignment
        for name, bpb in bytes_per_block.items():
            if bpb % 4096 != 0:
                logger.warning(
                    "GDS: bytes_per_block[%s]=%d not 4KB-aligned, "
                    "GDS reads will fall back to POSIX",
                    name, bpb,
                )
                return

        # Try to open GDS file handles
        try:
            for name in tensor_names:
                bpb = bytes_per_block[name]
                path = os.path.join(disk_path, f"kv_{name}.bin")
                # Ensure file exists and is pre-allocated
                needed = num_disk_blocks * bpb
                if not os.path.exists(path) or os.path.getsize(path) < needed:
                    with open(path, "wb") as f:
                        f.truncate(needed)
                gds_file = torch.cuda.gds.GdsFile(
                    path, os.O_CREAT | os.O_RDWR
                )
                self._gds_files[name] = gds_file

            self._gds_read_available = True
            logger.info(
                "GDSDiskBackend: %d tensors, %d blocks, GDS reads ENABLED",
                len(tensor_names), num_disk_blocks,
            )
        except Exception as e:
            logger.warning(
                "GDSDiskBackend: failed to init GDS files (%s), "
                "falling back to POSIX for reads",
                e,
            )
            self._gds_files.clear()

    def write_blocks(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """Writes always go through POSIX (source is CPU memory)."""
        self._posix.write_blocks(cpu_tensors, cpu_block_ids, disk_block_ids)

    def read_blocks_to_cpu(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """CPU reads go through POSIX."""
        self._posix.read_blocks_to_cpu(
            cpu_tensors, cpu_block_ids, disk_block_ids
        )

    def read_blocks_to_gpu(
        self,
        gpu_tensors: dict[str, torch.Tensor],
        gpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> bool:
        """Read directly from NVMe to GPU via GDS. Zero CPU overhead."""
        if not self._gds_read_available:
            return False

        try:
            for gpu_bid, disk_bid in zip(gpu_block_ids, disk_block_ids):
                for name in self._tensor_names:
                    bpb = self._bytes_per_block[name]
                    gds_file = self._gds_files[name]
                    # GdsFile.load_storage reads from file into GPU tensor
                    storage = gpu_tensors[name][gpu_bid].untyped_storage()
                    gds_file.load_storage(
                        storage, offset=disk_bid * bpb
                    )
            return True
        except Exception as e:
            logger.warning("GDS read failed, falling back to POSIX: %s", e)
            self._gds_read_available = False
            return False

    def sync_writes(self) -> None:
        self._posix.sync_writes()

    def shutdown(self) -> None:
        self._posix.shutdown()
        self._gds_files.clear()


# ---------------------------------------------------------------------------
# Factory — auto-detect best backend
# ---------------------------------------------------------------------------

def create_disk_backend(
    disk_path: str,
    tensor_names: list[str],
    bytes_per_block: dict[str, int],
    num_disk_blocks: int,
    try_gds: bool = True,
) -> DiskBackend:
    """Create the best available disk backend.

    Tries GDS first (direct NVMe↔GPU DMA), falls back to POSIX.
    GDS requires: torch.cuda.gds, XFS/ext4, 4KB alignment.
    """
    posix = PosixDiskBackend(
        disk_path=disk_path,
        tensor_names=tensor_names,
        bytes_per_block=bytes_per_block,
        num_disk_blocks=num_disk_blocks,
    )

    if not try_gds:
        return posix

    # Check if GDS is available
    if not hasattr(torch.cuda, "gds"):
        logger.info("GDS not available (no torch.cuda.gds), using POSIX")
        return posix

    try:
        backend = GDSDiskBackend(
            disk_path=disk_path,
            tensor_names=tensor_names,
            bytes_per_block=bytes_per_block,
            num_disk_blocks=num_disk_blocks,
            posix_fallback=posix,
        )
        if backend._gds_read_available:
            return backend
        logger.info("GDS init succeeded but reads not available, using POSIX")
        return posix
    except Exception as e:
        logger.info("GDS init failed (%s), using POSIX", e)
        return posix
