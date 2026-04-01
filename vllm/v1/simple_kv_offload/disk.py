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
    """High-performance POSIX disk backend for KV cache blocks.

    Optimizations:
    - posix_fadvise(RANDOM): disables wasteful sequential readahead
    - Zero-copy reads: pread directly into tensor data_ptr via ctypes
    - 16 read threads: saturates NVMe queue depth
    - Single writer thread + SimpleQueue: avoids GIL contention
    - posix_fadvise(WILLNEED): prefetch hint for upcoming blocks
    """

    def __init__(
        self,
        disk_path: str,
        tensor_names: list[str],
        bytes_per_block: dict[str, int],
        num_disk_blocks: int,
        read_threads: int = 16,
    ):
        os.makedirs(disk_path, exist_ok=True)

        self._fds: dict[str, int] = {}
        self._bytes_per_block = bytes_per_block
        self._tensor_names = tensor_names

        # libc for zero-copy I/O and fadvise
        import ctypes
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        POSIX_FADV_RANDOM = 1

        # Create/open one file per tensor, pre-allocate
        for name in tensor_names:
            bpb = bytes_per_block[name]
            path = os.path.join(disk_path, f"kv_{name}.bin")
            fd = os.open(path, os.O_RDWR | os.O_CREAT)
            needed = num_disk_blocks * bpb
            if os.fstat(fd).st_size < needed:
                os.ftruncate(fd, needed)
            # Disable readahead — our access is random, not sequential
            self._libc.posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)
            self._fds[name] = fd

        # Single writer thread — avoids GIL contention
        self._write_queue: queue.SimpleQueue = queue.SimpleQueue()
        # GDS-aligned file descriptors (set by GDSDiskBackend if padding needed)
        self._gds_fds: dict[str, int] = {}
        self._gds_aligned_bpb: dict[str, int] = {}
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="kv-disk-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # 16 read threads to saturate NVMe queue depth (~32-64 outstanding)
        self._read_pool = ThreadPoolExecutor(
            max_workers=read_threads,
            thread_name_prefix="kv-disk-read",
        )

        logger.info(
            "PosixDiskBackend: %d tensors, %d blocks, %d read threads, "
            "fadvise=RANDOM, path=%s",
            len(tensor_names), num_disk_blocks, read_threads, disk_path,
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
                        # Get raw pointer to tensor data for zero-copy write
                        tensor_slice = cpu_tensors[name][cpu_bid]
                        data = tensor_slice.numpy().tobytes()

                        # Write to POSIX file
                        os.pwrite(self._fds[name], data, disk_bid * bpb)

                        # Also write to GDS-aligned file if available
                        if name in self._gds_fds:
                            aligned_bpb = self._gds_aligned_bpb[name]
                            padded = (
                                data + b'\x00' * (aligned_bpb - len(data))
                                if len(data) < aligned_bpb
                                else data
                            )
                            os.pwrite(
                                self._gds_fds[name], padded,
                                disk_bid * aligned_bpb,
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
        """Read blocks from disk into CPU tensors. Blocking.

        Uses zero-copy reads: pread via ctypes directly into the
        tensor's data pointer, avoiding Python bytes intermediary.
        """
        futures: list[Future] = []
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for name in self._tensor_names:
                fut = self._read_pool.submit(
                    self._read_one_block_zerocopy, name, cpu_bid, disk_bid,
                    cpu_tensors[name],
                )
                futures.append(fut)
        # Wait for all reads to complete
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                logger.warning("Disk read failed: %s", e)

    def _read_one_block_zerocopy(
        self,
        name: str,
        cpu_block_id: int,
        disk_block_id: int,
        cpu_tensor: torch.Tensor,
    ) -> None:
        """Zero-copy read: pread directly into tensor memory via ctypes.

        Avoids: os.pread → bytes → bytearray → torch.frombuffer → copy_
        Instead: libc.pread → tensor.data_ptr (single copy from NVMe to RAM)
        """
        import ctypes

        bpb = self._bytes_per_block[name]
        fd = self._fds[name]
        file_offset = disk_block_id * bpb

        # Get raw pointer to the target tensor slice
        target = cpu_tensor[cpu_block_id]
        data_ptr = target.data_ptr()

        # pread directly into tensor memory — zero intermediate copies
        nbytes = self._libc.pread(
            fd,
            ctypes.c_void_p(data_ptr),
            ctypes.c_size_t(bpb),
            ctypes.c_longlong(file_offset),
        )
        if nbytes != bpb:
            raise OSError(
                f"pread returned {nbytes}, expected {bpb} "
                f"(fd={fd}, offset={file_offset})"
            )

    def prefetch_hint(self, disk_block_ids: list[int]) -> None:
        """Issue POSIX_FADV_WILLNEED for upcoming block reads.

        Triggers kernel async readahead — hides disk latency
        by prefetching blocks we know we'll need soon.
        """
        POSIX_FADV_WILLNEED = 3
        for disk_bid in disk_block_ids:
            for name in self._tensor_names:
                bpb = self._bytes_per_block[name]
                self._libc.posix_fadvise(
                    self._fds[name],
                    disk_bid * bpb,
                    bpb,
                    POSIX_FADV_WILLNEED,
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
        """Best-effort sync — drain what we can."""
        pass

    def shutdown(self) -> None:
        self._write_queue.put(None)
        self._writer_thread.join(timeout=10)
        self._read_pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        for fd in self._gds_fds.values():
            os.close(fd)


# ---------------------------------------------------------------------------
# GDSDiskBackend — cuFile/KvikIO for direct NVMe↔GPU DMA
# ---------------------------------------------------------------------------

GDS_ALIGNMENT = 4096  # cuFile optimal alignment for DMA path


def _align_up(n: int, alignment: int = GDS_ALIGNMENT) -> int:
    """Round up to the next multiple of alignment."""
    return ((n + alignment - 1) // alignment) * alignment


class GDSDiskBackend(DiskBackend):
    """Disk backend using NVIDIA GPUDirect Storage (GDS).

    Uses torch.cuda.gds.GdsFile for direct NVMe↔GPU DMA transfers,
    bypassing CPU entirely. Optimized for maximum throughput:

    - 4KB-aligned file layout (avoids read-modify-write on NVMe)
    - GPU buffer registration (cuFileBufRegister equivalent via
      gds_register_buffer) eliminates internal staging copies
    - Batched reads: consecutive disk blocks read in single large
      I/O (~1MB optimal) instead of per-block calls
    - Staging buffer for non-aligned block sizes with GPU-to-GPU copy

    Falls back to PosixDiskBackend if GDS is not available.

    Requirements:
      - CUDA 12.6+ (for torch.cuda.gds)
      - XFS or ext4 filesystem (not overlay/tmpfs)
      - nvidia-fs kernel module OR P2PDMA support (CUDA 12.8+)
    """

    # Optimal I/O size for GDS throughput (~1MB per NVIDIA benchmarks)
    OPTIMAL_IO_SIZE = 1024 * 1024

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
        self._aligned_bpb: dict[str, int] = {}
        self._tensor_names = tensor_names
        self._num_disk_blocks = num_disk_blocks
        self._gds_read_available = False
        self._needs_padding = False

        # Compute aligned block sizes
        for name, bpb in bytes_per_block.items():
            aligned = _align_up(bpb)
            self._aligned_bpb[name] = aligned
            if aligned != bpb:
                self._needs_padding = True

        if self._needs_padding:
            logger.info(
                "GDS: block sizes not 4KB-aligned, using padded layout "
                "(e.g., %d → %d bytes, %.1f%% overhead)",
                next(iter(bytes_per_block.values())),
                next(iter(self._aligned_bpb.values())),
                (next(iter(self._aligned_bpb.values()))
                 / next(iter(bytes_per_block.values())) - 1) * 100,
            )

        # Try to open GDS file handles with aligned layout
        try:
            gds_dir = os.path.join(disk_path, "gds")
            os.makedirs(gds_dir, exist_ok=True)

            for name in tensor_names:
                aligned = self._aligned_bpb[name]
                path = os.path.join(gds_dir, f"kv_{name}.bin")
                needed = num_disk_blocks * aligned
                if not os.path.exists(path) or os.path.getsize(path) < needed:
                    fd = os.open(path, os.O_RDWR | os.O_CREAT)
                    os.ftruncate(fd, needed)
                    os.close(fd)
                gds_file = torch.cuda.gds.GdsFile(
                    path, os.O_CREAT | os.O_RDWR
                )
                self._gds_files[name] = gds_file

            self._gds_read_available = True

            # Staging buffer for batched reads — sized for optimal I/O
            # Allocated lazily on first read, registered with GDS
            self._staging_buffers: dict[str, torch.Tensor] = {}
            self._buffers_registered = False

            # Give POSIX backend the GDS-aligned file descriptors
            # so the writer thread mirrors writes to padded files
            if self._needs_padding:
                for name in tensor_names:
                    aligned = self._aligned_bpb[name]
                    path = os.path.join(gds_dir, f"kv_{name}.bin")
                    fd = os.open(path, os.O_RDWR)
                    posix_fallback._gds_fds[name] = fd
                    posix_fallback._gds_aligned_bpb[name] = aligned

            # Compute optimal batch size (blocks per read for ~1MB I/O)
            first_aligned = next(iter(self._aligned_bpb.values()))
            self._batch_size = max(
                1, self.OPTIMAL_IO_SIZE // first_aligned
            )

            logger.info(
                "GDSDiskBackend: %d tensors, %d blocks, GDS reads ENABLED, "
                "batch_size=%d blocks (~%d KB per I/O)%s",
                len(tensor_names), num_disk_blocks,
                self._batch_size,
                self._batch_size * first_aligned // 1024,
                " (padded)" if self._needs_padding else "",
            )
        except Exception as e:
            logger.warning(
                "GDSDiskBackend: failed to init GDS files (%s), "
                "falling back to POSIX for reads",
                e,
            )
            self._gds_files.clear()

    def _ensure_staging_buffers(
        self, device: torch.device
    ) -> None:
        """Lazily allocate and register GPU staging buffers."""
        if self._staging_buffers:
            return

        for name in self._tensor_names:
            aligned = self._aligned_bpb[name]
            # Size for batch_size blocks — enables large I/O
            buf_size = self._batch_size * aligned
            buf = torch.empty(buf_size, dtype=torch.int8, device=device)
            self._staging_buffers[name] = buf

        # Register buffers with GDS for optimal DMA performance
        # (avoids internal staging copies on every read/write)
        if not self._buffers_registered:
            try:
                for buf in self._staging_buffers.values():
                    torch.cuda.gds.gds_register_buffer(
                        buf.untyped_storage()
                    )
                self._buffers_registered = True
                logger.info(
                    "GDS: registered %d staging buffers "
                    "(%d KB each) for optimal DMA",
                    len(self._staging_buffers),
                    next(iter(self._staging_buffers.values())).nbytes // 1024,
                )
            except Exception as e:
                logger.warning(
                    "GDS: buffer registration failed (%s), "
                    "continuing without registration",
                    e,
                )

    def write_blocks(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """Writes go through POSIX (source is CPU memory).

        The writer thread automatically mirrors to GDS-padded files
        when configured, so GDS reads find aligned data.
        """
        self._posix.write_blocks(cpu_tensors, cpu_block_ids, disk_block_ids)

    def read_blocks_to_cpu(
        self,
        cpu_tensors: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> None:
        """CPU reads go through POSIX (unpadded files)."""
        self._posix.read_blocks_to_cpu(
            cpu_tensors, cpu_block_ids, disk_block_ids
        )

    def read_blocks_to_gpu(
        self,
        gpu_tensors: dict[str, torch.Tensor],
        gpu_block_ids: list[int],
        disk_block_ids: list[int],
    ) -> bool:
        """Read directly from NVMe to GPU via GDS. Zero CPU overhead.

        Batches consecutive disk blocks into large I/O operations
        (~1MB per call) for maximum throughput. Uses registered
        staging buffers to avoid internal GDS copies.
        """
        if not self._gds_read_available or not gpu_block_ids:
            return False

        try:
            device = next(iter(gpu_tensors.values())).device
            self._ensure_staging_buffers(device)

            # Process blocks in batches for optimal I/O size
            n_blocks = len(gpu_block_ids)
            for batch_start in range(0, n_blocks, self._batch_size):
                batch_end = min(batch_start + self._batch_size, n_blocks)
                batch_gpu_ids = gpu_block_ids[batch_start:batch_end]
                batch_disk_ids = disk_block_ids[batch_start:batch_end]
                batch_len = batch_end - batch_start

                for name in self._tensor_names:
                    actual_bpb = self._bytes_per_block[name]
                    aligned_bpb = self._aligned_bpb[name]
                    gds_file = self._gds_files[name]
                    staging = self._staging_buffers[name]

                    # Check if disk blocks are consecutive — if so,
                    # read entire range in one I/O call
                    first_disk = batch_disk_ids[0]
                    is_consecutive = all(
                        batch_disk_ids[i] == first_disk + i
                        for i in range(batch_len)
                    )

                    if is_consecutive and batch_len > 1:
                        # Single large I/O for consecutive blocks
                        total_bytes = batch_len * aligned_bpb
                        file_offset = first_disk * aligned_bpb
                        gds_file.load_storage(
                            staging[:total_bytes].untyped_storage(),
                            offset=file_offset,
                        )
                        # Scatter from staging to target GPU blocks
                        for i, gpu_bid in enumerate(batch_gpu_ids):
                            src_start = i * aligned_bpb
                            gpu_tensors[name][gpu_bid].copy_(
                                staging[src_start:src_start + actual_bpb]
                                .view(gpu_tensors[name][gpu_bid].shape)
                            )
                    else:
                        # Non-consecutive: read each block separately
                        for gpu_bid, disk_bid in zip(
                            batch_gpu_ids, batch_disk_ids
                        ):
                            file_offset = disk_bid * aligned_bpb

                            if not self._needs_padding:
                                # Direct read — no staging needed
                                storage = (
                                    gpu_tensors[name][gpu_bid]
                                    .untyped_storage()
                                )
                                gds_file.load_storage(
                                    storage, offset=file_offset
                                )
                            else:
                                # Padded: read into staging, copy actual
                                gds_file.load_storage(
                                    staging[:aligned_bpb].untyped_storage(),
                                    offset=file_offset,
                                )
                                gpu_tensors[name][gpu_bid].copy_(
                                    staging[:actual_bpb].view(
                                        gpu_tensors[name][gpu_bid].shape
                                    )
                                )
            return True
        except Exception as e:
            logger.warning("GDS read failed, falling back to POSIX: %s", e)
            self._gds_read_available = False
            return False

    def sync_writes(self) -> None:
        self._posix.sync_writes()

    def shutdown(self) -> None:
        # Deregister buffers before freeing
        if self._buffers_registered:
            try:
                for buf in self._staging_buffers.values():
                    torch.cuda.gds.gds_deregister_buffer(
                        buf.untyped_storage()
                    )
            except Exception:
                pass
        self._posix.shutdown()
        self._staging_buffers.clear()
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
