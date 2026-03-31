# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU ↔ Disk background I/O handler.

Runs in background threads inside the worker process. Each KV tensor
is backed by a single pre-allocated file on NVMe. Blocks are stored
at fixed offsets: disk_block_id * block_size_bytes.

Writes are fire-and-forget (write-through from CPU after GPU→CPU).
Reads are prefetches (disk→CPU before CPU→GPU load).
"""
import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DiskIOWorker:
    """
    Background disk I/O worker for KV cache blocks.

    Manages a thread pool that reads/writes blocks between CPU tensors
    and NVMe-backed files. Each tensor has its own file.
    """

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        disk_path: str,
        num_disk_blocks: int,
        io_threads: int = 4,
    ):
        self.cpu_tensors = cpu_tensors
        self.block_sizes = [t.shape[1] for t in cpu_tensors]

        # Create/open one file per tensor
        os.makedirs(disk_path, exist_ok=True)
        self._fds: list[int] = []
        for idx, bsize in enumerate(self.block_sizes):
            path = os.path.join(disk_path, f"kv_tier3_{idx}.bin")
            fd = os.open(path, os.O_RDWR | os.O_CREAT)
            needed = num_disk_blocks * bsize
            if os.fstat(fd).st_size < needed:
                os.ftruncate(fd, needed)
            self._fds.append(fd)

        self._pool = ThreadPoolExecutor(
            max_workers=io_threads,
            thread_name_prefix="kv-disk",
        )
        self._pending_writes: deque[Future] = deque()
        self._pending_reads: deque[Future] = deque()
        self._lock = threading.Lock()

        logger.info(
            "DiskIOWorker: %d tensors, %d blocks, %d threads, path=%s",
            len(cpu_tensors), num_disk_blocks, io_threads, disk_path,
        )

    def _write_block(self, tensor_idx: int, cpu_block: int,
                     disk_block: int) -> None:
        """Copy one block from CPU tensor to disk file."""
        bsize = self.block_sizes[tensor_idx]
        data = self.cpu_tensors[tensor_idx][cpu_block].numpy().tobytes()
        os.pwrite(self._fds[tensor_idx], data, disk_block * bsize)

    def _read_block(self, tensor_idx: int, cpu_block: int,
                    disk_block: int) -> None:
        """Copy one block from disk file to CPU tensor."""
        bsize = self.block_sizes[tensor_idx]
        data = os.pread(self._fds[tensor_idx], bsize, disk_block * bsize)
        self.cpu_tensors[tensor_idx][cpu_block].copy_(
            torch.frombuffer(bytearray(data), dtype=torch.int8)
        )

    def submit_writes(
        self, cpu_block_ids: list[int], disk_block_ids: list[int]
    ) -> None:
        """
        Queue background writes for a batch of blocks.
        Each (cpu_block, disk_block) pair writes ALL tensors for that block.
        Fire-and-forget — we don't wait for completion.
        """
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for tidx in range(len(self.cpu_tensors)):
                fut = self._pool.submit(
                    self._write_block, tidx, cpu_bid, disk_bid
                )
                with self._lock:
                    self._pending_writes.append(fut)

    def submit_reads(
        self, cpu_block_ids: list[int], disk_block_ids: list[int]
    ) -> None:
        """
        Queue reads and WAIT for them (blocking).
        Use submit_async_reads for non-blocking.
        """
        futures: list[Future] = []
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for tidx in range(len(self.cpu_tensors)):
                fut = self._pool.submit(
                    self._read_block, tidx, cpu_bid, disk_bid
                )
                futures.append(fut)
        for fut in futures:
            fut.result()

    def submit_async_reads(
        self, cpu_block_ids: list[int], disk_block_ids: list[int]
    ) -> None:
        """
        Queue reads without waiting (non-blocking).
        Reads complete in background. The data lands in the CPU
        tensors asynchronously — caller should not assume data is
        ready until the next engine step.
        """
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for tidx in range(len(self.cpu_tensors)):
                self._pool.submit(
                    self._read_block, tidx, cpu_bid, disk_bid
                )

    def drain_completed_writes(self) -> int:
        """Drain completed write futures. Returns count drained."""
        count = 0
        with self._lock:
            while self._pending_writes:
                if self._pending_writes[0].done():
                    fut = self._pending_writes.popleft()
                    exc = fut.exception()
                    if exc is not None:
                        logger.warning("Disk write failed: %s", exc)
                    count += 1
                else:
                    break
        return count

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        for fd in self._fds:
            os.close(fd)
