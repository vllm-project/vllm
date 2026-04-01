# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU ↔ Disk background I/O handler.

Runs in background threads inside the worker process. Each KV tensor
is backed by a single pre-allocated file on NVMe. Blocks are stored
at fixed offsets: disk_block_id * block_size_bytes.

Writes use a single dedicated thread with a queue to avoid GIL
contention from multiple ThreadPoolExecutor workers.
Reads use a thread pool for parallel prefetch I/O.
"""
import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DiskIOWorker:
    """
    Background disk I/O worker for KV cache blocks.

    Writes go through a single dedicated thread (minimal GIL contention).
    Reads go through a thread pool for parallelism.
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

        # Single writer thread with queue — avoids GIL contention
        self._write_queue: queue.SimpleQueue[
            list[tuple[int, int]] | None
        ] = queue.SimpleQueue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="kv-disk-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # Thread pool for reads (need parallelism for prefetch)
        self._read_pool = ThreadPoolExecutor(
            max_workers=io_threads,
            thread_name_prefix="kv-disk-read",
        )

        logger.info(
            "DiskIOWorker: %d tensors, %d blocks, path=%s",
            len(cpu_tensors), num_disk_blocks, disk_path,
        )

    def _writer_loop(self) -> None:
        """Dedicated writer thread. Processes batches from the queue."""
        while True:
            batch = self._write_queue.get()
            if batch is None:
                break  # Shutdown signal
            for cpu_bid, disk_bid in batch:
                for tidx in range(len(self.cpu_tensors)):
                    try:
                        bsize = self.block_sizes[tidx]
                        # Pass numpy array directly — avoids .tobytes() copy
                        data = self.cpu_tensors[tidx][cpu_bid].numpy()
                        os.pwrite(self._fds[tidx], bytes(data), disk_bid * bsize)
                    except Exception as e:
                        logger.warning("Disk write failed: %s", e)

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
        Non-blocking — just puts the batch on the writer queue.
        """
        batch = list(zip(cpu_block_ids, disk_block_ids))
        self._write_queue.put(batch)

    def drain_completed_writes(self) -> int:
        """No-op — single writer thread handles everything."""
        return 0

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
                fut = self._read_pool.submit(
                    self._read_block, tidx, cpu_bid, disk_bid
                )
                futures.append(fut)
        for fut in futures:
            fut.result()

    def submit_async_reads(
        self, cpu_block_ids: list[int], disk_block_ids: list[int]
    ) -> None:
        """
        Queue reads without waiting (non-blocking, fire-and-forget).
        """
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for tidx in range(len(self.cpu_tensors)):
                self._read_pool.submit(
                    self._read_block, tidx, cpu_bid, disk_bid
                )

    def submit_async_reads_tracked(
        self, cpu_block_ids: list[int], disk_block_ids: list[int]
    ) -> list[Future]:
        """
        Queue reads without waiting, return futures for completion tracking.
        """
        futures: list[Future] = []
        for cpu_bid, disk_bid in zip(cpu_block_ids, disk_block_ids):
            for tidx in range(len(self.cpu_tensors)):
                fut = self._read_pool.submit(
                    self._read_block, tidx, cpu_bid, disk_bid
                )
                futures.append(fut)
        return futures

    def shutdown(self) -> None:
        self._write_queue.put(None)  # Signal writer to stop
        self._writer_thread.join(timeout=5)
        self._read_pool.shutdown(wait=True)
        for fd in self._fds:
            os.close(fd)
