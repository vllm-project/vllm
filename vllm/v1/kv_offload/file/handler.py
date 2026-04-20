# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileOffloadingHandler: Worker-side file I/O for KV cache offloading.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.file.load_store_spec import FileLoadStoreSpec
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult

logger = init_logger(__name__)


@dataclass
class PendingTransfer:
    job_id: int
    done_event: threading.Event = field(default_factory=threading.Event)
    result: TransferResult | None = None


class FileOffloadingHandler(OffloadingHandler):
    """
    Handles KV data transfer between GPU memory and file storage.

    Uses a thread pool for async file I/O.

    Transfer types:
    - GPU -> FILE (offload): serialize GPU tensor slices to binary files
    - FILE -> GPU (restore): deserialize binary files to GPU tensor slices
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        block_size_bytes: int,
        num_threads: int = 4,
    ):
        self.gpu_tensors = gpu_tensors
        self.block_size_bytes = block_size_bytes
        self.num_tensors = len(gpu_tensors)

        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        self._pending: dict[int, PendingTransfer] = {}
        self._lock = threading.Lock()

        # Pre-allocate CPU buffers for each thread
        self._cpu_buffers: dict[int, torch.Tensor] = {}

    def _get_cpu_buffer(self) -> torch.Tensor:
        """Get a cached CPU buffer for transfers."""
        thread_id = threading.get_ident()
        if thread_id not in self._cpu_buffers:
            self._cpu_buffers[thread_id] = torch.empty(
                self.block_size_bytes * self.num_tensors,
                dtype=torch.int8,
                pin_memory=True,
            )
        return self._cpu_buffers[thread_id]

    def transfer_async(self, job_id: int, spec) -> bool:
        """
        Initiate an asynchronous file transfer.

        Args:
            job_id: unique job ID for completion tracking
            spec: (src_spec, dst_spec) tuple

        Returns:
            True if transfer was submitted successfully.
        """
        src_spec, dst_spec = spec

        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, FileLoadStoreSpec
        ):
            # GPU -> FILE (offload)
            pending = PendingTransfer(job_id=job_id)
            with self._lock:
                self._pending[job_id] = pending
            self._executor.submit(
                self._transfer_gpu_to_file, pending, src_spec, dst_spec
            )
        elif isinstance(src_spec, FileLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            # FILE -> GPU (restore)
            pending = PendingTransfer(job_id=job_id)
            with self._lock:
                self._pending[job_id] = pending
            self._executor.submit(
                self._transfer_file_to_gpu, pending, src_spec, dst_spec
            )
        else:
            logger.error(
                "Unsupported transfer: %s -> %s",
                type(src_spec).__name__,
                type(dst_spec).__name__,
            )
            return False

        return True

    def _transfer_gpu_to_file(
        self,
        pending: PendingTransfer,
        gpu_spec: GPULoadStoreSpec,
        file_spec: FileLoadStoreSpec,
    ) -> None:
        """Transfer KV data from GPU to file."""
        t0 = time.monotonic()
        transfer_size = 0

        try:
            src_blocks = gpu_spec.block_ids
            dst_paths = file_spec.file_paths
            dst_offsets = file_spec.block_offsets

            # Handle multiple tensor groups
            group_sizes = (
                gpu_spec.group_sizes
                if hasattr(gpu_spec, "group_sizes")
                else [len(src_blocks)]
            )
            block_indices = (
                gpu_spec.block_indices if hasattr(gpu_spec, "block_indices") else None
            )
            del block_indices  # not used yet

            tensor_idx = 0
            block_idx = 0
            for group_size in group_sizes:
                gpu_tensor = self.gpu_tensors[tensor_idx]
                group_blocks = src_blocks[block_idx : block_idx + group_size]
                group_paths = dst_paths[block_idx : block_idx + group_size]
                group_offsets = dst_offsets[block_idx : block_idx + group_size]

                for block_id, file_path, offset in zip(
                    group_blocks, group_paths, group_offsets
                ):
                    # Copy GPU tensor slice to CPU using torch
                    gpu_slice = gpu_tensor[int(block_id)].cpu()
                    src_bytes = gpu_slice.numpy().tobytes()

                    # Write to file (create if not exists)
                    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.seek(offset)
                        f.write(src_bytes)

                    transfer_size += self.block_size_bytes

                tensor_idx += 1
                block_idx += group_size

            result = TransferResult(
                job_id=pending.job_id,
                success=True,
                transfer_size=transfer_size,
                transfer_time=time.monotonic() - t0,
                transfer_type=("GPU", "FILE"),
            )
        except Exception as e:
            logger.error("GPU->FILE transfer failed for job %d: %r", pending.job_id, e)
            result = TransferResult(
                job_id=pending.job_id,
                success=False,
                transfer_time=time.monotonic() - t0,
                transfer_type=("GPU", "FILE"),
            )

        with self._lock:
            pending.result = result
            pending.done_event.set()

    def _transfer_file_to_gpu(
        self,
        pending: PendingTransfer,
        file_spec: FileLoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
    ) -> None:
        """Transfer KV data from file to GPU."""
        t0 = time.monotonic()
        transfer_size = 0

        try:
            src_paths = file_spec.file_paths
            src_offsets = file_spec.block_offsets
            dst_blocks = gpu_spec.block_ids

            # Handle multiple tensor groups
            group_sizes = (
                gpu_spec.group_sizes
                if hasattr(gpu_spec, "group_sizes")
                else [len(dst_blocks)]
            )
            block_indices = (
                gpu_spec.block_indices if hasattr(gpu_spec, "block_indices") else None
            )
            del block_indices  # not used yet

            tensor_idx = 0
            block_idx = 0
            for group_size in group_sizes:
                gpu_tensor = self.gpu_tensors[tensor_idx]
                group_paths = src_paths[block_idx : block_idx + group_size]
                group_offsets = src_offsets[block_idx : block_idx + group_size]
                group_blocks = dst_blocks[block_idx : block_idx + group_size]

                for file_path, offset, block_id in zip(
                    group_paths, group_offsets, group_blocks
                ):
                    # Read from file
                    with open(file_path, "rb") as f:
                        f.seek(offset)
                        data = f.read(self.block_size_bytes)

                    # Convert to torch tensor and copy to GPU
                    data_tensor = torch.from_numpy(
                        np.frombuffer(data, dtype=np.int8)
                    ).clone()
                    gpu_tensor[int(block_id)].copy_(data_tensor)

                    transfer_size += self.block_size_bytes

                tensor_idx += 1
                block_idx += group_size

            result = TransferResult(
                job_id=pending.job_id,
                success=True,
                transfer_size=transfer_size,
                transfer_time=time.monotonic() - t0,
                transfer_type=("FILE", "GPU"),
            )
        except Exception as e:
            logger.error("FILE->GPU transfer failed for job %d: %r", pending.job_id, e)
            result = TransferResult(
                job_id=pending.job_id,
                success=False,
                transfer_time=time.monotonic() - t0,
                transfer_type=("FILE", "GPU"),
            )

        with self._lock:
            pending.result = result
            pending.done_event.set()

    def get_finished(self) -> list[TransferResult]:
        """Get list of finished transfers."""
        results = []
        with self._lock:
            done_ids = [
                job_id for job_id, p in self._pending.items() if p.done_event.is_set()
            ]
            for job_id in done_ids:
                pending = self._pending.pop(job_id)
                if pending.result:
                    results.append(pending.result)

        return results

    def wait(self, job_ids: set[int]) -> None:
        """Wait for specified jobs to complete (blocking)."""
        for job_id in job_ids:
            with self._lock:
                pending = self._pending.get(job_id)
            if pending:
                pending.done_event.wait()

    def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        # Wait for all pending transfers
        with self._lock:
            pending_ids = list(self._pending.keys())

        for job_id in pending_ids:
            self.wait({job_id})

        self._executor.shutdown(wait=True)
        self._cpu_buffers.clear()
        logger.info("FileOffloadingHandler shutdown complete")
