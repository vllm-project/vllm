# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base for filesystem-backed offloading workers."""

import time
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    DevicePointers,
    FSLoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.file_mapper import FileMapper

logger = init_logger(__name__)

DEFAULT_MAX_THREADS = 400


@dataclass
class _Transfer:
    job_id: int
    futures: list[Future]
    num_bytes: int
    start_time: float


class FSOffloadingWorker(OffloadingWorker):
    """Abstract base for device <-> filesystem offloading workers.

    Handles: key->path resolution, per-key pointer grouping, thread pool
    management, and completion tracking. Subclasses implement the actual
    I/O via write_block() and read_block().
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        block_size_factor: int,
        max_io_threads: int = DEFAULT_MAX_THREADS,
        pool_initializer=None,
    ):
        self._file_mapper = file_mapper
        self._block_size_factor = block_size_factor
        self._pool = ThreadPoolExecutor(
            max_workers=max_io_threads, initializer=pool_initializer
        )
        self._transfers: dict[int, _Transfer] = {}

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def write_block(
        self, file_path: str, ops: list[tuple[int, int, int]]
    ) -> object | None:
        """Write device data to a file.

        Args:
            file_path: destination file path.
            ops: list of (device_ptr, size_bytes, file_offset) tuples.
        """

    @abstractmethod
    def read_block(
        self, file_path: str, ops: list[tuple[int, int, int]]
    ) -> object | None:
        """Read file data into device memory.

        Args:
            file_path: source file path.
            ops: list of (device_ptr, size_bytes, file_offset) tuples.
        """

    def shutdown_backend(self) -> None:
        """Optional backend cleanup (e.g. cuFileBufDeregister)."""

    # --- OffloadingWorker interface ---

    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, FSLoadStoreSpec)
        torch.cuda.current_stream().synchronize()
        futures, num_bytes = self._submit_io(device_ptrs, dst_spec.keys, is_store=True)
        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            futures=futures,
            num_bytes=num_bytes,
            start_time=time.perf_counter(),
        )
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, device_ptrs: DevicePointers
    ) -> bool:
        assert isinstance(src_spec, FSLoadStoreSpec)
        torch.cuda.current_stream().synchronize()
        futures, num_bytes = self._submit_io(device_ptrs, src_spec.keys, is_store=False)
        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            futures=futures,
            num_bytes=num_bytes,
            start_time=time.perf_counter(),
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_ids: list[int] = []
        for job_id, t in self._transfers.items():
            if all(f.done() for f in t.futures):
                success = True
                for f in t.futures:
                    try:
                        f.result()
                    except Exception:
                        logger.exception("I/O failed for job %d", job_id)
                        success = False
                elapsed = time.perf_counter() - t.start_time
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=success,
                        transfer_size=t.num_bytes,
                        transfer_time=elapsed,
                    )
                )
                finished_ids.append(job_id)
        for job_id in finished_ids:
            del self._transfers[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            t = self._transfers.get(job_id)
            if t is not None:
                for f in t.futures:
                    f.result()

    def shutdown(self) -> None:
        for t in self._transfers.values():
            for f in t.futures:
                f.result()
        self._transfers.clear()
        self._pool.shutdown(wait=True)
        self.shutdown_backend()

    # --- Internal: group device pointers per key and submit I/O ---

    def _submit_io(
        self,
        device_ptrs: DevicePointers,
        keys: list,
        is_store: bool,
    ) -> tuple[list[Future], int]:
        """Group device pointers by offload key and submit per-file I/O."""
        futures: list[Future] = []
        total_bytes = 0

        group_block_counts = device_ptrs.group_block_counts
        group_data_ref_counts = device_ptrs.group_data_ref_counts

        key_idx = 0
        ptr_offset = 0

        for group_size, n_data_refs in zip(group_block_counts, group_data_ref_counts):
            if group_size == 0:
                continue

            n_files = cdiv(group_size, self._block_size_factor)
            for i in range(n_files):
                key = keys[key_idx]
                key_idx += 1

                blk_start = i * self._block_size_factor
                blk_end = min(blk_start + self._block_size_factor, group_size)
                n_blks = blk_end - blk_start

                ops: list[tuple[int, int, int]] = []
                file_offset = 0
                for d in range(n_data_refs):
                    base = ptr_offset + d * group_size + blk_start
                    for b in range(n_blks):
                        idx = base + b
                        dev_ptr = int(device_ptrs.ptrs[idx])
                        size = int(device_ptrs.sizes[idx])
                        ops.append((dev_ptr, size, file_offset))
                        file_offset += size
                        total_bytes += size

                file_path = self._file_mapper.get_file_name(key)
                if is_store:
                    future = self._pool.submit(self.write_block, file_path, ops)
                else:
                    future = self._pool.submit(self.read_block, file_path, ops)
                futures.append(future)

            ptr_offset += n_data_refs * group_size

        return futures, total_bytes
