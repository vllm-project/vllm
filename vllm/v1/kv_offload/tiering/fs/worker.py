# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import TransferResult
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.common import FileSystemLoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

logger = init_logger(__name__)


def _write_all_at(fd: int, view: memoryview, offset: int) -> None:
    written = 0
    while written < len(view):
        n = os.pwrite(fd, view[written:], offset + written)
        if n <= 0:
            raise OSError("pwrite made no progress")
        written += n


def _read_exact_at(fd: int, view: memoryview, offset: int) -> None:
    read = 0
    while read < len(view):
        chunk = os.pread(fd, len(view) - read, offset + read)
        if not chunk:
            raise OSError(
                f"Short read: expected {len(view)} bytes, read {read}"
            )
        end = read + len(chunk)
        view[read:end] = chunk
        read = end


class FileSystemWorkerTransferHandler:
    """Worker-side CPU shard <-> filesystem transfer executor."""

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        rank: int,
        n_read_threads: int = 4,
        n_write_threads: int = 4,
    ) -> None:
        self._cpu_tensors = cpu_tensors
        self._rank = rank
        self._rank_size = sum(int(t.shape[1]) for t in cpu_tensors)
        self._rank_offset = rank * self._rank_size
        self._store_temp_paths: dict[int, list[str]] = {}
        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_worker_fs",
        )

    def _select_rank_paths(self, paths: list[str], num_ranks: int) -> list[str]:
        if num_ranks <= 0:
            raise ValueError(f"Invalid num_ranks={num_ranks}")
        if self._rank < 0 or self._rank >= num_ranks:
            raise ValueError(
                f"Worker rank {self._rank} is outside num_ranks={num_ranks}"
            )
        if len(paths) % num_ranks != 0:
            raise ValueError(
                f"Cannot split {len(paths)} paths across {num_ranks} ranks"
            )
        per_rank = len(paths) // num_ranks
        start = self._rank * per_rank
        return paths[start : start + per_rank]

    def submit_store(
        self,
        job_id: int,
        src_spec: CPULoadStoreSpec,
        dst_spec: FileSystemLoadStoreSpec,
    ) -> bool:
        if dst_spec.temp_file_paths is None:
            raise ValueError("Filesystem worker stores require temp file paths")
        # Select this rank's slice of the concatenated per-rank paths.
        num_ranks: int = getattr(dst_spec, "num_ranks", 1)
        my_final = self._select_rank_paths(dst_spec.file_paths, num_ranks)
        my_temp = self._select_rank_paths(dst_spec.temp_file_paths, num_ranks)

        if len(src_spec.block_ids) != len(my_final):
            raise ValueError(
                "CPU block count does not match filesystem path count: "
                f"{len(src_spec.block_ids)} != {len(my_final)}"
            )
        if len(src_spec.block_ids) != len(my_temp):
            raise ValueError(
                "CPU block count does not match filesystem temp path count: "
                f"{len(src_spec.block_ids)} != {len(my_temp)}"
            )
        self._store_temp_paths[job_id] = list(my_temp)

        tasks = (
            lambda bid=int(block_id), final_path=final_path, temp_path=temp_path: (
                self._store_one(
                    block_id=bid,
                    final_path=final_path,
                    temp_path=temp_path,
                    block_size=dst_spec.block_size,
                )
            )
            for block_id, final_path, temp_path in zip(
                src_spec.block_ids, my_final, my_temp
            )
        )
        self._pool.enqueue_store(job_id, len(src_spec.block_ids), tasks)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: FileSystemLoadStoreSpec,
        dst_spec: CPULoadStoreSpec,
    ) -> bool:
        # Select this rank's slice of the concatenated per-rank paths.
        num_ranks: int = getattr(src_spec, "num_ranks", 1)
        my_files = self._select_rank_paths(src_spec.file_paths, num_ranks)

        if len(my_files) != len(dst_spec.block_ids):
            raise ValueError(
                "Filesystem path count does not match CPU block count: "
                f"{len(my_files)} != {len(dst_spec.block_ids)}"
            )
        tasks = (
            lambda source_path=source_path, bid=int(block_id): self._load_one(
                source_path=source_path,
                block_id=bid,
            )
            for source_path, block_id in zip(my_files, dst_spec.block_ids)
        )
        self._pool.enqueue_load(job_id, len(dst_spec.block_ids), tasks)
        return True

    def _store_one(
        self,
        *,
        block_id: int,
        final_path: str,
        temp_path: str,
        block_size: int,
    ) -> None:
        if os.path.exists(final_path):
            return

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_TRUNC
        fd = os.open(temp_path, flags, 0o644)
        try:
            os.ftruncate(fd, block_size)
            offset = self._rank_offset
            for tensor in self._cpu_tensors:
                row = tensor[block_id].numpy()
                view = memoryview(row).cast("B")
                _write_all_at(fd, view, offset)
                offset += len(view)
        finally:
            os.close(fd)

    def _load_one(self, *, source_path: str, block_id: int) -> None:
        fd = os.open(source_path, os.O_RDONLY)
        try:
            offset = self._rank_offset
            for tensor in self._cpu_tensors:
                row = tensor[block_id].numpy()
                view = memoryview(row).cast("B")
                _read_exact_at(fd, view, offset)
                offset += len(view)
        finally:
            os.close(fd)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, success in self._pool.get_finished():
            if success:
                self._store_temp_paths.pop(job_id, None)
            else:
                self._cleanup_store_temps({job_id})
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=success,
                    transfer_size=None,
                    transfer_time=None,
                )
            )
        return results

    def _cleanup_store_temps(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            for temp_path in self._store_temp_paths.pop(job_id, []):
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass

    def wait(self, job_ids: set[int] | None = None) -> None:
        self._pool.wait_idle()
        if job_ids is not None:
            self._cleanup_store_temps(job_ids)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
