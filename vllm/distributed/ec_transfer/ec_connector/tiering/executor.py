# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Asynchronous filesystem I/O execution for encoder-cache entries."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from vllm.distributed.ec_transfer.ec_connector.tiering.io import (
    load_entry,
    remove_entry,
    store_entry,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

ECFileIOOperation = Literal["read", "write", "delete"]


@dataclass(frozen=True)
class ECFileIOCompletion:
    """Result and timing data for one accepted entry I/O job."""

    job_id: int
    operation: ECFileIOOperation
    cache_key: str
    success: bool
    file_size_bytes: int
    submit_time: float
    queue_time: float
    transfer_time: float
    error: str | None = None


@dataclass(frozen=True)
class ECFileIOStats:
    """Current executor pressure signals."""

    pending_read_jobs: int
    pending_read_bytes: int
    pending_write_jobs: int
    pending_write_bytes: int
    pending_delete_jobs: int
    pending_delete_bytes: int
    oldest_read_queue_age: float
    oldest_write_queue_age: float
    oldest_delete_queue_age: float


@dataclass
class _PendingJob:
    operation: ECFileIOOperation
    cache_key: str
    file_size_bytes: int
    submit_time: float
    start_time: float | None = None
    error: str | None = None


class ECFileIOExecutor:
    """Run one complete ``.ec`` file transfer per thread-pool task."""

    def __init__(self, n_read_threads: int, n_write_threads: int) -> None:
        if n_read_threads <= 0 or n_write_threads <= 0:
            raise ValueError("filesystem I/O thread counts must be positive")

        self._pool = DualQueueThreadPool(
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
            thread_name_prefix="vllm_ec_fs",
        )
        self._lock = Lock()
        self._next_job_id = 0
        self._accepting = True
        self._shutdown = False
        self._pending: dict[int, _PendingJob] = {}
        self._completed: deque[ECFileIOCompletion] = deque()
        self._pending_read_jobs = 0
        self._pending_read_bytes = 0
        self._pending_write_jobs = 0
        self._pending_write_bytes = 0
        self._pending_delete_jobs = 0
        self._pending_delete_bytes = 0

    def submit_read(
        self,
        cache_key: str,
        entry_path: Path,
        destinations: Sequence[memoryview],
        file_size_bytes: int,
    ) -> int | None:
        """Submit one complete entry read, or return ``None`` after shutdown."""
        buffers = tuple(destinations)
        return self._submit(
            operation="read",
            cache_key=cache_key,
            file_size_bytes=file_size_bytes,
            task_factory=lambda: load_entry(
                entry_path,
                buffers,
                file_size_bytes,
            ),
        )

    def submit_write(
        self,
        cache_key: str,
        entry_path: Path,
        temporary_path: Path,
        sources: Sequence[memoryview],
        file_size_bytes: int,
        *,
        fsync: bool = False,
    ) -> int | None:
        """Submit one complete entry write, or return ``None`` after shutdown."""
        buffers = tuple(sources)
        return self._submit(
            operation="write",
            cache_key=cache_key,
            file_size_bytes=file_size_bytes,
            task_factory=lambda: store_entry(
                entry_path,
                temporary_path,
                buffers,
                file_size_bytes,
                fsync=fsync,
            ),
        )

    def submit_delete(
        self,
        cache_key: str,
        entry_path: Path,
        file_size_bytes: int,
    ) -> int | None:
        """Submit one entry removal, or return ``None`` after shutdown."""
        return self._submit(
            operation="delete",
            cache_key=cache_key,
            file_size_bytes=file_size_bytes,
            task_factory=lambda: remove_entry(entry_path),
        )

    def _submit(
        self,
        operation: ECFileIOOperation,
        cache_key: str,
        file_size_bytes: int,
        task_factory: Callable[[], None],
    ) -> int | None:
        if file_size_bytes <= 0:
            raise ValueError("file_size_bytes must be positive")

        submit_time = time.monotonic()
        with self._lock:
            if not self._accepting:
                return None

            job_id = self._next_job_id
            self._next_job_id += 1
            self._pending[job_id] = _PendingJob(
                operation=operation,
                cache_key=cache_key,
                file_size_bytes=file_size_bytes,
                submit_time=submit_time,
            )
            self._increment_pending(operation, file_size_bytes)

            def task() -> None:
                with self._lock:
                    self._pending[job_id].start_time = time.monotonic()
                try:
                    task_factory()
                except Exception as exc:
                    with self._lock:
                        self._pending[job_id].error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    raise

            if operation == "read":
                self._pool.enqueue_load(job_id, 1, [task])
            else:
                self._pool.enqueue_store(job_id, 1, [task])
            return job_id

    def poll_completed(self) -> list[ECFileIOCompletion]:
        """Drain currently available completions without blocking."""
        with self._lock:
            self._collect_completed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def stats(self) -> ECFileIOStats:
        """Return current pending counts, bytes, and oldest queue ages."""
        now = time.monotonic()
        with self._lock:
            return ECFileIOStats(
                pending_read_jobs=self._pending_read_jobs,
                pending_read_bytes=self._pending_read_bytes,
                pending_write_jobs=self._pending_write_jobs,
                pending_write_bytes=self._pending_write_bytes,
                pending_delete_jobs=self._pending_delete_jobs,
                pending_delete_bytes=self._pending_delete_bytes,
                oldest_read_queue_age=self._oldest_queue_age("read", now),
                oldest_write_queue_age=self._oldest_queue_age("write", now),
                oldest_delete_queue_age=self._oldest_queue_age("delete", now),
            )

    def drain(self) -> None:
        """Wait for every accepted job and retain its completion for polling."""
        self._pool.wait_idle()
        with self._lock:
            self._collect_completed()

    def shutdown(self) -> None:
        """Stop submission, drain accepted jobs, then stop worker threads."""
        with self._lock:
            if self._shutdown:
                return
            self._accepting = False
        self.drain()
        self._pool.shutdown(wait=True)
        with self._lock:
            self._shutdown = True

    def _collect_completed(self) -> None:
        for job_id, success, transfer_time in self._pool.get_finished():
            job = self._pending.pop(job_id)
            self._decrement_pending(job.operation, job.file_size_bytes)
            assert job.start_time is not None
            self._completed.append(
                ECFileIOCompletion(
                    job_id=job_id,
                    operation=job.operation,
                    cache_key=job.cache_key,
                    success=success,
                    file_size_bytes=job.file_size_bytes,
                    submit_time=job.submit_time,
                    queue_time=job.start_time - job.submit_time,
                    transfer_time=transfer_time,
                    error=job.error,
                )
            )

    def _increment_pending(
        self, operation: ECFileIOOperation, file_size_bytes: int
    ) -> None:
        if operation == "read":
            self._pending_read_jobs += 1
            self._pending_read_bytes += file_size_bytes
        elif operation == "write":
            self._pending_write_jobs += 1
            self._pending_write_bytes += file_size_bytes
        else:
            self._pending_delete_jobs += 1
            self._pending_delete_bytes += file_size_bytes

    def _decrement_pending(
        self, operation: ECFileIOOperation, file_size_bytes: int
    ) -> None:
        if operation == "read":
            self._pending_read_jobs -= 1
            self._pending_read_bytes -= file_size_bytes
        elif operation == "write":
            self._pending_write_jobs -= 1
            self._pending_write_bytes -= file_size_bytes
        else:
            self._pending_delete_jobs -= 1
            self._pending_delete_bytes -= file_size_bytes

    def _oldest_queue_age(
        self, operation: ECFileIOOperation, now: float
    ) -> float:
        submit_times = (
            job.submit_time
            for job in self._pending.values()
            if job.operation == operation and job.start_time is None
        )
        oldest_submit_time = min(submit_times, default=now)
        return now - oldest_submit_time
