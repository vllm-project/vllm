# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-blocking interface for an encoder-cache filesystem tier."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    get_ec_entry_path,
    get_ec_temporary_path,
)
from vllm.v1.kv_offload.tiering.backpressure import (
    BackpressureDetector,
    EMABackpressureDetector,
)

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
        ECFileIOCompletion,
    )


@dataclass(frozen=True)
class ECFileEntry:
    """An entry published in the filesystem tier's in-memory index."""

    cache_key: str
    path: Path
    file_size_bytes: int


@dataclass(frozen=True)
class ECReadResult:
    """Completion reported for one accepted filesystem read."""

    cache_key: str
    success: bool
    file_size_bytes: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ECWriteResult:
    """Completion reported for one accepted filesystem write."""

    cache_key: str
    success: bool
    file_size_bytes: int = 0
    error: str | None = None


class _ECFileIOExecutor(Protocol):
    def submit_read(
        self,
        cache_key: str,
        entry_path: Path,
        destinations: Sequence[memoryview],
        file_size_bytes: int,
    ) -> int | None: ...

    def submit_write(
        self,
        cache_key: str,
        entry_path: Path,
        temporary_path: Path,
        sources: Sequence[memoryview],
        file_size_bytes: int,
        *,
        fsync: bool = False,
    ) -> int | None: ...

    def poll_completed(self) -> list[ECFileIOCompletion]: ...

    def drain(self) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class _IndexedEntry:
    entry: ECFileEntry
    generation: int


class ECFileSystemTierManager:
    """Minimal in-memory index and lifecycle around filesystem entry I/O."""

    def __init__(
        self,
        root_dir: Path,
        namespace_digest: str,
        n_read_threads: int,
        n_write_threads: int,
        *,
        fsync: bool = False,
        backpressure: ECFileSystemBackpressureConfig | None = None,
        executor: _ECFileIOExecutor | None = None,
    ) -> None:
        if executor is None:
            from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
                ECFileIOExecutor,
            )

            executor = ECFileIOExecutor(n_read_threads, n_write_threads)

        self._root_dir = root_dir
        self._namespace_digest = namespace_digest
        self._fsync = fsync
        self._executor = executor
        self._backpressure_detector: BackpressureDetector | None = None
        if backpressure is not None and backpressure.enabled:
            self._backpressure_detector = EMABackpressureDetector(
                high_water_s=backpressure.high_water_s,
                low_water_s=backpressure.low_water_s,
            )
        self._accepting = True
        self._shutdown = False
        self._next_generation = 0
        self._entries: dict[str, _IndexedEntry] = {}
        self._pending_reads: dict[int, _IndexedEntry] = {}
        self._pending_writes: dict[int, ECFileEntry] = {}
        self._completed_reads: deque[ECReadResult] = deque()
        self._completed_writes: deque[ECWriteResult] = deque()

    def contains(self, cache_key: str) -> bool:
        """Return an in-memory membership decision without filesystem I/O."""
        return cache_key in self._entries

    def submit_read(
        self,
        cache_key: str,
        destinations: Sequence[memoryview],
    ) -> bool:
        """Submit a read for an indexed entry without filesystem work here."""
        if not self._accepting:
            return False

        indexed_entry = self._entries.get(cache_key)
        if indexed_entry is None:
            raise KeyError(f"encoder-cache entry is not indexed: {cache_key}")

        entry = indexed_entry.entry
        job_id = self._executor.submit_read(
            cache_key,
            entry.path,
            destinations,
            entry.file_size_bytes,
        )
        if job_id is None:
            return False
        self._pending_reads[job_id] = indexed_entry
        return True

    def submit_write(
        self,
        cache_key: str,
        sources: Sequence[memoryview],
        file_size_bytes: int,
    ) -> bool:
        """Submit a write; publish it in memory only after completion."""
        if not self._accepting:
            return False
        if self._backpressure_detector is not None:
            if not self._backpressure_detector.should_store(len(sources)):
                return False

        entry_path = get_ec_entry_path(
            self._root_dir,
            self._namespace_digest,
            cache_key,
        )
        temporary_path = get_ec_temporary_path(entry_path, uuid4().hex)
        entry = ECFileEntry(cache_key, entry_path, file_size_bytes)
        job_id = self._executor.submit_write(
            cache_key,
            entry_path,
            temporary_path,
            sources,
            file_size_bytes,
            fsync=self._fsync,
        )
        if job_id is None:
            return False
        self._pending_writes[job_id] = entry
        return True

    def poll_completed_reads(self) -> list[ECReadResult]:
        """Drain currently available read completions without blocking."""
        self._collect_executor_completions()
        completed = list(self._completed_reads)
        self._completed_reads.clear()
        return completed

    def poll_completed_writes(self) -> list[ECWriteResult]:
        """Drain currently available write completions without blocking."""
        self._collect_executor_completions()
        completed = list(self._completed_writes)
        self._completed_writes.clear()
        return completed

    def drain(self) -> None:
        """Wait for accepted I/O and retain its results for later polling."""
        self._executor.drain()
        self._collect_executor_completions()

    def shutdown(self) -> None:
        """Stop submission, drain accepted I/O, and stop the executor."""
        if self._shutdown:
            return
        self._accepting = False
        self._executor.shutdown()
        self._collect_executor_completions()
        self._shutdown = True

    def _collect_executor_completions(self) -> None:
        for completion in self._executor.poll_completed():
            if completion.operation == "read":
                self._complete_read(completion)
            elif completion.operation == "write":
                self._complete_write(completion)
            else:
                raise RuntimeError(
                    f"unknown filesystem I/O operation: {completion.operation}"
                )

    def _complete_read(self, completion: ECFileIOCompletion) -> None:
        indexed_entry = self._pending_reads.pop(completion.job_id)
        if not completion.success:
            current_entry = self._entries.get(completion.cache_key)
            if current_entry == indexed_entry:
                del self._entries[completion.cache_key]
        self._completed_reads.append(
            ECReadResult(
                cache_key=completion.cache_key,
                success=completion.success,
                file_size_bytes=completion.file_size_bytes,
                error=completion.error,
            )
        )

    def _complete_write(self, completion: ECFileIOCompletion) -> None:
        entry = self._pending_writes.pop(completion.job_id)
        if completion.success:
            if self._backpressure_detector is not None:
                self._backpressure_detector.on_store_completed(
                    completion.queue_time + completion.transfer_time,
                    completion.file_size_bytes,
                )
            self._entries[completion.cache_key] = _IndexedEntry(
                entry=entry,
                generation=self._next_generation,
            )
            self._next_generation += 1
        self._completed_writes.append(
            ECWriteResult(
                cache_key=completion.cache_key,
                success=completion.success,
                file_size_bytes=completion.file_size_bytes,
                error=completion.error,
            )
        )


@runtime_checkable
class ECFileSystemTier(Protocol):
    """Scheduler-facing contract for the optional filesystem tier.

    Implementations own their worker threads and in-memory entry index. Calls
    made from the scheduler path must not perform filesystem I/O or wait for an
    I/O task. The caller keeps submitted buffers alive until the matching
    completion is returned by a poll method.
    """

    def contains(self, cache_key: str) -> bool:
        """Return an in-memory membership decision without filesystem I/O."""
        ...

    def submit_read(
        self,
        cache_key: str,
        destinations: Sequence[memoryview],
    ) -> bool:
        """Queue a filesystem-to-CPU read without blocking.

        Return ``False`` when the tier is no longer accepting work. On
        acceptance, exactly one completion must eventually be returned by
        :meth:`poll_completed_reads`.
        """
        ...

    def submit_write(
        self,
        cache_key: str,
        sources: Sequence[memoryview],
        file_size_bytes: int,
    ) -> bool:
        """Queue a CPU-to-filesystem write without blocking.

        ``file_size_bytes`` is the allocated size of all complete CPU blocks.
        The sources must contain every byte of every block, including undefined
        padding in the final block. On acceptance, exactly one completion must
        eventually be returned by :meth:`poll_completed_writes`. Return
        ``False`` when the tier is no longer accepting work or store pressure
        policy skips this cache opportunity.
        """
        ...

    def poll_completed_reads(self) -> list[ECReadResult]:
        """Drain currently available read completions without blocking."""
        ...

    def poll_completed_writes(self) -> list[ECWriteResult]:
        """Drain currently available write completions without blocking."""
        ...

    def shutdown(self) -> None:
        """Stop accepting work and release resources owned by the tier."""
        ...
