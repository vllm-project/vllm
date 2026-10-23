# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-blocking interface for an encoder-cache filesystem tier."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    EC_FILE_SUFFIX,
    get_ec_entry_path,
    get_ec_namespace_path,
    get_ec_temporary_path,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.backpressure import (
    BackpressureDetector,
    EMABackpressureDetector,
)

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
        ECFileIOCompletion,
    )

logger = init_logger(__name__)


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

    def submit_delete(
        self,
        cache_key: str,
        entry_path: Path,
        file_size_bytes: int,
    ) -> int | None: ...

    def poll_completed(self) -> list[ECFileIOCompletion]: ...

    def drain(self) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class _IndexedEntry:
    entry: ECFileEntry
    generation: int


class ECFileSystemTierManager:
    """Capacity-bounded LRU index and lifecycle around entry I/O."""

    def __init__(
        self,
        root_dir: Path,
        namespace_digest: str,
        capacity_bytes: int,
        n_read_threads: int,
        n_write_threads: int,
        *,
        fsync: bool = False,
        backpressure: ECFileSystemBackpressureConfig | None = None,
        executor: _ECFileIOExecutor | None = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("filesystem capacity_bytes must be positive")
        if executor is None:
            from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
                ECFileIOExecutor,
            )

            executor = ECFileIOExecutor(n_read_threads, n_write_threads)

        self._root_dir = root_dir
        self._namespace_digest = namespace_digest
        self._capacity_bytes = capacity_bytes
        self._used_bytes = 0
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
        self._entries: OrderedDict[str, _IndexedEntry] = OrderedDict()
        self._pending_reads: dict[int, _IndexedEntry] = {}
        self._pending_writes: dict[int, ECFileEntry] = {}
        self._pending_write_keys: set[str] = set()
        self._pending_deletes: dict[int, _IndexedEntry] = {}
        self._deleting_keys: set[str] = set()
        self._read_pins: Counter[str] = Counter()
        self._completed_reads: deque[ECReadResult] = deque()
        self._completed_writes: deque[ECWriteResult] = deque()
        self._recover_index()
        self._evict_to_capacity()

    @property
    def used_bytes(self) -> int:
        """Return bytes owned by entries in the in-memory index."""
        return self._used_bytes

    def contains(self, cache_key: str) -> bool:
        """Return an in-memory membership decision without filesystem I/O."""
        if cache_key not in self._entries:
            return False
        self._entries.move_to_end(cache_key)
        return True

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
        self._read_pins[cache_key] += 1
        self._entries.move_to_end(cache_key)
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
        if file_size_bytes <= 0:
            raise ValueError("file_size_bytes must be positive")
        if file_size_bytes > self._capacity_bytes:
            return False
        if (
            cache_key in self._pending_write_keys
            or cache_key in self._deleting_keys
        ):
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
        self._pending_write_keys.add(cache_key)
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
        # Write/read completions can enqueue internal eviction deletes.
        self._executor.drain()
        self._collect_executor_completions()

    def shutdown(self) -> None:
        """Stop submission, drain accepted I/O, and stop the executor."""
        if self._shutdown:
            return
        self._accepting = False
        self.drain()
        self._executor.shutdown()
        self._collect_executor_completions()
        self._shutdown = True

    def _collect_executor_completions(self) -> None:
        for completion in self._executor.poll_completed():
            if completion.operation == "read":
                self._complete_read(completion)
            elif completion.operation == "write":
                self._complete_write(completion)
            elif completion.operation == "delete":
                self._complete_delete(completion)
            else:
                raise RuntimeError(
                    f"unknown filesystem I/O operation: {completion.operation}"
                )

    def _complete_read(self, completion: ECFileIOCompletion) -> None:
        indexed_entry = self._pending_reads.pop(completion.job_id)
        self._read_pins[completion.cache_key] -= 1
        if self._read_pins[completion.cache_key] == 0:
            del self._read_pins[completion.cache_key]
        if not completion.success:
            current_entry = self._entries.get(completion.cache_key)
            if current_entry == indexed_entry:
                del self._entries[completion.cache_key]
                self._used_bytes -= indexed_entry.entry.file_size_bytes
        self._completed_reads.append(
            ECReadResult(
                cache_key=completion.cache_key,
                success=completion.success,
                file_size_bytes=completion.file_size_bytes,
                error=completion.error,
            )
        )
        self._evict_to_capacity()

    def _complete_write(self, completion: ECFileIOCompletion) -> None:
        entry = self._pending_writes.pop(completion.job_id)
        self._pending_write_keys.remove(completion.cache_key)
        if completion.success:
            if self._backpressure_detector is not None:
                self._backpressure_detector.on_store_completed(
                    completion.queue_time + completion.transfer_time,
                    completion.file_size_bytes,
                )
            previous = self._entries.pop(completion.cache_key, None)
            if previous is not None:
                self._used_bytes -= previous.entry.file_size_bytes
            self._entries[completion.cache_key] = _IndexedEntry(
                entry=entry,
                generation=self._next_generation,
            )
            self._used_bytes += entry.file_size_bytes
            self._next_generation += 1
        self._completed_writes.append(
            ECWriteResult(
                cache_key=completion.cache_key,
                success=completion.success,
                file_size_bytes=completion.file_size_bytes,
                error=completion.error,
            )
        )
        self._evict_to_capacity()

    def _complete_delete(self, completion: ECFileIOCompletion) -> None:
        indexed_entry = self._pending_deletes.pop(completion.job_id)
        self._deleting_keys.remove(indexed_entry.entry.cache_key)
        if not completion.success:
            logger.warning(
                "Failed to delete evicted encoder-cache entry %s: %s",
                indexed_entry.entry.path,
                completion.error,
            )

    def _recover_index(self) -> None:
        namespace_path = get_ec_namespace_path(
            self._root_dir,
            self._namespace_digest,
        )
        recovered: list[tuple[int, str, ECFileEntry]] = []
        for path in namespace_path.rglob(f"*{EC_FILE_SUFFIX}"):
            if path.is_symlink():
                continue
            cache_key = path.name.removesuffix(EC_FILE_SUFFIX)
            try:
                expected_path = get_ec_entry_path(
                    self._root_dir,
                    self._namespace_digest,
                    cache_key,
                )
                stat_result = path.stat()
            except (OSError, ValueError):
                continue
            if path != expected_path or not S_ISREG(stat_result.st_mode):
                continue
            if stat_result.st_size <= 0:
                continue
            recovered.append(
                (
                    stat_result.st_mtime_ns,
                    cache_key,
                    ECFileEntry(cache_key, path, stat_result.st_size),
                )
            )

        for _, cache_key, entry in sorted(recovered):
            self._entries[cache_key] = _IndexedEntry(
                entry=entry,
                generation=self._next_generation,
            )
            self._used_bytes += entry.file_size_bytes
            self._next_generation += 1

    def _evict_to_capacity(self) -> None:
        while self._used_bytes > self._capacity_bytes:
            victim: tuple[str, _IndexedEntry] | None = None
            for cache_key, indexed_entry in self._entries.items():
                if (
                    self._read_pins[cache_key] == 0
                    and cache_key not in self._pending_write_keys
                ):
                    victim = (cache_key, indexed_entry)
                    break
            if victim is None:
                return

            cache_key, indexed_entry = victim
            del self._entries[cache_key]
            self._used_bytes -= indexed_entry.entry.file_size_bytes
            job_id = self._executor.submit_delete(
                cache_key,
                indexed_entry.entry.path,
                indexed_entry.entry.file_size_bytes,
            )
            if job_id is None:
                self._entries[cache_key] = indexed_entry
                self._entries.move_to_end(cache_key, last=False)
                self._used_bytes += indexed_entry.entry.file_size_bytes
                return
            self._pending_deletes[job_id] = indexed_entry
            self._deleting_keys.add(cache_key)


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
        policy, capacity, or same-key lifecycle state skips this cache
        opportunity.
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
