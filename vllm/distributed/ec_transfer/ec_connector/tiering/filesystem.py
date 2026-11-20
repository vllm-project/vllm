# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-blocking interface for an encoder-cache filesystem tier."""

from __future__ import annotations

import fcntl
import os
from collections import Counter, OrderedDict, deque
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

import regex as re

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    EC_FILE_SUFFIX,
    get_ec_entry_path,
    get_ec_namespace_path,
    get_ec_temporary_path,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.lookup import ECFileLookup
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.tiering.backpressure import (
    BackpressureDetector,
    EMABackpressureDetector,
)

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
        ECFileIOCompletion,
        ECFileIOStats,
    )

logger = init_logger(__name__)

_OWNER = re.compile(r"[0-9a-f]{32}\Z")
_TEMPORARY = re.compile(r"\.([0-9a-f]{64})\.ec\.([0-9a-f]{32})\.[0-9a-f]{16}\.tmp\Z")


def _iter_entry_files(namespace_path: Path) -> Iterator[Path]:
    directories = [namespace_path]
    while directories:
        try:
            entries = os.scandir(directories.pop())
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        directories.append(Path(entry.path))
                    elif entry.name.endswith(EC_FILE_SUFFIX) and not entry.is_symlink():
                        yield Path(entry.path)
                except FileNotFoundError:
                    continue


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
    queue_time: float = 0.0
    transfer_time: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class ECWriteResult:
    """Completion reported for one accepted filesystem write."""

    cache_key: str
    success: bool
    file_size_bytes: int = 0
    queue_time: float = 0.0
    transfer_time: float = 0.0
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

    def stats(self) -> ECFileIOStats: ...

    def drain(self) -> None: ...

    def shutdown(self) -> None: ...


class ECTemporaryFileOwner:
    """Hold an instance lease until its writes and cleanup have finished."""

    def __init__(self, namespace_path: Path) -> None:
        self._directory = namespace_path / ".owners"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = namespace_path / ".tmp"
        self._locks: dict[str, int] = {}
        self.orphaned: dict[Path, int] = {}
        while True:
            self.owner_id = uuid4().hex
            fd = self._try_lock(self._directory / f"{self.owner_id}.lock", create=True)
            if fd is not None:
                self._locks[self.owner_id] = fd
                break
        try:
            with os.scandir(self._directory) as entries:
                owners = [
                    Path(entry.path)
                    for entry in entries
                    if entry.name.endswith(".lock")
                ]
            for path in owners:
                owner = path.stem
                if not _OWNER.fullmatch(owner) or owner == self.owner_id:
                    continue
                fd = self._try_lock(path)
                if fd is None:
                    continue
                self._locks[owner] = fd
                directory = self._temporary_directory / owner
                if directory.is_symlink():
                    raise OSError(
                        f"encoder-cache temporary directory is a symlink: {directory}"
                    )
                try:
                    temporary_entries = os.scandir(directory)
                except FileNotFoundError:
                    pass
                else:
                    with temporary_entries:
                        for item in temporary_entries:
                            match = _TEMPORARY.fullmatch(item.name)
                            if match is None or match.group(2) != owner:
                                continue
                            path = Path(item.path)
                            try:
                                info = path.lstat()
                            except FileNotFoundError:
                                continue
                            if S_ISREG(info.st_mode):
                                path.unlink(missing_ok=True)
                self._release(owner, remove=True, strict=True)
        except BaseException:
            for owner in tuple(self._locks):
                self._release(owner, remove=False)
            raise

    @staticmethod
    def _try_lock(path: Path, *, create: bool = False) -> int | None:
        flags = os.O_RDWR | os.O_NOFOLLOW
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(path, flags, 0o600)
        except FileNotFoundError:
            if create:
                raise
            return None
        except FileExistsError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = os.fstat(fd)
            current = path.lstat()
            if S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (
                current.st_dev,
                current.st_ino,
            ):
                return fd
        except (BlockingIOError, FileNotFoundError):
            pass
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
        return None

    def _release(self, owner: str, *, remove: bool, strict: bool = False) -> None:
        fd = self._locks.pop(owner)
        try:
            if remove:
                with suppress(FileNotFoundError):
                    (self._temporary_directory / owner).rmdir()
                (self._directory / f"{owner}.lock").unlink(missing_ok=True)
        except OSError:
            if strict:
                raise
            logger.warning("Failed to remove encoder-cache owner files %s", owner)
        finally:
            os.close(fd)

    def close(self, remaining: dict[Path, int]) -> None:
        unfinished = {
            match.group(2)
            for path in remaining
            if (match := _TEMPORARY.fullmatch(path.name)) is not None
        }
        for owner in tuple(self._locks):
            self._release(owner, remove=owner not in unfinished)


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
        owner = ECTemporaryFileOwner(get_ec_namespace_path(root_dir, namespace_digest))
        if executor is None:
            from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
                ECFileIOExecutor,
            )

            try:
                executor = ECFileIOExecutor(n_read_threads, n_write_threads)
            except BaseException:
                owner.close(owner.orphaned)
                raise

        self._root_dir = root_dir
        self._namespace_digest = namespace_digest
        self._capacity_bytes = capacity_bytes
        self._used_bytes = 0
        self._reserved_write_bytes = 0
        self._pending_delete_bytes = 0
        self._fsync = fsync
        self._executor = executor
        self._temporary_owner = owner
        self._temporary_cleanup = dict(owner.orphaned)
        self._temporary_cleanup_bytes = sum(self._temporary_cleanup.values())
        self._pending_temporary_deletes: dict[int, Path] = {}
        self._write_temporary: dict[int, Path] = {}
        self._backpressure_detector: BackpressureDetector | None = None
        if backpressure is not None and backpressure.enabled:
            self._backpressure_detector = EMABackpressureDetector(
                high_water_s=backpressure.high_water_s,
                low_water_s=backpressure.low_water_s,
            )
        self._accepting = True
        self._shutdown = False
        self._entries: OrderedDict[str, ECFileEntry] = OrderedDict()
        self._invalid_keys: set[str] = set()
        self._pending_reads: dict[int, ECFileEntry] = {}
        self._pending_writes: dict[int, ECFileEntry] = {}
        self._pending_write_keys: set[str] = set()
        self._pending_deletes: dict[int, ECFileEntry] = {}
        self._deleting_keys: set[str] = set()
        self._read_pins: Counter[str] = Counter()
        self._completed_reads: deque[ECReadResult] = deque()
        self._completed_writes: deque[ECWriteResult] = deque()
        try:
            self._recover_index()
            self._lookup = ECFileLookup(root_dir, namespace_digest)
            self._evict_to_capacity()
            self._retry_temporary_cleanup()
        except BaseException:
            if hasattr(self, "_lookup"):
                self._lookup.shutdown()
            self._executor.shutdown()
            owner.close(self._temporary_cleanup)
            raise

    @property
    def used_bytes(self) -> int:
        """Return indexed bytes, write reservations and unconfirmed deletions."""
        return (
            self._used_bytes
            + self._reserved_write_bytes
            + self._pending_delete_bytes
            + self._temporary_cleanup_bytes
        )

    @property
    def capacity_bytes(self) -> int:
        """Return the configured filesystem capacity."""
        return self._capacity_bytes

    def io_stats(self) -> ECFileIOStats:
        """Return the executor's current non-blocking pressure snapshot."""
        return self._executor.stats()

    def contains(self, cache_key: str) -> bool:
        """Return an in-memory membership decision without filesystem I/O."""
        if cache_key in self._invalid_keys or cache_key not in self._entries:
            return False
        self._entries.move_to_end(cache_key)
        return True

    def lookup(
        self, cache_key: str, file_size_bytes: int, request_id: str
    ) -> bool | None:
        """Use indexed entries immediately; probe unknown files asynchronously."""
        if file_size_bytes <= 0:
            raise ValueError("file_size_bytes must be positive")
        if not self._accepting or file_size_bytes > self._capacity_bytes:
            return False
        if cache_key in self._deleting_keys or cache_key in self._invalid_keys:
            return False
        if cache_key in self._pending_write_keys:
            return None
        entry = self._entries.get(cache_key)
        if entry is not None and entry.file_size_bytes != file_size_bytes:
            self._invalid_keys.add(cache_key)
            self._entries.move_to_end(cache_key, last=False)
            self._evict_to_capacity()
            return False
        key = self._lookup.key(cache_key, file_size_bytes)
        result = self._lookup.lookup(key, ReqContext(req_id=request_id))
        if entry is not None:
            # Register the owner so a failed indexed read can be marked MISS.
            # Supersede the pending probe before flush; no file check is queued.
            self._lookup.mark_present([key])
            self._entries.move_to_end(cache_key)
            result = True
        # EC has no schedule-end callback; flush this non-blocking probe here.
        self._lookup.flush()
        if result and entry is None:
            self._entries[cache_key] = ECFileEntry(
                cache_key,
                get_ec_entry_path(self._root_dir, self._namespace_digest, cache_key),
                file_size_bytes,
            )
            self._used_bytes += file_size_bytes
            self._evict_to_capacity()
            return cache_key in self._entries
        return result

    def release_request(self, request_id: str) -> None:
        self._lookup.cleanup(request_id)

    def submit_read(
        self,
        cache_key: str,
        destinations: Sequence[memoryview],
    ) -> bool:
        """Submit a read for an indexed entry without filesystem work here."""
        if not self._accepting:
            return False

        if cache_key in self._invalid_keys:
            return False
        entry = self._entries.get(cache_key)
        if entry is None:
            raise KeyError(f"encoder-cache entry is not indexed: {cache_key}")

        job_id = self._executor.submit_read(
            cache_key,
            entry.path,
            destinations,
            entry.file_size_bytes,
        )
        if job_id is None:
            return False
        self._pending_reads[job_id] = entry
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
        self._retry_temporary_cleanup()
        if cache_key in self._invalid_keys:
            self._evict_to_capacity()
            return False
        if (
            cache_key in self._entries
            or cache_key in self._pending_write_keys
            or cache_key in self._deleting_keys
        ):
            return False
        if self._backpressure_detector is not None and not (
            self._backpressure_detector.should_store(len(sources))
        ):
            return False
        if not self._evict_to_capacity(file_size_bytes):
            return False

        entry_path = get_ec_entry_path(
            self._root_dir,
            self._namespace_digest,
            cache_key,
        )
        temporary_path = get_ec_temporary_path(
            entry_path, uuid4().hex, self._temporary_owner.owner_id
        )
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
        self._write_temporary[job_id] = temporary_path
        self._pending_write_keys.add(cache_key)
        self._reserved_write_bytes += file_size_bytes
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
        self._retry_temporary_cleanup()
        self._lookup.shutdown()
        self.drain()
        self._executor.shutdown()
        self._collect_executor_completions()
        self._temporary_owner.close(self._temporary_cleanup)
        self._shutdown = True

    def _collect_executor_completions(self) -> None:
        for completion in self._executor.poll_completed():
            if completion.operation == "read":
                self._complete_read(completion)
            elif completion.operation == "write":
                self._complete_write(completion)
            elif completion.operation == "delete":
                if completion.job_id in self._pending_temporary_deletes:
                    path = self._pending_temporary_deletes.pop(completion.job_id)
                    if completion.success:
                        self._temporary_cleanup_bytes -= self._temporary_cleanup.pop(
                            path
                        )
                    else:
                        logger.warning(
                            "Failed to clean encoder-cache temporary file %s: %s",
                            path,
                            completion.error,
                        )
                else:
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
            if current_entry is indexed_entry:
                self._invalid_keys.add(completion.cache_key)
                self._entries.move_to_end(completion.cache_key, last=False)
            self._lookup.mark_miss(
                [self._lookup.key(completion.cache_key, indexed_entry.file_size_bytes)]
            )
        self._completed_reads.append(
            ECReadResult(
                cache_key=completion.cache_key,
                success=completion.success,
                file_size_bytes=completion.file_size_bytes,
                queue_time=completion.queue_time,
                transfer_time=completion.transfer_time,
                error=completion.error,
            )
        )
        self._evict_to_capacity()

    def _complete_write(self, completion: ECFileIOCompletion) -> None:
        entry = self._pending_writes.pop(completion.job_id)
        temporary_path = self._write_temporary.pop(completion.job_id)
        self._pending_write_keys.remove(completion.cache_key)
        self._reserved_write_bytes -= entry.file_size_bytes
        if completion.success:
            if self._backpressure_detector is not None:
                self._backpressure_detector.on_store_completed(
                    completion.queue_time + completion.transfer_time,
                    completion.file_size_bytes,
                )
            self._entries[completion.cache_key] = entry
            self._used_bytes += entry.file_size_bytes
            self._lookup.mark_present(
                [self._lookup.key(completion.cache_key, entry.file_size_bytes)]
            )
        else:
            self._temporary_cleanup[temporary_path] = entry.file_size_bytes
            self._temporary_cleanup_bytes += entry.file_size_bytes
            self._retry_temporary_cleanup()
        self._completed_writes.append(
            ECWriteResult(
                cache_key=completion.cache_key,
                success=completion.success,
                file_size_bytes=completion.file_size_bytes,
                queue_time=completion.queue_time,
                transfer_time=completion.transfer_time,
                error=completion.error,
            )
        )
        self._evict_to_capacity()

    def _retry_temporary_cleanup(self) -> None:
        pending = set(self._pending_temporary_deletes.values())
        for path, size in self._temporary_cleanup.items():
            if path in pending:
                continue
            try:
                job_id = self._executor.submit_delete(path.name, path, max(1, size))
            except Exception:
                logger.warning(
                    "Failed to submit encoder-cache temporary cleanup for %s",
                    path,
                    exc_info=True,
                )
                continue
            if job_id is not None:
                self._pending_temporary_deletes[job_id] = path

    def _complete_delete(self, completion: ECFileIOCompletion) -> None:
        indexed_entry = self._pending_deletes.pop(completion.job_id)
        self._deleting_keys.remove(indexed_entry.cache_key)
        self._pending_delete_bytes -= indexed_entry.file_size_bytes
        if completion.success:
            self._invalid_keys.discard(indexed_entry.cache_key)
        else:
            self._entries[indexed_entry.cache_key] = indexed_entry
            self._entries.move_to_end(indexed_entry.cache_key, last=False)
            self._used_bytes += indexed_entry.file_size_bytes
            logger.warning(
                "Failed to delete evicted encoder-cache entry %s: %s",
                indexed_entry.path,
                completion.error,
            )

    def _recover_index(self) -> None:
        namespace_path = get_ec_namespace_path(
            self._root_dir,
            self._namespace_digest,
        )
        recovered: list[tuple[int, str, ECFileEntry]] = []
        for path in _iter_entry_files(namespace_path):
            cache_key = path.name.removesuffix(EC_FILE_SUFFIX)
            try:
                expected_path = get_ec_entry_path(
                    self._root_dir,
                    self._namespace_digest,
                    cache_key,
                )
            except ValueError:
                continue
            try:
                stat_result = path.stat()
            except FileNotFoundError:
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
            self._entries[cache_key] = entry
            self._used_bytes += entry.file_size_bytes

    def _evict_to_capacity(self, required_bytes: int = 0) -> bool:
        target_bytes = self._capacity_bytes - required_bytes
        # Pending deletes will free space, but cannot fund a write until confirmed.
        while (
            self._invalid_keys - self._deleting_keys
            or self._used_bytes + self._reserved_write_bytes > target_bytes
        ):
            needs_capacity = (
                self._used_bytes + self._reserved_write_bytes > target_bytes
            )
            victim: tuple[str, ECFileEntry] | None = None
            for cache_key, indexed_entry in self._entries.items():
                if (
                    self._read_pins[cache_key] == 0
                    and cache_key not in self._pending_write_keys
                    and (cache_key in self._invalid_keys or needs_capacity)
                ):
                    victim = (cache_key, indexed_entry)
                    break
            if victim is None:
                return False

            cache_key, indexed_entry = victim
            job_id = self._executor.submit_delete(
                cache_key,
                indexed_entry.path,
                indexed_entry.file_size_bytes,
            )
            if job_id is None:
                return False
            del self._entries[cache_key]
            self._used_bytes -= indexed_entry.file_size_bytes
            self._pending_deletes[job_id] = indexed_entry
            self._pending_delete_bytes += indexed_entry.file_size_bytes
            self._deleting_keys.add(cache_key)
            self._lookup.mark_miss(
                [self._lookup.key(cache_key, indexed_entry.file_size_bytes)]
            )
        return self.used_bytes <= target_bytes


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

    def lookup(
        self, cache_key: str, file_size_bytes: int, request_id: str
    ) -> bool | None: ...

    def release_request(self, request_id: str) -> None: ...

    def submit_read(
        self,
        cache_key: str,
        destinations: Sequence[memoryview],
    ) -> bool:
        """Queue a filesystem-to-CPU read without blocking.

        Return `False` when the tier is no longer accepting work. On
        acceptance, exactly one completion must eventually be returned by
        `poll_completed_reads`.
        """
        ...

    def submit_write(
        self,
        cache_key: str,
        sources: Sequence[memoryview],
        file_size_bytes: int,
    ) -> bool:
        """Queue a CPU-to-filesystem write without blocking.

        `file_size_bytes` is the allocated size of all complete CPU blocks.
        The sources must contain every byte of every block, including undefined
        padding in the final block. On acceptance, exactly one completion must
        eventually be returned by `poll_completed_writes`. Return
        `False` when the tier is no longer accepting work or store pressure
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
