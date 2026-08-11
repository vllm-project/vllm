# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable execution-artifact objects in shared memory."""

from __future__ import annotations

import fcntl
import hashlib
import mmap
import os
import queue
import stat
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import regex as re

from vllm.logger import init_logger

logger = init_logger(__name__)

_SAFE_STORE_ID = re.compile(r"^[a-f0-9]{32,64}$")


@dataclass(frozen=True)
class ArtifactObject:
    """One immutable artifact object."""

    key: str
    payload: bytes


class ArtifactStoreError(RuntimeError):
    """Base class for artifact-store failures."""


class ArtifactCapacityError(ArtifactStoreError):
    """The artifact store cannot retain another object."""


class ArtifactCorruptionError(ArtifactStoreError):
    """An artifact object failed validation."""


class ArtifactNotFoundError(ArtifactStoreError):
    """A requested artifact object is not present."""


class ArtifactReader(Protocol):
    """Byte-object reads used to materialize artifacts."""

    def get_concatenated(
        self,
        keys: list[str],
        *,
        object_size: int,
    ) -> bytes: ...

    def close(self) -> None: ...


class ArtifactStore(ArtifactReader, Protocol):
    """Artifact reader that can publish immutable objects."""

    def put(self, objects: list[ArtifactObject]) -> None: ...


class BackgroundArtifactStore:
    """Publish objects in a background thread while preserving read order."""

    def __init__(self, store: ArtifactStore, *, max_pending_batches: int) -> None:
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        self._store = store
        self._queue: queue.Queue[list[ArtifactObject] | None] = queue.Queue(
            maxsize=max_pending_batches
        )
        self._error: BaseException | None = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-artifact-writer",
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            objects = self._queue.get()
            try:
                if objects is None:
                    return
                if self._error is None:
                    self._store.put(objects)
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise ArtifactStoreError("artifact publication failed") from self._error

    def put(self, objects: list[ArtifactObject]) -> None:
        if not objects:
            return
        with self._state_lock:
            if self._closed:
                raise RuntimeError("artifact store is closed")
            self._raise_if_failed()
            self._queue.put(objects)
            self._raise_if_failed()

    def get_concatenated(
        self,
        keys: list[str],
        *,
        object_size: int,
    ) -> bytes:
        self._queue.join()
        self._raise_if_failed()
        return self._store.get_concatenated(keys, object_size=object_size)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.join()
            self._queue.put(None)
        self._thread.join()
        try:
            self._raise_if_failed()
        finally:
            self._store.close()


def make_shm_store_id(instance_id: str, dp_rank: int) -> str:
    """Return the process-group-stable SHM store identity."""
    return hashlib.sha256(f"{instance_id}:{dp_rank}".encode()).hexdigest()[:32]


class LocalSharedMemoryArtifactStore:
    """Bounded immutable SHM store that fails closed after an eviction."""

    def __init__(
        self,
        root: str,
        instance_id: str,
        dp_rank: int,
        *,
        max_bytes: int,
        object_nbytes: int,
        ttl_seconds: int,
    ) -> None:
        if object_nbytes <= 0:
            raise ValueError("artifact object size must be positive")
        if max_bytes < object_nbytes:
            raise ValueError("artifact SHM must fit at least one object")
        self.store_id = make_shm_store_id(instance_id, dp_rank)
        self.root = Path(root) / self.store_id
        self.arena_path = self.root / "arena.bin"
        self.object_nbytes = object_nbytes
        self.num_slots = max_bytes // object_nbytes
        self.max_bytes = self.num_slots * object_nbytes
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._arena_fd: int | None = None
        self._arena: mmap.mmap | None = None

        root_path = Path(root)
        if root_path.resolve() == Path("/dev/shm"):
            raise ValueError("artifact SHM root must be below /dev/shm")
        self._prepare_directory(root_path)
        self._gc_stale_store_dirs(root_path)
        self._writer_lock_fd: int | None = self._acquire_writer_lock()
        try:
            self._create_arena()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _prepare_directory(path: Path) -> None:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError(f"artifact path is not a directory: {path}")
        if path_stat.st_uid != os.getuid():
            raise ValueError(f"artifact directory is not owned by this user: {path}")
        path.chmod(0o700)

    def _acquire_writer_lock(self) -> int:
        path = self.root / ".writer.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        while True:
            self._prepare_directory(self.root)
            try:
                fd = os.open(path, flags, 0o600)
            except FileNotFoundError:
                continue
            try:
                file_stat = os.fstat(fd)
                if (
                    not stat.S_ISREG(file_stat.st_mode)
                    or file_stat.st_uid != os.getuid()
                ):
                    raise ValueError(f"unsafe artifact writer lock: {path}")
                os.fchmod(fd, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ArtifactStoreError(
                        f"artifact SHM store already has a live writer: {self.root}"
                    ) from error
                try:
                    path_stat = path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    os.close(fd)
                    continue
                if (path_stat.st_dev, path_stat.st_ino) != (
                    file_stat.st_dev,
                    file_stat.st_ino,
                ):
                    os.close(fd)
                    continue
            except Exception:
                os.close(fd)
                raise
            return fd

    @staticmethod
    def _stale_store_files(
        store_root: Path,
    ) -> tuple[list[Path], float] | None:
        try:
            root_stat = store_root.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
            return None

        files: list[Path] = []
        newest_mtime = root_stat.st_mtime
        try:
            entries = list(os.scandir(store_root))
        except FileNotFoundError:
            return None
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            newest_mtime = max(newest_mtime, entry_stat.st_mtime)
            if entry.name not in (".writer.lock", "arena.bin"):
                return None
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_uid != os.getuid():
                return None
            if entry.name == "arena.bin":
                files.append(Path(entry.path))
        return files, newest_mtime

    @staticmethod
    def _open_lock_file(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid():
            os.close(fd)
            raise ValueError(f"unsafe artifact writer lock: {path}")
        return fd

    def _gc_stale_store_dirs(self, root: Path) -> None:
        now = time.time()
        for entry in os.scandir(root):
            if entry.name == self.store_id or not _SAFE_STORE_ID.fullmatch(entry.name):
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                store_root = Path(entry.path)
                scanned = self._stale_store_files(store_root)
                cutoff = now - self.ttl_seconds
                if scanned is None or scanned[1] >= cutoff:
                    continue
                lock_path = store_root / ".writer.lock"
                lock_fd = self._open_lock_file(lock_path)
                try:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    scanned = self._stale_store_files(store_root)
                    if scanned is None or scanned[1] >= cutoff:
                        continue
                    files, _ = scanned
                    for path in files:
                        path.unlink(missing_ok=True)
                    lock_path.unlink(missing_ok=True)
                    store_root.rmdir()
                    logger.info("Removed expired artifact SHM store %s", store_root)
                finally:
                    os.close(lock_fd)
            except (FileNotFoundError, OSError, ValueError):
                logger.debug(
                    "Could not collect stale artifact SHM store %s",
                    entry.path,
                    exc_info=True,
                )

    def _create_arena(self) -> None:
        for entry in os.scandir(self.root):
            if entry.name == ".writer.lock":
                continue
            entry_stat = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_uid != os.getuid():
                raise ValueError(f"unsafe artifact file: {entry.path}")
            Path(entry.path).unlink()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.arena_path, flags, 0o600)
        try:
            os.ftruncate(fd, self.max_bytes)
            self._arena = mmap.mmap(fd, self.max_bytes, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(fd)
            self.arena_path.unlink(missing_ok=True)
            raise
        self._arena_fd = fd

    def _evict_to_fit(self, additional_slots: int, protected: set[str]) -> None:
        if len(self._lru) + additional_slots <= self.num_slots:
            return
        protected_slots = sum(key in self._lru for key in protected)
        if protected_slots + additional_slots > self.num_slots:
            raise ArtifactCapacityError(
                "artifact SHM cannot retain the requested batch: "
                f"retained={protected_slots}, requested={additional_slots}, "
                f"limit={self.num_slots} objects"
            )
        while len(self._lru) + additional_slots > self.num_slots:
            victim = next(
                object_id for object_id in self._lru if object_id not in protected
            )
            self._free_slots.append(self._lru.pop(victim))

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def put(self, objects: list[ArtifactObject]) -> None:
        if not objects:
            return
        assert self._arena is not None
        with self._lock:
            unique: dict[str, ArtifactObject] = {}
            for obj in objects:
                if not obj.key or "\x00" in obj.key:
                    raise ValueError("artifact object key must be a non-empty string")
                if len(obj.payload) != self.object_nbytes:
                    raise ArtifactCorruptionError("artifact object has an invalid size")
                unique.setdefault(obj.key, obj)
            protected = set(unique)
            additional_slots = sum(object_id not in self._lru for object_id in unique)
            self._evict_to_fit(additional_slots, protected)
            for object_id, obj in unique.items():
                if object_id in self._lru:
                    self._lru.move_to_end(object_id)
                    continue
                slot = self._allocate_slot()
                offset = slot * self.object_nbytes
                self._arena[offset : offset + self.object_nbytes] = obj.payload
                self._lru[object_id] = slot

    def get_concatenated(
        self,
        keys: list[str],
        *,
        object_size: int,
    ) -> bytes:
        assert self._arena is not None
        if object_size != self.object_nbytes:
            raise ArtifactCorruptionError("artifact object has an invalid size")
        with self._lock:
            entries = self._lookup(keys)
            arena = memoryview(self._arena)
            try:
                payload = b"".join(
                    arena[slot * self.object_nbytes : (slot + 1) * self.object_nbytes]
                    for slot in entries
                )
            finally:
                arena.release()
            for key in keys:
                self._lru.move_to_end(key)
            return payload

    def _lookup(self, keys: list[str]) -> list[int]:
        try:
            return [self._lru[key] for key in keys]
        except KeyError as error:
            raise ArtifactNotFoundError(
                "artifact object does not exist; the object may have been "
                f"evicted from SHM (used={len(self._lru)}, "
                f"limit={self.num_slots} objects). Increase "
                "artifact_config.max_shm_bytes when a KV cache hit requires it."
            ) from error

    def close(self) -> None:
        arena = self._arena
        if arena is not None:
            self._arena = None
            arena.close()
        arena_fd = self._arena_fd
        if arena_fd is not None:
            self._arena_fd = None
            os.close(arena_fd)
            with suppress(FileNotFoundError):
                os.utime(self.arena_path)
        lock_fd = self._writer_lock_fd
        if lock_fd is not None:
            self._writer_lock_fd = None
            os.close(lock_fd)
