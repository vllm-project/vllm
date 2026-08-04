# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable execution-artifact objects in shared memory."""

from __future__ import annotations

import fcntl
import hashlib
import mmap
import os
import stat
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import regex as re

from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactNotFoundError,
    ArtifactObject,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SAFE_STORE_ID = re.compile(r"^[a-f0-9]{32,64}$")
_ARTIFACT_KEY_PREFIX = "vllm-artifact/"


def make_shm_store_id(instance_id: str, dp_rank: int) -> str:
    """Return the process-group-stable SHM store identity."""
    return hashlib.sha256(f"{instance_id}:{dp_rank}".encode()).hexdigest()[:32]


@dataclass
class _Entry:
    offset: int
    size: int


class LocalSharedMemoryArtifactStore:
    """Single-owner immutable object store backed by one SHM mmap arena."""

    @staticmethod
    def _object_id(key: str) -> str:
        if not key or "\x00" in key:
            raise ValueError("artifact object key must be a non-empty string")
        if key.startswith(_ARTIFACT_KEY_PREFIX):
            digest = key[len(_ARTIFACT_KEY_PREFIX) :]
            if len(digest) == 64 and digest == digest.lower():
                try:
                    int(digest, 16)
                except ValueError:
                    pass
                else:
                    return digest
        return hashlib.sha256(key.encode()).hexdigest()

    def __init__(
        self,
        root: str,
        instance_id: str,
        dp_rank: int,
        *,
        max_bytes: int,
        ttl_seconds: int,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("artifact SHM capacity must be positive")
        self.store_id = make_shm_store_id(instance_id, dp_rank)
        self.root = Path(root) / self.store_id
        self.arena_path = self.root / "arena.bin"
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._used_bytes = 0
        self._lru: OrderedDict[str, _Entry] = OrderedDict()
        self._free: list[tuple[int, int]] = []
        self._next_offset = 0
        self._arena_fd: int | None = None
        self._arena: mmap.mmap | None = None

        root_path = Path(root)
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
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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

    def _release(self, entry: _Entry) -> None:
        self._free.append((entry.offset, entry.size))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for offset, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous_offset, previous_size = merged[-1]
                merged[-1] = (previous_offset, previous_size + size)
            else:
                merged.append((offset, size))
        self._free = merged

    def _evict_to_fit(self, additional_bytes: int, protected: set[str]) -> None:
        if self._used_bytes + additional_bytes <= self.max_bytes:
            return
        protected_bytes = sum(
            self._lru[object_id].size
            for object_id in protected
            if object_id in self._lru
        )
        if protected_bytes + additional_bytes > self.max_bytes:
            raise ArtifactCapacityError(
                "artifact SHM cannot retain the requested batch: "
                f"retained={protected_bytes}, requested={additional_bytes}, "
                f"limit={self.max_bytes}"
            )
        while self._used_bytes + additional_bytes > self.max_bytes:
            victim = next(
                object_id for object_id in self._lru if object_id not in protected
            )
            entry = self._lru.pop(victim)
            self._used_bytes -= entry.size
            self._release(entry)

    def _compact(self) -> None:
        assert self._arena is not None
        cursor = 0
        for entry in self._lru.values():
            if entry.offset != cursor:
                payload = self._arena[entry.offset : entry.offset + entry.size]
                self._arena[cursor : cursor + entry.size] = payload
                entry.offset = cursor
            cursor += entry.size
        self._free.clear()
        self._next_offset = cursor

    def _allocate(self, size: int) -> int:
        for index, (offset, available) in enumerate(self._free):
            if available < size:
                continue
            if available == size:
                self._free.pop(index)
            else:
                self._free[index] = (offset + size, available - size)
            return offset
        if self._next_offset + size > self.max_bytes:
            self._compact()
        if self._next_offset + size > self.max_bytes:
            raise ArtifactCapacityError("artifact SHM arena is fragmented")
        offset = self._next_offset
        self._next_offset += size
        return offset

    def put(self, objects: list[ArtifactObject]) -> None:
        if not objects:
            return
        assert self._arena is not None
        with self._lock:
            unique: dict[str, ArtifactObject] = {}
            for obj in objects:
                unique.setdefault(self._object_id(obj.key), obj)
            protected = set(unique)
            additional_bytes = sum(
                len(obj.payload)
                for object_id, obj in unique.items()
                if object_id not in self._lru
            )
            self._evict_to_fit(additional_bytes, protected)
            for object_id, obj in unique.items():
                if object_id in self._lru:
                    self._lru.move_to_end(object_id)
                    continue
                offset = self._allocate(len(obj.payload))
                self._arena[offset : offset + len(obj.payload)] = obj.payload
                self._lru[object_id] = _Entry(offset, len(obj.payload))
                self._used_bytes += len(obj.payload)

    def get(self, keys: list[str]) -> list[bytes]:
        assert self._arena is not None
        with self._lock:
            object_ids = [self._object_id(key) for key in keys]
            try:
                entries = [self._lru[object_id] for object_id in object_ids]
            except KeyError as error:
                raise ArtifactNotFoundError(
                    "artifact object does not exist; the object may have been "
                    f"evicted from SHM (used={self._used_bytes}, "
                    f"limit={self.max_bytes}). Increase "
                    "artifact_config.max_shm_bytes when a KV cache hit requires it."
                ) from error
            payloads = [
                self._arena[entry.offset : entry.offset + entry.size]
                for entry in entries
            ]
            for object_id in object_ids:
                self._lru.move_to_end(object_id)
            return payloads

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
