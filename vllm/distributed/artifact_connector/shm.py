# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable execution-artifact objects in a shared-memory filesystem."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import regex as re

from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SAFE_STORE_ID = re.compile(r"^[a-f0-9]{32,64}$")


def make_shm_store_id(instance_id: str, dp_rank: int) -> str:
    """Return the process-group-stable SHM store identity."""
    return hashlib.sha256(f"{instance_id}:{dp_rank}".encode()).hexdigest()[:32]


class LocalSharedMemoryArtifactStore:
    """Single-writer immutable artifact store in `/dev/shm`."""

    @staticmethod
    def _object_id(key: str) -> str:
        if not key or "\x00" in key:
            raise ValueError("artifact object key must be a non-empty string")
        return hashlib.sha256(key.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._path_from_object_id(self._object_id(key))

    def _path_from_object_id(self, object_id: str) -> Path:
        return self.objects_dir / f"{object_id}.bin"

    @staticmethod
    def _open_regular_file(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid():
            os.close(fd)
            raise ArtifactCorruptionError(f"unsafe artifact file: {path}")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            os.close(fd)
            raise ArtifactCorruptionError(f"invalid artifact mode: {path}")
        return fd

    def _read(self, keys: list[str]) -> list[bytes]:
        payloads: list[bytes] = []
        for key in keys:
            path = self._path(key)
            try:
                fd = self._open_regular_file(path)
            except FileNotFoundError as error:
                raise ArtifactNotFoundError(
                    f"artifact object does not exist: {key}"
                ) from error
            file_size = os.fstat(fd).st_size
            with os.fdopen(fd, "rb") as file:
                payload = file.read()
            if len(payload) != file_size:
                raise ArtifactCorruptionError(f"artifact object is truncated: {key}")
            payloads.append(payload)
        return payloads

    def __init__(
        self,
        root: str,
        instance_id: str,
        dp_rank: int,
        *,
        max_bytes: int,
        ttl_seconds: int,
    ) -> None:
        self.store_id = make_shm_store_id(instance_id, dp_rank)
        self.root = Path(root) / self.store_id
        self.objects_dir = self.root / "objects"
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._used_bytes = 0
        self._lru: OrderedDict[str, int] = OrderedDict()

        root_path = Path(root)
        self._prepare_directory(root_path)
        self._gc_stale_store_dirs(root_path)
        self._writer_lock_fd: int | None = self._acquire_writer_lock()
        try:
            self._prepare_directory(self.objects_dir)
            self._cleanup_orphan_partials()
            self._load_lru()
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
    def _stale_store_entries(
        store_root: Path,
    ) -> tuple[list[Path], list[Path], float] | None:
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
            if entry.name == ".writer.lock":
                if not stat.S_ISREG(entry_stat.st_mode):
                    return None
                continue
            if entry.name != "objects" or not stat.S_ISDIR(entry_stat.st_mode):
                return None
            if entry_stat.st_uid != os.getuid():
                return None
            for child in os.scandir(entry.path):
                child_stat = child.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(child_stat.st_mode)
                    or child_stat.st_uid != os.getuid()
                ):
                    return None
                newest_mtime = max(newest_mtime, child_stat.st_mtime)
                files.append(Path(child.path))
        return files, [store_root / "objects"], newest_mtime

    def _gc_stale_store_dirs(self, root: Path) -> None:
        now = time.time()
        for entry in os.scandir(root):
            if entry.name == self.store_id or not _SAFE_STORE_ID.fullmatch(entry.name):
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                store_root = Path(entry.path)
                scanned = self._stale_store_entries(store_root)
                cutoff = now - self.ttl_seconds
                if scanned is None or scanned[2] >= cutoff:
                    continue
                lock_path = store_root / ".writer.lock"
                lock_fd = self._open_regular_file(lock_path)
                try:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    scanned = self._stale_store_entries(store_root)
                    if scanned is None or scanned[2] >= cutoff:
                        continue
                    files, directories, _ = scanned
                    for path in files:
                        path.unlink(missing_ok=True)
                    for path in directories:
                        path.rmdir()
                    lock_path.unlink(missing_ok=True)
                    store_root.rmdir()
                    logger.info("Removed expired artifact SHM store %s", store_root)
                finally:
                    os.close(lock_fd)
            except (ArtifactCorruptionError, FileNotFoundError, OSError, ValueError):
                logger.debug(
                    "Could not collect stale artifact SHM store %s",
                    entry.path,
                    exc_info=True,
                )

    def _load_lru(self) -> None:
        entries: list[tuple[int, str, int]] = []
        for path in self.objects_dir.glob("*.bin"):
            path_stat = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_uid != os.getuid():
                raise ArtifactCorruptionError(f"unsafe artifact file: {path}")
            entries.append((path_stat.st_mtime_ns, path.stem, path_stat.st_size))
        entries.sort()
        self._lru = OrderedDict((object_id, size) for _, object_id, size in entries)
        self._used_bytes = sum(self._lru.values())

    def _touch(self, object_id: str) -> None:
        if object_id in self._lru:
            self._lru.move_to_end(object_id)

    def _evict_to_fit(
        self,
        additional_bytes: int,
        protected: set[str],
    ) -> None:
        protected_bytes = sum(self._lru.get(object_id, 0) for object_id in protected)
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
            size = self._lru.pop(victim)
            (self.objects_dir / f"{victim}.bin").unlink(missing_ok=True)
            self._used_bytes -= size

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> bool:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        fd: int | None = None
        try:
            fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.posix_fallocate(fd, 0, len(payload))
            except OSError as error:
                if error.errno == errno.ENOSPC:
                    raise ArtifactCapacityError(
                        f"artifact SHM could not reserve {len(payload)} bytes"
                    ) from error
                if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
                os.ftruncate(fd, len(payload))
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(fd, view[offset:])
                if written <= 0:
                    raise OSError("short write while publishing artifact object")
                offset += written
            view.release()
            os.close(fd)
            fd = None
            try:
                os.link(temporary, path, follow_symlinks=False)
                return True
            except FileExistsError:
                return False
        finally:
            if fd is not None:
                os.close(fd)
            temporary.unlink(missing_ok=True)

    def _put_one(self, object_id: str, obj: ArtifactObject) -> None:
        if object_id in self._lru:
            self._touch(object_id)
            return
        path = self._path_from_object_id(object_id)
        created = self._write_immutable(path, obj.payload)
        if created:
            self._used_bytes += len(obj.payload)
            self._lru[object_id] = len(obj.payload)

    def put(self, objects: list[ArtifactObject]) -> None:
        if not objects:
            return
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
                self._put_one(object_id, obj)

    def get(self, keys: list[str]) -> list[bytes]:
        with self._lock:
            try:
                payloads = self._read(keys)
            except ArtifactNotFoundError as error:
                raise ArtifactNotFoundError(
                    f"{error}; the object may have been evicted from SHM "
                    f"(used={self._used_bytes}, limit={self.max_bytes}). "
                    "Increase artifact_config.max_shm_bytes when a KV cache hit "
                    "requires it."
                ) from error
            for key in keys:
                self._touch(self._object_id(key))
            return payloads

    def _cleanup_orphan_partials(self) -> None:
        # The exclusive writer lock proves no partial file is still active.
        for path in self.objects_dir.glob(".*.partial"):
            path.unlink(missing_ok=True)

    def close(self) -> None:
        fd = getattr(self, "_writer_lock_fd", None)
        if fd is not None:
            self._writer_lock_fd = None
            os.close(fd)
