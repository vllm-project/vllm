# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process execution-artifact storage."""

from __future__ import annotations

import mmap
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol


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


class InProcessArtifactStore:
    """Bounded immutable store that fails closed after an eviction."""

    def __init__(
        self,
        *,
        max_bytes: int,
        object_nbytes: int,
    ) -> None:
        if object_nbytes <= 0:
            raise ValueError("artifact object size must be positive")
        if max_bytes < object_nbytes:
            raise ValueError("artifact store must fit at least one object")
        self.object_nbytes = object_nbytes
        self.num_slots = max_bytes // object_nbytes
        self.max_bytes = self.num_slots * object_nbytes
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._arena: mmap.mmap | None = mmap.mmap(-1, self.max_bytes)

    def _evict_to_fit(self, additional_slots: int, protected: set[str]) -> None:
        if len(self._lru) + additional_slots <= self.num_slots:
            return
        protected_slots = sum(key in self._lru for key in protected)
        if protected_slots + additional_slots > self.num_slots:
            raise ArtifactCapacityError(
                "artifact store cannot retain the requested batch: "
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
                f"evicted (used={len(self._lru)}, "
                f"limit={self.num_slots} objects). Increase "
                "artifact_config.max_bytes when a KV cache hit requires it."
            ) from error

    def close(self) -> None:
        arena = self._arena
        if arena is not None:
            self._arena = None
            arena.close()
