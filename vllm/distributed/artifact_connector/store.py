# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process execution-artifact storage."""

from __future__ import annotations

import mmap
import queue
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactObject:
    """One immutable artifact object."""

    key: str
    payload: bytes


class ArtifactStoreError(RuntimeError):
    """Artifact storage or retrieval failed."""


class BackgroundArtifactStore:
    """Serialize store mutations on a background thread."""

    def __init__(
        self, store: InProcessArtifactStore, *, max_pending_batches: int
    ) -> None:
        self._store = store
        self._queue: queue.Queue[
            tuple[list[ArtifactObject], tuple[str, ...], tuple[str, ...]] | None
        ] = queue.Queue(maxsize=max_pending_batches)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-artifact-writer",
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            update = self._queue.get()
            try:
                if update is None:
                    return
                if self._error is None:
                    objects, retain_keys, release_keys = update
                    self._store.put(
                        objects,
                        retain_keys=retain_keys,
                        release_keys=release_keys,
                    )
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise ArtifactStoreError("artifact publication failed") from self._error

    def put(
        self,
        objects: list[ArtifactObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        retains = tuple(retain_keys)
        releases = tuple(release_keys)
        if not objects and not retains and not releases:
            return
        if self._closed:
            raise RuntimeError("artifact store is closed")
        self._queue.put((objects, retains, releases))
        self._raise_if_failed()

    def get_concatenated(self, keys: list[str]) -> bytes:
        self._queue.join()
        self._raise_if_failed()
        return self._store.get_concatenated(keys)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        self._store.close()
        self._raise_if_failed()


class InProcessArtifactStore:
    """Single-owner bounded store that fails closed after an eviction."""

    _UNALLOCATED_SLOT = -1

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
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._references: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._arena: mmap.mmap | None = mmap.mmap(
            -1, self.num_slots * self.object_nbytes
        )

    def _evict_to_fit(self, protected: set[str]) -> None:
        excess = len(self._lru) - self.num_slots
        if excess <= 0:
            return
        victims = []
        for key in self._lru:
            if key not in self._references and key not in protected:
                victims.append(key)
                if len(victims) == excess:
                    break
        if len(victims) != excess:
            raise ArtifactStoreError(
                "artifact store cannot retain the requested batch: "
                f"limit={self.num_slots} objects"
            )
        for victim in victims:
            slot = self._lru.pop(victim)
            if slot != self._UNALLOCATED_SLOT:
                self._free_slots.append(slot)

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def put(
        self,
        objects: list[ArtifactObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        arena = self._arena
        if arena is None:
            raise RuntimeError("artifact store is closed")
        unique = {obj.key: obj for obj in objects}
        self._retain(retain_keys)
        terminal_order = self._release(release_keys)
        for key in unique:
            self._lru.setdefault(key, self._UNALLOCATED_SLOT)
            self._lru.move_to_end(key)
        for key in terminal_order:
            if key in self._lru:
                self._lru.move_to_end(key)
        try:
            self._evict_to_fit(set(unique) - set(terminal_order))
        except ArtifactStoreError:
            for key in unique:
                if self._lru.get(key) == self._UNALLOCATED_SLOT:
                    del self._lru[key]
            raise
        for object_id, obj in unique.items():
            slot = self._lru.get(object_id)
            if slot is None:
                continue
            if slot != self._UNALLOCATED_SLOT:
                continue
            slot = self._allocate_slot()
            offset = slot * self.object_nbytes
            arena[offset : offset + self.object_nbytes] = obj.payload
            self._lru[object_id] = slot

    def _retain(self, keys: Iterable[str]) -> None:
        for key in keys:
            references = self._references.get(key, 0)
            self._references[key] = references + 1

    def _release(self, keys: Iterable[str]) -> list[str]:
        terminal_order = []
        for key in keys:
            references = self._references[key] - 1
            if references:
                self._references[key] = references
                continue
            del self._references[key]
            terminal_order.append(key)
        return terminal_order

    def get_concatenated(self, keys: list[str]) -> bytes:
        arena_obj = self._arena
        if arena_obj is None:
            raise RuntimeError("artifact store is closed")
        try:
            entries = [self._lru[key] for key in keys]
        except KeyError as error:
            raise ArtifactStoreError(
                "artifact object does not exist; the object may have been "
                f"evicted (used={len(self._lru)}, "
                f"limit={self.num_slots} objects). Increase "
                "artifact_config.max_bytes when a KV cache hit requires it."
            ) from error
        arena = memoryview(arena_obj)
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

    def close(self) -> None:
        arena = self._arena
        if arena is not None:
            self._arena = None
            arena.close()
