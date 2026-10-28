# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process auxiliary-output storage."""

from __future__ import annotations

import mmap
import queue
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class BlockObject:
    """One immutable auxiliary output object."""

    key: str
    payload: bytes


class BlockObjectStoreError(RuntimeError):
    """AuxOutput storage or retrieval failed."""


class BackgroundBlockObjectStore:
    """Serialize store mutations on a background thread."""

    def __init__(self, store: BlockObjectStore, *, max_pending_batches: int) -> None:
        self._store = store
        self._queue: queue.Queue[
            tuple[list[BlockObject], tuple[str, ...], tuple[str, ...]] | None
        ] = queue.Queue(maxsize=max_pending_batches)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-aux-output-writer",
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
            raise BlockObjectStoreError(
                "auxiliary output publication failed"
            ) from self._error

    def put(
        self,
        objects: list[BlockObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        retains = tuple(retain_keys)
        releases = tuple(release_keys)
        if not objects and not retains and not releases:
            return
        if self._closed:
            raise RuntimeError("auxiliary output store is closed")
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


class BlockObjectStore:
    """Single-owner store that fails closed after an eviction."""

    _UNALLOCATED_SLOT = -1

    def __init__(
        self,
        *,
        max_bytes: int,
        object_nbytes: int,
    ) -> None:
        if object_nbytes <= 0:
            raise ValueError("auxiliary output object size must be positive")
        if max_bytes < object_nbytes:
            raise ValueError("auxiliary output store must fit at least one object")
        self.object_nbytes = object_nbytes
        self.num_slots = max_bytes // object_nbytes
        self._slots: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._references: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._arena: mmap.mmap | None = mmap.mmap(
            -1, self.num_slots * self.object_nbytes
        )

    def _evict_to_fit(self, protected: set[str]) -> None:
        excess = len(self._slots) - self.num_slots
        if excess <= 0:
            return
        victims = []
        for key in self._lru:
            if key not in protected:
                victims.append(key)
                if len(victims) == excess:
                    break
        for victim in victims:
            del self._lru[victim]
            slot = self._slots.pop(victim)
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
        objects: list[BlockObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        arena = self._arena
        if arena is None:
            raise RuntimeError("auxiliary output store is closed")
        unique = {obj.key: obj for obj in objects}
        for key in unique.keys() & self._lru.keys():
            del self._lru[key]
            slot = self._slots.pop(key)
            if slot != self._UNALLOCATED_SLOT:
                self._free_slots.append(slot)
        self._retain(retain_keys)
        terminal_order = self._release(release_keys)
        for key in unique:
            self._slots.setdefault(key, self._UNALLOCATED_SLOT)
            if key not in self._references:
                self._lru[key] = None
                self._lru.move_to_end(key)
        for key in terminal_order:
            if key in self._lru:
                self._lru.move_to_end(key)
        self._evict_to_fit(set(unique) - set(terminal_order))
        if (nbytes := len(self._slots) * self.object_nbytes) > len(arena):
            nbytes = max(nbytes, 2 * len(arena))
            logger.warning(
                "Auxiliary output store grows to %d bytes: all objects are referenced.",
                nbytes,
            )
            grown = mmap.mmap(-1, nbytes)
            grown[: len(arena)] = arena
            self._arena = arena = grown
            self.num_slots = len(grown) // self.object_nbytes
        for object_id, obj in unique.items():
            if self._slots.get(object_id) != self._UNALLOCATED_SLOT:
                continue
            slot = self._allocate_slot()
            offset = slot * self.object_nbytes
            arena[offset : offset + self.object_nbytes] = obj.payload
            self._slots[object_id] = slot

    def _retain(self, keys: Iterable[str]) -> None:
        for key in keys:
            references = self._references.get(key, 0)
            if not references:
                self._lru.pop(key, None)
            self._references[key] = references + 1

    def _release(self, keys: Iterable[str]) -> list[str]:
        terminal_order = []
        for key in keys:
            references = self._references[key] - 1
            if references:
                self._references[key] = references
                continue
            del self._references[key]
            if key in self._slots:
                self._lru[key] = None
            terminal_order.append(key)
        return terminal_order

    def get_concatenated(self, keys: list[str]) -> bytes:
        arena_obj = self._arena
        if arena_obj is None:
            raise RuntimeError("auxiliary output store is closed")
        try:
            entries = [self._slots[key] for key in keys]
        except KeyError as error:
            raise BlockObjectStoreError(
                "auxiliary output object does not exist; the object may have been "
                f"evicted (used={len(self._slots)}, "
                f"limit={self.num_slots} objects)."
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
            if key in self._lru:
                self._lru.move_to_end(key)
        return payload

    def close(self) -> None:
        arena = self._arena
        if arena is not None:
            self._arena = None
            arena.close()
