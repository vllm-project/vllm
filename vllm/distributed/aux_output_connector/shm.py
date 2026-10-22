# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memory-mapped auxiliary-output block storage."""

import mmap
from collections import OrderedDict
from collections.abc import Iterable

from vllm.distributed.aux_output_connector.store import (
    BlockObject,
    BlockObjectStoreError,
)


class ShmBlockObjectStore:
    """Single-owner bounded store that fails closed after an eviction."""

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
        if len(victims) != excess:
            raise BlockObjectStoreError(
                "auxiliary output store cannot retain the requested batch: "
                f"limit={self.num_slots} objects"
            )
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
        try:
            self._evict_to_fit(set(unique) - set(terminal_order))
        except BlockObjectStoreError:
            for key in unique:
                if self._slots.get(key) == self._UNALLOCATED_SLOT:
                    del self._slots[key]
                    self._lru.pop(key, None)
            raise
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
                f"limit={self.num_slots} objects). Increase "
                "aux_output_config.max_bytes when a KV cache hit requires it."
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
