# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process auxiliary-output storage."""

from __future__ import annotations

import mmap
import queue
import threading
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory


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
    """Single-owner bounded store that fails closed after an eviction.

    Recency and victims come from a KV offload `CachePolicy` ("lru" by
    default), driven the way the KV CPU offload manager drives it.
    """

    _UNALLOCATED_SLOT = -1

    def __init__(
        self,
        *,
        max_bytes: int,
        object_nbytes: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
    ) -> None:
        if object_nbytes <= 0:
            raise ValueError("auxiliary output object size must be positive")
        if max_bytes < object_nbytes:
            raise ValueError("auxiliary output store must fit at least one object")
        self.object_nbytes = object_nbytes
        self.num_slots = max_bytes // object_nbytes
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy = policy_cls(cache_capacity=self.num_slots)
        # Keys are not tied to a request; built-in policies ignore the context.
        self._req_context = ReqContext(req_id="aux-output-store")
        self._num_objects = 0
        # References may arrive before the object is stored.
        self._references: dict[OffloadKey, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._arena: mmap.mmap | None = mmap.mmap(
            -1, self.num_slots * self.object_nbytes
        )

    def _evict_to_fit(self, protected: set[OffloadKey]) -> None:
        excess = self._num_objects - self.num_slots
        if excess <= 0:
            return
        victims = self._policy.evict(excess, protected)
        if victims is None:
            raise BlockObjectStoreError(
                "auxiliary output store cannot retain the requested batch: "
                f"limit={self.num_slots} objects"
            )
        self._num_objects -= len(victims)
        for _, chunk in victims:
            if chunk.chunk_id != self._UNALLOCATED_SLOT:
                self._free_slots.append(chunk.chunk_id)

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
        unique = {_offload_key(obj.key): obj for obj in objects}
        self._retain(map(_offload_key, retain_keys))
        terminal_order = self._release(map(_offload_key, release_keys))
        inserted = []
        for key in unique:
            if self._policy.get(key) is not None:
                # Storing a resident object reuses it, as in KV offload.
                self._policy.touch([key], self._req_context)
                continue
            # ARC learns from ghost entries before insert() drops them.
            self._policy.on_store_miss([key], self._req_context)
            chunk = ChunkStatus(self._UNALLOCATED_SLOT)
            chunk.ref_cnt = self._references.get(key, 0)
            self._policy.insert(key, chunk)
            inserted.append(key)
        self._num_objects += len(inserted)
        # Released keys arrive tail first; the policy expects prefix order.
        # Releasing refreshes recency only: reuse was recorded when read.
        released = tuple(
            key for key in reversed(terminal_order) if self._policy.get(key) is not None
        )
        self._policy.on_request_finished(
            (released,), set(released), set(), self._req_context
        )
        try:
            self._evict_to_fit(unique.keys() - set(terminal_order))
        except BlockObjectStoreError:
            for key in inserted:
                self._policy.remove(key)
            self._num_objects -= len(inserted)
            raise
        for key, obj in unique.items():
            stored = self._policy.get(key)
            if stored is None or stored.chunk_id != self._UNALLOCATED_SLOT:
                continue
            slot = self._allocate_slot()
            offset = slot * self.object_nbytes
            arena[offset : offset + self.object_nbytes] = obj.payload
            stored.chunk_id = slot

    def _retain(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            references = self._references.get(key, 0)
            self._references[key] = references + 1
            chunk = self._policy.get(key)
            if chunk is not None:
                if chunk.ref_cnt == 0:
                    self._policy.mark_non_evictable(key)
                chunk.ref_cnt += 1

    def _release(self, keys: Iterable[OffloadKey]) -> list[OffloadKey]:
        terminal_order = []
        for key in keys:
            references = self._references[key] - 1
            if references:
                self._references[key] = references
            else:
                del self._references[key]
                terminal_order.append(key)
            chunk = self._policy.get(key)
            if chunk is not None:
                chunk.ref_cnt -= 1
                if chunk.ref_cnt == 0:
                    self._policy.mark_evictable(key)
        return terminal_order

    def get_concatenated(self, keys: list[str]) -> bytes:
        arena_obj = self._arena
        if arena_obj is None:
            raise RuntimeError("auxiliary output store is closed")
        offload_keys = [_offload_key(key) for key in keys]
        entries = []
        for key in offload_keys:
            chunk = self._policy.get(key)
            if chunk is None:
                raise BlockObjectStoreError(
                    "auxiliary output object does not exist; the object may have "
                    f"been evicted (used={self._num_objects}, "
                    f"limit={self.num_slots} objects). Increase "
                    "aux_output_config.max_bytes when a KV cache hit requires it."
                )
            entries.append(chunk.chunk_id)
        arena = memoryview(arena_obj)
        try:
            payload = b"".join(
                arena[slot * self.object_nbytes : (slot + 1) * self.object_nbytes]
                for slot in entries
            )
        finally:
            arena.release()
        self._policy.touch(offload_keys, self._req_context)
        return payload

    def close(self) -> None:
        arena = self._arena
        if arena is not None:
            self._arena = None
            arena.close()


def _offload_key(key: str) -> OffloadKey:
    # Routed experts form one cache group.
    return make_offload_key(key.encode(), 0)
