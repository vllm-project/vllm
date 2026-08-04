# SPDX-License-Identifier: Apache-2.0
"""
SerdeL2AdapterWrapper: wraps an inner L2 adapter with a SerdeProcessor
so controllers see a plain ``L2AdapterInterface`` while data is
transparently serialized on store and deserialized on load.

Threading: the wrapper owns an internal poll thread that reacts to
inner-adapter and serde event notifiers and chains

    store : caller → serialize → inner.store            → signal store_efd
    load  : caller → inner.load → deserialize           → signal load_efd

Lookup / unlock / delete / eviction pass straight through to the inner
adapter (no serde transform involved).

Temp buffer lifecycle: temp byte buffers come from the injected
``L1Manager`` so the extra memory shows up in L1 accounting just like
the non-serde path's temporary KV buffers. For a store, the temp holds
serialized bytes; for a load, the temp catches the bytes L2 reads
before deserialize copies them into the caller-provided KV buffer.

Failure policy: all-or-nothing per submit. A partial temp-alloc
failure fails the whole task (``success=False`` for store, all-zeros
bitmap for load). This preserves the coarse-grained success semantic
of ``L2AdapterInterface`` and means the caller's lock / lifecycle
invariants don't need to change.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import enum
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import KeyListPage, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.serde import (
    SerdeProcessor,
    SerdeTaskId,
    make_temp_key,
    serialized_layout_desc,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import consume_fd, create_event_notifier

logger = init_logger(__name__)

_POLL_TIMEOUT_MS = 500


class _StorePhase(enum.Enum):
    SERIALIZE = enum.auto()
    INNER_STORE = enum.auto()


@dataclass
class _StoreTaskState:
    wrapped_id: L2TaskId
    keys: list[ObjectKey]
    temp_keys: list[ObjectKey]
    temp_objs: list[MemoryObj]
    phase: _StorePhase
    """SERIALIZE while temps are write-locked; INNER_STORE after the
    serialize→store transition. Only read on shutdown to pick the right
    lock-release path; assignment is done under ``self._lock``."""


@dataclass
class _LoadTaskState:
    wrapped_id: L2TaskId
    keys: list[ObjectKey]
    dst_objs: list[MemoryObj]
    temp_keys: list[ObjectKey]
    temp_objs: list[MemoryObj]
    load_bitmap: Bitmap = field(default_factory=lambda: Bitmap(0))
    """Inner adapter's per-key load bitmap; populated in
    ``_drain_inner_load`` before the task transitions to the deserialize
    stage. ``Bitmap(0)`` means "not populated yet" — by the time
    ``_drain_deserialize`` reads it, this placeholder has been
    overwritten with the real bitmap."""


class SerdeL2AdapterWrapper(L2AdapterInterface):
    """L2 adapter that adds transparent serde on top of an inner adapter.

    Args:
        inner: The wrapped L2 adapter doing the actual storage.
        serde: The SerdeProcessor used to (de)serialize KV data.
        l1_manager: L1 manager used to allocate temp byte buffers.
    """

    def __init__(
        self,
        inner: L2AdapterInterface,
        serde: SerdeProcessor,
        l1_manager: L1Manager,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._serde = serde
        self._l1_manager = l1_manager

        # Our own notifiers for store/load completion. Lookup passes the
        # inner adapter's fd straight through (no chaining needed there).
        self._store_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task-id space separate from inner's. Reverse maps let the
        # internal thread pair inner / serde completions back to our
        # wrapped task id.
        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._store_tasks: dict[L2TaskId, _StoreTaskState] = {}
        self._load_tasks: dict[L2TaskId, _LoadTaskState] = {}
        self._serde_to_store: dict[SerdeTaskId, L2TaskId] = {}
        self._inner_to_store: dict[L2TaskId, L2TaskId] = {}
        self._inner_to_load: dict[L2TaskId, L2TaskId] = {}
        self._serde_to_load: dict[SerdeTaskId, L2TaskId] = {}

        # User-visible completion queues (drained by controller polls).
        self._completed_store: dict[L2TaskId, L2StoreResult] = {}
        self._completed_load: dict[L2TaskId, Bitmap] = {}

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="serde-l2-wrapper",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Event fds
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        # Lookup doesn't touch serde; passing through the inner adapter's
        # fd avoids a useless thread-hop per lookup.
        return self._inner.get_lookup_and_lock_event_fd()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a wrapped store (serialize → inner.store).

        All-or-nothing: if temp alloc fails for any key or serialize
        submission raises, the whole task is marked failed and the
        caller's next ``pop_completed_store_tasks`` call sees it.
        """
        with self._lock:
            wrapped_id = self._next_task_id
            self._next_task_id += 1

        temp_keys, temp_objs = self._alloc_temp_buffers(keys, objects)
        if temp_objs is None:
            logger.warning(
                "Serde wrapper: temp alloc failed for store task %d",
                wrapped_id,
            )
            self._finalize_store(wrapped_id, success=False)
            return wrapped_id

        # Hold the wrapper lock across submit + reverse-map registration
        # so the internal drain thread cannot observe a half-state where
        # the serde already signaled completion but ``_serde_to_store``
        # has no entry — which would leave the wrapped task hanging.
        state = _StoreTaskState(
            wrapped_id=wrapped_id,
            keys=list(keys),
            temp_keys=temp_keys,
            temp_objs=temp_objs,
            phase=_StorePhase.SERIALIZE,
        )
        try:
            with self._lock:
                self._store_tasks[wrapped_id] = state
                serde_task_id = self._serde.submit_serialize(
                    objects, temp_objs, state.keys
                )
                self._serde_to_store[serde_task_id] = wrapped_id
        except Exception:
            logger.exception(
                "Serde wrapper: submit_serialize raised for store task %d",
                wrapped_id,
            )
            with self._lock:
                self._store_tasks.pop(wrapped_id, None)
            self._release_write_temps(temp_keys)
            self._finalize_store(wrapped_id, success=False)
            return wrapped_id
        return wrapped_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            result = self._completed_store
            self._completed_store = {}
        return result

    # ------------------------------------------------------------------
    # Lookup / unlock (pure delegation)
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        return self._inner.submit_lookup_and_lock_task(keys, layout_desc)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        return self._inner.query_lookup_and_lock_result(task_id)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        self._inner.submit_unlock(keys)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a wrapped load (inner.load → deserialize).

        All-or-nothing: if temp alloc or inner submission fails, the
        caller gets an all-zeros bitmap on next ``query_load_result``.
        """
        with self._lock:
            wrapped_id = self._next_task_id
            self._next_task_id += 1

        temp_keys, temp_objs = self._alloc_temp_buffers(keys, objects)
        if temp_objs is None:
            logger.warning(
                "Serde wrapper: temp alloc failed for load task %d",
                wrapped_id,
            )
            self._finalize_load(wrapped_id, Bitmap(len(keys)))
            return wrapped_id

        # Hold the wrapper lock across submit + reverse-map registration
        # so the internal drain thread cannot observe a half-state where
        # the inner already signaled completion but ``_inner_to_load``
        # has no entry.
        state = _LoadTaskState(
            wrapped_id=wrapped_id,
            keys=list(keys),
            dst_objs=list(objects),
            temp_keys=temp_keys,
            temp_objs=temp_objs,
        )
        try:
            with self._lock:
                self._load_tasks[wrapped_id] = state
                inner_task_id = self._inner.submit_load_task(keys, temp_objs)
                self._inner_to_load[inner_task_id] = wrapped_id
        except Exception:
            logger.exception(
                "Serde wrapper: inner.submit_load_task raised for task %d",
                wrapped_id,
            )
            with self._lock:
                self._load_tasks.pop(wrapped_id, None)
            self._release_write_temps(temp_keys)
            self._finalize_load(wrapped_id, Bitmap(len(keys)))
            return wrapped_id
        return wrapped_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load.pop(task_id, None)

    # ------------------------------------------------------------------
    # Eviction / metadata / listeners (delegate to inner)
    # ------------------------------------------------------------------

    @property
    def inner_adapter(self) -> L2AdapterInterface:
        """Return the wrapped L2 adapter."""
        return self._inner

    @property
    def supports_global_eviction(self) -> bool:
        return self._inner.supports_global_eviction

    def get_usage(self) -> AdapterUsage:
        return self._inner.get_usage()

    def delete(self, keys: list[ObjectKey]) -> None:
        self._inner.delete(keys)

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> KeyListPage:
        return self._inner.list_l2_keys(
            model_name=model_name,
            page_size=page_size,
            cursor=cursor,
        )

    def register_listener(self, listener: L2AdapterListener) -> None:
        # Listeners track what's actually stored — which is inner's job.
        self._inner.register_listener(listener)

    def set_backend_name(self, name: str) -> None:
        """Forward the backend name to the inner adapter (which owns the
        listener-notify funnel that tags cache events)."""
        self._inner.set_backend_name(name)

    def report_status(self) -> dict:
        inner_status = self._inner.report_status()
        return {**inner_status, "serde_wrapped": True}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._stop_flag.set()
        self._thread.join()

        # Shut down the inner adapter and serde processor BEFORE
        # releasing temp buffers. Both ``close()`` calls block until
        # their in-flight reads / writes against the temp MemoryObjs
        # finish; releasing the temps first would let L1 reclaim memory
        # that the inner adapter or serde thread pool is still touching
        # (use-after-free).
        self._inner.close()
        self._serde.close()

        # Now safe to release leftover temp buffers. Store tasks in
        # SERIALIZE phase hold write locks on their temps; tasks in
        # INNER_STORE phase hold read locks (transitioned after
        # serialize). Load tasks always hold write locks.
        with self._lock:
            write_locked: list[ObjectKey] = []
            read_locked: list[ObjectKey] = []
            for s in self._store_tasks.values():
                if s.phase is _StorePhase.SERIALIZE:
                    write_locked.extend(s.temp_keys)
                else:
                    read_locked.extend(s.temp_keys)
            for load in self._load_tasks.values():
                write_locked.extend(load.temp_keys)
            self._store_tasks.clear()
            self._load_tasks.clear()
            self._serde_to_store.clear()
            self._inner_to_store.clear()
            self._inner_to_load.clear()
            self._serde_to_load.clear()

        if write_locked:
            try:
                self._l1_manager.finish_write(write_locked)
                self._l1_manager.delete(write_locked)
            except Exception:
                logger.exception(
                    "Serde wrapper: error releasing write-locked leftover temps"
                )
        if read_locked:
            try:
                self._l1_manager.finish_read(read_locked)
            except Exception:
                logger.exception(
                    "Serde wrapper: error releasing read-locked leftover temps"
                )

        self._store_efd.close()
        self._load_efd.close()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        poller = select.poll()
        inner_store_efd = self._inner.get_store_event_fd()
        inner_load_efd = self._inner.get_load_event_fd()
        serialize_efd = self._serde.get_serialize_event_fd()
        deserialize_efd = self._serde.get_deserialize_event_fd()

        poller.register(inner_store_efd, select.POLLIN)
        poller.register(inner_load_efd, select.POLLIN)
        poller.register(serialize_efd, select.POLLIN)
        poller.register(deserialize_efd, select.POLLIN)

        while not self._stop_flag.is_set():
            ready = poller.poll(_POLL_TIMEOUT_MS)
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue
                try:
                    consume_fd(fd)
                except OSError:
                    pass
                try:
                    if fd == serialize_efd:
                        self._drain_serialize()
                    elif fd == inner_store_efd:
                        self._drain_inner_store()
                    elif fd == inner_load_efd:
                        self._drain_inner_load()
                    elif fd == deserialize_efd:
                        self._drain_deserialize()
                except Exception:
                    logger.exception("Serde wrapper: internal loop error on fd %d", fd)

    def _drain_serialize(self) -> None:
        """Poll pending serialize tasks; on success submit inner store."""
        with self._lock:
            pending = list(self._serde_to_store.keys())
        for serde_id in pending:
            result = self._serde.query_serialize_result(serde_id)
            if result is None:
                continue
            with self._lock:
                wrapped_id = self._serde_to_store.pop(serde_id, None)
                state = (
                    self._store_tasks.get(wrapped_id)
                    if wrapped_id is not None
                    else None
                )
            if wrapped_id is None or state is None:
                continue

            if not result:
                self._release_write_temps(state.temp_keys)
                self._finalize_store(wrapped_id, success=False)
                continue

            # Serialize succeeded — transition temps write → read so inner
            # can safely read them during the store.
            self._l1_manager.finish_write_and_reserve_read(state.temp_keys)
            try:
                inner_id = self._inner.submit_store_task(state.keys, state.temp_objs)
            except Exception:
                logger.exception(
                    "Serde wrapper: inner.submit_store_task raised for task %d",
                    wrapped_id,
                )
                # Temps are now read-locked and temporary — finish_read
                # is enough; the entries auto-delete.
                self._l1_manager.finish_read(state.temp_keys)
                self._finalize_store(wrapped_id, success=False)
                continue

            # Phase flip and reverse-map insert happen under the same
            # lock so ``close()``'s cleanup can't observe a half-way
            # transition.
            with self._lock:
                state.phase = _StorePhase.INNER_STORE
                self._inner_to_store[inner_id] = wrapped_id

    def _drain_inner_store(self) -> None:
        """Drain inner store completions; release temp read locks (auto-
        delete) and finalize the wrapped tasks."""
        completed = self._inner.pop_completed_store_tasks()
        for inner_id, result in completed.items():
            with self._lock:
                wrapped_id = self._inner_to_store.pop(inner_id, None)
                state = (
                    self._store_tasks.get(wrapped_id)
                    if wrapped_id is not None
                    else None
                )
            if wrapped_id is None:
                logger.warning(
                    "Serde wrapper: inner store task %d has no wrapped id",
                    inner_id,
                )
                continue
            if state is not None:
                self._l1_manager.finish_read(state.temp_keys)
            self._finalize_store(
                wrapped_id, result.is_successful(), result.bytes_transferred()
            )

    def _drain_inner_load(self) -> None:
        """Drain inner load completions; on per-key success submit
        deserialize, otherwise fail the keys immediately."""
        with self._lock:
            pending = list(self._inner_to_load.keys())
        for inner_id in pending:
            bitmap = self._inner.query_load_result(inner_id)
            if bitmap is None:
                continue
            with self._lock:
                wrapped_id = self._inner_to_load.pop(inner_id, None)
                state = (
                    self._load_tasks.get(wrapped_id) if wrapped_id is not None else None
                )
            if wrapped_id is None or state is None:
                continue

            src_objs: list[MemoryObj] = []
            dst_objs: list[MemoryObj] = []
            sel_keys: list[ObjectKey] = []
            for i in range(len(state.keys)):
                if bitmap.test(i):
                    src_objs.append(state.temp_objs[i])
                    dst_objs.append(state.dst_objs[i])
                    sel_keys.append(state.keys[i])

            if not src_objs:
                # Inner loaded nothing — skip deserialize, finalize.
                self._release_write_temps(state.temp_keys)
                self._finalize_load(wrapped_id, bitmap)
                continue

            state.load_bitmap = bitmap
            try:
                serde_id = self._serde.submit_deserialize(src_objs, dst_objs, sel_keys)
            except Exception:
                logger.exception(
                    "Serde wrapper: submit_deserialize raised for task %d",
                    wrapped_id,
                )
                self._release_write_temps(state.temp_keys)
                self._finalize_load(wrapped_id, Bitmap(len(state.keys)))
                continue
            with self._lock:
                self._serde_to_load[serde_id] = wrapped_id

    def _drain_deserialize(self) -> None:
        """Drain deserialize completions; report inner's load bitmap on
        success, all-zeros on deserialize failure."""
        with self._lock:
            pending = list(self._serde_to_load.keys())
        for serde_id in pending:
            result = self._serde.query_deserialize_result(serde_id)
            if result is None:
                continue
            with self._lock:
                wrapped_id = self._serde_to_load.pop(serde_id, None)
                state = (
                    self._load_tasks.get(wrapped_id) if wrapped_id is not None else None
                )
            if wrapped_id is None or state is None:
                continue

            if result:
                # ``load_bitmap`` was populated by _drain_inner_load before
                # the task was registered in ``_serde_to_load``.
                final_bitmap = state.load_bitmap
            else:
                logger.warning(
                    "Serde wrapper: deserialize failed for task %d; "
                    "reporting all keys as failed",
                    wrapped_id,
                )
                final_bitmap = Bitmap(len(state.keys))

            self._release_write_temps(state.temp_keys)
            self._finalize_load(wrapped_id, final_bitmap)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alloc_temp_buffers(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[list[ObjectKey], list[MemoryObj] | None]:
        """Reserve one temp byte buffer per input key. All-or-nothing:
        any single failure releases the partial successes and returns
        ``(temp_keys, None)``.

        Args:
            keys: Original logical keys; used only to derive temp keys.
            objects: Source (store) or destination (load) MemoryObjs.
                All entries must share a single ``(shape, dtype)`` — the
                caller (store/prefetch controller) is responsible for
                shape-grouping before submission.
        """
        shape_0 = objects[0].get_shapes()
        dtype_0 = objects[0].get_dtypes()
        temp_keys = [make_temp_key(k) for k in keys]
        layout = serialized_layout_desc(
            MemoryLayoutDesc(shapes=shape_0, dtypes=dtype_0), self._serde
        )
        results = self._l1_manager.reserve_write(
            keys=temp_keys,
            is_temporary=[True] * len(temp_keys),
            layout_desc=layout,
            mode="new",
        )
        # First pass: collect every key whose reserve_write succeeded.
        # We must scan the full list (not bail on the first failure)
        # so a mixed-success result still releases all reserved keys.
        successful_temp_keys: list[ObjectKey] = []
        for temp_key in temp_keys:
            r = results.get(temp_key)
            if r is not None and r[0] == L1Error.SUCCESS:
                successful_temp_keys.append(temp_key)
        if len(successful_temp_keys) != len(temp_keys):
            self._release_write_temps(successful_temp_keys)
            return temp_keys, None
        temp_objs = [results[tk][1] for tk in temp_keys]
        return temp_keys, temp_objs

    def _release_write_temps(self, temp_keys: list[ObjectKey]) -> None:
        """Release write-locked temps and delete them. No-op on empty."""
        if not temp_keys:
            return
        try:
            self._l1_manager.finish_write(temp_keys)
            self._l1_manager.delete(temp_keys)
        except Exception:
            logger.exception("Serde wrapper: failed releasing write-locked temps")

    def _finalize_store(
        self,
        wrapped_id: L2TaskId,
        success: bool,
        bytes_transferred: int = 0,
    ) -> None:
        with self._lock:
            self._store_tasks.pop(wrapped_id, None)
            self._completed_store[wrapped_id] = L2StoreResult(
                success, bytes_transferred
            )
        try:
            self._store_efd.notify()
        except OSError:
            logger.exception("Serde wrapper: failed to signal store notifier")

    def _finalize_load(self, wrapped_id: L2TaskId, bitmap: Bitmap) -> None:
        with self._lock:
            self._load_tasks.pop(wrapped_id, None)
            self._completed_load[wrapped_id] = bitmap
        try:
            self._load_efd.notify()
        except OSError:
            logger.exception("Serde wrapper: failed to signal load notifier")
