# SPDX-License-Identifier: Apache-2.0
"""
Interface for L2 adapters
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.native_storage_ops import Bitmap
    from lmcache.v1.distributed.api import KeyListPage, MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
    from lmcache.v1.memory_management import MemoryObj

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

logger = init_logger(__name__)

L2TaskId = int


_EMPTY_BY_CACHE_SALT: Mapping[str, int] = MappingProxyType({})


@dataclass(frozen=True)
class AdapterUsage:
    """Unified usage report for an L2 adapter.

    Replaces the old ``tuple[float, float]`` return shape with a structured
    record that exposes both aggregate and per cache_salt byte counts.
    Buckets are keyed by ``ObjectKey.cache_salt`` directly — the salt may
    represent a user, a vLLM deployment, or any other isolation
    granularity the caller chooses; the adapter stays agnostic.

    Instances are returned as immutable snapshots:
    ``bytes_by_cache_salt`` is a read-only ``Mapping``
    (``MappingProxyType``) so callers cannot mutate the snapshot after
    the fact. Each ``get_usage()`` call returns a fresh snapshot so a
    held reference will never reflect later state.
    """

    total_bytes_used: int
    """Aggregate bytes across all cache_salt buckets."""

    total_capacity_bytes: int
    """Adapter's maximum capacity. ``0`` means unknown / unlimited; the
    adapter does not support aggregate (global) usage-based eviction."""

    bytes_by_cache_salt: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_BY_CACHE_SALT
    )
    """Bytes used per ``cache_salt``. Only entries with positive usage
    appear; an empty mapping means no traffic has been tracked yet.
    Read-only — wrap with ``dict(...)`` if you need a mutable copy."""

    @property
    def usage_fraction(self) -> float:
        """Aggregate usage as a fraction in [0, 1].

        Returns ``-1.0`` when capacity is unknown (``total_capacity_bytes
        <= 0``) — matches the legacy sentinel from ``get_usage()`` so
        callers can keep using ``< 0`` to mean "no eviction signal".
        """
        if self.total_capacity_bytes <= 0:
            return -1.0
        return self.total_bytes_used / self.total_capacity_bytes


class L2AdapterInterface(ABC):
    """
    The abstracted interface for L2 I/O adapters.

    The L2 I/O adapter mainly provides 3 main functionalities with non-blocking
    primitives:
    1. Store: store a batch of memory objects associated with a batch of keys.
    2. Lookup and lock: look up and lock a batch of objects by the given keys.
       will also try to 'lock' the objects to prevent being evicted before
       loading them to L1.
    3. Load: load a batch of objects by the given keys. The load operation is
       not guaranteed to succeed, and the caller should check the return value.
       In most of cases, it should be likely to succeed if the objects are locked.

    Note that the store and the load operation are pre-provided with the data buffer
    (i.e., memory objects), which is managed by the caller (L2 controller). The L2
    adapter is not supposed to manage the lifecycle of the memory objects.

    The non-blocking interface is designed as follows:
    1. Submit task
    2. Query the completed tasks (either pop all the completed tasks or query
       a specific task by its id)
    3. Use event fd to signal the completion of the tasks. The event fd will be
       handled by the caller (L2 controller). Note that the event fd will be
       closed by the `close()` function.


    Error handling:
    1. For store operation, we only provide a coarse-grained error handling, which
       means that we only report the error at the task level. If a store task fails,
       we will report the failure of the whole task, instead of reporting the failure
       of each key-object pair in the task. The caller can choose to retry the failed
       task or not.
    2. For both lookup and load operations, we will return a bitmap indicating the
       success or failure of each key-object pair in the task.

    Thread-safe:
    The L2 adapter is designed to be called by a 2 controller threads (store controller
    and prefetch controller), therefore, it needs to be thread-safe.
    """

    def __init__(self, max_capacity_bytes: int = 0) -> None:
        """
        Args:
            max_capacity_bytes: Adapter's maximum byte capacity. ``0``
                (default) marks the adapter as not supporting global
                (aggregate) eviction — ``supports_global_eviction``
                returns ``False`` and ``get_usage`` returns
                ``usage_fraction == -1.0``. Per-cache_salt eviction
                policies (e.g. quota-based) can still operate on the
                per-bucket byte counts regardless of this value.
        """
        self._listeners: list[L2AdapterListener] = []
        self._backend_name: str = ""
        self._event_bus = get_event_bus()

        # Centralized byte accounting. Subclasses pass ``sizes`` to
        # ``_notify_keys_stored`` / ``_notify_keys_deleted`` and the base
        # class maintains both aggregate and per cache_salt totals so
        # every adapter exposes the same shape via ``get_usage()``.
        self._max_capacity_bytes: int = max_capacity_bytes
        self._total_bytes_used: int = 0
        self._bytes_by_cache_salt: dict[str, int] = {}
        self._usage_lock = threading.Lock()

    #####################
    # Event Fd Interface
    #####################

    # IMPORTANT: Each of the three event fd methods below MUST return a
    # distinct file descriptor.  The store controller and prefetch controller
    # build fd-to-adapter lookup maps; if any two methods return the same fd
    # (within one adapter or across adapters), poll-based dispatch will
    # silently misroute events.

    @abstractmethod
    def get_store_event_fd(self) -> int:
        """
        Get the event fd for store operation, which will be signaled on the
        completion of the store tasks.

        Returns:
            int: the event fd for store operation.

        Note:
            Must be distinct from the lookup and load event fds of this
            adapter, and from the event fds of all other adapters.
        """
        pass

    @abstractmethod
    def get_lookup_and_lock_event_fd(self) -> int:
        """
        Get the event fd for lookup and lock operation, which will be signaled
        on the completion of the lookup and lock tasks.

        Returns:
            int: the event fd for lookup and lock operation.

        Note:
            Must be distinct from the store and load event fds of this
            adapter, and from the event fds of all other adapters.
        """
        pass

    @abstractmethod
    def get_load_event_fd(self) -> int:
        """
        Get the event fd for load operation, which will be signaled on the completion
        of the load tasks.

        Returns:
            int: the event fd for load operation.

        Note:
            Must be distinct from the store and lookup event fds of this
            adapter, and from the event fds of all other adapters.
        """
        pass

    #####################
    # Store Interface
    #####################

    @abstractmethod
    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """
        Submit a store task to store a batch of memory objects associated with
        a batch of keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be stored.
            objects (list[MemoryObj]): the list of memory objects to be stored.
                The length of the objects list should be the same as the length of
                the keys list.

        Returns:
            L2TaskId: the task id of the submitted store task.
        """
        pass

    @abstractmethod
    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Pop all completed store tasks.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task
            id to an ``L2StoreResult`` that encodes both the success flag
            and the bytes actually transferred. Use
            ``result.is_successful()`` and ``result.bytes_transferred()``
            to inspect the outcome.
        """
        pass

    #####################
    # Lookup and Lock Interface
    #####################

    @abstractmethod
    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        """
        Submit a lookup and lock task to look up and lock a batch of objects
        by the given keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be looked up and locked.
            layout_desc (MemoryLayoutDesc): the memory layout of the objects.
                This is an advisory hint; most adapters ignore it. The P2P
                adapter forwards it to the peer cache server.

        Returns:
            L2TaskId: the task id of the submitted lookup and lock task.
        """
        pass

    @abstractmethod
    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """
        Non-blockingly query the result of a lookup and lock task by its task id.
        The result is a bitmap indicating the success or failure of each key-object
        pair in the task.

        For a single task id, this function will ONLY return a non-None value ONCE.
        (Which means this function is not idempotent)

        Args:
            task_id (L2TaskId): the task id of the lookup and lock task.

        Returns:
            Optional[Bitmap]: a bitmap indicating the success or failure of each
            key-object pair in the task. 1 means successful, and 0 means failed.
            None is returned when the lookup and lock task is not completed.
        """
        pass

    @abstractmethod
    def submit_unlock(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Submit an unlock task to unlock a batch of objects by the given keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be unlocked.

        Note:
            This function does not return any task id, meaning that the caller
            assumes the unlock operation will be eventually successful, and will
            NEVER retry.
            Therefore, the implementation MUST make sure that the unlock operation
            is successful (i.e., have error handling and retry mechanism if needed).
        """
        pass

    #####################
    # Load Interface
    ######################

    @abstractmethod
    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """
        Submit a load task to load a batch of objects by the given keys. The load
        operation is not guaranteed to succeed, and the caller should check the
        return value.

        Args:
            keys (list[ObjectKey]): the list of keys to be loaded.
            objects (list[MemoryObj]): the list of memory objects as the load buffer.
                The L2 adapter will write the loaded data to the memory buffer provided
                by the caller. The caller is responsible for managing the lifecycle of
                the memory objects, and should make sure that the memory buffer is valid
                until the load task is completed.
                The length of the objects list should be the same as the length of the
                keys list.

        Returns:
            L2TaskId: the task id of the submitted load task.
        """
        pass

    @abstractmethod
    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """
        Non-blockingly query the result of a load task by its task id. The result
        is a bitmap indicating the success or failure of each key-object pair in
        the task.

        For a single task id, this function will ONLY return a non-None value ONCE.
        (Which means this function is not idempotent)

        Args:
            task_id (L2TaskId): the task id of the load task.

        Returns:
            Optional[Bitmap]: a bitmap indicating the success or failure of each
            key-object pair in the task. 1 means successful, and 0 means failed.
            None is returned when the load task is not completed.
        """
        pass

    #####################
    # Listener Interface
    #####################

    def register_listener(self, listener: L2AdapterListener) -> None:
        """Register a listener to receive L2 adapter events."""
        self._listeners.append(listener)

    def set_backend_name(self, name: str) -> None:
        """Set the registered adapter type name used to tag cache events.

        Called by the storage manager right after construction (initial
        and runtime-added adapters alike).

        Args:
            name: The registered adapter type name (e.g. ``"fs"``;
                non-empty).

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError("backend name must be non-empty")
        self._backend_name = name

    def _notify_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Update byte accounting and notify listeners that ``keys`` were
        stored. ``sizes[i]`` is the byte size of ``keys[i]``.

        Accounting is held under ``_usage_lock``; listener callbacks fire
        outside the lock so a slow listener cannot stall further notifies.
        """
        # Aggregate per-salt deltas before touching
        # ``_bytes_by_cache_salt`` — one dict read/write per unique
        # salt instead of one per key. This matters when the registry is
        # large (10k+ salts) and keys/sizes are bulky.
        delta: dict[str, int] = {}
        total_delta = 0
        for key, size in zip(keys, sizes, strict=True):
            delta[key.cache_salt] = delta.get(key.cache_salt, 0) + size
            total_delta += size

        with self._usage_lock:
            self._total_bytes_used += total_delta
            for salt, d in delta.items():
                self._bytes_by_cache_salt[salt] = (
                    self._bytes_by_cache_salt.get(salt, 0) + d
                )
        for listener in self._listeners:
            listener.on_l2_keys_stored(keys, sizes)
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_KEYS_STORED,
                metadata={
                    "keys": keys,
                    "sizes": sizes,
                    "backend": self._backend_name,
                },
            )
        )

    def _notify_keys_accessed(self, keys: list[ObjectKey]) -> None:
        # ``_notify_keys_accessed`` carries no byte impact — only LRU
        # bookkeeping cares about it, so no accounting is needed here.
        for listener in self._listeners:
            listener.on_l2_keys_accessed(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_KEYS_ACCESSED,
                metadata={"keys": keys, "backend": self._backend_name},
            )
        )

    def _notify_keys_deleted(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Update byte accounting and notify listeners that ``keys`` were
        deleted. ``sizes[i]`` is the byte size of ``keys[i]`` (typically
        the same value the adapter passed to ``_notify_keys_stored``).

        Per-cache_salt buckets that drop to zero are removed so the
        ``bytes_by_cache_salt`` snapshot in ``AdapterUsage`` stays compact.

        Counters are clamped at zero — a delete that would drive
        ``_total_bytes_used`` negative indicates an accounting bug in the
        caller (size mismatch, double-delete, etc.). Without the clamp the
        sentinel ``usage_fraction == -1`` would silently disable eviction
        forever; with it we log a warning and recover.
        """
        # Same batching rationale as ``_notify_keys_stored`` — aggregate
        # per-salt deltas first so the hot path does one dict read/write
        # per unique salt, not per key.
        delta: dict[str, int] = {}
        total_delta = 0
        for key, size in zip(keys, sizes, strict=True):
            delta[key.cache_salt] = delta.get(key.cache_salt, 0) + size
            total_delta += size

        with self._usage_lock:
            self._total_bytes_used -= total_delta
            if self._total_bytes_used < 0:
                logger.warning(
                    "L2 adapter byte accounting underflow: "
                    "_total_bytes_used dropped to %d after deleting %d "
                    "keys (total size %d). Clamping to 0; this indicates "
                    "an accounting bug (double-delete or size mismatch) "
                    "in the adapter.",
                    self._total_bytes_used,
                    len(keys),
                    total_delta,
                )
                self._total_bytes_used = 0
            for salt, d in delta.items():
                new_total = self._bytes_by_cache_salt.get(salt, 0) - d
                if new_total <= 0:
                    self._bytes_by_cache_salt.pop(salt, None)
                else:
                    self._bytes_by_cache_salt[salt] = new_total
        for listener in self._listeners:
            listener.on_l2_keys_deleted(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_KEYS_DELETED,
                metadata={"keys": keys, "backend": self._backend_name},
            )
        )

    #####################
    # Eviction Interface
    #####################

    @property
    def supports_global_eviction(self) -> bool:
        """Whether this adapter supports **aggregate** (global) usage-driven
        eviction.

        ``True`` when the adapter declared a positive
        ``max_capacity_bytes`` at construction time. Adapters that don't
        track or cap aggregate byte usage (e.g. the FS adapter, which
        assumes unbounded disk) pass ``0`` and return ``False``. The
        storage manager skips creating an ``L2AdapterEvictionState``
        with a global policy for these adapters even if an
        ``eviction_config`` is otherwise present.

        Per-cache_salt eviction policies (e.g. quota-based) are
        orthogonal to this flag — they can operate on the per-bucket
        byte counts regardless of whether aggregate capacity is
        declared.
        """
        return self._max_capacity_bytes > 0

    def delete(self, keys: list[ObjectKey]) -> None:
        """
        Delete a batch of objects from L2 storage.

        Args:
            keys (list[ObjectKey]): The keys of the objects to delete.

        Note:
            Implementations should fire on_l2_keys_deleted on registered
            L2AdapterListeners once the deletion completes.

            The default implementation is a no-op. Subclasses that support
            eviction should override this method.
        """
        return None

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> KeyListPage:
        """List keys currently resident in this adapter, paginated.

        Args:
            model_name: if set, restrict to keys with this
                ``ObjectKey.model_name``.
            page_size: maximum entries to return in this page.
            cursor: opaque cursor from the previous page; ``None`` on
                the first call.

        Raises:
            NotImplementedError: the adapter does not support listing.
            ValueError: ``page_size`` is non-positive or ``cursor`` is
                malformed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement list_l2_keys"
        )

    def get_usage(self) -> AdapterUsage:
        """
        Return the current L2 storage utilization as an ``AdapterUsage``.

        The default implementation returns the totals maintained by the
        base class via ``_notify_keys_stored`` / ``_notify_keys_deleted``,
        so adapters that pass their stored sizes through those helpers do
        not need to override this method.

        ``AdapterUsage.usage_fraction`` returns ``-1.0`` when
        ``max_capacity_bytes <= 0`` — the legacy "no eviction signal"
        sentinel — so eviction-controller callers can keep the same
        ``< 0`` short-circuit they had with the old tuple API.
        """
        with self._usage_lock:
            per_salt_snapshot = {
                k: v for k, v in self._bytes_by_cache_salt.items() if v > 0
            }
            return AdapterUsage(
                total_bytes_used=self._total_bytes_used,
                total_capacity_bytes=self._max_capacity_bytes,
                # Wrap in a read-only view so callers can't mutate the
                # snapshot. The underlying dict is a fresh copy so the
                # view is fully detached from the adapter's live state.
                bytes_by_cache_salt=MappingProxyType(per_salt_snapshot),
            )

    #####################
    # Cleanup Interface
    #####################

    @abstractmethod
    def close(self) -> None:
        """
        Close the L2 adapter and release all the resources. After calling this function,
        the L2 adapter should not be used anymore.
        """
        pass

    #####################
    # Status Interface
    #####################

    def report_status(self) -> dict:
        """
        Return a status dict for this adapter.

        Must include at least ``is_healthy: bool``.
        Subclasses should override this with adapter-specific metrics.
        """
        return {
            "is_healthy": True,
            "extra_warning": "report_status is not implemented and runs default impl",
        }
