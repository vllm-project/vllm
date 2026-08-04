# SPDX-License-Identifier: Apache-2.0
"""
Store Controller: asynchronously copies data from L1 to L2 after writes complete.

The controller runs a background thread with an event-driven loop that:
1. Listens for L1 write-completion events via StoreListener.
2. Submits store tasks to L2 adapters based on StorePolicy decisions.
3. Monitors L2 task completion via event fds.
4. Releases L1 read locks and optionally deletes keys from L1.
"""

# Standard
from collections import Counter, defaultdict
from dataclasses import dataclass
import enum
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.adapter_lifecycle import (
    AddAdapterOp,
    RemoveAdapterOp,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    StorePolicy,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

logger = init_logger(__name__)

# Poll timeout in milliseconds for the store loop
STORE_LOOP_POLL_TIMEOUT_MS = 500


def _group_keys_by_shape(
    keys: list[ObjectKey],
) -> dict[tuple, list[ObjectKey]]:
    """Group ``keys`` by the fields that determine their KV cache shape.

    Each bucket shares a single ``(shape, dtype)``, so each bucket can be
    submitted as one ``submit_store_task`` call. Today the shape is pinned
    by ``(model_name, kv_rank)`` — ``kv_rank`` packs ``world_size`` and
    parallelism config, so different TP/PP setups land in different
    buckets. Extend the grouping tuple when a new shape-affecting field is
    added to ``ObjectKey``.
    """
    groups: dict[tuple, list[ObjectKey]] = defaultdict(list)
    for key in keys:
        groups[(key.model_name, key.kv_rank)].append(key)
    return groups


# Helper classes (module-level, before main class)


class StoreListener(L1ManagerListener):
    """
    Listener that receives L1 write-completion callbacks and enqueues
    keys for the StoreController's background loop.

    The ``on_keys_write_finished`` callback is invoked inside L1Manager's
    lock, so it must be non-blocking. It appends keys to an internal list
    and signals an eventfd to wake up the controller's select.poll().
    """

    def __init__(self) -> None:
        self._pending_keys: list[ObjectKey] = []
        self._lock = threading.Lock()
        self._event_fd = create_event_notifier()

    def get_event_fd(self) -> int:
        """
        Return the notifier fd that is signaled when new keys are
        available.

        Returns:
            int: The readable file descriptor for poller registration.
        """
        return self._event_fd.fileno()

    def notify(self) -> None:
        """Signal the notifier to wake any blocked poll() waiter."""
        self._event_fd.notify()

    def pop_pending_keys(self) -> list[ObjectKey]:
        """
        Pop all pending keys from the queue.

        This is non-blocking and should be called by the StoreController's
        main loop after select.poll() indicates the eventfd is ready.

        Returns:
            list[ObjectKey]: All keys enqueued since the last pop.
        """
        with self._lock:
            keys = self._pending_keys
            self._pending_keys = []
        return keys

    def pending_count(self) -> int:
        """Return the number of pending keys waiting to be processed."""
        with self._lock:
            return len(self._pending_keys)

    # L1ManagerListener implementation

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """
        Enqueue keys and signal the notifier.

        Called inside L1Manager's lock. Must be fast and must not
        call any L1Manager methods (would deadlock).

        Args:
            keys (list[ObjectKey]): Keys that finished writing.
        """
        with self._lock:
            self._pending_keys.extend(keys)
        self._event_fd.notify()

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        # No op here because we don't want to trigger store when the
        # objects are prefetched to L1.
        pass

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def close(self) -> None:
        """Close the notifier."""
        self._event_fd.close()


class StorePhase(enum.Enum):
    """Phases a store task may be in. Currently there is only an
    L2_STORE phase; new phases (e.g. verify, ack) would be added here and
    threaded through ``signaled_adapters`` without changing
    ``_advance_request``'s signature."""

    L2_STORE = enum.auto()


@dataclass
class InFlightStoreTask:
    """
    Tracks a single submitted L2 store task so the controller can
    release L1 read locks and perform cleanup when it completes.
    """

    adapter_index: int
    """Which L2 adapter this task was submitted to."""

    keys: list[ObjectKey]
    """All keys that were submitted in this store task."""

    read_locked_keys: list[ObjectKey]
    """The subset of keys for which reserve_read succeeded
    (i.e., keys holding an L1 read lock that must be released)."""

    l2_store_result: bool | None = None
    """L2 outcome (True=success, False=failure, None=still in flight)."""

    l2_bytes_transferred: int = 0
    """Bytes actually transferred by the adapter for this task."""


# Main class


class StoreController(StorageControllerInterface):
    """
    Asynchronously stores L1 data to L2 adapters after write completion.

    The controller:
    1. Registers a StoreListener with L1Manager to receive
       on_keys_write_finished callbacks.
    2. Runs a background thread with an event-driven loop using
       select.poll() on the listener eventfd and all L2 adapter
       store eventfds.
    3. On new keys: consults StorePolicy to decide targets,
       calls reserve_read to get MemoryObjs, and submits store
       tasks to L2 adapters.
    4. On L2 task completion: releases read locks, optionally
       deletes keys from L1 per policy.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The store policy for deciding targets and deletions.
    """

    # Singleton dispatch for ``lmcache_mp.num_inflight_l2_stores``: tests may
    # construct multiple controllers but the OTel SDK only honors the first
    # gauge registration, so the callback reads from the most recently built
    # instance via ``_gauge_target``.
    _gauge_registered: bool = False
    _gauge_target: "StoreController | None" = None

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: StorePolicy,
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters: dict[int, L2AdapterInterface] = {
            desc.index: adapter
            for desc, adapter in zip(adapter_descriptors, l2_adapters, strict=True)
        }
        self._adapter_descriptors: dict[int, AdapterDescriptor] = {
            desc.index: desc for desc in adapter_descriptors
        }
        self._policy = policy

        # Adapters that are being drained and will be removed after all
        # the in-flight operations are done.
        self._draining: dict[int, threading.Event] = {}

        # Control-plane queue for runtime add/remove, used by the internal
        # loop thread
        self._adapter_ops_lock = threading.Lock()
        self._pending_adapter_ops: list[AddAdapterOp | RemoveAdapterOp] = []
        self._adapter_ctrl_efd = create_event_notifier()

        self._listener = StoreListener()
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()

        # (adapter_index, task_id) -> InFlightStoreTask
        # Composite key is needed because task IDs are only unique
        # within a single adapter, not across adapters.
        self._in_flight_tasks: dict[tuple[int, L2TaskId], InFlightStoreTask] = {}

        # Shadow counter for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0

        StoreController._gauge_target = self
        if not StoreController._gauge_registered:
            StoreController._gauge_registered = True
            register_gauge(
                "lmcache.l2_store",
                "lmcache_mp.num_inflight_l2_stores",
                "L2 store tasks currently executing, per adapter",
                lambda: (
                    StoreController._gauge_target.get_inflight_stores_observations()
                    if StoreController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_store",
                "lmcache_mp.l2_adapters",
                (
                    "Count of L2 adapters attached to the store controller, "
                    "tagged by ``state`` (active or draining)."
                ),
                lambda: (
                    StoreController._gauge_target.get_adapter_state_observations()
                    if StoreController._gauge_target is not None
                    else []
                ),
            )

        # Map store eventfd -> adapter id for quick lookup in poll results
        self._efd_to_adapter_index: dict[int, int] = {}
        for adapter_id, adapter in self._l2_adapters.items():
            self._efd_to_adapter_index[adapter.get_store_event_fd()] = adapter_id

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._store_loop,
            daemon=True,
        )

    def start(self) -> None:
        """Start the background store loop thread."""
        logger.info("Starting StoreController...")
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop to stop, wait for the thread to join.

        Releases all in-flight read locks on shutdown so that
        L1 objects are not permanently locked.
        """
        self._stop_flag.set()
        # Wake up the poll loop so it can exit promptly
        self._listener.notify()
        self._thread.join()
        self._cleanup_in_flight_tasks()
        self._listener.close()
        self._adapter_ctrl_efd.close()

    def report_status(self) -> dict:
        """Return a status dict for the store controller."""
        is_healthy = self._thread.is_alive()
        num_draining = len(self._draining)
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "pending_keys_count": self._listener.pending_count(),
            "in_flight_task_count": self._status_in_flight_count,
            "num_l2_adapters": len(self._l2_adapters),
            "num_active_adapters": len(self._l2_adapters) - num_draining,
            "num_draining_adapters": num_draining,
        }

    def add_adapter(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        descriptor: AdapterDescriptor,
    ) -> None:
        """Blocking function to add a new adapter into the store controller
        with the specified adapter ID and descriptor.

        Args:
            adapter_id: Stable id assigned by the StorageManager.
            adapter: The adapter instance to attach.
            descriptor: The adapter's descriptor (``descriptor.index`` must
                equal ``adapter_id``).

        Raises:
            RuntimeError: If the background loop did not apply the op in
                time (e.g. the loop is not running).
        """
        op = AddAdapterOp(
            adapter_id=adapter_id,
            adapter=adapter,
            descriptor=descriptor,
            done=threading.Event(),
        )
        with self._adapter_ops_lock:
            self._pending_adapter_ops.append(op)
        self._adapter_ctrl_efd.notify()
        if not op.done.wait(timeout=STORE_LOOP_POLL_TIMEOUT_MS / 1000 + 5.0):
            raise RuntimeError(
                f"StoreController did not attach adapter {adapter_id} in time"
            )

    def request_remove_adapter(self, adapter_id: int) -> threading.Event:
        """Non-blocking function to request the removal of a L2 adapter
        specified by the adapter ID.

        New store tasks stop routing to the adapter immediately; in-flight
        tasks are allowed to complete.

        Args:
            adapter_id: Stable id of the adapter to drain.

        Returns:
            An Event signaled when the adapter is fully drained.
        """
        op = RemoveAdapterOp(adapter_id=adapter_id, done=threading.Event())
        with self._adapter_ops_lock:
            self._pending_adapter_ops.append(op)
        self._adapter_ctrl_efd.notify()
        return op.done

    def get_adapter_state_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """``(count, {"state": ...})`` tuples for the ``lmcache_mp.l2_adapters``
        gauge. ``len()`` reads are GIL-atomic, safe from the OTel thread."""
        num_draining = len(self._draining)
        return [
            (len(self._l2_adapters) - num_draining, {"state": "active"}),
            (num_draining, {"state": "draining"}),
        ]

    def get_inflight_stores_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(count, attributes)`` snapshot for the
        ``lmcache_mp.num_inflight_l2_stores`` gauge.

        ``dict.copy()`` is GIL-atomic in CPython, so reading from the
        OTel reader thread while the store loop mutates is safe; the
        snapshot may be one mutation stale, which is fine at the 10 s
        scrape cadence.
        """
        counts: dict[int, int] = defaultdict(int)
        for adapter_index, _ in self._in_flight_tasks.copy():
            counts[adapter_index] += 1
        observations: list[tuple[int | float, dict[str, object]]] = []
        for idx, count in counts.items():
            # A just-finalized adapter may be gone from the descriptor map
            # while a stale in-flight snapshot still references it; skip it
            # rather than KeyError on the OTel reader thread.
            desc = self._adapter_descriptors.get(idx)
            if desc is None:
                continue
            observations.append(
                (count, {"l2_name": desc.type_name, "adapter_index": idx})
            )
        return observations

    # Private methods

    def _store_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The StoreListener's eventfd (new keys from L1 writes).
        - Each L2 adapter's store eventfd (completed store tasks).

        Exits when the stop flag is set.
        """
        poller = select.poll()

        listener_efd = self._listener.get_event_fd()
        poller.register(listener_efd, select.POLLIN)
        poller.register(self._adapter_ctrl_efd.fileno(), select.POLLIN)

        for efd in self._efd_to_adapter_index:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            # First, apply runtime add/remove of the L2 adapters.
            self._apply_pending_adapter_ops(poller)

            ready = poller.poll(STORE_LOOP_POLL_TIMEOUT_MS)

            signaled_adapters: dict[StorePhase, set[int]] = {
                phase: set() for phase in StorePhase
            }
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                # Consume the notifier value
                try:
                    consume_fd(fd)
                except (OSError, BlockingIOError):
                    pass

                try:
                    if fd == listener_efd:
                        keys = self._listener.pop_pending_keys()
                        if keys:
                            self._process_new_keys(keys)
                    else:
                        adapter_idx = self._efd_to_adapter_index.get(fd)
                        if adapter_idx is not None:
                            signaled_adapters[StorePhase.L2_STORE].add(adapter_idx)
                except Exception:
                    logger.exception(
                        "Unexpected error in store loop while processing fd %d",
                        fd,
                    )

            if any(signaled_adapters.values()):
                try:
                    self._drain_l2_store_completions(
                        signaled_adapters[StorePhase.L2_STORE]
                    )
                except Exception:
                    logger.exception("Unexpected error draining L2 store completions")
                for task_key, task in list(self._in_flight_tasks.items()):
                    try:
                        self._advance_request(task_key, task)
                    except Exception:
                        logger.exception(
                            "Unexpected error advancing in-flight store task "
                            "(adapter %d, task %d)",
                            task_key[0],
                            task_key[1],
                        )

            # Finalize any draining adapter no longer have any in-flight
            # tasks.
            self._finalize_drained_adapters(poller)

    def _apply_pending_adapter_ops(self, poller: "select.poll") -> None:
        """Apply queued add/remove ops on the store loop thread."""
        with self._adapter_ops_lock:
            ops = self._pending_adapter_ops
            self._pending_adapter_ops = []
        for op in ops:
            if isinstance(op, AddAdapterOp):
                self._l2_adapters[op.adapter_id] = op.adapter
                self._adapter_descriptors[op.adapter_id] = op.descriptor
                efd = op.adapter.get_store_event_fd()
                self._efd_to_adapter_index[efd] = op.adapter_id
                poller.register(efd, select.POLLIN)
                logger.info("StoreController attached adapter %d", op.adapter_id)
                op.done.set()
            elif isinstance(op, RemoveAdapterOp):
                if op.adapter_id not in self._l2_adapters:
                    # Already gone (double remove); signal so the caller
                    # doesn't block.
                    op.done.set()
                    continue
                # Mark draining; routing skips it immediately. The adapter
                # stays registered so in-flight completions still process.
                self._draining[op.adapter_id] = op.done
                logger.info(
                    "StoreController draining adapter %d (no new stores routed)",
                    op.adapter_id,
                )

    def _adapter_in_use(self, adapter_id: int) -> bool:
        """True if any in-flight store task targets ``adapter_id``."""
        for task_adapter_id, _ in self._in_flight_tasks:
            if task_adapter_id == adapter_id:
                return True
        return False

    def _finalize_drained_adapters(self, poller: "select.poll") -> None:
        """Detach draining adapters that no longer have in-flight tasks."""
        for adapter_id in list(self._draining):
            if self._adapter_in_use(adapter_id):
                continue
            adapter = self._l2_adapters.pop(adapter_id)
            self._adapter_descriptors.pop(adapter_id, None)
            efd = adapter.get_store_event_fd()
            self._efd_to_adapter_index.pop(efd, None)
            try:
                poller.unregister(efd)
            except (KeyError, OSError):
                pass
            done = self._draining.pop(adapter_id)
            logger.info("StoreController detached adapter %d", adapter_id)
            done.set()

    def _process_new_keys(self, keys: list[ObjectKey]) -> None:
        """
        Process a batch of newly written keys.

        1. Ask the policy which adapters each key should go to.
        2. For each adapter target, reserve read access on L1 to get
           MemoryObj references (skip keys that fail — best-effort).
        3. Submit store tasks to L2 adapters.
        4. Track in-flight tasks for later cleanup.

        Args:
            keys (list[ObjectKey]): Keys that finished writing to L1.
        """

        for group in _group_keys_by_shape(keys).values():
            self._submit_store_for_single_shape(group)

    def _submit_store_for_single_shape(self, keys: list[ObjectKey]) -> None:
        """Submit ``keys`` (all same shape) to their target adapters."""
        # Only route to adapters that are live (not draining). Descriptors
        # for draining adapters are kept for in-flight completion handling
        # but excluded here so no new task targets them.
        routing_descriptors = [
            desc
            for adapter_id, desc in self._adapter_descriptors.items()
            if adapter_id not in self._draining
        ]
        plan = self._policy.select_store_targets(keys, routing_descriptors)

        l1_mgr = self._l1_manager

        for adapter_index, target_keys in plan.items():
            if not target_keys:
                continue

            if adapter_index not in self._l2_adapters:
                logger.error(
                    "StorePolicy returned invalid adapter id %d "
                    "(not among attached adapters). Skipping.",
                    adapter_index,
                )
                continue

            # Reserve read to get MemoryObj references and hold read locks
            read_results = l1_mgr.reserve_read(target_keys)

            successful_keys = []
            successful_objs = []
            not_found_keys: list[ObjectKey] = []
            write_locked_keys: list[ObjectKey] = []
            for key in target_keys:
                result = read_results.get(key)
                if result is None:
                    continue
                err, obj = result
                if err != L1Error.SUCCESS or obj is None:
                    if err == L1Error.KEY_NOT_EXIST:
                        not_found_keys.append(key)
                    elif err == L1Error.KEY_NOT_READABLE:
                        write_locked_keys.append(key)
                    logger.debug(
                        "Skipping key %s for L2 store (adapter %d): %s",
                        key,
                        adapter_index,
                        err,
                    )
                    continue
                successful_keys.append(key)
                successful_objs.append(obj)

            # L1 read-failure anomaly reporting: target_keys come from an
            # L1_WRITE_FINISHED notification, so failing to reserve_read them
            # immediately after means an unexpected eviction or lock race.
            if not_found_keys:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L1_READ_FAILED,
                        metadata={
                            "during": "l2_store",
                            "reason": "not_found",
                            "keys": not_found_keys,
                        },
                    )
                )
            if write_locked_keys:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L1_READ_FAILED,
                        metadata={
                            "during": "l2_store",
                            "reason": "write_locked",
                            "keys": write_locked_keys,
                        },
                    )
                )

            if not successful_keys:
                continue

            adapter = self._l2_adapters[adapter_index]
            task_id = adapter.submit_store_task(successful_keys, successful_objs)

            self._in_flight_tasks[(adapter_index, task_id)] = InFlightStoreTask(
                adapter_index=adapter_index,
                keys=successful_keys,
                read_locked_keys=list(successful_keys),
            )
            self._status_in_flight_count += 1

            # All objects for a single store task share one layout (L1
            # allocates uniform MemoryObjs per chunk), so total bytes is
            # size * count — avoids summing N identical values.
            total_bytes = successful_objs[0].get_size() * len(successful_objs)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_STORE_SUBMITTED,
                    metadata={
                        "adapter_index": adapter_index,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_index].type_name,
                        "key_count": len(successful_keys),
                        "total_bytes": total_bytes,
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in successful_keys
                        ),
                    },
                )
            )

            logger.debug(
                "Submitted store task %d to adapter %d with %d keys.",
                task_id,
                adapter_index,
                len(successful_keys),
            )

    def _drain_l2_store_completions(self, signaled_adapters: set[int]) -> None:
        """Deposit each signaled adapter's L2 outcomes onto their in-flight
        tasks, to be consumed by ``_advance_request``."""
        for adapter_index in signaled_adapters:
            adapter = self._l2_adapters[adapter_index]
            completed = adapter.pop_completed_store_tasks()
            for task_id, result in completed.items():
                task = self._in_flight_tasks.get((adapter_index, task_id))
                if task is None:
                    logger.warning(
                        "Completed store task %d (adapter %d) not found in tracking.",
                        task_id,
                        adapter_index,
                    )
                    continue
                task.l2_store_result = result.is_successful()
                task.l2_bytes_transferred = result.bytes_transferred()

    def _advance_request(
        self,
        task_key: tuple[int, L2TaskId],
        task: InFlightStoreTask,
    ) -> None:
        """State-transition dispatcher. Delegate to ``_finalize_store``
        once the L2 outcome has been recorded by
        ``_drain_l2_store_completions``."""
        if task.l2_store_result is None:
            return
        self._finalize_store(task_key, task)

    def _finalize_store(
        self,
        task_key: tuple[int, L2TaskId],
        task: InFlightStoreTask,
    ) -> None:
        """Release read locks, publish completion, apply policy L1 deletions
        on success, and remove the tracking entry."""
        adapter_index, task_id = task_key
        l1_mgr = self._l1_manager
        success = task.l2_store_result

        l1_mgr.finish_read(task.read_locked_keys)
        del self._in_flight_tasks[task_key]
        self._status_in_flight_count -= 1

        l2_name = self._adapter_descriptors[adapter_index].type_name
        completion_meta: dict[str, object] = {
            "adapter_index": adapter_index,
            "task_id": task_id,
            "l2_name": l2_name,
            "bytes_transferred": task.l2_bytes_transferred,
        }
        if success:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        **completion_meta,
                        "succeeded_count": len(task.keys),
                        "failed_count": 0,
                        "key_count_per_salt": Counter(k.cache_salt for k in task.keys),
                    },
                )
            )
            logger.debug(
                "L2 store task %d completed: adapter %d, %d keys.",
                task_id,
                adapter_index,
                len(task.keys),
            )
            delete_keys = self._policy.select_l1_deletions(task.keys)
            if delete_keys:
                l1_mgr.delete(delete_keys)
        else:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        **completion_meta,
                        "succeeded_count": 0,
                        "failed_count": len(task.keys),
                    },
                )
            )
            logger.warning(
                "Store task %d to adapter %d failed for keys: %s",
                task_id,
                adapter_index,
                task.keys,
            )

    def _cleanup_in_flight_tasks(self) -> None:
        """
        Release all held read locks for any in-flight tasks that
        haven't completed. Called during stop().
        """
        l1_mgr = self._l1_manager
        for (adapter_index, task_id), task in self._in_flight_tasks.items():
            logger.warning(
                "Cleaning up in-flight store task %d (adapter %d, %d keys).",
                task_id,
                adapter_index,
                len(task.read_locked_keys),
            )
            l1_mgr.finish_read(task.read_locked_keys)
        self._in_flight_tasks.clear()
