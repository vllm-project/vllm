# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

The controller runs a background thread with an event-driven loop that:
1. Accepts prefetch requests from external threads via submit_prefetch_request.
2. Submits lookup_and_lock tasks to all L2 adapters.
3. Computes a load plan, keeping the keys retained by the TrimPolicy
   (PREFIX, SEGMENTED_PREFIX, or SPARSE).
4. Reserves L1 write buffers and submits load tasks to L2 adapters.
5. On load completion, transitions L1 entries from write-locked to read-locked.
6. Reports the retained-key bitmap.
"""

# Standard
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable
import enum
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.adapter_lifecycle import (
    AddAdapterOp,
    RemoveAdapterOp,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    PrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


# HELPER FUNCTIONS
def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge bitmaps with a bitwise OR into a ``num_keys``-sized bitmap.

    Always returns a ``num_keys``-sized bitmap (empty input -> all zeros), so
    downstream ``&`` operations never hit a size mismatch.
    """
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


def build_trim_mask(
    found: Bitmap,
    num_keys: int,
    policy: TrimPolicy = TrimPolicy.PREFIX,
) -> Bitmap:
    """Subset of ``found`` to keep (load + read-lock + report); the rest is
    released.

    PREFIX trims at the first gap (leading contiguous run). The non-PREFIX
    policies keep every set bit, gaps included, and differ only in intent:
    SEGMENTED_PREFIX keeps the keys that loaded when an L2 hit fails to load
    into L1 (e.g. OOM) mid-prefix; SPARSE keeps an intentionally scattered set.

    Args:
        found: Bitmap of found keys, over key indices ``0..num_keys-1``.
        num_keys: Total number of requested keys.
        policy: Trim policy to apply (see :class:`TrimPolicy`).

    Returns:
        Bitmap of the retained key indices.
    """
    if policy is TrimPolicy.PREFIX:
        return Bitmap(num_keys, found.count_leading_ones())
    return found


def trim_load_plan_with_mask(
    load_plan: dict[int, Bitmap],
    mask: Bitmap,
) -> dict[int, Bitmap]:
    """Trim the load plan to the key indices set in ``mask`` (gap-tolerant).

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        mask: Bitmap of key indices to retain.

    Returns:
        Trimmed load plan; adapter indices retaining no keys are dropped.
    """
    trimmed_plan: dict[int, Bitmap] = {}
    for adapter_idx, bitmap in load_plan.items():
        new_bitmap = bitmap & mask
        if new_bitmap.popcount() == 0:
            continue
        trimmed_plan[adapter_idx] = new_bitmap
    return trimmed_plan


# Poll timeout in milliseconds for the prefetch loop
PREFETCH_LOOP_POLL_TIMEOUT_MS = 500

PrefetchRequestId = int


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    phase: PrefetchPhase
    extra_count: int = 0
    """Extra read locks per key (on top of the default 1) to acquire when
    transitioning from write-locked to read-locked.  Must match the
    ``extra_count`` used in the corresponding ``submit_prefetch_task`` call."""

    policy: TrimPolicy = TrimPolicy.PREFIX
    """Which retained-subset policy to apply (see :class:`TrimPolicy`)."""

    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    """Cross-chunk attention windows of all object groups, in object-group
    order."""
    mode: PrefetchMode = PrefetchMode.LOOKUP
    """The prefetch intent (see :class:`PrefetchMode`).  ``WARM`` forces all
    loaded keys permanent and acquires no read lock; ``LOOKUP`` defers
    retention to the policy and read-locks loaded keys."""

    # Lookup phase: adapter_idx -> task_id (removed as results arrive)
    pending_lookup_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Lookup phase: adapter_idx -> bitmap (populated as results arrive)
    lookup_results: dict[int, Bitmap] = field(default_factory=dict)

    # Load phase: adapter_idx -> bitmap of key indices to load
    load_plan: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: adapter_idx -> task_id (removed as results arrive)
    pending_load_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Load phase: adapter_idx -> L1 bytes reserved for that adapter's
    # in-flight load.  Read by the inflight_load_memory_usage_bytes gauge.
    load_bytes_by_adapter: dict[int, int] = field(default_factory=dict)
    # Load phase: adapter_idx -> bitmap (populated as results arrive)
    load_results: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: keys that were write-reserved in L1
    write_reserved_keys: list[ObjectKey] = field(default_factory=list)
    write_reserved_objs: dict[ObjectKey, "MemoryObj"] = field(default_factory=dict)

    def all_lookups_done(self) -> bool:
        return len(self.pending_lookup_tasks) == 0

    def all_loads_done(self) -> bool:
        return len(self.pending_load_tasks) == 0


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Runs a background thread that submits lookup_and_lock to all adapters.
    3. Uses PrefetchPolicy to compute a load plan from lookup results.
    4. Reserves L1 write buffers and submits load tasks to adapters.
    5. On completion, transitions loaded keys to read-locked state.
    6. Reports the number of prefix hits via query_prefetch_result.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan decisions.
        max_in_flight: Maximum number of concurrent prefetch requests.
    """

    # Singleton dispatch for the in-flight load gauges: tests may construct
    # multiple controllers but the OTel SDK only honors the first gauge
    # registration, so the callbacks read from the most recently built
    # instance via ``_gauge_target``.
    _gauges_registered: bool = False
    _gauge_target: "PrefetchController | None" = None

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: PrefetchPolicy,
        max_in_flight: int = 8,
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
        self._max_in_flight = max_in_flight

        # Adapters that are being drained and will be removed after all
        # the in-flight operations are done.
        self._draining: dict[int, threading.Event] = {}

        # Control-plane queue for runtime add/remove, used by the internal
        # loop thread
        self._adapter_ops_lock = threading.Lock()
        self._pending_adapter_ops: list[AddAdapterOp | RemoveAdapterOp] = []
        self._adapter_ctrl_efd = create_event_notifier()

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        self._pending_queue: list[tuple[PrefetchRequestId, PrefetchRequestSpec]] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[tuple[PrefetchRequestId, PrefetchRequestSpec]] = []
        self._next_request_id: PrefetchRequestId = 0
        self._submission_efd = create_event_notifier()

        # Thread-safe lookup results (background -> external)
        self._lookup_results_lock = threading.Lock()
        self._completed_lookups: dict[PrefetchRequestId, int] = {}

        # Thread-safe prefetch results (background -> external).  The condition
        # variable lets a WAIT_PREFETCH_STATUS handler block until a result is
        # published instead of busy-polling QUERY_PREFETCH_STATUS.
        self._prefetch_results_lock = threading.Lock()
        self._prefetch_results_cv = threading.Condition(self._prefetch_results_lock)
        self._completed_results: dict[PrefetchRequestId, Bitmap] = {}

        # Map eventfds to adapter indices for quick lookup in poll.
        # Relies on the L2AdapterInterface contract that every adapter
        # returns distinct fds for store/lookup/load, and no two adapters
        # share an fd.  See the docstrings in L2AdapterInterface.
        self._lookup_efd_to_adapter: dict[int, int] = {}
        self._load_efd_to_adapter: dict[int, int] = {}
        for adapter_id, adapter in self._l2_adapters.items():
            self._lookup_efd_to_adapter[adapter.get_lookup_and_lock_event_fd()] = (
                adapter_id
            )
            self._load_efd_to_adapter[adapter.get_load_event_fd()] = adapter_id

        self._event_bus = get_event_bus()

        PrefetchController._gauge_target = self
        if not PrefetchController._gauges_registered:
            PrefetchController._gauges_registered = True
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.num_inflight_l2_loads",
                "L2 -> L1 prefetch load tasks currently executing, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_loads_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.inflight_load_memory_usage_bytes",
                "L1 bytes reserved by in-flight L2 -> L1 prefetch loads, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_load_bytes_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.l2_adapters",
                (
                    "Count of L2 adapters attached to the prefetch controller, "
                    "tagged by ``state`` (active or draining)."
                ),
                lambda: (
                    PrefetchController._gauge_target.get_adapter_state_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
        )

    # =========================================================================
    # External API (thread-safe)
    # =========================================================================

    def submit_prefetch_request(
        self,
        spec: PrefetchRequestSpec,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request.

        Thread-safe. Can be called from any thread.

        The retained subset of found keys is chosen by ``spec.policy`` (see
        :class:`TrimPolicy`).  With the default ``PREFIX`` policy, only the
        **contiguous prefix** of found keys is loaded from L2: if L2 has keys
        {0, 1, 3, 4} but not key 2, only keys {0, 1} are loaded because the gap
        at index 2 breaks the prefix.  Keys outside the retained set are never
        transferred, saving I/O bandwidth and L1 memory.  Use
        :meth:`query_prefetch_result` to retrieve the retained set once the
        request completes.

        Args:
            spec: The prefetch request inputs (see :class:`PrefetchRequestSpec`).

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._submission_queue.append((request_id, spec))
        self._submission_efd.notify()
        return request_id

    def query_lookup_result(self, request_id: PrefetchRequestId) -> int | None:
        """
        Query the keys that are found during the lookup for a specific request.

        Thread-safe. Returns the prefix-hit count if the lookup phase
        has completed, None if still in progress, or the prefetch request
        has already been consumed by query_prefetch_result.

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits from the lookup phase, or None if not yet complete
            or if the request has already been consumed by a previous call to this
            method.

        Note:
            This function does not pop the result. The caller need to make sure to call
            the query_prefetch_result after calling this function, otherwise nobody
            will clean up the completed lookups dictionary, causing memory leak.
        """
        with self._lookup_results_lock:
            return self._completed_lookups.get(request_id, None)

    def query_prefetch_result(self, request_id: PrefetchRequestId) -> Bitmap | None:
        """
        Query the result of a prefetch request.

        Thread-safe. Returns the retained-key bitmap if the request
        has completed, None if still in progress. Each result can only
        be retrieved once (subsequent calls return None).

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits, or None if not yet complete.

        Note:
            This function will pop the completed lookup results as well.
            Therefore, the caller need to make sure that never call
            query_lookup_result after calling this function, otherwise it will
            get None forever.
        """
        with self._prefetch_results_lock:
            result = self._completed_results.pop(request_id, None)
        if result is not None:
            with self._lookup_results_lock:
                self._completed_lookups.pop(request_id, None)
        return result

    def wait_prefetch_result(
        self, request_id: PrefetchRequestId, timeout: float
    ) -> bool:
        """
        Block until a prefetch request's result is published, or until timeout.

        Thread-safe. Lets a handler wait for prefetch completion instead of
        busy-polling query_prefetch_result. Does not consume the result; the
        caller still retrieves it via query_prefetch_result.

        Args:
            request_id: The request ID from submit_prefetch_request.
            timeout: Maximum number of seconds to wait for the result.

        Returns:
            True if the result became available within the timeout, False if
            the wait timed out.
        """
        with self._prefetch_results_cv:
            return self._prefetch_results_cv.wait_for(
                lambda: request_id in self._completed_results, timeout
            )

    def report_status(self) -> dict:
        """Return a status dict for the prefetch controller."""
        is_healthy = self._thread.is_alive()
        with self._submission_lock:
            submission_queue_size = len(self._submission_queue)
        with self._prefetch_results_lock:
            completed_results_count = len(self._completed_results)
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "max_in_flight": self._max_in_flight,
            "submission_queue_size": submission_queue_size,
            "pending_queue_size": self._status_pending_count,
            "in_flight_request_count": self._status_in_flight_count,
            "lookup_phase_count": self._status_lookup_phase_count,
            "load_phase_count": self._status_load_phase_count,
            "completed_results_count": completed_results_count,
            "num_l2_adapters": len(self._l2_adapters),
            "num_active_adapters": len(self._l2_adapters) - len(self._draining),
            "num_draining_adapters": len(self._draining),
        }

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

    def _snapshot_inflight_loads(self) -> dict[int, tuple[int, int]]:
        """``{adapter_idx: (count, reserved_bytes)}`` for in-flight L2 -> L1
        loads, computed via GIL-atomic ``dict.copy()`` snapshots so the
        OTel reader thread can call this concurrently with the prefetch
        loop without locking.
        """
        counts: dict[int, int] = defaultdict(int)
        bytes_by_adapter: dict[int, int] = defaultdict(int)
        for request in self._in_flight_requests.copy().values():
            for idx, reserved in request.load_bytes_by_adapter.copy().items():
                counts[idx] += 1
                bytes_by_adapter[idx] += reserved
        return {idx: (counts[idx], bytes_by_adapter[idx]) for idx in counts}

    def get_inflight_loads_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(count, attributes)`` for the
        ``lmcache_mp.num_inflight_l2_loads`` gauge."""
        observations: list[tuple[int | float, dict[str, object]]] = []
        for idx, (count, _) in self._snapshot_inflight_loads().items():
            desc = self._adapter_descriptors.get(idx)
            if desc is None:
                continue
            observations.append(
                (count, {"l2_name": desc.type_name, "adapter_index": idx})
            )
        return observations

    def get_inflight_load_bytes_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(reserved_bytes, attributes)`` for the
        ``lmcache_mp.inflight_load_memory_usage_bytes`` gauge."""
        observations: list[tuple[int | float, dict[str, object]]] = []
        for idx, (_, reserved_bytes) in self._snapshot_inflight_loads().items():
            desc = self._adapter_descriptors.get(idx)
            if desc is None:
                continue
            observations.append(
                (reserved_bytes, {"l2_name": desc.type_name, "adapter_index": idx})
            )
        return observations

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the background prefetch loop thread."""
        logger.info("Starting PrefetchController...")
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop to stop and wait for the thread to join.

        Cleans up any in-flight requests (releases L1 write locks,
        L2 locks) before returning.
        """
        self._stop_flag.set()
        self._submission_efd.notify()
        self._thread.join()
        self._cleanup_in_flight_requests()
        self._submission_efd.close()
        self._adapter_ctrl_efd.close()

    def add_adapter(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        descriptor: AdapterDescriptor,
    ) -> None:
        """Blocking function to add a new adapter into the prefetch
        controller with the specified adapter ID and descriptor.

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
        if not op.done.wait(timeout=PREFETCH_LOOP_POLL_TIMEOUT_MS / 1000 + 5.0):
            raise RuntimeError(
                f"PrefetchController did not attach adapter {adapter_id} in time"
            )

    def request_remove_adapter(self, adapter_id: int) -> threading.Event:
        """Non-blocking function to request the removal of a L2 adapter
        specified by the adapter ID.

        New lookups stop routing to the adapter immediately; in-flight
        requests are allowed to complete.

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

    # =========================================================================
    # Background loop
    # =========================================================================

    def _prefetch_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The submission eventfd (new prefetch requests).
        - Each L2 adapter's lookup eventfd (completed lookups).
        - Each L2 adapter's load eventfd (completed loads).
        """
        poller = select.poll()
        submission_fd = self._submission_efd.fileno()
        poller.register(submission_fd, select.POLLIN)
        poller.register(self._adapter_ctrl_efd.fileno(), select.POLLIN)
        for efd in self._lookup_efd_to_adapter:
            poller.register(efd, select.POLLIN)
        for efd in self._load_efd_to_adapter:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            # First, apply runtime add/remove of the L2 adapters.
            self._apply_pending_adapter_ops(poller)

            ready = poller.poll(PREFETCH_LOOP_POLL_TIMEOUT_MS)

            signaled_adapters: dict[PrefetchPhase, set[int]] = {
                phase: set() for phase in PrefetchPhase
            }
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                try:
                    consume_fd(fd)
                except (OSError, BlockingIOError):
                    pass

                try:
                    if fd == submission_fd:
                        self._drain_submission_queue()
                    elif fd in self._lookup_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.LOOKUP].add(
                            self._lookup_efd_to_adapter[fd]
                        )
                    elif fd in self._load_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.PLAN_AND_LOAD].add(
                            self._load_efd_to_adapter[fd]
                        )
                except Exception:
                    logger.exception(
                        "Unexpected error in prefetch loop while processing fd %d",
                        fd,
                    )

            if any(signaled_adapters.values()):
                for request in list(self._in_flight_requests.values()):
                    try:
                        self._advance_request(request, signaled_adapters)
                    except Exception:
                        logger.exception(
                            "Unexpected error advancing in-flight prefetch request %d",
                            request.request_id,
                        )

            try:
                self._start_pending_requests()
            except Exception:
                logger.exception(
                    "Unexpected error in prefetch loop while starting pending requests"
                )

            # Finalize any draining adapter no longer have any in-flight
            # requests.
            self._finalize_drained_adapters(poller)

    def _apply_pending_adapter_ops(self, poller: "select.poll") -> None:
        """Apply queued add/remove ops on the prefetch loop thread."""
        with self._adapter_ops_lock:
            ops = self._pending_adapter_ops
            self._pending_adapter_ops = []
        for op in ops:
            if isinstance(op, AddAdapterOp):
                self._l2_adapters[op.adapter_id] = op.adapter
                self._adapter_descriptors[op.adapter_id] = op.descriptor
                lookup_efd = op.adapter.get_lookup_and_lock_event_fd()
                load_efd = op.adapter.get_load_event_fd()
                self._lookup_efd_to_adapter[lookup_efd] = op.adapter_id
                self._load_efd_to_adapter[load_efd] = op.adapter_id
                poller.register(lookup_efd, select.POLLIN)
                poller.register(load_efd, select.POLLIN)
                logger.info("PrefetchController attached adapter %d", op.adapter_id)
                op.done.set()
            elif isinstance(op, RemoveAdapterOp):
                if op.adapter_id not in self._l2_adapters:
                    op.done.set()
                    continue
                # Mark draining; new lookups skip it. The adapter stays
                # registered so in-flight requests can still complete.
                self._draining[op.adapter_id] = op.done
                logger.info(
                    "PrefetchController draining adapter %d (no new lookups routed)",
                    op.adapter_id,
                )

    def _adapter_in_use(self, adapter_id: int) -> bool:
        """True if any in-flight request still references ``adapter_id``."""
        for request in self._in_flight_requests.values():
            if (
                adapter_id in request.pending_lookup_tasks
                or adapter_id in request.pending_load_tasks
                or adapter_id in request.load_plan
                or adapter_id in request.lookup_results
            ):
                return True
        return False

    def _finalize_drained_adapters(self, poller: "select.poll") -> None:
        """Detach draining adapters no longer referenced by any request."""
        for adapter_id in list(self._draining):
            if self._adapter_in_use(adapter_id):
                continue
            adapter = self._l2_adapters.pop(adapter_id)
            self._adapter_descriptors.pop(adapter_id, None)
            lookup_efd = adapter.get_lookup_and_lock_event_fd()
            load_efd = adapter.get_load_event_fd()
            self._lookup_efd_to_adapter.pop(lookup_efd, None)
            self._load_efd_to_adapter.pop(load_efd, None)
            for efd in (lookup_efd, load_efd):
                try:
                    poller.unregister(efd)
                except (KeyError, OSError):
                    pass
            done = self._draining.pop(adapter_id)
            logger.info("PrefetchController detached adapter %d", adapter_id)
            done.set()

    def _drain_submission_queue(self) -> None:
        """Move items from the thread-safe submission queue to the
        pending queue."""
        with self._submission_lock:
            items = self._submission_queue
            self._submission_queue = []
        self._pending_queue.extend(items)
        self._status_pending_count += len(items)

    def _start_pending_requests(self) -> None:
        """Start pending requests up to the max in-flight limit."""
        while (
            self._pending_queue and len(self._in_flight_requests) < self._max_in_flight
        ):
            request_id, spec = self._pending_queue.pop(0)
            self._status_pending_count -= 1
            self._start_lookup_phase(request_id, spec)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        spec: PrefetchRequestSpec,
    ) -> None:
        """Submit lookup_and_lock to all live (non-draining) adapters for a
        new request."""
        # Skip adapters being drained so a new request never locks keys on
        # an adapter that is on its way out.
        routing_adapters = {
            adapter_id: adapter
            for adapter_id, adapter in self._l2_adapters.items()
            if adapter_id not in self._draining
        }
        if not routing_adapters:
            self._complete_request(request_id, Bitmap(len(spec.keys)))
            return

        pending_lookup_tasks: dict[int, L2TaskId] = {}
        for adapter_id, adapter in routing_adapters.items():
            task_id = adapter.submit_lookup_and_lock_task(spec.keys, spec.layout_desc)
            pending_lookup_tasks[adapter_id] = task_id

        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=spec.keys,
            layout_desc=spec.layout_desc,
            phase=PrefetchPhase.LOOKUP,
            extra_count=spec.extra_count,
            policy=spec.policy,
            attn_desc=spec.attn_desc,
            mode=spec.mode,
            pending_lookup_tasks=pending_lookup_tasks,
        )
        self._in_flight_requests[request_id] = request
        self._status_in_flight_count += 1
        self._status_lookup_phase_count += 1

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(spec.keys),
                    "adapter_count": len(pending_lookup_tasks),
                    "key_count_per_salt": Counter(k.cache_salt for k in spec.keys),
                },
            )
        )

    # =========================================================================
    # Load phase
    # =========================================================================
    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Compute load plan, reserve L1 buffers, and submit load tasks."""
        request.phase = PrefetchPhase.PLAN_AND_LOAD
        self._status_lookup_phase_count -= 1
        self._status_load_phase_count += 1

        # Step 1: get load plan from policy. Exclude draining adapters so no
        # new load targets them; any keys they locked during lookup fall
        # outside the plan and get unlocked in _unlock_unneeded_keys.
        routing_descriptors = [
            desc
            for adapter_id, desc in self._adapter_descriptors.items()
            if adapter_id not in self._draining
        ]
        load_plan = self._policy.select_load_plan(
            request.keys,
            request.lookup_results,
            routing_descriptors,
        )

        # Step 2: trim the load plan to the policy's retained subset
        num_keys = len(request.keys)
        merged_lookup = merge_bitmaps(load_plan.values(), num_keys)
        retained = build_trim_mask(merged_lookup, num_keys, request.policy)
        trimmed_plan = trim_load_plan_with_mask(load_plan, retained)

        if not trimmed_plan:
            # Nothing to load after trimming. Unlock all lookup locks and
            # complete with an empty retained set.
            self._unlock_all_lookups(request)
            self._update_lookup_results(request.request_id, 0)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "prefix_hit_count": 0,
                    },
                )
            )
            self._complete_request(request.request_id, Bitmap(num_keys))
            return

        # Step 3: reserve L1 write buffers
        merged_bitmap = merge_bitmaps(trimmed_plan.values(), len(request.keys))
        keys_to_reserve = merged_bitmap.gather(request.keys)
        l1_mgr = self._l1_manager

        # WARM retains every loaded key; LOOKUP follows the configured policy.
        if request.mode is PrefetchMode.WARM:
            retentions = [True] * len(keys_to_reserve)
        else:
            retentions = self._policy.select_l1_retentions(
                keys_to_reserve,
            )
        write_results = l1_mgr.reserve_write(
            keys=keys_to_reserve,
            is_temporary=[not r for r in retentions],
            layout_desc=request.layout_desc,
            mode="new",
        )

        # Step 4: filter to successfully reserved keys
        reserved_key_set: set[ObjectKey] = set()
        oom_keys: list[ObjectKey] = []
        for key, (err, mem_obj) in write_results.items():
            if err == L1Error.SUCCESS and mem_obj is not None:
                request.write_reserved_keys.append(key)
                request.write_reserved_objs[key] = mem_obj
                reserved_key_set.add(key)
            else:
                if err == L1Error.OUT_OF_MEMORY:
                    oom_keys.append(key)
                logger.debug(
                    "Prefetch request %d: reserve write failed for %s: %s",
                    request.request_id,
                    key,
                    err,
                )

        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l2_prefetch", "keys": oom_keys},
                )
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "l1_oom", "keys": oom_keys},
                )
            )

        # Step 5: recompute load plan excluding failed reservations
        reserved_bitmap = Bitmap(num_keys)
        for i, key in enumerate(request.keys):
            if key in reserved_key_set:
                reserved_bitmap.set(i)

        retained = build_trim_mask(reserved_bitmap, num_keys, request.policy)
        trimmed_plan = trim_load_plan_with_mask(load_plan, retained)
        request.load_plan = trimmed_plan

        ## Step 6: phase 1 unlock — keys locked in lookup but not in plan
        self._unlock_unneeded_keys(request)

        if not trimmed_plan:
            # Nothing loadable after filtering
            if request.write_reserved_keys:
                l1_mgr.finish_write(request.write_reserved_keys)
                l1_mgr.delete(request.write_reserved_keys)
            self._update_lookup_results(request.request_id, 0)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "prefix_hit_count": 0,
                    },
                )
            )
            self._complete_request(request.request_id, Bitmap(num_keys))
            return

        ## Step 7: submit load tasks per adapter
        for adapter_idx, bitmap in trimmed_plan.items():
            per_adapter_keys = bitmap.gather(request.keys)
            per_adapter_objs = [
                request.write_reserved_objs[key] for key in per_adapter_keys
            ]
            task_id = self._l2_adapters[adapter_idx].submit_load_task(
                per_adapter_keys, per_adapter_objs
            )
            request.pending_load_tasks[adapter_idx] = task_id
            # Per-adapter byte accounting for L2_LOAD_TASK_* throughput
            # events.  Uniform layout per chunk -> size * count.
            total_bytes = (
                per_adapter_objs[0].get_size() * len(per_adapter_objs)
                if per_adapter_objs
                else 0
            )
            request.load_bytes_by_adapter[adapter_idx] = total_bytes

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_SUBMITTED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                        "key_count": len(per_adapter_keys),
                        "total_bytes": total_bytes,
                    },
                )
            )

        ## Step 8: update the lookup result based on the final load plan
        self._update_lookup_results(request.request_id, retained.count_leading_ones())

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "prefix_hit_count": retained.count_leading_ones(),
                },
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": request.request_id,
                    "key_count": len(reserved_key_set),
                    "adapter_count": len(trimmed_plan),
                    "key_count_per_salt": Counter(
                        k.cache_salt for k in reserved_key_set
                    ),
                },
            )
        )

        logger.debug(
            "Prefetch request %d: submitted load tasks to %d adapters for %d keys",
            request.request_id,
            len(trimmed_plan),
            len(reserved_key_set),
        )

    def _update_lookup_results(
        self, request_id: PrefetchRequestId, prefix_hit_count: int
    ) -> None:
        """Store the prefix-hit count from the lookup phase."""
        with self._lookup_results_lock:
            self._completed_lookups[request_id] = prefix_hit_count

    def _advance_request(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: dict[PrefetchPhase, set[int]],
    ) -> None:
        """State-transition dispatcher by phase: poll signaled adapters for
        the request's current phase via the per-phase helper, then trigger
        the phase transition when done."""
        phase_adapters = signaled_adapters[request.phase]
        if not phase_adapters:
            return
        if request.phase == PrefetchPhase.LOOKUP:
            self._poll_lookup_results(request, phase_adapters)
            if request.all_lookups_done():
                self._transition_to_load_phase(request)
        elif request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._poll_load_results(request, phase_adapters)
            if request.all_loads_done():
                self._finalize_load(request)

    def _poll_lookup_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending lookup-and-lock results from signaled adapters."""
        for adapter_idx in list(request.pending_lookup_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_lookup_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_lookup_and_lock_result(
                task_id
            )
            if result is None:
                continue
            request.lookup_results[adapter_idx] = result
            del request.pending_lookup_tasks[adapter_idx]

    def _poll_load_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending load results from signaled adapters."""
        for adapter_idx in list(request.pending_load_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_load_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_load_result(task_id)
            if result is None:
                continue
            request.load_results[adapter_idx] = result
            del request.pending_load_tasks[adapter_idx]
            request.load_bytes_by_adapter.pop(adapter_idx, None)

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                    },
                )
            )

    def _finalize_load(self, request: InFlightPrefetchRequest) -> None:
        """
        Finalize a completed load: build result bitmap, transition L1
        state, release read locks outside the retained set, and report the
        retained-key bitmap.

        Partial load failures can create gaps, so a loaded key may fall
        outside the policy's retained set; its read lock must be released.
        """
        num_keys = len(request.keys)

        # Scatter per-adapter local load results into global positions.
        # Each adapter's load bitmap is locally indexed (size == adapter's
        # key count).  The plan bitmap maps local → global indices via
        # get_indices_list().
        result_bitmap = Bitmap(num_keys)
        for adapter_idx, plan_bitmap in request.load_plan.items():
            load_bitmap = request.load_results.get(adapter_idx)
            if load_bitmap is None:
                continue
            plan_indices = plan_bitmap.get_indices_list()
            for global_i in load_bitmap.gather(plan_indices):
                result_bitmap.set(global_i)

        # Separate loaded vs. failed among write-reserved keys
        loaded_keys: list[ObjectKey] = result_bitmap.gather(request.keys)
        loaded_set = set(loaded_keys)
        failed_keys = [k for k in request.write_reserved_keys if k not in loaded_set]

        # Phase 2 unlock: release L2 locks for all keys in the load plan
        self._unlock_all_plan_keys(request)

        l1_mgr = self._l1_manager

        # Transition loaded keys out of write-locked state.
        if loaded_keys:
            if request.mode is PrefetchMode.WARM:
                # Warm: make ready, pin nothing.
                l1_mgr.finish_write(loaded_keys)
            else:
                # write-locked -> read-locked; extra_count so each TP worker
                # gets its own read lock.
                l1_mgr.finish_write_and_reserve_read(
                    loaded_keys, extra_count=request.extra_count
                )

        # Clean up failed keys
        if failed_keys:
            l1_mgr.finish_write(failed_keys)
            l1_mgr.delete(failed_keys)

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "loaded_count": len(loaded_keys),
                    "failed_count": len(failed_keys),
                    "key_count_per_salt": Counter(k.cache_salt for k in loaded_keys),
                },
            )
        )

        # L2 prefetch-failure anomaly reporting: keys were reserved in L1
        # (expected to load from L2) but did not appear in the load bitmap.
        # Classified as ``not_found`` — the serde_failure reason will be
        # added once the serde PR lands and adapters can distinguish
        # deserialization errors from missing objects.
        if failed_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "not_found", "keys": failed_keys},
                )
            )

        # Release read locks for any loaded key outside the retained set
        # (partial load failures can create gaps).
        retained = build_trim_mask(result_bitmap, num_keys, request.policy)
        released_bitmap = result_bitmap & (~retained)
        released = released_bitmap.gather(request.keys)
        if released:
            l1_mgr.finish_read(released, extra_count=request.extra_count)

        self._complete_request(request.request_id, retained)

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _unlock_unneeded_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 1 unlock: keys locked in lookup but not in the load plan."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            plan_bitmap = request.load_plan.get(adapter_idx, Bitmap(len(request.keys)))
            to_unlock_bitmap = lookup_bitmap & (~plan_bitmap)
            unlock_keys = to_unlock_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_plan_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 2 unlock: release L2 locks for all keys in the load plan."""
        for adapter_idx, load_bitmap in request.load_plan.items():
            unlock_keys = load_bitmap.gather(request.keys)
            self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_lookups(self, request: InFlightPrefetchRequest) -> None:
        """Unlock all keys locked during lookup (nothing to load case)."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            unlock_keys = lookup_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _complete_request(self, request_id: PrefetchRequestId, result: Bitmap) -> None:
        """Store the retained-key bitmap and remove from in-flight tracking."""
        with self._prefetch_results_lock:
            self._completed_results[request_id] = result
            # Wake any WAIT_PREFETCH_STATUS handler blocked on this result.
            self._prefetch_results_cv.notify_all()
        removed = self._in_flight_requests.pop(request_id, None)
        if removed is not None:
            self._status_in_flight_count -= 1
            if removed.phase == PrefetchPhase.LOOKUP:
                self._status_lookup_phase_count -= 1
            elif removed.phase == PrefetchPhase.PLAN_AND_LOAD:
                self._status_load_phase_count -= 1
        logger.debug(
            "Prefetch request %d completed: %d retained keys",
            request_id,
            result.popcount(),
        )

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown."""
        l1_mgr = self._l1_manager
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write(request.write_reserved_keys)
                    l1_mgr.delete(request.write_reserved_keys)
                self._unlock_all_plan_keys(request)
            elif request.phase == PrefetchPhase.LOOKUP:
                self._unlock_all_lookups(request)
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.keys),
            )
        self._in_flight_requests.clear()
