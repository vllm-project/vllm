# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Encoder-cache (EC) connector backed by Mooncake TransferEngine.

Used in disaggregated setups where an encoder / prefill instance produces
multimodal encoder outputs and a decode instance loads them over RDMA-capable
Mooncake transport instead of shared filesystem.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

logger = init_logger(__name__)

_T = TypeVar("_T")

_LEASE_TTL_SECONDS = 300
_RESERVATION_REFRESH_SECONDS = _LEASE_TTL_SECONDS / 2
_RESERVATION_REAP_INTERVAL_SECONDS = 1
_DRAIN_MIN_INTERVAL = 0.005
# Readiness notifications are advisory: the scheduler also learns from the
# reserve reply. Cap the queue so a shard nobody subscribed to cannot grow
# without bound.
_MAX_PENDING_EVENTS = 4096

_MOONCAKE_IMPORT_ERROR: ImportError | None
try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    TransferEngine = None  # type: ignore[misc, assignment]
    _MOONCAKE_IMPORT_ERROR = e
else:
    _MOONCAKE_IMPORT_ERROR = None


@dataclass
class ECMooncakeLoadSpec:
    """Per-item metadata shipped from scheduler to worker (pickle-friendly)."""

    mm_hash: str
    num_token: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    pushed: bool = False
    transfer_id: str = ""
    reservation_id: str = ""
    # The consumer pool still holds this item, so the load is a local handoff:
    # no transfer, no producer.
    local: bool = False


@dataclass
class ECMooncakePushSpec:
    """Destination reservation requested before an encoder tensor is ready."""

    mm_hash: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    consumer_zmq: str
    transfer_id: str
    request_id: str = ""


@dataclass
class ECMooncakeConnectorMetadata(ECConnectorMetadata):
    """Worker-side metadata for one scheduler step."""

    loads: list[ECMooncakeLoadSpec] = field(default_factory=list)
    pushes: list[ECMooncakePushSpec] = field(default_factory=list)

    def add_load(self, spec: ECMooncakeLoadSpec) -> None:
        self.loads.append(spec)

    def add_push(self, spec: ECMooncakePushSpec) -> None:
        self.pushes.append(spec)


@dataclass
class ECMooncakeWorkerMetadata(ECConnectorWorkerMetadata):
    """Completion state reported from workers to the scheduler."""

    loaded: set[str] = field(default_factory=set)
    failed_loads: set[str] = field(default_factory=set)
    # Items the receive pool dropped under pressure. The scheduler assumes an
    # evicted item stays resident until told otherwise.
    reclaimed: set[str] = field(default_factory=set)
    pending_loads: bool = False
    pending_saves: bool = False

    def aggregate(self, other: ECConnectorWorkerMetadata) -> ECMooncakeWorkerMetadata:
        assert isinstance(other, ECMooncakeWorkerMetadata)
        return ECMooncakeWorkerMetadata(
            # Every tensor-parallel rank gathers the embedding from its own
            # cache, so an item counts as loaded only where all of them have
            # it; one rank falling short must fail the load rather than leave
            # the scheduler believing it is ready.
            loaded=self.loaded & other.loaded,
            failed_loads=self.failed_loads | other.failed_loads,
            reclaimed=self.reclaimed | other.reclaimed,
            pending_loads=self.pending_loads or other.pending_loads,
            pending_saves=self.pending_saves or other.pending_saves,
        )


@dataclass
class _PushSourceRegistration:
    tensor: torch.Tensor
    nbytes: int
    users: int = 1


@dataclass
class _ConsumerPoolAllocation:
    offset: int
    size: int
    tensor: torch.Tensor


@dataclass
class _PushReservation:
    mm_hash: str
    reservation_id: str
    allocation: _ConsumerPoolAllocation
    shape: tuple[int, ...]
    dtype: str
    ready: bool = False
    owns_allocation: bool = True
    discard_on_complete: bool = False
    created_at: float = field(default_factory=time.monotonic)
    expires_at: float = 0


@dataclass(frozen=True)
class _PushCompletion:
    accepted: bool
    became_ready: bool = False


@dataclass
class _PendingPush:
    tensor: torch.Tensor
    spec: ECMooncakePushSpec
    reservation: Future[list[dict[str, Any]]]
    ready_event: torch.Event | None
    enqueued_at: float


@dataclass
class _PushPerfWindow:
    started_at: float = field(default_factory=time.monotonic)
    batches: int = 0
    items: int = 0
    bytes: int = 0
    skipped_items: int = 0
    failures: int = 0
    stage_totals_ms: dict[str, float] = field(default_factory=dict)
    stage_max_ms: dict[str, float] = field(default_factory=dict)


class _ControlChannel:
    """Reusable REQ sockets for the ZMQ control plane.

    One context and one connection per message costs a thread spawn plus a
    TCP handshake, and the push path sends one message per reserve, complete
    and cancel. Sockets are cached per thread because a REQ socket is neither
    thread-safe nor usable after a failed exchange.
    """

    def __init__(self, timeout_ms: int):
        self._context = zmq.Context()
        self._timeout_ms = timeout_ms
        self._local = threading.local()

    def _sockets(self) -> dict[str, zmq.Socket]:
        sockets = getattr(self._local, "sockets", None)
        if sockets is None:
            sockets = {}
            self._local.sockets = sockets
        return sockets

    def _discard(self, addr: str) -> None:
        socket = self._sockets().pop(addr, None)
        if socket is not None:
            socket.close(linger=0)

    def send(self, addr: str, payload: dict[str, Any]) -> dict[str, Any]:
        sockets = self._sockets()
        socket = sockets.get(addr)
        if socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(addr)
            sockets[addr] = socket
        try:
            socket.send_json(payload)
            response = socket.recv_json()
        except Exception:
            # A REQ socket cannot recover from a half-finished exchange.
            self._discard(addr)
            raise
        assert isinstance(response, dict)
        return response

    def request(self, addr: str, payload: dict[str, Any]) -> Any:
        response = self.send(addr, payload)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "EC control request failed"))
        return response.get("result")

    def close(self) -> None:
        # Callers must have stopped every thread that used this channel.
        self._context.destroy(linger=0)


class _ResidentPool(Generic[_T]):
    """Content-addressed entries kept until their space is needed.

    Both sides of the connector hold the same thing under different names: a
    map from mm_hash to a device resource, a count of who is using it, and an
    eviction order over the rest. This is `BlockPool`'s accounting for
    variable-sized entries: `acquire`/`release` mirror `touch`/`free_blocks`,
    and `evict_lru` mirrors the reclaim inside `get_new_blocks`.

    An unreferenced entry stays resident. Eviction is driven by pressure, so
    the entry serves whoever needs it next instead of being transferred again.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0
        self._entries: dict[str, tuple[_T, int]] = {}
        self._refs: Counter[str] = Counter()
        # Unreferenced entries in eviction order, oldest first.
        self._evictable: OrderedDict[str, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    @property
    def num_evictable(self) -> int:
        return len(self._evictable)

    def referenced(self) -> list[str]:
        """Keys that are in use. `_refs` only holds entries above zero."""
        return list(self._refs)

    def referenced_or_retired(self) -> list[str]:
        """Every key held, in insertion order."""
        return list(self._entries)

    def get(self, key: str) -> _T | None:
        entry = self._entries.get(key)
        return entry[0] if entry is not None else None

    def insert(self, key: str, value: _T, nbytes: int) -> None:
        """Add a referenced entry, replacing any previous one."""
        previous = self._entries.get(key)
        if previous is not None:
            self.used -= previous[1]
        self._entries[key] = (value, nbytes)
        self.used += nbytes
        self.pin(key)

    def pin(self, key: str) -> _T | None:
        """Mark an entry as in use without counting a new reference.

        For a holder whose references are discovered by scanning rather than
        released in pairs, `pin`/`retire` are the matching operations.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._evictable.pop(key, None)
        self._refs[key] = max(1, self._refs[key])
        return entry[0]

    def retire(self, key: str) -> None:
        """Drop every reference; the entry is evictable from now on."""
        if key not in self._entries:
            return
        self._refs.pop(key, None)
        self._evictable[key] = None

    def refresh(self, key: str) -> None:
        """Move an unreferenced entry to the back of the eviction order."""
        if key in self._evictable:
            self._evictable.move_to_end(key)

    def acquire(self, key: str) -> _T | None:
        """Take one reference so pressure cannot evict the entry."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._evictable.pop(key, None)
        self._refs[key] += 1
        return entry[0]

    def release(self, key: str) -> None:
        """Drop one reference; the entry becomes evictable at zero."""
        if key not in self._entries:
            return
        count = self._refs[key] - 1
        if count > 0:
            self._refs[key] = count
            return
        self._refs.pop(key, None)
        self._evictable[key] = None

    def evict_lru(self, evict: Callable[[str, _T], bool]) -> str | None:
        """Drop the oldest entry `evict` accepts, and return its key.

        `evict` returns False for an entry that cannot go yet (a lease the
        remote side still holds, a deregistration that failed). Those keep
        their place in the order and the next candidate is tried.
        """
        for key in list(self._evictable):
            value, nbytes = self._entries[key]
            if not evict(key, value):
                continue
            self._evictable.pop(key, None)
            del self._entries[key]
            self._refs.pop(key, None)
            self.used -= nbytes
            return key
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._refs.clear()
        self._evictable.clear()
        self.used = 0


class _ContiguousAllocator:
    def __init__(self, capacity: int, alignment: int = 256):
        self.capacity = capacity
        self.alignment = alignment
        self._free = [(0, capacity)]

    def allocate(self, nbytes: int) -> tuple[int, int] | None:
        size = math.ceil(nbytes / self.alignment) * self.alignment
        for index, (offset, available) in enumerate(self._free):
            if size > available:
                continue
            if size == available:
                self._free.pop(index)
            else:
                self._free[index] = (offset + size, available - size)
            return offset, size
        return None

    def free(self, offset: int, size: int) -> None:
        index = bisect.bisect_left(self._free, (offset, size))
        self._free.insert(index, (offset, size))
        # Coalesce with the neighbours only; the rest of the list is already
        # merged, so a full re-scan per free is wasted work.
        if index + 1 < len(self._free):
            next_offset, next_size = self._free[index + 1]
            if offset + size == next_offset:
                self._free[index] = (offset, size + next_size)
                self._free.pop(index + 1)
        if index > 0:
            previous_offset, previous_size = self._free[index - 1]
            current_offset, current_size = self._free[index]
            if previous_offset + previous_size == current_offset:
                self._free[index - 1] = (
                    previous_offset,
                    previous_size + current_size,
                )
                self._free.pop(index)


class ECMooncakeControlServer:
    """Expose consumer reservations over a lightweight ZMQ control channel."""

    def __init__(
        self,
        host: str,
        port: int,
        reserve: Callable[[dict[str, Any]], dict[str, Any]],
        status: Callable[[str], dict[str, Any] | None],
        complete: Callable[[str, str], _PushCompletion],
        cancel: Callable[[str, str, bool], bool],
        reap: Callable[[], int],
        metrics_log_interval: float = 10,
        peer_ports: list[int] | None = None,
        device: torch.device | None = None,
    ):
        self.host = host
        self.port = port
        self.peer_ports = peer_ports or [port]
        self._device = device
        self.event_port: int | None = None
        self._reserve = reserve
        self._status = status
        self._complete = complete
        self._cancel = cancel
        self._reap = reap
        self._metrics_log_interval = metrics_log_interval
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None

    def start(self) -> None:
        def loop() -> None:
            if self._device is not None and self._device.type == "cuda":
                # Reserving can retire an entry, and the event that orders its
                # reuse is created on the recording thread's device rather
                # than the stream's. A thread starts on device 0, which under
                # a shard-local CUDA_VISIBLE_DEVICES is a peer's GPU, so
                # without this every shard but the first strands a primary
                # context there. The event orders correctly either way; what
                # it costs is a few hundred MiB on someone else's card.
                torch.accelerator.set_device_index(self._device.index or 0)
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            event_socket = context.socket(zmq.PUSH)
            pending_events: deque[dict[str, Any]] = deque()
            metrics: Counter[str] = Counter()

            def queue_event(event: dict[str, Any]) -> None:
                # The shard tag lets the scheduler tell each rank's readiness
                # apart; a transfer is only loadable once every rank has it.
                event["shard"] = self.port
                if len(pending_events) >= _MAX_PENDING_EVENTS:
                    pending_events.popleft()
                    metrics["events_dropped"] += 1
                pending_events.append(event)
                metrics["events_queued"] += 1

            metrics_started_at = time.monotonic()
            last_reap_at = metrics_started_at
            socket.setsockopt(zmq.RCVTIMEO, 100)
            try:
                socket.bind(f"tcp://{self.host}:{self.port}")
                self.event_port = event_socket.bind_to_random_port(f"tcp://{self.host}")
            except Exception as e:
                self._startup_error = e
                self._started.set()
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()
                return
            self._started.set()
            try:
                while not self._stop.is_set():
                    while pending_events:
                        try:
                            event_socket.send_json(
                                pending_events[0], flags=zmq.DONTWAIT
                            )
                        except zmq.Again:
                            break
                        pending_events.popleft()
                        metrics["events_sent"] += 1
                    now = time.monotonic()
                    if now - last_reap_at >= _RESERVATION_REAP_INTERVAL_SECONDS:
                        metrics["reservations_reaped"] += self._reap()
                        last_reap_at = now
                    if (
                        self._metrics_log_interval > 0
                        and now - metrics_started_at >= self._metrics_log_interval
                    ):
                        logger.info(
                            "EC Mooncake consumer control: requests=%s, "
                            "events_queued=%d, events_sent=%d, events_dropped=%d, "
                            "event_backlog=%d, reservations_reaped=%d",
                            {
                                key.removeprefix("request_"): value
                                for key, value in metrics.items()
                                if key.startswith("request_")
                            },
                            metrics["events_queued"],
                            metrics["events_sent"],
                            metrics["events_dropped"],
                            len(pending_events),
                            metrics["reservations_reaped"],
                        )
                        metrics.clear()
                        metrics_started_at = now
                    try:
                        request = socket.recv_json()
                    except zmq.Again:
                        continue
                    try:
                        op = request.get("op")
                        result: Any = None
                        metrics[f"request_{op}"] += 1
                        if op == "reserve":
                            result = self._reserve(request)
                            if result.get("ready"):
                                transfer_id = str(request["transfer_id"])
                                status = self._status(transfer_id)
                                if status is not None:
                                    queue_event({"transfer_id": transfer_id, **status})
                        elif op == "status":
                            result = self._status(str(request["transfer_id"]))
                        elif op == "event_port":
                            result = self.event_port
                        elif op == "peers":
                            # Every consumer shard receives its own copy, so a
                            # producer holding one address needs the rest.
                            result = {"ports": self.peer_ports}
                        elif op in ("complete", "complete_batch"):
                            items = (
                                request["items"]
                                if op == "complete_batch"
                                else [request]
                            )
                            completions = []
                            for item in items:
                                transfer_id = str(item["transfer_id"])
                                completion = self._complete(
                                    transfer_id,
                                    str(item["reservation_id"]),
                                )
                                completions.append(
                                    {
                                        "completed": completion.accepted,
                                        "became_ready": completion.became_ready,
                                    }
                                )
                                if not completion.became_ready:
                                    continue
                                status = self._status(transfer_id)
                                if status is not None:
                                    queue_event({"transfer_id": transfer_id, **status})
                            result = (
                                {"items": completions}
                                if op == "complete_batch"
                                else completions[0]
                            )
                        elif op == "cancel":
                            result = {
                                "cancelled": self._cancel(
                                    str(request["transfer_id"]),
                                    str(request.get("reservation_id", "")),
                                    bool(request.get("abandon", False)),
                                )
                            }
                        else:
                            raise ValueError(f"unknown control op: {op!r}")
                        socket.send_json({"ok": True, "result": result})
                    except Exception as e:
                        socket.send_json({"ok": False, "error": str(e)})
            finally:
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()

        self._thread = threading.Thread(
            target=loop, name="ec-mooncake-control", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("EC Mooncake control channel failed to start")
        if self._startup_error is not None:
            raise RuntimeError("EC Mooncake control channel failed to bind") from (
                self._startup_error
            )
        logger.info(
            "EC Mooncake control channel listening on tcp://%s:%d (events tcp://%s:%d)",
            self.host,
            self.port,
            self.host,
            self.event_port,
        )

    def shutdown(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()


class ECMooncakeConnector(ECConnectorBase):
    """
    EC connector using Mooncake TransferEngine for GPU tensor transport.

    The producer pushes each encoder output into a receive buffer the consumer
    reserved for it, so the transfer overlaps encoding instead of waiting for
    the consumer to ask. An item the consumer's encoder cache evicted stays in
    that pool and is handed back locally; when neither has it, the load fails
    with a retryable error so the caller can re-issue the request.

    Extra config (``ec_connector_extra_config``):

    - ``mooncake_protocol`` (optional): Passed to ``TransferEngine.initialize``
      (default ``"rdma"``).
    - ``consumer_buffer_pool_size`` (consumer, optional): Bytes reserved for a
      long-lived registered CUDA receive arena (default ``ec_buffer_size``).
    - ``reservation_zmq_port`` (consumer worker, required): Exposes registered
      receive addresses over ZMQ. Tensor-parallel rank ``r`` of the first
      pipeline stage listens on ``port + r``, and rank 0 reports the whole set,
      so a producer only needs the first address.
    - ``reservation_zmq_addr`` (consumer scheduler, required): Address of the
      consumer control channel. Defaults to ``tcp://127.0.0.1:<port>``.
    - ``transfer_max_workers`` (optional): Maximum concurrent Mooncake transfer
      batches (default ``4``).
    - ``control_max_workers`` (optional): Maximum concurrent reservation requests
      issued by a producer (default ``8``).
    - ``transfer_metrics_log_interval`` (optional): Seconds between aggregated
      push-transfer performance logs (default ``10``; ``0`` disables them).
    - ``consumer_metrics_log_interval`` (optional): Seconds between aggregated
      consumer lifecycle logs (default ``10``; ``0`` disables them).

    Parallelism: consumers may use tensor and pipeline parallelism. Only the
    first pipeline stage holds encoder outputs, and each tensor-parallel rank
    there gathers from its own cache, so every rank exposes a control channel
    and the producer writes into all of them concurrently from one registered
    source. That costs bandwidth but not latency, and avoids the second hop a
    receive-then-broadcast would add. Producers must be unsharded, and data
    parallelism is unsupported on either side.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        if _MOONCAKE_IMPORT_ERROR is not None or TransferEngine is None:
            raise ImportError(
                "Install mooncake-transfer-engine (see "
                "https://github.com/kvcache-ai/Mooncake ) to use ECMooncakeConnector."
            ) from _MOONCAKE_IMPORT_ERROR

        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            raise ValueError(
                "ECMooncakeConnector does not support data parallelism yet: the "
                "consumer exposes one control channel per instance, so a push "
                "cannot be routed to the replica that will run the request."
            )
        ec_cfg_early = vllm_config.ec_transfer_config
        assert ec_cfg_early is not None
        if ec_cfg_early.is_ec_producer:
            # The producer holds one copy of each encoder output and addresses
            # consumers directly; sharding it would only duplicate the push.
            if parallel_config.tensor_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers require tensor_parallel_size=1."
                )
            if parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers do not support pipeline parallelism."
                )

        self._role = role
        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None
        self._ec_cfg = ec_cfg
        self._extra = self._ec_cfg.ec_connector_extra_config
        self._protocol: str = self._extra.get("mooncake_protocol", "rdma")
        reservation_port = self._extra.get("reservation_zmq_port")
        self._reservation_zmq_port = (
            int(reservation_port) if reservation_port is not None else None
        )
        self._reservation_zmq_addr: str | None = self._extra.get("reservation_zmq_addr")
        if (
            self._reservation_zmq_addr is None
            and self._reservation_zmq_port is not None
        ):
            self._reservation_zmq_addr = f"tcp://127.0.0.1:{self._reservation_zmq_port}"
        self._registered_capacity = int(self._ec_cfg.ec_buffer_size)
        if self._registered_capacity <= 0:
            raise ValueError("ECMooncakeConnector requires ec_buffer_size > 0.")
        self._model_config = vllm_config.model_config
        self._metadata_fields_cache: dict[str, set[str]] = {}

        pool_size = self._extra.get(
            "consumer_buffer_pool_size", self._registered_capacity
        )
        self._consumer_pool_capacity = int(pool_size)
        self._consumer_pool: torch.Tensor | None = None
        self._consumer_pool_allocator: _ContiguousAllocator | None = None
        # The receive pool is orders of magnitude larger than the encoder
        # cache, so an item the encoder cache evicted stays resident here and
        # a later request gets it for a dict lookup instead of a transfer.
        self._consumer_residents: _ResidentPool[_ConsumerPoolAllocation] = (
            _ResidentPool(self._consumer_pool_capacity)
        )
        self._consumer_retire_events: dict[str, torch.Event] = {}
        self._consumer_pending_frees: list[
            tuple[torch.Event, _ConsumerPoolAllocation]
        ] = []
        self._consumer_reclaimed: set[str] = set()
        self._consumer_rank_resolved = False
        self._is_receiving_rank = True
        self._tp_rank = 0
        self._tp_size = 1
        self._consumer_pool_disabled = self._consumer_pool_capacity <= 0
        self._consumer_lock = threading.Lock()
        self._push_reservations: dict[str, _PushReservation] = {}
        self._cancelled_transfers: dict[str, float] = {}
        self._control_server: ECMooncakeControlServer | None = None
        self._consumer_metrics_log_interval = float(
            self._extra.get("consumer_metrics_log_interval", 10)
        )
        self._consumer_metrics_started_at = time.monotonic()
        self._consumer_worker_metrics: Counter[str] = Counter()
        self._consumer_scheduler_metrics: Counter[str] = Counter()
        self._consumer_missing_since: dict[str, float] = {}
        self._stalled_hashes: set[str] = set()
        self._unavailable_requests: set[str] = set()
        self._active_push_sources: Counter[tuple[str, int]] = Counter()
        self._active_push_sources_lock = threading.Lock()
        self._push_wait_timeout = float(self._extra.get("push_wait_timeout_s", 60))
        self._drain_pending = True
        self._drained_at = 0.0
        self._consumer_loading_since: dict[str, float] = {}
        self._consumer_pending_since: dict[str, float] = {}
        self._pending_spec_deadlines: dict[str, float] = {}
        self._pending_cancels: dict[str, Future[Any]] = {}
        self._cancelled_transfer_ids: set[str] = set()

        # Scheduler (consumer): transfer_id -> pending tensor layout.
        self._pending_specs: dict[str, ECMooncakeLoadSpec] = {}
        self._pending_specs_by_hash: dict[str, deque[str]] = {}
        self._load_specs: dict[str, ECMooncakeLoadSpec] = {}
        self._mm_datas_need_loads: dict[str, int] = {}
        self._loading_hashes: set[str] = set()
        self._ready_hashes: set[str] = set()
        # Scheduler-side mirror of the worker's receive pool, oldest first. An
        # item stays here after the encoder cache evicts it, so the next
        # request that needs it is served locally instead of consuming another
        # transfer. The worker reports what it reclaims under pressure; the
        # byte budget only guards against drift.
        self._resident_specs: OrderedDict[str, ECMooncakeLoadSpec] = OrderedDict()
        self._resident_bytes = 0
        self._scheduler_pending_work = False
        self._pushes_to_prepare: dict[str, ECMooncakePushSpec] = {}

        # Worker producer
        self._engine: TransferEngine | None = None
        self._engine_lock = threading.Lock()
        self._hostname = get_ip()
        # Published encoder outputs, referenced while a pull is reading them.
        self._pending_unregister: dict[int, torch.Tensor] = {}
        self._push_source_registrations: dict[int, _PushSourceRegistration] = {}
        self._push_source_registration_lock = threading.Lock()
        producer_pool = self._extra.get(
            "producer_buffer_pool_size", self._registered_capacity
        )
        self._producer_pool_capacity = int(producer_pool)
        self._producer_pool: torch.Tensor | None = None
        self._producer_pool_allocator: _ContiguousAllocator | None = None
        self._producer_pool_disabled = self._producer_pool_capacity <= 0
        self._producer_pool_lock = threading.Lock()
        transfer_workers = int(self._extra.get("transfer_max_workers", 4))
        control_workers = int(self._extra.get("control_max_workers", 8))
        self._transfer_metrics_log_interval = float(
            self._extra.get("transfer_metrics_log_interval", 10)
        )
        self._control_channel = _ControlChannel(
            int(float(self._extra.get("control_timeout_s", 30)) * 1000)
        )
        self._producer_metrics: Counter[str] = Counter()
        self._io_executor = ThreadPoolExecutor(
            max_workers=transfer_workers, thread_name_prefix="ec-mooncake-transfer"
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=control_workers, thread_name_prefix="ec-mooncake-control"
        )
        self._consumer_shard_cache: dict[str, list[str]] = {}
        self._shard_pool: ThreadPoolExecutor | None = None
        self._shard_pool_lock = threading.Lock()
        self._pending_saves: list[tuple[str, Future[None]]] = []
        self._pending_reservations: dict[
            str, deque[tuple[ECMooncakePushSpec, Future[list[dict[str, Any]]]]]
        ] = {}
        self._pending_pushes: list[_PendingPush] = []
        self._push_perf_lock = threading.Lock()
        self._push_perf = _PushPerfWindow()
        self._active_transfer_batches = 0
        self._queued_transfer_batches = 0
        self._event_zmq_ctx: zmq.Context | None = None
        self._event_zmq_socket: zmq.Socket | None = None
        self._event_shard_count = 1
        # transfer_id -> shards that reported it ready, oldest first. A sharded
        # consumer writes one copy per rank, so the item is only loadable once
        # every rank has reported. Bounded: a transfer whose last rank never
        # arrives is given up on by the push-wait timeout, not by this map.
        self._event_ready_shards: OrderedDict[str, set[int]] = OrderedDict()
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()
        self._shutdown = False

        if (
            role == ECConnectorRole.SCHEDULER
            and self.is_consumer
            and not self._reservation_zmq_addr
        ):
            raise ValueError(
                "ec_consumer with ECMooncakeConnector requires "
                "reservation_zmq_port or reservation_zmq_addr."
            )

    def _ensure_engine(self) -> TransferEngine:
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            eng = TransferEngine()
            ret = eng.initialize(self._hostname, "P2PHANDSHAKE", self._protocol, "")
            if ret != 0:
                raise RuntimeError("Mooncake TransferEngine initialization failed.")
            self._engine = eng
            logger.info(
                "ECMooncakeConnector TransferEngine ready at %s:%d",
                self._hostname,
                eng.get_rpc_port(),
            )
        return self._engine

    def _resolve_consumer_rank(self) -> None:
        """Place this worker in the consumer's receive topology.

        Encoder outputs only exist on the first pipeline stage, and every
        tensor-parallel rank there gathers from its own cache, so each of them
        receives its own copy on its own control channel. Ports run
        consecutively from the configured one so a producer holding the first
        address can reach the rest.
        """
        if self._consumer_rank_resolved:
            return
        self._consumer_rank_resolved = True
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            tp_group = get_tp_group()
            self._tp_rank = tp_group.rank_in_group
            self._tp_size = tp_group.world_size
            self._is_receiving_rank = get_pp_group().is_first_rank
        except AssertionError:
            # Groups are only absent outside a distributed run, where this
            # worker is the whole consumer.
            self._tp_rank = 0
            self._tp_size = 1
            self._is_receiving_rank = True

    def start_worker_services(self) -> None:
        if (
            self._role != ECConnectorRole.WORKER
            or not self.is_consumer
            or self._reservation_zmq_port is None
            or self._control_server is not None
        ):
            return
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Later pipeline stages hold no encoder outputs, so they need
            # neither a receive pool nor a control channel.
            return
        raw_device = self._ec_cfg.ec_buffer_device
        device_name = (
            raw_device.lower() if isinstance(raw_device, str) and raw_device else "cuda"
        )
        self._ensure_consumer_pool(torch.device(device_name), allow_host=True)
        if self._consumer_pool is None:
            raise RuntimeError(
                "Mooncake push mode requires a registered consumer buffer pool."
            )
        self._control_server = ECMooncakeControlServer(
            "0.0.0.0",
            self._reservation_zmq_port + self._tp_rank,
            self._reserve_push_destination,
            self._push_status,
            self._complete_push,
            self._cancel_push,
            self._expire_push_reservations,
            self._consumer_metrics_log_interval,
            peer_ports=[
                self._reservation_zmq_port + rank for rank in range(self._tp_size)
            ],
            device=self._consumer_pool.device,
        )
        self._control_server.start()

    def _unregister_memory(self, tensor: torch.Tensor) -> bool:
        assert self._engine is not None
        ret = self._engine.unregister_memory(tensor.data_ptr())
        if ret != 0:
            logger.error(
                "Mooncake EC memory unregistration failed for address %d: %d",
                tensor.data_ptr(),
                ret,
            )
            self._pending_unregister[tensor.data_ptr()] = tensor
            return False
        self._pending_unregister.pop(tensor.data_ptr(), None)
        return True

    def _unregister_memories(self, tensors: list[torch.Tensor]) -> None:
        assert self._engine is not None
        addresses = [tensor.data_ptr() for tensor in tensors]
        ret = self._engine.batch_unregister_memory(addresses)
        if ret != 0:
            for tensor in tensors:
                self._pending_unregister[tensor.data_ptr()] = tensor
            logger.warning(
                "Keeping %d EC tensors alive after Mooncake unregistration failure",
                len(tensors),
            )
            return
        for address in addresses:
            self._pending_unregister.pop(address, None)

    @staticmethod
    def _push_source_range(tensor: torch.Tensor) -> tuple[int, int]:
        # Register exactly the bytes that will be transferred. One encoder
        # batch returns its items as views of a single storage (models split
        # the batched embeddings, e.g. `image_embeds.split(sizes)`), so
        # registering the whole storage would overlap the per-tensor
        # registration a sibling item takes -- and Mooncake rejects
        # overlapping memory regions.
        return tensor.data_ptr(), tensor.nbytes

    def _acquire_push_source_registrations(
        self, tensors: list[torch.Tensor]
    ) -> list[int]:
        ranges: dict[int, tuple[int, torch.Tensor]] = {}
        for tensor in tensors:
            address, nbytes = self._push_source_range(tensor)
            ranges.setdefault(address, (nbytes, tensor))

        eng = self._ensure_engine()
        acquired: list[int] = []
        new_addresses: list[int] = []
        new_lengths: list[int] = []
        with self._push_source_registration_lock:
            for address, (nbytes, tensor) in ranges.items():
                entry = self._push_source_registrations.get(address)
                if entry is not None:
                    if entry.nbytes != nbytes:
                        raise RuntimeError(
                            "Mooncake EC source storage changed size while registered"
                        )
                    entry.users += 1
                    acquired.append(address)
                    continue
                new_addresses.append(address)
                new_lengths.append(nbytes)
                self._push_source_registrations[address] = _PushSourceRegistration(
                    tensor=tensor,
                    nbytes=nbytes,
                )
                acquired.append(address)

            if new_addresses:
                ret = eng.batch_register_memory(new_addresses, new_lengths)
                if ret != 0:
                    for address in acquired:
                        entry = self._push_source_registrations[address]
                        entry.users -= 1
                        if entry.users == 0:
                            del self._push_source_registrations[address]
                    raise RuntimeError("Mooncake EC source registration failed")
        return acquired

    def _release_push_source_registrations(self, addresses: list[int]) -> bool:
        if not addresses:
            return True
        with self._push_source_registration_lock:
            unused = []
            for address in addresses:
                entry = self._push_source_registrations.get(address)
                if entry is None:
                    continue
                entry.users -= 1
                if entry.users == 0:
                    unused.append(address)
            if not unused:
                return True
            ret = self._ensure_engine().batch_unregister_memory(unused)
            if ret != 0:
                logger.warning(
                    "Keeping %d EC source tensors registered after Mooncake "
                    "unregistration failure",
                    len(unused),
                )
                return False
            for address in unused:
                del self._push_source_registrations[address]
                self._pending_unregister.pop(address, None)
            return True

    def _ensure_consumer_pool(
        self, device: torch.device, *, allow_host: bool = False
    ) -> None:
        if (
            self._consumer_pool is not None
            or self._consumer_pool_disabled
            or (device.type != "cuda" and not allow_host)
        ):
            return
        try:
            pool = torch.empty(
                self._consumer_pool_capacity, dtype=torch.uint8, device=device
            )
            if self._is_receiving_rank:
                # Producers write into this pool directly, so it needs a memory
                # region. Later pipeline stages never receive and skip it.
                ret = self._ensure_engine().batch_register_memory(
                    [pool.data_ptr()], [pool.nbytes]
                )
                if ret != 0:
                    raise RuntimeError(f"Mooncake returned {ret}")
        except (RuntimeError, torch.OutOfMemoryError) as e:
            self._consumer_pool_disabled = True
            logger.warning(
                "Could not initialize the EC consumer buffer pool; falling back "
                "to per-tensor registration: %s",
                e,
            )
            return
        self._consumer_pool = pool
        self._consumer_pool_allocator = _ContiguousAllocator(pool.nbytes)
        logger.info(
            "Prepared %d-byte CUDA receive pool for Mooncake EC (registered=%s)",
            pool.nbytes,
            self._is_receiving_rank,
        )

    def _ensure_producer_pool(self, device: torch.device) -> None:
        """Register one staging slab so pushes never register per transfer.

        Registering the encoder output itself costs more than the transfer
        (register+unregister dominated the push path); staging into a slab
        that is registered once trades that for a device-to-device copy.
        """
        if self._producer_pool is not None or self._producer_pool_disabled:
            return
        with self._producer_pool_lock:
            if self._producer_pool is not None or self._producer_pool_disabled:
                return
            try:
                pool = torch.empty(
                    self._producer_pool_capacity, dtype=torch.uint8, device=device
                )
                ret = self._ensure_engine().batch_register_memory(
                    [pool.data_ptr()], [pool.nbytes]
                )
                if ret != 0:
                    raise RuntimeError(f"Mooncake returned {ret}")
            except (RuntimeError, torch.OutOfMemoryError) as e:
                self._producer_pool_disabled = True
                logger.warning(
                    "Could not initialize the EC producer staging pool; falling "
                    "back to per-transfer registration: %s",
                    e,
                )
                return
            self._producer_pool = pool
            self._producer_pool_allocator = _ContiguousAllocator(pool.nbytes)
            logger.info(
                "Registered %d-byte staging pool for Mooncake EC pushes",
                pool.nbytes,
            )

    def _stage_push_sources(
        self, tensors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]] | None:
        """Copy the batch into the staging pool; None if it does not fit."""
        if not tensors:
            return [], []
        self._ensure_producer_pool(tensors[0].device)
        pool = self._producer_pool
        allocator = self._producer_pool_allocator
        if pool is None or allocator is None:
            return None
        staged: list[torch.Tensor] = []
        regions: list[tuple[int, int]] = []
        with self._producer_pool_lock:
            for tensor in tensors:
                region = allocator.allocate(tensor.nbytes)
                if region is None:
                    for offset, size in regions:
                        allocator.free(offset, size)
                    return None
                regions.append(region)
                offset = region[0]
                staged.append(
                    pool.narrow(0, offset, tensor.nbytes)
                    .view(tensor.dtype)
                    .view(tensor.shape)
                )
        for destination, source in zip(staged, tensors):
            destination.copy_(source, non_blocking=True)
        return staged, regions

    def _release_push_staging(self, regions: list[tuple[int, int]]) -> None:
        allocator = self._producer_pool_allocator
        if allocator is None or not regions:
            return
        with self._producer_pool_lock:
            for offset, size in regions:
                allocator.free(offset, size)

    def _poll_consumer_pool_frees(self) -> None:
        allocator = self._consumer_pool_allocator
        if allocator is None:
            return
        with self._consumer_lock:
            pending = []
            for event, allocation in self._consumer_pending_frees:
                if event.query():
                    allocator.free(allocation.offset, allocation.size)
                else:
                    pending.append((event, allocation))
            self._consumer_pending_frees = pending

    def _reclaim_residents_locked(
        self, allocator: _ContiguousAllocator, nbytes: int
    ) -> tuple[int, int] | None:
        """Give up retired items, oldest first, until `nbytes` fits.

        Called only when the pool cannot satisfy an allocation, so a retired
        item survives until its memory is genuinely needed.
        """

        def evict(mm_hash: str, allocation: _ConsumerPoolAllocation) -> bool:
            event = self._consumer_retire_events.pop(mm_hash, None)
            if event is None or event.query():
                allocator.free(allocation.offset, allocation.size)
            else:
                self._consumer_pending_frees.append((event, allocation))
            self._consumer_reclaimed.add(mm_hash)
            self._consumer_worker_metrics["residents_reclaimed"] += 1
            return True

        while self._consumer_residents.evict_lru(evict) is not None:
            region = allocator.allocate(nbytes)
            if region is not None:
                return region
        return None

    def _take_resident_tensor(self, spec: ECMooncakeLoadSpec) -> torch.Tensor | None:
        """Hand back a copy the pool still holds.

        Retired and in-use entries live in the same map, so an item a later
        push reserved again still serves this load.
        """
        with self._consumer_lock:
            allocation = self._consumer_residents.get(spec.mm_hash)
            if allocation is None:
                self._consumer_worker_metrics["residents_missed"] += 1
                return None
            tensor = allocation.tensor
            if (
                tuple(tensor.shape) != tuple(spec.shape)
                or str(tensor.dtype).split(".")[-1] != spec.dtype
            ):
                self._consumer_worker_metrics["residents_mismatched"] += 1
                return None
            self._consumer_residents.pin(spec.mm_hash)
            self._consumer_retire_events.pop(spec.mm_hash, None)
            self._consumer_worker_metrics["residents_promoted"] += 1
            return tensor

    def _release_stale_consumer_allocations(
        self, encoder_cache: dict[str, torch.Tensor]
    ) -> None:
        if self._consumer_pool is None:
            return
        with self._consumer_lock:
            reserved_allocations = {
                id(reservation.allocation)
                for reservation in self._push_reservations.values()
            }
            # Walk only the referenced entries: the retired set grows to
            # thousands and none of it can change state here.
            for mm_hash in self._consumer_residents.referenced():
                allocation = self._consumer_residents.get(mm_hash)
                if allocation is None:
                    continue
                if encoder_cache.get(mm_hash) is allocation.tensor:
                    continue
                if id(allocation) in reserved_allocations:
                    continue
                # Retire rather than free: the bytes stay valid and serve the
                # next request that needs this item. The event orders the
                # eventual reuse behind whatever still reads the tensor.
                event = torch.Event()
                event.record(
                    torch.accelerator.current_stream(self._consumer_pool.device)
                )
                self._consumer_retire_events[mm_hash] = event
                self._consumer_residents.retire(mm_hash)
                self._consumer_worker_metrics["residents_retired"] += 1
        self._poll_consumer_pool_frees()

    def _clear_item_timers(self, mm_hash: str) -> None:
        self._consumer_missing_since.pop(mm_hash, None)
        self._consumer_loading_since.pop(mm_hash, None)
        self._consumer_pending_since.pop(mm_hash, None)
        self._stalled_hashes.discard(mm_hash)

    def _note_awaiting_push(
        self,
        mm_hash: str,
        transfer_id: str | None = None,
        request_id: str | None = None,
    ) -> bool:
        """Wait for an item with nothing in flight, and give up on timeout.

        Nothing on this side can produce the item, so a push that never
        arrives would defer the request forever. Past the timeout the request
        is reported unavailable instead: the scheduler fails it with a
        retryable error and the caller can re-issue it, which re-runs the
        encode and produces a fresh transfer.

        Returns:
            True once this request has been given up on.
        """
        now = time.monotonic()
        since = self._consumer_missing_since.setdefault(mm_hash, now)
        self._consumer_scheduler_metrics["missing_event"] += 1
        elapsed = now - since
        if elapsed < self._push_wait_timeout:
            return False
        stale = mm_hash in self._stalled_hashes
        if request_id is not None:
            self._unavailable_requests.add(request_id)
            self._consumer_scheduler_metrics["given_up"] += 1
            # Start a fresh window: a re-issued request pushes this item again,
            # and it must be allowed to wait for that push rather than inherit
            # this one's deadline and be given up on immediately. Only the
            # deadline resets -- `_stalled_hashes` keeps the warning to one per
            # hash, while `given_up` counts every occurrence.
            self._consumer_missing_since.pop(mm_hash, None)
        if stale:
            return request_id is not None
        self._stalled_hashes.add(mm_hash)
        self._consumer_scheduler_metrics["stalled"] += 1
        # Ask the worker what it knows about this transfer: whether the
        # reservation exists at all separates "the producer never sent it"
        # from "it arrived and the scheduler missed it".
        reservation: Any = "unknown"
        if transfer_id and self._reservation_zmq_addr is not None:
            try:
                reservation = self._send_control(
                    self._reservation_zmq_addr,
                    {"op": "status", "transfer_id": transfer_id},
                )
            except Exception as e:  # noqa: BLE001 - diagnostic only
                reservation = f"status failed: {e}"
        logger.warning(
            "EC Mooncake waited %.1fs for a push of mm_hash=%s "
            "(transfer_id=%s) that never arrived; worker reservation=%s; "
            "requests needing it fail with a retryable error.",
            elapsed,
            mm_hash,
            transfer_id,
            reservation,
        )
        return request_id is not None

    def take_unavailable_requests(self) -> set[str]:
        given_up = self._unavailable_requests
        self._unavailable_requests = set()
        return given_up

    @staticmethod
    def _hash_samples(values: list[str], limit: int = 5) -> list[str]:
        return [value[:16] for value in values[:limit]]

    def _maybe_log_consumer_worker_metrics(self) -> None:
        now = time.monotonic()
        if (
            self._consumer_metrics_log_interval <= 0
            or now - self._consumer_metrics_started_at
            < self._consumer_metrics_log_interval
        ):
            return
        with self._consumer_lock:
            ready = [
                mm_hash
                for mm_hash, reservation in self._push_reservations.items()
                if reservation.ready
            ]
            pending = [
                mm_hash
                for mm_hash, reservation in self._push_reservations.items()
                if not reservation.ready
            ]
            metrics = dict(self._consumer_worker_metrics)
            self._consumer_worker_metrics.clear()
            residents = len(self._consumer_residents)
            live = len(self._consumer_residents.referenced())
            retired = self._consumer_residents.num_evictable
            pending_frees = len(self._consumer_pending_frees)
            oldest_reservation_ms = max(
                (
                    (now - reservation.created_at) * 1000
                    for reservation in self._push_reservations.values()
                ),
                default=0.0,
            )
        logger.info(
            "EC Mooncake consumer worker: lifecycle=%s, reservations_ready=%d, "
            "reservations_pending=%d, residents=%d, live=%d, retired=%d, "
            "pending_frees=%d, "
            "oldest_reservation_ms=%.1f, ready_hashes=%s, pending_hashes=%s",
            metrics,
            len(ready),
            len(pending),
            residents,
            live,
            retired,
            pending_frees,
            oldest_reservation_ms,
            self._hash_samples(ready),
            self._hash_samples(pending),
        )
        self._consumer_metrics_started_at = now

    def _maybe_log_consumer_scheduler_metrics(self) -> None:
        now = time.monotonic()
        if (
            self._consumer_metrics_log_interval <= 0
            or now - self._consumer_metrics_started_at
            < self._consumer_metrics_log_interval
        ):
            return
        missing = sorted(self._consumer_missing_since.items(), key=lambda item: item[1])
        loading = sorted(self._consumer_loading_since.items(), key=lambda item: item[1])
        pending = sorted(self._consumer_pending_since.items(), key=lambda item: item[1])
        oldest_missing_ms = round((now - missing[0][1]) * 1000, 1) if missing else 0.0
        oldest_loading_ms = round((now - loading[0][1]) * 1000, 1) if loading else 0.0
        oldest_pending_ms = round((now - pending[0][1]) * 1000, 1) if pending else 0.0
        logger.info(
            "EC Mooncake consumer scheduler: decisions=%s, ready=%d, loading=%d, "
            "resident=%d, pending_specs=%d, needs_load=%d, missing=%d, "
            "oldest_missing_ms=%.1f, oldest_loading_ms=%.1f, "
            "oldest_pending_ms=%.1f, missing_hashes=%s, loading_hashes=%s, "
            "pending_hashes=%s",
            dict(self._consumer_scheduler_metrics),
            len(self._ready_hashes),
            len(self._loading_hashes),
            len(self._resident_specs),
            len(self._pending_specs),
            len(self._mm_datas_need_loads),
            len(missing),
            oldest_missing_ms,
            oldest_loading_ms,
            oldest_pending_ms,
            self._hash_samples([mm_hash for mm_hash, _ in missing]),
            self._hash_samples([mm_hash for mm_hash, _ in loading]),
            self._hash_samples([mm_hash for mm_hash, _ in pending]),
        )
        self._consumer_scheduler_metrics.clear()
        self._consumer_metrics_started_at = now

    def _expire_push_reservations_locked(self) -> None:
        now = time.monotonic()
        allocator = self._consumer_pool_allocator
        assert allocator is not None
        for transfer_id, reservation in list(self._push_reservations.items()):
            if reservation.expires_at > now:
                continue
            if reservation.owns_allocation:
                allocator.free(
                    reservation.allocation.offset, reservation.allocation.size
                )
            self._push_reservations.pop(transfer_id)
            self._consumer_worker_metrics["reservations_expired"] += 1
        for transfer_id, expires_at in list(self._cancelled_transfers.items()):
            if expires_at <= now:
                self._cancelled_transfers.pop(transfer_id)

    def _expire_push_reservations(self) -> int:
        with self._consumer_lock:
            before = len(self._push_reservations)
            self._expire_push_reservations_locked()
            return before - len(self._push_reservations)

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        transfer_id = str(payload["transfer_id"])
        mm_hash = str(payload["mm_hash"])
        nbytes = int(payload["nbytes"])
        shape = tuple(int(value) for value in payload["shape"])
        dtype_name = str(payload["dtype"])
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype string: {dtype_name!r}")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if expected_nbytes != nbytes:
            raise ValueError("shape and dtype do not match nbytes")

        with self._consumer_lock:
            self._expire_push_reservations_locked()
            if transfer_id in self._cancelled_transfers:
                self._consumer_worker_metrics["reservations_cancelled_early"] += 1
                return {
                    "reservation_id": "",
                    "dst_session": "",
                    "dst_ptr": 0,
                    "nbytes": nbytes,
                    "write": False,
                    "ready": False,
                    "cancelled": True,
                }
            existing = self._push_reservations.get(transfer_id)
            if existing is not None:
                if (
                    existing.mm_hash != mm_hash
                    or existing.shape != shape
                    or existing.dtype != dtype_name
                ):
                    raise ValueError("conflicting reservation for transfer_id")
                reservation = existing
                should_write = False
                key = (
                    "reservations_reused_ready"
                    if existing.ready
                    else ("reservations_reused_pending")
                )
                self._consumer_worker_metrics[key] += 1
                if not existing.ready:
                    existing.expires_at = time.monotonic() + _LEASE_TTL_SECONDS
            else:
                cached = self._consumer_residents.get(mm_hash)
                if cached is not None:
                    if (
                        tuple(cached.tensor.shape) != shape
                        or cached.tensor.dtype != dtype
                    ):
                        raise ValueError("conflicting cached tensor for mm_hash")
                    reservation = _PushReservation(
                        mm_hash=mm_hash,
                        reservation_id=uuid.uuid4().hex,
                        allocation=cached,
                        shape=shape,
                        dtype=dtype_name,
                        ready=True,
                        owns_allocation=False,
                        expires_at=time.monotonic() + _LEASE_TTL_SECONDS,
                    )
                    should_write = False
                    # Live again: it must not be reclaimed under pressure.
                    self._consumer_residents.pin(mm_hash)
                    self._consumer_retire_events.pop(mm_hash, None)
                    self._consumer_worker_metrics["reservations_cached"] += 1
                else:
                    pool = self._consumer_pool
                    allocator = self._consumer_pool_allocator
                    assert pool is not None and allocator is not None
                    region = allocator.allocate(nbytes)
                    if region is None:
                        self._expire_push_reservations_locked()
                        region = allocator.allocate(nbytes)
                    if region is None:
                        region = self._reclaim_residents_locked(allocator, nbytes)
                    if region is None:
                        raise RuntimeError("EC consumer buffer pool is full")
                    offset, size = region
                    tensor = pool.narrow(0, offset, nbytes).view(dtype).view(shape)
                    allocation = _ConsumerPoolAllocation(offset, size, tensor)
                    reservation = _PushReservation(
                        mm_hash=mm_hash,
                        reservation_id=uuid.uuid4().hex,
                        allocation=allocation,
                        shape=shape,
                        dtype=dtype_name,
                        expires_at=time.monotonic() + _LEASE_TTL_SECONDS,
                    )
                    should_write = True
                    self._consumer_worker_metrics["reservations_created"] += 1
                self._push_reservations[transfer_id] = reservation

        eng = self._ensure_engine()
        return {
            "reservation_id": reservation.reservation_id,
            "dst_session": f"{self._hostname}:{eng.get_rpc_port()}",
            "dst_ptr": reservation.allocation.tensor.data_ptr(),
            "nbytes": reservation.allocation.tensor.nbytes,
            "write": should_write,
            "ready": reservation.ready,
            "cached": not reservation.owns_allocation,
        }

    def _push_status(self, transfer_id: str) -> dict[str, Any] | None:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if reservation is None:
                return None
            return {
                "mm_hash": reservation.mm_hash,
                "ready": reservation.ready,
                "reservation_id": reservation.reservation_id,
                "nbytes": reservation.allocation.tensor.nbytes,
                "shape": list(reservation.shape),
                "dtype": reservation.dtype,
            }

    def _complete_push(self, transfer_id: str, reservation_id: str) -> _PushCompletion:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if reservation is None or reservation.reservation_id != reservation_id:
                self._consumer_worker_metrics["completions_rejected"] += 1
                return _PushCompletion(False)
            if reservation.ready:
                self._consumer_worker_metrics["completions_repeated"] += 1
                return _PushCompletion(True)
            self._consumer_worker_metrics["completions_accepted"] += 1
            if reservation.discard_on_complete:
                allocator = self._consumer_pool_allocator
                assert allocator is not None
                self._push_reservations.pop(transfer_id)
                if reservation.owns_allocation:
                    allocator.free(
                        reservation.allocation.offset, reservation.allocation.size
                    )
                self._consumer_worker_metrics["reservations_discarded"] += 1
                return _PushCompletion(True)
            reservation.ready = True
            reservation.expires_at = time.monotonic() + _LEASE_TTL_SECONDS
            return _PushCompletion(True, became_ready=True)

    def _cancel_push(
        self, transfer_id: str, reservation_id: str, abandon: bool = False
    ) -> bool:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if (
                reservation is not None
                and reservation_id
                and reservation.reservation_id != reservation_id
            ):
                self._consumer_worker_metrics["cancellations_rejected"] += 1
                return False
            self._cancelled_transfers[transfer_id] = (
                time.monotonic() + _LEASE_TTL_SECONDS
            )
            if reservation is None:
                self._consumer_worker_metrics["cancellations_pre_reserved"] += 1
                return True
            allocator = self._consumer_pool_allocator
            assert allocator is not None
            if not reservation.ready and not abandon:
                reservation.discard_on_complete = True
                self._consumer_worker_metrics["cancellations_deferred"] += 1
                return True
            self._push_reservations.pop(transfer_id)
            if reservation.owns_allocation:
                allocator.free(
                    reservation.allocation.offset, reservation.allocation.size
                )
            self._consumer_worker_metrics["reservations_cancelled"] += 1
            return True

    def _take_pushed_tensor(
        self, spec: ECMooncakeLoadSpec
    ) -> tuple[torch.Tensor, _ConsumerPoolAllocation]:
        with self._consumer_lock:
            reservation = self._push_reservations.get(spec.transfer_id)
            # Not compared against `spec.reservation_id`: each shard mints its
            # own, while the spec carries the one from whichever shard's event
            # the scheduler observed. `transfer_id` is assigned per request
            # item and is already unique, and a stale reservation for a reused
            # one is rejected by `_reserve_push_destination`.
            if reservation is None or not reservation.ready:
                self._consumer_worker_metrics["takes_rejected"] += 1
                raise RuntimeError(
                    f"Pushed EC tensor is not ready for mm_hash={spec.mm_hash}"
                )
            self._push_reservations.pop(spec.transfer_id)
            self._consumer_residents.insert(
                spec.mm_hash, reservation.allocation, reservation.allocation.size
            )
            self._consumer_worker_metrics["reservations_taken"] += 1
            return reservation.allocation.tensor, reservation.allocation

    def _send_control(self, addr: str, request: dict[str, Any]) -> Any:
        return self._control_channel.request(addr, request)

    def _shard_executor(self) -> ThreadPoolExecutor:
        """Threads for the extra shards of a sharded consumer.

        Reserving and writing both fan out from a task that already holds a
        worker of the control or transfer pool, so the extra shards need a
        pool of their own: queueing them behind their own caller deadlocks as
        soon as every worker there is waiting. Nothing submitted here fans out
        again, so this pool cannot deadlock on itself.
        """
        with self._shard_pool_lock:
            if self._shard_pool is None:
                self._shard_pool = ThreadPoolExecutor(
                    max_workers=32, thread_name_prefix="ec-mooncake-shard"
                )
            return self._shard_pool

    def _consumer_shards(self, base_addr: str) -> list[str]:
        """Every control channel of the consumer reachable at `base_addr`.

        A tensor-parallel consumer gathers from each rank's own cache, so each
        rank receives its own copy. Asking the first one for the roster keeps
        the address list out of the request and the proxy configuration.
        """
        cached = self._consumer_shard_cache.get(base_addr)
        if cached is not None:
            return cached
        shards = [base_addr]
        try:
            reply = self._send_control(base_addr, {"op": "peers"})
            ports = reply.get("ports") if isinstance(reply, dict) else None
            if ports:
                prefix = base_addr.rsplit(":", 1)[0]
                shards = [f"{prefix}:{int(port)}" for port in ports]
        except Exception:
            # An older consumer does not answer this, and it can only be
            # unsharded, so its single address is the whole roster.
            logger.warning(
                "EC Mooncake consumer at %s did not report its shards; "
                "assuming it is unsharded.",
                base_addr,
                exc_info=True,
            )
        self._consumer_shard_cache[base_addr] = shards
        if len(shards) > 1:
            logger.info(
                "EC Mooncake consumer at %s has %d shards", base_addr, len(shards)
            )
        return shards

    def _reserve_one(self, addr: str, spec: ECMooncakePushSpec) -> dict[str, Any]:
        result = self._send_control(
            addr,
            {
                "op": "reserve",
                "transfer_id": spec.transfer_id,
                "mm_hash": spec.mm_hash,
                "nbytes": spec.nbytes,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("Invalid EC reservation response")
        result["_received_at"] = time.monotonic()
        result["addr"] = addr
        return result

    def _reserve_remote(self, spec: ECMooncakePushSpec) -> list[dict[str, Any]]:
        """Reserve a destination on every shard of the consumer."""
        shards = self._consumer_shards(spec.consumer_zmq)
        if len(shards) == 1:
            return [self._reserve_one(shards[0], spec)]
        # This already runs on the control pool, so the extra shards go to the
        # fan-out pool: queueing them behind their own caller would deadlock
        # once every control worker is holding a reservation.
        extra = [
            self._shard_executor().submit(self._reserve_one, addr, spec)
            for addr in shards[1:]
        ]
        return [self._reserve_one(shards[0], spec)] + [f.result() for f in extra]

    def _cancel_remote(
        self, consumer_zmq: str, transfer_id: str, reservation_id: str
    ) -> bool:
        """Release this transfer on every shard that reserved for it.

        A sharded consumer holds one reservation per rank, so cancelling only
        the first would leave the rest pinning pool slots until they expire.
        """
        cancelled = False
        for addr in self._consumer_shards(consumer_zmq):
            result = self._send_control(
                addr,
                {
                    "op": "cancel",
                    "transfer_id": transfer_id,
                    "reservation_id": reservation_id,
                },
            )
            cancelled |= isinstance(result, dict) and bool(result.get("cancelled"))
        return cancelled

    def _poll_pending_cancels(self) -> None:
        pending = {}
        for transfer_id, future in self._pending_cancels.items():
            if not future.done():
                pending[transfer_id] = future
                continue
            try:
                cancelled = future.result()
            except Exception:
                self._cancelled_transfer_ids.discard(transfer_id)
                self._consumer_scheduler_metrics["cancellations_failed"] += 1
                logger.warning(
                    "EC Mooncake reservation cancellation failed", exc_info=True
                )
            else:
                key = "cancellations_completed" if cancelled else "cancellations_stale"
                self._consumer_scheduler_metrics[key] += 1
        self._pending_cancels = pending

    def start_save_caches(self, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        for spec in metadata.pushes:
            reservation = self._control_executor.submit(self._reserve_remote, spec)
            self._pending_reservations.setdefault(spec.mm_hash, deque()).append(
                (spec, reservation)
            )
        encoder_cache = kwargs.get("encoder_cache")
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._submit_reserved_pushes(tensor, mm_hash)

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Reached on steps with no work, from a stage that never gathers
            # multimodal embeddings. Taking a transfer here would fail for
            # want of a reservation and fail the load for everyone.
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        self._ensure_engine()
        raw_buf = self._ec_cfg.ec_buffer_device
        buf = raw_buf.lower() if isinstance(raw_buf, str) and raw_buf else "cuda"
        if buf == "cuda" and not torch.accelerator.is_available():
            raise RuntimeError(
                "ECMooncakeConnector requires CUDA for ec_buffer_device=cuda"
            )
        self._release_stale_consumer_allocations(encoder_cache)

        for spec in metadata.loads:
            if spec.mm_hash in encoder_cache:
                if spec.pushed:
                    # The spec's id is one shard's; cancel by transfer.
                    self._cancel_push(spec.transfer_id, "")
                self._completed_loads.add(spec.mm_hash)
                continue
            if spec.local:
                resident = self._take_resident_tensor(spec)
                if resident is None:
                    # Reclaimed before the scheduler heard about it; the load
                    # falls back to a transfer on a later step.
                    self._failed_loads.add(spec.mm_hash)
                else:
                    encoder_cache[spec.mm_hash] = resident
                    self._completed_loads.add(spec.mm_hash)
                continue
            if spec.pushed:
                try:
                    pushed_tensor, _ = self._take_pushed_tensor(spec)
                except RuntimeError as e:
                    logger.warning("EC Mooncake pushed load failed: %s", e)
                    self._failed_loads.add(spec.mm_hash)
                    continue
                encoder_cache[spec.mm_hash] = pushed_tensor
                self._completed_loads.add(spec.mm_hash)
                continue
            logger.warning(
                "EC Mooncake load for mm_hash=%s has no transfer to take",
                spec.mm_hash,
            )
            self._failed_loads.add(spec.mm_hash)

    def _push_batch(self, pushes: list[_PendingPush]) -> None:
        started_at = time.monotonic()
        with self._push_perf_lock:
            self._queued_transfer_batches -= 1
            self._active_transfer_batches += 1

        queue_waits_ms = [
            max(0, started_at - push.enqueued_at) * 1000 for push in pushes
        ]
        stage_ms = {
            "queue": sum(queue_waits_ms),
            "reserve": 0.0,
            "cuda": 0.0,
            "register": 0.0,
            "rdma": 0.0,
            "unregister": 0.0,
            "complete": 0.0,
        }
        ready: list[tuple[_PendingPush, dict[str, Any]]] = []
        notifications: list[tuple[_PendingPush, dict[str, Any]]] = []
        failed = False
        try:
            synchronized: set[int] = set()
            for push in pushes:
                stage_started_at = time.monotonic()
                reservations = push.reservation.result()
                stale = [
                    index
                    for index, shard in enumerate(reservations)
                    if not shard.get("ready", False)
                    and time.monotonic() - float(shard.get("_received_at", started_at))
                    >= _RESERVATION_REFRESH_SECONDS
                ]
                if stale:
                    reservations = self._reserve_remote(push.spec)
                stage_ms["reserve"] += (time.monotonic() - stage_started_at) * 1000
                for shard in reservations:
                    if shard.get("cached", False) or shard.get("cancelled", False):
                        continue
                    if not shard.get("write", True):
                        continue
                    if push.ready_event is not None and id(push) not in synchronized:
                        stage_started_at = time.monotonic()
                        push.ready_event.synchronize()
                        stage_ms["cuda"] += (time.monotonic() - stage_started_at) * 1000
                        synchronized.add(id(push))
                    if int(shard["nbytes"]) != push.tensor.nbytes:
                        raise RuntimeError(
                            "Reserved EC size does not match tensor for "
                            f"mm_hash={push.spec.mm_hash}"
                        )
                    ready.append((push, shard))
                    notifications.append((push, shard))
            if not ready and not notifications:
                return

            if ready:
                eng = self._ensure_engine()
                # One source per push: a sharded consumer reads the same bytes
                # into each of its ranks, so staging and registration happen
                # once however many destinations there are.
                unique: list[_PendingPush] = []
                source_index: dict[int, int] = {}
                for push, _ in ready:
                    if id(push) not in source_index:
                        source_index[id(push)] = len(unique)
                        unique.append(push)
                tensors = [push.tensor for push in unique]
                lengths = [tensor.nbytes for tensor in tensors]
                stage_started_at = time.monotonic()
                staged = self._stage_push_sources(tensors)
                registered_sources: list[int] = []
                staged_regions: list[tuple[int, int]] = []
                if staged is not None:
                    sources, staged_regions = staged
                    # The NIC reads outside the CUDA stream, so the staging
                    # copies have to have landed before the transfer starts.
                    if sources and sources[0].device.type == "cuda":
                        torch.accelerator.current_stream(
                            sources[0].device
                        ).synchronize()
                else:
                    sources = tensors
                    registered_sources = self._acquire_push_source_registrations(
                        tensors
                    )
                addresses = [tensor.data_ptr() for tensor in sources]
                stage_ms["register"] = (time.monotonic() - stage_started_at) * 1000
                try:
                    by_session: dict[str, list[tuple[int, int]]] = {}
                    for push, shard in ready:
                        by_session.setdefault(str(shard["dst_session"]), []).append(
                            (source_index[id(push)], int(shard["dst_ptr"]))
                        )
                    stage_started_at = time.monotonic()

                    def write(session: str, items: list[tuple[int, int]]) -> None:
                        ret = eng.batch_transfer_sync_write(
                            session,
                            [addresses[index] for index, _ in items],
                            [dst for _, dst in items],
                            [lengths[index] for index, _ in items],
                        )
                        if ret != 0:
                            raise RuntimeError(
                                f"Mooncake EC push to {session} failed with "
                                f"status {ret}"
                            )

                    sessions = list(by_session.items())
                    # Shards are written concurrently: serialising them would
                    # make the transfer cost the sum of the ranks instead of
                    # the slowest one.
                    extra = [
                        self._shard_executor().submit(write, session, items)
                        for session, items in sessions[1:]
                    ]
                    try:
                        write(*sessions[0])
                    finally:
                        for future in extra:
                            future.result()
                    stage_ms["rdma"] = (time.monotonic() - stage_started_at) * 1000
                finally:
                    stage_started_at = time.monotonic()
                    self._release_push_staging(staged_regions)
                    self._release_push_source_registrations(registered_sources)
                    stage_ms["unregister"] = (
                        time.monotonic() - stage_started_at
                    ) * 1000

            stage_started_at = time.monotonic()
            self._notify_completions(notifications)
            stage_ms["complete"] = (time.monotonic() - stage_started_at) * 1000
        except Exception:
            # A failed batch must not take the engine down with it: the
            # consumer is told to drop its reservations and this item falls
            # back to whatever the consumer can still do (pull, or a local
            # re-encode). Raising here would surface in
            # `build_connector_worker_meta` as a fatal EngineCore error.
            failed = True
            logger.exception(
                "EC Mooncake push batch failed for mm_hashes=%s",
                [push.spec.mm_hash for push in pushes],
            )
            self._abandon_pushes(pushes)
        finally:
            with self._active_push_sources_lock:
                for push in pushes:
                    key = (push.spec.mm_hash, id(push.tensor))
                    self._active_push_sources[key] -= 1
                    if self._active_push_sources[key] == 0:
                        del self._active_push_sources[key]
            stage_ms["total"] = (time.monotonic() - started_at) * 1000
            self._record_push_perf(
                stage_ms,
                stage_max_ms={"queue": max(queue_waits_ms, default=0.0)},
                item_count=len(pushes),
                # `ready` holds one entry per destination shard, so count the
                # distinct items rather than the writes.
                byte_count=sum(
                    push.tensor.nbytes for push in {id(p): p for p, _ in ready}.values()
                ),
                skipped_items=len(pushes) - len({id(push) for push, _ in ready}),
                failed=failed,
            )

    def _notify_completions(
        self, notifications: list[tuple[_PendingPush, dict[str, Any]]]
    ) -> None:
        """Tell the consumer, in one message per destination, what landed."""
        if not notifications:
            return
        by_destination: dict[str, list[tuple[_PendingPush, dict[str, Any]]]] = {}
        for push, reservation in notifications:
            by_destination.setdefault(
                str(reservation.get("addr", push.spec.consumer_zmq)), []
            ).append((push, reservation))
        for consumer_zmq, items in by_destination.items():
            result = self._send_control(
                consumer_zmq,
                {
                    "op": "complete_batch",
                    "items": [
                        {
                            "transfer_id": push.spec.transfer_id,
                            "reservation_id": reservation["reservation_id"],
                        }
                        for push, reservation in items
                    ],
                },
            )
            completions = result.get("items", []) if isinstance(result, dict) else []
            if len(completions) != len(items):
                raise RuntimeError("Malformed EC completion response")
            for (push, _), completion in zip(items, completions):
                if not completion.get("completed"):
                    raise RuntimeError(
                        f"Unknown EC reservation for mm_hash={push.spec.mm_hash}"
                    )

    def _abandon_pushes(self, pushes: list[_PendingPush]) -> None:
        """Release the consumer-side reservations of a batch that failed."""
        for push in pushes:
            shards: list[dict[str, Any]] = []
            if push.reservation.done() and not push.reservation.cancelled():
                with suppress(Exception):
                    shards = push.reservation.result()
            if not shards:
                shards = [{"addr": push.spec.consumer_zmq, "reservation_id": ""}]
            for shard in shards:
                with suppress(Exception):
                    self._send_control(
                        str(shard.get("addr", push.spec.consumer_zmq)),
                        {
                            "op": "cancel",
                            "transfer_id": push.spec.transfer_id,
                            "reservation_id": str(shard.get("reservation_id", "")),
                            "abandon": True,
                        },
                    )

    def _record_push_perf(
        self,
        stage_ms: dict[str, float],
        *,
        stage_max_ms: dict[str, float],
        item_count: int,
        byte_count: int,
        skipped_items: int,
        failed: bool,
    ) -> None:
        now = time.monotonic()
        report: tuple[_PushPerfWindow, int, int] | None = None
        with self._push_perf_lock:
            self._active_transfer_batches -= 1
            perf = self._push_perf
            perf.batches += 1
            perf.items += item_count
            perf.bytes += byte_count
            perf.skipped_items += skipped_items
            perf.failures += int(failed)
            for stage, elapsed_ms in stage_ms.items():
                perf.stage_totals_ms[stage] = (
                    perf.stage_totals_ms.get(stage, 0.0) + elapsed_ms
                )
                perf.stage_max_ms[stage] = max(
                    perf.stage_max_ms.get(stage, 0.0),
                    stage_max_ms.get(stage, elapsed_ms),
                )
            if (
                self._transfer_metrics_log_interval > 0
                and now - perf.started_at >= self._transfer_metrics_log_interval
            ):
                report = (
                    perf,
                    self._active_transfer_batches,
                    self._queued_transfer_batches,
                )
                self._push_perf = _PushPerfWindow(started_at=now)
        if report is None:
            return
        perf, active_batches, queued_batches = report
        batches = max(perf.batches, 1)
        items = max(perf.items, 1)
        stage_parts = []
        for stage in (
            "queue",
            "reserve",
            "cuda",
            "register",
            "rdma",
            "unregister",
            "complete",
            "total",
        ):
            divisor = items if stage == "queue" else batches
            average = perf.stage_totals_ms.get(stage, 0.0) / divisor
            maximum = perf.stage_max_ms.get(stage, 0.0)
            stage_parts.append(f"{stage}_ms={average:.1f}/{maximum:.1f}")
        stage_summary = " ".join(stage_parts)
        producer_metrics = dict(self._producer_metrics)
        self._producer_metrics.clear()
        logger.info(
            "EC Mooncake push perf: batches=%d items=%d bytes=%d "
            "batch_items=%.1f skipped=%d failures=%d active=%d queued=%d "
            "producer=%s queue_item_avg/max and stage_batch_avg/max: %s",
            perf.batches,
            perf.items,
            perf.bytes,
            perf.items / batches,
            perf.skipped_items,
            perf.failures,
            active_batches,
            queued_batches,
            producer_metrics,
            stage_summary,
        )

    def _flush_pending_pushes(self) -> None:
        if not self._pending_pushes:
            return
        grouped: dict[str, list[_PendingPush]] = {}
        for push in self._pending_pushes:
            grouped.setdefault(push.spec.consumer_zmq, []).append(push)
        self._pending_pushes = []
        for pushes in grouped.values():
            with self._push_perf_lock:
                self._queued_transfer_batches += 1
            future = self._io_executor.submit(self._push_batch, pushes)
            hashes = ",".join(push.spec.mm_hash for push in pushes)
            self._pending_saves.append((hashes, future))

    def _submit_push(
        self,
        tensor: torch.Tensor,
        spec: ECMooncakePushSpec,
        reservation: Future[list[dict[str, Any]]],
    ) -> None:
        ready_event = None
        if tensor.device.type == "cuda":
            ready_event = torch.Event()
            ready_event.record(torch.accelerator.current_stream(tensor.device))
        self._pending_pushes.append(
            _PendingPush(
                tensor=tensor,
                spec=spec,
                reservation=reservation,
                ready_event=ready_event,
                enqueued_at=time.monotonic(),
            )
        )

    def _submit_reserved_pushes(self, tensor: torch.Tensor, mm_hash: str) -> None:
        reservations = self._pending_reservations.pop(mm_hash, deque())
        if reservations:
            with self._active_push_sources_lock:
                self._active_push_sources[(mm_hash, id(tensor))] += len(reservations)
        for spec, reservation in reservations:
            self._submit_push(tensor, spec, reservation)

    def _cancel_orphaned_reservation(
        self,
        spec: ECMooncakePushSpec,
        reservation: Future[list[dict[str, Any]]],
    ) -> None:
        try:
            for shard in reservation.result():
                if shard.get("cached", False) or shard.get("cancelled", False):
                    continue
                self._send_control(
                    str(shard.get("addr", spec.consumer_zmq)),
                    {
                        "op": "cancel",
                        "transfer_id": spec.transfer_id,
                        "reservation_id": str(shard.get("reservation_id", "")),
                        "abandon": True,
                    },
                )
        except Exception:
            logger.exception(
                "Failed to cancel orphaned EC reservation for transfer_id=%s",
                spec.transfer_id,
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_producer or self._role != ECConnectorRole.WORKER:
            return None, None

        Reserved = tuple[ECMooncakePushSpec, Future[list[dict[str, Any]]]]
        orphaned: list[Reserved] = []
        for mm_hash, reservations in list(self._pending_reservations.items()):
            remaining: deque[Reserved] = deque()
            for spec, reservation in reservations:
                if spec.request_id in finished_req_ids:
                    orphaned.append((spec, reservation))
                else:
                    remaining.append((spec, reservation))
            if remaining:
                self._pending_reservations[mm_hash] = remaining
            else:
                self._pending_reservations.pop(mm_hash)

        for spec, reservation in orphaned:
            future = self._io_executor.submit(
                self._cancel_orphaned_reservation, spec, reservation
            )
            self._pending_saves.append((f"cancel:{spec.transfer_id}", future))
        return None, None

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        if not self.is_producer or self._role != ECConnectorRole.WORKER:
            return
        tensor = encoder_cache[mm_hash]
        if mm_hash in self._pending_reservations:
            self._submit_reserved_pushes(tensor, mm_hash)

    def _index_pending_spec(self, spec: ECMooncakeLoadSpec) -> None:
        transfer_id = spec.transfer_id or spec.mm_hash
        if transfer_id in self._pending_specs:
            self._consumer_scheduler_metrics["events_duplicate"] += 1
            return
        self._pending_specs[transfer_id] = spec
        self._pending_specs_by_hash.setdefault(spec.mm_hash, deque()).append(
            transfer_id
        )
        self._pending_spec_deadlines[transfer_id] = (
            time.monotonic() + _LEASE_TTL_SECONDS
        )
        self._consumer_missing_since.pop(spec.mm_hash, None)
        self._consumer_pending_since.setdefault(spec.mm_hash, time.monotonic())

    def _pop_pending_spec(self, transfer_id: str) -> ECMooncakeLoadSpec | None:
        spec = self._pending_specs.pop(transfer_id, None)
        self._pending_spec_deadlines.pop(transfer_id, None)
        self._forget_shard_readiness(transfer_id)
        if spec is not None:
            if not self._pending_specs_by_hash.get(spec.mm_hash):
                self._consumer_pending_since.pop(spec.mm_hash, None)
            transfer_ids = self._pending_specs_by_hash.get(spec.mm_hash)
            if transfer_ids is not None:
                with suppress(ValueError):
                    transfer_ids.remove(transfer_id)
                if not transfer_ids:
                    self._pending_specs_by_hash.pop(spec.mm_hash, None)
                    self._consumer_pending_since.pop(spec.mm_hash, None)
            else:
                self._consumer_pending_since.pop(spec.mm_hash, None)
        return spec

    def _first_pending_spec(self, mm_hash: str) -> ECMooncakeLoadSpec | None:
        transfer_ids = self._pending_specs_by_hash.get(mm_hash)
        if transfer_ids is None:
            return None
        while transfer_ids:
            spec = self._pending_specs.get(transfer_ids[0])
            if spec is not None:
                return spec
            transfer_ids.popleft()
        self._pending_specs_by_hash.pop(mm_hash, None)
        self._consumer_pending_since.pop(mm_hash, None)
        return None

    def _note_shard_ready(self, data: dict[str, Any]) -> bool:
        """Whether every consumer shard has now reported this transfer ready.

        Loading before the last rank has its copy makes that rank miss, which
        `ECMooncakeWorkerMetadata.aggregate` catches by intersecting `loaded`
        across ranks -- at the cost of rescheduling the whole load.
        """
        if self._event_shard_count <= 1:
            return True
        transfer_id = str(data["transfer_id"])
        if transfer_id in self._pending_specs:
            # Already indexed; later shards are just confirmations.
            return False
        shard = data.get("shard")
        shards = self._event_ready_shards.setdefault(transfer_id, set())
        self._event_ready_shards.move_to_end(transfer_id)
        shards.add(int(shard) if shard is not None else len(shards))
        if len(shards) < self._event_shard_count:
            self._consumer_scheduler_metrics["events_awaiting_shards"] += 1
            while len(self._event_ready_shards) > _MAX_PENDING_EVENTS:
                self._event_ready_shards.popitem(last=False)
                self._consumer_scheduler_metrics["events_partial_dropped"] += 1
            return False
        self._event_ready_shards.pop(transfer_id, None)
        self._consumer_scheduler_metrics["events_all_shards_ready"] += 1
        return True

    def _forget_shard_readiness(self, transfer_id: str) -> None:
        self._event_ready_shards.pop(transfer_id, None)

    def _store_pushed_spec(self, data: dict[str, Any]) -> None:
        transfer_id = str(data["transfer_id"])
        identifier = str(data["mm_hash"])
        reservation_id = str(data["reservation_id"])
        self._index_pending_spec(
            ECMooncakeLoadSpec(
                mm_hash=identifier,
                num_token=0,
                nbytes=int(data["nbytes"]),
                shape=tuple(int(value) for value in data["shape"]),
                dtype=str(data["dtype"]),
                pushed=True,
                transfer_id=transfer_id,
                reservation_id=reservation_id,
            )
        )

    def _note_resident(self, spec: ECMooncakeLoadSpec) -> None:
        """Record that the worker's receive pool now holds this item."""
        self._drop_resident(spec.mm_hash)
        self._resident_specs[spec.mm_hash] = ECMooncakeLoadSpec(
            mm_hash=spec.mm_hash,
            num_token=0,
            nbytes=spec.nbytes,
            shape=spec.shape,
            dtype=spec.dtype,
            local=True,
        )
        self._resident_bytes += spec.nbytes
        while (
            self._resident_specs and self._resident_bytes > self._consumer_pool_capacity
        ):
            _, dropped = self._resident_specs.popitem(last=False)
            self._resident_bytes -= dropped.nbytes

    def _drop_resident(self, mm_hash: str) -> None:
        spec = self._resident_specs.pop(mm_hash, None)
        if spec is not None:
            self._resident_bytes -= spec.nbytes

    def _queue_cancel(self, transfer_id: str, reservation_id: str = "") -> None:
        if (
            self._reservation_zmq_addr is None
            or transfer_id in self._pending_cancels
            or transfer_id in self._cancelled_transfer_ids
        ):
            return
        self._cancelled_transfer_ids.add(transfer_id)
        self._pending_cancels[transfer_id] = self._control_executor.submit(
            self._cancel_remote,
            self._reservation_zmq_addr,
            transfer_id,
            reservation_id,
        )

    def _expire_pending_specs(self) -> None:
        now = time.monotonic()
        for transfer_id, deadline in list(self._pending_spec_deadlines.items()):
            if deadline > now:
                continue
            spec = self._pop_pending_spec(transfer_id)
            if spec is not None:
                self._consumer_pending_since.pop(spec.mm_hash, None)
                self._consumer_scheduler_metrics["pending_specs_expired"] += 1
                self._queue_cancel(transfer_id)

    def _ensure_event_channel(self) -> None:
        if self._event_zmq_socket is not None:
            return
        assert self._reservation_zmq_addr is not None
        shards = self._consumer_shards(self._reservation_zmq_addr)
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PULL)
        # One PULL fair-queues across every shard's PUSH. Subscribing to the
        # first shard alone leaves the others' notifications queued on their
        # side forever, and hides their readiness from the scheduler.
        connected = 0
        for addr in shards:
            try:
                event_port = self._send_control(addr, {"op": "event_port"})
                address, _ = addr.rsplit(":", 1)
                socket.connect(f"{address}:{int(event_port)}")
            except Exception:
                logger.warning(
                    "EC Mooncake could not subscribe to the event channel of "
                    "consumer shard %s; its readiness will only be seen "
                    "through reserve replies.",
                    addr,
                )
                continue
            connected += 1
        if not connected:
            socket.close(linger=0)
            ctx.term()
            return
        self._event_zmq_ctx = ctx
        self._event_zmq_socket = socket
        self._event_shard_count = connected

    def _drain_push_notifications(self) -> None:
        # `has_cache_item` and `ensure_cache_available` run once per request
        # per multimodal item, so draining on every call rescans the cancel
        # and deadline tables thousands of times per step. Once per step is
        # enough: `build_connector_meta` re-arms this at the end of each one.
        now = time.monotonic()
        if not self._drain_pending and now - self._drained_at < _DRAIN_MIN_INTERVAL:
            return
        self._drain_pending = False
        self._drained_at = now
        self._poll_pending_cancels()
        self._expire_pending_specs()
        if self._reservation_zmq_addr is not None:
            self._ensure_event_channel()
        socket = self._event_zmq_socket
        if socket is None:
            return
        while True:
            try:
                data = socket.recv_json(flags=zmq.DONTWAIT)
            except zmq.Again:
                return
            identifier = str(data["mm_hash"])
            self._consumer_scheduler_metrics["events_received"] += 1
            if data.get("ready"):
                self._consumer_scheduler_metrics["events_ready"] += 1
                if identifier in self._ready_hashes:
                    # Redundant only for as long as the hash stays ready; hold
                    # on to the spec so an eviction does not strand whoever
                    # this transfer belongs to.
                    self._consumer_scheduler_metrics["events_redundant"] += 1
                if not self._note_shard_ready(data):
                    continue
                self._store_pushed_spec(data)
            else:
                self._consumer_scheduler_metrics["events_not_ready"] += 1

    def has_cache_item(self, identifier: str) -> bool:
        if not self.is_consumer or self._role != ECConnectorRole.SCHEDULER:
            return False
        self._drain_push_notifications()
        self._maybe_log_consumer_scheduler_metrics()
        if identifier in self._ready_hashes:
            self._consumer_scheduler_metrics["ready"] += 1
            self._clear_item_timers(identifier)
            return True
        if identifier in self._loading_hashes:
            self._consumer_scheduler_metrics["loading"] += 1
            return False
        if identifier in self._resident_specs:
            self._consumer_scheduler_metrics["resident"] += 1
            self._consumer_missing_since.pop(identifier, None)
            return True
        pending = self._first_pending_spec(identifier)
        if pending is not None:
            self._consumer_scheduler_metrics["pending_spec"] += 1
            self._consumer_missing_since.pop(identifier, None)
            return True
        self._consumer_scheduler_metrics["missing_event"] += 1
        self._consumer_missing_since.setdefault(identifier, time.monotonic())
        return False

    @staticmethod
    def _request_transfer_id(request: Any, index: int) -> str | None:
        params = getattr(request, "ec_transfer_params", None) or {}
        items = params.get("ec_items") or []
        mm_hash = request.mm_features[index].identifier
        if index < len(items):
            item = items[index]
            if item.get("mm_hash") in (None, mm_hash) and item.get("transfer_id"):
                return str(item["transfer_id"])
        for item in items:
            if item.get("mm_hash") == mm_hash and item.get("transfer_id"):
                return str(item["transfer_id"])
        return None

    def ensure_cache_available(
        self,
        request: Any,
        num_computed_tokens: int,
        local_cache_hashes: Collection[str] | None = None,
    ) -> bool:
        if self.is_producer:
            for index, feature in enumerate(request.mm_features):
                if (
                    feature.mm_position.offset + feature.mm_position.length
                    > num_computed_tokens
                ):
                    self._prepare_push_spec(request, index)
        if not self.is_consumer or self._role != ECConnectorRole.SCHEDULER:
            return True

        self._drain_push_notifications()
        local_cache_hashes = local_cache_hashes or set()
        all_ready = True
        for index, feature in enumerate(request.mm_features):
            if (
                feature.mm_position.offset + feature.mm_position.length
                <= num_computed_tokens
            ):
                continue
            mm_hash = feature.identifier
            transfer_id = self._request_transfer_id(request, index)
            if transfer_id is not None and transfer_id in self._pending_spec_deadlines:
                # A live request still references this transfer, so keep it out
                # of the orphan sweep in `_expire_pending_specs`.
                self._pending_spec_deadlines[transfer_id] = (
                    time.monotonic() + _LEASE_TTL_SECONDS
                )
            if mm_hash in local_cache_hashes:
                # Keep the transfer: `local_cache_hashes` is a snapshot, and
                # the entry can be evicted before this request is scheduled.
                # Cancelling here used to strand the request with no way to
                # get the item back. `request_finished` releases it instead.
                continue
            if mm_hash in self._ready_hashes:
                self._consumer_scheduler_metrics["ready"] += 1
                self._clear_item_timers(mm_hash)
                continue
            if mm_hash in self._loading_hashes:
                self._consumer_scheduler_metrics["loading"] += 1
                all_ready = False
                continue
            # A resident copy is preferred over a transfer: it is already in
            # this instance's memory, and using it leaves the transfer for
            # whoever has no copy at all.
            spec = self._resident_specs.get(mm_hash)
            if spec is not None:
                self._consumer_scheduler_metrics["resident_hit"] += 1
            else:
                spec = (
                    self._pending_specs.get(transfer_id)
                    if transfer_id is not None
                    else None
                )
                if spec is None:
                    spec = self._first_pending_spec(mm_hash)
            if spec is not None:
                self._loading_hashes.add(mm_hash)
                self._load_specs[mm_hash] = spec
                self._consumer_loading_since.setdefault(mm_hash, time.monotonic())
                self._consumer_pending_since.pop(mm_hash, None)
                self._mm_datas_need_loads[mm_hash] = request.get_num_encoder_embeds(
                    index
                )
                self._scheduler_pending_work = True
                all_ready = False
            else:
                # Keep waiting until the timeout, then let the request fail
                # rather than hold a scheduler slot forever.
                self._note_awaiting_push(mm_hash, transfer_id, request.request_id)
                all_ready = False
        return all_ready

    def _prepare_push_spec(self, request: Any, index: int) -> None:
        params = getattr(request, "ec_transfer_params", None) or {}
        consumer_zmq = params.get("consumer_zmq")
        mm_hash = request.mm_features[index].identifier
        transfer_id = self._request_transfer_id(request, index)
        if transfer_id is None:
            transfer_id = f"{request.request_id}:{index}"
        if not consumer_zmq or transfer_id in self._pushes_to_prepare:
            return
        num_tokens = request.get_num_encoder_embeds(index)
        dtype = self._model_config.dtype
        assert isinstance(dtype, torch.dtype)
        dtype_name = str(dtype).split(".")[-1]
        shape = (num_tokens, self._model_config.get_hidden_size())
        nbytes = math.prod(shape) * dtype.itemsize
        self._pushes_to_prepare[transfer_id] = ECMooncakePushSpec(
            mm_hash=mm_hash,
            nbytes=nbytes,
            shape=shape,
            dtype=dtype_name,
            consumer_zmq=str(consumer_zmq),
            transfer_id=transfer_id,
            request_id=request.request_id,
        )

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        if self.is_producer:
            self._prepare_push_spec(request, index)
        if not self.is_consumer:
            return
        if mm_hash in self._ready_hashes:
            return
        if mm_hash in self._loading_hashes:
            return
        num_encoder_token = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def update_state_after_free(self, request: Any, index: int) -> None:
        """Release this request's transfer as soon as it consumed the item.

        Waiting for `request_finished` would keep a consumer buffer (and the
        pool slot its reservation pins) alive for the whole generation.
        """
        if not self.is_consumer or self._role != ECConnectorRole.SCHEDULER:
            return
        transfer_id = self._request_transfer_id(request, index)
        if transfer_id is None:
            return
        self._pop_pending_spec(transfer_id)
        self._queue_cancel(transfer_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self._ready_hashes.discard(mm_hash)
            self._clear_item_timers(mm_hash)
        meta = ECMooncakeConnectorMetadata()
        for push_spec in self._pushes_to_prepare.values():
            meta.add_push(push_spec)
        self._pushes_to_prepare.clear()
        for mm_hash, num_token in self._mm_datas_need_loads.items():
            load_spec = self._load_specs.pop(mm_hash, None)
            if load_spec is None:
                logger.warning("Missing EC Mooncake spec for mm_hash=%s", mm_hash)
                continue
            meta.add_load(
                ECMooncakeLoadSpec(
                    mm_hash=load_spec.mm_hash,
                    num_token=num_token,
                    nbytes=load_spec.nbytes,
                    shape=load_spec.shape,
                    dtype=load_spec.dtype,
                    pushed=load_spec.pushed,
                    transfer_id=load_spec.transfer_id,
                    reservation_id=load_spec.reservation_id,
                    local=load_spec.local,
                )
            )
            # Either way the pool holds the item once this load lands, so it
            # can serve the next request without another transfer.
            self._note_resident(load_spec)
            if not load_spec.local:
                self._pop_pending_spec(load_spec.transfer_id or load_spec.mm_hash)
        self._mm_datas_need_loads.clear()
        self._poll_pending_cancels()
        self._maybe_log_consumer_scheduler_metrics()
        self._drain_pending = True
        return meta

    def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
        if self._role != ECConnectorRole.WORKER:
            return None
        if self.is_consumer and not self._is_receiving_rank:
            # `loaded` is intersected across reporting ranks, so a stage that
            # never loads must not report at all rather than report nothing.
            return None

        self._flush_pending_pushes()
        saves = self._pending_saves
        completed_saves = []
        self._pending_saves = [
            (mm_hash, future) for mm_hash, future in saves if not future.done()
        ]
        for mm_hash, future in saves:
            if future.done():
                completed_saves.append((mm_hash, future))
        for mm_hash, future in completed_saves:
            try:
                future.result()
            except Exception:
                # Publishing is best-effort: a consumer that cannot fetch this
                # item falls back to encoding it locally. Failing the step
                # instead would take the whole engine down.
                self._producer_metrics["saves_failed"] += 1
                logger.exception(
                    "EC Mooncake async save failed for mm_hash=%s", mm_hash
                )
        with self._consumer_lock:
            reclaimed = self._consumer_reclaimed
            self._consumer_reclaimed = set()
        meta = ECMooncakeWorkerMetadata(
            loaded=self._completed_loads,
            failed_loads=self._failed_loads,
            reclaimed=reclaimed,
            pending_loads=False,
            pending_saves=bool(self._pending_saves),
        )
        self._completed_loads = set()
        self._failed_loads = set()
        if self.is_consumer:
            self._maybe_log_consumer_worker_metrics()
        return meta

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, ECMooncakeWorkerMetadata):
            return
        for mm_hash in meta.loaded:
            self._loading_hashes.discard(mm_hash)
            self._ready_hashes.add(mm_hash)
            self._clear_item_timers(mm_hash)
            self._consumer_scheduler_metrics["loads_completed"] += 1
        for mm_hash in meta.failed_loads:
            self._loading_hashes.discard(mm_hash)
            self._load_specs.pop(mm_hash, None)
            self._drop_resident(mm_hash)
            self._clear_item_timers(mm_hash)
            self._consumer_scheduler_metrics["loads_failed"] += 1
        for mm_hash in meta.reclaimed:
            self._drop_resident(mm_hash)
            self._consumer_scheduler_metrics["resident_reclaimed"] += 1
        self._scheduler_pending_work = meta.pending_loads or meta.pending_saves

    def has_pending_push_work(self) -> bool:
        return self._scheduler_pending_work

    def _placeholder_metadata_fields(self, modality: str) -> set[str]:
        if modality in self._metadata_fields_cache:
            return self._metadata_fields_cache[modality]

        fields: set[str] = set()
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            info = MULTIMODAL_REGISTRY.create_processor(self._model_config).info
            fields = info.data_parser.placeholder_metadata_fields(modality)
        except Exception:
            logger.warning(
                "Could not determine the placeholder metadata fields for "
                "modality %s; the consumer will preprocess the media itself.",
                modality,
                exc_info=True,
            )

        self._metadata_fields_cache[modality] = fields
        return fields

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        if self.is_consumer and self._role == ECConnectorRole.SCHEDULER:
            for index in range(len(request.mm_features)):
                transfer_id = self._request_transfer_id(request, index)
                if transfer_id is None:
                    continue
                self._pop_pending_spec(transfer_id)
                self._queue_cancel(transfer_id)
        if not self.is_producer:
            return False, None

        items = []
        for index, feature in enumerate(request.mm_features):
            metadata = {}
            if feature.data is not None:
                wanted = self._placeholder_metadata_fields(feature.modality)
                metadata = {
                    key: value.tolist()
                    for key, value in feature.data.get_data().items()
                    if key in wanted and isinstance(value, torch.Tensor)
                }
            transfer_id = self._request_transfer_id(request, index)
            item = {"mm_hash": feature.identifier, **metadata}
            if transfer_id is not None:
                item["transfer_id"] = transfer_id
            items.append(item)

        if not items:
            return False, None
        return False, {"ec_items": items}

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._flush_pending_pushes()
        self._io_executor.shutdown(wait=True, cancel_futures=True)
        if self._shard_pool is not None:
            self._shard_pool.shutdown(wait=True, cancel_futures=True)
        self._control_executor.shutdown(wait=True, cancel_futures=True)
        # Every thread that could hold a control socket is stopped by now.
        self._control_channel.close()
        if self._control_server is not None:
            self._control_server.shutdown()
        if self._event_zmq_socket is not None:
            self._event_zmq_socket.close(linger=0)
        if self._event_zmq_ctx is not None:
            self._event_zmq_ctx.term()

        if self._engine is not None:
            if self._consumer_pool is not None and self._unregister_memory(
                self._consumer_pool
            ):
                self._consumer_pool = None
                self._consumer_pool_allocator = None
                self._consumer_residents.clear()
                self._consumer_retire_events.clear()
                self._consumer_pending_frees.clear()
            # Published tensors and in-flight push sources share one refcounted
            # registration table, so a single pass covers both.
            with self._push_source_registration_lock:
                addresses = list(self._push_source_registrations)
                addresses.extend(self._pending_unregister)
                unregistered = True
                if addresses:
                    ret = self._engine.batch_unregister_memory(
                        list(dict.fromkeys(addresses))
                    )
                    if ret != 0:
                        unregistered = False
                        logger.error(
                            "Mooncake EC batch memory unregistration failed: %d", ret
                        )
                if unregistered:
                    self._push_source_registrations.clear()
                    self._pending_unregister.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.shutdown()
