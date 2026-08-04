# SPDX-License-Identifier: Apache-2.0

"""EventBus: unified pub/sub dispatcher for MP observability events."""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable
import collections
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge

try:
    # Third Party
    import torch  # noqa: F401 — must be imported before lmcache.c_ops

    # First Party
    import lmcache.c_ops as _lmc_ops

    _has_native_recorder = hasattr(_lmc_ops, "record_event_on_stream")
except ImportError:
    _has_native_recorder = False

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EventCallback = Callable[[Event], None]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EventBusConfig:
    """Configuration for the EventBus.

    Attributes:
        enabled: Whether the event bus is active.  When disabled,
            ``publish()`` is a no-op and the drain thread is not started.
        max_queue_size: Maximum number of events in the queue.  When the
            queue is full, new events are silently dropped with a
            rate-limited warning.
    """

    enabled: bool = True
    max_queue_size: int = 10_000


# ---------------------------------------------------------------------------
# Subscriber ABC
# ---------------------------------------------------------------------------


class EventSubscriber(ABC):
    """Base class for per-component event subscribers.

    Subclasses declare which ``EventType``\\s they care about via
    ``get_subscriptions()``.  The ``register()`` helper wires them up to
    an ``EventBus``.
    """

    @abstractmethod
    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return event_type -> callback mapping.

        Called once during ``register()``.  The EventBus stores these
        callbacks directly.
        """
        ...

    def register(self, bus: EventBus) -> None:
        """Subscribe all declared handlers to *bus*."""
        for event_type, callback in self.get_subscriptions().items():
            bus.subscribe(event_type, callback)

    def shutdown(self) -> None:  # noqa: B027
        """Optional cleanup hook.  Called by ``EventBus.stop()``."""
        pass


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Manages event ingestion, queueing, and dispatch to subscribers.

    Events are appended to a deque on the hot path (``publish()``) and
    drained by a background thread that dispatches to registered callbacks.
    """

    def __init__(self, config: EventBusConfig | None = None) -> None:
        if config is None:
            config = EventBusConfig()
        self._config = config
        self._subscribers: dict[EventType, list[EventCallback]] = defaultdict(list)
        self._queue: collections.deque[Event] = collections.deque()
        self._wake = threading.Event()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._registered_subscribers: list[EventSubscriber] = []
        self._discard_count: int = 0
        self._last_discard_warning: float = 0.0
        self._subscriber_exception_counts: dict[str, int] = {}

        self._register_self_gauges()

    def _register_self_gauges(self) -> None:
        """Register the two self-monitoring gauges via ``register_gauge``."""
        register_gauge(
            "lmcache.event_bus",
            "lmcache_mp.event_bus.queue_depth",
            "Events currently queued in the EventBus.",
            self.queue_depth,
        )
        register_gauge(
            "lmcache.event_bus",
            "lmcache_mp.event_bus.drain_lag_seconds",
            (
                "Seconds since the oldest queued event was published; 0.0 "
                "when empty.  Rising values mean the drain thread is "
                "falling behind."
            ),
            self.oldest_event_lag_seconds,
        )

    # -- Public API --------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Register a callback for a specific event type (thread-safe)."""
        with self._lock:
            self._subscribers[event_type].append(callback)

    def register_subscriber(self, subscriber: EventSubscriber) -> None:
        """Register an ``EventSubscriber`` and wire up its callbacks."""
        subscriber.register(self)
        with self._lock:
            self._registered_subscribers.append(subscriber)

    def has_subscribers(self, event_type: EventType) -> bool:
        """Return True if at least one callback is registered for *event_type*.

        Use this to skip expensive event construction on the hot path when
        no subscriber is listening::

            if bus.has_subscribers(EventType.MP_LOOKUP):
                bus.publish(Event(event_type=EventType.MP_LOOKUP, ...))
        """
        return bool(self._subscribers.get(event_type))

    def publish_on_stream(self, stream: Any, event: Event) -> None:
        """Schedule event recording as a CUDA host function on *stream*.

        Uses a C++ callback via ``cudaLaunchHostFunc`` so the callback
        never touches the GIL, avoiding the CUDA-driver/GIL deadlock.

        No-op when the EventBus is disabled, avoiding the overhead of
        scheduling a host function on the CUDA stream entirely.
        """
        if not self._config.enabled:
            return
        if _has_native_recorder:
            str_metadata: dict[str, str] = {}
            int_metadata: dict[str, int] = {}
            for k, v in event.metadata.items():
                if isinstance(v, int):
                    int_metadata[k] = v
                else:
                    str_metadata[k] = str(v)
            _lmc_ops.record_event_on_stream(
                stream.ptr,
                event.event_type.value,
                event.session_id,
                str_metadata,
                int_metadata,
            )
        else:
            stream.launch_host_func(self.publish, event)

    def publish(self, event: Event) -> None:
        """Submit an event (hot path — non-blocking).

        The event's ``timestamp`` is set to ``time.time()`` at call time.
        When the queue is full the event is silently discarded with a
        rate-limited warning (at most once per second).
        """
        if not self._config.enabled:
            return

        if len(self._queue) >= self._config.max_queue_size:
            self._discard_count += 1
            now = time.monotonic()
            if now - self._last_discard_warning >= 1.0:
                logger.warning(
                    "EventBus queue full (max_queue_size=%d), "
                    "%d event(s) discarded so far",
                    self._config.max_queue_size,
                    self._discard_count,
                )
                self._last_discard_warning = now
            return

        event.timestamp = time.time()
        self._queue.append(event)
        self._wake.set()

    def start(self) -> None:
        """Start the background drain thread.  No-op when disabled or
        already running."""
        if not self._config.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EventBus",
        )
        logger.debug("Starting EventBus drain thread...")
        self._thread.start()

    def stop(self) -> None:
        """Stop the drain thread, flush remaining events, and shut down
        all registered subscribers.  Safe to call when not started."""
        self._stop_flag.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        # Final drain
        self._drain_all()

        # Shutdown subscribers
        with self._lock:
            snapshot = list(self._registered_subscribers)
        for sub in snapshot:
            try:
                sub.shutdown()
            except Exception:
                logger.exception(
                    "EventBus: error shutting down %s",
                    type(sub).__name__,
                )

    # -- Self-monitoring ---------------------------------------------------

    def queue_depth(self) -> int:
        """Number of events currently queued waiting for dispatch.

        Reads ``len`` on the underlying deque, which is atomic in CPython,
        so callers can poll this from a metrics scrape callback without
        holding the bus lock.
        """
        return len(self._queue)

    def oldest_event_lag_seconds(self) -> float:
        """Wall-clock age of the oldest queued event, or 0.0 when empty.

        Used as a drain-lag gauge: a rising value means the drain thread
        is not keeping up with publish rate.  Read without the lock; if
        the deque is concurrently popped during the peek, returns 0.0.
        """
        try:
            oldest = self._queue[0]
        except IndexError:
            return 0.0
        return max(0.0, time.time() - oldest.timestamp)

    def dropped_events_count(self) -> int:
        """Cumulative count of events dropped because the queue was full."""
        return self._discard_count

    def subscriber_exception_counts(self) -> dict[str, int]:
        """Snapshot of per-subscriber callback exception counts.

        Maps ``subscriber_name`` (the owning class name for bound methods,
        ``__qualname__`` for free functions) to the cumulative count of
        exceptions raised by that subscriber's callbacks during dispatch.
        Returns a copy that callers may iterate without holding the bus
        lock.
        """
        with self._lock:
            return dict(self._subscriber_exception_counts)

    # -- Internal ----------------------------------------------------------

    def _run(self) -> None:
        """Drain loop: wait for wake signal or timeout, then drain."""
        while not self._stop_flag.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            self._drain_all()

    def _drain_all(self) -> None:
        """Pop all queued events and dispatch to subscribers."""
        # Drain events buffered on the C++ side (from CUDA host callbacks)
        if _has_native_recorder:
            for name, sid, ts, str_meta, int_meta in _lmc_ops.drain_recorded_events():
                metadata: dict[str, Any] = dict(str_meta)
                metadata.update(int_meta)
                self._queue.append(
                    Event(
                        event_type=EventType(name),
                        session_id=sid,
                        timestamp=ts,
                        metadata=metadata,
                    )
                )

        with self._lock:
            snapshot = dict(self._subscribers)

        while True:
            try:
                event = self._queue.popleft()
            except IndexError:
                break
            for cb in snapshot.get(event.event_type, []):
                try:
                    cb(event)
                except Exception:
                    instance = getattr(cb, "__self__", None)
                    name = (
                        type(instance).__name__
                        if instance is not None
                        else getattr(cb, "__qualname__", repr(cb))
                    )
                    with self._lock:
                        self._subscriber_exception_counts[name] = (
                            self._subscriber_exception_counts.get(name, 0) + 1
                        )
                    logger.exception(
                        "EventBus: error in callback %s for %s",
                        name,
                        event.event_type.value,
                    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_bus = EventBus(EventBusConfig(enabled=False))
_observability_enabled: bool = False


def is_observability_enabled() -> bool:
    """Fast check for whether observability is active.

    Use this to guard expensive event-construction or CUDA host-function
    scheduling when observability is disabled.
    """
    return _observability_enabled


def get_event_bus() -> EventBus:
    """Return the current global EventBus singleton."""
    return _global_bus


def init_event_bus(config: EventBusConfig | None = None) -> EventBus:
    """Replace the global singleton with a new EventBus built from *config*.

    Returns the newly created bus.
    """
    global _global_bus, _observability_enabled
    _global_bus = EventBus(config)
    _observability_enabled = config.enabled if config else True
    return _global_bus
