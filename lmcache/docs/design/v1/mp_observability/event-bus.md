# MP Observability: EventBus Design

This document describes the design rationale behind the EventBus-based
observability system.  For configuration reference see [README.md](README.md).
For metrics, see [METRICS.md](METRICS.md).  For event metadata contracts,
see [EVENTS.md](EVENTS.md).

---

## 1. Architecture: Event Bus with Pub/Sub

```
Producers (L1Manager, StorageManager, MPCacheServer)
    │
    │  bus.publish(Event(...))
    ▼
EventBus  (single deque + drain thread)
    │
    │  dispatches by EventType
    │
    ├──► Metrics subscribers   → OTel counter.add(...)
    ├──► Logging subscribers   → logger.debug(...)
    └──► Tracing subscribers   → OTel span start/end (explicit timestamps)

OpenTelemetry SDK  (configured once at startup)
    │
    ├──► OTLP push (production)     → OTel collector → Prometheus / Grafana / etc.
    └──► Prometheus pull (dev/debug) → /metrics endpoint
```

Producers publish a single `Event` object.  Subscribers register callbacks
for the `EventType`s they care about.  The EventBus queues events in a
lock-free deque and dispatches them from a background drain thread.

---

## 2. Event Model

```python
class EventType(Enum):
    L1_READ_RESERVED     = "l1.read.reserved"
    L1_READ_FINISHED     = "l1.read.finished"
    # ... (see event.py for the full list)

@dataclass
class Event:
    event_type: EventType
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
```

### Timestamp semantics

`timestamp` is set by `EventBus.publish()` at the moment it is called — **not**
when the drain thread processes the event.  This matters because:

- Subscribers that compute durations (tracing, latency metrics) need the
  real operation time, not the queue-processing time.
- For MP server events, `publish()` is invoked from **CUDA host callbacks**
  (`cupy_stream.launch_host_func`).  The CUDA runtime calls the host function
  when the GPU stream reaches that point, so `time.time()` inside `publish()`
  captures the GPU-accurate moment.
- For L1Manager / StorageManager events, `publish()` is called directly from
  Python, so the timestamp reflects the actual operation time.

### Metadata

All event-specific data lives in `metadata: dict[str, Any]`.  Each `EventType`
has a documented metadata schema (see [EVENTS.md](EVENTS.md)).  This design
keeps the `Event` dataclass generic and avoids coupling between producers and
subscribers.

---

## 3. Event Bus Internals

```python
class EventBus:
    def publish(self, event: Event) -> None:
        """Hot path — stamps timestamp, appends to deque, signals drain."""
        event.timestamp = time.time()
        self._queue.append(event)
        self._wake.set()

    def _drain_all(self) -> None:
        """Background thread pops events and dispatches to subscribers."""
        while event := self._queue.popleft():
            for cb in self._subscribers.get(event.event_type, []):
                cb(event)  # exceptions are caught and logged
```

Key design choices:

- **Async-only dispatch.**  All events go through the queue.  Subscribers
  never run on the producer's thread.  This keeps the hot path (L1Manager,
  StorageManager GPU copies) fast.
- **Tail-drop backpressure.**  When the queue exceeds `max_queue_size`, new
  events are silently dropped with a rate-limited warning.  This prevents
  unbounded memory growth without blocking producers.
- **Exception isolation.**  A failing subscriber callback does not affect other
  subscribers or the drain thread.
- **Self-monitoring.**  The bus exposes four read-only accessors —
  `queue_depth()`, `oldest_event_lag_seconds()`, `dropped_events_count()`,
  `subscriber_exception_counts()` — that surface its own health.  These
  drive the `lmcache_mp.event_bus.*` metrics registered by
  `EventBusSelfMetricsSubscriber` (see [METRICS.md](METRICS.md)). Keeping
  the accessors OTel-free preserves the bus's decoupling from the metrics
  system.

---

## 4. Subscriber Model

```python
class EventSubscriber(ABC):
    @abstractmethod
    def get_subscriptions(self) -> dict[EventType, EventCallback]: ...

    def shutdown(self) -> None:  # optional cleanup
        pass
```

Subscribers are organized by concern:

| Subdirectory | Concern | Examples |
|---|---|---|
| `subscribers/metrics/` | OTel counters / histograms | `L1MetricsSubscriber`, `SMMetricsSubscriber` |
| `subscribers/logging/` | Debug log output | `L1LoggingSubscriber`, `MPServerLoggingSubscriber` |
| `subscribers/tracing/` | OTel spans from START/END pairs | `MPServerTracingSubscriber` |

Each concern can be toggled independently via `ObservabilityConfig`
(`metrics_enabled`, `logging_enabled`, `tracing_enabled`).

---

## 5. OTel Span Strategy

All MP server operations have their START and END in **different call sites**:

- **`store` / `retrieve`:** START and END fire from separate CUDA host callbacks.
- **`lookup_and_prefetch`:** START is in `lookup()`, END is in
  `query_prefetch_status()` — two separate RPC methods.

OTel's idiomatic `with tracer.start_as_current_span(...)` pattern requires
start and end in the same lexical scope, which does not apply here.  Instead,
the tracing subscriber uses explicit span management with caller-provided
timestamps:

```python
class MPServerTracingSubscriber(EventSubscriber):
    def _on_start(self, event: Event) -> None:
        span = tracer.start_span("mp.store", start_time=ns(event.timestamp))
        self._pending[key] = span

    def _on_end(self, event: Event) -> None:
        span = self._pending.pop(key)
        span.end(end_time=ns(event.timestamp))
```

This preserves GPU-accurate timing while producing real OTel spans exportable
to any OTLP-compatible backend (Jaeger, Tempo, Grafana, etc.).

---

## 6. OTel SDK Initialization

Configured once at startup via `ObservabilityConfig`:

- **`otlp_endpoint` is set** → OTLP push mode.  Metrics and traces are
  exported to an OTel collector.
- **`otlp_endpoint` is `None`** → Prometheus pull fallback.  An in-process
  `/metrics` endpoint is served on `prometheus_port`.

Tracing requires an OTLP endpoint (there is no local fallback for spans).
When `tracing_enabled` is set without `otlp_endpoint`, the TracerProvider
init is skipped and spans are silently dropped.

---

## 7. L1Manager Listener Coexistence

`L1Manager` still uses its `L1ManagerListener` ABC for **business-logic
consumers** (`StoreController`, `EvictionPolicy`).  These are not observability
— they drive store scheduling and LRU eviction, and must not be subject to the
bus's bounded-queue drop semantics.

The bus additionally carries the **cache-event stream** for the coordinator's
key directory (`mp_coordinator/cache_events.py`): eventually-consistent
placement hints that tolerate loss by design (a dropped event surfaces as a
``seq`` gap and a resync flag), so bus semantics fit.  For that consumer the
L1 store/evict events carry a ``meta`` list (``L1ObjectMeta``: per-object
size and backing medium, parallel to ``keys``), ``touch_keys`` publishes
``l1.keys.accessed`` with keys only, and the L2 base adapter's
listener-notify funnel
publishes ``l2.keys.stored`` / ``l2.keys.accessed`` / ``l2.keys.deleted``
(``keys`` + ``backend`` [adapter type name] + ``sizes`` for stores).  The
subscriber batches inside its event callbacks on the drain thread and
self-paces flushing by its flush interval, plus a final flush from its
shutdown hook.

For observability, L1Manager publishes events **directly** to the EventBus
alongside its existing listener iteration:

```python
# Business logic — listener pattern stays
for listener in self._registered_listeners:
    listener.on_l1_keys_read_finished(successful_keys)

# Observability — EventBus
self._event_bus.publish(Event(
    event_type=EventType.L1_READ_FINISHED,
    metadata={"keys": successful_keys},
))
```

`StorageManager`, by contrast, has no business-logic listeners and publishes
exclusively to the EventBus.
