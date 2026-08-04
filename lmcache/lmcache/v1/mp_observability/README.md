# MP Observability

Event-driven observability for LMCache's multiprocess (MP) mode, built on
[OpenTelemetry](https://opentelemetry.io/).

For metrics, see [METRICS.md](../../../docs/design/v1/mp_observability/METRICS.md).
For event metadata contracts, see
[EVENTS.md](../../../docs/design/v1/mp_observability/EVENTS.md).
For design rationale, see
[event-bus.md](../../../docs/design/v1/mp_observability/event-bus.md).
For the trace recording subsystem (`lmcache trace`), see
[trace.md](../../../docs/design/v1/mp_observability/trace.md).

---

## Architecture

```
Producers (L1Manager, StorageManager, MPCacheServer)
    │
    │  event_bus.publish(Event(...))
    ▼
EventBus  (async queue + drain thread)
    │
    ├──► L1MetricsSubscriber          → OTel counter.add(...)
    ├──► SMMetricsSubscriber          → OTel counter.add(...)
    ├──► EventBusSelfMetricsSubscriber → OTel observable gauges/counters
    │                                    (bus health: queue depth, drain
    │                                    lag, drops, subscriber exceptions)
    ├──► L1LoggingSubscriber          → logger.debug(...)
    ├──► SMLoggingSubscriber          → logger.debug(...)
    ├──► MPServerLoggingSubscriber    → logger.debug(...)
    ├──► ExtraStatsLoggingSubscriber  → logger.info(...) (opt-in periodic
    │                                    per-GPU L0<->L1 transfer stats)
    └──► MPServerTracingSubscriber    → OTel span start/end

OTel SDK  (configured at startup)
    │
    ├──► OTLP push (production)       → OTel collector → Prometheus / Grafana / etc.
    └──► Prometheus pull (dev/debug)   → /metrics on configured port
```

---

## Configuration

All observability behaviour is controlled by `ObservabilityConfig`
(defined in `config.py`).  When running the LMCache MP mode server from the
CLI, pass the flags below; when embedding programmatically, construct an
`ObservabilityConfig` directly.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--disable-observability` | off | Disable the EventBus entirely. No events are published or consumed. |
| `--disable-metrics` | off | Skip registering metrics subscribers (OTel counters). |
| `--disable-logging` | off | Skip registering logging subscribers. |
| `--enable-tracing` | off | Register tracing subscribers (OTel spans). Disabled by default. **Requires `--otlp-endpoint`.** |
| `--event-bus-queue-size N` | `10000` | Maximum number of events in the EventBus queue before tail-drop. |
| `--otlp-endpoint URL` | *(none)* | OTLP gRPC endpoint (e.g. `http://localhost:4317`). When set, metrics and traces are pushed to an OTel collector. When unset, metrics fall back to Prometheus pull mode. |
| `--prometheus-port PORT` | `9090` | Port for the Prometheus `/metrics` endpoint. Only used when `--otlp-endpoint` is not set. |
| `--enable-gc-monitor` | off | Time every CPython garbage collection and log the slow ones at INFO. Logs directly rather than publishing events, so it is independent of every other flag in this table. |
| `--gc-monitor-min-pause-ms MS` | `1.0` | Only log collections that took at least this long. `0` logs every collection, including the very frequent sub-millisecond gen-0 sweeps. |
| `--gc-monitor-top-objects N` | `0` | Append the N most common object types in the collected generation. Walks the whole generation on **every** collection (O(heap)) — debugging only. |
| `--enable-extra-logging` | off | Periodically log per-GPU L0<->L1 store/retrieve throughput and token counts, plus L1 memory usage (from the eviction loop), at INFO. **Conflicts with `--disable-observability`.** |
| `--extra-logging-interval SECONDS` | `10.0` | Seconds between extra-logging emissions. Values below 1.0 are limited by the 1 Hz internal heartbeat. |

### `ObservabilityConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Master switch for the EventBus. |
| `max_queue_size` | `int` | `10000` | Maximum events in the EventBus queue before tail-drop. |
| `metrics_enabled` | `bool` | `True` | Register metrics subscribers (OTel counters / histograms). |
| `logging_enabled` | `bool` | `True` | Register logging subscribers. |
| `tracing_enabled` | `bool` | `False` | Register tracing subscribers (OTel spans). |
| `otlp_endpoint` | `str \| None` | `None` | OTLP gRPC endpoint. When set, metrics and traces are pushed. When `None`, metrics use Prometheus pull fallback. |
| `prometheus_port` | `int` | `9090` | Port for the Prometheus `/metrics` endpoint (pull fallback only). |
| `gc_monitor` | `GCMonitorConfig` | disabled | CPython GC timing (`gc_monitor.py`). Not an EventBus subscriber: the hook runs inside the collector and logs directly, so GC timing is never coupled to the bus's queue depth or drain lag. |
| `extra_logging_enabled` | `bool` | `False` | Register `ExtraStatsLoggingSubscriber` (periodic per-GPU L0<->L1 transfer stats at INFO). Independent of `logging_enabled`. |
| `extra_logging_interval` | `float` | `10.0` | Seconds between extra-stats log flushes. |
| `service_instance_id` | `str \| None` | `None` | OTel Resource attribute `service.instance.id`, attached to every metric and span. No CLI flag: `run_cache_server` projects the MP server's `--instance-id` onto it. `None` (standalone callers only) falls back to a random UUID v4 at `init_observability` time; an explicit value is preserved. |

### Metrics export modes

| `otlp_endpoint` | Mode | How to query |
|---|---|---|
| `http://host:4317` | OTLP push | Query the OTel collector's Prometheus exporter |
| `None` | Prometheus pull fallback | `curl http://localhost:<prometheus-port>/metrics` |

> **Note:** OTel counters only appear on `/metrics` after the first increment.
> If you see only Python runtime metrics, trigger a store/retrieve first.

### Tracing

Tracing is opt-in (`--enable-tracing`).  When enabled, `MPServerTracingSubscriber`
creates OTel spans from MP server START/END event pairs (store, retrieve,
lookup/prefetch).  Trace export requires an OTLP endpoint — there is no local
fallback.  `--enable-tracing` **requires** `--otlp-endpoint`; the server will
raise a `ValueError` at startup if the endpoint is missing.

---

## How to Add a New Event and Subscriber

### Step 1 — Define the event type

Add a new member to `EventType` in `event.py`:

```python
class EventType(Enum):
    # ... existing events ...

    # My new component events
    MY_COMPONENT_OPERATION = "my_component.operation"
```

### Step 2 — Publish the event from the producer

In your component (e.g., a manager class), publish to the EventBus:

```python
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

class MyComponent:
    def __init__(self):
        self._event_bus = get_event_bus()

    def do_operation(self, keys):
        # ... business logic ...

        self._event_bus.publish(Event(
            event_type=EventType.MY_COMPONENT_OPERATION,
            metadata={"keys": keys},
        ))
```

### Step 3 — Create a subscriber

Create a file under the appropriate `subscribers/` subdirectory:

- `subscribers/metrics/` for OTel counters / histograms
- `subscribers/logging/` for debug log output
- `subscribers/tracing/` for OTel spans

Example metrics subscriber (`subscribers/metrics/my_component.py`):

```python
from opentelemetry import metrics
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class MyComponentMetricsSubscriber(EventSubscriber):
    def __init__(self):
        meter = metrics.get_meter("lmcache.my_component")
        self._op_counter = meter.create_counter(
            "lmcache_mp.my_component_operations",
            description="Total operations on my component",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MY_COMPONENT_OPERATION: self._on_operation,
        }

    def _on_operation(self, event: Event) -> None:
        self._op_counter.add(len(event.metadata["keys"]))
```

### Step 4 — Export from `__init__.py`

Add the subscriber to the corresponding `__init__.py` so it can be
imported from the package:

```python
# subscribers/metrics/__init__.py
from lmcache.v1.mp_observability.subscribers.metrics.my_component import (
    MyComponentMetricsSubscriber,
)
```

### Step 5 — Register the subscriber at startup

In the server startup function (e.g., `run_cache_server()` in `server.py`),
register conditionally based on `ObservabilityConfig`:

```python
if obs_config.metrics_enabled:
    from lmcache.v1.mp_observability.subscribers.metrics import (
        MyComponentMetricsSubscriber,
    )
    bus.register_subscriber(MyComponentMetricsSubscriber())
```

### Step 6 — Document the metadata contract

Add a row to the metadata contracts table in
[EVENTS.md](../../../docs/design/v1/mp_observability/EVENTS.md) so
subscribers can rely on the schema:

```markdown
| `MY_COMPONENT_OPERATION` | `keys` | `list[ObjectKey]` |
```

---

## Design rules

| Rule | Reason |
|---|---|
| Create meters and counters in `__init__()`, not at module level | `MeterProvider` must be set before `get_meter()` is called. Module-level calls happen at import time, before setup. |
| Prefix OTel metric names with `lmcache_mp.` | Keeps the MP namespace separate from `lmcache.` (the single-process engine namespace). |
| Use `metadata: dict[str, Any]` for event payloads | Flexible, no coupling between producers and subscribers. See metadata contracts in [EVENTS.md](../../../docs/design/v1/mp_observability/EVENTS.md). |
| Separate metrics, logging, and tracing subscribers | Single responsibility. Can enable/disable independently via config. |
| Store `self._event_bus = get_event_bus()` in `__init__` | Avoids calling the singleton getter on every publish. |
