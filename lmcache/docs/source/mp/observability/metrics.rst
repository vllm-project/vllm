Metrics
=======

Metrics are collected via OpenTelemetry and made available to Prometheus in
one of two ways, depending on whether ``--otlp-endpoint`` is set:

- **Pull mode (default, no collector).** When ``--otlp-endpoint`` is *not*
  set, the server publishes a Prometheus ``/metrics`` endpoint that Prometheus
  scrapes directly.

  .. important::

     For ``lmcache server``, ``/metrics`` is served by the **HTTP frontend**
     on ``--http-port`` (default **8080**), e.g.
     ``http://<host>:8080/metrics`` — **not** on ``--prometheus-port``.
     ``--prometheus-port`` is *ignored* by ``lmcache server``: the standalone
     Prometheus HTTP server is disabled because the HTTP frontend already
     serves ``/metrics``. The frontend-less entrypoints
     (``python -m lmcache.v1.multiprocess.server`` and ``lmcache trace
     replay``) have no HTTP frontend, so *they* serve ``/metrics`` on
     ``--prometheus-port`` (default 9090). See
     :ref:`mp-obs-metrics-endpoint` for the full breakdown. Either way,
     ``/metrics`` is empty until the first store/retrieve — drive some
     traffic before you go looking.

- **Push mode (OTLP).** When ``--otlp-endpoint`` is set, metrics are pushed to
  an OpenTelemetry Collector, which re-exposes them for Prometheus to scrape.
  See :doc:`index` for the bundled Collector + Prometheus + Grafana stack.

All metrics use the ``lmcache_mp.`` prefix (multiprocess). On Prometheus,
dots are converted to underscores and counters get a ``_total`` suffix
(e.g. ``lmcache_mp_l1_read_chunks_total``); histograms gain a unit suffix
plus ``_sum`` / ``_count`` / ``_bucket`` (e.g.
``lmcache_mp_l2_store_throughput_GB_per_second_sum``).

.. _mp-observability-resource:

Global Resource Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every metric and span exported by an MP server carries Resource-level
attributes built at startup. These identify the process producing the
telemetry and are orthogonal to per-metric attributes such as
``cache_salt``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Attribute
     - CLI flag / config
     - Default when unset
   * - ``service.instance.id``
     - ``--instance-id`` / ``MPServerConfig.instance_id``
     - Random UUID v4 minted at startup.

Resource attributes attach to the ``MeterProvider`` / ``TracerProvider``
and propagate to every exported datapoint via OTLP. On Prometheus, SDK
resource attributes surface on the ``target_info`` series rather than
on each time-series — this is standard OTel behavior.

L1 Metrics
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_read``
     - Counter (attr: ``cache_salt``)
     - Number of chunks read from L1, grouped by tenant.
   * - ``lmcache_mp.l1_write``
     - Counter (attr: ``cache_salt``)
     - Number of chunks written to L1, grouped by tenant.
   * - ``lmcache_mp.l1_evicted``
     - Counter (attr: ``cache_salt``)
     - Number of chunks evicted by the EvictionController, grouped by
       tenant.
   * - ``lmcache_mp.l1_eviction_loop_ticks``
     - Counter
     - L1 eviction-loop iterations (every cycle, regardless of whether
       the watermark was crossed). Driven by ``L1_EVICTION_LOOP_TICK``.
   * - ``lmcache_mp.l1_eviction_loop_triggered``
     - Counter
     - L1 eviction-loop iterations where ``usage >= watermark`` and the
       eviction policy actually ran. The two counters distinguish "loop
       is alive" from "eviction fired" — important when debugging
       short-lived benchmarks that complete faster than the 1 Hz
       polling cycle.

L1 Chunk Lifecycle Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampled (default 1%) chunk-level lifecycle tracking via
``L1LifecycleSubscriber``. Only sampled chunks contribute to histograms;
counters above always count all events. Sampling is deterministic
(hash-based), so the same key always gets the same decision with zero
memory overhead.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_chunk_lifetime``
     - Histogram
     - Time from allocation to eviction per sampled chunk.
   * - ``lmcache_mp.l1_chunk_idle_before_evict``
     - Histogram
     - Time from last access to eviction per sampled chunk.
   * - ``lmcache_mp.l1_chunk_reuse_gap``
     - Histogram
     - Time gap between consecutive touches (read or write) of the same chunk.
   * - ``lmcache_mp.l1_chunk_evict_reuse_gap``
     - Histogram
     - Time from eviction to next reuse (capped at 300 s).

StorageManager Real-Reuse Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workload-level reuse histograms emitted by ``SMLifecycleSubscriber``,
driven by caller-facing StorageManager events
(``SM_READ_PREFETCHED_FINISHED``, ``SM_WRITE_FINISHED``).  Internal
read-lock releases by the store/prefetch controllers are excluded so
the signal reflects user-driven access only.

Both histograms are tagged with ``cache_salt`` for per-tenant
isolation.  The per-salt access counter advances on every read and
write of every chunk (regardless of sampling) so the chunks-gap
reflects true storage volume; the histogram itself records gaps only
for chunks that pass the (deterministic, hash-based) sampling gate.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.real_reuse_gap``
     - Histogram (tag: ``cache_salt``)
     - Time gap between a chunk's last access (read or write) and its
       next read.  Captures storage cost — how long a stored chunk sat
       between accesses.  Emitted only on read events.
   * - ``lmcache_mp.real_reuse_gap_objects``
     - Histogram (tag: ``cache_salt``)
     - Per-``cache_salt`` access-counter gap between two reads of the
       same chunk.  Captures storage volume — how many chunk-accesses
       occurred while this chunk waited for its next read.  Emitted on
       read events for sampled chunks.

L2 Metrics
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l2_store_submitted``
     - Counter
     - Number of L2 store requests submitted.
   * - ``lmcache_mp.l2_store_submitted_objects``
     - Counter (attr: ``cache_salt``)
     - Number of chunks submitted for L2 store, grouped by tenant.
   * - ``lmcache_mp.l2_store_completed``
     - Counter (attr: ``l2_name``)
     - Number of L2 store requests completed, labeled by adapter type.
   * - ``lmcache_mp.l2_store_completed_objects``
     - Counter (attr: ``cache_salt``)
     - Number of chunks successfully stored to L2, grouped by tenant.
   * - ``lmcache_mp.l2_prefetch_lookup``
     - Counter
     - Number of L2 prefetch lookup requests.
   * - ``lmcache_mp.l2_prefetch_lookup_objects``
     - Counter (attr: ``cache_salt``)
     - Number of chunks submitted for L2 prefetch lookup, grouped by
       tenant.
   * - ``lmcache_mp.l2_prefetch_hit``
     - Counter
     - Number of prefix chunks found in L2 lookup.
   * - ``lmcache_mp.l2_prefetch_load_submitted``
     - Counter
     - Number of L2 prefetch load requests submitted.
   * - ``lmcache_mp.l2_prefetch_load_submitted_objects``
     - Counter (attr: ``cache_salt``)
     - Number of chunks submitted for L2 load, grouped by tenant.
   * - ``lmcache_mp.l2_prefetch_load_completed``
     - Counter (attr: ``cache_salt``)
     - Number of chunks successfully loaded from L2, grouped by tenant.
   * - ``lmcache_mp.l2_load_completed``
     - Counter (attr: ``l2_name``)
     - Number of per-adapter L2 load requests completed, labeled by adapter type.
   * - ``lmcache_mp.l2_evicted_objects``
     - Counter (attr: ``cache_salt``)
     - Number of chunks evicted from L2, grouped by tenant.

The ``l2_name``-labeled counters (``l2_store_completed`` and
``l2_load_completed``) exist so dashboards can compute per-backend IOPS on
demand via ``rate(lmcache_mp_l2_store_completed_requests_total{l2_name="..."}[1m])``
(and the equivalent for loads).  No separate ``*_iops`` metric is exported;
keeping the raw counter lets dashboard users pick their own window.

Failure & Health Counters
~~~~~~~~~~~~~~~~~~~~~~~~~

Health-monitoring counters emitted on the dedicated ``lmcache_mp.health``
OTel meter. Driven by the ``L1FailureMetricsSubscriber`` and
``L2FailureMetricsSubscriber``, which are registered automatically when
metrics are enabled. All three counters carry ``model_name`` (extracted
from each ``ObjectKey``) so operators can slice per-model on the
Prometheus ``/metrics`` endpoint.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_allocation_failure``
     - Counter
     - L1 memory allocation failures (OOM) during ``reserve_write``.
       Tagged by ``during`` ∈ {``l1_store``, ``l2_prefetch``} to
       distinguish user-initiated stores from prefetch-triggered
       allocations, plus ``model_name``.
   * - ``lmcache_mp.l1_read_failure``
     - Counter
     - L1 ``reserve_read`` failures. Tagged by ``during`` ∈
       {``l2_store``, ``l1_retrieve``}, ``reason`` ∈ {``not_found``,
       ``write_locked``}, plus ``model_name``. **Post-lookup anomaly
       counter**, not a cache-miss counter — in MP mode ``reserve_read``
       is only called after a successful lookup, so any non-zero value
       indicates a lookup/reserve race or unexpected eviction and should
       stay near zero in healthy operation.
   * - ``lmcache_mp.l2_prefetch_failure``
     - Counter
     - Chunks that L2 reported present at lookup but failed to land in L1.
       Tagged by ``reason`` ∈ {``l1_oom``, ``not_found``} plus
       ``model_name``. ``l1_oom`` means L1 had no room to receive the
       prefetched object; ``not_found`` means the adapter returned no
       data despite a positive lookup (e.g. concurrent delete).

A ``reason=serde_failure`` value will be added to ``l2_prefetch_failure``
as an additive, non-breaking extension once L2 adapters distinguish
deserialization errors from missing objects — no dashboard migration
needed when that lands.

For the full design rationale (including which event types drive each
counter and why ``lmcache_instance_id`` is deferred), see
``docs/design/v1/mp_observability/METRICS.md`` in the source tree.

Lookup Hit-Rate Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Token-level counters whose ratio gives the fraction of tokens requested by
a lookup that were served from either L1 or L2. L0 (GPU prefix cache) is
intentionally excluded — it is vLLM-owned and not observable from LMCache.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.lookup_requested``
     - Counter (attrs: ``model_name``, ``cache_salt``)
     - Total tokens submitted for lookup (denominator of the L1+L2
       token-level hit rate). Only chunk-aligned tokens are counted.
   * - ``lmcache_mp.lookup_hit``
     - Counter (attrs: ``model_name``, ``cache_salt``)
     - Total tokens found in L1 or L2 during lookup (numerator of the
       L1+L2 token-level hit rate). Counts the contiguous prefix hit only.

Both counters are driven by the same event (``MP_LOOKUP_PREFETCH_END``),
so they always advance together per completed lookup. Early-exit lookups
contribute ``0`` to both, and abandoned lookups contribute to neither.

The ``model_name`` and ``cache_salt`` attributes are captured at lookup
time from ``IPCCacheServerKey`` so dashboards can compute per-model or
per-tenant hit rate. ``cache_salt`` can be high-cardinality (one entry
per tenant or isolation domain); drop it at scrape time with
``metric_relabel_configs`` if storage cost matters.

**PromQL for hit rate:**

.. code-block:: promql

    # Aggregate (all models, all salts):
    rate(lmcache_mp_lookup_hit_tokens_total[5m])
    / rate(lmcache_mp_lookup_requested_tokens_total[5m])

    # Per-model:
    sum(rate(lmcache_mp_lookup_hit_tokens_total[5m])) by (model_name)
    / sum(rate(lmcache_mp_lookup_requested_tokens_total[5m])) by (model_name)

L0 (GPU) Block Lifecycle Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampled (default 1%) GPU KV cache block lifecycle tracking via
``L0LifecycleSubscriber``. Eviction is detected at reallocation time
(when a block is assigned different tokens). Sampling uses random
selection with a ``_skipped`` set (bounded by the finite number of
physical GPU blocks).

All L0 histograms are emitted with ``instance_id`` and ``model_name``
OTel attributes, enabling per-instance and per-model metric slicing
in Prometheus (e.g.
``lmcache_mp_l0_block_lifetime_seconds{instance_id="12345",model_name="llama-7b"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l0_block_lifetime``
     - Histogram
     - Time from allocation to eviction per sampled GPU block.
   * - ``lmcache_mp.l0_block_idle_before_evict``
     - Histogram
     - Time from last access to eviction per sampled GPU block.
   * - ``lmcache_mp.l0_block_reuse_gap``
     - Histogram
     - Time gaps between consecutive accesses of the same GPU block.

L0 ↔ L1 Throughput Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-request throughput of GPU↔CPU copies via
``L0L1ThroughputSubscriber``. Every store/retrieve request contributes
one sample to the appropriate histogram:
``total_bytes / (end_ts - start_ts)`` in GB/s. Timestamps come from
``MP_{STORE,RETRIEVE}_{START,END}`` events published on the GPU cupy
stream, so they reflect true GPU-stream copy time — not Python/lock
overhead.

All throughput histograms are emitted with ``engine_id`` (vLLM worker
instance id), ``device`` (e.g. ``"cuda:3"``), and ``model_name`` OTel
attributes, enabling per-worker, per-device, and per-model slicing in
Prometheus (e.g.
``lmcache_mp_l0_l1_store_throughput_GB_per_second{engine_id="0",device="cuda:3",model_name="meta-llama/Llama-3.1-8B"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l0_l1_store_throughput``
     - Histogram
     - GPU→CPU (L0→L1) store throughput in GB/s per request.
   * - ``lmcache_mp.l0_l1_load_throughput``
     - Histogram
     - CPU→GPU (L1→L0) load throughput in GB/s per request.

L1 ↔ L2 Throughput Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-request throughput of L1↔L2 transfers via
``L2ThroughputSubscriber``. The store path correlates
``L2_STORE_SUBMITTED`` → ``L2_STORE_COMPLETED`` by
``(adapter_index, task_id)``. The load path correlates the per-adapter
``L2_LOAD_TASK_SUBMITTED`` → ``L2_LOAD_TASK_COMPLETED`` events by
``(request_id, adapter_index)``; the request-level
``L2_PREFETCH_LOAD_*`` events used by the chunk-count counters aggregate
across adapters and cannot be attributed to a specific ``l2_name``.

Timestamps span **submit → complete**, so the duration includes adapter
queue, network, and disk I/O — the value is *bytes / end-to-end
latency*, not raw transfer rate. Use these histograms to compare
adapter types and catch regressions; use the L0↔L1 histograms when you
need pure copy-time throughput.

All L1↔L2 throughput histograms carry a single ``l2_name`` OTel
attribute — the registered adapter type (e.g. ``"fs"``, ``"nixl_store"``,
``"mooncake_store"``) — enabling per-backend slicing in Prometheus (e.g.
``lmcache_mp_l2_store_throughput_GB_per_second{l2_name="nixl_store"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l2_store_throughput``
     - Histogram
     - L1→L2 store throughput in GB/s per request.
   * - ``lmcache_mp.l2_load_throughput``
     - Histogram
     - L2→L1 load throughput in GB/s per (request, adapter) pair.

**PromQL for average throughput (GB/s), per backend:**

.. code-block:: promql

    # L1 -> L2 store throughput, averaged over the last minute, per l2_name:
    sum by (l2_name) (rate(lmcache_mp_l2_store_throughput_GB_per_second_sum[1m]))
    / sum by (l2_name) (rate(lmcache_mp_l2_store_throughput_GB_per_second_count[1m]))

    # L2 -> L1 load throughput (same shape):
    sum by (l2_name) (rate(lmcache_mp_l2_load_throughput_GB_per_second_sum[1m]))
    / sum by (l2_name) (rate(lmcache_mp_l2_load_throughput_GB_per_second_count[1m]))

.. note::

   ``l2_store_throughput`` populates whenever chunks are written to L2.
   ``l2_load_throughput`` only populates when chunks are read **from L2 into
   L1** — i.e. on a prefetch load after the entry has aged out of L1. If your
   working set fits entirely in L1 (common with small models or a large
   ``--l1-size-gb``), lookups are served from L1 and the load histogram stays
   empty even though store throughput is non-zero. Drive enough distinct data
   to force L1 eviction, or restart the server between store and load passes,
   to exercise the L2 load path.

Engine Counters
~~~~~~~~~~~~~~~

Worker-scoped counters tied to what the MP server delivers back to each
vLLM worker via ``retrieve()``.  Labeled by ``worker_id`` (the vLLM
worker instance id) — distinct from any scheduler-scoped id that may
appear on other metrics.

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.num_chunks_loaded``
     - Counter (attrs: ``worker_id``, ``model_name``, ``cache_salt``)
     - Total number of LMCache chunks loaded into the engine, summed
       over all ``retrieve()`` completions.  Sliceable per worker, per
       model, and per tenant / isolation domain (``cache_salt``).
       ``cache_salt`` may be high-cardinality; drop it at scrape time
       with ``metric_relabel_configs`` if storage cost matters.

Observable Gauges
~~~~~~~~~~~~~~~~~

Point-in-time state snapshots registered via ``register_gauge``
(pull-based OTel observable gauges).

The three in-flight metrics carry two attributes that distinguish
adapters even when more than one is registered with the same backend
type — same shape as ``lmcache_mp.l2_store_completed``:

- ``l2_name`` — the registered adapter type (e.g. ``"fs"``,
  ``"nixl_store"``, ``"mooncake_store"``).
- ``adapter_index`` — position in the controller's adapter list.

Adapters with no in-flight work emit no datapoint for that scrape.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.active_prefetch_jobs``
     - ObservableGauge
     - Number of prefetch jobs currently in-flight. A sustained high
       value may indicate slow L2 backends or polling delays.
   * - ``lmcache_mp.l1_memory_usage_bytes``
     - ObservableGauge
     - Bytes currently held in L1.  Rising without plateauing typically
       indicates a leak; saturating at the configured ``--l1-size-gb``
       indicates working set exceeds capacity.
   * - ``lmcache_mp.l1_usage_ratio``
     - ObservableGauge
     - L1 used/total ratio (``0.0``–``1.0``), sampled at scrape time
       from ``L1Manager.get_memory_usage()``. Returns ``0.0`` when the
       gauge target is not yet wired up or ``total_bytes`` is zero, so
       the callback never raises during a scrape. Compare against the
       eviction watermark (default ``0.8``) to read whether the
       eviction loop is below or above its trigger threshold.
   * - ``lmcache_mp.l2_usage_bytes``
     - ObservableGauge (attr: ``l2_name``)
     - Bytes currently held in each L2 adapter, sampled at scrape time
       from ``adapter.get_usage()``.  One observation per configured
       adapter, tagged by ``l2_name`` (the adapter type, e.g. ``"fs"``,
       ``"nixl_store"``, ``"mooncake_store"``).  Parallel to
       ``l1_memory_usage_bytes`` for the L2 tier — use it to see how
       much each L2 backend currently holds.  Adapters whose
       ``get_usage()`` raises are skipped silently rather than poisoning
       the observation, so a missing datapoint for one ``l2_name`` can
       mean either "not configured" or "adapter errored on this
       scrape" — cross-check with the L2 store/load counters.
   * - ``lmcache_mp.num_inflight_l2_stores``
     - ObservableGauge (attrs: ``l2_name``, ``adapter_index``)
     - L2 store tasks currently executing, per adapter.  Sustained
       non-zero values indicate the adapter cannot keep up with the
       L1 → L2 write rate.
   * - ``lmcache_mp.num_inflight_l2_loads``
     - ObservableGauge (attrs: ``l2_name``, ``adapter_index``)
     - L2 → L1 prefetch load tasks currently executing, per adapter.
       Pair with ``num_inflight_l2_stores`` to see whether read or write
       traffic dominates a given backend.
   * - ``lmcache_mp.inflight_load_memory_usage_bytes``
     - ObservableGauge (attrs: ``l2_name``, ``adapter_index``)
     - L1 bytes reserved by in-flight L2 → L1 prefetch loads, per
       adapter.  Rising in-flight bytes alongside rising
       ``l1_memory_usage_bytes`` is a signal that prefetch reservations
       are crowding out cacheable data.  Per-adapter byte attribution
       follows each request's ``load_plan`` bitmap, so summing across
       adapters never double-counts.

EventBus Self-Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

Health metrics for the EventBus itself, registered by
``EventBusSelfMetricsSubscriber`` on the ``lmcache.event_bus`` OTel
meter.  These metrics observe bus state directly via the ``EventBus``
accessors and report on every OTel scrape — they are not driven by
events, so dropping or failing subscribers cannot silence them.

Use them to answer: is the EventBus keeping up with publishers, is
anything being dropped, and are any subscriber callbacks raising?
A non-zero ``dropped_events_total`` or a sustained non-zero
``drain_lag_seconds`` indicates the bus is at ``--event-bus-queue-size``
and tail-dropping; raise that flag or investigate slow subscribers.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.event_bus.queue_depth``
     - ObservableGauge
     - Events currently queued in the EventBus (``len(_queue)`` at
       scrape time).
   * - ``lmcache_mp.event_bus.drain_lag_seconds``
     - ObservableGauge
     - Seconds since the oldest queued event was published; ``0.0``
       when empty.  Rising values mean the drain thread is falling
       behind.
   * - ``lmcache_mp.event_bus.dropped_events_total``
     - ObservableCounter
     - Cumulative events dropped because the EventBus queue was at
       ``--event-bus-queue-size``.
   * - ``lmcache_mp.event_bus.subscriber_exceptions``
     - ObservableCounter (attr: ``subscriber_name``)
     - Cumulative exceptions raised by subscriber callbacks during
       EventBus dispatch, tagged by ``subscriber_name`` (the failing
       callback's owning class for bound methods, or ``__qualname__``
       for free functions).

For the full design rationale and the in-process accessors that back
each metric see ``docs/design/v1/mp_observability/METRICS.md`` and
``docs/design/v1/mp_observability/event-bus.md`` in the source tree.

Prometheus Scrape Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In **pull mode** (no ``--otlp-endpoint``), point Prometheus at the server's
HTTP-frontend port — ``--http-port``, default **8080** — not at
``--prometheus-port``:

.. code-block:: yaml

    scrape_configs:
      - job_name: "lmcache-mp"
        static_configs:
          - targets: ["<lmcache-host>:8080"]   # --http-port, NOT --prometheus-port

In **push mode** (``--otlp-endpoint`` set), the server does not expose
``/metrics`` itself; scrape the OpenTelemetry Collector's Prometheus exporter
instead. The bundled stack in ``examples/observability/`` wires this up for
you — see :doc:`index`.

