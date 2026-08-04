Observability
=============

LMCache multiprocess mode provides three complementary observability modes:
**metrics** (Prometheus counters via OTel), **logging** (Python logging with
optional OTel log forwarding), and **tracing** (OTel spans for per-request
latency).

All three modes are powered by an internal **EventBus** that decouples
producers (L1Manager, StorageManager, MPCacheServer) from subscribers.

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

By default, **metrics** and **logging** are enabled; **tracing** is disabled.
No extra flags are needed:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU

The server then exposes Prometheus metrics at ``/metrics`` on its **HTTP
frontend port** (``--http-port``, default ``8080``):

.. code-block:: bash

    curl http://localhost:8080/metrics | grep lmcache_mp_

.. important::

   For ``lmcache server``, ``/metrics`` lives on ``--http-port`` (default
   ``8080``), **not** on ``--prometheus-port``: the HTTP frontend already
   serves ``/metrics``, so the standalone Prometheus server is disabled and
   ``--prometheus-port`` has no effect under this command. ``--prometheus-port``
   *is* the metrics endpoint for the frontend-less entrypoints
   (``python -m lmcache.v1.multiprocess.server`` and ``lmcache trace replay``)
   — see :ref:`mp-obs-metrics-endpoint`. Also note metrics are lazy: a series
   only appears *after* the first store/retrieve that produces it, so drive
   some traffic before scraping.

To also see L2 (storage-tier) metrics, attach an L2 backend with
``--l2-adapter``. The simplest is the local filesystem:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

To enable tracing instead of (or alongside) Prometheus pull, supply an OTLP
endpoint — this switches metrics to **push mode** (see
:ref:`mp-obs-grafana`):

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --enable-tracing --otlp-endpoint http://localhost:4317

.. _mp-obs-metrics-endpoint:

Where ``/metrics`` lives
~~~~~~~~~~~~~~~~~~~~~~~~~~

The pull-mode ``/metrics`` endpoint is served in one of two places depending
on the entrypoint. Entrypoints that embed the uvicorn HTTP frontend serve it
there (and disable the standalone Prometheus server); entrypoints with no
HTTP frontend start the standalone server on ``--prometheus-port`` instead.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Entrypoint
     - HTTP frontend?
     - Pull-mode ``/metrics`` endpoint
   * - ``lmcache server``
     - yes
     - ``--http-port`` (default ``8080``); ``--prometheus-port`` ignored
   * - ``python -m lmcache.v1.multiprocess.server``
     - no
     - ``--prometheus-port`` (default ``9090``)
   * - ``lmcache trace replay``
     - no
     - ``--prometheus-port`` (default ``9090``)

In **push mode** (``--otlp-endpoint`` set) none of these serve ``/metrics`` —
metrics are pushed to the collector instead.

.. _mp-obs-grafana:

Viewing Metrics in Grafana
--------------------------

There are two ways to get LMCache metrics into Grafana. Pick based on whether
you also want **traces**.

.. list-table::
   :header-rows: 1
   :widths: 22 20 30 28

   * - Path
     - Server flags
     - What you run
     - Gives you
   * - **A. Bundled stack**
     - ``--otlp-endpoint`` (push)
     - ``docker compose up`` in ``examples/observability/``
     - Metrics **+ traces**, Grafana with the LMCache dashboard
       auto-provisioned
   * - **B. Pull mode**
     - none (default)
     - Your own Prometheus + Grafana scraping ``:8080``
     - Metrics only, minimal moving parts, no collector

Path A — bundled Prometheus + Tempo + Grafana
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository ships a ready-to-run stack (OpenTelemetry Collector →
Prometheus + Tempo → Grafana) under ``examples/observability/``. Grafana comes
with the **LMCache dashboard and datasources pre-provisioned** and anonymous
access enabled, so there is nothing to click to log in.

.. code-block:: bash

    # 1. Start the observability stack (Collector :4320, Prometheus, Tempo,
    #    Grafana :3000)
    cd examples/observability
    docker compose up -d

    # 2. Start the LMCache server (+ vLLM) pushing OTLP to the collector
    MODEL=/path/to/model bash start-server.sh

    # 3. Generate traffic, then open Grafana
    #    http://localhost:3000  ->  Dashboards  ->  "LMCache"

In this path the server pushes to the Collector and **Prometheus scrapes the
Collector**, so you do *not* scrape the server's ``:8080`` directly.

Path B — pull mode (metrics only, no collector)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only want metrics, skip the collector entirely and have Prometheus
scrape the server's ``/metrics`` endpoint directly. Start the server **without**
``--otlp-endpoint`` (see Quick Start above), then:

.. code-block:: bash

    # 1. Prometheus config: scrape the server's HTTP-frontend port (8080)
    cat > prometheus.yml <<'YAML'
    global:
      scrape_interval: 5s
    scrape_configs:
      - job_name: lmcache
        static_configs:
          - targets: ["localhost:8080"]   # --http-port, NOT --prometheus-port
    YAML

    # 2. Run Prometheus (:9090) and Grafana (:3000) on the host network so
    #    they can reach localhost:8080 and each other.
    docker run -d --name lmcache-prom --network host \
        -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
        prom/prometheus

    docker run -d --name lmcache-grafana --network host \
        -e GF_AUTH_ANONYMOUS_ENABLED=true \
        -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
        grafana/grafana

Then in Grafana (``http://localhost:3000``):

1. **Add a datasource** → Prometheus → URL ``http://localhost:9090`` → Save.
2. **Import the dashboard**: Dashboards → New → Import → upload
   ``examples/observability/grafana/provisioning/dashboards/lmcache.json``
   and select the Prometheus datasource. This is the same dashboard the
   bundled stack provisions (cache hit rate, L1/L2 cache ops, L1↔L2
   throughput, eviction loop, EventBus health, and more).

Verify the pipeline end to end:

.. code-block:: bash

    # target should be "up"
    curl -s localhost:9090/api/v1/targets | grep -o '"health":"[a-z]*"'

    # after driving traffic, L2 store throughput (GB/s) per backend:
    curl -s localhost:9090/api/v1/query --data-urlencode \
      'query=sum by (l2_name) (rate(lmcache_mp_l2_store_throughput_GB_per_second_sum[1m]))
             / sum by (l2_name) (rate(lmcache_mp_l2_store_throughput_GB_per_second_count[1m]))'

See :doc:`metrics` for the full metric catalog and more PromQL examples.

.. note::

   ``--network host`` (used above) is the simplest option on Linux. On Docker
   Desktop (macOS/Windows), drop ``--network host``, publish ports with
   ``-p 9090:9090`` / ``-p 3000:3000``, and set the scrape target to
   ``host.docker.internal:8080`` and the Grafana datasource URL to
   ``http://host.docker.internal:9090``.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--disable-observability``
     - off
     - Master switch: disable the EventBus entirely (no metrics, logging, or
       tracing subscribers are registered).
   * - ``--disable-metrics``
     - off
     - Skip metrics subscribers (Prometheus endpoint is not started).
   * - ``--disable-logging``
     - off
     - Skip logging subscribers.
   * - ``--enable-tracing``
     - off
     - Register tracing subscribers. Requires ``--otlp-endpoint``.
   * - ``--event-bus-queue-size``
     - ``10000``
     - Maximum events in the EventBus queue before tail-drop.
   * - ``--otlp-endpoint``
     - *(none)*
     - OTLP gRPC endpoint (e.g. ``http://localhost:4317``). Used for
       exporting metrics (push mode) and traces.
   * - ``--prometheus-port``
     - ``9090``
     - Port of the standalone Prometheus ``/metrics`` server. Started only by
       frontend-less entrypoints (``python -m lmcache.v1.multiprocess.server``,
       ``lmcache trace replay``). **Ignored by** ``lmcache server`` — there the
       HTTP frontend serves ``/metrics`` on ``--http-port`` instead, so the
       standalone server is disabled. See :ref:`mp-obs-metrics-endpoint`.
   * - ``--http-port``
     - ``8080``
     - Port of the HTTP frontend, which serves the Prometheus ``/metrics``
       endpoint in pull mode (when ``--otlp-endpoint`` is unset) for
       ``lmcache server``.
   * - ``--metrics-sample-rate``
     - ``0.01``
     - Fraction of chunks/blocks to track for lifecycle histograms
       (0, 1.0]. Counters always count all events. Default is 1%.
   * - ``--enable-extra-logging``
     - off
     - Periodically log per-GPU L0<->L1 store/retrieve throughput and token
       counts, plus L1 memory usage, at INFO level. Conflicts with
       ``--disable-observability``. See :doc:`logs`.
   * - ``--extra-logging-interval``
     - ``10.0``
     - Seconds between extra-logging emissions. Values below ``1.0`` are
       limited by the 1 Hz internal heartbeat.
   * - ``--trace-level``
     - *(none)*
     - Enable trace recording at the given level. Currently only
       ``storage`` is supported (records ``StorageManager`` public-API
       calls for offline replay). When unset, trace recording is off.
       See :ref:`trace-recording` for details.
   * - ``--trace-output``
     - *(none)*
     - Path to write the trace file. If omitted while ``--trace-level``
       is set, a timestamped file under ``$TMPDIR`` is minted
       (``lmcache-trace-<pid>-<UTC>.lct``) and its path is logged at INFO.

**Environment variables:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``LMCACHE_LOG_LEVEL``
     - ``INFO``
     - Controls the log level for all LMCache loggers. Valid values:
       ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.


.. toctree::
   :maxdepth: 1

   metrics
   logs
   traces
