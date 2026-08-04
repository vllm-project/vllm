Tracing and Debugging
=====================

LMCache MP mode can record every ``StorageManager`` public-API call to a
binary **trace file** and reissue those calls later against a fresh
server via ``lmcache trace replay``. The feature is designed for:

- **Regression hunting** — capture a production workload, then replay it
  against a build under investigation to reproduce a bug offline.
- **Performance characterization** — measure L1/L2 latency distributions
  under a realistic storage-level access pattern, without needing vLLM
  or a GPU.
- **Configuration tuning** — replay the same trace against different
  L1 sizes, eviction policies, and L2 adapters to compare their
  behavior on identical input.

.. note::

   Trace recording is **independent** from ``--enable-tracing``
   (OTel spans). OTel tracing exports *live* spans to an OTLP
   endpoint for online observability; trace recording persists a
   replayable binary file for offline analysis. Both can be enabled
   simultaneously.

.. _trace-recording-guide:

Recording a Trace
-----------------

Recording is **off by default**. Enable it by adding
``--trace-level storage`` to ``lmcache server``:

.. code-block:: bash

    # Explicit output path
    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --trace-level storage --trace-output /tmp/run.lct

    # Implicit timestamped path under $TMPDIR
    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --trace-level storage
    # → INFO log: "trace recording enabled (level=storage); no
    #   --trace-output given, writing to
    #   /tmp/lmcache-trace-<pid>-<UTC>.lct"

Drive traffic through the server as usual (vLLM requests, benchmark
scripts, etc.). The trace file is closed cleanly on ``SIGTERM`` via
the EventBus stop path — no ``--stop-tracing`` command needed.

**What is captured:**

- The fully-qualified name of every decorated ``StorageManager`` call
  (e.g. ``StorageManager.reserve_write``,
  ``StorageManager.submit_prefetch_task``).
- Each call's input arguments (``keys``, ``layout_desc``, ``mode``,
  ``extra_count``, ``external_request_id``, …).
- Wall-clock and monotonic timestamps per call.
- A header with the file format version, trace schema version, start
  timestamps, and a SHA-256 digest of the active
  ``StorageManagerConfig`` so replay can flag mismatched
  configurations.

**What is not captured:**

- **KV tensor bytes.** Replay exercises bookkeeping and controller
  logic; payloads at replay time are zeros. The trace file stays
  bounded even for long runs.
- Calls inside ``MPCacheServer``, the message queue, or GPU-copy code.
  Those layers are out of scope for the ``storage`` trace level.

**Overhead:**

- Off: a single boolean check per ``StorageManager`` call. Effectively free.
- On: encoding and file I/O happen on the EventBus drain thread, off
  the request path. In practice this has no visible impact on
  request latency.

Inspecting a Trace
------------------

Before replaying, ``lmcache trace info`` prints a one-screen summary:

.. code-block:: bash

    lmcache trace info /tmp/run.lct

.. code-block:: text

    Trace file: /tmp/run.lct
      level:                storage
      format_version:       1
      trace_schema_version: 1
      duration:             226.691s
      sm_config_digest:     0f685d8a...
      total_records:        1318
      ops:
        lmcache.v1.distributed.storage_manager.StorageManager.finish_read_prefetched: 133
        lmcache.v1.distributed.storage_manager.StorageManager.finish_write: 349
        lmcache.v1.distributed.storage_manager.StorageManager.read_prefetched_results.__enter__: 96
        lmcache.v1.distributed.storage_manager.StorageManager.read_prefetched_results.__exit__: 96
        lmcache.v1.distributed.storage_manager.StorageManager.reserve_write: 349
        lmcache.v1.distributed.storage_manager.StorageManager.submit_prefetch_task: 295

Use this to sanity-check that the trace you intend to replay covers
the expected operation mix and duration.

Replaying a Trace
-----------------

``lmcache trace replay FILE`` reissues every recorded call against a
**fresh** ``StorageManager`` built from CLI flags you supply. The
replay-side config is **chosen by you**, not copied from the
recording. This is the feature's main value — you can compare
different L1/L2 setups on identical input.

Minimal invocation:

.. code-block:: bash

    lmcache trace replay /tmp/run.lct \
        --l1-size-gb 100 --eviction-policy LRU

``--l1-size-gb`` and ``--eviction-policy`` are required, just like on
``lmcache server``. Any storage-manager flag accepted by the server
also works here (``--l2-adapter``, ``--l1-use-lazy``,
``--l2-store-policy``, …); run ``lmcache trace replay --help`` for the
full list.

**Pacing.** The driver always honors the recorded inter-call timings
by sleeping to align each dispatch with its recorded ``t_mono``
offset. There is **no** as-fast-as-possible mode: ``StorageManager``
reads and writes are asynchronous and carry cross-call dependencies
(for example, a retrieve may depend on an earlier L2 load completing),
so collapsing the recorded gaps races the internal queues and causes
non-deterministic retrieve misses. If the replay host is slower than
the recording host, the loop simply lags the recorded schedule.

**Output.** Every replay prints a terminal metrics table and writes
a per-qualname CSV by default:

.. code-block:: text

    =================== Trace Replay Result ======================
    --------------------------- Overall --------------------------
    Trace level:                                         storage
    Records replayed:                                       1318
    Records skipped:                                           0
    Records failed:                                            0
    Replay duration (s):                                  226.69
    Config digest:                          match (0f685d8a)
    --------------------- Per-Op Latency (ms) --------------------
    reserve_write count:                                     349
    reserve_write mean:                                     0.16
    reserve_write p50:                                      0.13
    reserve_write p99:                                      0.93
    ...

Additional per-record output is controlled by:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Purpose
   * - ``--output-dir DIR``
     - Directory for aggregated summary files. Default: current dir.
   * - ``--no-csv``
     - Skip the ``trace_replay_ops.csv`` export.
   * - ``--json``
     - Also write ``trace_replay_summary.json`` (per-qualname
       count / mean / p50 / p90 / p99 / min / max, plus total
       duration).
   * - ``--verbose``
     - Print one ``[N/total] OK|FAIL <qualname> (Xms)`` line per
       record to stdout in addition to the INFO log.
   * - ``--jsonl-out PATH``
     - Write one JSON object per replayed record to ``PATH``
       (``{qualname, latency_ms, failed}``) for post-hoc
       analysis.
   * - ``-q`` / ``--quiet``
     - Suppress the terminal metrics table. The aggregated
       files are still written.

Even without ``--verbose``, the driver logs each dispatch at INFO:

.. code-block:: text

    [1/1318] OK lmcache...StorageManager.reserve_write (0.252ms)
    [2/1318] OK lmcache...StorageManager.finish_write (0.032ms)
    ...

Progress numbers come from a cheap pre-scan of the trace file, so you
always see ``[N/total]`` rather than just a running counter.

Monitoring During Replay
------------------------

The replay driver initializes the full observability EventBus
**before** constructing the replay-side ``StorageManager``. Internal
events (L1/L2 lifecycle, eviction ticks, store/retrieve publishes,
etc.) therefore flow through a live bus during replay and the
standard subscribers — logging, metrics, OTel tracing — can attach
to them.

The same observability CLI flags that the server accepts are
available on ``lmcache trace replay``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Effect
   * - ``--disable-observability``
     - Turn the EventBus off entirely. No subscribers fire.
   * - ``--disable-metrics``
     - Skip OTel metrics init and metrics subscribers. Useful to
       avoid binding the Prometheus port when you only want logs.
   * - ``--disable-logging``
     - Skip logging subscribers.
   * - ``--enable-tracing``
     - Enable OTel span subscribers. Requires ``--otlp-endpoint``.
   * - ``--otlp-endpoint URL``
     - Export metrics/traces to an OTLP gRPC collector (e.g.
       ``http://localhost:4317``). When unset, metrics fall back
       to the in-process Prometheus pull endpoint.
   * - ``--prometheus-port PORT``
     - Port for the Prometheus ``/metrics`` endpoint in pull mode.
       Default ``9090``.
   * - ``--metrics-sample-rate FLOAT``
     - Sampling rate for lifecycle histograms. Counters always
       count all events.

Typical monitoring setups:

**Raw log trail (SM/L1/L2 events to stdout):**

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache trace replay /tmp/run.lct \
        --l1-size-gb 100 --eviction-policy LRU \
        --disable-metrics

**Prometheus metrics in pull mode:**

.. code-block:: bash

    lmcache trace replay /tmp/run.lct \
        --l1-size-gb 100 --eviction-policy LRU \
        --prometheus-port 9095
    # scrape http://localhost:9095/metrics from another terminal

**OTel metrics + traces to a collector:**

.. code-block:: bash

    lmcache trace replay /tmp/run.lct \
        --l1-size-gb 100 --eviction-policy LRU \
        --otlp-endpoint http://localhost:4317 \
        --enable-tracing

.. note::

   The ``--trace-level`` and ``--trace-output`` flags are **recording-only**
   and are not accepted by ``lmcache trace replay``. A replay never
   writes a new trace file.

Notes, Hints, and Caveats
-------------------------

**Retrieve misses are expected when the replay environment differs.**
At replay start, the CLI prints a visible warning banner:

.. code-block:: text

    ==============================================================================
      !! REPLAY ENVIRONMENT MISMATCH MAY CAUSE RETRIEVE MISSES !!
    ==============================================================================

Because KV payloads are not captured and the replay-side config and
host speed may differ from recording, retrieve calls that hit at
record time can miss at replay time — for instance, an async L2 load
that had finished by the time the recorded retrieve was issued may
still be in flight when the replayed retrieve fires. Treat
retrieve-miss counts as a signal about the replay environment, **not**
as a defect in the trace.

**Config-digest mismatch is informational, not fatal.** The replay
always runs whether the digests match or not. A mismatch simply tells
you the replay-side ``StorageManagerConfig`` differs from what was
recorded — often exactly what you intended (comparing two configs
on the same trace).

**Prometheus port binding.** The server's ``--prometheus-port``
defaults to ``9090``. Running ``lmcache trace replay`` concurrently
with the server — or running two replays at once — on the same port
will fail. Either pass a different ``--prometheus-port`` or
``--disable-metrics`` on the secondary runs.

**Trace-recording overhead.** Recording happens on the EventBus drain
thread, not the request-handling threads. The gate is a single
boolean check when disabled (default), so production builds with
recording off pay no measurable cost.

**Trace files are not encrypted.** Arguments such as ``ObjectKey``
chunk hashes are written in plaintext. Treat trace files with the
same care as cache hash logs.

**Forward compatibility.** The header carries a format version and a
trace schema version. Readers reject files with unknown versions
rather than silently producing garbage. Captured API surface
changes (new arguments on a traced method, new codec tags) bump the
schema version; framing changes bump the format version.

**Extensibility.** The format is designed to accommodate future
trace **levels** (``mq``, ``gpu``). Adding a new traced method in an
existing level requires only decorating it on the recording side and
registering a handler on the replay side — no format changes.

See Also
--------

- :ref:`trace-recording` — the short ``Trace Recording`` section in
  the Observability page focuses on the recording-side flags.
- ``docs/design/v1/mp_observability/trace.md`` in the source tree —
  full design doc: architecture, replay dispatcher, context-manager
  pairing, stats collector, and test matrix.
