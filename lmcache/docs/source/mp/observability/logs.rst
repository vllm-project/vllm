Logging
=======

Logging subscribers emit debug-level messages for store, retrieve, lookup,
L1, and StorageManager events via Python's standard ``logging`` module.

When OpenTelemetry is installed, ``init_logger`` automatically attaches an
OTel ``LoggingHandler`` so that log records are forwarded to any configured
OTel ``LoggerProvider``. The handler respects the ``LMCACHE_LOG_LEVEL``
environment variable.

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server ...

Key log messages:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Level
     - Message
   * - INFO
     - ``Stored N tokens in X seconds``
   * - INFO
     - ``Retrieved N tokens in X seconds``
   * - INFO
     - ``Prefetch request completed (L1+L2): N/M prefix hits``
   * - DEBUG
     - ``MP store start: session=... device=...``
   * - DEBUG
     - ``MP retrieve end: session=... retrieved_count=...``

GC Monitoring (garbage-collection pauses)
-----------------------------------------

A full (generation-2) collection is stop-the-world and walks the whole
heap, so it lands inside request handling as tail latency that nothing else
in the server accounts for. Opt in to time collections and log the slow
ones:

.. code-block:: bash

    lmcache server ... --enable-gc-monitor --gc-monitor-min-pause-ms 10 --gc-monitor-top-objects 3

.. code-block:: text

    GC gen2 173.2ms collected=0 uncollectable=0 builtins.function=86911 builtins.tuple=70493 builtins.dict=45107

Notes:

- ``--gc-monitor-min-pause-ms`` (default ``1.0``) drops faster collections.
  ``0`` logs everything, including the sub-millisecond gen-0 sweeps CPython
  runs roughly every 700 net container allocations.
- ``--gc-monitor-top-objects`` shows *what* the collector is walking, but
  scans the whole generation on **every** collection (O(heap), can itself
  cost hundreds of ms). Debugging only.
- Non-zero ``uncollectable`` means CPython found garbage it could not free —
  a reference-cycle leak worth chasing.
- The monitor logs directly instead of publishing events, so it is
  independent of the ``--disable-observability`` / ``--disable-metrics`` /
  ``--disable-logging`` flags.

Extra Logging (periodic transfer stats)
---------------------------------------

Opt-in periodic INFO-level stats, aimed at watching a running server from
its logs without a metrics stack:

.. code-block:: bash

    lmcache server ... --enable-extra-logging --extra-logging-interval 10

Every interval, the server logs — per GPU — the L0<->L1 store/retrieve
activity of the last window and the cumulative totals since start
(operation count, tokens, size in GB, and average copy throughput in GB/s):

.. code-block:: text

    L0<->L1 stats [cuda:0] last 10.0s: store ops=12 tokens=24576 size=1.50GB avg_copy=12.34GB/s | retrieve ops=3 tokens=6144 size=0.40GB avg_copy=18.10GB/s
    L0<->L1 stats [cuda:0] cumulative: store ops=120 tokens=245760 size=15.02GB avg_copy=12.10GB/s | retrieve ops=31 tokens=63488 size=4.11GB avg_copy=17.92GB/s

The eviction loop also reports L1 memory usage at the same interval:

.. code-block:: text

    L1 memory usage: 45.20/100.00 GiB (45.2%)

Notes:

- Transfer-stat lines are only emitted for windows with activity; the L1
  memory usage line is emitted every interval regardless of load.
- ``avg_copy`` measures throughput of the GPU-stream copy time only (not
  wall clock), and shows ``n/a`` when no timed operation completed in the
  window.
- ``--extra-logging-interval`` values below ``1.0`` are limited by the 1 Hz
  internal heartbeat.
- ``--enable-extra-logging`` works even with ``--disable-logging``, but
  conflicts with ``--disable-observability`` (the server raises a
  ``ValueError`` at startup).

