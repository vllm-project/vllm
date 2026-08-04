Tracing
=======

.. note::

   ``--enable-tracing`` **requires** ``--otlp-endpoint`` to be set.
   The server will refuse to start if tracing is enabled without an
   OTLP endpoint, since there is no local fallback for trace export.

When tracing is enabled (``--enable-tracing --otlp-endpoint <URL>``),
the tracing subscriber creates OTel spans from START/END event pairs:

- ``mp.store`` — from ``MP_STORE_START`` to ``MP_STORE_END``
- ``mp.retrieve`` — from ``MP_RETRIEVE_START`` to ``MP_RETRIEVE_END``
- ``mp.lookup_prefetch`` — from ``MP_LOOKUP_PREFETCH_START`` to ``MP_LOOKUP_PREFETCH_END``

Each span carries event metadata as span attributes (e.g. ``device``,
``stored_count``, ``found_count``).

View traces in any OTel-compatible backend such as **Jaeger** or
**Grafana Tempo**.

.. code-block:: bash

    # Start Jaeger all-in-one (OTLP gRPC on 4317)
    docker run -d --name jaeger \
        -p 16686:16686 -p 4317:4317 \
        jaegertracing/all-in-one:latest

    # Start LMCache with tracing
    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --enable-tracing --otlp-endpoint http://localhost:4317

Per-Request Hit-Rate Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each session is wrapped in a per-request root span — ``request`` for the
standard MP path and ``cb.request`` for the CacheBlend path — that nests
all child spans (``mp.store``, ``mp.retrieve``, ``mp.lookup_prefetch``)
beneath it.  When the lookup phase ends, the root span is annotated with
three OTel attributes that summarise the request-level cache hit rate:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - OTel type
     - Description
   * - ``hit_tokens``
     - ``int``
     - Tokens served from L1+L2 (numerator).
   * - ``requested_tokens``
     - ``int``
     - Chunk-aligned tokens submitted for lookup (denominator).
   * - ``hit_rate``
     - ``float``
     - ``hit_tokens / requested_tokens``; ``0.0`` when the denominator is
       zero.  Stored as a precomputed float because trace UIs (Tempo,
       Jaeger) cannot derive it from two integer attributes at query time.

The attributes are written when ``MP_LOOKUP_PREFETCH_END`` (standard MP
path) or ``CB_LOOKUP_END`` (CacheBlend path) is processed — while the
root span is still open.  **Store-only requests** that never call
``lookup_prefetch_start()`` emit no end event for the lookup phase, so
their root span will not carry these attributes.

Example TraceQL queries (Grafana Tempo):

.. code-block:: text

    # Requests with less than 50% cache hit rate
    { name = "request" && span.hit_rate < 0.5 }

    # Full cache hits only
    { name = "request" && span.hit_rate = 1.0 }

    # Complete misses (lookup ran but nothing was cached)
    { name = "request" && span.requested_tokens > 0 && span.hit_tokens = 0 }

For the full event-to-span mapping and the registry pattern that links
child spans back to the root see
``docs/design/observability/request-event-span.md`` in the source tree.

.. _trace-recording:

Trace Recording
~~~~~~~~~~~~~~~

.. note::

   Trace recording is **distinct from** ``--enable-tracing`` (OTel
   spans). Trace recording captures every ``StorageManager`` public-API
   call to a binary file so the same workload can be **replayed** later
   for testing, regression hunting, and benchmarking — without needing
   vLLM and (eventually) without a GPU. ``--enable-tracing`` exports
   live OTel spans to an OTLP endpoint for online observability.
   The two features are independent and can be used together.

When ``--trace-level storage`` is set, LMCache records every call to
``StorageManager.{reserve_write, finish_write, submit_prefetch_task,
read_prefetched_results, finish_read_prefetched}`` to a binary file
for later replay.

Recording is **off by default** and adds near-zero overhead when off
(a single boolean check per ``StorageManager`` call). When on,
recording happens on the EventBus drain thread, off the request path.

Capturing a trace
^^^^^^^^^^^^^^^^^

With an explicit output path:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --trace-level storage --trace-output /tmp/run.lct

With an implicit timestamped output path under ``$TMPDIR``:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --trace-level storage
    # → INFO log: "trace recording enabled (level=storage); no
    #   --trace-output given, writing to
    #   /tmp/lmcache-trace-<pid>-<UTC>.lct"

The trace file is closed cleanly on shutdown (SIGTERM is handled by
the EventBus stop path).

Replay
^^^^^^

Replaying a recorded trace, plus the full set of CLI flags for
driving, monitoring, and exporting replay results, is covered in
its own page: :doc:`/mp/tracing_and_debugging`.

What is captured (and what is not)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Captured:**

- The fully-qualified name of every decorated ``StorageManager`` call.
- Each call's input arguments (e.g. ``keys``, ``layout_desc``,
  ``mode``, ``extra_count``, ``external_request_id``).
- Wall-clock and monotonic timestamps of each call.
- A header carrying a trace schema version, start times, and a
  SHA-256 digest of the active ``StorageManagerConfig`` so replay can
  detect mismatched configurations.

**Not captured:**

- KV tensor bytes. Replay exercises bookkeeping and controller logic;
  payloads at replay time are zeros.
- Calls inside the ``MPCacheServer``, the message queue, or any
  GPU-copy code. These layers are **out of scope** for the storage
  trace level.

File format
^^^^^^^^^^^

A length-prefixed `msgpack <https://msgpack.org/>`_ stream:

::

    [4-byte big-endian length][msgpack Header]
    [4-byte big-endian length][msgpack Record]
    [4-byte big-endian length][msgpack Record]
    ...

The ``Header`` carries a magic prefix (``LMCT``), a format version,
the trace level (``storage`` today), a trace schema version, start
timestamps, and the StorageManagerConfig digest. Each ``Record``
carries a relative timestamp, a wall-clock timestamp, the
fully-qualified call site (``qualname``), and an argument dict.

The format is deliberately extensible: future trace **levels**
(``mq``, ``gpu``) will share this layout and use the ``level`` header
field to discriminate. Additional captured ops add new ``qualname``
strings without bumping the format version.

For the full design rationale see

.. toctree::
   :hidden:
   :maxdepth: 1

   /mp/tracing_and_debugging
