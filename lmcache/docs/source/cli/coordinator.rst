lmcache coordinator
===================

The ``lmcache coordinator`` command launches the LMCache MP **coordinator**, a
standalone HTTP service that tracks the MP server instances in a deployment. MP
servers register with it and send periodic heartbeats; the coordinator evicts
any instance whose heartbeat lapses past ``--instance-timeout``.

It replaces ``python -m lmcache.v1.mp_coordinator``. The process runs in the
foreground; stop it with ``Ctrl-C``.

.. code-block:: bash

   lmcache coordinator [options]

Quick start
-----------

.. code-block:: bash

   lmcache coordinator \
       --host 0.0.0.0 --port 9300 \
       --instance-timeout 30 \
       --health-check-interval 10

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Description
   * - ``--host HOST``
     - Bind address for the coordinator's HTTP server (default: ``0.0.0.0``).
   * - ``--port PORT``
     - HTTP port (default: ``9300``).
   * - ``--instance-timeout SECS``
     - Seconds without a heartbeat after which an instance is evicted
       (default: ``30``).
   * - ``--health-check-interval SECS``
     - Seconds between health-check sweeps; ``0`` disables the loop
       (default: ``10``).
   * - ``--eviction-check-interval SECS``
     - Seconds between L2 eviction sweeps; ``0`` disables the loop
       (default: ``5``).
   * - ``--eviction-ratio RATIO``
     - Fraction of tracked keys (by count) to evict per cycle, ``0.0`` to
       ``1.0`` (default: ``0.2``).
   * - ``--trigger-watermark RATIO``
     - Eviction fires when usage reaches this fraction of the quota, ``0.0``
       (exclusive) to ``1.0`` (default: ``1.0``).
   * - ``--chunk-size N``
     - Tokens per KV chunk: the CacheBlend match unit and the unit used to
       resolve pin ``token_ids`` to keys. Must equal the MP servers'
       ``--chunk-size`` (default: ``256``).
   * - ``--hash-algorithm NAME``
     - Token hash algorithm for pin key resolution; must equal the MP servers'
       ``--hash-algorithm``. ``blake3`` (default) is self-contained; other
       algorithms require vLLM importable in the coordinator.
   * - ``--blend-probe-stride N``
     - Positions between CacheBlend match probes; ``1`` probes every offset
       for full recall (default: ``1``).
   * - ``--timeout-keep-alive SECS``
     - Seconds the HTTP server keeps idle connections open before closing
       them. Must be greater than the MP servers' heartbeat interval
       (default ``5``), otherwise heartbeat requests may hit a closing
       connection and fail with ``Server disconnected without sending a
       response`` (default: ``10``).

Configuration
-------------

Every flag is optional. Unset flags fall back to the
``LMCACHE_MP_COORDINATOR_*`` environment variables (``HOST``, ``PORT``,
``INSTANCE_TIMEOUT``, ``HEALTH_CHECK_INTERVAL``, ``EVICTION_CHECK_INTERVAL``,
``EVICTION_RATIO``, ``TRIGGER_WATERMARK``, ``CHUNK_SIZE``, ``HASH_ALGORITHM``,
``BLEND_PROBE_STRIDE``, ``TIMEOUT_KEEP_ALIVE``), and then to the built-in
defaults. A supplied flag always overrides the matching env-derived value, so
env-only deployments keep working unchanged.

A second set of env-only knobs controls the startup L2 resync —
``LMCACHE_MP_COORDINATOR_ENABLE_STARTUP_RESYNC`` (default ``True``),
``LMCACHE_MP_COORDINATOR_RESYNC_POLL_INTERVAL`` (``1``),
``LMCACHE_MP_COORDINATOR_RESYNC_MAX_WAIT`` (``60``), and
``LMCACHE_MP_COORDINATOR_RESYNC_PAGE_SIZE`` (``1000``). See
:doc:`/mp/coordinator` for the boot-time resync flow and the active
eviction loop.

The coordinator drives fleet-wide L2 eviction by calling each MP
server's ``DELETE /l2`` endpoint, and resync paginates ``GET /l2/keys``
on a registered MP server. Both endpoints are documented at
:ref:`mp-http-l2-keys-api`.

See :doc:`/mp/coordinator` for the coordinator's architecture, registration
protocol, and HTTP API.
