Multi-Server Coordination
=========================

When you run more than one LMCache multiprocess (MP) server, the **MP
Coordinator** is a standalone service they register with, giving you a single,
fleet-wide view of every running server. Each MP server caches independently;
the coordinator ties them together into one coordinated fleet.

.. contents::
   :local:
   :depth: 2

Running the coordinator
-----------------------

The coordinator is a FastAPI service. Start it with:

.. code-block:: bash

    lmcache coordinator

Expected log output:

.. code-block:: text

    LMCache INFO: MP coordinator listening on http://0.0.0.0:9300

The CLI accepts ``--host``, ``--port``, ``--instance-timeout``,
``--health-check-interval``, ``--eviction-check-interval``,
``--eviction-ratio``, ``--trigger-watermark``, ``--chunk-size``,
``--hash-algorithm``, ``--blend-probe-stride``, and ``--timeout-keep-alive``;
any flag overrides the matching environment variable below. See
:doc:`/cli/coordinator` for details.
Equivalently, the coordinator can still be launched as a module with
``python3 -m lmcache.v1.mp_coordinator``.

Configuration
-------------

The coordinator is configured through ``LMCACHE_MP_COORDINATOR_*`` environment
variables:

.. list-table::
   :header-rows: 1
   :widths: 38 14 48

   * - Environment variable
     - Default
     - Description
   * - ``LMCACHE_MP_COORDINATOR_HOST``
     - ``0.0.0.0``
     - Host the HTTP server binds to.
   * - ``LMCACHE_MP_COORDINATOR_PORT``
     - ``9300``
     - Port the HTTP server binds to.
   * - ``LMCACHE_MP_COORDINATOR_INSTANCE_TIMEOUT``
     - ``30``
     - Seconds without a heartbeat after which a server is dropped from the
       fleet.
   * - ``LMCACHE_MP_COORDINATOR_HEALTH_CHECK_INTERVAL``
     - ``10``
     - Seconds between health-check sweeps that expire stale MP-server
       registrations. ``0`` disables the stale-instance eviction loop; it does
       **not** affect the ``/quota`` L2 eviction loop (see
       ``LMCACHE_MP_COORDINATOR_EVICTION_CHECK_INTERVAL`` below).
   * - ``LMCACHE_MP_COORDINATOR_EVICTION_CHECK_INTERVAL``
     - ``5``
     - Seconds between L2 eviction sweeps. ``0`` disables the loop.
   * - ``LMCACHE_MP_COORDINATOR_EVICTION_RATIO``
     - ``0.2``
     - Fraction of tracked keys (by count) to evict per cycle (0.0 to 1.0).
   * - ``LMCACHE_MP_COORDINATOR_TRIGGER_WATERMARK``
     - ``1.0``
     - Eviction fires when usage reaches this fraction of the quota
       (0.0 exclusive to 1.0).
   * - ``LMCACHE_MP_COORDINATOR_CHUNK_SIZE``
     - ``256``
     - Tokens per KV chunk: the CacheBlend match unit and the unit used to
       resolve pin ``token_ids`` to keys. Must equal the MP servers'
       ``--chunk-size``.
   * - ``LMCACHE_MP_COORDINATOR_HASH_ALGORITHM``
     - ``blake3``
     - Token hash algorithm for pin key resolution. Must equal the MP servers'
       ``--hash-algorithm``. ``blake3`` is self-contained; other algorithms
       require vLLM importable in the coordinator process.
   * - ``LMCACHE_MP_COORDINATOR_BLEND_PROBE_STRIDE``
     - ``1``
     - Positions between CacheBlend match probes. ``1`` probes every offset
       for full recall.
   * - ``LMCACHE_MP_COORDINATOR_TIMEOUT_KEEP_ALIVE``
     - ``10``
     - Seconds the HTTP server keeps idle connections open before closing
       them. Must be greater than the MP servers' heartbeat interval
       (default ``5``), otherwise heartbeat requests may hit a closing
       connection and fail with ``Server disconnected without sending a
       response``.
   * - ``LMCACHE_MP_COORDINATOR_ENABLE_STARTUP_RESYNC``
     - ``True``
     - When ``True``, the coordinator runs a one-shot L2 resync on
       startup that paginates an MP server's ``GET /cache/objects`` and
       backfills usage + eviction trackers from existing L2 contents.
       Disable to start from empty trackers (handy for tests, or
       deployments that start the coordinator before any MP server).
   * - ``LMCACHE_MP_COORDINATOR_RESYNC_POLL_INTERVAL``
     - ``1``
     - Seconds between registry checks while waiting for the first
       MP server to register so startup resync can begin.
   * - ``LMCACHE_MP_COORDINATOR_RESYNC_MAX_WAIT``
     - ``60``
     - Maximum seconds startup resync waits for an MP server before
       giving up. The coordinator keeps running with empty trackers
       until normal usage events fill them in.
   * - ``LMCACHE_MP_COORDINATOR_RESYNC_PAGE_SIZE``
     - ``1000``
     - ``page_size`` forwarded to the MP server's ``GET /cache/objects``
       during resync. Larger values reduce RTT count; the server
       clamps to its own ceiling.

Connecting MP servers
---------------------

An MP server (``lmcache server``) joins the coordinator when you point it at one
with ``--coordinator-url``. It registers on startup, heartbeats while running,
and deregisters on shutdown -- all on the server's own event loop. This is
opt-in: with no URL set, the server runs exactly as before. Each flag falls back
to a matching ``LMCACHE_COORDINATOR_*`` environment variable (handy for the
Kubernetes downward API); an explicit flag wins over the env var.

.. list-table::
   :header-rows: 1
   :widths: 38 24 38

   * - Flag (on the MP server)
     - Env fallback
     - Description
   * - ``--coordinator-url``
     - ``LMCACHE_COORDINATOR_URL``
     - Coordinator base URL, e.g. ``http://coordinator:9300``. Enables
       registration when set.
   * - ``--coordinator-advertise-ip``
     - ``LMCACHE_COORDINATOR_ADVERTISE_IP``
     - IP the coordinator should reach this server at (defaults to the server's
       outbound IP).
   * - ``--coordinator-heartbeat-interval``
     - ``LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL``
     - Seconds between heartbeats (must be ``> 0``, default ``5``). Keep it well
       below the coordinator's ``INSTANCE_TIMEOUT``.
   * - ``--coordinator-event-reporting``
     - ``LMCACHE_COORDINATOR_EVENT_REPORTING``
     - Enable reporting cache events to the coordinator: the key
       directory's store/access/delete stream (fleet-wide placement
       tracking) and the L2 usage stream (quota tracking and eviction).
   * - ``--coordinator-event-flush-interval``
     - ``LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL``
     - Seconds between cache-event batch flushes (must be ``> 0``, default
       ``1``).

The server registers under its stable identity (``--instance-id`` / OTel
``service.instance.id``); if the flag is not passed, the server mints a
random UUID v4 at startup and registers under that.

Registration is best-effort: if the coordinator is unreachable, the MP server
logs a warning, keeps retrying, and continues serving. A malformed
heartbeat-interval value is rejected at startup.

HTTP endpoints
--------------

The coordinator's HTTP surface (base URL ``http://localhost:9300``) groups into:

- **Fleet membership and health** -- registration and liveness
  (``/instances``, ``/healthz``).
- **Quota, usage, and eviction** -- the ``/quota`` group: per-tenant byte
  budgets, usage accounting, and the usage-event ingest that drives fleet-wide
  eviction.
- **Cache control** -- the ``/cache`` group: cache operations dispatched to a
  named server (warm prefetch, pin/unpin, and delete, with more to come).
- **CacheBlend fingerprint directory** -- the ``/blend`` group: the fleet-wide
  fingerprint index blend-enabled MP servers publish to on STORE and query on
  LOOKUP. Server-to-coordinator only; not usually called by hand.

Each endpoint is documented below. Success is ``200`` unless noted, and
``{cache_salt}`` uses the ``_default`` sentinel for the empty salt. The wire
types live in ``lmcache/v1/mp_coordinator/schemas.py``.

Fleet membership and health
---------------------------

MP servers register, heartbeat, and deregister automatically (see
`Connecting MP servers`_); ``GET /instances`` and ``GET /healthz`` are read-only
operator views.

``POST /instances``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register (or re-register) an MP server. Called automatically by each server on
startup.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``ip``
     - string
     - IP/host of the server's HTTP API; the coordinator dials this address, so
       it must be non-empty.
   * - ``http_port``
     - int
     - Port of the server's HTTP API.
   * - ``instance_id``
     - string
     - Optional. Server identifier; if omitted (or blank) the coordinator
       generates one and returns it.
   * - ``metadata``
     - object
     - Optional. Free-form ``string -> string`` registration hints.
   * - ``p2p_advertised_url``
     - string
     - Optional. URL the server advertises for peer-to-peer transfers; empty
       when it is not in P2P.
   * - ``mq_port``
     - int
     - Optional (default ``0``). ZMQ message-queue port P2P peers send
       lookup/unlock RPCs to; ``0`` when P2P is disabled.

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1", "re_registered": false}

``instance_id`` is the registered id (the generated one when the request omitted
it); ``re_registered`` is ``true`` when this replaced an existing registration.

**HTTP status codes:**

- ``200``: registered.
- ``422``: request body fails field-level validation (e.g. blank ``ip`` or
  out-of-range ``http_port``).

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/instances \
        -H 'Content-Type: application/json' \
        -d '{"ip": "10.0.0.5", "http_port": 8080}'
    # -> {"instance_id": "mp-3f2c9d...", "re_registered": false}

``PUT /instances/{instance_id}/heartbeat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record a liveness heartbeat. Called automatically while the server runs.

**Path parameters:** ``instance_id`` — the instance recording the heartbeat.

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1"}

**HTTP status codes:**

- ``200``: heartbeat recorded.
- ``404``: unknown instance — the caller should re-register via
  ``POST /instances``.

**Example:**

.. code-block:: bash

    curl -s -X PUT http://localhost:9300/instances/server-1/heartbeat
    # -> {"instance_id": "server-1"}

``DELETE /instances/{instance_id}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deregister an MP server. Called automatically on shutdown.

**Path parameters:** ``instance_id`` — the server to deregister.

**Response:** ``204 No Content`` with an empty body, returned whether or not the
instance was registered (idempotent).

**HTTP status codes:**

- ``204``: deregistered (also returned for an unknown instance).

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:9300/instances/server-1 -o /dev/null -w '%{http_code}\n'
    # -> 204

``GET /instances``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List every registered MP server.

**Response** (``200 OK``):

.. code-block:: json

    {
      "instances": [
        {
          "instance_id": "server-1",
          "ip": "10.0.0.5",
          "http_port": 8080,
          "registration_time": 1719000000.0,
          "metadata": {},
          "p2p_advertised_url": "",
          "mq_port": 0
        }
      ]
    }

Each entry reports the server's ``instance_id``, the ``ip`` / ``http_port`` the
coordinator reaches it at, the wall-clock ``registration_time`` (epoch seconds),
any ``metadata`` supplied at registration, and the ``p2p_advertised_url`` /
``mq_port`` used for peer-to-peer transfers (empty / ``0`` when P2P is disabled).

**HTTP status codes:**

- ``200``: fleet listed (an empty fleet returns ``{"instances": []}``).

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/instances

``GET /healthz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coordinator liveness probe (for Kubernetes).

**Response** (``200 OK``):

.. code-block:: json

    {"status": "healthy"}

**HTTP status codes:**

- ``200``: the coordinator is up.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/healthz
    # -> {"status": "healthy"}

Quota, usage, and eviction
--------------------------

The ``/quota`` group owns per-``cache_salt`` byte budgets, the live usage
accounting behind them, and the usage-event stream that drives fleet-wide
eviction. (The MP server exposes a node-local ``/quota`` with the same shape;
this is its fleet-wide counterpart.) Use ``_default`` as the path parameter to
target the empty-string salt.

.. warning::

   Do **not** use the MP server's node-local ``/quota`` API together with the
   coordinator's. The two are independent, unsynchronized quota registries
   enforcing eviction on the **same shared L2**: the server-side enforcer
   (active when the server runs a per-salt eviction policy) uses strict
   allowlist semantics — any salt missing from *its own* table is fully
   evicted — and it never sees quotas registered on the coordinator, and vice
   versa. Mixing the two produces competing eviction decisions: the server can
   wipe data the coordinator considers within quota (or still exempt before
   the default limit is armed). Pick one owner per deployment — in
   coordinator-managed deployments, register quotas **only** through the
   coordinator's ``/quota`` API and leave the servers' node-local quota tables
   untouched.

Salts without an explicit quota are governed by the registry's **default
limit** (``PUT /quota/config``). On boot the default is unset, and unquota'd
salts are **exempt** from eviction — quotas live in memory, so a freshly
(re)started coordinator has an empty quota table until the external quota
controller re-syncs it, and the exempt default keeps that window from
mass-evicting unknown tenants. After re-registering every per-salt quota, the
controller sets the default to ``0`` — the signal that arms strict allowlist
enforcement (all bytes under unquota'd salts become evictable on the next
cycle):

.. code-block:: bash

    # 1. re-register every tenant quota
    curl -s -X PUT http://localhost:9300/quota/user-a \
        -H 'Content-Type: application/json' -d '{"limit_gb": 10.0}'
    # ... one PUT per tenant ...

    # 2. arm eviction of everything else
    curl -s -X PUT http://localhost:9300/quota/config \
        -H 'Content-Type: application/json' -d '{"default_limit_gb": 0}'
    # -> {"default_limit_gb": 0.0}

When MP servers enable ``--coordinator-event-reporting``, they stream L2
``store``, ``lookup``, and ``delete`` events to the coordinator, which aggregates
per-``cache_salt`` usage, enforces quotas, and selects LRU keys to evict. Each
batch carries the server's ``instance_id`` and a monotonically increasing
sequence number (``seq``) scoped to that instance, enabling future gap detection.

**Active eviction loop.** Every
``LMCACHE_MP_COORDINATOR_EVICTION_CHECK_INTERVAL`` seconds, the
coordinator inspects per-salt usage against the registered quotas and,
for any salt over the trigger watermark, picks LRU victims and
dispatches a single ``DELETE /cache/objects`` to a uniformly random registered MP
server. Because all MP servers share the same backing L2 (e.g. one S3
bucket), one dispatch evicts the keys for the whole fleet. The MP
server's L2 adapter fires ``on_l2_keys_deleted`` listeners after the
delete completes; those listeners ship ``delete`` events back through
``POST /quota/events``, which is what updates the coordinator's LRU +
per-salt totals. Dispatch failures or no-instances-registered fall
through to the next cycle — at-least-once semantics, safe because the
S3 delete is idempotent.

**Startup resync.** On boot, the coordinator waits up to
``LMCACHE_MP_COORDINATOR_RESYNC_MAX_WAIT`` seconds for the first MP
server to register, then paginates its
``GET /cache/objects`` and seeds the in-memory usage + eviction trackers
with whatever is already resident in L2 — so a fresh coordinator
does not start from zero usage. Set
``LMCACHE_MP_COORDINATOR_ENABLE_STARTUP_RESYNC=False`` to skip this
phase. Best-effort: resync failures are logged and the manager gives
up; the ongoing usage-event stream from MP servers eventually corrects
any initial blind spots.

``PUT /quota/config`` / ``GET /quota/config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set / read the default limit applied to salts with no explicit quota entry.

**Request body** (``PUT``):

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``default_limit_gb``
     - float or null
     - ``null`` (the boot default) exempts unquota'd salts from eviction;
       ``0`` arms strict allowlist enforcement (all unquota'd bytes become
       evictable next cycle); a positive value grants every unquota'd salt
       that byte budget.
   * - ``tier``
     - string
     - Optional (default ``l2``). Only ``l2`` is supported today.

**Response** (``200 OK``):

.. code-block:: json

    {"default_limit_gb": 0.0}

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/quota/config
    # -> {"default_limit_gb": null}          (boot state: unquota'd exempt)

    curl -s -X PUT http://localhost:9300/quota/config \
        -H 'Content-Type: application/json' -d '{"default_limit_gb": 0}'
    # -> {"default_limit_gb": 0.0}           (allowlist enforcement armed)

``PUT /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create or update a tenant's byte budget.

**Path parameters:** ``cache_salt`` — tenant identifier (``_default`` for the
empty salt).

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Type
     - Description
   * - ``limit_gb``
     - float
     - Byte budget in GiB; must be ``>= 0`` (``0`` clears the tenant's data on
       the next eviction cycle).
   * - ``tier``
     - string
     - Optional (default ``l2``). Cache tier the quota applies to; only ``l2`` is
       supported today.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "user-a", "limit_gb": 10.0, "status": "ok"}

**HTTP status codes:**

- ``200``: quota applied.
- ``400``: invalid limit (negative or non-finite).
- ``422``: request body fails field-level validation.

**Example:**

.. code-block:: bash

    curl -s -X PUT http://localhost:9300/quota/user-a \
        -H 'Content-Type: application/json' \
        -d '{"limit_gb": 10.0}'
    # -> {"cache_salt": "user-a", "limit_gb": 10.0, "status": "ok"}

``DELETE /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove a salt's quota entry. Any bytes still cached under it become over-budget
on the next eviction cycle (effective limit drops to ``0``).

**Path parameters:** ``cache_salt`` — tenant identifier (``_default`` for the
empty salt).

**Query parameters:** ``tier`` — optional (default ``l2``); cache tier the quota
applies to.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "user-a", "limit_gb": 0.0, "status": "removed"}

When no quota was registered for the salt, ``status`` is ``"not_found"`` (still
``200 OK``).

**HTTP status codes:**

- ``200``: removed, or ``not_found`` if no quota existed.

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:9300/quota/user-a
    # -> {"cache_salt": "user-a", "limit_gb": 0.0, "status": "removed"}

``GET /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the quota and live usage for a single salt.

**Path parameters:** ``cache_salt`` — tenant identifier (``_default`` for the
empty salt).

**Query parameters:** ``tier`` — optional (default ``l2``).

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}

``quota_limit_gb`` is the configured limit in GiB (``0.0`` when no quota is set),
``quota_exists`` whether an explicit quota is registered, and ``usage_gb`` the
current aggregate usage. This endpoint never returns ``404`` for an unknown salt.

**HTTP status codes:**

- ``200``: quota and usage reported.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/quota/user-a
    # -> {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}

``GET /quota``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List total usage and a per-salt breakdown.

**Query parameters:** ``tier`` — optional (default ``l2``).

**Response** (``200 OK``):

.. code-block:: json

    {
      "total_gb": 0.005,
      "by_cache_salt": [
        {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}
      ]
    }

``total_gb`` is aggregate usage across all salts in GiB; each ``by_cache_salt``
entry has the same fields as the ``GET /quota/{cache_salt}`` response.

**HTTP status codes:**

- ``200``: usage reported.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/quota
    # -> {"total_gb": 0.005, "by_cache_salt": [...]}

``POST /quota/events``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingest a batch of usage events. Sent automatically by reporting MP servers; not
usually called by hand.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Field
     - Type
     - Description
   * - ``instance_id``
     - string
     - The MP server that produced this batch.
   * - ``seq``
     - int
     - Monotonic per-instance sequence number (``>= 1``); supports future gap
       detection of lost batches.
   * - ``tier``
     - string
     - Optional (default ``l2``). Cache tier the events apply to.
   * - ``events``
     - list[object]
     - The events to record. Each is ``{"type", "key", "bytes"}``: ``type`` is
       ``"store"``, ``"lookup"``, or ``"delete"``; ``key`` is the encoded object
       key; ``bytes`` (``>= 0``) is the stored size — counted for ``store`` and
       ignored for ``lookup`` / ``delete`` (a ``delete`` subtracts the size
       recorded at the original ``store``).

**Response** (``200 OK``):

.. code-block:: json

    {"recorded": 3}

``recorded`` is the number of events processed.

**HTTP status codes:**

- ``200``: events processed.
- ``422``: request body fails field-level validation.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/quota/events \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "seq": 1,
            "events": [
                {"type": "store",  "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 1024},
                {"type": "lookup", "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 0},
                {"type": "delete", "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 0}
            ]
        }'
    # -> {"recorded": 3}

Cache control
-------------

The ``/cache`` group dispatches cache operations to a named MP server. It covers
**warm prefetch**, **pin/unpin**, and **delete**; further cache-control
operations will be documented as endpoints here as they land.

**Warm prefetch (pre-loading L1 from L2).** Pre-warm one MP server's L1 with the
KV for a known prompt **before** the requests arrive, so the first request hits
L1 instead of paying the L2 fetch inline -- useful when you know a workload is
about to be routed to a node (a traffic shift, a hot shared system prompt).

You describe the content by **token ids** -- the unit the cache speaks -- never
by internal cache keys, which you cannot construct (a key is a content hash
plus a per-rank layout bitmap). The coordinator forwards the request to the
named server, which hashes the tokens, expands them across the node's ranks,
loads the chunks from L2 into L1, and **retains** them so a later lookup hits.
The submit returns a ``request_id``; poll the status endpoint until
``completed``. The warm acquires no lock -- the poll simply reports progress and
clears the server-side job once the load finishes.

``POST /cache/prefetches``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Submit a warm prefetch of a token sequence on one named server.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Field
     - Type
     - Description
   * - ``instance_id``
     - string
     - Target MP server; must be registered.
   * - ``model_name``
     - string
     - Model whose layout sizes the target's L1 buffers.
   * - ``world_size``
     - int
     - World size (``>= 1``) selecting the KV layout and the per-rank fan-out
       (``1`` for a single-GPU, TP=1 deployment).
   * - ``token_ids``
     - list[int]
     - Prompt tokens whose complete ``chunk_size`` chunks are warmed; must match
       what was stored (same tokenizer / special tokens). A sub-chunk sequence
       is a ``noop``.
   * - ``cache_salt``
     - string
     - Optional (default ``""``). Per-tenant isolation salt applied to the
       produced keys.

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1", "request_id": "abc123", "chunks": 12, "status": "submitted"}

When the sequence is shorter than one chunk, nothing is submitted and
``request_id`` is empty:

.. code-block:: json

    {"instance_id": "server-1", "request_id": "", "chunks": 0, "status": "noop"}

``request_id`` is the id to poll; ``chunks`` is the number of whole chunks
submitted to warm.

**HTTP status codes:**

- ``200``: submitted (or a ``noop`` as above).
- ``404``: unknown ``instance_id`` (not registered).
- ``502``: the target server was unreachable or rejected the submit.
- ``422``: request body fails field-level validation.

.. note::

   **Single-node scope:** one ``instance_id`` warms only that node's shards. For
   a model sharded across multiple nodes, submit one request per node's instance.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/cache/prefetches \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"instance_id": "server-1", "request_id": "abc123", "chunks": 12, "status": "submitted"}

``GET /cache/prefetches/{instance_id}/{request_id}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poll a submitted warm prefetch; the response relays the owning server's status
verbatim with its code.

**Path parameters:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``instance_id``
     - string
     - The server the prefetch was submitted to.
   * - ``request_id``
     - string
     - The id returned by ``POST /cache/prefetches``.

**Response** (``200 OK``) while the load runs:

.. code-block:: json

    {"status": "pending"}

…and once complete:

.. code-block:: json

    {"status": "completed", "found_keys": 12, "total_keys": 12}

``found_keys`` of ``total_keys`` requested chunks were resident.

**HTTP status codes:**

- ``200``: status reported (``pending`` or ``completed``).
- ``404``: unknown ``instance_id``, or unknown ``request_id`` relayed from the
  server.
- ``502``: the target server was unreachable.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/cache/prefetches/server-1/abc123
    # -> {"status": "completed", "found_keys": 12, "total_keys": 12}

**Pin/unpin (protecting cache from eviction).** Pin a token sequence's cache so
it is not evicted from L2 until unpinned. The coordinator resolves the token
sequence to its object keys **locally** (no MP-server round-trip) and records
them in its L2 eviction plan (``POST``) or releases them (``DELETE``), excluding
pinned keys from quota-based eviction. L2 pins are fleet-wide (per
``cache_salt``), so no target instance is named.

Local resolution requires the coordinator's ``chunk_size`` and
``hash_algorithm`` (see `Configuration`_) to match the MP servers' ``--chunk-size``
/ ``--hash-algorithm``; otherwise the resolved keys will not match what was
stored and the pin protects nothing. It also requires the MP servers to be
launched with ``--no-separate-object-groups`` (the coordinator resolves keys in
a single object group).

``POST /cache/pins``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pin a token sequence's keys in the L2 eviction plan.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Field
     - Type
     - Description
   * - ``model_name``
     - string
     - Model whose rank fan-out to use when resolving keys.
   * - ``world_size``
     - int
     - World size (``>= 1``) selecting the per-rank fan-out.
   * - ``token_ids``
     - list[int]
     - Prompt tokens whose complete chunks are pinned; must match what was
       stored. A sub-chunk sequence pins nothing (``affected`` 0).
   * - ``cache_salt``
     - string
     - Optional (default ``""``). Per-tenant isolation salt.

**Response** (``200 OK``):

.. code-block:: json

    {"requested": 12, "affected": 12, "status": "pinned"}

``requested`` is the number of whole chunks resolved; ``affected`` is the number
of L2 keys pinned (chunks times the per-rank fan-out).

**HTTP status codes:**

- ``200``: pinned.
- ``400``: ``token_ids`` exceeds the per-request cap, or ``cache_salt`` violates
  its invariants.
- ``422``: request body fails field-level validation.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/cache/pins \
        -H 'Content-Type: application/json' \
        -d '{
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"requested": 12, "affected": 12, "status": "pinned"}

.. note::

   **Requires event reporting.** The coordinator can only exclude keys from
   eviction for a salt it is tracking, which requires the MP servers started with
   ``--coordinator-event-reporting`` (see `Connecting MP servers`_).

``DELETE /cache/pins``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unpin a token sequence's keys from the L2 eviction plan. Same request body as
``POST /cache/pins``. The response mirrors the pin (``affected`` is the number
of keys unpinned), with ``status`` ``"unpinned"``. Pins are reference-counted: a
chunk pinned *N* times needs *N* unpins before it can be evicted.

**HTTP status codes:** same as ``POST /cache/pins``.

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:9300/cache/pins \
        -H 'Content-Type: application/json' \
        -d '{
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"requested": 12, "affected": 12, "status": "unpinned"}

**Delete (removing cache by token sequence).** Delete a token sequence's cache
on one named server, addressed by token ids. The coordinator resolves the tokens
to object keys locally (like pin) and issues a single key-addressed
``DELETE /cache/objects`` to the named server, which removes them from the
requested tier(s). The ``tier`` field selects the tier(s): ``l1`` deletes only
the named server's L1, ``l2`` only L2, ``all`` both. When the tier includes L2,
the coordinator first drops any key it is protecting with an L2 pin from the
delete set unless ``force`` is set — so a pinned key is retained in every tier
the delete would have touched; ``force`` deletes them and drops those pins.

``POST /cache/delete``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete a token sequence on one named server.

**Request body:** (``model_name``,
``world_size``, ``token_ids``, ``cache_salt``) plus ``tier`` (``l1`` / ``l2`` /
``all``) and ``force`` (bool, default ``false``). When ``force`` is ``true``,
locked keys are deleted anyway (L1 read/write locks on the node and the
coordinator's L2 pin set).

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1", "requested": 12, "affected": 24, "skipped": 0, "status": "deleted"}

``requested`` is the number of whole chunks resolved. ``affected`` and
``skipped`` are **totals across the tiers acted on**: ``affected`` counts L1 keys
removed by the node plus L2 keys removed by the coordinator, and ``skipped``
counts L1 keys the node refused plus L2 keys held back for an L2 pin (non-force
only). A chunk resident in both tiers (``tier=all``) contributes to both counts,
so ``affected`` may be up to ``2 x requested x world_size``. A sub-chunk
sequence returns ``status`` ``"noop"``.

**HTTP status codes:**

- ``200``: deleted (or a ``noop``).
- ``404``: no server is registered under ``instance_id``.
- ``502``: the target server was unreachable or rejected the delete.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/cache/delete \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a",
            "tier": "all",
            "force": false
        }'
    # -> {"instance_id": "server-1", "requested": 12, "affected": 24, "skipped": 0, "status": "deleted"}

CacheBlend fingerprint directory
--------------------------------

The ``/blend`` group is the fleet-wide fingerprint index behind CacheBlend
cross-request reuse. Blend-enabled MP servers publish chunk fingerprints on
STORE (``POST /blend/fingerprints``) and query them on LOOKUP
(``POST /blend/match``); the index is chunked at
``LMCACHE_MP_COORDINATOR_CHUNK_SIZE`` tokens and rolling-hash-matched at
``LMCACHE_MP_COORDINATOR_BLEND_PROBE_STRIDE``. These endpoints are
server-to-coordinator; they are not usually called by hand.

The wire types (``StoreRangeModel``, ``BlendFingerprintRequest``,
``BlendMatchRequest``, ``GlobalMatchModel`` and their responses) live in
``lmcache/v1/mp_coordinator/schemas.py``.

``POST /blend/fingerprints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register stored chunk fingerprints (idempotent).

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``ranges``
     - list
     - Stored token ranges. Each entry carries ``model_scope`` (the reuse
       scope, typically the model name), ``tokens`` (the raw stored token
       ids), ``object_keys`` (the shared-L2 storage key per chunk, in order),
       and ``old_st_base`` (the token position of the range's first token).
       The directory chunks ``tokens`` at the coordinator's chunk size and
       hashes each chunk; chunk ``i`` maps to ``object_keys[i]``.

**Response** (``200 OK``):

.. code-block:: json

    {"inserted": 3}

``inserted`` reports how many fingerprints were newly registered (existing
entries are left in place; re-publishing is safe).

**HTTP status codes:**

- ``200``: fingerprints processed.
- ``422``: request body fails field-level validation.

``DELETE /blend/fingerprints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evict fingerprints by shared-L2 storage key (idempotent). MP servers call this
when the underlying L2 objects are dropped, so the directory does not keep
handing out stale matches.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``object_keys``
     - list[string]
     - Storage keys whose fingerprint entries should be removed. Unknown keys
       are ignored.

**Response** (``200 OK``):

.. code-block:: json

    {"removed": 2}

``removed`` reports how many entries were actually evicted.

**HTTP status codes:**

- ``200``: keys processed.
- ``422``: request body fails field-level validation.

``POST /blend/match``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Match a request's token buffer against the directory and return the reusable
chunks.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``model_scope``
     - string
     - Reuse scope to match within (typically the model name).
   * - ``tokens_b64``
     - string
     - Request tokens packed as base64 little-endian ``uint32`` (see
       ``encode_tokens`` / ``decode_tokens`` in ``schemas.py``). The
       coordinator hashes them at
       ``LMCACHE_MP_COORDINATOR_CHUNK_SIZE`` positions striding by
       ``LMCACHE_MP_COORDINATOR_BLEND_PROBE_STRIDE``.

**Response** (``200 OK``):

.. code-block:: json

    {
      "matches": [
        {"object_key": "ab12...", "old_st": 0,    "cur_st": 512},
        {"object_key": "cd34...", "old_st": 256,  "cur_st": 768}
      ]
    }

Each entry names one reusable chunk: ``object_key`` is the shared-L2 key,
``old_st`` is its token position in the stored sequence (re-RoPE source), and
``cur_st`` is the position in the request (re-RoPE target). Matches are sorted
ascending by ``cur_st``. An empty request or an unknown ``model_scope``
returns ``{"matches": []}``.

**HTTP status codes:**

- ``200``: match completed (an empty match list is not an error).
- ``422``: ``tokens_b64`` is not valid base64 or not a whole number of
  ``uint32`` tokens.
