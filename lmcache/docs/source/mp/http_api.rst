HTTP API
========

When the MP server is started via ``lmcache server`` (the recommended entry
point), a FastAPI-based HTTP frontend is exposed alongside the ZMQ socket
used by vLLM. This HTTP API is intended for operators, orchestrators
(e.g. Kubernetes), and debugging tools — it is **not** on the inference
data path.

Where the routes come from
--------------------------

Routes are assembled from three sources, all merged into one FastAPI app by
:class:`~lmcache.v1.multiprocess.http_api_registry.HTTPAPIRegistry` at startup:

- **MP-native routes** — any module named ``*_api.py`` under
  ``lmcache/v1/multiprocess/http_apis/`` that exposes a module-level
  ``router`` (a :class:`fastapi.APIRouter`) is auto-discovered. This covers
  the operational surface: status, cache control, L2 management, quota, and
  runtime reconfiguration.
- **Shared "common" routes** —
  ``lmcache/v1/multiprocess/http_apis/common_api.py`` aggregates every
  compatible router under ``lmcache/v1/internal_api_server/common/`` (skipping
  any module listed in ``_MP_INCOMPATIBLE_MODULES``, currently empty) and
  forwards them to the auto-discovery pipeline. These are the cross-server
  diagnostics shared with the vLLM-embedded API server (``/env``,
  ``/loglevel``, ``/metrics``, ``/threads``, ``/periodic-threads*``,
  ``/run_script``). Adding a new compatible module under
  ``internal_api_server/common`` requires no wiring changes on the MP side.
- **Re-exported version routes** — the basic-info group
  ``lmcache/v1/multiprocess/http_apis/info_api.py`` includes the router
  from ``lmcache/v1/internal_api_server/vllm/version_api.py``, exposing
  ``/version``, ``/lmc_version``, and ``/commit_id`` alongside ``/``,
  ``/healthcheck`` and ``/status``.

.. contents::
   :local:
   :depth: 2

Server Configuration
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--http-host``
     - ``0.0.0.0``
     - Host to bind the HTTP server.
   * - ``--http-port``
     - ``8080``
     - Port to bind the HTTP server.

Example:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --http-host 0.0.0.0 --http-port 8080

All examples below assume the server is reachable at
``http://localhost:8080``.

Endpoint Overview
-----------------

The routes are grouped by purpose below. The operational surface (health,
status, cache and storage control) lives at top-level paths; routes inherited
from the shared ``internal_api_server`` package keep their original paths for
compatibility with the vLLM-embedded API server.

.. note::

   Several handlers report failure in the response **body** rather than via a
   non-200 status code (e.g. ``DELETE /cache/objects`` returns ``200`` with ``ok=false``,
   and ``/periodic-threads-health`` returns ``200`` with ``healthy=false``).
   The error-field name is also not uniform: ``/healthcheck`` and
   ``/cache/clear`` use ``reason`` on failure, while ``/status``, ``/config``,
   and ``/cache/checksums`` use ``error``. Per-endpoint details below are
   authoritative.

**Liveness and health**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/``
     - Static liveness ping (does not touch the engine).
   * - GET
     - ``/healthcheck``
     - K8s liveness/readiness probe; ``503`` until the engine is initialized.

**Inspection and status**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/status``
     - Detailed engine snapshot (L1, L2, registered contexts, sessions,
       prefetch jobs) for inspection and debugging.
   * - GET
     - ``/config``
     - Dump the merged server configuration objects (``mp``,
       ``storage_manager``, ``observability``).
   * - GET
     - ``/config/adapters``
     - Enumerate the live cache adapters (``type_name``, ``tier``, ``primary``,
       ``reconfigurable``). Supersedes ``/reconfigure/backends``.
   * - GET
     - ``/version``
     - Combined version string (``"<version>-<commit_id>"``).
   * - GET
     - ``/lmc_version``
     - LMCache package version string.
   * - GET
     - ``/commit_id``
     - Build commit id.

**Cache management**

.. list-table::
   :header-rows: 1
   :widths: 10 38 52

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/cache/objects``
     - Paginate objects resident in a tier/adapter (query: ``tier``,
       ``adapter``, ``model_name``, ``page_size``, ``page_token``).
   * - DELETE
     - ``/cache/objects``
     - Delete a caller-supplied list of object keys from L1, L2, or both (body:
       ``keys``, ``tier``, ``force``, ``adapter``). ``force`` deletes L1 keys
       even if locked.
   * - POST
     - ``/cache/prefetches``
     - Warm a node's L1 by loading a token sequence from L2 ahead of traffic;
       returns a ``request_id``.
   * - GET
     - ``/cache/prefetches/{request_id}``
     - Poll a submitted warm prefetch (``pending`` / ``completed``).
   * - POST
     - ``/cache/clear``
     - Force-clear a tier's resident cache (body: ``tier`` = ``l1``, ``force``).
   * - POST
     - ``/cache/checksums``
     - Compute MD5 checksums over KV cache blocks (diagnostics / round-trip
       integrity checks).

**Quota management**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/quota``
     - List every registered ``cache_salt`` quota with live usage.
   * - PUT
     - ``/quota/{cache_salt}``
     - Set or update the quota (in GB) for a ``cache_salt``.
   * - GET
     - ``/quota/{cache_salt}``
     - Read the quota and live usage for a single ``cache_salt``.
   * - DELETE
     - ``/quota/{cache_salt}``
     - Remove a ``cache_salt``'s quota entry (its data is evicted next cycle).

**Runtime L2 reconfiguration**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/reconfigure/{backend}/status``
     - Report runtime-manageable L2 adapters for one backend type. (To discover
       reconfigurable backends, use ``GET /config/adapters`` and read the
       ``reconfigurable`` flag.)
   * - POST
     - ``/reconfigure/{backend}/{operation}``
     - Apply one runtime reconfiguration operation to a backend adapter.

**Observability**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/metrics``
     - Prometheus exposition format.
   * - POST
     - ``/metrics/reset``
     - Reset all observability metrics to their initial state.

**Diagnostics and debugging**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/loglevel``
     - List or inspect logger levels; also accepts ``level`` to mutate one.
   * - GET
     - ``/threads``
     - Enumerate active Python threads and their stack traces.
   * - GET
     - ``/periodic-threads``
     - List registered periodic threads with summary counts.
   * - GET
     - ``/periodic-threads/{thread_name}``
     - Detailed status for a single periodic thread.
   * - GET
     - ``/periodic-threads-health``
     - Quick health check for critical/high-level periodic threads.
   * - GET
     - ``/env``
     - Dump process environment variables (JSON body, ``text/plain``).
   * - POST
     - ``/run_script``
     - Execute an uploaded Python script in a restricted sandbox.

Liveness and Health
-------------------

``GET /``
~~~~~~~~~

Basic liveness check. Returns a static payload indicating the HTTP server
is running; it does **not** touch the cache engine. Use ``/healthcheck``
instead for probes that also verify the engine is initialized.

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "ok",
      "service": "LMCache HTTP API"
    }

**HTTP status codes:**

- ``200``: server is alive (unconditional; does not touch the engine).

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/

``GET /healthcheck``
~~~~~~~~~~~~~~~~~~~~

Health check endpoint suitable for Kubernetes liveness and readiness
probes. A ``200`` response means the HTTP server is alive **and** the MP
cache engine object is wired onto ``app.state``. A ``503`` response
indicates the engine is not yet present (still initializing, or failed to
initialize). The check verifies that the engine attribute is set; it does
not call into the engine to assert deeper liveness.

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "healthy"
    }

**Response** (``503 Service Unavailable``):

.. code-block:: json

    {
      "status": "unhealthy",
      "reason": "engine not initialized"
    }

**HTTP status codes:**

- ``200``: engine is wired onto ``app.state`` (healthy).
- ``503``: engine not initialized (still starting up, or failed to initialize).

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/healthcheck

**Kubernetes probe snippet:**

.. code-block:: yaml

    livenessProbe:
      httpGet:
        path: /healthcheck
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /healthcheck
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5

Inspection and Status
---------------------

``GET /status``
~~~~~~~~~~~~~~~

Returns a detailed snapshot of the MP engine's internal state. The payload is
assembled by ``MPCacheServer.report_status()``: a fixed set of engine-level
fields, the full storage-manager status, plus whatever keys each loaded module
contributes (so the exact key set depends on which modules are active —
``registered_gpu_ids`` / ``cache_context_meta`` come from the transfer module,
``active_prefetch_jobs`` from the lookup module, and blend modes add their own
fields). Intended for operators and debugging, not for monitoring (use
Prometheus metrics for time-series data — see :doc:`observability/index`).

**Response** (``200 OK``):

.. code-block:: json

    {
      "is_healthy": true,
      "engine_type": "MPCacheServer",
      "chunk_size": 256,
      "hash_algorithm": "builtin-hash",
      "active_sessions": 2,
      "registered_gpu_ids": [0, 1],
      "cache_context_meta": {
        "0": {
          "model_name": "meta-llama/Llama-3.1-8B-Instruct",
          "world_size": 1,
          "kv_cache_layout": {
            "num_layers": 32,
            "num_blocks": 12345,
            "cache_size_per_token": 131072,
            "kernel_groups": [
              {
                "kernel_group_idx": 0,
                "engine_group_idx": 0,
                "object_group_idx": 0,
                "num_layers": 32,
                "layer_indices": [0, 1, "..."],
                "tokens_per_block": 16,
                "slots_per_block": 16,
                "dtype": "torch.bfloat16",
                "engine_kv_concrete_shape": "...",
                "is_mla": false,
                "engine_kv_format": "...",
                "engine_kv_shape": "...",
                "attention_backend": "..."
              }
            ]
          }
        }
      },
      "active_prefetch_jobs": 0,
      "storage_manager": {
        "is_healthy": true,
        "...": "backend-specific fields"
      }
    }

**Response** (``503 Service Unavailable``) when the engine has not yet
been initialized:

.. code-block:: json

    {
      "error": "engine not initialized"
    }

**HTTP status codes:**

- ``200``: status snapshot returned.
- ``503``: engine not yet initialized on ``app.state``.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/status | jq

``GET /config``
~~~~~~~~~~~~~~~

Returns every server-side configuration object registered on
``app.state.configs`` (typically ``mp``, ``storage_manager`` and
``observability``) as a single indented JSON document. Dataclasses are
serialized via ``safe_asdict``; other values go through ``make_json_safe``.
Useful for confirming what the process actually loaded — including
environment overrides — without restarting.

**Response** (``200 OK``):

.. code-block:: json

    {
      "mp": {
        "http_host": "0.0.0.0",
        "http_port": 8080,
        "...": "..."
      },
      "storage_manager": {
        "...": "..."
      },
      "observability": {
        "...": "..."
      }
    }

**Response** (``503 Service Unavailable``) when configs are not wired
onto ``app.state`` yet:

.. code-block:: json

    {
      "error": "configs not initialized"
    }

**HTTP status codes:**

- ``200``: configuration document returned.
- ``503``: configs not yet wired onto ``app.state``.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/config | jq

``GET /config/adapters``
~~~~~~~~~~~~~~~~~~~~~~~~~

Enumerate every L2 adapter the engine has loaded, in configuration
order. Which storage backends are loaded is a configuration-inspection
concern, so this lives in the config group rather than under ``/cache``.
This is the single live adapter listing; it supersedes the old
``GET /reconfigure/backends`` (the reconfigurable backends are the
``type_name`` values whose ``reconfigurable`` flag is ``true``).

**Response** (``200 OK``):

.. code-block:: json

    {
      "adapters": [
        {"index": 0, "type_name": "S3L2Adapter", "tier": "l2", "primary": true, "reconfigurable": false},
        {"index": 1, "type_name": "dax", "tier": "l2", "primary": false, "reconfigurable": true}
      ]
    }

``primary`` is ``true`` only on the first entry. ``reconfigurable`` is
``true`` for adapters that accept ``/reconfigure`` operations — pass that
adapter's ``type_name`` as the ``{backend}`` path parameter to
``GET /reconfigure/{backend}/status`` and the reconfigure operations. An
engine that has no L2 backends returns ``{"adapters": []}`` (still ``200`` —
the engine is initialized, it just has no L2 storage).

**HTTP status codes:**

- ``200``: success (including the no-adapters case).
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/config/adapters | jq

``GET /version``, ``GET /lmc_version``, ``GET /commit_id``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version descriptors. Each returns a bare JSON **string** (not an object):

- ``GET /version`` — the combined descriptor from
  ``lmcache.utils.get_version()``, formatted ``"<version>-<commit_id>"``
  (e.g. ``"0.3.1-ca79ea33"``). On a source checkout without build-time
  version metadata, each missing component falls back to the literal
  ``"NA"`` (so a metadata-less build returns ``"NA-NA"``).
- ``GET /lmc_version`` — the raw package version string
  (``lmcache.utils.VERSION``); empty string ``""`` when the generated
  ``lmcache._version`` module is absent.
- ``GET /commit_id`` — the git commit id baked into the build
  (``lmcache.utils.COMMIT_ID``); empty string ``""`` when unavailable.

All three are unconditional ``200 OK``.

**Response** (``200 OK``):

.. code-block:: json

    "0.3.1-ca79ea33"

**HTTP status codes:**

- ``200``: version string returned (unconditional for all three routes).

**Examples:**

.. code-block:: bash

    curl -s http://localhost:8080/version
    curl -s http://localhost:8080/lmc_version
    curl -s http://localhost:8080/commit_id

Cache Management
----------------

``POST /cache/clear``
~~~~~~~~~~~~~~~~~~~~~~

Force-clears **all** KV cache data currently held in a tier (today ``l1``).

.. warning::

   This endpoint is destructive and bypasses read/write locks. In-flight
   store or prefetch operations may be corrupted. Use only when the
   server is idle, or when recovering from a known-bad cache state.

**Request body:** optional -- an absent (or empty) body uses the defaults below.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Type
     - Description
   * - ``tier``
     - string
     - Optional (default ``l1``). Tier to clear; today only ``l1`` is
       supported. Any other value returns ``400``.
   * - ``force``
     - bool
     - Optional (default ``true``). Currently accepted but **not honored** --
       the clear always force-clears (active locks are ignored) regardless of
       this value.

**Response** (``200 OK``):

.. code-block:: json

    {"status": "ok", "cleared": {"tier": "l1"}}

**HTTP status codes:**

- ``200``: tier cleared.
- ``400``: unsupported ``tier`` (anything other than ``l1``).
- ``503``: server not initialized (``{"detail": "server not initialized"}``).

**Example:**

.. code-block:: bash

    # the body is optional; this clears the default tier (l1)
    curl -s -X POST http://localhost:8080/cache/clear

``POST /cache/checksums``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute MD5 checksums over the engine KV cache, grouped ``chunk_size`` blocks
per hashed chunk. MP mode addresses KV storage by block IDs natively (the
same units used by ``STORE`` / ``RETRIEVE``), so the endpoint is fully
block-centric. Intended for diagnostics and round-trip integrity checks from
``lmcache bench server`` -- not for the inference data path.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Required
     - Description
   * - ``block_ids``
     - yes
     - Engine block IDs as a JSON list of ints, e.g. ``[0, 1, 2, 3]``.
   * - ``chunk_size``
     - yes
     - Positive integer — number of blocks per hashed chunk.
   * - ``instance_id``
     - no (default ``0``)
     - Registered KV context ID on the engine.
   * - ``layerwise``
     - no (default ``false``)
     - If ``true``, return per-layer checksums keyed by ``"layer_<idx>"``;
       otherwise a single aggregated digest per chunk over all layers.

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "success",
      "chunk_size": 2,
      "num_chunks": 2,
      "chunk_checksums": ["<md5>", "<md5>"],
      "layerwise": false,
      "block_id_ranges": "0,[2,5],8"
    }

When ``layerwise=true``, ``chunk_checksums`` is a dict keyed by
``"layer_<idx>"`` whose values are per-layer lists.

**HTTP status codes:**

- ``200``: success.
- ``400``: ``block_ids`` empty, or ``chunk_size`` missing or non-positive.
- ``404``: ``instance_id`` not registered, or the registered KV tensors
  are empty.
- ``501``: engine has no ``cache_contexts``, or the KV format is not
  supported by this endpoint (page-buffer-fused and cross-layer layouts
  are declined until a real need appears).
- ``503``: engine not yet initialized on ``app.state``.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/cache/checksums \
        -H 'Content-Type: application/json' \
        -d '{"block_ids": [0, 1, 2, 3], "chunk_size": 2}'

.. _mp-http-l2-keys-api:

Cache Objects And Prefetch
--------------------------

Two endpoints — ``DELETE /cache/objects`` and ``GET /cache/objects`` — let
operators purge keys from a configured cache backend and enumerate what is
currently resident. (To enumerate the configured backends themselves, use
``GET /config/adapters`` in the config group.)

A further pair — ``POST /cache/prefetches`` and
``GET /cache/prefetches/{request_id}`` — lets an operator (or the
coordinator) **pre-warm** a node's L1 from L2 ahead of traffic and poll
the load to completion; they are documented at the end of this section.
The coordinator exposes an ``instance_id``-routed variant of both — see
:doc:`coordinator`.

Both object endpoints take an optional ``adapter`` selector — a query
parameter on ``GET /cache/objects`` (``?adapter=<type_name>``) and a body
field on ``DELETE /cache/objects``. Omit it to target the **primary**
(first-configured) adapter. When multiple adapters share a ``type_name``,
the first match wins. Use ``GET /config/adapters`` to learn the valid
selectors.

Both are intended for operator / admin workflows ("purge this
user's keys", "show me what's resident", "garbage-collect orphans
after a rename"). They are **not** on the inference data path.

``DELETE /cache/objects`` deletes from L1, L2, or both, selected by its ``tier``
field (default ``l2``). ``GET /cache/objects`` lists L2 only.

The coordinator's eviction loop uses ``DELETE /cache/objects`` automatically (see
:doc:`coordinator` — "L2 usage tracking and eviction"); the
``GET /cache/objects`` endpoint also powers the coordinator's startup
resync. Manual ``curl`` usage is reserved for ad-hoc operator
actions and debugging.

``DELETE /cache/objects``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete a caller-supplied list of keys from L1, L2, or both (``tier``).
Idempotent: absent keys are skipped silently. L2 deletes skip keys locked by
in-flight store/load tasks so an active transfer is never corrupted; L1 deletes
skip read/write-locked keys unless ``force`` is set. Blocking adapter I/O is run
off the event loop. When the tier includes L2 the primary (or selected) L2
adapter must be configured, else ``503``; a pure ``l1`` delete needs no adapter.

Per-key successful L2 deletions fire ``on_l2_keys_deleted`` on the
adapter's listeners — when the coordinator is wired (see
``--coordinator-event-reporting``), the deletions show up at the
coordinator's ``POST /quota/events`` as ``"type": "delete"`` events. The
coordinator's eviction + usage trackers learn about the deletion from
that event flow, not from the response of this call.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Type
     - Description
   * - ``keys``
     - list[EncodedObjectKey]
     - Required. The object keys to delete (schema below). The batch is
       capped at ``10000`` keys per request.
   * - ``tier``
     - string
     - Optional (default ``l2``). One of ``l1`` / ``l2`` / ``all``.
   * - ``force``
     - bool
     - Optional (default ``false``). When ``true``, delete an L1 key even if it
       is read/write-locked (no effect on L2).
   * - ``adapter``
     - string
     - Optional (default: primary, first-configured adapter). The ``type_name``
       of the target L2 adapter (see ``GET /config/adapters``); ignored when
       ``tier`` is ``l1``.

Each ``EncodedObjectKey`` is

.. code-block:: json

    {
      "chunk_hash_hex": "abc123...",
      "model_name": "meta-llama/Llama-3-8B",
      "kv_rank": 0,
      "object_group_id": 0,
      "cache_salt": "user-a"
    }

``object_group_id`` (default ``0``) and ``cache_salt`` (default ``""``)
are optional for backward compatibility with older wire payloads.

**Response** (``200 OK``):

.. code-block:: json

    {
      "deleted": 4,
      "skipped": 0,
      "ok": true
    }

``deleted`` is the total keys removed across the requested tiers (L1 removals
plus the L2 batch size); ``skipped`` is the L1 keys refused because they were
locked (non-force only). On an L2 adapter failure the response is still ``200``
with ``ok=false`` and an ``error`` field carrying the reason.

**HTTP status codes:**

- ``200``: request processed (check ``ok`` for the L2 adapter outcome).
- ``400``: batch exceeds the limit, or a key payload violates an
  ``ObjectKey`` invariant (bad hex, ``@`` in ``model_name``, forbidden
  ``cache_salt`` character).
- ``404``: ``adapter`` (body) does not match any configured adapter.
- ``422``: Pydantic-level body-shape failure (missing ``keys``,
  wrong field types).
- ``503``: engine not initialized, or no L2 adapters configured (only when
  ``tier`` includes L2).

**Example:** delete from both tiers.

.. code-block:: bash

    curl -s -X DELETE http://localhost:8080/cache/objects \
        -H 'Content-Type: application/json' \
        -d '{
            "tier": "all",
            "keys": [
              {"chunk_hash_hex": "aa", "model_name": "m",
               "kv_rank": 0, "object_group_id": 0, "cache_salt": "user-a"}
            ]
        }'
    # -> {"deleted": 2, "skipped": 0, "ok": true}

``GET /cache/objects``
~~~~~~~~~~~~~~~~~~~~~~

Paginate keys currently resident in one L2 adapter.

**Query parameters:**

.. list-table::
   :header-rows: 1
   :widths: 22 13 65

   * - Name
     - Default
     - Description
   * - ``adapter``
     - primary
     - ``type_name`` of the target adapter (see ``GET /config/adapters``).
       Omit to target the primary (first-configured) adapter. First
       match wins when multiple adapters share a ``type_name``.
   * - ``model_name``
     - none
     - Restrict the result to keys whose ``model_name`` matches.
   * - ``page_size``
     - ``500``
     - Max entries per page. Must be in ``[1, 5000]``; an out-of-range
       value is rejected with ``422`` (it is not silently clamped).
   * - ``page_token``
     - none
     - Opaque cursor from the previous page's ``next_page_token``.
       Omit on the first call; pass back verbatim on subsequent calls.

The page token is private to the adapter; do not parse or modify it.
Adapters that support listing (currently only the S3 adapter via
``ListObjectsV2``) guarantee best-effort consistency, not snapshot
isolation — concurrent stores or deletes during a paginated walk may
cause keys to appear, disappear, or shift between pages.

**Response** (``200 OK``):

.. code-block:: json

    {
      "adapter": "S3L2Adapter",
      "entries": [
        {
          "key": {
            "chunk_hash_hex": "abc123",
            "model_name": "meta-llama/Llama-3-8B",
            "kv_rank": 0,
            "object_group_id": 0,
            "cache_salt": "user-a"
          },
          "size_bytes": 4194304
        }
      ],
      "next_page_token": "opaque-cursor-string"
    }

``next_page_token`` is ``null`` when the listing is exhausted.

**HTTP status codes:**

- ``200``: success.
- ``400``: malformed ``page_token`` (adapter-level).
- ``404``: ``?adapter=<name>`` does not match any configured adapter.
- ``422``: ``page_size`` outside ``[1, 5000]``.
- ``501``: selected adapter does not implement listing. In v1 only
  ``S3L2Adapter`` does; adapters wrapped by ``SerdeL2AdapterWrapper``
  inherit the wrapped adapter's behavior.
- ``503``: engine not initialized, or no L2 adapters configured.

**Example:** paginate every key for a model.

.. code-block:: bash

    next=""
    while :; do
      page=$(curl -s "http://localhost:8080/cache/objects?model_name=meta-llama/Llama-3-8B&page_size=500&page_token=$next")
      echo "$page" | jq '.entries[]'
      next=$(echo "$page" | jq -r '.next_page_token // empty')
      [ -z "$next" ] && break
    done

``POST /cache/prefetches``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warm one node's L1 by loading a token sequence's chunks from L2 **ahead** of
the requests that will use them, so the first request hits L1 instead of
paying the L2 fetch inline. Useful when a workload is about to be routed to a
node (a traffic shift, a hot shared system prompt).

The caller describes content by **token ids**, never by internal cache keys (a
key is a content hash plus a per-rank layout bitmap, which callers cannot
construct). The server hashes the tokens, expands each chunk across the node's
ranks, and submits a *warm* prefetch: loaded chunks are **retained**
(permanent) and left **unlocked** — there is no downstream reader to pin them
for, so a later real lookup takes its own lock. The call returns immediately;
the load runs in the storage manager's own thread. It coalesces across all
configured L2 adapters, so there is no ``?adapter=`` selector.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Field
     - Type
     - Description
   * - ``model_name``
     - string
     - The served model id, exactly as registered (e.g. ``Qwen/Qwen3-8B``).
   * - ``world_size``
     - int
     - The value vLLM registered for the node's KV layout and rank fan-out
       (``1`` for a single-GPU, TP=1 deployment).
   * - ``token_ids``
     - list[int]
     - The prompt token ids. Must use the same tokenizer / special-token
       settings as the store, and contain at least one full ``chunk_size`` of
       tokens — only complete chunks are warmed.
   * - ``cache_salt``
     - string
     - Per-tenant key isolation; must match the store (default ``""``).

**Response** (``202 Accepted``):

.. code-block:: json

    {"request_id": "abc123", "chunks": 12, "status": "submitted"}

When the sequence is shorter than one chunk, nothing is submitted and there is
no ``request_id`` to poll:

.. code-block:: json

    {"chunks": 0, "status": "noop"}

**HTTP status codes:**

- ``202``: submitted (or a ``noop`` as above).
- ``400``: ``token_ids`` exceeds the per-request cap, or ``cache_salt``
  violates its invariants.
- ``409``: no layout registered for ``(model_name, world_size)`` — the model
  has not allocated KV cache on this node yet (start vLLM first).
- ``422``: request body fails field-level validation.
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/cache/prefetches \
        -H 'Content-Type: application/json' \
        -d '{"model_name": "Qwen/Qwen3-8B", "world_size": 1,
             "token_ids": [101, 102, 103], "cache_salt": "user-a"}'
    # -> {"request_id": "abc123", "chunks": 1, "status": "submitted"}

``GET /cache/prefetches/{request_id}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poll a submitted warm prefetch. The warm holds no lock, so the poll only
reports progress; the first poll that observes completion drops the job
(exactly-once), so a later poll for the same id returns ``404``.

**Response** (``200 OK``) while the load runs:

.. code-block:: json

    {"status": "pending"}

…and once complete:

.. code-block:: json

    {"status": "completed", "found_keys": 12, "total_keys": 12}

``found_keys`` / ``total_keys`` count only chunks **loaded from L2** by this
request; chunks already resident in L1 are skipped at ``reserve_write`` and not
counted, so a partially-resident warm undercounts by the resident chunk count
(a cold request loads and counts everything). The warm uses the gap-tolerant
``SPARSE`` trim policy, so an already-resident chunk does not stop the rest from
loading — it loads every not-yet-resident chunk and reports that count. Not
counting the resident chunks is deliberate: an already-present entry may be a
transient temporary from another lookup, so claiming it as warmed could mislead.

**HTTP status codes:**

- ``200``: status reported (``pending`` or ``completed``).
- ``404``: unknown ``request_id`` — already completed-and-consumed, or never
  submitted.
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/cache/prefetches/abc123
    # -> {"status": "completed", "found_keys": 1, "total_keys": 1}

.. _mp-http-quota-api:

Quota Management
----------------

These endpoints manage the per-``cache_salt`` storage budgets consumed by
the ``IsolatedLRU`` eviction policy (selected via
``--eviction-policy IsolatedLRU``). Quotas are **soft**: setting a limit
does not reject writes — any over-budget ``cache_salt`` is evicted at
the next eviction cycle (~1 s).
A ``cache_salt`` with no registered quota has an effective limit of
``0`` bytes, so its data is cleared next cycle (allowlist semantics).

These endpoints are no-ops on engines that did not start with
``--eviction-policy IsolatedLRU``: the ``QuotaManager`` is still
present, but the LRU policy ignores the registered quotas.

**URL escaping for the empty salt.** ``cache_salt=""`` (un-salted /
anonymous traffic) cannot appear in a URL path parameter, so the API
accepts the sentinel ``_default`` in its place. ``PUT /quota/_default``
sets the quota for ``cache_salt=""``, and ``_default`` is echoed back in
responses for the empty salt. A user that legitimately stores data with
``cache_salt="_default"`` cannot be managed via this HTTP API distinctly
from anonymous traffic — both map to the same path parameter; pick any
other value (e.g. ``"default"``) to disambiguate.

``PUT /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create or update a quota.

**Path parameters:** ``cache_salt`` — tenant identifier (use ``_default``
for the empty salt; see the section intro).

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Type
     - Description
   * - ``limit_gb``
     - float
     - Required. The quota in GB. Must be finite and non-negative.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "alice", "limit_gb": 10.0, "status": "ok"}

**HTTP status codes:**

- ``200``: quota set or updated.
- ``400``: malformed JSON, missing ``limit_gb``, non-numeric ``limit_gb``,
  ``nan`` / ``inf``, or a negative value.
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s -X PUT http://localhost:8080/quota/alice \
        -H 'Content-Type: application/json' \
        -d '{"limit_gb": 10.0}'

``GET /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the current quota and live usage for one ``cache_salt``.

**Path parameters:** ``cache_salt`` — tenant identifier (use ``_default``
for the empty salt).

**Response** (``200 OK``):

.. code-block:: json

    {
      "cache_salt": "alice",
      "limit_gb": 10.0,
      "current_usage_gb": 2.137,
      "exists": true
    }

``exists`` is ``false`` when no quota was ever registered for this
``cache_salt`` (``limit_gb`` is then ``0.0`` and ``current_usage_gb``
reflects whatever bytes are currently cached for that salt — those bytes
will evict next cycle under ``IsolatedLRU``). This endpoint never returns
``404`` for an unknown salt.

**HTTP status codes:**

- ``200``: quota and usage returned (also when the salt has no registered
  quota — ``exists`` is then ``false``; never ``404``).
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/quota/alice | jq

``DELETE /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove a ``cache_salt``'s quota entry. Any bytes still cached under this
``cache_salt`` become over-budget on the next eviction cycle (effective
limit drops to ``0``) and will be evicted.

**Path parameters:** ``cache_salt`` — tenant identifier (use ``_default``
for the empty salt).

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "alice", "status": "removed"}

When no quota was registered for the given ``cache_salt``, the response
is ``{"cache_salt": "...", "status": "not_found"}`` (still ``200 OK``).

**HTTP status codes:**

- ``200``: quota entry removed (``"removed"``) or none existed
  (``"not_found"``); never ``404``.
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:8080/quota/alice

``GET /quota``
~~~~~~~~~~~~~~~~~~

List every registered quota alongside its live usage.

**Response** (``200 OK``):

.. code-block:: json

    {
      "users": {
        "alice": {"limit_gb": 10.0, "current_usage_gb": 2.137},
        "bob":   {"limit_gb":  4.0, "current_usage_gb": 0.812}
      }
    }

Only ``cache_salt`` values with a **registered** quota appear; the empty
salt is reported under the ``_default`` key.

**HTTP status codes:**

- ``200``: quota listing returned (``{"users": {}}`` when none registered).
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/quota | jq

.. _mp-http-dax-api:

Runtime L2 Reconfiguration
--------------------------

These endpoints are available when the server has a runtime-reconfigurable L2
adapter. They only change LMCache runtime mappings and metadata; backend
resources such as DAX device paths must already exist and be readable and
writable by the server. The endpoint routes ``backend``, ``operation``, and the
JSON request body into the generic L2 adapter reconfiguration API, while
backend-specific validation and migration semantics stay inside the adapter.

``backend`` and ``operation`` path segments are normalized (stripped and
lower-cased). Within a request body, ``adapter_index`` (default ``0``) is
**backend-local** — it indexes only the adapters of that backend, not the
engine-wide adapter list. If an L2 adapter is wrapped by serde, the backend
string is still the configured L2 adapter type, not the serde wrapper type.

.. note::

   The backend strings accepted here are discovered via
   ``GET /config/adapters``: each adapter whose ``reconfigurable`` flag is
   ``true`` can be addressed by its ``type_name``.

``GET /reconfigure/{backend}/status``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Report the runtime-manageable adapters for one backend type. Each adapter
entry's ``adapter_index`` is rewritten to its **backend-local** 0-based index
(the value to pass back in operation request bodies).

**Path parameters:** ``backend`` — adapter ``type_name`` (normalized:
stripped and lower-cased; discover valid values via ``GET /config/adapters``).

**Response** (``200 OK``):

.. code-block:: json

    {
      "enabled": true,
      "backend": "dax",
      "num_adapters": 1,
      "adapters": [
        {"adapter_index": 0, "...": "backend-specific adapter fields"}
      ]
    }

An unknown or empty backend returns ``enabled=false``, ``num_adapters=0``,
``adapters=[]`` (it is **not** a ``404``).

**HTTP status codes:**

- ``200``: status returned (including the unknown-backend case above).
- ``400``: ``backend`` is empty.
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/reconfigure/dax/status | jq

``POST /reconfigure/{backend}/{operation}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply one reconfiguration operation to a backend adapter. The request body is
a JSON object whose accepted fields depend on the backend and operation. The
``200`` response is whatever the storage manager's
``reconfigure_l2_adapter`` returns (a backend-defined dict).

**Path parameters:** ``backend`` (adapter ``type_name``) and ``operation``
(e.g. ``add`` / ``remove`` / ``resize`` for ``dax``). Both are normalized
(stripped and lower-cased).

For the **generic** path (any backend other than ``dax``), the body carries
``adapter_index`` plus any backend-specific fields, which are forwarded
verbatim to the adapter.

For **Device-DAX** (``backend=dax``), JSON request bodies are used because DAX
paths contain slashes. The accepted operations and fields are:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operation
     - Body fields
   * - ``add``
     - ``device_path`` (str, required), ``size`` (int byte count or string
       such as ``"100GiB"``, required), ``adapter_index`` (default ``0``).
   * - ``remove``
     - ``device_path`` (str, required), ``mode`` (``migrate`` | ``evict`` |
       ``drain``, default ``migrate``), ``force`` (bool, default ``false``),
       ``adapter_index`` (default ``0``).
   * - ``resize``
     - ``device_path`` (str, required), ``size`` (int or string, required),
       ``mode`` (``migrate`` | ``evict``, default ``migrate``), ``force``
       (bool, default ``false``), ``adapter_index`` (default ``0``).

``size`` accepts an integer byte count or a string with a base-1024 unit
suffix (``b``, ``kib``, ``mib``, ``gib``, ``tib`` and the ``k``/``m``/``g``/``t``
aliases), e.g. ``"100GiB"``; it must resolve to a positive value.

**Response** (``200 OK``):

The body is the backend's ``reconfigure_l2_adapter`` result (a backend-defined
dict). A successful DAX ``add`` looks like:

.. code-block:: json

    {
      "status": "ok",
      "operation": "add",
      "adapter_index": 0,
      "device": {"device_path": "/dev/dax0.0", "state": "active", "size_bytes": 107374182400}
    }

**HTTP status codes:**

- ``200``: success (body is the storage manager's reconfigure result).
- ``400``: empty ``backend``/``operation``, an unsupported DAX operation, or
  an invalid ``size``.
- ``404``: ``adapter_index`` is out of range for the backend.
- ``422``: request body fails validation (e.g. a missing required field, or
  an unknown field in a DAX body — DAX bodies reject extras).
- ``503``: engine not initialized.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/reconfigure/dax/add \
        -H 'Content-Type: application/json' \
        -d '{"device_path": "/dev/dax0.0", "size": "100GiB"}'

See :doc:`/kv_cache/storage_backends/dax` for detailed request examples,
mode semantics, and validation guidance.

Observability
-------------

``GET /metrics``
~~~~~~~~~~~~~~~~

Prometheus exposition format for every metric registered on the default
``prometheus_client`` registry (``Content-Type: text/plain``). Scrape this
directly from Prometheus. See :doc:`observability/index` for the list of
exported metrics.

**Response** (``200 OK``, ``text/plain``): the Prometheus exposition-format
metrics body.

**HTTP status codes:**

- ``200``: metrics returned.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/metrics

``POST /metrics/reset``
~~~~~~~~~~~~~~~~~~~~~~~

Resets all LMCache observability metrics to their initial state
(``reset_observability_metrics``). Intended for test harnesses and
benchmarks — not for production.

**Response** (``200 OK``, ``text/plain``):

.. code-block:: text

    ok

**HTTP status codes:**

- ``200``: metrics reset.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/metrics/reset

Diagnostics and Debugging
-------------------------

``GET /loglevel``
~~~~~~~~~~~~~~~~~

Inspect or mutate Python logger levels at runtime. All responses are
``text/plain``. The endpoint has three modes driven by query parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Query
     - Behavior
   * - (no params)
     - List every logger registered with :mod:`logging` and its level.
   * - ``?logger_name=<name>``
     - Return the effective level of the named logger.
   * - ``?logger_name=<name>&level=<LEVEL>``
     - Set the named logger (and its handlers) to ``LEVEL``
       (``DEBUG``/``INFO``/``WARNING``/``ERROR``/``CRITICAL``;
       case-insensitive). Returns ``400`` on an unknown level.

Passing ``level`` without ``logger_name`` matches none of the modes and
returns ``200`` with a ``null`` body.

**Response** (``200 OK``, ``text/plain``): one ``<logger>: <LEVEL>`` line per
registered logger (list mode), a single ``<logger>: <LEVEL>`` (read mode), or a
confirmation of the updated level (set mode).

**HTTP status codes:**

- ``200``: levels listed, read, or set (also the ``null``-body case above).
- ``400``: ``level`` is not a known logging level.

**Examples:**

.. code-block:: bash

    # list everything
    curl -s http://localhost:8080/loglevel

    # read one
    curl -s 'http://localhost:8080/loglevel?logger_name=lmcache'

    # elevate to DEBUG
    curl -s 'http://localhost:8080/loglevel?logger_name=lmcache&level=DEBUG'

``GET /threads``
~~~~~~~~~~~~~~~~

Enumerate active Python threads in the server process along with their
stack traces, plus a total-count summary (``Content-Type: text/plain``).
Useful for live debugging of hangs or runaway workers.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Query
     - Behavior
   * - ``?name=<substr>``
     - Keep only threads whose name contains ``<substr>``
       (case-insensitive).
   * - ``?thread_id=<int>``
     - Keep only the thread with the matching ``ident``.

.. warning::

   The response contains live stack traces and can disclose internal code
   paths and state. Restrict network access to this endpoint in production.

**Response** (``200 OK``, ``text/plain``): a total-thread-count summary followed
by each thread's name, ``ident``, and current stack trace.

**HTTP status codes:**

- ``200``: thread listing returned.

**Example:**

.. code-block:: bash

    curl -s 'http://localhost:8080/threads?name=periodic'

``GET /periodic-threads``
~~~~~~~~~~~~~~~~~~~~~~~~~

Returns a JSON snapshot of the
:class:`~lmcache.v1.periodic_thread.PeriodicThreadRegistry`: counts by
level plus per-thread status (last run timestamp, latest summary, etc.).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Query
     - Behavior
   * - ``?level=critical|high|medium|low``
     - Only include threads at the given level. ``400`` on unknown.
   * - ``?running_only=true``
     - Only include threads currently running.
   * - ``?active_only=true``
     - Only include threads considered active (recent tick).

**Response** (``200 OK``):

.. code-block:: json

    {
      "summary": {
        "total_count": 4,
        "running_count": 4,
        "active_count": 4,
        "by_level": {
          "critical": {"total": 1, "running": 1, "active": 1},
          "high":     {"total": 2, "running": 2, "active": 2},
          "medium":   {"total": 1, "running": 1, "active": 1},
          "low":      {"total": 0, "running": 0, "active": 0}
        }
      },
      "threads": [
        {
          "name": "...",
          "level": "high",
          "interval": 5.0,
          "is_running": true,
          "is_active": true,
          "last_run_ago": 1.2,
          "total_runs": 120,
          "failed_runs": 0,
          "success_rate": 100.0,
          "last_summary": {"...": "..."}
        }
      ]
    }

**HTTP status codes:**

- ``200``: snapshot returned.
- ``400``: unknown ``level`` filter.

**Example:**

.. code-block:: bash

    curl -s 'http://localhost:8080/periodic-threads?level=critical' | jq

``GET /periodic-threads/{thread_name}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed status for a single periodic thread (the same per-thread object
shown in the ``threads`` list above).

**Path parameters:** ``thread_name`` — the registered periodic thread name.

**Response** (``200 OK``):

.. code-block:: json

    {
      "name": "storage-flush",
      "level": "critical",
      "interval": 5.0,
      "is_running": true,
      "is_active": true,
      "last_run_ago": 1.2,
      "total_runs": 120,
      "failed_runs": 0,
      "success_rate": 100.0,
      "last_summary": {"...": "..."}
    }

**Response** (``404 Not Found``) when the name is unknown:

.. code-block:: json

    {"error": "Thread not found: <name>"}

**HTTP status codes:**

- ``200``: thread status returned.
- ``404``: no periodic thread with that name.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/periodic-threads/storage-flush | jq

``GET /periodic-threads-health``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast health check covering only ``critical`` and ``high`` level periodic
threads. A thread is flagged unhealthy when it is marked running but has
not ticked within its expected interval. Always returns ``200`` — health
is conveyed by the ``healthy`` boolean, not the HTTP status.

**Response** (``200 OK``):

.. code-block:: json

    {
      "healthy": true,
      "unhealthy_count": 0,
      "unhealthy_threads": []
    }

When something is lagging:

.. code-block:: json

    {
      "healthy": false,
      "unhealthy_count": 1,
      "unhealthy_threads": [
        {
          "name": "storage-flush",
          "level": "critical",
          "last_run_ago": 42.5,
          "interval": 5.0
        }
      ]
    }

**HTTP status codes:**

- ``200``: health reported — always ``200``, even when unhealthy
  (the ``healthy`` boolean is ``false``).

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/periodic-threads-health

``GET /env``
~~~~~~~~~~~~

Dumps the process environment variables as a sorted, pretty-printed
JSON document. The response ``Content-Type`` is ``text/plain`` so it can be
piped directly to a terminal.

.. warning::

   The payload contains **every** environment variable, including any
   secrets injected via the environment. There is no redaction or auth —
   restrict network access to this endpoint in production.

**Response** (``200 OK``, ``text/plain``):

.. code-block:: json

    {
      "HOME": "/root",
      "LMCACHE_LOG_LEVEL": "INFO",
      "PATH": "/usr/local/bin:/usr/bin"
    }

**HTTP status codes:**

- ``200``: environment dump returned.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/env

``POST /run_script``
~~~~~~~~~~~~~~~~~~~~

Execute an uploaded Python script inside the server process. The script is
uploaded as multipart form data under the field name ``script`` and is
``exec``'d with a restricted ``__builtins__`` (only ``print``, ``str``,
``int``, ``float``, ``list``, ``dict``, ``tuple``, ``set``, and a guarded
``__import__``). Only modules listed in ``--script-allowed-imports`` can be
imported; the running FastAPI ``app`` is injected into the script globals.
If the script assigns a variable named ``result``, its stringified value is
returned; otherwise the body is ``"Script executed successfully"``
(``Content-Type: text/plain``).

**Request body:** multipart form data with a single ``script`` file field
(the Python source to execute). Not a JSON body.

.. danger::

   This endpoint runs caller-supplied code in-process. The restricted
   builtins are **not** a security sandbox — combined with the injected
   ``app`` object and any allowed imports, treat it as full remote code
   execution. Never expose it on an untrusted network.

**Response** (``200 OK``, ``text/plain``): the stringified ``result`` variable if
the script assigns one, otherwise ``Script executed successfully``.

**HTTP status codes:**

- ``200``: script executed.
- ``400``: no ``script`` file provided.
- ``500``: an exception was raised during import setup or execution
  (body: ``"Error executing script: <reason>"``).

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/run_script \
        -F 'script=@my_script.py'

Adding New Endpoints
--------------------

Endpoints are auto-discovered from
``lmcache/v1/multiprocess/http_apis/``. To add a new MP-only endpoint:

1. Create a new module in that directory named ``<name>_api.py``.
2. Define a module-level ``router = APIRouter()``.
3. Register handlers on ``router`` using FastAPI decorators.
4. Access the engine via ``request.app.state.engine`` and guard for the
   ``None`` case (engine not yet initialized).

The :class:`~lmcache.v1.multiprocess.http_api_registry.HTTPAPIRegistry`
will pick the module up automatically at startup — no central
registration list to edit.

If the route is generic enough to be shared with the vLLM-embedded API
server, add it under ``lmcache/v1/internal_api_server/common/`` instead.
It will be picked up on the MP side via ``common_api.py`` unless its
module name is listed in ``_MP_INCOMPATIBLE_MODULES`` there (reserved
for modules that require vLLM-specific ``app.state`` attributes; the
list is currently empty). A handler that lives under
``internal_api_server/vllm/`` can still be surfaced on the MP server by
including its router from a group module under ``http_apis/`` (as
``info_api.py`` does for the version endpoints).

When adding a new endpoint, please also add a matching section to this
page documenting the endpoint's purpose, request/response schema, and
an example ``curl`` invocation.
