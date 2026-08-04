Overview
========

LMCache multiprocess (MP) mode runs LMCache as a **standalone service** that
vLLM instances connect to over ZMQ.  One LMCache server per node can serve
multiple vLLM pods, providing process isolation, shared caching, and
independent resource scaling.

.. contents::
   :local:
   :depth: 2

Key Benefits
------------

- **Process isolation** -- LMCache and vLLM run in separate processes (or
  containers), so a cache-related issue does not crash the inference engine.
- **No GIL contention or Python overhead on the inference path** -- By running
  LMCache in a separate process, its Python GIL and CPU work (hashing,
  memory management, L2 I/O) do not compete with vLLM's inference threads.
- **Shared caching across pods** -- Multiple vLLM instances on the same node
  share a single L1 cache, maximizing KV reuse.
- **Independent resource scaling** -- Allocate CPU memory for caching
  independently of GPU memory for inference.
- **Multi-tier storage (L1 + L2)** -- An L1 cache (in CPU DRAM, or an NVMe
  slab via GPUDirect Storage) backed by persistent L2 storage via NIXL (GDS,
  POSIX, HF3FS, and more).
- **Built-in observability** -- Prometheus metrics and a telemetry event system
  out of the box.

Prerequisites
-------------

- **vLLM** latest version is recommended for best compatibility
- **LMCache** latest dev branch

Server Variants
---------------

LMCache ships two server entry points:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry Point
     - Description
   * - ``lmcache server``
     - **Recommended.** ZMQ + FastAPI HTTP frontend — see :doc:`http_api`.
   * - ``python3 -m lmcache.v1.multiprocess.server``
     - (Legacy) ZMQ-only server with no HTTP endpoints; same
       ``--engine-type`` / ``--supported-transfer-mode`` flags as
       ``lmcache server``. Prefer ``lmcache server``.

The sections below describe LMCache MP internals -- useful if you want to
understand, debug, or extend the system.

High-Level Architecture
-----------------------

.. code-block:: text

    vLLM Instance(s)
         |
         | ZMQ (tcp)
         v
    MessageQueueServer (mq.py)
         |
         | dispatch by RequestType
         v
    MPCacheServer (server.py)
         |
         |--- TokenHasher / SessionManager
         |
         v
    StorageManager (distributed/storage_manager.py)
         |
         |--- L1Manager (l1_manager.py)
         |       |--- L1MemoryManager (CPU DRAM),
         |       |    DevDaxL1MemoryManager (Device-DAX slab), or
         |       |    GDSL1MemoryManager (NVMe slab via cuFile / hipFile)
         |       |--- TTLLock per object (read/write)
         |
         |--- StoreController  -----> L2 Adapter(s) (async L1->L2 push)
         |--- PrefetchController ---> L2 Adapter(s) (async L2->L1 load)
         |--- EvictionController ----> L1Manager (watermark-triggered eviction)
         |
         v
    EventBus + OTel providers (observability)

Engine and Modules
------------------

All server entry points share the same ``MPCacheServer`` and
``StorageManager`` core. ``MPCacheServer`` is now a thin compositor:
it holds an ``MPCacheServerContext`` and a list of ``EngineModule``
instances assembled by ``_build_modules()`` (in ``server.py``)
based on ``--engine-type`` and ``--supported-transfer-mode``.

**``server.py``** -- The default ZMQ-only server.  Creates an
``MPCacheServer``, assembles the engine modules
(``LookupModule`` + ``ManagementModule`` + ``LMCacheDrivenTransferModule``
and/or ``EngineDrivenTransferModule`` depending on
``--supported-transfer-mode`` — ``lmcache_driven`` or ``engine_driven`` loads
just one,
``auto`` (default) loads both — plus a CacheBlend module when
``--engine-type`` is set: ``blend`` appends ``BlendV3Module`` (the
current paged-aware implementation), and ``blend_legacy`` appends
``BlendModule`` (the original)). Starts a ``MessageQueueServer``,
registers handlers for every ``RequestType`` exposed by the loaded
modules, and blocks in a keep-alive loop.

**``modules/blend.py``** -- Defines ``BlendModule`` and ``BlendEngineV2``,
which add the original CacheBlend operations (``CB_REGISTER_KV_CACHE``,
``CB_LOOKUP_PRE_COMPUTED``, ``CB_STORE_PRE_COMPUTED``,
``CB_RETRIEVE_PRE_COMPUTED``, ``CB_STORE_FINAL`` and their V2
variants). Enables non-prefix KV cache reuse across document
paragraphs. Selected by passing ``--engine-type blend_legacy`` to
``lmcache server``.

**``modules/blend_v3.py``** -- Defines ``BlendV3Module``, the
paged-aware CacheBlend V3 pipeline that runs on the sparse-prefetch
path. Adds the V3 RPCs (``CB_REGISTER_ROPE_V3``,
``CB_UNREGISTER_ROPE_V3``, ``CB_RETRIEVE_PRE_COMPUTED_V3``,
``CB_UNIFIED_LOOKUP``) and reuses the existing
``LMCacheDrivenTransferModule`` and ``LookupModule``. Selected by
passing ``--engine-type blend`` to
``lmcache server``.

Both blend variants require ``--supported-transfer-mode`` to be
``lmcache_driven`` or ``auto`` and will refuse to load when it is
``engine_driven``.

**``http_server.py``** -- Wraps ``run_cache_server()`` (from ``server.py``)
inside a FastAPI application.  Endpoints are contributed by modules under
``http_apis/`` and auto-registered via ``HTTPAPIRegistry``: ``GET /`` (basic
liveness), ``GET /healthcheck`` for Kubernetes probes, ``POST /cache/clear``
for clearing all KV cache data in L1 (CPU) memory, and ``GET /status``
for inspecting detailed internal state.  The ZMQ server runs as part of the
same process, and any configured runtime plugins are spawned by
``MPRuntimePluginLauncher`` during FastAPI startup.

ZMQ Protocol
------------

Communication between vLLM and LMCache uses ZMQ (DEALER/ROUTER pattern).

**RequestType enum** (defined in ``protocols/base.py``):

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Request Type
     - Handler Type
     - Description
   * - ``REGISTER_KV_CACHE``
     - SYNC
     - Register GPU KV cache tensors for a vLLM instance.
   * - ``UNREGISTER_KV_CACHE``
     - SYNC
     - Unregister KV cache tensors.
   * - ``REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT``
     - SYNC
     - Register an engine-driven KV cache context (CPU/accelerator
       workers using the PREPARE/COMMIT transfer path). Loaded only when
       ``--supported-transfer-mode`` is ``engine_driven`` or ``auto``.
       Returns a ``RegisterEngineDrivenContextResponse`` carrying the
       SHM segment name and pool size when the SHM path is in use
       (empty for the pickle path).
   * - ``UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT``
     - SYNC
     - Unregister an engine-driven KV cache context.
   * - ``STORE``
     - BLOCKING
     - Store KV cache chunks from GPU to L1 (CPU). LMCache-driven
       transfer path (CUDA IPC); loaded only when
       ``--supported-transfer-mode`` is ``lmcache_driven`` or ``auto``.
   * - ``RETRIEVE``
     - BLOCKING
     - Copy KV cache chunks from L1 (CPU) back to GPU. LMCache-driven
       transfer path (CUDA IPC); loaded only when
       ``--supported-transfer-mode`` is ``lmcache_driven`` or ``auto``.
   * - ``PREPARE_STORE``
     - BLOCKING
     - (Engine-driven path) Worker asks the server to prepare store-side
       transfer state for a key. Loaded when ``--supported-transfer-mode``
       is ``engine_driven`` or ``auto``.
   * - ``COMMIT_STORE``
     - BLOCKING
     - (Engine-driven path) Worker commits the chunk's serialized bytes
       (pickle path) or releases the prepared SHM slot (SHM path) so the
       server can persist into L1 storage.
   * - ``PREPARE_RETRIEVE``
     - BLOCKING
     - (Engine-driven path) Worker asks the server to prepare the
       retrieval payload for a key. The pickle path returns the bytes
       inline; the SHM path returns slot info so the worker can read
       from shared memory.
   * - ``COMMIT_RETRIEVE``
     - BLOCKING
     - (Engine-driven path) Worker acknowledges retrieval completion so
       the server can release the underlying read locks and reclaim any
       transport state.
   * - ``LOOKUP``
     - BLOCKING
     - Submit a prefix lookup; the prefetch job is tracked server-side by
       request_id.
   * - ``QUERY_PREFETCH_STATUS``
     - BLOCKING
     - Poll a prefetch job by request_id. Returns the loaded chunk count
       when done, or ``None`` while the prefetch is still in progress.
   * - ``WAIT_PREFETCH_STATUS``
     - BLOCKING
     - (SGLang only) Block until a prefetch job completes, then return its
       loaded chunk count, or ``None`` on timeout. The blocking alternative
       to polling ``QUERY_PREFETCH_STATUS``.
   * - ``QUERY_PREFETCH_LOOKUP_HITS``
     - BLOCKING
     - Query the lookup-phase hit chunk count by request_id, before the
       prefetch finishes. Returns ``None`` while the lookup is still
       running.
   * - ``FREE_LOOKUP_LOCKS``
     - BLOCKING
     - Release read locks from a cancelled lookup without doing a full
       RETRIEVE.
   * - ``END_SESSION``
     - BLOCKING
     - Remove session state for a finished request.
   * - ``CLEAR``
     - BLOCKING
     - Clear all cached data.
   * - ``GET_CHUNK_SIZE``
     - SYNC
     - Return the server's chunk size.
   * - ``PING``
     - BLOCKING
     - Liveness ping; the handler always returns ``True``.
   * - ``REPORT_BLOCK_ALLOCATION``
     - BLOCKING
     - Fire-and-forget channel for the vLLM scheduler to report GPU block
       allocation events to the observability subsystem.
   * - ``NOOP``
     - SYNC
     - Debug heartbeat -- returns a confirmation string.
   * - ``CB_REGISTER_KV_CACHE``
     - SYNC
     - (Blend) Register CacheBlend KV buffer.
   * - ``CB_UNREGISTER_KV_CACHE``
     - SYNC
     - (Blend) Unregister CacheBlend KV buffer.
   * - ``CB_STORE_PRE_COMPUTED``
     - BLOCKING
     - (Blend) Store pre-computed paragraph chunks.
   * - ``CB_LOOKUP_PRE_COMPUTED``
     - BLOCKING
     - (Blend) Lookup pre-computed paragraph chunks.
   * - ``CB_RETRIEVE_PRE_COMPUTED``
     - BLOCKING
     - (Blend) Retrieve pre-computed paragraph chunks to GPU.
   * - ``CB_STORE_FINAL``
     - BLOCKING
     - (Blend) Store final blended chunks.
   * - ``CB_LOOKUP_PRE_COMPUTED_V2``
     - BLOCKING
     - (Blend V2) Lookup pre-computed chunks; returns
       ``CBMatchResult`` entries (with old/cur ranges and per-chunk hashes)
       so the retrieve step can skip re-hashing.
   * - ``CB_RETRIEVE_PRE_COMPUTED_V2``
     - BLOCKING
     - (Blend V2) Retrieve pre-computed chunks using the
       ``CBMatchResult`` list returned by ``CB_LOOKUP_PRE_COMPUTED_V2``.
   * - ``CB_REGISTER_ROPE_V3``
     - SYNC
     - (Blend V3) Share the RoPE cos/sin cache onto a context already
       registered via ``REGISTER_KV_CACHE``.
   * - ``CB_UNREGISTER_ROPE_V3``
     - SYNC
     - (Blend V3) Drop the RoPE state (paged KV cache lives on; use
       ``UNREGISTER_KV_CACHE`` to release that).
   * - ``CB_RETRIEVE_PRE_COMPUTED_V3``
     - BLOCKING
     - (Blend V3) Scatter all matched chunks (prefix- and non-prefix-hit)
       into paged KV by per-token block ID; re-RoPE only the shifted subset.
   * - ``CB_UNIFIED_LOOKUP``
     - BLOCKING
     - (Blend V3) Sole live lookup path: one RPC runs prefix + non-prefix
       match, reconciles, issues one sparse-coalesced prefetch, and
       classifies per-TP-rank. Returns ``CBUnifiedLookupResult`` (or
       ``None`` while the prefetch is still in flight).
   * - ``P2P_LOOKUP_AND_LOCK``
     - BLOCKING
     - (P2P) Look up the given keys and read-lock the locally cached
       prefix. Returns a task id which the caller passes to
       ``P2P_QUERY_LOOKUP_RESULTS`` to poll for the transfer addresses.
       Served by ``P2PController`` (loaded unconditionally by
       ``_build_modules()``); whether this server also acts as a P2P
       client is controlled by ``--p2p-advertise-url`` -- see
       :doc:`p2p`.
   * - ``P2P_QUERY_LOOKUP_RESULTS``
     - BLOCKING
     - (P2P) Poll the transfer addresses for a lookup task. Returns a
       list of ``TransferChannelAddress`` once the lookup is complete,
       or ``None`` while the lookup is still in progress or its
       results have already been consumed.
   * - ``P2P_UNLOCK_OBJECTS``
     - BLOCKING
     - (P2P) Release the read locks previously taken by
       ``P2P_LOOKUP_AND_LOCK`` on the given keys.

**Handler types:**

- **SYNC** -- Runs directly in the ZMQ main loop (fast, non-blocking).
- **BLOCKING** -- Dispatched to a thread pool (may involve GPU copies or I/O).

Config System
-------------

Each config module exposes a composable triple:

.. code-block:: text

    (DataclassConfig, add_*_args(parser), parse_args_to_*_config(args))

``server.py:parse_args()`` composes them:

.. code-block:: python

    parser = argparse.ArgumentParser(...)
    add_mp_server_args(parser)        # from multiprocess/config.py
                                      # includes runtime-plugin args
                                      # (--runtime-plugin-locations,
                                      #  --runtime-plugin-config)
    add_storage_manager_args(parser)  # from distributed/config.py
      # which internally calls add_l2_adapters_args(parser)
    add_observability_args(parser)    # from mp_observability/config.py

``http_server.py`` reuses this pattern, adding
``add_http_frontend_args()`` and ``add_coordinator_args()`` for the
``lmcache server`` CLI. CacheBlend is no longer a separate entry point —
it is opted into at runtime by passing ``--engine-type`` to
``server.py`` (or ``lmcache server``). ``--engine-type blend`` appends
``BlendV3Module`` (the current paged-aware implementation), while
``--engine-type blend_legacy`` appends ``BlendModule`` (the original).

Distributed Storage
-------------------

StorageManager
~~~~~~~~~~~~~~

``lmcache/v1/distributed/storage_manager.py``

The top-level manager that wires together L1, L2, and all controllers.  Key
methods:

- ``reserve_write()`` / ``finish_write()`` -- Two-phase write into L1.
- ``submit_prefetch_task()`` / ``query_prefetch_status()`` -- Async lookup +
  L2 prefetch. ``query_prefetch_status()`` is non-blocking and returns
  ``None`` while the L2 prefetch is still in flight.
- ``wait_prefetch_status()`` -- Blocking alternative to polling
  ``query_prefetch_status()``: waits on a controller condition variable
  until the prefetch result is published (or a timeout expires).
  L1-only prefetches return immediately. Used by the
  ``WAIT_PREFETCH_STATUS`` RPC to avoid busy-polling on the load path.
- ``read_prefetched_results()`` / ``finish_read_prefetched()`` -- Read
  prefetched data from L1 with automatic lock management.

L1Manager
~~~~~~~~~

``lmcache/v1/distributed/l1_manager.py``

Manages objects in CPU memory with a state machine:

.. code-block:: text

    None --> write_locked --> ready --> read_locked
              (reserve_write)  (finish_write)  (reserve_read)
                                  |                |
                                  v                v
                               evictable      finish_read -> ready

Each object has two ``TTLLock`` instances (read and write) with configurable
timeouts to prevent deadlocks from crashed clients.

The underlying memory allocation is handled by one of three interchangeable
tiers selected at startup (all satisfy ``L1ManagerProtocol``):

- ``L1MemoryManager`` (default) -- pinned CPU DRAM, with lazy growth up to
  ``--l1-size-gb``.
- ``DevDaxL1MemoryManager`` -- a Device-DAX-backed L1 slab when
  ``--l1-devdax-path`` is set. A pure Device-DAX configuration maps the DAX
  device as the full L1 arena; a hybrid configuration uses DRAM first and
  spills overflow allocations into Device-DAX. See the *L1 Memory Manager*
  section of :doc:`configuration` for the accepted knobs.
- ``GDSL1MemoryManager`` -- an NVMe slab file when ``--gds-l1-path`` is set.
  The bytes live on disk; reads/writes DMA directly between the GPU staging
  buffer and the slab, driven by the process-global ``GDSContext``
  (``gpu_connector/gds_context.py``) and dispatched from ``gpu_ops``. The DMA
  backend is selected by platform via ``gpu_connector/_gds_async.py`` --
  cuFile (``libcufile.so``) on NVIDIA and hipFile (``libhipfile.so``) on AMD
  ROCm; see the *GDS L1 Tier* section of :doc:`configuration` for the
  vendor-specific requirements. The CPU tier is disabled in this mode.

L2 Adapters
~~~~~~~~~~~

``lmcache/v1/distributed/l2_adapters/``

The ``L2AdapterInterface`` (in ``base.py``) defines three async task methods:

- ``submit_store_task(key, data)`` -- Push data to L2.
- ``submit_lookup_and_lock_task(keys)`` -- Check if keys exist in L2.
- ``submit_load_task(keys, layout_desc)`` -- Load data from L2 into L1.

The factory function ``create_l2_adapter()`` (in ``__init__.py``) uses
``isinstance()`` on the config type to instantiate the correct adapter.

New adapter types are registered via ``register_l2_adapter_type()`` in
``config.py``.

Controllers
~~~~~~~~~~~

**StoreController** (``storage_controllers/store_controller.py``):
Event-driven background thread that uses ``select.poll()`` on listener eventfd
and adapter store eventfds.  When new objects appear in L1 (signaled via
``StoreListener``), it submits async store tasks to each L2 adapter based on
the ``StorePolicy``.

**EvictionController** (``storage_controllers/eviction_controller.py``):
Periodically checks L1 memory usage against the watermark threshold.  When
triggered, evicts objects using the configured policy (``LRU``,
``IsolatedLRU``, or ``noop``) until usage drops below the target.
``IsolatedLRU`` evicts per ``cache_salt`` against limits registered through
the ``/quota`` HTTP endpoints; see :ref:`mp-http-quota-api`.

**PrefetchController** (``storage_controllers/prefetch_controller.py``):
Handles L2 lookup and load requests submitted by ``StorageManager`` during
``LOOKUP`` RPCs.  When keys are not in L1, it queries L2 adapters and loads
found data back into L1.

Request Flows
-------------

LOOKUP Flow
~~~~~~~~~~~

.. code-block:: text

    vLLM                MPCacheServer          StorageManager         L1Manager       L2 (PrefetchController)
     |                       |                       |                    |                    |
     |---LOOKUP(key)-------->|                       |                    |                    |
     |                       |--submit_prefetch------>|                    |                    |
     |                       |                       |--reserve_read----->|                    |
     |                       |                       |<--hit_count--------|                    |
     |                       |                       |--submit_prefetch_request--------------->|
     |                       |                       |    (remaining keys)                     |
     |                       |--query_prefetch------->|                    |                    |
     |                       |                       |--query_prefetch_result----------------->|
     |                       |<--found_count----------|                    |                    |
     |<--found_count---------|                       |                    |                    |

STORE Flow
~~~~~~~~~~

.. code-block:: text

    vLLM                MPCacheServer          StorageManager         L1Manager
     |                       |                       |                    |
     |---STORE(key,blocks)-->|                       |                    |
     |                       |--reserve_write-------->|                    |
     |                       |                       |--reserve_write---->|
     |                       |                       |<--memory_objs------|
     |                       |  (GPU->CPU copy)      |                    |
     |                       |--finish_write--------->|                    |
     |                       |                       |--finish_write----->|
     |                       |                       |                    |
     |                       |                       |  [StoreController detects new objects]
     |                       |                       |  [async L1->L2 push via adapters]
     |<--event_handle--------|                       |                    |

RETRIEVE Flow
~~~~~~~~~~~~~

.. code-block:: text

    vLLM                MPCacheServer          StorageManager         L1Manager
     |                       |                       |                    |
     |---RETRIEVE(key)------>|                       |                    |
     |                       |--read_prefetched------>|                    |
     |                       |                       |--unsafe_read------>|
     |                       |                       |<--memory_objs------|
     |                       |  (CPU->GPU copy)      |                    |
     |                       |--finish_read_prefetch->|                    |
     |                       |                       |--finish_read------>|
     |<--event_handle--------|                       |                    |

Observability Internals
-----------------------

**EventBus** (``lmcache/v1/mp_observability/event_bus.py``) is a global
singleton initialized at server startup by ``init_observability()``.
Producers (L1Manager, StorageManager, MPCacheServer) publish ``Event``
objects to a bounded queue (``--event-bus-queue-size``, default 10000,
tail-drop on overflow).  A background drain thread dispatches each
event to all registered subscribers.

**Subscribers** live under ``lmcache/v1/mp_observability/subscribers/``
and are grouped by concern: ``metrics/`` (OTel counters and lifecycle
histograms), ``logging/`` (Python logging handlers, lookup-hash JSONL),
and ``tracing/`` (OTel spans built from START/END event pairs).
``init_observability()`` registers the set selected by CLI flags
(``--disable-metrics``, ``--disable-logging``, ``--enable-tracing``).

**OTel providers** are set up via ``otel_init.py`` before subscribers
are constructed, so module-level ``get_meter()`` / ``get_tracer()``
calls bind to the real provider. Metrics are exported both to an
in-process Prometheus ``/metrics`` endpoint (``--prometheus-port``,
default 9090) and, when ``--otlp-endpoint`` is set, pushed to an OTel
collector.

How to Extend
-------------

Adding a new L2 adapter
~~~~~~~~~~~~~~~~~~~~~~~~

Create a new ``*_l2_adapter.py`` module under
``lmcache/v1/distributed/l2_adapters/`` — ``__init__.py`` auto-discovers
modules matching that suffix via ``pkgutil`` and imports them lazily on
first use, so no other files need to be modified.

1. Create a config class subclassing ``L2AdapterConfigBase`` with
   ``from_dict()`` and ``help()`` methods.
2. Create an adapter class implementing ``L2AdapterInterface``, and
   a small factory function
   ``(config, l1_memory_desc) -> L2AdapterInterface``.
3. At module level, self-register both the config and the factory:

   .. code-block:: python

       register_l2_adapter_type("my_adapter", MyAdapterConfig)
       register_l2_adapter_factory("my_adapter", _create_my_adapter)

See ``mock_l2_adapter.py`` or ``s3_l2_adapter.py`` for reference
implementations.

Adding an observability subscriber
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a subscriber class subclassing ``EventSubscriber`` (defined
   in ``lmcache/v1/mp_observability/event_bus.py``): implement
   ``get_subscriptions()`` to return an ``{EventType: callback}``
   mapping; optionally override ``shutdown()`` for cleanup.
2. Place the class under the appropriate concern group
   (``subscribers/metrics/``, ``subscribers/logging/``, or
   ``subscribers/tracing/``) and export it from that package's
   ``__init__.py``.
3. Register the subscriber in ``init_observability()``
   (``lmcache/v1/mp_observability/config.py``) via
   ``bus.register_subscriber(...)`` inside the branch matching its
   concern (metrics / logging / tracing), gated on the corresponding
   CLI flag if needed.

Adding a new request type
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add a new member to ``RequestType`` in ``protocols/base.py``.
2. Create a ``ProtocolDefinition`` in the appropriate ``protocols/*.py`` file
   (``engine``, ``controller``, ``observability``, ``debug``, ``blend``,
   ``blend_v2``, ``blend_v3``, or ``p2p``) and add the request name to that
   module's ``REQUEST_NAMES``.
3. Implement the handler method on the appropriate ``EngineModule``
   (e.g. ``LookupModule``, ``LMCacheDrivenTransferModule``, ``BlendV3Module``) and
   expose it as a ``HandlerSpec`` from that module's ``get_handlers()``.
4. ``run_cache_server()`` registers every ``HandlerSpec`` returned by the
   loaded modules via ``add_handler_helper()`` — no manual registration
   step is needed.

Key Source Files
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``lmcache/v1/multiprocess/server.py``
     - MPCacheServer + ZMQ server entry point
   * - ``lmcache/v1/multiprocess/config.py``
     - MPServerConfig, HTTPFrontendConfig
   * - ``lmcache/v1/multiprocess/engine_context.py``
     - MPCacheServerContext (shared state passed to every EngineModule)
   * - ``lmcache/v1/multiprocess/engine_module.py``
     - ``EngineModule`` protocol, ``HandlerSpec``, ``ThreadPoolType``
       (per-module handler registration)
   * - ``lmcache/v1/multiprocess/modules/``
     - Engine module implementations: ``lookup.py`` (``LookupModule``),
       ``management.py`` (``ManagementModule``), ``lmcache_driven_transfer.py``
       (``LMCacheDrivenTransferModule``), ``engine_driven_transfer.py``
       (``EngineDrivenTransferModule``), ``blend.py``
       (``BlendModule`` / ``BlendEngineV2``, selected by
       ``--engine-type blend_legacy``), and ``blend_v3.py``
       (``BlendV3Module``, the paged-aware CacheBlend V3 pipeline
       selected by ``--engine-type blend``).
   * - ``lmcache/v1/multiprocess/http_server.py``
     - FastAPI wrapper with health check and many other useful APIs
   * - ``lmcache/v1/multiprocess/http_api_registry.py``
     - ``HTTPAPIRegistry`` that auto-discovers routers in ``http_apis/``
   * - ``lmcache/v1/multiprocess/http_apis/``
     - Extensible HTTP endpoints (``/``, ``/healthcheck``,
       ``/cache/clear``, ``/status``)
   * - ``lmcache/v1/multiprocess/mp_runtime_plugin_launcher.py``
     - ``MPRuntimePluginLauncher`` that spawns runtime plugins with the
       full server config serialized into environment variables
   * - ``lmcache/v1/multiprocess/protocols/base.py``
     - RequestType, HandlerType, ProtocolDefinition
   * - ``lmcache/v1/distributed/storage_manager.py``
     - StorageManager (top-level manager)
   * - ``lmcache/v1/distributed/config.py``
     - StorageManagerConfig hierarchy
   * - ``lmcache/v1/distributed/l1_manager.py``
     - L1Manager (object state machine)
   * - ``lmcache/v1/distributed/l2_adapters/config.py``
     - L2 adapter config registry
   * - ``lmcache/v1/distributed/l2_adapters/base.py``
     - L2AdapterInterface
   * - ``lmcache/v1/distributed/storage_controllers/store_controller.py``
     - StoreController (event-driven L1->L2)
   * - ``lmcache/v1/distributed/storage_controllers/eviction_controller.py``
     - EvictionController (watermark-triggered)
   * - ``lmcache/v1/distributed/storage_controllers/prefetch_controller.py``
     - PrefetchController (L2->L1 on miss)
   * - ``lmcache/v1/mp_observability/config.py``
     - ObservabilityConfig + ``init_observability()`` entry point
   * - ``lmcache/v1/mp_observability/event_bus.py``
     - EventBus singleton and ``EventSubscriber`` base class
   * - ``lmcache/v1/mp_observability/event.py``
     - ``Event`` / ``EventType`` definitions
   * - ``lmcache/v1/mp_observability/otel_init.py``
     - OTel metrics / tracing provider setup
   * - ``lmcache/v1/mp_observability/subscribers/``
     - Metrics, logging, and tracing subscribers
   * - ``lmcache/v1/mp_observability/trace/``
     - Trace recording (``--trace-level storage``) capture stack
