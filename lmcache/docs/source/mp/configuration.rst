Configuration Reference
=======================

This page documents every CLI argument accepted by the LMCache multiprocess
server.  Arguments are grouped by the config module that defines them.

.. contents::
   :local:
   :depth: 2

MP Server
---------

Source: ``lmcache/v1/multiprocess/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--instance-id``
     - *(unset, default UUID v4)*
     - Stable identity of this MP server. Used as the coordinator
       membership key and projected onto the OTel
       ``service.instance.id`` resource attribute on every metric and
       span (so telemetry and coordinator membership share one id).
       When the flag is not passed, defaults to a random UUID v4
       minted at startup.
   * - ``--host``
     - ``localhost``
     - Host address to bind the ZMQ server.
   * - ``--port``
     - ``5555``
     - Port to bind the ZMQ server.
   * - ``--chunk-size``
     - ``256``
     - Chunk size for KV cache operations (in tokens).
   * - ``--max-workers``
     - ``1``
     - Base number of worker threads. Sets the default for both the GPU
       (affinity) pool and the CPU (normal) pool. Can be overridden
       per-pool with ``--max-gpu-workers`` and ``--max-cpu-workers``.
   * - ``--max-gpu-workers``
     - (inherits ``--max-workers``)
     - Worker threads for the GPU affinity pool (STORE/RETRIEVE).
       Requests from the same vLLM instance are always dispatched to the
       same thread, eliminating GPU transfer lock contention.
   * - ``--max-cpu-workers``
     - (inherits ``--max-workers``)
     - Worker threads for the normal CPU pool (LOOKUP, etc.).
   * - ``--hash-algorithm``
     - ``blake3``
     - Hash algorithm for token-based operations.
       Choices: ``builtin``, ``sha256_cbor``, ``blake3``.
   * - ``--engine-type``
     - ``default``
     - Cache engine backend type. ``default`` uses standard prefix
       caching; ``blend`` selects the current CacheBlend V3 implementation
       (composes a ``BlendV3Module`` into the engine);
       ``blend_legacy`` selects the original CacheBlend
       (composes a ``BlendModule``). Both blend variants require
       ``--supported-transfer-mode`` to be ``lmcache_driven`` or ``auto``.
       Choices: ``default``, ``blend``, ``blend_legacy``.
   * - ``--supported-transfer-mode``
     - ``auto``
     - Which worker → server transfer paths the server loads.
       ``lmcache_driven`` enables only the server-driven transfer
       path (STORE/RETRIEVE, supports both CUDA IPC and CPU SHM);
       ``engine_driven`` enables only the non-GPU (PREPARE/COMMIT)
       transfer path; ``auto`` (default) loads both
       so workers of either device type can connect without manual
       configuration.
       Choices: ``lmcache_driven``, ``engine_driven``, ``auto``.
   * - ``--runtime-plugin-locations``
     - ``[]``
     - Zero or more paths to runtime plugin scripts or directories to
       launch alongside the server. Plugins are spawned by
       ``MPRuntimePluginLauncher`` and receive the full server config
       via the ``LMCACHE_RUNTIME_PLUGIN_CONFIG`` environment variable.
   * - ``--runtime-plugin-config``
     - ``"{}"``
     - JSON string of extra key-value config forwarded to runtime
       plugins via ``LMCACHE_RUNTIME_PLUGIN_EXTRA_CONFIG``. Example:
       ``'{"plugin.frontend.heartbeat_url": "http://localhost:5000/heartbeat"}'``.
   * - ``--script-allowed-imports``
     - ``[]``
     - Space-separated list of Python module names that scripts posted
       to the HTTP ``/run_script`` endpoint are allowed to import.
       Example: ``--script-allowed-imports numpy pandas``.
   * - ``--shm-name``
     - *(not set)*
     - SHM segment name for non-GPU KV transfer (only used when the
       non-GPU path is loaded, i.e. ``--supported-transfer-mode`` is
       ``auto`` or ``engine_driven``).
       Not set (default): auto-allocate a shared-memory pool.
       ``""`` (empty string): disable SHM and force the pickle transfer
       path.  Any other value: use that exact name for the SHM pool
       segment.
   * - ``--worker-reap-timeout-seconds``
     - ``120.0``
     - Silence budget (seconds) after which a worker that has sent at
       least one heartbeat PING but then gone quiet has its KV cache
       registration reaped, freeing the leaked GPU context and CUDA IPC
       handles. ``0`` disables reaping. Keep this at least 3x the engine
       adapter's ``lmcache.mp.heartbeat_interval`` (default 10s) so a few
       missed pings never reap a live worker; the adapter warns at startup
       if its interval is raised without raising this.
   * - ``--worker-registration-grace-seconds``
     - ``3600.0``
     - Silence budget (seconds) for a worker that registered but has never
       sent a PING (still warming up, or died before its first request).
       Must be >= ``--worker-reap-timeout-seconds``. Generous by default so
       slow model warmup is never mistaken for a dead worker.
   * - ``--enable-segmented-prefix``
     - ``False``
     - CacheBlend (``--engine-type blend``) only: on a mid-prefix L2 retrieve
       failure, retain the gapped prefix so the post-gap chunks stay
       L1-resident and only the dropped gap is recomputed, instead of
       truncating the prefix at the gap. No effect for other engines. See
       :doc:`/mp/l2_storage/fault_inject` for a way to exercise it.
   * - ``--separate-object-groups`` / ``--no-separate-object-groups``
     - ``True``
     - Split a hybrid model's kernel groups into one object group per
       cross-chunk attention window (full attention, each sliding-window
       size, mamba/GDN) at KV-cache registration. On by default; pass
       ``--no-separate-object-groups`` to keep all layers in a single
       full-attention object group. Transparent to correctness; a non-hybrid
       model always resolves to one object group. See :doc:`/mp/hybrid_models`.

Lookup Hash Logging
-------------------

Source: ``lmcache/v1/mp_observability/subscribers/logging/lookup_hash.py``

When enabled, the server publishes chunk hashes computed during ``lookup()``
as ``MP_LOOKUP`` events on the EventBus.  The
``LookupHashLoggingSubscriber`` writes these to rotating JSONL files for
offline analysis.  Disabled by default.  These arguments are part of the
Observability group.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--lookup-hash-log-dir``
     - ``""`` (disabled)
     - Directory to write lookup hash JSONL files.
       An empty string disables logging.
   * - ``--lookup-hash-log-rotation-interval``
     - ``21600`` (6 h)
     - Time interval in seconds before rotating to a new log file.
   * - ``--lookup-hash-log-rotation-max-size``
     - ``104857600`` (100 MB)
     - Max file size in bytes before rotating even if the time
       interval has not elapsed.
   * - ``--lookup-hash-log-max-files``
     - ``100``
     - Max number of log files to keep.  Oldest files are deleted
       when this limit is exceeded.

HTTP Frontend
-------------

Source: ``lmcache/v1/multiprocess/config.py``

The HTTP frontend is included when running ``lmcache server``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--http-host``
     - ``0.0.0.0``
     - Host to bind the HTTP (FastAPI/uvicorn) server.
   * - ``--http-port``
     - ``8080``
     - Port to bind the HTTP server.

P2P
---

Source: ``lmcache/v1/multiprocess/config.py``

These flags configure peer-to-peer KV cache sharing between MP servers
(see :doc:`p2p`). They are registered by ``add_p2p_args()`` on the
``lmcache server`` parser. P2P is enabled when ``--p2p-advertise-url``
is set, which additionally requires a coordinator URL via
``--coordinator-url`` (or ``LMCACHE_COORDINATOR_URL``).

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--p2p-advertise-url``
     - ``""`` (P2P disabled)
     - Transfer-channel server ``host:port`` this instance advertises to
       peers. Setting it enables P2P (also requires ``--coordinator-url``).
   * - ``--p2p-listen-url``
     - ``""``
     - Transfer-channel server ``host:port`` to bind. Defaults to
       ``--p2p-advertise-url``.
   * - ``--p2p-lookup-timeout``
     - ``30.0``
     - Seconds before a peer lookup result counts as a miss.
   * - ``--p2p-load-timeout``
     - ``30.0``
     - Seconds before a peer load counts as a failure.
   * - ``--p2p-transfer-engine``
     - ``nixl``
     - Transfer-channel implementation to use.

L1 Memory Manager
------------------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--l1-size-gb``
     - *required*
     - Size of the L1 tier in GB. Sizes the pinned-DRAM L1 by default, or the
       GDS slab file when ``--gds-l1-path`` is set (see *GDS L1 Tier* below).
   * - ``--l1-use-lazy`` / ``--no-l1-use-lazy``
     - ``True``
     - Enable or disable lazy allocation for L1 memory.
       Pass ``--l1-use-lazy`` to enable (default) or
       ``--no-l1-use-lazy`` to explicitly disable.
   * - ``--l1-init-size-gb``
     - ``20``
     - Initial allocation size (GB) when using lazy allocation.
   * - ``--l1-align-bytes``
     - ``4096``
     - Alignment size in bytes (default 4 KB).
   * - ``--l1-devdax-path``
     - *(not set)*
     - Optional ``/dev/dax*`` device or mmap-able file to use as the L1
       backing arena.  When set, disable lazy allocation with
       ``--no-l1-use-lazy`` and disable SHM transfer advertising with
       ``--shm-name ""`` because the L1 bytes live in the DAX mapping.  If a
       DAX L2 adapter with the same ``device_path`` is registered, that
       adapter's ``max_dax_size_gb`` is used as the L1 Device-DAX overflow
       size.

GDS L1 Tier
-----------

Source: ``lmcache/v1/distributed/config.py``

Opt-in. Setting ``--gds-l1-path`` switches the L1 medium from pinned DRAM to
an NVMe slab file accessed via GPUDirect Storage DMA. The CPU pinned-DRAM tier
is then disabled, and ``--l1-size-gb`` sizes the slab. Disable byte-array L2
adapters when this is on (the GDS tier exposes no L1 memory buffer for them to
register).

The DMA path is selected automatically by platform: **cuFile**
(``libcufile.so``) on NVIDIA and **hipFile** (``libhipfile.so``,
`ROCm/hipFile <https://github.com/ROCm/hipFile>`_) on AMD ROCm. The same
flags apply to both; no configuration change is needed to switch vendors.

.. note::

   AMD hipFile requires ROCm >= 7.2.0. The zero-copy GPUDirect fast path
   additionally needs a kernel built with ``CONFIG_PCI_P2PDMA``,
   ``amdgpu-dkms >= 30.20.1``, and the slab on a local NVMe ext4/xfs
   filesystem; where those are unavailable hipFile transparently falls back to
   a host-bounce compatibility path (correct, but not zero-copy).

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--gds-l1-path``
     - Not set
     - NVMe directory for the GDS L1 slab. Setting this enables the GDS L1
       tier; one shared slab per process lives at
       ``<path>/lmcache_gds_slab.bin``.
   * - ``--gds-l1-use-direct-io`` / ``--no-gds-l1-use-direct-io``
     - ``True``
     - Open the slab with ``O_DIRECT`` (required for the GDS DMA fast path on
       ext4).

L1 Manager TTLs
----------------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--l1-write-ttl-seconds``
     - ``600``
     - Time-to-live for each object's write lock (seconds).
   * - ``--l1-read-ttl-seconds``
     - ``300``
     - Time-to-live for each object's read lock (seconds).

Eviction Policy
---------------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--eviction-policy``
     - *required*
     - Eviction policy.
       Choices: ``LRU``, ``IsolatedLRU``, ``noop``.
       Use ``noop`` for buffer-only mode where L1 acts as a pure
       write buffer (data is deleted from L1 after L2 store).
       ``IsolatedLRU`` maintains one LRU list per ``cache_salt``
       and requires per-``cache_salt`` quotas to be configured at
       runtime via the ``/quota`` HTTP endpoints
       (see :ref:`mp-http-quota-api`); a ``cache_salt`` with no
       registered quota has an effective limit of ``0`` bytes,
       so its data is evicted at the next eviction cycle
       (allowlist semantics).
   * - ``--eviction-trigger-watermark``
     - ``0.8``
     - Memory usage ratio (0.0--1.0) that triggers eviction.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of allocated memory to evict when triggered (0.0--1.0).

L2 Policies
-----------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--l2-store-policy``
     - ``default``
     - L2 store policy.  Determines which adapters receive each key
       and whether keys are deleted from L1 after L2 store.
       The ``default`` policy stores all keys to all adapters and keeps L1.
       The ``skip_l1`` policy stores all keys to all adapters and then
       deletes them from L1 (buffer-only mode).
       Choices: ``default``, ``skip_l1``.
   * - ``--l2-prefetch-policy``
     - ``default``
     - L2 prefetch policy.  Determines which adapter loads each key
       when multiple adapters have it.
       The ``default`` policy picks the first adapter (lowest index).
       Prefetched keys are temporary (deleted after the reader finishes).
       The ``retain`` policy uses the same load plan but keeps
       prefetched keys permanently in L1.
       Choices: ``default``, ``retain``.
   * - ``--l2-prefetch-max-in-flight``
     - ``8``
     - Maximum number of concurrent prefetch (L2 load) requests.
       Limits how many in-flight loads the PrefetchController may
       issue at once, preventing excessive L1 memory pressure.
   * - ``--periodic-notifier-interval-ms``
     - ``5``
     - Interval in milliseconds for the periodic event notifier
       heartbeat.  A native C++ background thread writes to all
       registered file descriptors at this interval, waking
       controller poll loops for L2 adapters that lack native
       async completion callbacks.

L2 Adapters
-----------

Source: ``lmcache/v1/distributed/l2_adapters/config.py``

L2 adapters are configured via repeatable ``--l2-adapter <JSON>`` arguments.
Each JSON object must include a ``"type"`` field that selects the adapter type.
The order of ``--l2-adapter`` arguments determines the adapter order (cascade).

Registered adapter types: ``nixl_store``, ``nixl_store_dynamic``, ``fs``,
``fs_native``, ``mock``, ``mooncake_store``, ``aerospike``, ``bigtable``,
``sagemaker-hyperpod``, ``s3``, ``hfbucket``, ``resp``, ``valkey``,
``plugin``, ``native_plugin``, ``raw_block``, ``dax``, ``fault_inject``.
(A ``p2p`` type is also registered, but it is wired in dynamically by the
:doc:`P2P subsystem </mp/p2p>` rather than configured via ``--l2-adapter``.)

Each adapter type's required and optional fields, plus per-backend examples, are
documented on its own page under :doc:`Secondary KV Storage <l2_storage/index>`
-- including the adapters not detailed inline here (``fs_native``,
``raw_block``, ``dax``, ``mooncake_store``, ``aerospike``, ``bigtable``,
``sagemaker-hyperpod``, ``hfbucket``, ``resp``, ``valkey``).

Multiple adapters (cascade)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``--l2-adapter`` multiple times.  Adapters are used in the order given:

.. code-block:: bash

    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/ssd/l2", "use_direct_io": "false"}, "pool_size": 64}' \
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"}, "pool_size": 128}'

Observability
-------------

Source: ``lmcache/v1/mp_observability/config.py``

See :doc:`observability/index` for full details on the three modes (metrics,
logging, tracing).

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--disable-observability``
     - off
     - Master switch: disable the EventBus entirely.
   * - ``--disable-metrics``
     - off
     - Skip metrics subscribers (no Prometheus endpoint).
   * - ``--disable-logging``
     - off
     - Skip logging subscribers.
   * - ``--enable-tracing``
     - off
     - Register tracing subscribers. Requires ``--otlp-endpoint``.
   * - ``--event-bus-queue-size``
     - ``10000``
     - Max events in the EventBus queue before tail-drop.
   * - ``--otlp-endpoint``
     - *(none)*
     - OTLP gRPC endpoint for exporting metrics and traces.
   * - ``--prometheus-port``
     - ``9090``
     - Port for the Prometheus ``/metrics`` endpoint.
   * - ``--metrics-sample-rate``
     - ``0.01``
     - Fraction of chunks/blocks in ``(0, 1.0]`` to track for lifecycle
       histograms. Counters always count every event regardless of this
       setting.
   * - ``--trace-level``
     - *(none)*
     - Enable trace recording at the given level. Currently only
       ``storage`` is supported (records ``StorageManager`` public-API
       calls for offline replay via ``lmcache trace``). See
       :doc:`tracing_and_debugging`.
   * - ``--trace-output``
     - *(none)*
     - Path to write the trace file. If omitted while ``--trace-level``
       is set, a timestamped file under ``$TMPDIR``
       (``lmcache-trace-<pid>-<UTC>.lct``) is minted and its path is
       logged at INFO.
   * - ``--enable-extra-logging``
     - off
     - Periodic INFO logs: per-GPU L0<->L1 transfer stats and L1 memory
       usage. See :doc:`observability/logs`.
   * - ``--extra-logging-interval``
     - ``10.0``
     - Seconds between extra-logging emissions.

vLLM Client Configuration
--------------------------

On the vLLM side, specify the LMCache server host and port via the
``kv_connector_extra_config`` parameter. The ``tcp://`` transport prefix
on ``lmcache.mp.host`` is optional -- a bare host is accepted and
normalized to ``tcp://`` by the connector:

.. code-block:: bash

    vllm serve Qwen/Qwen3-14B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "127.0.0.1", "lmcache.mp.port": 6000}}'

To target multiple LMCache servers from a single vLLM deployment, pass a
list (or comma-separated string) of server URLs via
``lmcache.mp.server_urls``. When set, ``server_urls`` takes precedence
over the single-server ``host`` / ``port`` keys; vLLM's world size must
be divisible by the number of servers, and each worker connects only to
its locally-assigned server (global ranks are sliced into contiguous
blocks, one block per server). Multi-server mode currently supports
tensor parallelism only -- pipeline parallelism (``pp_size > 1``) and
data parallelism (``dp_size > 1``) are rejected with a clear error.

.. code-block:: bash

    vllm serve Qwen/Qwen3-14B \
        --tensor-parallel-size 4 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.server_urls": "tcp://host1:6667,tcp://host2:6667"}}'

``LMCacheMPConnector`` reads the following keys from
``kv_connector_extra_config``:

Connector ``extra_config`` Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All connector-level options are passed through
``kv_connector_extra_config`` and use the ``lmcache.mp.`` prefix.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``lmcache.mp.server_urls``
     - *(unset)*
     - Multi-server deployment: list (or comma-separated string) of
       ``<transport>://<host>:<port>`` URLs, e.g.
       ``"tcp://host1:6667,tcp://host2:6667"``. The transport prefix
       may be omitted -- bare ``host:port`` entries such as
       ``"host1:6667,host2:6667"`` are normalized to ``tcp://`` by the
       connector. When set, takes precedence over ``lmcache.mp.host`` /
       ``lmcache.mp.port``; the vLLM world size must be divisible by the
       number of servers, and each worker connects to its
       locally-assigned server.
   * - ``lmcache.mp.host``
     - ``tcp://localhost``
     - Single-server deployment: host of the LMCache MP server. A ZMQ
       transport prefix (e.g. ``tcp://``) is optional -- a bare
       ``localhost`` / ``127.0.0.1`` is normalized to ``tcp://`` by the
       connector. Ignored when ``lmcache.mp.server_urls`` is set.
   * - ``lmcache.mp.port``
     - ``5555``
     - Single-server deployment: port of the LMCache MP server. Must
       match the server's ``--port``. Ignored when
       ``lmcache.mp.server_urls`` is set.
   * - ``lmcache.mp.mq_timeout``
     - ``300.0``
     - Timeout (seconds) for blocking message-queue requests, including
       the initial chunk-size query and KV cache
       registration/unregistration. If the server does not respond within
       this window, the connector raises ``ConnectionError`` on startup.
   * - ``lmcache.mp.heartbeat_interval``
     - ``10.0``
     - Interval (seconds) between periodic heartbeat pings sent from the
       connector to the server.
   * - ``lmcache.mp.mp_transfer_mode``
     - ``auto``
     - Routing mode for the worker -> server transfer context. One of
       ``auto`` (CUDA -> lmcache_driven, others -> engine_driven),
       ``lmcache_driven`` (force the IPC / SHM zero-copy handle path —
       LMCache server pulls data via device handles), or
       ``engine_driven`` (force the worker-side gather/scatter copy
       path). Overrides the ``LMCACHE_MP_TRANSFER_MODE`` env var when
       set.

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``LMCACHE_LOG_LEVEL``
     - Log level for LMCache (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
       Set to ``DEBUG`` to see L2 store activity, prefetch results, etc.
   * - ``PYTHONHASHSEED``
     - Set to a fixed value for reproducible hashing across processes
       (relevant when using ``--hash-algorithm builtin``).
   * - ``LMCACHE_TRACK_USAGE``
     - Set to ``false`` to disable anonymous usage statistics (see below).
   * - ``DO_NOT_TRACK``
     - Set to ``1`` to disable anonymous usage statistics (cross-tool
       convention).
   * - ``LMCACHE_USAGE_TRACK_INTERVAL``
     - Seconds between continuous usage-telemetry flushes (default
       ``600``). See :ref:`usage-stats-collection`.

Full Example
------------

.. code-block:: bash

    lmcache server \
        --host 0.0.0.0 \
        --port 6555 \
        --chunk-size 512 \
        --max-workers 4 \
        --max-gpu-workers 2 \
        --hash-algorithm blake3 \
        --engine-type default \
        --lookup-hash-log-dir /data/lmcache/lookup_hashes \
        --lookup-hash-log-rotation-interval 21600 \
        --lookup-hash-log-rotation-max-size 104857600 \
        --lookup-hash-log-max-files 100 \
        --l1-size-gb 100 \
        --l1-use-lazy \
        --l1-init-size-gb 20 \
        --l1-align-bytes 4096 \
        --l1-write-ttl-seconds 600 \
        --l1-read-ttl-seconds 300 \
        --eviction-policy noop \
        --l2-store-policy skip_l1 \
        --eviction-trigger-watermark 0.9 \
        --eviction-ratio 0.1 \
        --l2-prefetch-policy default \
        --l2-prefetch-max-in-flight 8 \
        --periodic-notifier-interval-ms 5 \
        --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}' \
        --prometheus-port 9090 \
        --metrics-sample-rate 0.01 \
        --enable-tracing \
        --otlp-endpoint http://localhost:4317

Anonymous Usage Statistics
--------------------------

The MP server reports anonymous usage statistics: a one-time
environment/configuration snapshot at startup and interval counters
(tokens retrieved/stored, bytes stored, uptime) every
``LMCACHE_USAGE_TRACK_INTERVAL`` seconds. No prompts, keys, KV-cache
data, model names, or ``--instance-id`` are ever sent, and reporting can
never affect serving. Opt out with ``LMCACHE_TRACK_USAGE=false`` or
``DO_NOT_TRACK=1``; see :ref:`usage-stats-collection` for details.
