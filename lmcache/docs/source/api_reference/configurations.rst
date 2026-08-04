Configuring LMCache
===================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/configuration`.


LMCache supports two types of configurations:

1. **Configuration file**: a YAML (recommended) or JSON file that contains the configuration items.
2. **Environment variables**: environment variables that start with ``LMCACHE_``. 

To use a configuration file, you can set the ``LMCACHE_CONFIG_FILE`` environment variable to the path of the configuration file.

.. note::

    The environment variable configurations will be ignored if the configuration file is present.


General Configurations
----------------------

Basic cache settings that control the core functionality of LMCache.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - chunk_size
     - LMCACHE_CHUNK_SIZE
     - Size of cache chunks. Default: 256
   * - local_cpu
     - LMCACHE_LOCAL_CPU
     - Whether to enable CPU caching. Values: true/false. Default: true
   * - max_local_cpu_size
     - LMCACHE_MAX_LOCAL_CPU_SIZE
     - Maximum CPU cache size in GB. Default: 5.0
   * - local_cpu_use_hugepages
     - LMCACHE_LOCAL_CPU_USE_HUGEPAGES
     - Whether to use Linux hugepages (2 MB) for CPU-pinned KV cache memory. Not compatible with P2P mode or shared memory (multiprocess). Requires pre-allocated hugepages (``sysctl vm.nr_hugepages``). Values: true/false. Default: false
   * - local_disk
     - LMCACHE_LOCAL_DISK
     - Path (or comma-separated paths) to local disk cache directories. Format: ``"file:///path/to/cache"`` or ``"/path/a,/path/b"`` for multi-device I/O. See ``local_disk_path_sharding`` for how paths are assigned to GPUs.
   * - local_disk_path_sharding
     - LMCACHE_LOCAL_DISK_PATH_SHARDING
     - Strategy for selecting a path when multiple paths are provided. Currently only ``"by_gpu"`` is supported, which selects paths based on GPU device ID (default: "by_gpu").
   * - max_local_disk_size
     - LMCACHE_MAX_LOCAL_DISK_SIZE
     - Maximum disk cache size in GB. Default: 0.0
   * - remote_url
     - LMCACHE_REMOTE_URL
     - Remote storage URL. Format: "protocol://host:port".
   * - remote_serde
     - LMCACHE_REMOTE_SERDE
     - Serialization format. Values: "naive" or "cachegen". Default: "naive"
   * - save_decode_cache
     - LMCACHE_SAVE_DECODE_CACHE
     - Whether to store decode KV cache. Values: true/false. Default: false
   * - use_layerwise
     - LMCACHE_USE_LAYERWISE
     - Whether to enable layerwise pipelining. Values: true/false. Default: false
   * - pre_caching_hash_algorithm
     - LMCACHE_PRE_CACHING_HASH_ALGORITHM
     - Hash algorithm for prefix-caching. Default: "builtin"
   * - save_unfull_chunk
     - LMCACHE_SAVE_UNFULL_CHUNK
     - Whether to save unfull chunks. Values: true/false. Default: false
   * - blocking_timeout_secs
     - LMCACHE_BLOCKING_TIMEOUT_SECS
     - Timeout for blocking operations in seconds. Default: 10
   * - py_enable_gc
     - LMCACHE_PY_ENABLE_GC
     - Whether to enable Python garbage collection. Values: true/false. Default: true
   * - cache_policy
     - LMCACHE_CACHE_POLICY
     - Cache eviction policy (e.g. "LRU", "LFU", "FIFO"). Default: "LRU"
   * - numa_mode
     - LMCACHE_NUMA_MODE
     - NUMA-aware memory allocation mode. Values: "auto" (detect from system), "manual" (use extra_config mapping), null (disabled). When enabled, allocates pinned memory on specific NUMA nodes for better GPU-CPU memory bandwidth. Default: null
   * - external_lookup_client
     - LMCACHE_EXTERNAL_LOOKUP_CLIENT
     - External KV lookup service URI (e.g., "mooncakestore://address"). If null, defaults to LMCache's internal lookup client. Default: null
   * - priority_limit
     - LMCACHE_PRIORITY_LIMIT
     - Caches requests only if priority value ≤ limit. (**Not applicable for PD Disaggregation**) Type: int. Default: None
   * - min_retrieve_tokens
     - LMCACHE_MIN_RETRIEVE_TOKENS
     - Minimum number of hit tokens required to perform retrieve. If hit tokens < this value, skip retrieve but still record the hits to avoid re-storing existing chunks. See :ref:`performance_tuning` for a working example. Default: 0 (disabled)
   * - store_location
     - LMCACHE_STORE_LOCATION
     - A single storage backend name to store KV caches into. When specified, only the matching backend receives store operations. Valid values are the backend class names registered in the storage manager, including: ``"LocalCPUBackend"``, ``"LocalDiskBackend"``, ``"RemoteBackend"``, ``"PDBackend"``, ``"P2PBackend"``, ``"GdsBackend"``, etc, and any storage plugin backends. Note: ``"PDBackend"`` cannot be used as a store location for a decoder instance in a PD setup, since PDBackend is one-way from prefiller to decoder only. Default: null (store to all active backends)
   * - retrieve_locations
     - LMCACHE_RETRIEVE_LOCATIONS
     - List of storage backend names to search when retrieving or looking up KV caches. When specified, only the listed backends are searched. Valid values are the backend class names registered in the storage manager, including: ``"LocalCPUBackend"``, ``"LocalDiskBackend"``, ``"RemoteBackend"``, ``"PDBackend"``, ``"P2PBackend"``, ``"GdsBackend"``, etc, and any storage plugin backends. Default: null (search all active backends)
   * - extra_config
     - LMCACHE_EXTRA_CONFIG={"key": value, ...}
     - Additional configuration as JSON dict. For NUMA manual mode, include "gpu_to_numa_mapping": {gpu_id: numa_node, ...}. Default: {}

Lazy Memory Allocator Configurations
------------------------------------

Settings for the lazy memory allocator that enables gradual memory allocation to reduce startup time and initial memory footprint.

.. note::

    The lazy memory allocator is designed for scenarios with large CPU memory configurations. It starts with a small initial allocation and gradually expands as needed, reducing startup wait time and avoiding unnecessary memory consumption when the full capacity is not required.
    
    **Key characteristics:**
    
    - **One-time expansion**: Memory expands until target size is reached, then stops
    - **No shrinking**: Once allocated, memory is never released back to the system
    - **Automatic activation**: Only activates when ``max_local_cpu_size`` exceeds ``lazy_memory_safe_size``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_lazy_memory_allocator
     - LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR
     - Whether to enable lazy memory allocator. Values: true/false. Default: false
   * - lazy_memory_initial_ratio
     - LMCACHE_LAZY_MEMORY_INITIAL_RATIO
     - Initial memory allocation ratio (0.0-1.0). Determines the fraction of max_local_cpu_size to allocate at startup. Default: 0.2 (20%)
   * - lazy_memory_expand_trigger_ratio
     - LMCACHE_LAZY_MEMORY_EXPAND_TRIGGER_RATIO
     - Memory usage ratio (0.0-1.0) that triggers expansion. When used memory exceeds this ratio of current capacity, expansion begins. Default: 0.5 (50%)
   * - lazy_memory_step_ratio
     - LMCACHE_LAZY_MEMORY_STEP_RATIO
     - Memory expansion step ratio (0.0-1.0). Each expansion adds this fraction of max_local_cpu_size. Default: 0.1 (10%)
   * - lazy_memory_safe_size
     - LMCACHE_LAZY_MEMORY_SAFE_SIZE
     - Threshold in GB above which lazy allocator activates. If max_local_cpu_size ≤ this value, lazy allocator is disabled regardless of enable_lazy_memory_allocator setting. Default: 0.0
   * - reserve_local_cpu_size
     - LMCACHE_RESERVE_LOCAL_CPU_SIZE
     - Reserved system memory in GB that should not be allocated by LMCache. Used to prevent out-of-memory conditions. Default: 0.0
     
Cache Blending Configurations
-----------------------------

Settings related to cache blending functionality.

.. note::

    We have an end-to-end `example <https://github.com/LMCache/LMCache/tree/dev/examples/blend_kv_v1>`_.
    We also have more :doc:`detailed documentation <../kv_cache_optimizations/blending>`.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_blending
     - LMCACHE_ENABLE_BLENDING
     - Whether to enable blending. Values: true/false. Default: false
   * - blend_recompute_ratios
     - LMCACHE_BLEND_RECOMPUTE_RATIOS
     - Ratio of blending recompute. Default: 0.15
   * - blend_check_layers
     - LMCACHE_BLEND_CHECK_LAYERS
     - Layers to determine the recomputed tokens. Default: 1
   * - blend_special_str
     - LMCACHE_BLEND_SPECIAL_STR
     - Separator string for blending. Default: " # # "

Peer-to-Peer Sharing Configurations
-----------------------------------

Settings for enabling and configuring peer-to-peer CPU KV cache sharing and global KV cache lookup.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_p2p
     - LMCACHE_ENABLE_P2P
     - Whether to enable peer-to-peer sharing. Requires ``enable_controller=true`` and the controller settings below. Values: true/false. Default: false
   * - p2p_host
     - LMCACHE_P2P_HOST
     - Ip address. Required if enable_p2p is true
   * - p2p_init_ports
     - LMCACHE_P2P_INIT_PORTS
     - Ports for p2p peer init. Required if enable_p2p is true
   * - p2p_lookup_ports
     - LMCACHE_P2P_LOOKUP_PORTS
     - Ports for p2p peer lookup. Required if enable_p2p is true
   * - transfer_channel
     - LMCACHE_TRANSFER_CHANNEL
     - Such as `nixl`. Required if enable_p2p is true

Controller Configurations
-------------------------

Settings for the KV cache controller functionality.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_controller
     - LMCACHE_ENABLE_CONTROLLER
     - Whether to enable controller. Values: true/false. Default: false
   * - lmcache_instance_id
     - LMCACHE_LMCACHE_INSTANCE_ID
     - ID of the LMCache instance. Must be non-null if ``enable_controller`` is true. If omitted, a unique ``lmcacheengineconfig_<uuid>`` value is generated
   * - controller_pull_url
     - LMCACHE_CONTROLLER_PULL_URL
     - URL used by workers to pull commands from the controller server. Required if ``enable_controller`` is true
   * - controller_reply_url
     - LMCACHE_CONTROLLER_REPLY_URL
     - URL used by workers to send replies to the controller server. Required if ``enable_controller`` is true
   * - lmcache_worker_ports
     - LMCACHE_LMCACHE_WORKER_PORTS
     - Non-empty list of port numbers for LMCache workers. Required if ``enable_controller`` is true

Disaggregated Prefill Configurations
-------------------------------------------

Settings for disaggregated prefill functionality. The latest/default PD is implemented inside of `lmcache/v1/storage_backend/pd_backend.py`.

.. note::

    When PD is enabled, the following restrictions apply (welcome contributions to remove these restrictions):
    
    - remote_url must be null
    - save_decode_cache must be false
    - enable_p2p must be false

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_pd
     - LMCACHE_ENABLE_PD
     - Whether to enable PD. Values: true/false. Default: false
   * - transfer_channel
     - LMCACHE_TRANSFER_CHANNEL
     - Transfer channel used for PD. Values: "nixl". Default: none
   * - pd_role
     - LMCACHE_PD_ROLE
     - PD role. Values: "sender" (prefiller) or "receiver" (decoder).
   * - pd_buffer_size
     - LMCACHE_PD_BUFFER_SIZE
     - Upper bound of PD transport buffer size (in bytes), aligned to chunk size. Required for both senders and receivers when enable_pd=true
   * - pd_buffer_device
     - LMCACHE_PD_BUFFER_DEVICE
     - Device for PD buffer. Values: "cpu", "cuda". Required for both senders and receivers when enable_pd=true
   * - nixl_backends
     - LMCACHE_NIXL_BACKENDS
     - List of Nixl transport backends. Useful for non-disaggregated use case (see below). UCX default is sufficient for disagg use case. Default: ["UCX"]
   * - pd_peer_host
     - LMCACHE_PD_PEER_HOST
     - Host for peer connections. Required for receivers to bind to
   * - pd_peer_init_port
     - LMCACHE_PD_PEER_INIT_PORT
     - Initialization port for peer connections. Required for receivers to bind to
   * - pd_peer_alloc_port
     - LMCACHE_PD_PEER_ALLOC_PORT
     - Allocation port for peer connections. Required for receivers to bind to
   * - pd_proxy_host
     - LMCACHE_PD_PROXY_HOST
     - Host for proxy server. Required for senders to connect to inform the proxy when transfer to decoder has been completed
   * - pd_proxy_port
     - LMCACHE_PD_PROXY_PORT
     - Port for proxy server. Required for senders to connect to inform the proxy when transfer to decoder has been completed
   * - pd_allocation_timeout_sec
     - LMCACHE_PD_ALLOCATION_TIMEOUT_SEC
     - Maximum seconds to retry memory allocation before giving up. Default: infinity (no timeout)
   * - pd_shutdown_timeout_sec
     - LMCACHE_PD_SHUTDOWN_TIMEOUT_SEC
     - Maximum seconds to wait for event loop shutdown and thread join. Default: 5.0
   * - pd_condition_poll_interval_sec
     - LMCACHE_PD_CONDITION_POLL_INTERVAL_SEC
     - Polling interval in seconds when waiting on a threading/asyncio Condition. Small enough to be responsive, large enough not to spin-waste CPU. Default: 0.005
   * - pd_max_prefill_len
     - LMCACHE_PD_MAX_PREFILL_LEN
     - Maximum prefill token length that the PD buffer must be able to hold. If > 0, initialization raises ValueError when the buffer capacity (in tokens) is smaller than this value. Set to 0 (default) to skip the check.
   * - pd_backend_mode
     - LMCACHE_PD_BACKEND_MODE
     - Select the PD backend implementation: 'async' (default) uses the asyncio-based implementation; 'sync' uses the original thread-based synchronous implementation. Default: "async"
   * - pd_skip_proxy_notification
     - LMCACHE_PD_SKIP_PROXY_NOTIFICATION
     - When true, the sender skips ZMQ proxy notification after KV transfer and does not require pd_proxy_host/pd_proxy_port. This option is intended for external orchestrators only (e.g., vLLM Production Stack router) that manage the prefill-decode request flow via HTTP and do not rely on ZMQ notifications. It must not be used with LMCache's built-in disaggregation proxy (``disagg_proxy_server.py``), which depends on ZMQ notifications to know when KV transfer is complete before forwarding the decode request. Values: true/false. Default: false
   * - pd_bidirectional
     - LMCACHE_PD_BIDIRECTIONAL
     - When true, enables bidirectional NIXL cache probe. The prefiller queries the decoder for cached KV blocks before transfer, and reads cached blocks via NIXL RDMA instead of recomputing. Values: true/false. Default: false
   * - pd_peer_query_port
     - LMCACHE_PD_PEER_QUERY_PORT
     - ZMQ ports for the bidirectional cache query channel (one per TP rank). Required on both prefiller and decoder when pd_bidirectional=true. Example: [7500, 7501, 7502, 7503]

P2P Backend Configurations
--------------------------

Settings for P2P (peer-to-peer) backend timeout behavior. These configurations are specified through ``extra_config``.

.. code-block:: yaml

    extra_config:
      p2p_socket_recv_timeout_ms: 30000
      p2p_socket_send_timeout_ms: 10000

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Configuration Key
     - Default
     - Description
   * - p2p_socket_recv_timeout_ms
     - 30000
     - Timeout in milliseconds for socket receive operations
   * - p2p_socket_send_timeout_ms
     - 10000
     - Timeout in milliseconds for socket send operations

Disk I/O Backend Configurations
---------------------------------

Settings shared by disk-based storage backends (local disk and GDS). These configurations are specified through ``extra_config``.

.. code-block:: yaml

    extra_config:
      use_odirect: true
      disk_io_threads: 8

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Configuration Key
     - Default
     - Description
   * - use_odirect
     - false
     - Enable O_DIRECT for disk I/O, bypassing the kernel page cache. Recommended when most local CPU memory is already used for KV cache offloading.
   * - disk_io_threads
     - 4
     - Number of worker threads in the disk I/O thread pool. Applies to both the local disk backend and the GDS backend. Increase for higher parallelism on fast NVMe drives. (Replaces the deprecated ``gds_io_threads`` key.)

Nixl (as a storage backend) Configurations
------------------------------------------

Settings for using Nixl as a storage backend instead of disaggregated prefill. This mode requires additional configurations in ``extra_config``.

.. note::

    This is a different mode from disaggregated prefill. When using Nixl as a storage backend, you need to configure it through ``extra_config``.

.. code-block:: yaml

  
    extra_config: 
      # enable_nixl_storage will disable disaggregated prefill mode.
      enable_nixl_storage: true
      nixl_backend: "POSIX"  # Options: "GDS", "GDS_MT", "POSIX", "HF3FS", "OBJ"
      nixl_path: "/path/to/storage/"
      nixl_pool_size: 64

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Configuration Key
     - Description
   * - enable_nixl_storage
     - Whether to enable Nixl storage backend. Values: true/false
   * - nixl_backend
     - Storage backend type. Options: "GDS", "GDS_MT", "POSIX", "HF3FS", "OBJ"
   * - nixl_path
     - File system path for Nixl storage
   * - nixl_pool_size
     - Number of files or objects in the storage pool
   * - nixl_endpoint_list
     - List of object-storage endpoint URLs for per-worker distribution. Each TP worker selects an entry round-robin by ``local_worker_id``, overriding ``nixl_backend_params.endpoint_override``. Only applied when ``nixl_backend`` is ``"OBJ"`` (silently ignored otherwise). Each entry must start with ``http://`` or ``https://``; an empty list raises ``ValueError`` at engine init.
   * - nixl_use_hugepages
     - **Deprecated.** Use ``local_cpu_use_hugepages`` instead. When set, the value is copied into ``local_cpu_use_hugepages`` (a warning is logged) and the key is dropped. Hugepages have never applied to GPU buffers; in CPU mode the NIXL pool is now owned by ``LocalCPUBackend``.


Additional Storage Configurations
---------------------------------

Settings for different storage backends and paths.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - gds_path
     - LMCACHE_GDS_PATH
     - Path for GDS backend. Supports comma-separated paths for multi-device I/O (e.g. ``/mnt/nvme0/cache,/mnt/nvme1/cache``). See ``gds_path_sharding`` for how paths are assigned to GPUs.
   * - gds_path_sharding
     - LMCACHE_GDS_PATH_SHARDING
     - Strategy for selecting a path when multiple paths are provided. Currently only ``"by_gpu"`` is supported, which selects paths based on GPU device ID (default: "by_gpu").
   * - gds_buffer_size
     - LMCACHE_GDS_BUFFER_SIZE
     - Buffer size for GDS operations
   * - use_gds
     - LMCACHE_USE_GDS
     - Enable or disable GPU Direct Storage API usage (default: true)
   * - gds_backend
     - LMCACHE_GDS_BACKEND
     - GDS library backend to use (default: "cufile")

Custom Prometheus Histogram Buckets
------------------------------------

You can override the default bucket boundaries for any Prometheus histogram metric
by adding a key ``histogram_bucket_<name>`` to ``extra_config``, where ``<name>``
is the metric name **without** the ``lmcache:`` prefix.

The value must be a list of numeric boundaries (floats or ints).

.. code-block:: yaml

    extra_config:
      histogram_bucket_time_to_retrieve: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
      histogram_bucket_retrieve_speed: [100, 500, 1000, 5000, 10000]

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Available Histogram Names
     - Description
   * - ``time_to_retrieve``
     - Time to retrieve from cache (seconds)
   * - ``time_to_store``
     - Time to store to cache (seconds)
   * - ``time_to_lookup``
     - Time to lookup in cache (seconds)
   * - ``retrieve_process_tokens_time``
     - Time to process tokens in retrieve (seconds)
   * - ``retrieve_broadcast_time``
     - Time to broadcast memory objects in retrieve (seconds)
   * - ``retrieve_to_gpu_time``
     - Time to move data to GPU in retrieve (seconds)
   * - ``remote_backend_batched_get_blocking_time``
     - Time to get data from remote backend (seconds)
   * - ``instrumented_connector_batched_get_time``
     - Time used by the connector (seconds)
   * - ``store_process_tokens_time``
     - Time to process tokens in store (seconds)
   * - ``store_from_gpu_time``
     - Time to move data from GPU in store (seconds)
   * - ``store_put_time``
     - Time to put data to storage in store (seconds)
   * - ``retrieve_speed``
     - Retrieve speed (tokens per second)
   * - ``store_speed``
     - Store speed (tokens per second)
   * - ``p2p_time_to_transfer``
     - Time to transfer via P2P (seconds)
   * - ``p2p_transfer_speed``
     - P2P transfer speed (tokens per second)
   * - ``remote_time_to_get``
     - Time to get from remote backends (ms)
   * - ``remote_time_to_put``
     - Time to put to remote backends (ms)
   * - ``remote_time_to_get_sync``
     - Time to get from remote backends synchronously (ms)
   * - ``request_cache_hit_rate``
     - Request cache hit rate (0.0 to 1.0)
   * - ``request_cache_lifespan``
     - Request cache lifespan (minutes)

Internal API Server Configurations
----------------------------------

Settings for the internal API server that provides management and debugging APIs for LMCache engines. The API server runs on each worker and scheduler, allowing you to inspect and control LMCache behavior at runtime.

.. note::

    The internal API server provides endpoints for:
    
    - **Metrics**: Performance and cache statistics 
    - **Configuration**: Runtime configuration inspection
    - **Metadata**: Engine and model metadata
    - **Threads**: Thread debugging information
    - **Log Level**: Dynamic log level adjustment
    - **Script Execution**: Run custom Python scripts with access to the LMCache engine

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - internal_api_server_enabled
     - LMCACHE_INTERNAL_API_SERVER_ENABLED
     - Whether to enable internal API server. Default: false
   * - internal_api_server_host
     - LMCACHE_INTERNAL_API_SERVER_HOST
     - Host for internal API server to bind to. Default: "0.0.0.0"
   * - internal_api_server_port_start
     - LMCACHE_INTERNAL_API_SERVER_PORT_START
     - Starting port for internal API server. Port assignment: Scheduler = port_start + 0, Worker i = port_start + i + 1. Example: If port_start=6999, then Scheduler=6999, Worker 0=7000, Worker 1=7001, etc. Default: 6999
   * - internal_api_server_include_index_list
     - LMCACHE_INTERNAL_API_SERVER_INCLUDE_INDEX_LIST
     - List of worker/scheduler indices to enable API server on. Use 0 for scheduler, 1 for worker 0, 2 for worker 1, etc. If null, enables on all workers/scheduler. Example: [0, 1] enables only on scheduler and worker 0. Default: null
   * - internal_api_server_socket_path_prefix
     - LMCACHE_INTERNAL_API_SERVER_SOCKET_PATH_PREFIX
     - If specified, use Unix domain sockets instead of TCP ports. Socket paths will be "{prefix}_{port}". Example: "/tmp/lmcache_api_socket" creates "/tmp/lmcache_api_socket_6999", "/tmp/lmcache_api_socket_7000", etc. Default: null

Plugin Configurations
---------------------

Settings for plugin system.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - runtime_plugin_locations
     - LMCACHE_RUNTIME_PLUGIN_LOCATIONS
     - List of runtime plugin locations. Default: null

Deprecated Configurations
-------------------------

These configurations are deprecated and may be removed in future versions.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - audit_actual_remote_url
     - LMCACHE_AUDIT_ACTUAL_REMOTE_URL
     - (Deprecated) URL of actual remote LMCache instance for auditing. Use extra_config['audit_actual_remote_url'] instead
   * - plugin_locations
     - LMCACHE_PLUGIN_LOCATIONS
     - Deprecated alias for ``runtime_plugin_locations``
     
