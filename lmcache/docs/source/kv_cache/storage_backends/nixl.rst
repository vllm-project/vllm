
Nixl
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/nixl`.


.. _nixl-overview:

Overview
--------

NIXL (NVIDIA Inference Xfer Library) is a high-performance library designed for accelerating point to point communications in AI inference frameworks. It provides an abstraction over various types of memory (CPU and GPU) and storage through a modular plug-in architecture, enabling efficient data transfer and coordination between different components of the inference pipeline.

LMCache supports using NIXL as a storage backend, allowing using NIXL to save either GPU or CPU memory into storage.

Prerequisites
~~~~~~~~~~~~~

- **LMCache**: Install with ``pip install lmcache``
- **NIXL**: Install from `NIXL GitHub repository <https://github.com/ai-dynamo/nixl>`_
- **Model Access**: Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct

Ways to configure LMCache NIXL Offloading
-----------------------------------------

**Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=lmcache-config.yaml``

Example ``lmcache-config.yaml`` for POSIX backend:

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_device: cpu
    local_cpu_use_hugepages: true  # optional, requires pre-allocated hugepages
    extra_config:
      enable_nixl_storage: true
      nixl_backend: POSIX
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      use_direct_io: true

Key settings:

- ``nixl_buffer_size``: buffer size for NIXL transfers. **GPU mode only** (``nixl_buffer_device: cuda``). Setting this with ``nixl_buffer_device: cpu`` is a configuration error and will be rejected — in CPU mode NIXL shares ``LocalCPUBackend``'s pinned pool, which is sized by ``max_local_cpu_size``.

- ``max_local_cpu_size``: size of ``LocalCPUBackend``'s pinned pool in GiB. In CPU mode, this pool is shared with NIXL and must accommodate both the hot cache and concurrent NIXL I/O in flight. Must be > 0 when ``nixl_buffer_device: cpu``. Default: ``5.0``.

- ``nixl_pool_size``: number of descriptors opened at init time for nixl backend. Set to 0 for dynamic mode.

- ``nixl_path``: directory (or list of directories) under which the storage files will be saved (e.g. /mnt/nixl/). Needed for NIXL backends that store to file. When using a list of paths with ``path_sharding``, paths will be selected based on the sharding strategy.

- ``nixl_buffer_device``: dictates where the memory managed by NIXL should be on. "cpu" or "cuda" is supported for "GDS", "GDS_MT", and "OBJ" backends - for "POSIX", "HF3FS", "AZURE_BLOB" & "DOCA_MEMOS", must be "cpu". In CPU mode, NIXL shares ``LocalCPUBackend``'s pinned buffer; ``LocalCPUBackend`` is always created when ``nixl_buffer_device: cpu``, regardless of the ``local_cpu`` setting. ``local_cpu: false`` still suppresses hot-cache promotions — the backend acts as a staging buffer only, mirroring how ``local_disk`` already uses ``LocalCPUBackend``.

- ``nixl_backend``: configuration of which nixl backend to use for storage.

- ``nixl_path_sharding``: strategy for selecting path when multiple paths are provided. Currently only "by_gpu" is supported, which selects paths based on GPU device ID.
- ``local_cpu_use_hugepages``: whether to use Linux hugepages (2 MiB) for ``LocalCPUBackend``'s pinned pool (which NIXL shares in CPU mode). Requires pre-allocated hugepages (``sysctl vm.nr_hugepages``). Default: ``false``. **Deprecated alias:** ``extra_config.nixl_use_hugepages`` — accepted with a warning and copied into this field; will be removed in a future release.

.. note::

    In CPU mode, the shared paged allocator consumes one full page per object. With ``save_unfull_chunk: true`` (only valid in static mode — dynamic mode rejects it; see "Dynamic Mode" → "Restrictions" below), partial chunks still occupy a full page each, so effective capacity degrades proportionally to the fraction of unfull last chunks across active sequences.

.. note::

    ``enable_p2p: true`` is rejected together with ``nixl_buffer_device: cpu``. The combination is structurally supported — both backends share ``LocalCPUBackend``'s pinned pool, each runs its own NIXL agent over it, and allocations route through ``LocalCPUBackend.allocate()`` — but it has not been exercised end-to-end and has no CI coverage. Use ``enable_p2p: true`` with ``nixl_buffer_device: cuda`` instead, or disable ``enable_p2p`` when running the NIXL CPU shared pool.

- ``nixl_presence_cache``: whether to keep an in-DRAM presence cache of keys known to exist, so repeated existence checks for the same key are answered locally instead of via a NIXL ``query_memory`` call. Applies to the dynamic backend (``nixl_pool_size: 0``). Default: ``false``.

- ``nixl_presence_cache_only``: when ``true``, the dynamic NIXL backend treats the local presence cache (and in-progress put set) as authoritative for existence checks. If a key is not known locally, lookup reports a miss **without** issuing a NIXL ``query_memory`` check. This requires ``nixl_presence_cache: true`` and can intentionally produce **false negatives** for objects that exist in the underlying storage but are absent from local presence metadata — giving "DRAM-only metadata" semantics where a process restart always yields a logically empty cache. Default: ``false``.

  .. note::

     This is a **lookup/existence-check** mode, not a "never touch the underlying storage" policy. It gates ``contains`` / ``batched_contains`` (and the async variant); direct retrieval still reads from the underlying storage. In normal operation a retrieval is only issued for a key that lookup already reported as present, so locally-unknown keys are not fetched. The option is only consulted by the dynamic backend (``nixl_pool_size: 0``); it is accepted but unused for static configurations.

.. note::

    Supported backends are: ["GDS", "GDS_MT", "POSIX", "HF3FS", "OBJ", "AZURE_BLOB", "DOCA_MEMOS"].

    Backend specific params should be provided via ``extra_config.nixl_backend_params``. Please refer to NIXL documentation for specifics.

Example ``lmcache-config.yaml`` for POSIX backend with multipath support:

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_size: 1073741824 # 1GB
    nixl_buffer_device: cpu
    extra_config:
      enable_nixl_storage: true
      nixl_backend: POSIX
      nixl_pool_size: 64
      nixl_path: 
        - /mnt/nixl/cache0/
        - /mnt/nixl/cache1/
        - /mnt/nixl/cache2/
      nixl_path_sharding: by_gpu
      use_direct_io: True

Example ``lmcache-config.yaml`` for OBJ backend using S3 API:

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_device: cpu
    max_local_cpu_size: 1  # GiB
    extra_config:
      enable_nixl_storage: true
      nixl_backend: OBJ
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      nixl_backend_params:
        access_key: <your_access_key>
        secret_key: <your_secret_key>
        bucket: <your_bucket>
        region: <your_region>

Example ``lmcache-config.yaml`` for POSIX backend using liburing:

.. note::

    using POSIX backend with liburing requires NIXL to be built with liburing support.

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_device: cpu
    max_local_cpu_size: 1  # GiB
    extra_config:
      enable_nixl_storage: true
      nixl_backend: POSIX
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      use_direct_io: True
      nixl_backend_params:
        use_uring: "true"

Example ``lmcache-config.yaml`` for AZURE_BLOB backend to offload using Azure Blob Storage API:

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_device: cpu
    max_local_cpu_size: 1  # GiB
    extra_config:
      enable_nixl_storage: true
      nixl_backend: AZURE_BLOB
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      nixl_backend_params:
        account_url: https://<your_azure_storage_account_name>.blob.core.windows.net
        container_name: <your_container_name>

Per-Worker Endpoint Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the OBJ backend with multiple tensor-parallel (TP) workers, you can
distribute workers across multiple object-storage endpoints by providing a list of
endpoints via ``nixl_endpoint_list``. Each worker selects an endpoint in
round-robin order based on its ``local_worker_id`` (the worker ID within its host).

.. code-block:: yaml

    extra_config:
      enable_nixl_storage: true
      nixl_backend: OBJ
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      nixl_endpoint_list:
        - https://node-0.object-storage:9021
        - https://node-1.object-storage:9021
        - https://node-2.object-storage:9021
      nixl_backend_params:
        access_key: <your_access_key>
        secret_key: <your_secret_key>
        bucket: <your_bucket>
        region: <your_region>

.. note::

    When ``nixl_endpoint_list`` is set, any ``endpoint_override`` value in
    ``nixl_backend_params`` is ignored (a warning is logged).

    ``nixl_endpoint_list`` is only honored for the OBJ backend; it is ignored
    for all other backends (including DOCA_MEMOS, AZURE_BLOB, and the file
    backends).

Dynamic Mode
~~~~~~~~~~~~~

Nixl Storage Backend also supports a dynamic mode, which creates nixl storage descriptors on demand instead of at init time.

In order to use dynamic mode, extra_config.nixl_pool_size should be set to 0.

Restrictions
^^^^^^^^^^^^

- Dynamic mode is supported for object backends ("OBJ", "AZURE_BLOB", "DOCA_MEMOS") and file backends ("POSIX", "GDS", "GDS_MT", "HF3FS").
- save_unfull_chunk must be set to False.

Example ``lmcache-config.yaml`` for OBJ backend with dynamic mode:

.. code-block:: yaml

  chunk_size: 256
  local_cpu: False
  save_unfull_chunk: False
  enable_async_loading: False # set to True to test async loading
  nixl_buffer_device: cpu
  max_local_cpu_size: 3  # GiB
  extra_config:
    enable_nixl_storage: true
    nixl_backend: OBJ
    nixl_pool_size: 0
    nixl_presence_cache: False
    nixl_async_put: False
    nixl_backend_params:
      access_key: <your_access_key>
      secret_key: <your_secret_key>
      bucket: <your_bucket>
      region: <your_region>
      endpoint_override: https://url-to-object-storage
      ca_bundle: path to self-signed certificate # remove this line if not using self-signed certificate


Example ``lmcache-config.yaml`` for AZURE_BLOB backend with dynamic mode:

.. code-block:: yaml

  chunk_size: 256
  local_cpu: False
  save_unfull_chunk: False
  enable_async_loading: False # set to True to test async loading
  nixl_buffer_device: cpu
  max_local_cpu_size: 3  # GiB
  extra_config:
    enable_nixl_storage: true
    nixl_backend: AZURE_BLOB
    nixl_pool_size: 0
    nixl_presence_cache: False
    nixl_async_put: False
    nixl_backend_params:
      account_url: https://<your_azure_storage_account_name>.blob.core.windows.net
      container_name: <your_container_name>

DOCA_MEMOS Backend (NVIDIA CMX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DOCA_MEMOS`` stores KV cache on NVIDIA CMX (Context Memory Storage), a
BlueField-4 context-memory tier accessed through NIXL. It is an object-style
backend (like ``OBJ``), supported in both static (``nixl_pool_size`` > 0) and
dynamic (``nixl_pool_size`` = 0) mode. ``nixl_buffer_device`` must be ``cpu``.
``nixl_endpoint_list`` is not supported for DOCA_MEMOS.

Object names are 128-bit lowercase-hex strings: the NIXL DOCA_MEMOS plugin
passes object names as strings and hex-decodes them on the device side, so
each name is exactly 32 hex characters. In dynamic mode this name is a
truncated SHA-256 of the cache key, so names are opaque (they carry no
model/chunk debug information) and uniqueness is probabilistic at 128 bits.

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_device: cpu
    max_local_cpu_size: 1  # GiB
    extra_config:
      enable_nixl_storage: true
      nixl_backend: DOCA_MEMOS
      nixl_pool_size: 64
      nixl_backend_params:
        # refer to NIXL DOCA_MEMOS plugin docs for connection params
