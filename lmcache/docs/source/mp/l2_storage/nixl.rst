NIXL
====

NIXL-based persistent storage — the primary production L2 backend, using NIXL
(NVIDIA Interconnect Library) for high-performance storage I/O. Two adapter
types share this backend:

- ``nixl_store`` — a fixed pool of storage descriptors pre-allocated at init.
- ``nixl_store_dynamic`` — opens and registers files per operation, adding
  persist/recover across restarts and removing the open-file-descriptor limit.

.. note::

   Both adapters require the NIXL runtime, which ships as the optional
   ``lmcache[nixl]`` extra:

   .. code-block:: bash

       uv pip install lmcache[nixl]

Static pool — ``nixl_store``
----------------------------

The primary production adapter. Pre-allocates a pool of storage descriptors at
initialization.

**Required fields:**

- ``backend``: Storage backend -- one of ``POSIX``, ``GDS``, ``GDS_MT``,
  ``HF3FS``, ``OBJ``, ``AZURE_BLOB``.
- ``pool_size``: Number of storage descriptors to pre-allocate (must be > 0).

**Backend-specific parameters (``backend_params``):**

File-based backends (``GDS``, ``GDS_MT``, ``POSIX``, ``HF3FS``) require:

- ``file_path``: Directory path for storing L2 data.
- ``use_direct_io``: ``"true"`` or ``"false"`` -- whether to use direct I/O.

The ``OBJ`` and ``AZURE_BLOB`` backends (object stores) do not require ``file_path``.

**Backend descriptions:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Backend
     - Description
   * - ``POSIX``
     - Standard POSIX file I/O.  Works on any file system.  No direct I/O.
   * - ``GDS``
     - NVIDIA GPU Direct Storage.  Enables direct GPU-to-storage transfers
       bypassing the CPU.  Requires NVMe SSDs with GDS support.
   * - ``GDS_MT``
     - Multi-threaded variant of GDS for higher throughput.
   * - ``HF3FS``
     - Shared file system backend (e.g., for distributed/networked storage).
   * - ``OBJ``
     - Object store backend.  No local file path required.
   * - ``AZURE_BLOB``
     - Object store backend for Azure Blob Storage.  No local file path required.

**Configuration examples:**

.. code-block:: bash

    # POSIX backend
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

    # GDS backend
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "true"}, "pool_size": 128}'

    # GDS_MT backend
    --l2-adapter '{"type": "nixl_store", "backend": "GDS_MT", "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "true"}, "pool_size": 128}'

    # HF3FS backend
    --l2-adapter '{"type": "nixl_store", "backend": "HF3FS", "backend_params": {"file_path": "/mnt/hf3fs/lmcache", "use_direct_io": "false"}, "pool_size": 64}'

    # OBJ backend
    --l2-adapter '{"type": "nixl_store", "backend": "OBJ", "backend_params": {}, "pool_size": 32}'

    # AZURE_BLOB backend
    --l2-adapter '{"type": "nixl_store", "backend": "AZURE_BLOB", "backend_params": {"account_url": "https://<account_name>.blob.core.windows.net", "container_name": "<container_name>"}, "pool_size": 32}'

Dynamic (persist / recover) — ``nixl_store_dynamic``
----------------------------------------------------

A dynamic variant of the NIXL adapter that opens and registers files
per-operation instead of pre-allocating them at init. This enables:

- **Persist/recover** -- cached KV metadata survives restarts.
- **No fd limits** -- files are opened and closed per transfer, so the
  cache can grow beyond OS open-file-descriptor limits.

.. note::

   Only file-based backends are supported (``POSIX``, ``GDS``, ``GDS_MT``,
   ``HF3FS``). The ``OBJ`` and ``AZURE_BLOB`` backends are not supported yet.

**Required fields:**

- ``backend``: Storage backend -- one of ``POSIX``, ``GDS``, ``GDS_MT``,
  ``HF3FS``.

**Backend-specific parameters (``backend_params``):**

- ``file_path``: Directory path for storing L2 data files.
- ``use_direct_io``: ``"true"`` or ``"false"``.
- ``max_capacity_gb``: Maximum storage capacity in GB. The adapter
  rejects stores when this limit is reached. Required for the eviction
  controller to compute usage.

**Optional fields (for persist):**

- ``persist_enabled`` (bool, default ``true``): If ``true``, data files
  are kept on disk at shutdown. If ``false``, all data files are deleted
  on shutdown.

Lookup always checks secondary storage (disk) on miss and lazily
populates the in-memory index when a file is found.

**Configuration examples:**

.. code-block:: bash

    # Basic dynamic POSIX backend (persist enabled by default)
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10"}}'

    # Explicitly disable persist
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10"}, "persist_enabled": false}'

    # With eviction
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true", "max_capacity_gb": "50"}, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'

**Persist / secondary lookup behaviour:**

- On **shutdown**, the adapter keeps data files on disk by default
  (``persist_enabled`` defaults to ``true``). If explicitly set to
  ``false``, all data files are deleted to avoid orphaned storage.
- On **startup**, the in-memory index is empty. Every lookup miss falls
  through to a secondary lookup on disk: if the deterministic file
  exists, it is treated as a hit and the in-memory index is populated
  lazily from the file size.
