.. _vllm_apis:

vLLM / Inference APIs
=====================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/http_api`.


These APIs are specific to vLLM inference workers and provide cache management,
configuration, freeze control, chunk statistics, and version information.

.. contents:: Endpoints
   :local:
   :depth: 2


Version & Info
--------------

``GET /lmc_version`` — LMCache Version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the LMCache library version string.

- **Method**: ``GET``
- **Path**: ``/lmc_version``
- **Parameters**: None
- **Response**: Plain version string.

.. code-block:: bash

    curl http://localhost:7000/lmc_version


``GET /commit_id`` — LMCache Commit ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the LMCache git commit ID.

- **Method**: ``GET``
- **Path**: ``/commit_id``
- **Parameters**: None
- **Response**: Plain commit ID string.

.. code-block:: bash

    curl http://localhost:7000/commit_id


``GET /version`` — Full Version Info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get full version info (version + commit ID).

- **Method**: ``GET``
- **Path**: ``/version``
- **Parameters**: None
- **Response**: Combined version string.

.. code-block:: bash

    curl http://localhost:7000/version


``GET /inference_info`` — Inference Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get inference information including vLLM config and LMCache details.

- **Method**: ``GET``
- **Path**: ``/inference_info``
- **Parameters**:

  ========== ======= =============================================
  Name       Type    Description
  ========== ======= =============================================
  ``format`` str     (Optional) Reserved for future use
  ========== ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/inference_info

**Error Response** (HTTP 500):

.. code-block:: json

    {
      "error": "Failed to get inference info",
      "message": "..."
    }


``GET /inference_version`` — vLLM Version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the vLLM version information.

- **Method**: ``GET``
- **Path**: ``/inference_version``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/inference_version

**Example Response**:

.. code-block:: json

    {
      "vllm_version": "0.8.0"
    }


Configuration & Metadata
-------------------------

``GET /conf`` — Get Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get current LMCache engine configuration values.

- **Method**: ``GET``
- **Path**: ``/conf``
- **Parameters**:

  ========= ======= =============================================
  Name      Type    Description
  ========= ======= =============================================
  ``names`` str     (Optional) Comma-separated list of config names to filter
  ========= ======= =============================================

- **Response**: ``application/json`` — JSON object of configuration key-value pairs.

.. code-block:: bash

    # Get all config
    curl http://localhost:7000/conf

    # Get specific config values
    curl "http://localhost:7000/conf?names=min_retrieve_tokens,save_decode_cache"


``POST /conf`` — Update Configuration (Experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update one or more configuration values at runtime.

.. warning::

    This feature is currently **experimental**. All configuration keys are
    mutable at runtime by default unless explicitly marked as
    ``"mutable": False`` in ``_CONFIG_DEFINITIONS``. The default will be
    changed to **immutable** once the feature is stabilized.

    Updating a configuration only modifies the value in the
    ``LMCacheEngineConfig`` object. If a component has already cached the
    value elsewhere, the change will **not** take effect for that component.

- **Method**: ``POST``
- **Path**: ``/conf``
- **Content-Type**: ``application/json``
- **Request Body**: JSON object with config name-value pairs.
- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST http://localhost:7000/conf \
      -H "Content-Type: application/json" \
      -d '{"min_retrieve_tokens": 512, "save_decode_cache": true}'

**Example Response** (HTTP 200):

.. code-block:: json

    {
      "updated": {
        "min_retrieve_tokens": 512,
        "save_decode_cache": true
      }
    }

**Example Response** (partial failure, HTTP 400):

.. code-block:: json

    {
      "updated": {"min_retrieve_tokens": 512},
      "errors": {"unknown_key": "Unknown config"}
    }

**Error Cases**:

- Unknown config key → ``"Unknown config"``
- Immutable config key → ``"Config is not mutable at runtime"``
- Invalid JSON body → HTTP 400


``GET /meta`` — Engine Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get metadata of the LMCache engine (e.g., worker_id, model_name, kv_shape).

- **Method**: ``GET``
- **Path**: ``/meta``
- **Parameters**:

  ========= ======= =============================================
  Name      Type    Description
  ========= ======= =============================================
  ``names`` str     (Optional) Comma-separated list of attribute names to filter
  ========= ======= =============================================

- **Response**: ``application/json`` — JSON object of metadata attributes.

.. code-block:: bash

    # Get all metadata
    curl http://localhost:7000/meta

    # Get specific attributes
    curl "http://localhost:7000/meta?names=worker_id,model_name"


Cache Operations
-----------------

``DELETE /cache/clear`` — Clear Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clear cached KV data from the LMCache engine.

- **Method**: ``DELETE``
- **Path**: ``/cache/clear``
- **Parameters**:

  =================== ========== =============================================
  Name                Type       Description
  =================== ========== =============================================
  ``locations``       list[str]  (Optional) Storage backends to clear (e.g. ``LocalCPUBackend``, ``LocalDiskBackend``). If not specified, clears all.
  =================== ========== =============================================

- **Response**: ``application/json``

.. code-block:: bash

    # Clear all cache
    curl -X DELETE http://localhost:7000/cache/clear

    # Clear specific backends
    curl -X DELETE "http://localhost:7000/cache/clear?locations=LocalCPUBackend&locations=LocalDiskBackend"

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "num_removed": 10
    }


``POST /cache/store`` — Store KV Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Store KV cache data into the LMCache engine using mock tokens.

- **Method**: ``POST``
- **Path**: ``/cache/store``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``tokens_mock``  str     Two comma-separated numbers: ``"start,end"`` (e.g. ``"0,100"`` generates tokens [0..99])
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST "http://localhost:7000/cache/store?tokens_mock=0,100"

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "num_tokens": 100
    }

**Error Response** (missing params, HTTP 400):

.. code-block:: json

    {
      "error": "Missing parameters",
      "message": "Must specify either tokens_input or tokens_mock"
    }


``POST /cache/retrieve`` — Retrieve KV Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retrieve KV cache data from the LMCache engine using mock tokens.

- **Method**: ``POST``
- **Path**: ``/cache/retrieve``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``tokens_mock``  str     Two comma-separated numbers: ``"start,end"`` (e.g. ``"0,100"``)
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST "http://localhost:7000/cache/retrieve?tokens_mock=0,100"

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "num_tokens": 100,
      "num_retrieved": 80
    }


``GET /cache/kvcache/check`` — KVCache Checksum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute MD5 checksums for kvcaches at specified slot_mapping positions.
Used for verifying that stored and retrieved kvcaches are identical.

- **Method**: ``GET``
- **Path**: ``/cache/kvcache/check``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``slot_mapping`` str     Slot indices, comma-separated. Supports ranges: ``"0,1,2,3"`` or ``"1,2,3,[9,12],17,19"``
  ``chunk_size``   int     Chunk size for computing per-chunk checksums (required)
  ``layerwise``    bool    If ``true``, output per-layer checksums per chunk (default: ``false``)
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    # Per-chunk checksum (all layers combined)
    curl "http://localhost:7000/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"

    # Per-layer per-chunk checksum
    curl "http://localhost:7000/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2&layerwise=true"

**Example Response** (``layerwise=false``):

.. code-block:: json

    {
      "status": "success",
      "slot_mapping_ranges": [[0, 3]],
      "chunk_size": 2,
      "num_chunks": 2,
      "chunk_checksums": ["abc123...", "def456..."],
      "layerwise": false
    }

**Example Response** (``layerwise=true``):

.. code-block:: json

    {
      "status": "success",
      "slot_mapping_ranges": [[0, 3]],
      "chunk_size": 2,
      "num_chunks": 2,
      "chunk_checksums": {
        "layer_0": ["abc123...", "def456..."],
        "layer_1": ["ghi789...", "jkl012..."]
      },
      "layerwise": true
    }


``POST /cache/kvcache/record_slot`` — Toggle Slot Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable or disable KVCache slot_mapping logging during store/retrieve operations.

- **Method**: ``POST``
- **Path**: ``/cache/kvcache/record_slot``
- **Parameters**:

  =========== ======= =============================================
  Name        Type    Description
  =========== ======= =============================================
  ``enabled`` str     ``"true"`` to enable, ``"false"`` to disable. Omit to query current status.
  =========== ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    # Enable logging
    curl -X POST "http://localhost:7000/cache/kvcache/record_slot?enabled=true"

    # Disable logging
    curl -X POST "http://localhost:7000/cache/kvcache/record_slot?enabled=false"

    # Check current status
    curl -X POST http://localhost:7000/cache/kvcache/record_slot

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "kvcache_check_log_enabled": true
    }


``GET /cache/kvcache/info`` — KVCache Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get information about the current kvcaches structure including layer names,
shapes, and device info.

- **Method**: ``GET``
- **Path**: ``/cache/kvcache/info``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/cache/kvcache/info

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "num_layers": 32,
      "layers": {
        "layer_0": {
          "shape": [2, 128, 16, 64, 128],
          "dtype": "torch.bfloat16",
          "device": "cuda:0"
        }
      }
    }


``POST /cache/load-fs-chunks`` — Load FS Chunks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load chunk files from FSConnector storage into LocalCPUBackend's hot cache.

- **Method**: ``POST``
- **Path**: ``/cache/load-fs-chunks``
- **Content-Type**: ``application/json``
- **Request Body**:

  ==================== ======= =============================================
  Field                Type    Description
  ==================== ======= =============================================
  ``config_path``      str     Path to LMCache engine configuration YAML file (required)
  ``max_chunks``       int     (Optional) Maximum number of chunks to load
  ``max_failed_keys``  int     Maximum failed keys to report (default: 10)
  ==================== ======= =============================================

- **Response**: ``application/json``
- **Tags**: ``cache-management``

.. code-block:: bash

    curl -X POST http://localhost:7000/cache/load-fs-chunks \
      -H "Content-Type: application/json" \
      -d '{"config_path": "/path/to/lmcache.yaml", "max_chunks": 100}'

**Example Response** (HTTP 200):

.. code-block:: json

    {
      "status": "success",
      "loaded_chunks": 95,
      "total_files": 100,
      "failed_keys": ["key1", "key2"],
      "config_path": "/path/to/lmcache.yaml"
    }

**Error Response** (invalid config, HTTP 400):

.. code-block:: json

    {
      "error": "Failed to load chunks from FSConnector",
      "message": "Configuration file not found",
      "config_path": "/path/to/lmcache.yaml"
    }


Freeze Mode
------------

``PUT /freeze/enable`` — Enable Freeze Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable freeze mode for the LMCache engine. When enabled:

- All store operations will be skipped (no new data stored)
- Only ``local_cpu`` backend will be used for retrieval
- No admit/evict messages will be generated

This protects the local_cpu hot cache from changes.

- **Method**: ``PUT``
- **Path**: ``/freeze/enable``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X PUT http://localhost:7000/freeze/enable

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "freeze": true,
      "message": "Freeze mode enabled successfully"
    }


``PUT /freeze/disable`` — Disable Freeze Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disable freeze mode. Store operations will proceed normally.

- **Method**: ``PUT``
- **Path**: ``/freeze/disable``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X PUT http://localhost:7000/freeze/disable

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "freeze": false,
      "message": "Freeze mode disabled successfully"
    }


``GET /freeze/status`` — Freeze Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the current freeze mode status.

- **Method**: ``GET``
- **Path**: ``/freeze/status``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/freeze/status

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "freeze": true,
      "message": "Freeze mode is enabled"
    }


Hot Cache
----------

These endpoints control the hot cache feature of LocalCPUBackend.
When hot cache is enabled, frequently accessed KV cache data will be kept
in CPU memory for faster retrieval.

``PUT /hot_cache/enable`` — Enable Hot Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable hot cache for the LocalCPUBackend.

- **Method**: ``PUT``
- **Path**: ``/hot_cache/enable``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X PUT http://localhost:7000/hot_cache/enable

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "hot_cache": true,
      "message": "Hot cache enabled successfully"
    }


``PUT /hot_cache/disable`` — Disable Hot Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disable hot cache for the LocalCPUBackend. Existing hot cache entries
will be cleared and no new data will be written.

- **Method**: ``PUT``
- **Path**: ``/hot_cache/disable``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X PUT http://localhost:7000/hot_cache/disable

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "hot_cache": false,
      "message": "Hot cache disabled successfully"
    }


``GET /hot_cache/status`` — Hot Cache Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the current hot cache status of LocalCPUBackend.

- **Method**: ``GET``
- **Path**: ``/hot_cache/status``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/hot_cache/status

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "hot_cache": true,
      "message": "Hot cache is enabled"
    }


Chunk Statistics
-----------------

These endpoints manage chunk-level statistics collection via
``ChunkStatisticsLookupClient``. They are only available when the
lookup client supports statistics.


Lookup Client/Server Management
-----------------------------------

These endpoints allow runtime management of the lookup client and server.
They are useful for dynamically reconfiguring the lookup mechanism without
restarting the service.

.. important::

    **Configuration Update Required First**

    Before calling ``/lookup/create`` or ``/lookup/recreate``, you **MUST**
    update the configuration via the ``/conf`` API first. The new lookup
    client/server will be created using ``LookupClientFactory``.

    For some configurations (e.g., switching ``enable_scheduler_bypass_lookup``),
    you only need to update the **scheduler's** configuration and recreate its
    lookup client. The workers don't need changes in this case.


``GET /lookup/info`` — Lookup Client/Server Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get information about the current lookup client and server status.
Shows wrapper chain if applicable (e.g., ``HitLimitLookupClient(LMCacheLookupClient)``).

- **Method**: ``GET``
- **Path**: ``/lookup/info``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:6999/lookup/info

**Example Response** (scheduler):

.. code-block:: json

    {
      "client": "HitLimitLookupClient(LMCacheBypassLookupClient)",
      "server": "None",
      "role": "scheduler"
    }

**Example Response** (worker):

.. code-block:: json

    {
      "client": "None",
      "server": "LMCacheLookupServer",
      "role": "worker"
    }


``POST /lookup/close`` — Close Lookup Client/Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Close the current lookup client (scheduler) or server (worker).

- **Method**: ``POST``
- **Path**: ``/lookup/close``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST http://localhost:6999/lookup/close

**Example Response**:

.. code-block:: json

    {
      "old": "HitLimitLookupClient(LMCacheBypassLookupClient)",
      "role": "scheduler"
    }


``POST /lookup/create`` — Create Lookup Client/Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new lookup client (scheduler) or server (worker) using current config.

- **Method**: ``POST``
- **Path**: ``/lookup/create``
- **Parameters**:

  =========== ======= =============================================
  Name        Type    Description
  =========== ======= =============================================
  ``dryrun``  bool    If ``true``, only show what would be created
  =========== ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    # Dryrun - preview what would be created
    curl -X POST "http://localhost:6999/lookup/create?dryrun=true"

    # Actually create
    curl -X POST http://localhost:6999/lookup/create

**Example Response** (dryrun):

.. code-block:: json

    {
      "new": "LMCacheLookupClient",
      "dryrun": true,
      "role": "scheduler"
    }

**Example Response** (actual create):

.. code-block:: json

    {
      "new": "LMCacheLookupClient",
      "role": "scheduler"
    }


``POST /lookup/recreate`` — Recreate Lookup Client/Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recreate the lookup client or server (equivalent to close + create).
The endpoint automatically determines which component based on role:

- **scheduler** role: recreates lookup client
- **worker** role: recreates lookup server

- **Method**: ``POST``
- **Path**: ``/lookup/recreate``
- **Parameters**: None
- **Response**: ``application/json``

**Usage Flow**:

.. code-block:: bash

    # Step 1: Update worker configuration (if needed)
    curl -X POST "http://localhost:7000/conf" \
      -H "Content-Type: application/json" \
      -d '{"enable_async_loading": true}'

    # Step 2: Recreate lookup server on worker
    curl -X POST "http://localhost:7000/lookup/recreate"

    # Step 3: Update scheduler configuration
    curl -X POST "http://localhost:6999/conf" \
      -H "Content-Type: application/json" \
      -d '{"enable_scheduler_bypass_lookup": true}'

    # Step 4: Recreate lookup client on scheduler
    curl -X POST "http://localhost:6999/lookup/recreate"

**Example Response** (scheduler):

.. code-block:: json

    {
      "old": "HitLimitLookupClient(LMCacheBypassLookupClient)",
      "new": "LMCacheLookupClient",
      "role": "scheduler"
    }

**Example Response** (worker):

.. code-block:: json

    {
      "old": "LMCacheLookupServer",
      "new": "LMCacheAsyncLookupServer",
      "role": "worker"
    }

.. note::

    **Client-only Changes**

    For some configuration changes (e.g., switching ``enable_scheduler_bypass_lookup``),
    you only need to update the scheduler's configuration and recreate its lookup
    client. Worker-side lookup servers don't need to be recreated in this case.


``POST /chunk_statistics/start`` — Start Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start collecting chunk statistics.

- **Method**: ``POST``
- **Path**: ``/chunk_statistics/start``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST http://localhost:7000/chunk_statistics/start

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "message": "Started"
    }

**Error Response** (not supported, HTTP 400):

.. code-block:: json

    {
      "error": "Not available",
      "message": "Client does not support statistics."
    }


``POST /chunk_statistics/stop`` — Stop Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stop collecting chunk statistics.

- **Method**: ``POST``
- **Path**: ``/chunk_statistics/stop``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST http://localhost:7000/chunk_statistics/stop

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "message": "Stopped"
    }


``POST /chunk_statistics/reset`` — Reset Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reset all collected chunk statistics.

- **Method**: ``POST``
- **Path**: ``/chunk_statistics/reset``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl -X POST http://localhost:7000/chunk_statistics/reset

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "message": "Reset"
    }


``GET /chunk_statistics/status`` — Statistics Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get current chunk statistics and auto-exit configuration.

- **Method**: ``GET``
- **Path**: ``/chunk_statistics/status``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/chunk_statistics/status

**Example Response**:

.. code-block:: json

    {
      "is_collecting": true,
      "total_chunks": 1000,
      "unique_chunks": 500,
      "timestamp": 1706745600.0,
      "auto_exit_enabled": true,
      "auto_exit_timeout_hours": 24.0,
      "auto_exit_target_unique_chunks": 1000
    }


.. _bypass_mode:

Bypass Mode
------------

Bypass mode allows dynamically skipping specific storage backends at runtime.
Bypassed backends are excluded from ``contains``/``put``/``get`` operations.
This is useful for fault injection testing, isolating a problematic backend,
or debugging without restarting the engine.


``GET /bypass/list`` — List Bypassed Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List all currently bypassed backends and all available backend names.

- **Method**: ``GET``
- **Path**: ``/bypass/list``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:7000/bypass/list

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "bypassed_backends": ["RemoteBackend"],
      "all_backends": ["LocalCPUBackend", "RemoteBackend"]
    }


``PUT /bypass/add`` — Add a Backend to Bypass List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a backend to the bypass list. The bypassed backend will be excluded
from ``contains``/``put``/``get`` operations.

- **Method**: ``PUT``
- **Path**: ``/bypass/add``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``backend_name`` str     Name of the backend to bypass (required)
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    curl -X PUT "http://localhost:7000/bypass/add?backend_name=RemoteBackend"

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "backend_name": "RemoteBackend",
      "bypassed": true,
      "was_already_bypassed": false,
      "bypassed_backends": ["RemoteBackend"]
    }

**Error Response** (unknown backend, HTTP 400):

.. code-block:: json

    {
      "error": "Unknown backend",
      "message": "Backend 'FooBackend' not found. Available: ['LocalCPUBackend', 'RemoteBackend']"
    }


``PUT /bypass/remove`` — Remove a Backend from Bypass List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove a backend from the bypass list, restoring it to normal operation.

- **Method**: ``PUT``
- **Path**: ``/bypass/remove``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``backend_name`` str     Name of the backend to restore (required)
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    curl -X PUT "http://localhost:7000/bypass/remove?backend_name=RemoteBackend"

**Example Response**:

.. code-block:: json

    {
      "status": "success",
      "backend_name": "RemoteBackend",
      "bypassed": false,
      "was_bypassed": true,
      "bypassed_backends": []
    }

**Error Response** (unknown backend, HTTP 400):

.. code-block:: json

    {
      "error": "Unknown backend",
      "message": "Backend 'FooBackend' not found. Available: ['LocalCPUBackend', 'RemoteBackend']"
    }
