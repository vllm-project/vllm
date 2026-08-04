Mooncake
========

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/mooncake_store`.


.. _mooncake-overview:

Overview
--------

`Mooncake <https://github.com/kvcache-ai/Mooncake>`_ is an open-source distributed KV cache storage system designed specifically for LLM inference scenarios. 
The system creates a distributed memory pool by aggregating memory space contributed by various client nodes, enabling efficient resource utilization across clusters.

By pooling underutilized DRAM and SSD resources from multiple nodes, the system forms a unified distributed storage service that maximizes resource efficiency.

.. image:: ../../assets/mooncake-store-preview.png
    :alt: Mooncake Architecture Diagram

Key Features
~~~~~~~~~~~~

- **Distributed memory pooling**: Aggregates memory contributions from multiple client nodes into a unified storage pool
- **High bandwidth utilization**: Supports striping and parallel I/O transfer of large objects, fully utilizing multi-NIC aggregated bandwidth
- **RDMA optimization**: Built on Transfer Engine with support for TCP, RDMA (InfiniBand/RoCEv2/eRDMA/NVIDIA GPUDirect)
- **Dynamic resource scaling**: Supports dynamically adding and removing nodes for elastic resource management

For detailed architecture information, see the `Mooncake Architecture Guide <https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html>`_.

Quick Start
-----------

Install Mooncake via pip:

.. code-block:: bash

    pip install mooncake-transfer-engine

This package includes all necessary components:

- ``mooncake_master``: Master service that manages cluster metadata and coordinates distributed storage operations
- ``mooncake_http_metadata_server``: HTTP-based metadata server used by the underlying transfer engine for connection establishment
- Mooncake Python bindings

For production deployments or custom builds, see the `Build Instructions <https://kvcache-ai.github.io/Mooncake/getting_started/build.html>`_.

Setup and Deployment
~~~~~~~~~~~~~~~~~~~~

**Prerequisites:**

- Machine with at least one GPU for vLLM inference
- RDMA-capable network hardware and drivers (recommended) or TCP network
- Python 3.8+ with pip
- vLLM and LMCache installed

**Step 1: Start Infrastructure Services**

Start the Mooncake master service (with built‑in HTTP metadata server):

.. code-block:: bash

    # Master service (use -v=1 for verbose logging)
    # The flag enables the integrated HTTP metadata server
    mooncake_master --enable_http_metadata_server=1

Expected output:

.. code-block:: text

    Master service started on port 50051
    HTTP metrics server started on port 9003
    Master Metrics: Storage: 0.00 B / 0.00 B | Keys: 0 | ...

**Step 2: Create Configuration File**

Create your ``mooncake-config.yaml``:

.. code-block:: yaml

    # LMCache Configuration
    local_cpu: False
    remote_url: "mooncakestore://localhost:50051/"
    max_local_cpu_size: 2  # small local buffer
    numa_mode: "auto"      # reduce tail latency with multi-NUMA/multi-NIC
    pre_caching_hash_algorithm: sha256_cbor_64bit

    # Mooncake Configuration (via extra_config)
    extra_config:
      use_exists_sync: true
      save_chunk_meta: False  # Enable chunk metadata optimization
      local_hostname: "localhost"
      metadata_server: "http://localhost:8080/metadata"
      protocol: "rdma"
      device_name: ""        # leave empty; autodetect device(s)
      global_segment_size: 21474836480   # 20 GiB per worker
      master_server_address: "localhost:50051"
      local_buffer_size: 0    # rely on LMCache local_cpu as the buffer
      mooncake_prefer_local_alloc: true  # prefer local segment if available

**Step 3: Start vLLM with Mooncake**

.. code-block:: bash

    # If you see persistent misses (no Mooncake hits), make sure
    # PYTHONHASHSEED is fixed across processes (e.g., export PYTHONHASHSEED=0).
    LMCACHE_CONFIG_FILE="mooncake-config.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 65536 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 4: Verify the Setup**

Test the integration with a sample request:

.. code-block:: bash

    curl -X POST "http://localhost:8000/v1/completions" \
         -H "Content-Type: application/json" \
         -d '{
           "model": "meta-llama/Llama-3.1-8B-Instruct",
           "prompt": "The future of AI is",
           "max_tokens": 100,
           "temperature": 0.7
         }'

**Debugging Tips:**

1. **Enable verbose logging:**

   .. code-block:: bash

       mooncake_master -v=1

2. **Check service status:**

   .. code-block:: bash

       # Check if services are running
       ps aux | grep mooncake
       netstat -tlnp | grep -E "(8080|50051)"

3. **Monitor metrics:**

   Access metrics at ``http://localhost:9003`` when master service is running.

Configuration
-------------

**LMCache Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``chunk_size``
     - 256
     - Number of tokens per KV chunk
   * - ``remote_url``
     - Required
     - Mooncake store connection URL (format: ``mooncakestore://host:port/``).
   * - ``remote_serde``
     - "naive"
     - Serialization method for remote storage
   * - ``local_cpu``
     - False
     - Enable/disable local CPU caching (set to False for pure Mooncake evaluation)
   * - ``max_local_cpu_size``
     - Required
     - Maximum local CPU cache size in GB (required even when local_cpu is False)
   * - ``numa_mode``
     - "auto"
     - NUMA binding mode. "auto" is recommended on multi‑NIC/multi‑NUMA systems to reduce tail latency.
   * - ``pre_caching_hash_algorithm``
     - "sha256_cbor_64bit"
     - Hash used for pre-caching keying. For cross‑process consistency, fix ``PYTHONHASHSEED`` (e.g., export ``PYTHONHASHSEED=0``).

**Mooncake Parameters (via extra_config):**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``local_hostname``
     - Required
     - Hostname/IP of the local node for Mooncake client identification
   * - ``metadata_server``
     - Required
     - HTTP metadata server address. When starting master with ``--enable_http_metadata_server=1``, it exposes this endpoint.
   * - ``master_server_address``
     - Required
     - Mooncake master service address (host:port format)
   * - ``protocol``
     - "rdma"
     - Communication protocol ("rdma" for high performance; "tcp" for compatibility)
   * - ``device_name``
     - ""
     - RDMA device specification (e.g., "erdma_0,erdma_1" or "mlx5_0,mlx5_1"). Leave empty for autodetection in most setups.
   * - ``global_segment_size``
     - 21474836480
     - **Memory size contributed by each vLLM worker** in bytes (e.g., 20 GiB recommended)
   * - ``local_buffer_size``
     - 0
     - Local buffer size in bytes used by Mooncake. Behavior depends on ``save_chunk_meta``:
       - When ``save_chunk_meta: False`` (recommended), LMCache uses its local CPU backend for zero‑copy RDMA, so Mooncake's ``local_buffer_size`` can be ``0``.
       - When ``save_chunk_meta: True``, Mooncake uses its own local buffer; set this to a proper value (e.g., several GiB).
       - Note: Some RDMA NICs have memory registration limits; registering LMCache's large CPU buffer can fail on constrained devices. In those cases, consider enabling ``save_chunk_meta: True`` and sizing ``local_buffer_size`` instead.
   * - ``transfer_timeout``
     - 1
     - Timeout for transfer operations in seconds
   * - ``storage_root_dir``
     - ""
     - The root directory for persistence (e.g., "/mnt/mooncake")
   * - ``save_chunk_meta``
     - False
     - Whether to save chunk metadata alongside data. Set to ``False`` to enable the optimized zero‑copy path in LMCache.
   * - ``use_exists_sync``
     - False
     - Use synchronous existence checks to avoid async scheduling overhead in hot paths.
   * - ``mooncake_prefer_local_alloc``
     - False
     - Prefer allocating on the local segment when possible.

.. important::
   **Understanding global_segment_size**: This parameter defines the amount of memory each vLLM worker contributes to the distributed memory pool. 
   The total cluster memory available for KV cache storage will be: ``number_of_vllm_workers × global_segment_size``.
   
   Adjust this value based on your available system memory and expected cache requirements.

.. tip::
   If you consistently get misses (no Mooncake hits), ensure all processes use the same hashing seed: ``export PYTHONHASHSEED=0``. This keeps pre‑caching keys consistent across processes.

.. note::
   RDMA device(s) usually do not need to be specified; leaving ``device_name`` empty works for most deployments.

Additional Resources
--------------------

- `Mooncake Store Architecture <https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html>`_
- `Mooncake Store Deployment Guide <https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html>`_
- `Mooncake Store Python API Reference <https://kvcache-ai.github.io/Mooncake/python-api-reference/mooncake-store.html>`_
- `Transfer Engine Documentation <https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html>`_
- `Build Instructions <https://kvcache-ai.github.io/Mooncake/getting_started/build.html>`_
- `GitHub Repository <https://github.com/kvcache-ai/Mooncake>`_
- `LMCache Integration Guide <https://kvcache-ai.github.io/Mooncake/getting_started/examples/lmcache-integration.html>`_
