Google Cloud Bigtable Remote Cache
==================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _bigtable-overview:

Overview: Why Choose Bigtable?
------------------------------

Cloud Bigtable is ideal for LLM serving workloads that require massive scale without sacrificing performance. It offers:

- **Massive Scalability**: Seamlessly scales to handle petabytes of KV cache data with consistent, single-digit millisecond latency.
- **Enterprise Reliability**: Fully managed with built-in replication, zero-downtime scaling, and robust IAM security.
- **Flexible Storage Tiers**: Choose the right balance of cost and performance for your deployment:
  - **SSD Tier (Recommended)**: Optimized for low-latency, high-throughput caching. Recommended for primary L2 cache setups where retrieval speed is critical.
  - **HDD Tier**: Best for cost-effective, massive-scale archival storage where capacity is the priority over sub-millisecond latency.

---

Quickstart: Lossless Raw Storage
--------------------------------

This tutorial walks you through setting up LMCache with a persistent Cloud Bigtable SSD tier using raw FP16 values (lossless storage).

**Step 1: Enable GCP Bigtable APIs**

Run the following command to enable the necessary Google Cloud services:

.. code-block:: bash

   gcloud services enable bigtable.googleapis.com bigtableadmin.googleapis.com --project=your-gcp-project-id

**Step 2: Provision a Bigtable Instance**

Create a single-node Bigtable SSD instance in your preferred zone:

.. code-block:: bash

   gcloud beta bigtable instances create lmcache-bt-instance \
       --display-name="LMCache SSD Instance" \
       --edition=ENTERPRISE \
       --cluster-storage-type=ssd \
       --cluster-config=id=lmcache-cluster,zone=us-central1-a,nodes=1 \
       --project=your-gcp-project-id

**Step 3: Create the Database Table**

Create a table and provision a column family named ``cf``:

.. code-block:: bash

   gcloud bigtable instances tables create lmcache-kv-table \
       --instance=lmcache-bt-instance \
       --column-families=cf \
       --project=your-gcp-project-id

**Step 4: Install LMCache and Bigtable SDK**

Install LMCache and the Google Cloud Bigtable client library on your serving machine:

.. code-block:: bash

   export NO_NATIVE_EXT=1
   pip install --no-cache-dir lmcache google-cloud-bigtable cachetools

**Step 5: Configure LMCache**

Create a configuration YAML file (e.g., ``lmcache_config.yaml``) with the following setup:

.. code-block:: yaml

   # Setup lossless, sharded Bigtable cache
   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 10.0
   remote_url: "bigtable://your-gcp-project-id/lmcache-bt-instance"
   remote_serde: "naive"

   extra_config:
     bigtable_project_id: "your-gcp-project-id"
     bigtable_instance_id: "lmcache-bt-instance"
     bigtable_table_name: "lmcache-kv-table"
     bigtable_family_name: "cf"
     bigtable_layer_group_size: 10  # Splits KV chunks across columns to stay under cell size limits

**Step 6: Start serving with vLLM**

Launch your vLLM engine pointing to the LMCache configuration:

.. code-block:: bash

   LMCACHE_CONFIG_FILE=lmcache_config.yaml vllm serve facebook/opt-6.7b

---

Advanced Configuration Tutorials
--------------------------------

Tutorial 1: Serving Massive Models (Quantized Compression)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are serving extremely large models (e.g., Llama-3.1-405B or long-context 70B models) where raw KV cache chunks exceed 240 MB, you should enable LMCache's native **CacheGen** compression to quantize payloads before storing them in Bigtable.

Create an ``lmcache_config_compressed.yaml`` file:

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 10.0
   remote_url: "bigtable://your-gcp-project-id/lmcache-bt-instance"
   
   # Enable CacheGen quantization compression
   remote_serde: "cachegen"
   
   extra_config:
     bigtable_project_id: "your-gcp-project-id"
     bigtable_instance_id: "lmcache-bt-instance"
     bigtable_table_name: "lmcache-kv-table"
     bigtable_family_name: "cf"
     bigtable_layer_group_size: 0  # Disable sharding (CacheGen output fits easily in a single cell)

*   **Benefit**: Compresses the KV cache footprint by **10x to 20x** (reducing a 128MB payload to ~8MB), drastically lowering Bigtable network transfer latency and storage costs.

Tutorial 2: 3-Tier Hybrid Storage (Local CPU -> Redis -> Bigtable SSD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For low-latency hot caching, you can place a Redis instance in front of Bigtable. Hot requests are served from Redis, while long-tail persistent caches are offloaded to Bigtable.

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 15.0

   # Define the caching pipeline order
   remote_storage_plugins:
     - "redis"
     - "bigtable"

   extra_config:
     # Redis L2 Configuration
     remote_storage_plugin.redis.redis_url: "redis://your-redis-host:6379"
     
     # Bigtable L3 Configuration
     remote_storage_plugin.bigtable.bigtable_project_id: "your-gcp-project-id"
     remote_storage_plugin.bigtable.bigtable_instance_id: "lmcache-bt-instance"
     remote_storage_plugin.bigtable.bigtable_table_name: "lmcache-kv-table"
     remote_storage_plugin.bigtable.bigtable_family_name: "cf"
     remote_storage_plugin.bigtable.bigtable_layer_group_size: 10

---

Configuration Reference
-----------------------

Configure the following options inside the ``extra_config`` dictionary in your configuration file:

.. list-table:: Bigtable Configuration Parameters
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter Key
     - Type
     - Default
     - Description
   * - ``bigtable_project_id``
     - string
     - None
     - Your Google Cloud Project ID.
   * - ``bigtable_instance_id``
     - string
     - None
     - Your Cloud Bigtable Instance ID.
   * - ``bigtable_table_name``
     - string
     - None
     - Bigtable table name.
   * - ``bigtable_family_name``
     - string
     - ``cf``
     - Bigtable column family name.
   * - ``bigtable_layer_group_size``
     - integer
     - ``10``
     - Number of layers per column group. Set to ``0`` to disable Layer-Group Sharding.
   * - ``bigtable_max_chunk_size_mb``
     - float
     - ``90.0``
     - The maximum allowed write limit when sharding is disabled. Writes exceeding this are safely skipped.
   * - ``credentials_path``
     - string
     - None
     - Absolute path to a GCP Service Account JSON key file. If omitted, LMCache defaults to Application Default Credentials (ADC).
   * - ``exists_cache_ttl_seconds``
     - float
     - ``10.0``
     - TTL for shielding lookups on Bigtable nodes.
   * - ``bigtable_write_timeout_ms``
     - float
     - ``10000.0``
     - Maximum timeout for database writes.
   * - ``bigtable_read_timeout_ms``
     - float
     - ``5000.0``
     - Maximum timeout for database reads.

---

Troubleshooting
---------------

*   **Authentication Failures**: If ``credentials_path`` is omitted, ensure your environment is authenticated via ``gcloud auth application-default login`` or configured with GKE Workload Identity Federation.
*   **Writes are skipped / "Bigtable chunk size exceeds threshold"**:
    *   If using uncompressed storage (``remote_serde: "naive"``), verify that ``bigtable_layer_group_size`` is set to ``10`` to enable Layer-Group Sharding (allows writes up to 240MB).
    *   Alternatively, enable compression by setting ``remote_serde: "cachegen"`` and set ``bigtable_layer_group_size: 0``.
*   **Redundant RPC Warnings**: If you see a warning about ``use_layerwise`` running with sharding, disable it by setting ``use_layerwise: false`` in your YAML configuration. Layer-Group Sharding optimizes reads into a single network roundtrip, whereas ``use_layerwise: true`` forces 32+ sequential network calls.
