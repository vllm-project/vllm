Bigtable
========

An L2 adapter backed by Google Cloud Bigtable, enabling out-of-process remote KV caching. It supports Layer-Group Sharding to partition large tensor payloads across multiple column qualifiers to stay safely under Bigtable cell size limits.

Why Choose Bigtable?
--------------------

Cloud Bigtable is an excellent choice as an L2 storage backend for enterprise-scale LLM serving workloads:

- **Sub-millisecond/Single-digit Latency**: SSD-backed Bigtable clusters deliver the low latency required for real-time prompt caching, preventing remote cache retrieval from bottlenecking LLM generation.
- **High Concurrency and Throughput**: Scales horizontally to handle thousands of concurrent queries from multiple vLLM instances, shielding the L1 memory from cache misses.
- **Enterprise-Grade Managed Service**: Fully managed by Google Cloud, providing built-in multi-region replication, IAM access controls, and zero-downtime scaling.
- **Tier-Agnostic Storage Flexibility**: Decouples LMCache config from database deployments. Choose the **SSD Tier** for primary low-latency caching, or the **HDD Tier** for cost-effective, large-scale archival storage.

**Required fields:**

- ``bigtable_project_id`` (str): Your Google Cloud Project ID.
- ``bigtable_instance_id`` (str): Your Cloud Bigtable Instance ID.
- ``bigtable_table_name`` (str): Bigtable table name.

**Optional fields:**

- ``bigtable_family_name`` (str, default ``"cf"``): Column family name. Must be pre-created in the Bigtable table.
- ``bigtable_column_name`` (str, default ``"data"``): Column qualifier name to use when sharding is disabled.
- ``layer_group_size`` (int, default ``10``): Number of layers per column group. Set to ``0`` to disable Layer-Group Sharding.
- ``bigtable_max_chunk_size_mb`` (float, default ``90.0``): The maximum allowed write limit when sharding is disabled (or when using CacheGen). Writes exceeding this are safely skipped to avoid database exceptions.
- ``credentials_path`` (str, default ``None``): Absolute path to a GCP Service Account JSON key file. If omitted, it defaults to Application Default Credentials (ADC).
- ``exists_cache_ttl_seconds`` (float, default ``10.0``): TTL in seconds for the internal cache that shields Bigtable from redundant lookup requests.
- ``bigtable_write_timeout_ms`` (float, default ``10000.0``): Maximum timeout for database writes.
- ``bigtable_read_timeout_ms`` (float, default ``5000.0``): Maximum timeout for database reads.

**Environment variable fallbacks.** When the corresponding config value is empty, these environment variables are used: ``BT_PROJECT_ID``, ``BT_INSTANCE_ID``, ``BT_TABLE_NAME``.

Tutorials & Configuration Examples
----------------------------------

Tutorial 1: Serving Massive Models (FP8 Compression)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are serving extremely large models where individual KV cache chunks are massive, you can enable FP8 quantization compression. This compresses the KV cache footprint by **2x** and reduces network transfer overhead to Bigtable.

Add the ``"serde"`` sub-dictionary inside the Bigtable L2 adapter JSON spec:

.. code-block:: bash

    lmcache server \
        --port 65432 \
        --http-port 8080 \
        --l2-adapter '{
            "type": "bigtable",
            "bigtable_project_id": "your-gcp-project-id",
            "bigtable_instance_id": "lmcache-bt-instance",
            "bigtable_table_name": "lmcache-kv-table",
            "layer_group_size": 10,
            "serde": {
                "type": "fp8",
                "fp8_dtype": "float8_e4m3fn"
            }
        }' \
        --l1-size-gb 10.0 \
        --eviction-policy LRU

Tutorial 2: 3-Tier Hybrid Storage (Local CPU -> Redis L2 -> Bigtable SSD L3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can place a Redis/Valkey instance in front of Bigtable to serve as a fast hot cache, cascading requests down to Bigtable on misses. 

Configure multiple L2 adapters sequentially using multiple ``--l2-adapter`` arguments:

.. code-block:: bash

    lmcache server \
        --port 65432 \
        --http-port 8080 \
        --l2-adapter '{"type": "resp", "host": "your-redis-host", "port": 6379}' \
        --l2-adapter '{
            "type": "bigtable",
            "bigtable_project_id": "your-gcp-project-id",
            "bigtable_instance_id": "lmcache-bt-instance",
            "bigtable_table_name": "lmcache-kv-table",
            "layer_group_size": 10
        }' \
        --l1-size-gb 10.0 \
        --eviction-policy LRU

Tutorial 3: Multi-Tier Storage Cascading (SSD Hot Tier -> HDD Archive)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To balance low-latency performance for hot cache hits and cost-effectiveness for long-term archival storage, you can configure tiered storage (SSD to HDD) in two ways:

Method A: Native Bigtable Tiered Storage (Recommended - Zero Configuration)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Google Cloud Bigtable supports native **Tiered Storage**. By enabling an age-based tiering policy on your table via the Google Cloud Console, gcloud CLI, or Terraform, Bigtable automatically moves older/colder cells to a cost-effective infrequent-access storage tier in the background.

* **No LMCache configuration changes**: You only need to configure a single Bigtable L2 adapter in LMCache.
* **No code changes**: Bigtable handles routing reads and writes to the correct tier transparently.
* **Natural aging**: Since cells are written with the default server timestamp, they will naturally age out based on the configured policy window.

For more information, see the official `Bigtable Tiered Storage Overview <https://cloud.google.com/bigtable/docs/tiered-storage>`_.

Method B: Manual LMCache Storage Cascade (Alternative)
""""""""""""""""""""""""""""""""""""""""""""""""""""""

If you prefer to manage separate physical Bigtable instances or tables (e.g. one SSD and one HDD), you can cascade them directly in LMCache by configuring multiple L2 adapters sequentially in your startup command:

.. code-block:: bash

    lmcache server \
        --port 65432 \
        --http-port 8080 \
        --l2-adapter '{
            "type": "bigtable",
            "bigtable_project_id": "your-gcp-project-id",
            "bigtable_instance_id": "lmcache-bt-ssd-instance",
            "bigtable_table_name": "lmcache-kv-hot-table",
            "layer_group_size": 10
        }' \
        --l2-adapter '{
            "type": "bigtable",
            "bigtable_project_id": "your-gcp-project-id",
            "bigtable_instance_id": "lmcache-bt-hdd-instance",
            "bigtable_table_name": "lmcache-kv-archive-table",
            "layer_group_size": 10
        }' \
        --l1-size-gb 10.0 \
        --eviction-policy LRU

Configuration Examples: Custom Timeouts and Explicit Service Account
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can configure custom database timeout settings and pass an explicit Service Account key file path in the JSON payload:

.. code-block:: bash

    lmcache server \
        --port 65432 \
        --http-port 8080 \
        --l2-adapter '{
            "type": "bigtable", 
            "bigtable_project_id": "your-gcp-project-id", 
            "bigtable_instance_id": "lmcache-bt-instance", 
            "bigtable_table_name": "lmcache-kv-table", 
            "credentials_path": "/secrets/gcp/sa_key.json", 
            "bigtable_write_timeout_ms": 5000.0, 
            "bigtable_read_timeout_ms": 2000.0
        }' \
        --l1-size-gb 10.0 \
        --eviction-policy LRU
