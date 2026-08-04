.. _observability_metrics:

Metrics Reference
=================

LMCache provides comprehensive metrics via Prometheus to help you monitor performance, cache efficiency, and system health. These metrics are exposed via the vLLM ``/metrics`` endpoint when LMCache is integrated with vLLM, or via the LMCache internal API server.

Available Metrics
-----------------

The following tables list all available LMCache metrics organized by category.

Core Request Metrics
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Core Request Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:num_retrieve_requests``
     - Counter
     - Total number of retrieve requests sent to LMCache.
   * - ``lmcache:num_store_requests``
     - Counter
     - Total number of store requests sent to LMCache.
   * - ``lmcache:num_lookup_requests``
     - Counter
     - Total number of lookup requests sent to LMCache.

Token Metrics
~~~~~~~~~~~~~

.. list-table:: Token Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:num_requested_tokens``
     - Counter
     - Total number of tokens requested for retrieval.
   * - ``lmcache:num_hit_tokens``
     - Counter
     - Total number of tokens hit in LMCache during retrieval.
   * - ``lmcache:num_stored_tokens``
     - Counter
     - Total number of tokens stored in LMCache.
   * - ``lmcache:num_lookup_tokens``
     - Counter
     - Total number of tokens requested in lookup operations.
   * - ``lmcache:num_lookup_hits``
     - Counter
     - Total number of tokens hit in lookup operations.
   * - ``lmcache:num_vllm_hit_tokens``
     - Counter
     - Number of hit tokens in vLLM.
   * - ``lmcache:num_prompt_tokens``
     - Counter
     - Number of prompt tokens in LMCache.

Hit Rate Metrics
~~~~~~~~~~~~~~~~

.. list-table:: Hit Rate Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:retrieve_hit_rate``
     - Gauge
     - The hit rate for retrieve requests since last log.
   * - ``lmcache:lookup_hit_rate``
     - Gauge
     - The hit rate for lookup requests since last log.
   * - ``lmcache:request_cache_hit_rate``
     - Histogram
     - Distribution of hit rates per request.
   * - ``lmcache:lookup_0_hit_requests``
     - Counter
     - Total number of lookup requests with zero hits.

Performance & Latency Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance & Latency Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:time_to_retrieve``
     - Histogram
     - Time taken to retrieve from the cache (seconds).
   * - ``lmcache:time_to_store``
     - Histogram
     - Time taken to store to the cache (seconds).
   * - ``lmcache:time_to_lookup``
     - Histogram
     - Time taken to perform a lookup in the cache (seconds).
   * - ``lmcache:retrieve_speed``
     - Histogram
     - Retrieval speed (tokens per second).
   * - ``lmcache:store_speed``
     - Histogram
     - Storage speed (tokens per second).
   * - ``lmcache:num_slow_retrieval_by_time``
     - Counter
     - Total number of slow retrievals exceeding the time threshold.
   * - ``lmcache:num_slow_retrieval_by_speed``
     - Counter
     - Total number of slow retrievals below the speed threshold.

Detailed Profiling Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Profiling Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:retrieve_process_tokens_time``
     - Histogram
     - Time to process tokens in retrieve (seconds).
   * - ``lmcache:retrieve_broadcast_time``
     - Histogram
     - Time to broadcast memory objects in retrieve (seconds).
   * - ``lmcache:retrieve_to_gpu_time``
     - Histogram
     - Time to move data to GPU in retrieve (seconds).
   * - ``lmcache:store_process_tokens_time``
     - Histogram
     - Time to process tokens in store (seconds).
   * - ``lmcache:store_from_gpu_time``
     - Histogram
     - Time to move data from GPU in store (seconds).
   * - ``lmcache:store_put_time``
     - Histogram
     - Time to put data to storage in store (seconds).
   * - ``lmcache:remote_backend_batched_get_blocking_time``
     - Histogram
     - Time spent waiting for data from remote backend (seconds).
   * - ``lmcache:instrumented_connector_batched_get_time``
     - Histogram
     - Time spent in the connector layer (seconds).

Cache Usage & Lifecycle Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Cache Usage Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:local_cache_usage``
     - Gauge
     - Local cache usage in bytes.
   * - ``lmcache:remote_cache_usage``
     - Gauge
     - Remote cache usage in bytes.
   * - ``lmcache:local_storage_usage``
     - Gauge
     - Local storage usage in bytes.
   * - ``lmcache:request_cache_lifespan``
     - Histogram
     - Distribution of request cache lifespan in minutes.

Remote Backend & Network Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Remote Backend Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:num_remote_read_requests``
     - Counter
     - Total number of read requests to remote backends.
   * - ``lmcache:num_remote_read_bytes``
     - Counter
     - Total number of bytes read from remote backends.
   * - ``lmcache:num_remote_write_requests``
     - Counter
     - Total number of write requests to remote backends.
   * - ``lmcache:num_remote_write_bytes``
     - Counter
     - Total number of bytes written to remote backends.
   * - ``lmcache:remote_time_to_get``
     - Histogram
     - Time taken to get data from remote backends (ms).
   * - ``lmcache:remote_time_to_put``
     - Histogram
     - Time taken to put data to remote backends (ms).
   * - ``lmcache:remote_time_to_get_sync``
     - Histogram
     - Time taken to get data from remote backends synchronously (ms).
   * - ``lmcache:remote_ping_latency``
     - Gauge
     - Latest ping latency to remote backends (ms).
   * - ``lmcache:remote_ping_errors``
     - Counter
     - Total number of ping errors to remote backends.
   * - ``lmcache:remote_ping_successes``
     - Counter
     - Total number of successful pings to remote backends.
   * - ``lmcache:remote_ping_error_code``
     - Gauge
     - Latest ping error code to remote backends.

Local CPU Backend Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Local CPU Backend Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:local_cpu_evict_count``
     - Counter
     - Total number of evictions in local CPU backend.
   * - ``lmcache:local_cpu_evict_keys_count``
     - Counter
     - Total number of evicted keys in local CPU backend.
   * - ``lmcache:local_cpu_evict_failed_count``
     - Counter
     - Total number of failed evictions in local CPU backend.
   * - ``lmcache:local_cpu_hot_cache_count``
     - Gauge
     - Current number of items in the hot cache.
   * - ``lmcache:local_cpu_keys_in_request_count``
     - Gauge
     - Current number of keys being processed in requests.

Memory Management Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Memory Management Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:active_memory_objs_count``
     - Gauge
     - The number of currently active memory objects.
   * - ``lmcache:pinned_memory_objs_count``
     - Gauge
     - The number of currently pinned memory objects.
   * - ``lmcache:forced_unpin_count``
     - Counter
     - Total number of forced unpins due to timeout.
   * - ``lmcache:pin_monitor_pinned_objects_count``
     - Gauge
     - The number of pinned objects tracked by the PinMonitor.

P2P Transfer Metrics
~~~~~~~~~~~~~~~~~~~~

.. list-table:: P2P Transfer Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:num_p2p_requests``
     - Counter
     - Total number of P2P transfer requests.
   * - ``lmcache:num_p2p_transferred_tokens``
     - Counter
     - Total number of tokens transferred via P2P.
   * - ``lmcache:p2p_time_to_transfer``
     - Histogram
     - Time taken for P2P transfers (seconds).
   * - ``lmcache:p2p_transfer_speed``
     - Histogram
     - P2P transfer speed (tokens per second).

Health & Internal System Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Health & Internal Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:lmcache_is_healthy``
     - Gauge
     - Overall health status of LMCache (1 = healthy, 0 = unhealthy).
   * - ``lmcache:interval_get_blocking_failed_count``
     - Gauge
     - Number of failed blocking get operations in the current interval.
   * - ``lmcache:kv_msg_queue_size``
     - Gauge
     - Size of the KV message queue in the BatchedMessageSender.
   * - ``lmcache:remote_put_task_num``
     - Gauge
     - Number of pending remote put tasks.
   * - ``lmcache:storage_events_ongoing_count``
     - Gauge
     - Number of storage events currently in progress.
   * - ``lmcache:storage_events_done_count``
     - Gauge
     - Number of storage events completed successfully.
   * - ``lmcache:storage_events_not_found_count``
     - Gauge
     - Number of storage events where the requested data was not found.

Chunk Statistics Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Chunk Statistics Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:chunk_statistics_enabled``
     - Gauge
     - Whether chunk statistics collection is enabled (1 = enabled, 0 = disabled).
   * - ``lmcache:chunk_statistics_total_requests``
     - Gauge
     - Total number of requests processed by chunk statistics.
   * - ``lmcache:chunk_statistics_total_chunks``
     - Gauge
     - Total number of chunks processed.
   * - ``lmcache:chunk_statistics_unique_chunks``
     - Gauge
     - Estimated number of unique chunks encountered.
   * - ``lmcache:chunk_statistics_reuse_rate``
     - Gauge
     - Chunk reuse rate (0.0 to 1.0).
   * - ``lmcache:chunk_statistics_bloom_filter_size_mb``
     - Gauge
     - Memory usage of the Bloom filter in megabytes.
   * - ``lmcache:chunk_statistics_bloom_filter_fill_rate``
     - Gauge
     - Fill rate of the Bloom filter (0.0 to 1.0).
   * - ``lmcache:chunk_statistics_file_count``
     - Gauge
     - Number of files created when using the ``file_hash`` strategy.
   * - ``lmcache:chunk_statistics_current_file_size``
     - Gauge
     - Current size of the active statistics file in bytes.

Connector Metrics
~~~~~~~~~~~~~~~~~

.. list-table:: Connector Metrics
   :header-rows: 1
   :widths: 40 15 45

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:scheduler_unfinished_requests_count``
     - Gauge
     - Current count of unfinished requests in the scheduler.
   * - ``lmcache:connector_load_specs_count``
     - Gauge
     - Number of load specifications currently in the connector.
   * - ``lmcache:connector_request_trackers_count``
     - Gauge
     - Number of active request trackers in the connector.
   * - ``lmcache:connector_kv_caches_count``
     - Gauge
     - Number of KV caches currently managed by the connector.
   * - ``lmcache:connector_layerwise_retrievers_count``
     - Gauge
     - Number of layer-wise retrievers active in the connector.
   * - ``lmcache:connector_invalid_block_ids_count``
     - Gauge
     - Number of invalid block IDs encountered by the connector.
   * - ``lmcache:connector_requests_priority_count``
     - Gauge
     - Number of requests prioritized by the connector.
