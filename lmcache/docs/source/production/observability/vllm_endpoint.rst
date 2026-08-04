.. _observability_vllm_endpoint:

Metrics by vLLM API
==========================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


LMCache provides detailed metrics via a Prometheus endpoint, allowing for in-depth monitoring of cache performance and behavior.
This section outlines how to enable and configure observability from embedded vLLM ``/metrics`` API endpoint.


Quick Start Guide
-----------------

1) On vLLM/LMCache side
^^^^^^^^^^^^^^^^^^^^^^^

In v1, vLLM and LMCache run in separate processes, so you have to use multi‑process Prometheus.

The ``PROMETHEUS_MULTIPROC_DIR`` environment variable must be the same in both processes, as a IPC directory.

.. code-block:: bash

   PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus \
   #.. other environment variables \
   vllm serve $MODEL -port 8000 ...

Once the HTTP server is running, you can access the LMCache metrics at the ``/metrics`` endpoint.

.. code-block:: bash

   curl http://$<vllm-worker-ip>:8000/metrics | grep lmcache

   # Replace $IP with the IP address of a vLLM worker


And you will also find some ``.db`` files in the ``$PROMETHEUS_MULTIPROC_DIR`` directory.


2) Prometheus Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To scrape the LMCache metrics with a Prometheus server, add the following job to your ``prometheus.yml`` configuration,
or equivalent configuration to scrape the metrics endpoint:

.. code-block:: yaml

   scrape_configs:
     - job_name: 'lmcache'
       static_configs:
         - targets: ['<vllm-worker-ip>:8000']
       scrape_interval: 15s

Available Metrics
-----------------

LMCache exposes a variety of metrics to monitor its performance. The following table lists all available metrics organized by category:

.. list-table:: LMCache Metrics
   :header-rows: 1
   :widths: 30 15 55

   * - Metric Name
     - Type
     - Description
   * - **Core Request Metrics**
     - 
     - 
   * - ``lmcache:num_retrieve_requests``
     - Counter
     - Total number of retrieve requests
   * - ``lmcache:num_store_requests``
     - Counter
     - Total number of store requests
   * - ``lmcache:num_lookup_requests``
     - Counter
     - Total number of lookup requests
   * - ``lmcache:num_requested_tokens``
     - Counter
     - Total number of tokens requested for retrieval
   * - ``lmcache:num_hit_tokens``
     - Counter
     - Total number of cache hit tokens from retrieval
   * - ``lmcache:num_lookup_tokens``
     - Counter
     - Total number of tokens requested in lookup operations
   * - ``lmcache:num_lookup_hits``
     - Counter
     - Total number of tokens hit in lookup operations
   * - ``lmcache:num_vllm_hit_tokens``
     - Counter
     - Number of hit tokens in vLLM
   * - **Hit Rate Metrics**
     - 
     - 
   * - ``lmcache:retrieve_hit_rate``
     - Gauge
     - The hit rate for retrieve requests
   * - ``lmcache:lookup_hit_rate``
     - Gauge
     - The hit rate for lookup requests
   * - **Cache Usage Metrics**
     - 
     - 
   * - ``lmcache:local_cache_usage``
     - Gauge
     - Local cache usage in bytes
   * - ``lmcache:remote_cache_usage``
     - Gauge
     - Remote cache usage in bytes
   * - ``lmcache:local_storage_usage``
     - Gauge
     - Local storage usage in bytes
   * - **Performance Metrics**
     - 
     - 
   * - ``lmcache:time_to_retrieve``
     - Histogram
     - Time taken to retrieve from the cache (seconds)
   * - ``lmcache:time_to_store``
     - Histogram
     - Time taken to store to the cache (seconds)
   * - ``lmcache:retrieve_speed``
     - Histogram
     - Retrieval speed (tokens per second)
   * - ``lmcache:store_speed``
     - Histogram
     - Storage speed (tokens per second)
   * - **Remote Backend Metrics**
     - 
     - 
   * - ``lmcache:num_remote_read_requests``
     - Counter
     - Total number of read requests to remote backends
   * - ``lmcache:num_remote_read_bytes``
     - Counter
     - Total number of bytes read from remote backends
   * - ``lmcache:num_remote_write_requests``
     - Counter
     - Total number of write requests to remote backends
   * - ``lmcache:num_remote_write_bytes``
     - Counter
     - Total number of bytes written to remote backends
   * - ``lmcache:remote_time_to_get``
     - Histogram
     - Time taken to get data from remote backends (milliseconds)
   * - ``lmcache:remote_time_to_put``
     - Histogram
     - Time taken to put data to remote backends (milliseconds)
   * - ``lmcache:remote_time_to_get_sync``
     - Histogram
     - Time taken to get data from remote backends synchronously (milliseconds)
   * - **Network Monitoring Metrics**
     - 
     - 
   * - ``lmcache:remote_ping_latency``
     - Gauge
     - Latest ping latency to remote backends (milliseconds)
   * - ``lmcache:remote_ping_errors``
     - Counter
     - Number of ping errors to remote backends
   * - ``lmcache:remote_ping_successes``
     - Counter
     - Number of ping successes to remote backends
   * - ``lmcache:remote_ping_error_code``
     - Gauge
     - Latest ping error code to remote backends
   * - **Local CPU Backend Metrics**
     - 
     - 
   * - ``lmcache:local_cpu_evict_count``
     - Counter
     - Total number of evictions in local CPU backend
   * - ``lmcache:local_cpu_evict_keys_count``
     - Counter
     - Total number of evicted keys in local CPU backend
   * - ``lmcache:local_cpu_evict_failed_count``
     - Counter
     - Total number of failed evictions in local CPU backend
   * - ``lmcache:local_cpu_hot_cache_count``
     - Gauge
     - The size of the hot cache
   * - ``lmcache:local_cpu_keys_in_request_count``
     - Gauge
     - The size of the keys in request
   * - **Memory Management Metrics**
     - 
     - 
   * - ``lmcache:active_memory_objs_count``
     - Gauge
     - The number of active memory objects
   * - ``lmcache:pinned_memory_objs_count``
     - Gauge
     - The number of pinned memory objects


