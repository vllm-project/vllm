.. _observability_chunk_statistics:

Chunk Statistics
================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The chunk statistics feature provides insights into KV cache chunk reuse patterns, helping you understand cache efficiency and optimize your deployment.

Overview
--------

Chunk statistics tracks and analyzes KV cache chunks to provide metrics on:

- **Total chunks processed**: The total number of chunks that have been processed
- **Unique chunks**: The number of distinct chunks encountered
- **Duplicate chunks**: The number of repeated chunks
- **Reuse rate**: The ratio of duplicate chunks to total chunks, indicating cache efficiency

This information is valuable for:

- Understanding cache hit patterns
- Optimizing cache size and eviction policies
- Analyzing workload characteristics
- Capacity planning for production deployments

Recording Strategies
--------------------

LMCache supports multiple recording strategies, each optimized for different use cases:

Memory Bloom Filter Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A memory-efficient strategy using Bloom filters for probabilistic duplicate detection.

**Advantages:**

- Low memory footprint
- Fast lookup operations
- Suitable for large-scale deployments

**Configuration:**

.. code-block:: yaml

    enable_chunk_statistics: true
    chunk_statistics_strategy: "memory_bloom_filter"
    extra_config:
      chunk_statistics_mem_bf_expected_chunks: 20000000  # Expected number of chunks
      chunk_statistics_mem_bf_false_positive_rate: 0.01  # Target false positive rate

**Environment Variables:**

.. code-block:: bash

    LMCACHE_ENABLE_CHUNK_STATISTICS=true
    LMCACHE_CHUNK_STATISTICS_STRATEGY=memory_bloom_filter
    LMCACHE_EXTRA_CONFIG='{"chunk_statistics_mem_bf_expected_chunks": 20000000, "chunk_statistics_mem_bf_false_positive_rate": 0.01}'

File Hash Strategy
~~~~~~~~~~~~~~~~~~

A file-based strategy that writes chunk hashes to disk for exact tracking and offline analysis.

**Advantages:**

- Exact duplicate detection (no false positives)
- Persistent storage for offline analysis
- Automatic file rotation and cleanup

**Configuration:**

.. code-block:: yaml

    enable_chunk_statistics: true
    chunk_statistics_strategy: "file_hash"
    extra_config:
      chunk_statistics_file_output_dir: "/tmp/lmcache_chunk_statistics"
      chunk_statistics_file_rotation_size: 104857600      # 100MB rotation size
      chunk_statistics_file_max_count: 100                # Maximum number of files

**Environment Variables:**

.. code-block:: bash

    LMCACHE_ENABLE_CHUNK_STATISTICS=true
    LMCACHE_CHUNK_STATISTICS_STRATEGY=file_hash
    LMCACHE_EXTRA_CONFIG='{"chunk_statistics_file_output_dir": "/tmp/lmcache_chunk_statistics", "chunk_statistics_file_rotation_size": 104857600, "chunk_statistics_file_max_count": 100}'

Quick Start Guide
-----------------

Step 1: Enable Chunk Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure your LMCache instance with chunk statistics enabled:

**Using YAML Configuration:**

.. code-block:: yaml

    # Enable internal API server for interacting with the chunk statistics API
    internal_api_server_enabled: True
    # Base port for the API server
    # actual_port = internal_api_server_port_start + index
    # Scheduler → 6999 + 0 = 6999
    # Worker 0 → 6999 + 1 = 7000
    internal_api_server_port_start: 6999
    # Enable chunk statistics with memory bloom filter strategy
    enable_chunk_statistics: true
    chunk_statistics_strategy: "memory_bloom_filter"
    chunk_statistics_auto_start_statistics: true

    # Bloom filter configuration
    extra_config:
      chunk_statistics_mem_bf_expected_chunks: 20000000
      chunk_statistics_mem_bf_false_positive_rate: 0.01

**Using vLLM with LMCache:**

.. code-block:: bash

    LMCACHE_CONFIG_FILE=lmcache.yaml \
    PYTHONHASHSEED=0 \
    python3 -m vllm.entrypoints.cli.main serve <model_path> \
    --load-format dummy \
    -tp 2 \
    --trust-remote-code \
    --served-model-name vllm_cpu_offload \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 64 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Step 2: Access Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve statistics through the internal API server:

.. code-block:: bash

    # Get current statistics (default port: 6999 for scheduler)
    curl http://localhost:6999/chunk_statistics/status

**Example Response:**

.. code-block:: json

    {
      "enabled": true,
      "total_requests": 3,
      "timing": {
        "lookup_time_seconds": 0.044486284255981445,
        "record_statistics_time_seconds": 6.246566772460938e-05,
        "check_exit_conditions_time_seconds": 5.7220458984375e-06,
        "total_time_seconds": 0.04455447196960449,
        "overhead_time_seconds": 6.818771362304688e-05,
        "overhead_percentage": 0.1530434782608696
      },
      "total_chunks": 12,
      "unique_chunks": 9,
      "duplicate_chunks": 3,
      "reuse_rate": 0.25,
      "async_queue": {
        "enabled": true,
        "capacity": 100000,
        "current_size": 0,
        "max_size_reached": 0,
        "full_blocks": 0,
        "utilization": 0.0
      },
      "bloom_filter": {
        "size_mb": 11.426279067993164,
        "hash_count": 6,
        "item_count": 9,
        "bits_set": 54,
        "fill_rate": 5.633768549952377e-07,
        "expected_elements": 10000000,
        "false_positive_rate": 0.01
      },
      "timestamp": 1763026696.7670634,
      "auto_exit_enabled": false,
      "auto_exit_timeout_hours": 0.0,
      "auto_exit_target_unique_chunks": null
    }

Configuration Options
---------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Basic Configuration Options
   :header-rows: 1
   :widths: 40 20 100

   * - Configuration Key
     - Default Value
     - Description
   * - ``enable_chunk_statistics``
     - ``false``
     - Enable chunk statistics tracking
   * - ``chunk_statistics_strategy``
     - ``memory_bloom_filter``
     - Recording strategy: ``memory_bloom_filter`` or ``file_hash``
   * - ``chunk_statistics_auto_start_statistics``
     - ``false``
     - Automatically start statistics collection on initialization
   * - ``chunk_statistics_auto_exit_timeout_hours``
     - ``0.0``
     - Auto-stop after specified hours (0 = disabled)
   * - ``chunk_statistics_auto_exit_target_unique_chunks``
     - ``0``
     - Auto-stop after reaching target unique chunks (0 = disabled)

Memory Bloom Filter Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure these options in the ``extra_config`` section:

.. list-table:: Bloom Filter Configuration
   :header-rows: 1
   :widths: 40 20 100

   * - Configuration Key
     - Default Value
     - Description
   * - ``chunk_statistics_mem_bf_expected_chunks``
     - ``20000000``
     - Expected number of chunks for capacity planning
   * - ``chunk_statistics_mem_bf_false_positive_rate``
     - ``0.01``
     - Target false positive rate (1%)

File Hash Options
~~~~~~~~~~~~~~~~~

Configure these options in the ``extra_config`` section:

.. list-table:: File Hash Configuration
   :header-rows: 1
   :widths: 40 20 100

   * - Configuration Key
     - Default Value
     - Description
   * - ``chunk_statistics_file_output_dir``
     - ``/tmp/lmcache_chunk_statistics``
     - Directory for storing chunk hash files
   * - ``chunk_statistics_file_rotation_size``
     - ``104857600``
     - File size threshold for rotation (bytes, default 100MB)
   * - ``chunk_statistics_file_max_count``
     - ``100``
     - Maximum number of files to keep

Advanced Usage
--------------

Programmatic Control
~~~~~~~~~~~~~~~~~~~~

Control statistics collection programmatically through the internal API:

.. code-block:: bash

    # Get current statistics (default port: 6999 for scheduler)
    curl http://localhost:6999/chunk_statistics/status

    # Pretty print JSON output
    curl http://localhost:6999/chunk_statistics/status | jq .

    # Start statistics collection (if not auto-started)
    curl -X POST http://localhost:6999/chunk_statistics/start

    # Stop statistics collection
    curl -X POST http://localhost:6999/chunk_statistics/stop

    # Reset statistics
    curl -X POST http://localhost:6999/chunk_statistics/reset

Auto-Stop Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Configure automatic stopping based on time or chunk count:

.. code-block:: yaml
    chunk_statistics_auto_exit_timeout_hours: 1.0  # Stop after 1 hour
    chunk_statistics_auto_exit_target_unique_chunks: 100000  # Stop after 100K unique chunks

Prometheus Metrics
~~~~~~~~~~~~~~~~~~

When using the internal API server, chunk statistics are exposed as Prometheus metrics:

- ``lmcache_chunk_statistics_total_chunks``: Total number of chunks processed
- ``lmcache_chunk_statistics_unique_chunks``: Number of unique chunks
- ``lmcache_chunk_statistics_reuse_rate``: Cache reuse rate (0.0 to 1.0)
- ``lmcache_chunk_statistics_bloom_filter_size_mb``: Bloom filter memory usage (MB)
- ``lmcache_chunk_statistics_bloom_filter_fill_rate``: Bloom filter fill rate (0.0 to 1.0)
- ``lmcache_chunk_statistics_file_count``: Number of hash files created
- ``lmcache_chunk_statistics_current_file_size``: Current file size (bytes)

Offline Analysis
~~~~~~~~~~~~~~~~

For the file hash strategy, you can perform detailed offline analysis of collected chunk hash data.

Using the Analysis Script
^^^^^^^^^^^^^^^^^^^^^^^^^^

LMCache provides a comprehensive analysis script at ``examples/chunk_statistics/analyze_chunk_hashes.py`` that supports multiple analysis modes.

Best Practices
--------------

1. **Choose the Right Strategy:**
   
   - Use **memory_bloom_filter** for real-time monitoring with minimal overhead
   - Use **file_hash** when exact tracking is required or for offline analysis

2. **Tune Bloom Filter Parameters:**
   
   - Set ``expected_chunks`` based on your workload size
   - Lower ``false_positive_rate`` increases memory usage but improves accuracy

3. **Monitor Memory Usage:**
   
   - Track ``bloom_filter_size_mb`` metric to ensure it fits in available memory
   - Adjust ``expected_chunks`` if memory usage is too high

4. **File Rotation:**
   
   - Configure appropriate ``file_rotation_size`` to balance file size and count
   - Set ``file_max_count`` to prevent unlimited disk usage

5. **Production Deployment:**
   
   - Enable auto-stop to prevent indefinite data collection
   - Use internal API server for centralized metrics collection
   - Integrate with your monitoring stack (Prometheus, Grafana, etc.)

Troubleshooting
---------------

Statistics Not Updating
~~~~~~~~~~~~~~~~~~~~~~~

**Issue:** Statistics remain at zero or don't update.

**Solutions:**

- Verify ``enable_chunk_statistics`` is set to ``true``
- Check that statistics collection is started (auto-start or manual start)
- Ensure requests are being processed by the LMCache instance

High Memory Usage
~~~~~~~~~~~~~~~~~

**Issue:** Bloom filter consuming too much memory.

**Solutions:**

- Reduce ``chunk_statistics_mem_bf_expected_chunks``
- Increase ``chunk_statistics_mem_bf_false_positive_rate`` (trade accuracy for memory)
- Consider switching to ``file_hash`` strategy

File System Full
~~~~~~~~~~~~~~~~

**Issue:** Disk space exhausted with file hash strategy.

**Solutions:**

- Reduce ``chunk_statistics_file_max_count``
- Decrease ``chunk_statistics_file_rotation_size``
- Implement external log rotation or archival
