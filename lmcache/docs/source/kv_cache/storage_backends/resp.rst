RESP (Native Redis/Valkey)
==========================

.. _resp-overview:

Overview
--------

The RESP backend is a high-performance native C++ storage connector for Redis and Valkey servers,
using the RESP2 wire protocol over TCP. It is designed for maximum throughput on KV cache
store and retrieval operations, achieving **6+ GB/s** on reads with optimal configuration.

Key advantages over the standard Redis connector:

- **Multi-threaded C++ I/O**: Worker threads operate in parallel with zero-copy buffer passing and full GIL release
- **Batched tiling**: Large batch operations are automatically split across worker threads for maximum parallelism
- **eventfd-based completions**: The kernel wakes Python on completion -- no polling overhead
- **Dual-mode support**: The same C++ connector works in both non-MP mode (via ``ConnectorClientBase``) and MP mode (via ``NativeConnectorL2Adapter`` as an L2 adapter)

The native C++ source lives in ``csrc/storage_backends/redis/``. See :doc:`Adding Native Connectors <../../developer_guide/extending_lmcache/native_connectors>` for the full architecture.


Prerequisites
-------------

- LMCache installed from source (``pip install -e .``) to compile the C++ extension
- A Redis 8.2+ or Valkey server (Redis 8.2 recommended for IO threads support)
- A machine with at least one GPU for vLLM inference


Redis Server Setup
------------------

.. important::
   Redis version and server configuration have a **major** impact on throughput.
   Using Redis 8.2 with IO threads yields ~6 GB/s reads vs ~1.5 GB/s with Redis 6.0 defaults.

**Build Redis 8.2 from source (recommended):**

.. code-block:: bash

    git clone https://github.com/redis/redis.git
    cd redis
    git checkout 8.2
    make -j

**Start the server with IO threads enabled:**

.. code-block:: bash

    ./src/redis-server \
        --protected-mode no \
        --save '' \
        --appendonly no \
        --io-threads 4 \
        --port 6379

.. list-table:: Recommended Server Flags
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Why
   * - ``--protected-mode no``
     - Allow connections from other hosts (use auth in production)
   * - ``--save '' --appendonly no``
     - Disable persistence -- KV cache is ephemeral, persistence wastes bandwidth
   * - ``--io-threads 4``
     - Enable multi-threaded I/O for parallel read/write handling
   * - ``--port 6379``
     - Default port (adjust if running multiple instances)

.. tip::
   The number of ``--io-threads`` should roughly match the number of physical cores
   available to the Redis process. 4 is a good starting point; benchmark with your
   hardware to find the optimum.


Chunk Size Selection and Throughput Tuning
------------------------------------------

The chunk size (in tokens) determines how many bytes each Redis key-value pair holds.
This is the **single most important parameter** for throughput.

**The sweet spot is ~4 MB per chunk.** Both smaller and larger chunks degrade throughput:

.. list-table:: Chunk Size vs Throughput (Redis 8.2, 8 workers)
   :header-rows: 1
   :widths: 20 20 20 20

   * - Chunk Size
     - Total Data
     - SET Throughput
     - GET Throughput
   * - 1 MB (500 keys)
     - 500 MB
     - ~3.5 GB/s
     - ~5.2 GB/s
   * - **4 MB (500 keys)**
     - **2 GB**
     - **~4.4 GB/s**
     - **~5.9 GB/s**
   * - 8 MB (200 keys)
     - 1.6 GB
     - ~4.2 GB/s
     - ~1.4 GB/s

**Why 4 MB?**

- Below ~2 MB, per-key overhead (RESP framing, TCP round-trips) dominates
- Above ~4 MB, Redis server-side memory allocation and TCP window sizes become bottlenecks
- At 4 MB, the balance between amortized overhead and memory pressure is optimal

**Calculating chunk size in tokens:**

The chunk size in bytes depends on the model's hidden dimension, number of KV heads,
number of layers, and dtype:

.. code-block:: text

    bytes_per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes

For ``meta-llama/Llama-3.1-8B-Instruct`` with BFloat16:

.. code-block:: text

    bytes_per_token = 2 * 8 * 128 * 32 * 2 = 131,072 bytes (~128 KB)
    chunk_size_tokens = 4 MB / 128 KB = 32 tokens

    # But typically chunk_size is set as token count in config:
    chunk_size: 16   # ~2 MB per chunk (conservative)
    chunk_size: 32   # ~4 MB per chunk (optimal for throughput)

.. note::
   The bytes-per-token calculation varies by model architecture. Larger models
   (e.g., 70B) have more layers and larger hidden dimensions, so fewer tokens
   are needed per chunk to reach the 4 MB sweet spot.


Throughput Sweep
~~~~~~~~~~~~~~~~

To find the optimal configuration for your hardware, use the included benchmark:

.. code-block:: bash

    cd examples/kv_cache_reuse/remote_backends/resp

    # Sweep chunk sizes
    for mb in 0.5 1 2 4 8; do
        echo "=== Chunk: ${mb} MB ==="
        python benchmark_resp_client.py \
            --host 127.0.0.1 --port 6379 \
            --chunk-mb $mb --num-workers 8 --num-keys 500
    done

    # Sweep worker counts
    for w in 1 2 4 8 16; do
        echo "=== Workers: $w ==="
        python benchmark_resp_client.py \
            --host 127.0.0.1 --port 6379 \
            --chunk-mb 4 --num-workers $w --num-keys 500
    done

Expected output:

.. code-block:: text

    Redis RESP Client Benchmark
    Server: 127.0.0.1:6379, Workers: 8
    Chunk size: 4096KB, Keys: 500
    ------------------------------------------------------------
    Batch SET:      4.36 GB/s  (1.95 GB written)
    Batch GET:      5.91 GB/s  (1.95 GB read)
    Batch EXISTS: 143528 ops/s  (500/500 hits)
    ------------------------------------------------------------
    All tests passed


Environment Variable Configuration
------------------------------------

Sensitive credentials (and optionally host/port) can be provided via environment
variables instead of config files or CLI arguments. This prevents secrets from
appearing in logged configuration at startup.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Description
   * - ``LMCACHE_RESP_USERNAME``
     - Redis ACL username. Used as default when ``username`` is not set in config/JSON.
   * - ``LMCACHE_RESP_PASSWORD``
     - Redis AUTH password. Used as default when ``password`` is not set in config/JSON.
   * - ``LMCACHE_RESP_HOST``
     - Redis hostname or IP. Used as default when ``host`` is not set in config/JSON/URL.
   * - ``LMCACHE_RESP_PORT``
     - Redis port. Used as default when ``port`` is not set in config/JSON/URL.

Config files (non-MP) and ``--l2-adapter`` JSON (MP) take precedence over
environment variables. Environment variables serve as defaults — they are used
when the corresponding config value is empty or unset. They are read at adapter
creation time inside the adapter itself, so they are **never stored in the
config object** and **never printed in startup logs**.

**Example — MP mode with env vars:**

.. code-block:: bash

    export LMCACHE_RESP_USERNAME="default"
    export LMCACHE_RESP_PASSWORD="secret"

    lmcache server \
        --l1-size-gb 10 \
        --eviction-policy LRU \
        --chunk-size 16 \
        --l2-adapter '{"type": "resp", "host": "localhost", "port": 6379, "num_workers": 8}' \
        --port 6555

**Example — Non-MP mode with env vars:**

.. code-block:: bash

    export LMCACHE_RESP_USERNAME="default"
    export LMCACHE_RESP_PASSWORD="secret"

    LMCACHE_CONFIG_FILE=resp-config.yaml \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --no-enable-prefix-caching \
        --load-format dummy

.. tip::
   For production deployments, always use environment variables for credentials
   rather than embedding them in config files or CLI arguments.


Non-MP Mode (Single Process)
-----------------------------

In non-MP mode, the RESP connector is used directly as a remote storage backend
via the ``RESPClient`` asyncio wrapper.

**Configuration file** (``resp-config.yaml``):

.. code-block:: yaml

    chunk_size: 16
    remote_url: "resp://localhost:6379"
    remote_serde: "naive"

Credentials can be set via environment variables (recommended) or in the config file
under ``extra_config`` (see `Environment Variable Configuration`_ above).

**Launch vLLM:**

.. code-block:: bash

    LMCACHE_CONFIG_FILE=resp-config.yaml \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --no-enable-prefix-caching \
        --load-format dummy

.. note::
   ``save_unfull_chunk`` must be off (default) and chunk metadata saving must be
   disabled for optimal throughput with the native RESP connector.


MP Mode (Multiprocess)
-----------------------

In MP mode, LMCache runs as a separate server process communicating with vLLM over
ZMQ. The RESP connector serves as an L2 adapter with variable-size chunk support.

**Step 1: Start Redis** (see `Redis Server Setup`_ above)

**Step 2: Launch LMCache MP Server:**

.. code-block:: bash

    lmcache server \
        --l1-size-gb 10 \
        --eviction-policy LRU \
        --chunk-size 16 \
        --l2-adapter '{"type": "resp", "host": "localhost", "port": 6379, "num_workers": 8}' \
        --port 6555

**Step 3: Launch vLLM with LMCache MP Connector:**

.. code-block:: bash

    PORT=8000
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --kv-transfer-config '{
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": 6555
            }
        }' \
        --no-enable-prefix-caching \
        --port $PORT \
        --load-format dummy


L2 Adapter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``--l2-adapter`` JSON accepts these fields:

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Field
     - Type
     - Default
     - Description
   * - ``type``
     - str
     - (required)
     - Must be ``"resp"``
   * - ``host``
     - str
     - (required)
     - Redis/Valkey hostname or IP
   * - ``port``
     - int
     - (required)
     - Redis/Valkey port
   * - ``num_workers``
     - int
     - 8
     - C++ worker threads for parallel I/O
   * - ``username``
     - str
     - ``""``
     - Redis ACL username (leave empty for no auth). Falls back to ``LMCACHE_RESP_USERNAME`` env var if empty.
   * - ``password``
     - str
     - ``""``
     - Redis AUTH password (leave empty for no auth). Falls back to ``LMCACHE_RESP_PASSWORD`` env var if empty.
   * - ``max_capacity_gb``
     - float
     - 0
     - Maximum L2 storage capacity in GB for client-side usage tracking. Required for L2 eviction. Set to 0 (default) to disable usage tracking.

L2 Eviction
~~~~~~~~~~~~

To enable automatic eviction of least-recently-used keys when the Redis backend fills up,
set ``max_capacity_gb`` and add an ``"eviction"`` block:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 10 \
        --eviction-policy LRU \
        --chunk-size 16 \
        --l2-adapter '{
            "type": "resp",
            "host": "localhost",
            "port": 6379,
            "num_workers": 8,
            "max_capacity_gb": 10,
            "eviction": {
                "eviction_policy": "LRU",
                "trigger_watermark": 0.8,
                "eviction_ratio": 0.2
            }
        }' \
        --port 6555

This configures a 10 GB capacity limit. When usage exceeds 80% (``trigger_watermark``),
the eviction controller will delete the least-recently-used ~20% of stored keys
(``eviction_ratio``) using the Redis ``DEL`` command.

.. note::
   ``max_capacity_gb`` enables **client-side** size tracking. It does not configure
   the Redis server's ``maxmemory`` setting. You should set ``max_capacity_gb`` to
   match or be slightly below your Redis server's available memory.


Testing the Setup
------------------

Send the same prompt twice. The first request stores KV cache to Redis; the second retrieves it.

.. code-block:: bash

    PORT=8000
    PROMPT="$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"

    # First request: store
    curl -s -X POST http://localhost:${PORT}/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"'"$PROMPT"'","max_tokens":10}'

    # Second request with same prefix: retrieve from Redis
    curl -s -X POST http://localhost:${PORT}/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"'"$PROMPT"'","max_tokens":10}'

Verify data was stored:

.. code-block:: bash

    redis-cli -p 6379 DBSIZE

Clear state between runs:

.. code-block:: bash

    redis-cli -p 6379 FLUSHALL


Best Practices
--------------

**Server deployment:**

- Use Redis 8.2+ with ``--io-threads 4`` (or more, matching available cores)
- Disable persistence (``--save '' --appendonly no``) for KV cache workloads
- Pin Redis to its own NUMA node if running on multi-socket systems
- For production, enable authentication with ``--requirepass`` and supply credentials via ``LMCACHE_RESP_USERNAME`` / ``LMCACHE_RESP_PASSWORD`` environment variables to keep them out of logs

**Client tuning:**

- Start with ``num_workers: 8`` and increase if the server has spare CPU and you're not saturating the network
- More workers help when chunk sizes are smaller (more keys per batch = more parallelism needed)
- On NUMA systems, ensure the LMCache process runs on the same NUMA node as the NIC

**Chunk size:**

- Target ~4 MB per chunk for maximum throughput
- Calculate the token count using your model's per-token byte size (see formula above)
- If unsure, run the benchmark sweep to find the optimum for your specific hardware

**Network:**

- Use localhost or loopback for single-machine deployments
- For cross-machine setups, ensure low-latency networking (ideally <100 us RTT)
- The RESP connector uses TCP; RDMA is not currently supported (consider :doc:`Mooncake <./mooncake>` for RDMA)


Additional Resources
--------------------

- Benchmark script: ``examples/kv_cache_reuse/remote_backends/resp/benchmark_resp_client.py``
- C++ source: ``csrc/storage_backends/redis/``
- Native connector architecture: ``csrc/storage_backends/README.md``
- Developer guide for adding new native connectors: :doc:`Adding Native Connectors <../../developer_guide/extending_lmcache/native_connectors>`
