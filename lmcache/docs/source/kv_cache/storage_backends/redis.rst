Redis
=====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/resp`.


.. _redis-overview:

Overview
--------

Redis is an in-memory key-value store and is a supported option for remote KV Cache offloading in LMCache.
Some other remote backends are :doc:`Mooncake <./mooncake>`, :doc:`Valkey <./valkey>`, and :doc:`InfiniStore <./infinistore>`.
This guide will mainly focus on single-node Redis but also shows you how to set up Redis Sentinels and an LMCache Server.

Two ways to configure LMCache Redis Offloading:
-----------------------------------------------

**1. Environment Variables:**

.. code-block:: bash

    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Redis host
    export LMCACHE_REMOTE_URL="redis://your-redis-host:6379"
    # Redis Sentinel hosts (for high availability)
    # export LMCACHE_REMOTE_URL="redis-sentinel://localhost:26379,localhost:26380,localhost:26381"
    # LMCache Server host
    # export LMCACHE_REMOTE_URL="lm://localhost:65432"

    # How to serialize and deserialize KV cache on remote transmission
    export LMCACHE_REMOTE_SERDE="naive" # "naive" (default) or "cachegen"

**2. Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Redis host
    remote_url: "redis://your-redis-host:6379"
    # Redis Sentinel hosts (for high availability)
    # remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"
    # LMCache Server host
    # remote_url: "lm://localhost:65432"

    # How to serialize and deserialize KV cache on remote transmission
    remote_serde: "naive" # "naive" (default) or "cachegen"

Dynamic Plugin Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    local_cpu: true

    remote_storage_plugins:
      - "redis"

    extra_config:
      remote_storage_plugin.redis.redis_url: "redis://your-redis-host:6379"

Remote Storage Explanation:
----------------------------

LMCache's backend is obeys the natural memory hierarchy of prioritizing CPU RAM offloading, then Local Storage
offloading, and finally remote offloading.

For LMCache to know how to create a connector to a remote backend, you must specify in
``remote_url`` a connector type followed by one or most host:port pairs (depending on what connector type is used).
If ``remote_url`` is set to ``None``, LMCache will not use any remote storage.

Examples of ``remote_url``'s:

.. code-block:: yaml

    remote_url: "redis://your-redis-host:6379"
    remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"
    remote_url: "lm://localhost:65432"
    remote_url: "infinistore://127.0.0.1:12345"
    remote_url: "mooncakestore://127.0.0.1:50051"

Remote Storage Example
-----------------------

.. _redis-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. You can adjust the max model length of your vllm instance depending on your GPU memory.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- Hugging Face access to ``meta-llama/Meta-Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token

**Step 0. Set up a directory for this example:**

.. code-block:: bash

    mkdir lmcache-redis-offload-example
    cd lmcache-redis-offload-example

**Step 1. Start a Redis server:**

.. code-block:: bash

    # Ubuntu / Debian Installation
    sudo apt-get install redis
    redis-server # starts the server on default port 6379

Check if Redis is running:

.. code-block:: bash

    redis-cli ping

Expected Response:

.. code-block:: text

    PONG

**Optional: Setting up Sentinels:**

To enable high availability with Redis, you can configure Redis sentinels to
monitor the master and automatically fail over to a replica if needed.

**Step 1a. Start a Redis replica:**

.. code-block:: bash

    redis-server --port 6380 --replicaof 127.0.0.1 6379

**Step 1b. Create Sentinel configuration files:**

Create three files: ``sentinel-26379.conf``, ``sentinel-26380.conf``, and ``sentinel-26381.conf``, with contents like this:

.. code-block:: ini

    port 26379  # Use 26380 and 26381 in other files respectively
    sentinel monitor mymaster 127.0.0.1 6379 1
    sentinel down-after-milliseconds mymaster 5000
    sentinel failover-timeout mymaster 10000
    sentinel parallel-syncs mymaster 1

**Step 1c. Start each Sentinel:**

.. code-block:: bash

    redis-server sentinel-26379.conf --sentinel
    redis-server sentinel-26380.conf --sentinel
    redis-server sentinel-26381.conf --sentinel

**Step 1d. Make sure the Sentinels are tracking the master:**

.. code-block:: bash

    redis-cli -p 26379 sentinel master mymaster
    redis-cli -p 26380 sentinel master mymaster
    redis-cli -p 26381 sentinel master mymaster

**Step 1e. Verify everything is running:**

.. code-block:: bash

    ps aux | grep redis

You should see something like this (without the comments):

.. code-block:: text

    # Master (read-write)
    user      60816  0.1  0.0  69804 11132 ?        Sl   04:11   0:00 redis-server *:6379
    # Replica (read-only mirror of 6379)
    user      60903  0.1  0.0  80048 10928 ?        Sl   04:12   0:00 redis-server *:6380
    # Sentinels (monitor the master and hold quorums to decide when to failover)
    user      61301  0.1  0.0  67244 10944 ?        Sl   04:14   0:00 redis-server *:26379 [sentinel]
    user      61382  0.1  0.0  67244 10944 ?        Sl   04:14   0:00 redis-server *:26380 [sentinel]
    user      61462  0.1  0.0  67244 10944 ?        Sl   04:15   0:00 redis-server *:26381 [sentinel]


**Alternative: Starting an LMCache Server:**

The ``lmcache_server`` CLI entrypoint starts a remote LMCache server and comes with
the ``lmcache`` package.

.. code-block:: bash

    lmcache_server <host> <port> <device>

    lmcache_server localhost 65432

Currently, the only supported device is "cpu" (which is the default, so you don't need to specify it).


**Step 2. Start a vLLM server with remote offloading enabled:**

Create a an lmcache configuration file called: ``redis-offload.yaml``

.. code-block:: yaml

    # disabling CPU RAM offload not recommended (on by default) but
    # if you want to confirm that the remote backend works by itself
    # local_cpu: false
    chunk_size: 256
    remote_url: "redis://localhost:6379"
    remote_serde: "naive"

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # disabling CPU RAM offload not recommended (on by default) but
    # if you want to confirm that the remote backend works by itself
    # LMCACHE_LOCAL_CPU=False \
    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_REMOTE_URL="redis://localhost:6379" \
    # LMCACHE_REMOTE_SERDE="naive"
    LMCACHE_CONFIG_FILE="redis-offload.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Optional: Sentinels**

Create a an lmcache configuration file called: ``redis-sentinel-offload.yaml``

.. code-block:: yaml

    chunk_size: 256
    remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"
    remote_serde: "naive"

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_REMOTE_URL="redis-sentinel://localhost:26379,localhost:26380,localhost:26381" \
    # LMCACHE_REMOTE_SERDE="naive"
    LMCACHE_CONFIG_FILE="redis-sentinel-offload.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Alternative: LMCache Server**

Create a an lmcache configuration file called: ``lmcache-server-offload.yaml``

.. code-block:: yaml

    chunk_size: 256
    remote_url: "lm://localhost:65432"
    remote_serde: "naive"

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_REMOTE_URL="lm://localhost:65432" \
    # LMCACHE_REMOTE_SERDE="naive"
    LMCACHE_CONFIG_FILE="lmcache-server-offload.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 3. Viewing and Managing LMCache Entries in Redis:**

If you would like to feel the TTFT speed up with offloading and KV Cache reuse, feel free to use the same
``query-twice.py`` script and ``man-bash.txt`` long context as in :doc:`CPU RAM <./cpu_ram>` and :doc:`Local Storage <./local_storage>`.

Here, we are instead going to demonstrate how to search for and modify LMCache KV Chunk entries in Redis.

Please note that the official LMCache way to achieve this redis-specific functionality of viewing and modifying LMCache KV Chunks is available in :doc:`LMCache Controller <../../kv_cache_management/index>`.

Let's warm/populate LMCache first with ``curl`` this time:

.. code-block:: bash

    curl -X 'POST' \
    'http://127.0.0.1:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
        {"role": "system", "content": "You are a helpful AI coding assistant."},
        {"role": "user", "content": "Write a segment tree implementation in python"}
        ],
        "max_tokens": 150
    }'

LMCache stores data in Redis using a structured key format. Each key contains the following information in a delimited format:

.. code-block:: text

    model_name@world_size@worker_id@chunk_hash

- `model_name`: Name of the language model
- `world_size`: Total number of workers in distributed deployment
- `worker_id`: ID of the worker that created this cache entry, in the range of [0, world_size - 1]
- `chunk_hash`: Hash of the token chunk (SHA-256 based)

For example, a typical key might look like:

.. code-block:: text

    vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@a1b2c3d4e5f6...

**Using redis-cli to View LMCache Data**

To inspect and manage LMCache entries in Redis:

.. code-block:: bash

    redis-cli -h localhost -p 6379

**Optional: If you are using sentinels, first find the master port:**

.. code-block:: bash

    redis-cli -p 26379 sentinel get-master-addr-by-name mymaster
    redis-cli -h localhost -p <master-port>


**List LMCache keys:**

Notice (from the suffixes of the keys) that each LMCache KV Chunk has two entries: ``kv_bytes`` and ``metadata``

.. code-block:: bash

    # Show all keys
    localhost:6379> KEYS *
    1) "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...kv_bytes"
    2) "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...metadata"
    # Show keys for a specific model
    localhost:6379> KEYS *Llama-3.1-8B-Instruct*
    1) "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...kv_bytes"
    2) "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...metadata"

**Delete LMCache entries:**

.. code-block:: bash

    localhost:6379> DEL *

Delete a specific LMCache entry:

.. code-block:: bash

    localhost:6379> DEL "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...kv_bytes"
    localhost:6379> KEYS *
    1) "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...metadata"


**Check if a key exists:**

.. code-block:: bash

    localhost:6379> EXISTS "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...kv_bytes"

**View memory usage for a key:**

Notice that the ``kv_bytes`` entry is what is exactly holding the KV Chunk and is much
larger than the ``metadata`` entry.

.. code-block:: bash

    localhost:6379> MEMORY USAGE "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...metadata"
    (integer) 198
    localhost:6379> MEMORY USAGE "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...kv_bytes"
    (integer) 7340200

**Delete specific keys:**

.. code-block:: bash

    # Delete a single key
    localhost:6379> DEL "vllm@meta-llama/Llama-3.1-8B-Instruct@1@0@02783dafec...kv_bytes"

.. code-block:: bash

    # Delete all keys matching a pattern
    redis-cli -h localhost -p 6379 --scan --pattern "vllm@meta-llama/Llama-3.1-8B-Instruct*" \
        | xargs redis-cli -h localhost -p 6379 DEL


**Monitor Redis in real-time:**

.. code-block:: bash

    localhost:6379> MONITOR

**Get Redis stats for LMCache:**

.. code-block:: bash

    # Get memory stats
    localhost:6379> INFO memory

    # Get statistics about operations
    localhost:6379> INFO stats

This tutorial utilized the ``redis-cli`` to directly peak into a remote backend and manipualte
KV Chunks.

Once again, please refer to the :doc:`LMCache Controller <../../kv_cache_management/index>`
for the official LMCache way of controlling and routing your KV Caches in your LMCache instances.

**Step 4. Clean up:**

.. code-block:: bash

    redis-cli shutdown

    # Optional:

    # Shut down the Redis replica (if started)
    redis-cli -p 6380 shutdown

    # Shut down all Redis Sentinels (if started)
    redis-cli -p 26379 shutdown
    redis-cli -p 26380 shutdown
    redis-cli -p 26381 shutdown

    # (Optional) Remove temporary files or configs
    rm -f sentinel-26379.conf sentinel-26380.conf sentinel-26381.conf

    # Confirm no Redis processes are still running
    ps aux | grep redis


