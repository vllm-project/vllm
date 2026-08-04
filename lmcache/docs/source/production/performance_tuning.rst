.. _performance_tuning:

Performance Tuning
==================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


This guide covers key LMCache configuration options that can help
you optimize performance in production deployments.

Minimum Retrieve Tokens
------------------------

When LMCache finds a partial KV cache hit, it loads the cached tokens
into GPU memory to avoid recomputation. However, if only a small
number of tokens are hit, the overhead of loading them from the
cache may outweigh the benefit of skipping recomputation.

The ``min_retrieve_tokens`` setting lets you set a threshold: if the
number of tokens that need to be loaded is below this value, LMCache
will skip the retrieve and let the inference engine recompute them
instead.

.. note::

    Even when retrieve is skipped, LMCache still records the hit
    tokens internally so that it does **not** re-store chunks that
    already exist in the cache.

When to Use
~~~~~~~~~~~

Consider setting ``min_retrieve_tokens`` when:

- For the backend you are using, the transfer latency is noticeable
  for small payloads.
- Your workload has many requests with **low cache hit ratios**,
  where recomputation is faster than cache loading.
- You want to reduce unnecessary I/O for marginal cache hits.

You can increase or decrease based on your latency observations.

Configuration
~~~~~~~~~~~~~

**YAML configuration file:**

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    min_retrieve_tokens: 1024

**Environment variable:**

.. code-block:: bash

    export LMCACHE_MIN_RETRIEVE_TOKENS=1024

Working Example
~~~~~~~~~~~~~~~

1. Create an LMCache configuration file ``lmcache_config.yaml``:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5.0
    min_retrieve_tokens: 1024

2. Start vLLM with LMCache:

.. code-block:: bash

    LMCACHE_CONFIG_FILE=lmcache_config.yaml \
        vllm serve Qwen/Qwen3-0.6B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --disable-log-requests \
        --no-enable-prefix-caching

3. Send a request to populate the cache:

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Explain the theory of relativity.",
        "max_tokens": 50,
        "temperature": 0.7
      }'

4. Send a similar request that partially reuses the cached prefix:

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Explain the theory of relativity in simple terms....",
        "max_tokens": 50,
        "temperature": 0.7
      }'

5. Check the server logs. If the number of loadable hit tokens is
   below 1024, you will see a log message like:

.. code-block:: text

    LMCache hit tokens: 762, but need to load: 762 < min_retrieve 1024,
    skip retrieve but record for save skip

This confirms that the small cache hit was skipped in favor of
recomputation, avoiding unnecessary transfer overhead.
