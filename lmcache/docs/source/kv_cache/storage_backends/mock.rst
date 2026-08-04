Mock
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


LMCache provides a mock remote connector that allows you to manually set the peeking latency, read throughput, and write throughput inside of the remote url. It will create copies of your KV cache in unmanaged local RAM.

Configuration
-------------

Create a configuration file (e.g., ``mock.yaml``) with the following content:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: False
    max_local_cpu_size: 10
    remote_url: "mock://100/?peeking_latency=1&read_throughput=2&write_throughput=2"

The ``remote_url`` format is ``mock://SIZE/?peeking_latency=LATENCY&read_throughput=READ_GBPS&write_throughput=WRITE_GBPS`` where:

- ``SIZE``: Maximum storage size
- ``peeking_latency``: Latency for peeking operations (in milliseconds)
- ``read_throughput``: Read throughput (in GB/s)
- ``write_throughput``: Write throughput (in GB/s)

Usage
-----

Deploy a serving engine with the mock remote backend:

.. code-block:: bash

    LMCACHE_CONFIG_FILE=mock.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --disable-log-requests \
        --no-enable-prefix-caching

Check the retrieval (storing is async so the throughput there is meaningless) logs on the second query to confirm that the throughput is slightly lower than 2 GB/s (the CPU <-> GPU allocation/transfer also has overhead).

Example Query
-------------

Send a test request:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "'"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
        "max_tokens": 10
      }'

Expected Logs
-------------

You should see logs similar to the following:

.. code-block:: text

    (EngineCore_0 pid=586318) [2025-09-03 05:06:41,751] LMCache INFO: Reqid: cmpl-b34e7c5b2f3e46a592722db2c27f6fc0-0, Total tokens 12002, LMCache hit tokens: 12002, need to load: 12001 (vllm_v1_adapter.py:1049:lmcache.integration.vllm.vllm_v1_adapter)
    (EngineCore_0 pid=586318) [2025-09-03 05:06:42,736] LMCache INFO: Retrieved 12002 out of total 12002 out of total 12002 tokens. size: 1.651 gb, cost 980.6983 ms, throughput: 1.8939 GB/s; (cache_engine.py:503:lmcache.v1.cache_engine)

The logs confirm that the throughput is slightly lower than 2 GB/s due to CPU <-> GPU allocation/transfer overhead.

