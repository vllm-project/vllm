.. _p2p_sharing:

P2P KV Cache Sharing
====================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/p2p`.


P2P (Peer-to-Peer) KV cache sharing enables direct cache transfer between multiple serving engine instances without requiring a centralized cache server. This approach provides high-performance cache sharing with reduced latency and improved scalability, especially beneficial in distributed inference scenarios.

LMCache supports P2P sharing through a controller-based architecture using NIXL (NVIDIA Inference Xfer Library) for optimized data transfer between instances.

Prerequisites
-------------

- **Multi-GPU Setup**: Your server should have at least 2 GPUs
- **NIC**: RDMA is recommended for more performance.
- **NIXL**: Install from `NIXL <https://github.com/ai-dynamo/nixl>`_
- **vLLM**: v1 version is required, refer to :ref:`installation_guide` for details.
- **LMCache**: Install from :ref:`installation_guide`

Configuration
-------------

Create two configuration files for the P2P sharing setup.
 

**Instance 1 Configuration (example1.yaml)**:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 100
    enable_async_loading: True

    # P2P configurations
    enable_p2p: True
    p2p_host: "localhost"
    p2p_init_ports: 8200
    p2p_lookup_ports: 8201
    transfer_channel: "nixl"

    # Controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_1"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8500

    extra_config:
      lookup_backoff_time: 0.001

**Instance 2 Configuration (example2.yaml)**:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 100
    enable_async_loading: True

    # P2P configurations
    enable_p2p: True
    p2p_host: "localhost"
    p2p_init_ports: 8202
    p2p_lookup_ports: 8203
    transfer_channel: "nixl"

    # Controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_2"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8501

    extra_config:
      lookup_backoff_time: 0.001

Setup and Usage
---------------

**Step 1: Start the LMCache Controller**

.. code-block:: bash

    PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400, "heartbeat": 8082}'

Make sure that the 8300 and 8400 ports are set up in **controller_pull_url** and **controller_reply_url** in the configuration files.
Port 9000 is the controller main port, which is arbitrary and can be changed.

After starting the controller, access the WebUI at:

http://localhost:9000/

**Step 2: Start vLLM Engines with LMCache Workers**

If the NIC supports RDMA:

.. code-block:: bash

    export UCX_TLS=rc

If the NIC does not support RDMA:

.. code-block:: bash

    export UCX_TLS=tcp

Start vLLM engine 1 at port 8010:

.. code-block:: bash

    PYTHONHASHSEED=123  CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/path/to/example1.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8010 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start vLLM engine 2 at port 8011:

.. code-block:: bash

    PYTHONHASHSEED=123  CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=/path/to/example2.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8011 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 3: Test P2P Cache Sharing**

Send a request to vLLM engine 1 to populate the cache:

.. code-block:: bash

    curl -X POST http://localhost:8010/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

Send the same request to vLLM engine 2 to demonstrate cache retrieval from **engine 1**:

.. code-block:: bash

    curl -X POST http://localhost:8011/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

Expected Output
---------------

When the second request successfully retrieves cache from the first instance, you should see logs similar to:

.. code-block:: bash

    (EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,706] LMCache INFO:[0m Established connection to peer_init_url localhost:8200. The peer_lookup_url: localhost:8201 (p2p_backend.py:278:lmcache.v1.storage_backend.p2p_backend)
    (EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,792] LMCache INFO: Retrieved 1002 out of total 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s; (cache_engine.py:496:lmcache.v1.cache_engine)

These logs indicate successful P2P connection establishment and high-throughput cache retrieval.



**Step 4: Benchmarking P2P Cache Sharing**

Send a request workload to instance 1 to populate the cache:

.. code-block:: bash

    python benchmarks/long_doc_qa/long_doc_qa.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-documents 50 \
    --document-length 10000 \
    --output-len 100 \
    --repeat-count 1 \
    --repeat-mode tile \
    --port 8010 \
    --max-inflight-requests 4

Send the same request workload to instance 2 to demonstrate cache retrieval from **instance 1**:

.. code-block:: bash

    python benchmarks/long_doc_qa/long_doc_qa.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-documents 50 \
    --document-length 10000 \
    --output-len 100 \
    --repeat-count 1 \
    --repeat-mode tile \
    --port 8011 \
    --max-inflight-requests 4


Benchmark Results
-----------------

First instance metrics:

.. code-block:: text
 
    Warmup round mean TTFT: 2.286s
    Warmup round time: 37.957s
    Warmup round prompt count: 50
    Warmup round successful prompt count: 50
    
    === BENCHMARK RESULTS ===
    Query round mean TTFT: 2.028s
    Query round time: 38.323s
    Query round prompt count: 50
    Query round successful prompt count: 50

Second instance metrics:

.. code-block:: text
 
    Warmup round mean TTFT: 1.036s
    Warmup round time: 13.814s
    Warmup round prompt count: 50
    Warmup round successful prompt count: 50
    
    === BENCHMARK RESULTS ===
    Query round mean TTFT: 0.490s
    Query round time: 7.964s
    Query round prompt count: 50
    Query round successful prompt count: 50

In this example, the warm-up round metric in long_doc_qa is used because no existing KV cache is reused within an instance to benefit solely from P2P sharing. With LMCache P2P sharing enabled, the time to first token (TTFT) is reduced by 54.7%, from 2.286 s to 1.036 s, with a 63.6% reduction in total inference time (37.957 s → 13.814 s).

