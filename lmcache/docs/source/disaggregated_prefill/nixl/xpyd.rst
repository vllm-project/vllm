XpYd
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/disaggregated_prefill`.


X Prefiller, Y Decoder (XpYd) Example
--------------------------------------

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node with multiple prefiller and decoder instances. This configuration allows for horizontal scaling of both the compute-intensive prefill operations and the decode operations, enabling better resource utilization and higher throughput.

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

The XpYd setup consists of multiple components that can be scaled independently:

1. **Multiple Prefiller Servers** - Handle the prefill phase of inference (initial prompt processing)
2. **Multiple Decoder Servers** - Handle the decode phase of inference (token generation) 
3. **Proxy Server** - Coordinates requests between prefillers and decoders using round-robin load balancing

Example 2p2d Architecture:

.. code-block::

                ┌─────────────┐
                │   Client    │
                └─────┬───────┘
                      │
              ┌───────▼────────┐
              │  Proxy Server  │
              │    Port 9100   │──────────────────────|
              │  (Round-Robin) │                      |
              └───▲────────▲───┘                      |
                  │        │                          |
         ┌────────▼───┐  ┌─▼──────────┐               |
         │ Prefiller1 │  │ Prefiller2 │               |
         │ Port 7100  │  │ Port 7101  │               |
         │   GPU 0    │  │   GPU 1    │               |
         └─────▲──────┘  └─────▲──────┘               |
               │               │                      |
               │ NIXL transfer |                      |
               │               │                      |
          ┌────▼───────┐  ┌────▼──────┐               |
          │ Decoder 1  │  │ Decoder 2 │               |
          │ Port 7200  │  │ Port 7201 │               |
          │  GPU 2     │  │  GPU 3    │               |
          └────▲───────┘  └────▲──────┘               |
               │               │                      |
               └───────────────┴──────────────────────|

Prerequisites
~~~~~~~~~~~~~

- **LMCache**: Install with ``pip install lmcache``
- **NIXL**: Install from `NIXL GitHub repository <https://github.com/ai-dynamo/nixl>`_
- **Hardware**: At least 4 GPUs (2 for prefillers + 2 for decoders in 2p2d setup)
- **Model Access**: Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct

Quick Start (2p2d Example)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Set your Hugging Face token**:

   .. code-block:: bash

      export HF_TOKEN=hf_your_token_here

2. **Navigate to the example directory**:

   .. code-block:: bash

      cd examples/disagg_prefill/xpyd_experimental

3. **Run the example**:

   .. code-block:: bash

      bash disagg_example_xpyd.sh

The script will automatically:

- Launch two decoder instances on port 8200 and 8201 (GPU 2 and GPU 3)
- Launch two prefiller instances on ports 7100 and 7101 (GPU 0 and GPU 1)
- Launch a proxy server on port 9100 with round-robin load balancing
- Wait for all servers to be ready

Press ``Ctrl+C`` to stop all servers.

Configuration
~~~~~~~~~~~~~

**Important**: For correct KV cache transfer, ensure all processes use the same ``PYTHONHASHSEED`` to keep the hash of the KV cache consistent across processes:

   .. code-block:: bash

      export PYTHONHASHSEED=0

Prefiller Configuration
^^^^^^^^^^^^^^^^^^^^^^^

All prefillers share the same configuration via ``configs/lmcache-prefiller-config.yaml``:

.. code-block:: yaml

   local_cpu: False
   max_local_cpu_size: 0
   max_local_disk_size: 0

   enable_pd: True
   transfer_channel: "nixl"
   pd_role: "sender"
   pd_proxy_host: "localhost"
   pd_proxy_port: 7500
   pd_buffer_size: 1073741824 # 1GB
   pd_buffer_device: "cuda"

Key settings:

- ``pd_role: "sender"`` - Configures these instances to send KV cache data
- ``pd_buffer_size: 1073741824 # 1GB`` - Upper bound of PD transport buffer size (in bytes), aligned to chunk size
- ``pd_buffer_device: "cuda"`` - Uses GPU memory for buffering

Decoder Configuration
^^^^^^^^^^^^^^^^^^^^^

The decoder(s) are configured via ``configs/lmcache-decoder-x-config.yaml``:

.. code-block:: yaml

   local_cpu: False
   max_local_cpu_size: 0

   enable_pd: True
   transfer_channel: "nixl"
   pd_role: "receiver"
   pd_peer_host: "localhost"
   pd_peer_init_port: 730x
   pd_peer_alloc_port: 740x
   pd_buffer_size: 2147483648 # 2GB
   pd_buffer_device: "cuda"
   pd_backends: [UCX]

Key settings:

- ``pd_role: "receiver"`` - Configures these instances to receive KV cache data
- ``pd_buffer_size: 2147483648 # 2GB`` - Upper bound of PD transport buffer size (in bytes), aligned to chunk size
- ``pd_buffer_device: "cuda"`` - Uses GPU memory for buffering

Components Deep Dive
~~~~~~~~~~~~~~~~~~~~

Proxy Server (disagg_proxy_server.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The proxy server coordinates the multi-prefiller disaggregated workflow:

1. **Request Handling**: Receives client requests on port 9000
2. **Load Balancing**: Distributes requests across multiple prefillers using round-robin
3. **Prefill Coordination**: Sends requests to prefillers with ``max_tokens=1``
4. **Prefill Response**: Receives prefiller that says nixl transfer is done
5. **Response Streaming**: Streams the full response from the decoder
6. **Performance Monitoring**: Tracks Time-To-First-Token (TTFT) statistics

Key features:
- **Round-robin distribution**: Balances load across ``--num-prefillers`` instances
- **Fault tolerance**: Handles prefiller failures gracefully
- **Monitoring**: Provides detailed TTFT statistics for each prefiller

Supported endpoints:
- ``/v1/completions``
- ``/v1/chat/completions``

vLLM Server Launcher (disagg_vllm_launcher.sh)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script launches individual vLLM servers with appropriate configurations:

**Prefiller1 Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_CONFIG_FILE=$prefill_config_file \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=0 \
      vllm serve $MODEL \
      --port 7100 \
      --disable-log-requests \
      --enforce-eager \
      --no-enable-prefix-caching \
      --kv-transfer-config \
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'

**Prefiller2 Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_CONFIG_FILE=$prefill_config_file \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=1 \
      vllm serve $MODEL \
      --port 7101 \
      --disable-log-requests \
      --enforce-eager \
      --no-enable-prefix-caching \
      --kv-transfer-config \
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer2"}}'

**Decoder1 Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_CONFIG_FILE=$decode_config_file \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=2 \
      vllm serve $MODEL \
      --port 7200 \
      --disable-log-requests \
      --enforce-eager \
      --no-enable-prefix-caching \
      --kv-transfer-config \
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}'

**Decoder2 Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_CONFIG_FILE=$decode_config_file \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=3 \
      vllm serve $MODEL \
      --port 7201 \
      --disable-log-requests \
      --enforce-eager \
      --no-enable-prefix-caching \
      --kv-transfer-config \
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer2", "skip_last_n_tokens": 1}}'

Key differences from 1p1d:
- Each prefiller gets a unique ``lmcache_rpc_port`` (producer1, producer2, etc.)
- Each prefiller runs on a different GPU (CUDA_VISIBLE_DEVICES)
- Different ports for each prefiller (7100, 7101, etc.)
- Different ports for each decoder (7200, 7201, etc.)

Basic Test
~~~~~~~~~~

Once all servers are running, you can test with a simple curl command:

.. code-block:: bash

   curl -s -N -X POST http://127.0.0.1:9100/v1/completions   -H "Content-Type: application/json"   -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "prompt": "What date is today?",
      "max_tokens": 20,
      "temperature": 0.0
   }'

Performance Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^

For comprehensive performance testing, use vLLM's benchmark tool:

.. code-block:: bash

   vllm bench serve --port 9100 --seed $(date +%s) \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --dataset-name random --random-input-len 7500 --random-output-len 200 \
      --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos

Expected performance improvements with 2p2d:
- **Higher throughput**: Multiple prefillers can handle more concurrent requests
- **Better TTFT**: Load balancing reduces queuing delays
- **Improved utilization**: Better GPU utilization across multiple devices

Sample benchmark results:

.. code-block::

   ============ Serving Benchmark Result ============
   Successful requests:                     30
   Benchmark duration (s):                  31.34
   Total input tokens:                      224970
   Total generated tokens:                  6000
   Request throughput (req/s):              0.96
   Output token throughput (tok/s):         191.44
   Total Token throughput (tok/s):          7369.36
   ---------------Time to First Token----------------
   Mean TTFT (ms):                          313.41
   Median TTFT (ms):                        272.83
   P99 TTFT (ms):                           837.32
   -----Time per Output Token (excl. 1st token)------
   Mean TPOT (ms):                          8.84
   Median TPOT (ms):                        8.72
   P99 TPOT (ms):                           11.35
   ---------------Inter-token Latency----------------
   Mean ITL (ms):                           8.84
   Median ITL (ms):                         8.61
   P99 ITL (ms):                            11.43
   ==================================================

Log Files and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

The example generates multiple log files for comprehensive monitoring:

- ``prefiller1.log`` - First prefiller server logs and errors
- ``prefiller2.log`` - Second prefiller server logs and errors  
- ``decoder1.log`` - First decoder server logs and errors
- ``decoder1.log`` - First decoder server logs and errors
- ``proxy.log`` - Proxy server logs and TTFT statistics

The proxy server provides detailed statistics for each prefiller:

.. code-block::

   ===============================
   Num requests: 20
   Prefiller 1 TTFT stats:
    - Average (ms): 42.3
    - Median (ms): 40.1
    - 99th Percentile (ms): 48.7
   Prefiller 2 TTFT stats:
    - Average (ms): 43.8
    - Median (ms): 41.5
    - 99th Percentile (ms): 52.1
   ===============================

This helps identify performance differences between prefiller instances and optimize load balancing.

Troubleshooting
~~~~~~~~~~~~~~~

Common Issues
^^^^^^^^^^^^^

1. **GPU Memory**: Ensure each GPU has sufficient memory for the model
2. **NIXL Installation**: Verify NIXL is properly installed and accessible
3. **Port Conflicts**: Check that all required ports are available
4. **HF Token**: Ensure your Hugging Face token has access to Llama models
5. **GPU Assignment**: Verify CUDA_VISIBLE_DEVICES assignments don't conflict

Multi-Instance Specific Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Uneven Load**: Monitor prefiller statistics to ensure balanced distribution
2. **Resource Contention**: Watch for GPU memory pressure with multiple instances
3. **Network Bottlenecks**: Monitor NIXL transfer performance between instances
4. **Startup Timing**: Stagger prefiller launches to avoid resource conflicts



