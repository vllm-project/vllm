1p1d
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/disaggregated_prefill`.


One Prefiller, One Decoder (1p1d) Example
------------------------------------------

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node with a 1 prefiller + 1 decoder setup. This configuration separates the compute-intensive prefill operations from the decode operations, allowing for better resource utilization and performance optimization.

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

The 1p1d setup consists of three main components:

1. **Prefiller Server** - Handles the prefill phase of inference (initial prompt processing)
2. **Decoder Server** - Handles the decode phase of inference (token generation) 
3. **Proxy Server** - Coordinates requests between the prefiller and decoder

.. code-block::

                ┌─────────────┐
                │   Client    │
                └─────┬───────┘
                      │
              ┌───────▼───────┐
              │ Proxy Server  │
              │   Port 9100   │
              └───▲───────┬───┘
                  │       │
         ┌────────▼──┐  ┌─▼────────┐
         │ Prefiller │  │ Decoder  │
         │Port 7100  │  │Port 7200 │
         │  GPU 0    │  │  GPU 1   │
         └───────────┘  └──────────┘
                  ▲       ▲
                  │       │
                  └───────┘
                   NIXL Transfer

Prerequisites
~~~~~~~~~~~~~

- **LMCache**: Install with ``pip install lmcache``
- **NIXL**: Install from `NIXL GitHub repository <https://github.com/ai-dynamo/nixl>`_
- **Hardware**: At least 2 GPUs
- **Model Access**: Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct

Quick Start
~~~~~~~~~~~

1. **Set your Hugging Face token**:

   .. code-block:: bash

      export HF_TOKEN=hf_your_token_here

2. **Navigate to the example directory**:

   .. code-block:: bash

      cd examples/disagg_prefill/1p1d_experimental

3. **Run the example**:

   .. code-block:: bash

      bash disagg_example_1p1d.sh

The script will automatically:

- Launch a prefiller instance on port 7100 (GPU 0)
- Launch a decoder instance on port 7200 (GPU 1)  
- Launch a proxy server on port 9100
- Wait for all servers to be ready

Press ``Ctrl+C`` to stop all servers.

Configuration
~~~~~~~~~~~~~

**Important**: For correct KV cache transfer, ensure all processes use the same ``PYTHONHASHSEED`` to keep the hash of the KV cache consistent across processes:

   .. code-block:: bash

      export PYTHONHASHSEED=0

Prefiller Configuration
^^^^^^^^^^^^^^^^^^^^^^^

The prefiller is configured via ``configs/lmcache-prefiller-config.yaml``:

.. code-block:: yaml

   local_cpu: True
   max_local_cpu_size: 5
   max_local_disk_size: 0

   enable_pd: True
   transfer_channel: "nixl"
   pd_role: "sender"
   pd_proxy_host: "localhost"
   pd_proxy_port: 7500
   pd_buffer_size: 1073741824 # 1GB
   pd_buffer_device: "cuda"

Key settings:

- ``pd_role: "sender"`` - Configures this instance to send KV cache data
- ``pd_buffer_size: 1073741824 # 1GB`` - Buffer size for transfers
- ``pd_buffer_device: "cuda"`` - Uses GPU memory for buffering

Decoder Configuration
^^^^^^^^^^^^^^^^^^^^^

The decoder is configured via ``configs/lmcache-decoder-config.yaml``:

.. code-block:: yaml

   local_cpu: False
   max_local_cpu_size: 0

   enable_pd: True
   transfer_channel: "nixl"
   pd_role: "receiver"
   pd_peer_host: "localhost"
   pd_peer_init_port: 7300
   pd_peer_alloc_port: 7400
   pd_buffer_size: 2147483648 # 2GB
   pd_buffer_device: "cuda"
   nixl_backends: [UCX]

Key settings:

- ``pd_role: "receiver"`` - Configures this instance to receive KV cache data
- ``pd_buffer_size: 2147483648 # 2GB`` - Buffer size for transfers
- ``pd_buffer_device: "cuda"`` - Uses GPU memory for buffering

Components Deep Dive
~~~~~~~~~~~~~~~~~~~~

Proxy Server (disagg_proxy_server.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The proxy server coordinates the disaggregated prefill workflow:

1. **Request Handling**: Receives client requests on port 9100
2. **Prefill Coordination**: Sends requests to the prefiller with ``max_tokens=1``
3. **Prefill Response**: Receives prefiller that says nixl transfer is done
4. **Response Streaming**: Streams the full response from the decoder
5. **Performance Monitoring**: Tracks Time-To-First-Token (TTFT) statistics

Supported endpoints:

- ``/v1/completions``
- ``/v1/chat/completions``

vLLM Server Launcher (disagg_vllm_launcher.sh)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script launches individual vLLM servers with appropriate configurations:

**Prefiller Launch Command**:

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

**Decoder Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_CONFIG_FILE=$decode_config_file \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=1 \
      vllm serve $MODEL \
      --port 7200 \
      --disable-log-requests \
      --enforce-eager \
      --no-enable-prefix-caching \
      --kv-transfer-config \
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}'

Testing and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

Basic Test
^^^^^^^^^^

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

This benchmark:
- Sends requests to port 9100 (proxy server)
- Uses random prompts with 7500 input tokens
- Generates 200 output tokens per request
- Tests with 30 total prompts at 1 request/second

Log Files and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

The example generates three log files for monitoring:

- ``prefiller.log`` - Prefiller server logs and errors
- ``decoder.log`` - Decoder server logs and errors  
- ``proxy.log`` - Proxy server logs and TTFT statistics

The proxy server automatically calculates and displays TTFT statistics every 5 seconds:

.. code-block::

   ===============================
   Num requests: 10
   Prefill node TTFT stats:
    - Average (ms): 45.2
    - Median (ms): 43.1
    - 99th Percentile (ms): 52.8
   ===============================

Troubleshooting
~~~~~~~~~~~~~~~

Common Issues
^^^^^^^^^^^^^

1. **GPU Memory**: Ensure each GPU has sufficient memory for the model
2. **NIXL Installation**: Verify NIXL is properly installed and accessible
3. **Port Conflicts**: Check that ports 7100, 7200, and 9000 are available
4. **HF Token**: Ensure your Hugging Face token has access to Llama models

Error Recovery
^^^^^^^^^^^^^^

If any server fails to start:

1. Check the corresponding log file for error details
2. Verify GPU availability with ``nvidia-smi``
3. Ensure all dependencies are installed
4. Try restarting with ``Ctrl+C`` followed by re-running the script
