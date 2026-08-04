.. _share_kv_cache:

Example: Share KV cache across multiple LLMs
============================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


LMCache should be able to reduce the generation time of the second and following calls.

We have examples for the following types of across-instance KV cache sharing:

- KV cache sharing through a centralized cache server: ``centralized_sharing``
- KV cache sharing through p2p cache transfer: ``p2p_sharing``

Prerequisites
-------------

Your server should have at least 2 GPUs.

For Centralized sharing, this will use the port 8000 and 8001 (for vLLM) and port 65432 (for LMCache).  

For P2P sharing: 

- `NIXL <https://github.com/ai-dynamo/nixl>`_ installed on the host.
- Port 8010 and 8011 for 2 vllms servers.
- Port 8200 and 8202 for 2 p2p initialization connections.
- Port 8201 and 8203 for 2 p2p lookup connections.
- Port 8300 for controller pull requests.
- Port 8400 for controller reply requests.
- Port 8500 and 8501 for 2 LMCache workers.
- Port 9000 for controller main port (arbitrary and can be changed) to start the controller.

Centralized KV cache sharing
----------------------------

This section demonstrates how to share KV cache across multiple vLLM instances using a centralized LMCache server.

**Important**: For centralized cache sharing (which is cross-process cases), ensure all processes use the same `PYTHONHASHSEED` to keep the hash of the KV cache consistent across processes: ``export PYTHONHASHSEED=0``.

Setup centralized sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create a configuration file named ``lmcache_config.yaml`` with the following content:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    remote_url: "lm://localhost:65432"
    remote_serde: "cachegen"

Run centralized sharing example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Start the LMCache centralized server,

.. code-block:: bash

    lmcache_server localhost 65432

2. In a different terminal,

.. code-block:: bash

    PYTHONHASHSEED=0 \
    LMCACHE_CONFIG_FILE=lmcache_config.yaml \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8000 --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

In another terminal,

.. code-block:: bash

    PYTHONHASHSEED=0 \
    LMCACHE_CONFIG_FILE=lmcache_config.yaml \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8001 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Wait until both engines are ready.

3.  Send one request to the engine at port 8000,

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "prompt": "Explain the significance of KV cache in language models.",
            "max_tokens": 10
        }'

4. Send the same request to the engine at port 8001,

.. code-block:: bash

    curl -X POST http://localhost:8001/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "prompt": "Explain the significance of KV cache in language models.",
            "max_tokens": 10
        }'

The second request will automatically retrieve and reuse the KV cache from the first instance, significantly reducing generation time.

P2P KV cache sharing
--------------------

This section demonstrates how to share KV cache across multiple vLLM instances using peer-to-peer transfer.

Configure LMCache instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create two configuration files for the P2P sharing setup. The values that differ between the files are the ``lmcache_instance_id`` and the P2P/controller port assignments.

Instance 1 configuration (``p2p_example1.yaml``):

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    enable_async_loading: True

    # P2P configurations
    enable_p2p: true
    p2p_host: "localhost"
    p2p_init_ports: 8200
    p2p_lookup_ports: 8201
    transfer_channel: "nixl"

    # Controller configurations
    enable_controller: true
    lmcache_instance_id: "lmcache_instance_1"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8500

    extra_config:
      lookup_backoff_time: 0.001

Instance 2 configuration (``p2p_example2.yaml``):

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    enable_async_loading: True

    # P2P configurations
    enable_p2p: true
    p2p_host: "localhost"
    p2p_init_ports: 8202
    p2p_lookup_ports: 8203
    transfer_channel: "nixl"

    # Controller configurations
    enable_controller: true
    lmcache_instance_id: "lmcache_instance_2"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8501

    extra_config:
      lookup_backoff_time: 0.001

Save both files in the directory that you will mount into the container (referenced later as ``$YAML_FILES``).

Run the P2P sharing workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Configure the environment on the host and open a shell inside the container:

.. code-block:: bash

    docker pull vllm/vllm-openai:latest
    export WEIGHT_DIR="/models"          # model weights directory
    export CONTAINER_NAME="lmcache_vllm" # container name
    export YAML_FILES="/path/to/yaml"    # directory containing the YAML files
    docker run --name "$CONTAINER_NAME" \
            --detach \
            --ipc=host \
            --network host \
            --gpus all \
            --volume "$WEIGHT_DIR:$WEIGHT_DIR" \
            --volume "$YAML_FILES:$YAML_FILES" \
            --entrypoint "/bin/bash" \
            vllm/vllm-openai:latest -c "time sleep 452d"
    docker exec -it "$CONTAINER_NAME" /bin/bash
    pip install -U lmcache # update lmcache to the latest version

2. Start the LMCache controller and monitoring endpoints:

.. code-block:: bash

    PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'

3. Launch two vLLM engines, each with its own LMCache worker configuration.

Start vLLM engine 1 on GPU 0:

.. code-block:: bash

    PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=p2p_example1.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8010 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start vLLM engine 2 on GPU 1:

.. code-block:: bash

    PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=p2p_example2.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8011 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

4. Populate the KV cache by sending a request to the first engine:

.. code-block:: bash

    curl -X POST http://localhost:8010/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

5. Send the same request to the second engine to demonstrate cache retrieval:

.. code-block:: bash

    curl -X POST http://localhost:8011/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

Expected output
~~~~~~~~~~~~~~~

When the second request successfully retrieves the cache from the first instance, the logs should include entries similar to:

.. code-block:: bash

    (EngineCore_DP0 pid=305) [2025-11-16 07:24:11,522] LMCache INFO: Got layout info from controller: ('lmcache_instance_2', 'LocalCPUBackend', 3, 'localhost:8202') (p2p_backend.py:196:lmcache.v1.storage_backend.p2p_backend)                                                  
    (EngineCore_DP0 pid=305) [2025-11-16 07:24:11,607] LMCache INFO: Established connection to peer_init_url localhost:8202. The peer_lookup_url: localhost:8203 (p2p_backend.py:349:lmcache.v1.storage_backend.p2p_backend)                                                      
    (EngineCore_DP0 pid=305) [2025-11-16 07:24:11,706] LMCache INFO: Responding to scheduler for lookup id cmpl-e9ec2875bf954bd298ca26d14e083b80-0 with retrieved length 768 (storage_manager.py:531:lmcache.v1.storage_backend.storage_manager)                                  
    (EngineCore_DP0 pid=305) [2025-11-16 07:24:11,708] LMCache INFO: Reqid: cmpl-e9ec2875bf954bd298ca26d14e083b80-0, Total tokens 1002, LMCache hit tokens: 768, need to load: 768 (vllm_v1_adapter.py:1330:lmcache.integration.vllm.vllm_v1_adapter)                             
    (EngineCore_DP0 pid=305) [2025-11-16 07:24:11,724] LMCache INFO: Retrieved 768 out of 768 required tokens (from 768 total tokens). size: 0.0938 gb, cost 7.9816 ms, throughput: 11.7458 GB/s; (cache_engine.py:531:lmcache.v1.cache_engine)


These logs indicate that the peer connection was established and the cache was transferred successfully.
