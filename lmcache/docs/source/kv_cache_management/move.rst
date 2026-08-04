.. _move:

Move the KV cache
=================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The ``move`` interface is defined as the following:

.. code-block:: python

    move(old_position: Tuple[str, str], new_position: Tuple[str, str],
         tokens: Optional[List[int]] = [], copy: Optional[bool] = False) -> event_id: str, num_tokens: int

The function moves the KV cache chunks identified by ``tokens`` from
``old_position`` to ``new_position``. Each position is a tuple of
``(instance_id, location)``. Setting ``copy`` to ``True`` copies the
KV cache instead of moving it.

Note that NIXL is required to  be installed for P2P transfer. 
We'll support other transports later such as Python socket and Mooncake.

Example usage:
---------------------------------------

First, prepare two yaml files ``instance1.yaml`` and ``instance2.yaml`` to
configure two lmcache instances:

.. code-block:: yaml

    # instance1.yaml
    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 5

    # cache controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_1"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8500

    # P2P configurations
    enable_p2p: True
    p2p_host: "localhost"
    p2p_init_ports: 8200
    p2p_lookup_ports: 8201
    transfer_channel: "nixl"


.. code-block:: yaml

    # instance2.yaml
    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 5

    # cache controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_1"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8501

    # P2P configurations
    enable_p2p: True
    p2p_host: "localhost"
    p2p_init_ports: 8202
    p2p_lookup_ports: 8203
    transfer_channel: "nixl"

Start two vllm engines:

.. code-block:: bash

    PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=instance1.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 \
      --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

.. code-block:: bash

    PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=instance2.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 \
      --gpu-memory-utilization 0.8 --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start the lmcache controller at port 9000 and the monitor at port 9001:

.. code-block:: bash

    PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'

Send a request to vllm engine 1:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": "Explain the significance of KV cache in language models.",
            "max_tokens": 10
          }'

Tokenize the prompt to obtain token ids:

.. code-block:: bash

    curl -X POST http://localhost:8000/tokenize \
      -H "Content-Type: application/json" \
      -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": "Explain the significance of KV cache in language models."
          }'

Move the KV cache from engine 1's CPU to engine 2's CPU using the token ids:

.. code-block:: bash

    curl -X POST http://localhost:9000/move \
      -H "Content-Type: application/json" \
      -d '{
            "old_position": ["lmcache_instance_1", "LocalCPUBackend"],
            "new_position": ["lmcache_instance_2", "LocalCPUBackend"],
            "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
          }'

The controller responds with a message similar to:

.. code-block:: text

    {"num_tokens": 12, "event_id": "xxx"}

``num_tokens`` indicates how many tokens' KV cache are being moved. The
returned ``event_id`` can be used to query the status of the operation.
