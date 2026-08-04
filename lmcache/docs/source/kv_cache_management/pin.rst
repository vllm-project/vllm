.. _pin:

Pin the KV cache
================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The ``pin`` interface is defined as the following:

.. code-block:: python

    pin(instance_id: str, location: str, tokens: List[int]) -> event_id: str, num_tokens: int

The function pins (persists) the KV cache chunks specified by ``tokens`` in the
given ``location`` of the ``instance_id``. The controller returns an ``event_id``
and the number of tokens scheduled for pinning.

Example usage:
---------------------------------------

First, create a yaml file ``example.yaml`` to configure the lmcache instance:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 5

    # cache controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_default_instance"
    controller_pull_url: "localhost:9001"
    lmcache_worker_ports: 8001

    # Peer identifiers
    p2p_host: "localhost"
    p2p_init_ports: 8200


Start the vllm/lmcache instance at port 8000:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 \
      --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start the lmcache controller at port 9000 and the monitor at port 9001:

.. code-block:: bash

    lmcache_controller --host localhost --port 9000 --monitor-port 9001

Send a request to vllm:

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

Pin the KV cache in the system:

.. code-block:: bash

    curl -X POST http://localhost:9000/pin \
      -H "Content-Type: application/json" \
      -d '{
            "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13],
            "instance_id": "lmcache_default_instance",
            "location": "LocalCPUBackend"
          }'

The controller responds with a message similar to:

.. code-block:: text

    {"event_id": "xxx", "num_tokens": 12}

``num_tokens`` indicates how many tokens' KV cache are pinned. The
returned ``event_id`` can be used to query the status of the operation.
