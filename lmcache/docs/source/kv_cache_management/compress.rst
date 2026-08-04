.. _compress:

Compress and Decompress the KV cache
=====================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The ``compress`` interface is defined as the following:

.. code-block:: python

    compress(instance_id: str, method: str, location: str, tokens: list[int]) -> event_id: str, num_tokens: int
    decompress(instance_id: str, method: str, location: str, tokens: list[int]) -> event_id: str, num_tokens: int

These 2 functions compresses/decompresses the KV cache chunks specified by ``tokens`` using the
given ``method`` in the storage ``location``. The controller returns an ``event_id`` and the number of tokens scheduled for compression or decompression.

Example usage:
---------------------------------------

First, we need a yaml file ``example.yaml`` to properly configure the lmcache instance:

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

Second, we need to start the vllm/lmcache instance at port 8000:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Third, we need to start the lmcache controller at port 9000 and the monitor at port 9001:

.. code-block:: bash

    lmcache_controller --host localhost --port 9000 --monitor-port 9001

Then we can send a request to vllm to see if it works properly:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models.",
        "max_tokens": 10
      }'

Now we send a request to tokenize the prompt:

.. code-block:: bash

    curl -X POST http://localhost:8000/tokenize \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models."
      }'

We should be able to see token ids in response:

.. code-block:: text

    {"count":12,"max_model_len":4096,"tokens":[128000,849,21435,279,26431,315,85748,6636,304,4221,4211,13],"token_strs":null}

After all, we issue a ``compress`` request:

.. code-block:: bash

    curl -X POST http://localhost:9000/compress \
      -H "Content-Type: application/json" \
      -d '{
          "instance_id": "lmcache_default_instance",
          "method": "cachegen",
          "location": "LocalCPUBackend",
          "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
      }'

The controller responds with a message similar to:

.. code-block:: text

    {"event_id": "xxx", "num_tokens": 12}

This indicates that 12 tokens are being compressed. The ``event_id`` can be used to query the status of the operation.

Once the kv cache is compressed, we can use cachegen to decompress

.. code-block:: bash

    curl -X POST http://localhost:9000/decompress \
      -H "Content-Type: application/json" \
      -d '{
          "instance_id": "lmcache_default_instance",
          "method": "cachegen",
          "location": "LocalCPUBackend",
          "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
      }'

The controller responds with a message similar to:

.. code-block:: text

    {"event_id": "xxx", "num_tokens": 12}

This indicates that 12 tokens are being decompressed. The ``event_id`` can be used to query the status of the operation.

