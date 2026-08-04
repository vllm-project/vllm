.. _query_worker_info:

Query Worker Info
=================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The ``query_worker_info`` interface is defined as the following:

.. code-block:: python

    query_worker_info(instance_id: str, worker_ids: List[int]) -> event_id: str, worker_infos: List[WorkerInfo]

The function get the info of the workers which specified by ``instance_id`` and ``worker_ids``.
The controller returns an ``event_id`` and the worker infos.

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

Send a request to controller:

.. code-block:: bash

    curl -X POST http://localhost:9000/query_worker_info \
      -H "Content-Type: application/json" \
      -d '{
            "instance_id": "lmcache_default_instance",
            "worker_ids": [0]
          }'

The controller responds with a message similar to:

.. code-block:: text

    {"event_id": "xxx", "worker_infos": [{"instance_id": "lmcache_default_instance", "worker_id": 0, "ip": "127.0.0.1", "port": 8001, "peer_init_url": "127.0.0.1:8200", "registration_time": 123456, "last_heartbeat_time": 456789}]}

``worker_infos`` contains the queried worker information.
returned ``event_id`` can be used to query the status of the operation.
