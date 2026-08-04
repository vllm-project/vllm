Maru
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _maru-overview:

Overview
--------

`Maru <https://github.com/xcena-dev/maru>`_ is a high-performance KV cache storage engine built on CXL shared memory,
designed for LLM inference scenarios where multiple instances need to share a KV cache with minimal latency.

.. image:: ../../assets/maru-kvcache.png
    :alt: KV Cache Sharing: Without vs With Maru

For architecture details, see the `Maru documentation <https://xcena-dev.github.io/maru/>`_.

Quick Start
-----------

Install Maru:

.. code-block:: bash

    git clone https://github.com/xcena-dev/maru.git
    cd maru
    ./install.sh

This installs ``maru-server``, ``maru-resourced``, and the ``maru`` Python package.

Deploy Model With Maru
~~~~~~~~~~~~~~~~~~~~~~

**Prerequisites:** CXL device (``/dev/dax*``), Python 3.12+, vLLM and LMCache installed.

**1. Start the Maru Server**

.. code-block:: bash

    maru-server

**2. Create configuration file** (``maru-config.yaml``):

.. code-block:: yaml

    chunk_size: 256
    local_cpu: False
    max_local_cpu_size: 0
    save_unfull_chunk: True

    # Maru backend
    maru_path: "maru://localhost:5555"
    maru_pool_size: 4

**3. Start vLLM with Maru**

.. code-block:: bash

    LMCACHE_CONFIG_FILE="maru-config.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 65536 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Configuration
-------------

**LMCache Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``maru_path``
     - Required
     - Maru server URL (format: ``maru://host:port``)
   * - ``maru_pool_size``
     - ``4.0``
     - CXL memory pool size per instance in GB (e.g., ``4``, ``0.5``)

**Advanced Parameters (via extra_config):**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``maru_instance_id``
     - auto UUID
     - Unique client instance identifier
   * - ``maru_timeout_ms``
     - 5000
     - ZMQ RPC socket timeout in milliseconds
   * - ``maru_use_async_rpc``
     - true
     - Async DEALER-ROUTER RPC (``false`` for synchronous REQ-REP)
   * - ``maru_max_inflight``
     - 64
     - Max concurrent async RPC requests
   * - ``maru_eager_map``
     - true
     - Pre-map all shared regions on connect

Additional Resources
--------------------

- `Maru GitHub Repository <https://github.com/xcena-dev/maru>`_
- `Maru Documentation <https://xcena-dev.github.io/maru/>`_
