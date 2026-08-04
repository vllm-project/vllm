.. _observability_internal_api_server:

Internal API Server Metrics
====================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


Another approach to retrieve LMCache metrics is to use the internal API server.

Overview
--------

The internal API server exposes Prometheus-compatible metrics endpoints in your LMCache deployment.

Quick Start Guide
-----------------

Step 1: Enable Internal API Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure your vLLM instance to enable the internal API server:

.. code-block:: bash

    LMCACHE_INTERNAL_API_SERVER_ENABLED=true \
    vllm serve $model \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Step 2: Access Metrics Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve metrics from the worker's endpoint:

.. code-block:: bash

    curl http://$IP:7000/metrics

Port Configuration
------------------

The following environment variables are used implicitly with their default values:

.. list-table:: Default Port Configuration
   :header-rows: 1
   :widths: 40 20 100

   * - Environment Variable
     - Default Value
     - Description
   * - ``LMCACHE_INTERNAL_API_SERVER_HOST``
     - ``0.0.0.0``
     - Host address for the internal API server to bind to.
   * - ``LMCACHE_INTERNAL_API_SERVER_PORT_START``
     - ``6999``
     - Starting port number, e.g.:

       - Scheduler: port_start + 0 (6999)
       - Worker 0: port_start + 1 (7000)
       - Worker 1: port_start + 2 (7001)


Therefore, the metrics endpoint curl command above uses port 7000. 

Advanced Usage
--------------

For comprehensive testing and configuration options, refer to :ref:`testing_internal_api_server` for detailed examples and best practices.
