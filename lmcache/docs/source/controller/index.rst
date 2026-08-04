Controller WebUI
===============

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/coordinator`.


Overview
--------

The LMCache Controller provides a web-based dashboard for monitoring your LMCache instances. This web interface allows you to monitor system status, instances, workers, and performance metrics.

Quick Start
-----------

To enable the Controller WebUI, start your LMCache instance with the following command:

.. code-block:: bash

    python3 -m lmcache.v1.api_server \
        --host 0.0.0.0 \
        --port 9000 \
        --monitor-ports '{"pull":8300,"reply":8400}' \
        --lmcache-worker-timeout 100 \
        --health-check-interval 10

After starting the controller, access the WebUI at:

.. code-block:: text

    http://localhost:9000/

Configuration Options
-------------------

- ``--host``: Bind address for the API server (default: 0.0.0.0)
- ``--port``: Port for the API server (default: 9000)
- ``--monitor-ports``: ZMQ ports for controller communication
- ``--lmcache-worker-timeout``: Worker timeout in seconds
- ``--health-check-interval``: Health check interval in seconds

Dashboard Features
------------------

The Controller Dashboard provides:

- System overview and health monitoring
- Instance and worker management
- Performance metrics
- Thread information
- Environment variables inspection

Related Documentation
--------------------

- :doc:`../api_reference/configurations` - Complete configuration reference
- :doc:`../kv_cache_management/index` - KV cache management guide
- :doc:`freeze_mode` - Freeze Mode safety mechanism guide