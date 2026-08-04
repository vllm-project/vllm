LMCache Controller
==================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


Overview
--------
The overall architecture of the LMCache Controller is shown in the figure,
mainly consisting of two parts: the Controller Manager and LMCache Worker.

The Controller Manager mainly consists of KV Controller, Reg Controller, and Cluster Executor.

- KV Controller: The KV Controller handles the chunk information reported by LMCache Workers, and lookup requests query chunk information from the KV Controller.
- Reg Controller: The Reg Controller is responsible for handling register/deregister/heartbeat requests from LMCache Workers.
- Cluster Executor: When the Controller Manager receives user requests, such as Clear or Move, it sends the corresponding commands to LMCache Workers through the Cluster Executor.

The LMCache Worker is a thread within a rank process, which is responsible for the following tasks:

- sends register, deregister, heartbeat to the Reg Controller.
- send chunk information to the KV Controller, which include admit and evict message.
- listens on a port to receive commands from the Cluster Executor and performs corresponding processing.

.. image:: /assets/lmcache-controller.png
    :alt: LMCache Controller Architecture Diagram

P2P Related
^^^^^^^^^^^

If ``enable_p2p`` is enabled, the LMCache Controller must also be enabled. The LMCache Controller serves as the central node
and stores information for each chunk. The ``P2PBackend`` queries chunk information from the LMCache Controller and performs
data transmission through NIXL.


Key Features
------------

1. Exposes a set of APIs for users and orchestrators to manage the KV cache.

Currently, the controller provides the following APIs:

- :ref:`Clear <clear>`: Clear the KV caches.
- :ref:`Compress <compress>`: Compress the KV cache.
- :ref:`Health <health>`: Check the health status of cache workers.
- :ref:`Lookup <lookup>`: Lookup the KV cache for a given list of tokens.
- :ref:`Move <move>`: Move the KV cache to a different location.
- :ref:`Pin <pin>`: Persist the KV cache to prevent it from being evicted.
- :ref:`CheckFinish <check_finish>`: Check whether a (non-blocking) control event has finished or not.
- :ref:`QueryWorkerInfo <query_worker_info>`: Query the worker info.

2. Interacts with the LMCache worker.

Currently, the LMCache worker supports the following functions:

- register with the controller
- deregister from the controller
- heartbeat
- admit or evict chunk information(LocalCPUBackend or LocalDiskBackend)


Quick Start
-----------

**Start the Controller**

.. code-block:: bash

    python3 -m lmcache.v1.api_server

Expected output:

.. code-block:: text

    [2025-11-11 11:15:35,277] LMCache WARNING: Argument --monitor-port will be deprecated soon. Please use --monitor-ports instead. (__main__.py:361:__main__)
    INFO 11-11 11:15:36 [__init__.py:239] Automatically detected platform cuda.
    /usr/local/lib/python3.12/dist-packages/pydantic/_internal/_fields.py:198: UserWarning: Field name "copy" in "create_app.<locals>.MoveRequest" shadows an attribute in parent "BaseModel"
    warnings.warn(
    [2025-11-11 11:15:37,956] LMCache INFO: Starting LMCache controller at 0.0.0.0:9000 (__main__.py:371:__main__)
    [2025-11-11 11:15:37,956] LMCache INFO: Monitoring lmcache workers at ports None (__main__.py:372:__main__)
    INFO:     Started server process [50664]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)

**Controller Configuration**

- --host: default is 0.0.0.0
- --port: default is 9000, the externally exposed port through which interfaces like lookup can be accessed via this port.
- --monitor-port: default is 9001,  the port through which LMCache Worker communicates with Controller Manager (deprecated, indicates the pull port in --monitor-ports, reply port is None).
- --monitor-ports: default is None, if configured, requires a JSON format string input such as ``{"pull": 8300, "reply": 8400}``.

**YAML Configuration**

.. code-block:: yaml

    enable_controller: True
    lmcache_instance_id: "lmcache_instance_id"

    controller_pull_url: ip:pull_port
    # if controller reply port is None, no need to configure reply url
    controller_reply_url: ip:reply_port
    # the number of ports for LMCache Worker, must equal to the number of ranks
    lmcache_worker_ports: [1, 2, 3]

    # p2p configuration
    p2p_host: localhost
    p2p_init_ports: [11, 12, 13]

.. toctree::
   :maxdepth: 1
   :hidden:

   clear
   compress
   health
   lookup
   move
   pin
   check_finish
   query_worker_info
