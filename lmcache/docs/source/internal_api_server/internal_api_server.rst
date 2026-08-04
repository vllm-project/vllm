.. _internal_api_server:

Internal API Server
===================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/http_api`.


The ``internal_api_server`` provides HTTP APIs for managing and inspecting
the LMCache engine at runtime. APIs are organized into three categories:

- **Common APIs** — Available across all components (scheduler, worker, controller).
- **vLLM / Inference APIs** — Specific to vLLM inference workers.
- **Controller APIs** — Specific to the LMCache Controller.

.. toctree::
   :maxdepth: 2

   common_apis
   vllm_apis
   controller_apis


Configuration
-------------

The following parameters can be configured in the YAML file:

.. code-block:: yaml

    # Enable/disable the internal API server
    internal_api_server_enabled: True
    # Base port for the API server
    # actual_port = internal_api_server_port_start + index
    # Scheduler → 6999 + 0 = 6999
    # Worker 0 → 6999 + 1 = 7000
    internal_api_server_port_start: 6999
    # List of scheduler/worker indices: 0 for scheduler, 1 for worker 0, 2 for worker 1, etc.
    internal_api_server_include_index_list: [0, 1]
    # Socket path prefix for the API server. If configured, the server will use a Unix socket instead of listening on a port.
    internal_api_server_socket_path_prefix: "/tmp/lmcache_internal_api_server/socket"

    # Actual socket files will be:
    #   /tmp/lmcache_internal_api_server/socket_6999 (scheduler)
    #   /tmp/lmcache_internal_api_server/socket_7000 (worker 0)


Port Assignment
^^^^^^^^^^^^^^^

The port for each component is computed as:

.. code-block:: text

    actual_port = internal_api_server_port_start + port_offset

Where ``port_offset`` is:

- ``0`` for the Scheduler
- ``1 + worker_id`` for Workers (e.g., Worker 0 → offset 1, Worker 1 → offset 2)


API Category & Route Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The server uses ``APIRegistry`` to automatically discover and register
API endpoint modules. Any file named ``*_api.py`` under
``lmcache/v1/internal_api_server/{common,vllm,controller}/`` that
exports a ``router = APIRouter()`` will be automatically included.


Extending the Server
^^^^^^^^^^^^^^^^^^^^^

To add a new API endpoint:

1. Create a new file in the appropriate category directory
   (``common/``, ``vllm/``, or ``controller/``).
2. Name the file with ``_api.py`` suffix (e.g., ``my_feature_api.py``).
3. Define ``router = APIRouter()`` and add your endpoints.

The endpoint will be automatically discovered and registered on the
next server startup.
