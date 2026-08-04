.. _observability_frontend:

LMCache Frontend
================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/frontend_dashboard`.


LMCache Frontend is a monitoring and proxy service for LMCache clusters,
providing a web interface for cluster management and HTTP request proxying
to cluster nodes.

.. image:: https://raw.githubusercontent.com/LMCache/lmcache_frontend/main/res/img.png
   :alt: LMCache Frontend Dashboard
   :align: center


Features
--------

- **Cluster Monitoring**: Web-based dashboard for visualizing cluster status
- **Request Proxying**: HTTP proxy service to forward requests to any cluster node
- **Flexible Configuration**: Support for both IP:port and Unix domain sockets
- **Plugin System**: Integration with LMCache plugin framework


Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install lmcache_frontend

Or install from source:

.. code-block:: bash

   git clone https://github.com/LMCache/lmcache_frontend.git
   cd lmcache_frontend
   pip install -e .


Quick Start
-----------

Starting the Service
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   lmcache-frontend --port 8080 --host 0.0.0.0


Command Line Options
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Option
     - Description
   * - ``--port``
     - Service port (default: 8000)
   * - ``--host``
     - Bind host address (default: 0.0.0.0)
   * - ``--config``
     - Path to configuration file
   * - ``--nodes``
     - Direct node configuration (JSON string)


After starting the service, access the dashboard at ``http://localhost:8080/``.


Configuration
-------------

Node Configuration
^^^^^^^^^^^^^^^^^^

Create a ``config.json`` file with node definitions:

.. code-block:: json

   [
     {
       "name": "node1",
       "host": "127.0.0.1",
       "port": "/tmp/lmcache_internal_api_server/socket/9090"
     },
     {
       "name": "node2",
       "host": "127.0.0.1",
       "port": "/tmp/lmcache_internal_api_server/socket/9091"
     }
   ]

The ``port`` field can be configured as either an integer port number or
a string path for Unix domain sockets.


LMCache Plugin Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^

To start the frontend via the LMCache plugin framework,
add the following to your ``lmcache.yaml``:

.. code-block:: yaml

   extra_config:
     plugin.frontend.port: 8080
   internal_api_server_enabled: True
   internal_api_server_port_start: 9090
   plugin_locations: ["/scripts/scheduler_lmc_frontend_plugin.py"]
   internal_api_server_socket_path_prefix: "/tmp/lmcache_internal_api_server/socket"


Proxying Requests
-----------------

Proxy requests using the format:

.. code-block:: text

   /proxy/{target_host}/{target_port_or_socket}/{path}

Examples:

.. code-block:: bash

   # Proxy to a Unix socket
   curl "http://localhost:8080/proxy/localhost/%252Ftmp%252Flmcache_internal_api_server%252Fsocket_8081/metrics"

   # Proxy a POST request
   curl -X POST http://localhost:9090/proxy/localhost/8081/run_script \
       -F "script=@/root/scratch.py"


Contributing
------------

LMCache Frontend is an open-source project and we welcome contributions
from the community! Whether you want to fix bugs, add new features,
improve documentation, or share ideas, your contributions are appreciated.

Ways to contribute:

- **Report Issues**: Found a bug or have a feature request?
  Open an issue on GitHub.
- **Submit Pull Requests**: Fork the repository, make your changes,
  and submit a PR.
- **Improve Documentation**: Help us make the docs clearer and more helpful.
- **Share Feedback**: Let us know how you're using LMCache Frontend
  and what could be improved.

Join us in making LMCache Frontend better for everyone!


More Information
----------------

For more details, visit the
`LMCache Frontend GitHub repository <https://github.com/LMCache/lmcache_frontend>`_.
