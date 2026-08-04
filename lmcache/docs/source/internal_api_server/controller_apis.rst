.. _controller_apis:

Controller APIs
===============

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/http_api`.


These APIs are specific to the LMCache Controller component. They provide
visibility into registered instances, workers, and key statistics.

.. contents:: Endpoints
   :local:
   :depth: 2


``GET /controller/key-stats`` — Key Statistics
------------------------------------------------

Get key statistics across all instances and workers.

- **Method**: ``GET``
- **Path**: ``/controller/key-stats``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:6999/controller/key-stats

**Example Response** (HTTP 200):

.. code-block:: json

    {
      "total_key_count": 1500,
      "total_instance_count": 2,
      "total_worker_count": 4,
      "instances": [
        {
          "instance_id": "instance_001",
          "key_count": 800,
          "worker_count": 2
        },
        {
          "instance_id": "instance_002",
          "key_count": 700,
          "worker_count": 2
        }
      ]
    }

**Error Response** (controller not available, HTTP 503):

.. code-block:: json

    {
      "detail": "Controller manager not available"
    }

**Response Schema**:

  ========================= ======= =============================================
  Field                     Type    Description
  ========================= ======= =============================================
  ``total_key_count``       int     Total number of KV keys across all instances
  ``total_instance_count``  int     Total number of registered instances
  ``total_worker_count``    int     Total number of workers across all instances
  ``instances``             list    Per-instance breakdown (see below)
  ========================= ======= =============================================

**Instance Schema**:

  ================= ======= =============================================
  Field             Type    Description
  ================= ======= =============================================
  ``instance_id``   str     Unique identifier of the instance
  ``key_count``     int     Number of KV keys held by this instance
  ``worker_count``  int     Number of workers in this instance
  ================= ======= =============================================


``GET /controller/workers`` — Worker Information
--------------------------------------------------

Get worker information with flexible query parameters. Behavior depends
on the combination of parameters:

- **No parameters**: List all registered workers across all instances.
- **``instance_id`` only**: List all workers for a specific instance.
- **``instance_id`` and ``worker_id``**: Get detailed info about a specific worker.

- **Method**: ``GET``
- **Path**: ``/controller/workers``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``instance_id``  str     (Optional) Instance ID to filter workers
  ``worker_id``    int     (Optional) Worker ID for specific worker details (requires ``instance_id``)
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    # List all workers
    curl http://localhost:6999/controller/workers

    # List workers for a specific instance
    curl "http://localhost:6999/controller/workers?instance_id=instance_001"

    # Get a specific worker
    curl "http://localhost:6999/controller/workers?instance_id=instance_001&worker_id=0"

**Example Response** (list workers):

.. code-block:: json

    {
      "workers": [
        {
          "instance_id": "instance_001",
          "worker_id": 0,
          "ip": "10.0.0.1",
          "port": 8000,
          "peer_init_url": "http://10.0.0.1:8000/init",
          "registration_time": 1706745600.0,
          "last_heartbeat_time": 1706745660.0,
          "key_count": 400
        },
        {
          "instance_id": "instance_001",
          "worker_id": 1,
          "ip": "10.0.0.2",
          "port": 8001,
          "peer_init_url": "http://10.0.0.2:8001/init",
          "registration_time": 1706745600.0,
          "last_heartbeat_time": 1706745660.0,
          "key_count": 400
        }
      ],
      "total_count": 2
    }

**Example Response** (single worker):

.. code-block:: json

    {
      "instance_id": "instance_001",
      "worker_id": 0,
      "ip": "10.0.0.1",
      "port": 8000,
      "peer_init_url": "http://10.0.0.1:8000/init",
      "registration_time": 1706745600.0,
      "last_heartbeat_time": 1706745660.0,
      "key_count": 400
    }

**Error Response** (worker not found, HTTP 404):

.. code-block:: json

    {
      "detail": "Worker (instance_001, 99) not found"
    }

**Error Response** (instance not found, HTTP 404):

.. code-block:: json

    {
      "detail": "No workers found for instance unknown_instance"
    }

**Error Response** (controller not available, HTTP 503):

.. code-block:: json

    {
      "detail": "Controller manager not available"
    }

**Worker Response Schema**:

  ========================== ======= =============================================
  Field                      Type    Description
  ========================== ======= =============================================
  ``instance_id``            str     Instance this worker belongs to
  ``worker_id``              int     Worker index within the instance
  ``ip``                     str     Worker IP address
  ``port``                   int     Worker port number
  ``peer_init_url``          str     (Optional) Peer initialization URL
  ``registration_time``      float   Unix timestamp of worker registration
  ``last_heartbeat_time``    float   Unix timestamp of last heartbeat
  ``key_count``              int     Number of KV keys held by this worker
  ========================== ======= =============================================
