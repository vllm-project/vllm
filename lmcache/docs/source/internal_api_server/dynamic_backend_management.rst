.. _dynamic_backend_management:

Dynamic Backend Management
==========================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/http_api`.


LMCache provides a set of internal API endpoints that allow you to **list**,
**close**, and **create** storage backends at runtime without restarting the
serving engine.  This is useful when you need to switch between different
storage configurations on the fly — for example, migrating from a
``LocalDiskBackend`` to a ``GdsBackend``, or changing the remote connector
from a filesystem connector to Redis.

Overview
--------

The workflow for dynamically switching a storage backend is:

1. **Close** the backend you want to replace.
2. **Update** the relevant configuration via the ``POST /conf`` API.
3. **Create** new backends — only backends that are not already present
   will be created.

Any backend that was **not** closed will be skipped during creation,
so the operation is safe and idempotent.

API Endpoints
-------------

``GET /backends``
^^^^^^^^^^^^^^^^^

List all active storage backends.

.. code-block:: bash

    curl http://localhost:7000/backends

Response:

.. code-block:: json

    {
      "LocalCPUBackend": "LocalCPUBackend",
      "RemoteBackend": "RemoteBackend"
    }

``DELETE /backends/{backend_name}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Close and remove a specific storage backend.  After this call the backend
is fully shut down and removed from the internal dictionary, ensuring no
stale references remain.

.. code-block:: bash

    curl -X DELETE http://localhost:7000/backends/RemoteBackend

Response:

.. code-block:: json

    {
      "status": "success",
      "message": "Backend RemoteBackend closed",
      "backends": {
        "LocalCPUBackend": "LocalCPUBackend"
      }
    }

``POST /backends``
^^^^^^^^^^^^^^^^^^

Create new storage backends based on the current ``LMCacheEngineConfig``.
Existing backends are skipped.

.. code-block:: bash

    curl -X POST http://localhost:7000/backends

Response:

.. code-block:: json

    {
      "status": "success",
      "created": {
        "RemoteBackend": "RemoteBackend"
      },
      "backends": {
        "LocalCPUBackend": "LocalCPUBackend",
        "RemoteBackend": "RemoteBackend"
      }
    }

Use-Case Examples
-----------------

Switching from ``LocalDiskBackend`` to ``GdsBackend``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you originally configured a local-disk backend and want to migrate to
NVIDIA GPUDirect Storage (GDS) at runtime:

.. code-block:: bash

    # 1. Close the old disk backend
    curl -X DELETE http://localhost:7000/backends/LocalDiskBackend

    # 2. Disable local_disk and set the GDS path
    curl -X POST http://localhost:7000/conf \
      -H "Content-Type: application/json" \
      -d '{
        "local_disk": false,
        "gds_path": "/mnt/nvme/lmcache_gds"
      }'

    # 3. Create the new GDS backend
    curl -X POST http://localhost:7000/backends

    # 4. Verify the new backend list
    curl http://localhost:7000/backends

Switching ``RemoteBackend`` connector (FS → Redis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To change the remote connector type without restarting:

.. code-block:: bash

    # 1. Close the old remote backend
    curl -X DELETE http://localhost:7000/backends/RemoteBackend

    # 2. Update the remote URL to point to Redis
    curl -X POST http://localhost:7000/conf \
      -H "Content-Type: application/json" \
      -d '{
        "remote_url": "redis://redis-host:6379"
      }'

    # 3. Create the new remote backend with Redis connector
    curl -X POST http://localhost:7000/backends

    # 4. Verify
    curl http://localhost:7000/backends

Replacing only one backend while keeping others
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If only the ``RemoteBackend`` needs to be updated but the
``LocalCPUBackend`` should stay untouched:

.. code-block:: bash

    # Close only the remote backend
    curl -X DELETE http://localhost:7000/backends/RemoteBackend

    # Update config
    curl -X POST http://localhost:7000/conf \
      -H "Content-Type: application/json" \
      -d '{"remote_url": "redis://new-redis:6379"}'

    # Create — LocalCPUBackend is skipped (already present)
    curl -X POST http://localhost:7000/backends

Notes
-----

- Closing the ``LocalCPUBackend`` is possible but should be done with
  caution since many other backends rely on it as an intermediate buffer.
- The ``POST /backends`` endpoint calls the same ``CreateStorageBackends``
  factory that is used during engine initialization, so all backend types
  (local CPU, local disk, GDS, remote, P2P, plugins, etc.) are
  supported.
- After creating backends, the ``StorageManager`` automatically refreshes
  its internal references (``non_allocator_backends``,
  ``local_cpu_backend``, etc.).
