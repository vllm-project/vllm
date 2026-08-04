Extending the HTTP API
======================

You can add new endpoints to the ``lmcache server`` HTTP frontend **without
modifying any existing code**. An endpoint is just a Python module placed in
``lmcache/v1/multiprocess/http_apis/`` that exposes a FastAPI ``APIRouter``;
``HTTPAPIRegistry`` auto-discovers and mounts it at startup -- the same
zero-modification pattern used by the :doc:`L2 adapters
</mp/l2_storage/index>`.

How discovery works
-------------------

At startup, ``http_server.py`` hands the FastAPI app to ``HTTPAPIRegistry``
(``lmcache/v1/multiprocess/http_api_registry.py``), which scans the
``http_apis/`` directory with ``pkgutil``, imports every module whose name ends
with ``_api``, and includes any module-level ``router``. The built-in modules
follow this pattern:

.. list-table::
   :header-rows: 1
   :widths: 28 20 12 40

   * - Module
     - Endpoint
     - Method
     - Description
   * - ``info_api.py``
     - ``/``
     - GET
     - Basic liveness check
   * - ``info_api.py``
     - ``/healthcheck``
     - GET
     - Kubernetes probe endpoint
   * - ``cache_api.py``
     - ``/cache/clear``
     - POST
     - Force-clear the L1 cache
   * - ``info_api.py``
     - ``/status``
     - GET
     - Internal status report

Adding an endpoint
------------------

Create a file in ``lmcache/v1/multiprocess/http_apis/`` whose name ends with
``_api.py`` and expose a ``router``:

.. code-block:: python

   # lmcache/v1/multiprocess/http_apis/metrics_api.py
   # SPDX-License-Identifier: Apache-2.0
   from fastapi import APIRouter, Request
   from fastapi.responses import JSONResponse

   router = APIRouter()


   @router.get("/metrics")
   async def metrics(request: Request):
       """Return cache hit/miss metrics."""
       engine = getattr(request.app.state, "engine", None)
       if engine is None:
           return JSONResponse(
               status_code=503,
               content={"error": "engine not initialized"},
           )
       return {"hits": 42, "misses": 7}

That's it -- ``HTTPAPIRegistry`` discovers and mounts it on the next server
startup; no other file needs to change.

Module contract
---------------

An API module **must**:

- live in ``lmcache/v1/multiprocess/http_apis/`` with a filename ending in
  ``_api.py``;
- expose a module-level ``router`` of type ``fastapi.APIRouter``.

An API module **should**:

- guard against uninitialized state by checking ``request.app.state.engine``
  and returning ``503`` when it is ``None``;
- use ``lmcache.logging.init_logger(__name__)`` for logging;
- use ``async`` handlers and avoid blocking I/O.

An API module **must not** import or mutate the ``app`` object from
``http_server.py``.

Accessing shared state
----------------------

``app.state`` is the shared context populated during server startup. Reach it
through the request object:

.. code-block:: python

   @router.get("/my-endpoint")
   async def my_endpoint(request: Request):
       engine = request.app.state.engine          # main cache engine
       zmq_server = request.app.state.zmq_server   # underlying ZMQ server
       ...

For the full design rationale see
``docs/design/v1/multiprocess/http_api_extension.md`` in the source tree.
