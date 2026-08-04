lmcache ping
============

The ``lmcache ping`` command is a liveness check for an LMCache KV cache
server or a vLLM serving engine.

.. code-block:: bash

   # Ping the KV cache server (default: http://localhost:8080)
   lmcache ping kvcache

   # Ping the serving engine (default: http://localhost:8000)
   lmcache ping engine --url http://localhost:8000

.. code-block:: text

   ======= Ping KV Cache ========
   Status:                   OK
   Round trip time (ms):     3.42
   ==============================

``ping kvcache`` checks the ``/healthcheck`` endpoint; ``ping engine`` checks
``/health``.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``kvcache`` | ``engine``
     - Target to ping (positional, required).
   * - ``--url``
     - Server URL. Defaults to ``http://localhost:8080`` for ``kvcache``,
       ``http://localhost:8000`` for ``engine``.
   * - ``--format``
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - Save metrics to a file (format follows ``--format``).
   * - ``-q`` / ``--quiet``
     - Suppress stdout output. Exit code only.

JSON Output
-----------

.. code-block:: bash

   lmcache ping kvcache --format json

.. code-block:: json

   {
     "title": "Ping KV Cache",
     "metrics": {
       "status": "OK",
       "round_trip_time_ms": 3.42
     }
   }

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Server is reachable (HTTP 200).
   * - ``1``
     - Connection failure or non-200 response.
