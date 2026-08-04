.. _health:

Check controller health
=======================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The ``health`` interface is defined as the following:

.. code-block:: python

    health(instance_id: str) -> event_id: str, error_codes: Dict[int, int]

The function returns an ``event_id`` and a dictionary mapping ``worker_id`` to
``error_code``. A value of ``0`` indicates a healthy worker, while a non-zero
value signals an error.

Example usage:
---------------------------------------

First, start the lmcache controller at port 9000 and the monitor at port 9001:

.. code-block:: bash

    PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-port 9001

Then send a health check request:

.. code-block:: bash

    curl -X POST http://localhost:9000/health \
      -H "Content-Type: application/json" \
      -d '{"instance_id": "lmcache_default_instance"}'

The controller responds with a message similar to the following:

.. code-block:: json

    {"event_id": "health47ce328d-f27e-48ae-ab0c-c2218aabce95", "error_codes": {"0": 0, "1": 0}}

Here ``error_codes`` lists each worker's ``error_code``. ``0`` represents a healthy
worker, while non-zero values indicate an error.
