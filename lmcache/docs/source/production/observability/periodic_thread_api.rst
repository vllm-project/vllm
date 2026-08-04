.. _observability_periodic_thread_api:

Periodic Thread Monitoring API
==============================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


This document describes the Periodic Thread Monitoring API feature introduced
in LMCache.

Overview
--------

The Periodic Thread Monitoring API provides HTTP endpoints to monitor and
inspect the status of all periodic background threads running in the LMCache
system. This is useful for debugging, health checking, and operational
monitoring.

API Endpoints
-------------

Three API endpoints are available under the internal API server:

GET /periodic-threads
~~~~~~~~~~~~~~~~~~~~~

Returns information about all registered periodic threads.

**Query Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``level``
     - (none)
     - Filter by thread level (``critical``, ``high``, ``medium``, ``low``)
   * - ``running_only``
     - ``false``
     - Only show running threads
   * - ``active_only``
     - ``false``
     - Only show active threads

**Response Example:**

.. code-block:: json

    {
      "summary": {
        "total_count": 5,
        "running_count": 3,
        "active_count": 3,
        "by_level": {
          "critical": {"total": 1, "running": 1, "active": 1},
          "high": {"total": 2, "running": 1, "active": 1},
          "medium": {"total": 1, "running": 1, "active": 1},
          "low": {"total": 1, "running": 0, "active": 0}
        }
      },
      "threads": [
        {
          "name": "pin_monitor",
          "level": "high",
          "is_running": true,
          "is_active": true,
          "last_run_time": 1706000000.0,
          "last_run_ago": "2.5s"
        }
      ]
    }

GET /periodic-threads/{thread_name}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns detailed information about a specific periodic thread.

**Response Example:**

.. code-block:: json

    {
      "name": "pin_monitor",
      "level": "high",
      "is_running": true,
      "is_active": true,
      "last_run_time": 1706000000.0,
      "last_run_ago": "2.5s",
      "interval": 1.0,
      "total_runs": 100,
      "failed_runs": 0
    }

GET /periodic-threads-health
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quick health check for periodic threads. Returns whether all critical
and high-level threads are active.

**Response Example:**

.. code-block:: json

    {
      "healthy": true,
      "unhealthy_count": 0,
      "unhealthy_threads": []
    }

Thread Levels
-------------

Periodic threads are categorized by importance level:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Description
   * - ``critical``
     - Essential for system operation
   * - ``high``
     - Important for performance
   * - ``medium``
     - Standard background tasks
   * - ``low``
     - Optional/auxiliary tasks

The health check endpoint specifically monitors ``critical`` and ``high``
level threads to determine system health.

IrrecoverableException Handling
-------------------------------

Periodic threads now properly handle ``IrrecoverableException``. When such an
exception is raised during thread execution:

- The exception is logged with full traceback
- The thread run is marked as failed
- The thread **stops its execution loop** instead of continuing

This prevents threads from endlessly retrying operations that cannot succeed.

Usage Examples
--------------

Check Overall Thread Health
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8080/periodic-threads-health

List All Running Threads
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl "http://localhost:8080/periodic-threads?running_only=true"

Get Critical Threads Only
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl "http://localhost:8080/periodic-threads?level=critical"

Check Specific Thread Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8080/periodic-threads/pin_monitor
