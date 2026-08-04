.. _observability_health_monitor:

Health Monitor
==============

LMCache includes a comprehensive health monitoring framework that continuously monitors the health of the cache engine and its components. This feature is essential for production deployments to detect and respond to failures in remote storage backends.

Overview
--------

The Health Monitor provides:

- **Automatic health checks**: Periodically monitors the health of all registered components
- **Extensible framework**: Easily add custom health checks for new components
- **Remote backend monitoring**: Built-in support for monitoring remote storage backends via ping
- **Degraded mode support**: Automatically blocks operations when the system is unhealthy
- **Prometheus metrics integration**: Health status exposed via metrics endpoint

Architecture
------------

The health monitoring system consists of three main components:

1. **HealthCheck (Abstract Base Class)**
   
   Base class for individual health checks. Each health check represents one aspect of system health.

2. **HealthMonitor**
   
   The central monitor that orchestrates all health checks. It runs in a background thread and periodically executes all registered health checks.

3. **RemoteBackendHealthCheck**
   
   Built-in health check for remote storage backends. It pings the remote connector to verify connectivity.

Auto-Discovery
--------------

The Health Monitor uses an auto-discovery mechanism to find and instantiate health checks:

1. At startup, the monitor scans the ``lmcache.v1.health_monitor.checks`` package
2. All classes that inherit from ``HealthCheck`` are discovered
3. Each check's ``create_from_engine()`` method is called to create instances
4. The instances are registered with the monitor

This design allows you to add new health checks by simply creating a new module in the checks package.

Configuration
-------------

Health monitor configuration is done through the ``extra_config`` section of your LMCache configuration:

.. list-table:: Health Monitor Configuration Options
   :header-rows: 1
   :widths: 40 20 100

   * - Configuration Key
     - Default Value
     - Description
   * - ``ping_interval``
     - ``30.0``
     - Interval (in seconds) between health check cycles
   * - ``ping_timeout``
     - ``5.0``
     - Timeout (in seconds) for each ping operation
   * - ``get_blocking_failed_threshold``
     - ``10``
     - Max number of get_blocking failed count in check interval
   * - ``waiting_time_for_recovery``
     - ``300.0``
     - Waiting time (in seconds) for recovery if get_blocking failed


How It Works
------------

Runtime Behavior
~~~~~~~~~~~~~~~~

The health monitor runs in a background thread:

1. Every ``ping_interval`` seconds, all health checks are executed
2. If any check fails, the system is marked as unhealthy
3. When unhealthy, store/retrieve operations are blocked with a warning log
4. Once all checks pass again, the system is marked as healthy and operations resume

Initialization Failure Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When initialization or post-initialization fails irrecoverably:

1. The system is marked with ``_init_failed = True``
2. ``is_healthy()`` method returns ``False`` permanently
3. Health monitoring thread will not start (if initialization fails before it starts)
4. The system operates in degraded mode (recompute-only)

This ensures that irrecoverable initialization errors don't cause cascading failures
and the system can gracefully fall back to recomputation.

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

When the health monitor detects an unhealthy state:

- **Store operations**: Skipped with a warning message
- **Retrieve operations**: Return empty results with a warning message
- **Lookup operations**: Return 0 (no cache hits) with a warning message

This prevents cascading failures when remote backends are unavailable.

Built-in Health Checks
----------------------

RemoteBackendHealthCheck
~~~~~~~~~~~~~~~~~~~~~~~~

This check monitors the connectivity to remote storage backends (e.g., Redis, Valkey).

**What it checks:**

- Pings the remote connector to verify it's reachable
- Measures ping latency
- Reports error codes for failures

**When it's active:**

- Only when a remote backend is configured (``remote_url`` is set)
- Only if the connector supports the ``ping()`` operation

**Metrics reported:**

- ``lmcache:remote_ping_latency``: Latest ping latency (milliseconds)
- ``lmcache:remote_ping_error_code``: Latest error code (0 = success)
- ``lmcache:remote_ping_errors``: Total number of ping errors
- ``lmcache:remote_ping_successes``: Total number of successful pings

Prometheus Metrics
------------------

The health monitor exposes metrics through the Prometheus endpoint:

.. list-table:: Health Monitor Metrics
   :header-rows: 1
   :widths: 40 15 55

   * - Metric Name
     - Type
     - Description
   * - ``lmcache:is_healthy``
     - Gauge
     - Overall system health status (1 = healthy, 0 = unhealthy)
   * - ``lmcache:remote_ping_latency``
     - Gauge
     - Latest ping latency to remote backends (milliseconds)
   * - ``lmcache:remote_ping_error_code``
     - Gauge
     - Latest ping error code (0 = success, -1 = timeout, -2 = generic error)
   * - ``lmcache:remote_ping_errors``
     - Counter
     - Total number of ping errors to remote backends
   * - ``lmcache:remote_ping_successes``
     - Counter
     - Total number of successful pings to remote backends

Error Codes
-----------

The health check system uses the following error codes:

.. list-table:: Health Check Error Codes
   :header-rows: 1
   :widths: 20 80

   * - Code
     - Description
   * - ``0``
     - Success - the health check passed
   * - ``-1``
     - Timeout - the ping operation exceeded the configured timeout
   * - ``-2``
     - Generic error - an unexpected error occurred during the health check

Extending the Health Monitor
----------------------------

You can add custom health checks by creating a new module in the ``lmcache/v1/health_monitor/checks/`` directory.

The custom check will be automatically discovered and registered when LMCache starts.
