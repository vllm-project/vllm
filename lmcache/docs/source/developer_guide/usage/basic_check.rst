Basic Check Tool
================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The LMCache Basic Check Tool is a testing and validation utility that helps you verify your LMCache installation, configuration, and functionality. It provides multiple testing modes to validate different components of the LMCache system.

Overview
--------

The basic check tool (``lmcache.v1.basic_check``) is designed to:

* Test remote backend connectivity and functionality
* Validate storage manager operations
* Generate test keys for performance testing
* Verify configuration settings
* Provide diagnostic information for troubleshooting

Available Check Modes
---------------------

The tool supports several check modes, each targeting specific functionality:

test_remote
~~~~~~~~~~~

Tests the remote backend functionality including:

* Connection establishment to remote backends (fs, etc.)
* put/get operations with data integrity validation
* put/get/exists operations with performance reports

**Usage:**

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode test_remote

test_storage_manager
~~~~~~~~~~~~~~~~~~~~

Tests the storage manager operations including:

* Configuration validation
* batched_put/get operations with data integrity validation
* batched_put/get/contains operations with performance reports

**Usage:**

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode test_storage_manager

gen (Key Generation)
~~~~~~~~~~~~~~~~~~~~

Generates test keys for performance testing and benchmarking:

* Configurable number of keys and concurrency levels
* Memory-efficient batch processing
* Progress tracking and performance metrics
* Offset support for distributed testing

**Usage:**

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode gen --num-keys 1000 --concurrency 16

Command Line Interface
----------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode <MODE> [OPTIONS]

List Available Modes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode list

Command Line Options
~~~~~~~~~~~~~~~~~~~~

.. option:: --mode MODE

   **Required.** Operation mode to run. Use ``list`` to see available modes.

.. option:: --model MODEL

   Model name for testing, just a part of key of persist kv-cache. Default: ``/lmcache_test_model/``

.. option:: --num-keys NUM

   Number of keys to generate (gen mode only). Default: 100

.. option:: --concurrency NUM

   Concurrency level for operations (gen mode only). Default: 16

.. option:: --offset NUM

   Offset for key generation (gen mode only). Default: 0

Configuration
-------------

The basic check tool uses your existing LMCache configuration. You can specify configuration in several ways:

Environment Variable
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export LMCACHE_CONFIG_PATH=/path/to/config.yaml
   python -m lmcache.v1.basic_check --mode test_remote

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

Here's an example configuration optimized for basic checks:

.. code-block:: yaml

   # Basic cache settings
   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 1.0  # 1GB for basic checks

   # Remote backend (optional)
   remote_url: "file:///tmp/lmcache_basic_check"

Examples
--------

The ``examples/basic_check/`` directory contains comprehensive examples:
