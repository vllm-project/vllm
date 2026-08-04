Runtime Plugins
===============

The LMCache runtime plugin system provides the ability to extend functionality by running custom scripts alongside LMCache processes. Plugins can be written in Python and Bash for now, and are managed by the ``RuntimePluginLauncher`` class.

Key Use Cases
-------------
- Start metric reporters for centralized monitoring
- Implement log reporters for log collection systems
- Report process-level metrics to alerting systems
- Implement health checks and service discovery
- Custom cache management operations

Configuration
-------------
Runtime plugins are configured through environment variables and configuration files:

Environment Variables:
- ``LMCACHE_RUNTIME_PLUGIN_ROLE``: Process role (e.g., ``SCHEDULER``, ``WORKER``)
- ``LMCACHE_RUNTIME_PLUGIN_CONFIG``: JSON string containing plugin configuration
- ``LMCACHE_RUNTIME_PLUGIN_WORKER_ID``: Current worker ID
- ``LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT``: Total worker count in cluster

Configuration File (``lmcache.yaml``):

.. code-block:: yaml

   runtime_plugin_locations:
     - "/path/to/plugins"

   extra_config:
     custom_setting: value

Runtime Plugin Naming Convention
--------------------------------
Plugin filenames determine execution targets:

Role-Specific Plugins:
- Format: ``<ROLE>[_<WORKER_ID>][_<DESCRIPTION>].<EXTENSION>``
- Examples:

  - ``scheduler_foo_plugin.py``: Runs only on ``SCHEDULER``
  - ``worker_0_test.sh``: Runs only on worker ID 0
  - ``all_plugin.sh``: Runs on all workers

Notes:
- Role names are case-insensitive
- Worker ID must be numeric when specified
- To target a specific worker ID, the filename must have at least three parts separated by underscores (e.g., `worker_<ID>_<DESCRIPTION>.ext`). A file named `worker_<DESCRIPTION>.ext` will run on all workers.

Execution Model
---------------
1. **Interpreter Detection**:
   - Uses shebang line (e.g., ``#!/opt/venv/bin/python``)
   - Fallback interpreters:

     - ``.py`` → ``python``
     - ``.sh`` → ``bash``

2. **Output Handling**:
   - Stdout/stderr captured continuously
   - Logged with plugin name prefix

3. **Process Management**:
   - Launched as subprocesses
   - Terminated when parent process exits

Example Runtime Plugins
-----------------------
Python Plugin (``scheduler_foo_plugin.py``):

.. literalinclude:: ../../../../examples/plugins/scheduler_foo_plugin.py
   :language: python
   :linenos:

Bash Plugin (``all_plugin.sh``):

.. literalinclude:: ../../../../examples/plugins/all_plugin.sh
   :language: bash
   :linenos:

Best Practices
--------------
1. Keep runtime plugins lightweight and efficient
2. Use descriptive naming conventions
3. Implement graceful error handling
4. Include shebang for portability
5. Validate configuration inputs
6. Add timeout mechanisms for long operations