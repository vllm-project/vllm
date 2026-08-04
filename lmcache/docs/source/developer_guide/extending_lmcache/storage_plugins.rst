Storage Plugins
===============

LMCache supports out of the box storage backends like Mooncake, S3 and NIXL.
The LMCache storage plugin system provides the ability to add custom storage backends through dynamic loading or plug and play capability. In other words, extending cache storage capabilities without modifying core code.

Backend Definition Requirements
-------------------------------
1. Inherit from ``StoragePluginInterface``
2. Implement all the abstract methods of the parent interface of ``StoragePluginInterface``- ``StorageBackendInterface``
3. Package as an installable Python module

.. note::

  The interface constructor is the instantiation contract that the LMCache loading system will use when loading custom storage backends.
  If you wish to implement a constructor, it should have the same parameter signature and call the interface constructor.

How to Integrate the Backend with LMCache
-----------------------------------------
1. Install your backend package in the LMCache environment
2. Add ``storage_plugins`` and its related ``module_path`` and ``class_name`` to ``extra_config`` section of LMCache configuration as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    storage_plugins: <backend_name>
    extra_config:
      storage_plugin.<backend_name>.module_path: <module_path>
      storage_plugin.<backend_name>.class_name: <class_name>

An example configuration for a logging backend is as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    storage_plugins: "log_backend"
    extra_config:
      storage_plugin.log_backend.module_path: lmc_external_log_backend.lmc_external_log_backend
      storage_plugin.log_backend.class_name: ExternalLogBackend

.. note::

   - Storage backends are initialized in order during LMCache startup - earlier backends have higher priority during cache lookups
   - ``storage_plugin.<backend_name>`` distinguishes the different dynamic loaded backends

Backend Implementation Example
------------------------------
A sample custom backend implementation can be viewed at https://github.com/opendataio/lmc_external_log_backend/


MP-Mode L2 Adapter Plugins (``plugin``)
----------------------------------------

.. _plugin-l2-adapter-overview:

The storage plugin system described above applies to **non-MP mode** (single-process). For
**MP mode** (multiprocess), LMCache provides the ``plugin`` L2 adapter type, which dynamically
loads a third-party ``L2AdapterInterface`` implementation at runtime.

Overview
~~~~~~~~

The ``plugin`` adapter type lets you ship a full L2 adapter as a **separate, pip-installable
package**. At startup, LMCache imports your module, instantiates your adapter class, and uses it
just like a built-in adapter -- no LMCache source modifications required.

.. note::

   If your storage backend is a native C++ connector (pybind-wrapped), consider using the
   ``native_plugin`` type instead (see :doc:`native_connectors`). It reuses the built-in
   bridging logic so you only need to implement 6 connector methods rather than the full
   ``L2AdapterInterface``.

Configuration
~~~~~~~~~~~~~

.. code-block:: json

    {
      "type": "plugin",
      "module_path": "my_plugin.adapter",
      "class_name": "MyL2Adapter",
      "adapter_params": {
        "host": "localhost",
        "capacity": 1000
      }
    }

.. list-table:: ``PluginL2AdapterConfig`` fields
   :header-rows: 1
   :widths: 22 10 10 58

   * - Field
     - Type
     - Required
     - Description
   * - ``module_path``
     - ``str``
     - yes
     - Dotted Python import path of the module containing the adapter class.
   * - ``class_name``
     - ``str``
     - yes
     - Name of the class inside ``module_path`` that implements ``L2AdapterInterface``.
   * - ``adapter_params``
     - ``dict``
     - no
     - Forwarded to the adapter class constructor (as a typed config or raw dict).
   * - ``config_class_name``
     - ``str``
     - no
     - Explicit config class name; when omitted the factory auto-discovers it.

Config Class Auto-Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The factory automatically resolves the adapter's config class using the following priority
chain (first match wins):

1. **Explicit** ``config_class_name`` field in JSON config.
2. **Convention**: adapter class name + ``"Config"`` suffix (e.g. ``MyL2Adapter`` looks for
   ``MyL2AdapterConfig``).
3. **Attribute**: ``config_class_name`` attribute on the adapter class itself.
4. **Fallback**: no config class found -- pass raw ``adapter_params`` dict.

Each candidate is looked up in the loaded module and validated as an ``L2AdapterConfigBase``
subclass. This means most plugins that follow the naming convention will **automatically**
receive a typed config instance without any extra configuration.

Plugin Contract
~~~~~~~~~~~~~~~

A plugin adapter class **must**:

1. Subclass ``L2AdapterInterface`` from ``lmcache.v1.distributed.l2_adapters.base``.
2. Implement all abstract methods: ``submit_store_task``, ``pop_completed_store_tasks``,
   ``submit_lookup_and_lock_task``, ``query_lookup_and_lock_result``, ``submit_unlock``,
   ``submit_load_task``, ``query_load_result``, ``close``, and all three event-fd getters.
3. Provide **three distinct event fds** (store / lookup / load). The controllers build
   ``fd -> adapter`` maps; duplicates will misroute events.
4. Be **thread-safe**: ``StoreController`` and ``PrefetchController`` call adapter methods
   from different threads concurrently.
5. Accept ``**kwargs`` in ``__init__`` to stay forward-compatible.

A plugin adapter class **should**:

1. Create its own asyncio event loop and background thread if it needs async I/O.
2. Use ``create_event_notifier()`` from ``lmcache.v1.platform`` for the three event fds
   (cross-platform: ``os.eventfd`` on Linux, ``os.pipe`` fallback elsewhere).
3. Clean up all resources (event fds, threads, connections) in ``close()``.

Loading Flow
~~~~~~~~~~~~

.. code-block:: text

    CLI / config JSON
      |
      v
    PluginL2AdapterConfig.from_dict(d)
      |  validates module_path, class_name, adapter_params
      |
      v
    _create_plugin_adapter(config, ...)
      |
      +-- importlib.import_module(config.module_path)
      +-- getattr(module, config.class_name)
      +-- issubclass check against L2AdapterInterface
      |
      +-- _resolve_config_class(module, config, adapter_cls)
      |   +-- 1. config.config_class_name (explicit)
      |   +-- 2. class_name + "Config" (convention)
      |   +-- 3. adapter_cls.config_class_name (attribute)
      |   +-- 4. None (fall back to raw dict)
      |
      +-- [if config class found]
      |   +-- adapter_cls(cfg_cls.from_dict(adapter_params))
      |
      +-- [otherwise]
          +-- adapter_cls(adapter_params)
              |
              v
      L2AdapterInterface instance (ready for use)

Minimal Example
~~~~~~~~~~~~~~~

.. code-block:: python

    # my_plugin/adapter.py
    import asyncio
    import threading

    from lmcache.native_storage_ops import Bitmap
    from lmcache.v1.distributed.l2_adapters.base import (
        L2AdapterInterface,
        L2TaskId,
    )
    from lmcache.v1.platform import create_event_notifier


    class MyL2Adapter(L2AdapterInterface):
        def __init__(self, params, **_kw):
            self._store_efd = create_event_notifier()
            self._lookup_efd = create_event_notifier()
            self._load_efd = create_event_notifier()
            # ... set up connection, background thread, etc.

        # implement all abstract methods ...

        def close(self) -> None:
            self._store_efd.close()
            self._lookup_efd.close()
            self._load_efd.close()

Launch via CLI:

.. code-block:: bash

    --l2-adapter '{
      "type": "plugin",
      "module_path": "my_plugin.adapter",
      "class_name": "MyL2Adapter",
      "adapter_params": {"host": "localhost"}
    }'

Reference Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/lmc_external_l2_adapter/`` for a complete, pip-installable example plugin
(``InMemoryL2Adapter``) that demonstrates:

- FIFO eviction with configurable capacity.
- Simulated bandwidth delay for realistic testing.
- Background asyncio event loop with proper shutdown.
- Full test suite covering store, lookup, load, batch operations, and eviction behavior.

Additional Resources
~~~~~~~~~~~~~~~~~~~~

- Plugin adapter source: ``lmcache/v1/distributed/l2_adapters/plugin_l2_adapter.py``
- Native plugin adapter: ``lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py``
- Design document: ``lmcache/v1/distributed/l2_adapters/design_docs/plugin.md``
- L2 adapter base interface: ``lmcache/v1/distributed/l2_adapters/base.py``
