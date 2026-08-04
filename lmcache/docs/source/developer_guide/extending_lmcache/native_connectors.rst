Adding Native Backends
======================

.. _native-connectors-overview:

Overview
--------

Native connectors are high-performance C++ storage backends that integrate with LMCache
through pybind11. They work in **both** LMCache operating modes:

- **Non-MP mode** (single process): via ``ConnectorClientBase`` (asyncio integration)
- **MP mode** (multiprocess): via ``NativeConnectorL2Adapter`` (L2 adapter interface)

Write the connector once, get both modes for free.

The framework lives in ``csrc/storage_backends/`` with the Redis RESP connector as the
reference implementation.

Architecture
~~~~~~~~~~~~

.. code-block:: text

    Non-MP mode:
      CacheEngine -> RemoteBackend -> ConnectorClientBase -> native client (C++)
                                        (asyncio event loop)

    MP mode:
      StoreController / PrefetchController
            |
      NativeConnectorL2Adapter (Python bridge)
        +-- 3 eventfds (store, lookup, load)
        +-- completion demux thread
        +-- ObjectKey <-> string serialization
        +-- client-side lock tracking
            |
      native client (C++)
        +-- 1 eventfd, worker threads, GIL-free I/O

**Design principles:**

1. **GIL release** at the pybind layer for true concurrency between native threads
2. **Batching with tiling**: work for a batched request is split evenly among threads
3. **eventfd-based completions**: the kernel wakes Python -- no polling
4. **Non-blocking submission**: submission queue / completion queue architecture


Step 1: C++ Connector
---------------------

Create your connector directory (e.g., ``csrc/storage_backends/mybackend/``) and
inherit from ``ConnectorBase<YourConnectionType>``. You need to override 4 required methods
(and optionally ``do_single_delete`` to support eviction).

**connector.h:**

.. code-block:: cpp

    // csrc/storage_backends/mybackend/connector.h
    #pragma once
    #include "../connector_base.h"

    namespace lmcache {
    namespace connector {

    // Per-thread connection state
    struct MyConn {
      int fd = -1;
      // your connection fields
    };

    class MyConnector : public ConnectorBase<MyConn> {
     public:
      MyConnector(std::string host, int port, int num_workers)
          : ConnectorBase(num_workers), host_(host), port_(port) {
        start_workers();  // IMPORTANT: call at END of constructor
      }

     protected:
      // 1. Create a connection (called once per worker thread)
      MyConn create_connection() override {
        MyConn conn;
        // connect to server...
        return conn;
      }

      // 2. GET: read value for key into buf
      void do_single_get(MyConn& conn, const std::string& key,
                         void* buf, size_t len,
                         size_t chunk_size) override {
        // send GET command, recv response into buf
      }

      // 3. SET: write data from buf under key
      void do_single_set(MyConn& conn, const std::string& key,
                         const void* buf, size_t len,
                         size_t chunk_size) override {
        // send SET command with data from buf
      }

      // 4. EXISTS: check if key exists
      bool do_single_exists(MyConn& conn,
                            const std::string& key) override {
        // send EXISTS, return true/false
      }

      // 5. DELETE: remove key (optional, has default no-op)
      bool do_single_delete(MyConn& conn,
                            const std::string& key) override {
        // send DELETE, return true if deleted, false if not found
      }

      // Optional: clean shutdown
      void shutdown_connections() override {
        // close sockets, free resources
      }

     private:
      std::string host_;
      int port_;
    };

    }  // namespace connector
    }  // namespace lmcache

**What ConnectorBase gives you for free:**

- Worker thread pool with per-thread connections
- Submission queue (lock-free enqueue) and completion queue
- Automatic tiling: batch operations are split across workers
- eventfd signaling on completion (kernel wakes Python)
- Graceful shutdown (stop flag, drain, join)

.. important::
   Always call ``start_workers()`` at the **end** of your derived constructor,
   after all member variables are initialized. Worker threads call
   ``create_connection()`` immediately, so the object must be fully constructed.

**Reference:** ``csrc/storage_backends/redis/connector.h`` and ``connector.cpp``


Step 2: Pybind Module
---------------------

Use the ``LMCACHE_BIND_CONNECTOR_METHODS`` macro, which binds all 7 methods
(``event_fd``, ``submit_batch_get/set/exists/delete``, ``drain_completions``, ``close``)
with proper GIL release and Python buffer protocol handling.

.. code-block:: cpp

    // csrc/storage_backends/mybackend/pybind.cpp
    #include <pybind11/pybind11.h>
    #include "../connector_pybind_utils.h"
    #include "connector.h"

    namespace py = pybind11;

    PYBIND11_MODULE(lmcache_mybackend, m) {
      py::class_<lmcache::connector::MyConnector>(m, "LMCacheMyBackendClient")
          .def(py::init<std::string, int, int>(),
               py::arg("host"), py::arg("port"),
               py::arg("num_workers"))
          LMCACHE_BIND_CONNECTOR_METHODS(
              lmcache::connector::MyConnector);
    }

The pybind utilities automatically:

- Extract buffer pointers from Python ``memoryview`` objects under the GIL
- Release the GIL before calling into C++
- Convert C++ ``Completion`` structs to Python tuples ``(future_id, ok, error, result_bools)``

**Reference:** ``csrc/storage_backends/redis/pybind.cpp``


Step 3: Build System
--------------------

Register your C++ sources in ``setup.py`` alongside the existing Redis extension:

.. code-block:: python

    # In cuda_extension() and rocm_extension():
    mybackend_sources = [
        "csrc/storage_backends/mybackend/pybind.cpp",
        "csrc/storage_backends/mybackend/connector.cpp",
    ]

    # Add to ext_modules list:
    cpp_extension.CppExtension(
        "lmcache.lmcache_mybackend",
        sources=mybackend_sources,
        include_dirs=[
            "csrc/storage_backends",
            "csrc/storage_backends/mybackend",
        ],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    ),

Then rebuild:

.. code-block:: bash

    pip install -e .


Step 4: Python Client (Non-MP Mode)
------------------------------------

Inherit from ``ConnectorClientBase`` which provides asyncio event loop integration,
future management, and both sync and async methods.

.. code-block:: python

    # lmcache/v1/storage_backend/native_clients/mybackend_client.py
    from .connector_client_base import ConnectorClientBase
    from lmcache.lmcache_mybackend import LMCacheMyBackendClient

    class MyBackendClient(ConnectorClientBase[LMCacheMyBackendClient]):
        def __init__(self, host: str, port: int,
                     num_workers: int, loop=None):
            native = LMCacheMyBackendClient(host, port, num_workers)
            super().__init__(native, loop)

This gives you ``batch_get``, ``batch_set``, ``batch_exists`` (async), and their
synchronous variants, all with automatic eventfd-driven completion handling.

**Reference:** ``lmcache/v1/storage_backend/native_clients/resp_client.py``


Step 5: L2 Adapter (MP Mode)
-----------------------------

To use your connector as an L2 adapter in MP mode, create a single Python module that
defines the config class, factory function, and self-registers both. The
``NativeConnectorL2Adapter`` bridge handles all the complexity (eventfd demuxing,
key serialization, locking).

Create a new file in the L2 adapters package:

.. code-block:: python

    # lmcache/v1/distributed/l2_adapters/mybackend_l2_adapter.py
    from __future__ import annotations
    from typing import TYPE_CHECKING, Optional

    if TYPE_CHECKING:
        from lmcache.v1.distributed.internal_api import L1MemoryDesc

    from lmcache.v1.distributed.l2_adapters.base import (
        L2AdapterInterface,
    )
    from lmcache.v1.distributed.l2_adapters.config import (
        L2AdapterConfigBase,
        register_l2_adapter_type,
    )
    from lmcache.v1.distributed.l2_adapters.factory import (
        register_l2_adapter_factory,
    )


    class MyBackendL2AdapterConfig(L2AdapterConfigBase):
        def __init__(self, host: str, port: int,
                     num_workers: int = 8,
                     max_capacity_gb: float = 0):
            self.host = host
            self.port = port
            self.num_workers = num_workers
            self.max_capacity_gb = max_capacity_gb

        @classmethod
        def from_dict(cls, d: dict) -> "MyBackendL2AdapterConfig":
            host = d.get("host")
            if not isinstance(host, str) or not host:
                raise ValueError("host must be a non-empty string")
            port = d.get("port")
            if not isinstance(port, int) or port <= 0:
                raise ValueError("port must be a positive integer")
            num_workers = d.get("num_workers", 8)
            max_capacity_gb = d.get("max_capacity_gb", 0)
            return cls(host=host, port=port,
                       num_workers=num_workers,
                       max_capacity_gb=max_capacity_gb)

        @classmethod
        def help(cls) -> str:
            return (
                "MyBackend L2 adapter config fields:\n"
                "- host (str): server hostname (required)\n"
                "- port (int): server port (required)\n"
                "- num_workers (int): worker threads (default 8)"
            )


    def _create_mybackend_l2_adapter(
        config: L2AdapterConfigBase,
        l1_memory_desc: "Optional[L1MemoryDesc]" = None,
    ) -> L2AdapterInterface:
        from lmcache.lmcache_mybackend import LMCacheMyBackendClient
        from lmcache.v1.distributed.l2_adapters \
            .native_connector_l2_adapter import (
            NativeConnectorL2Adapter,
        )

        assert isinstance(config, MyBackendL2AdapterConfig)
        native_client = LMCacheMyBackendClient(
            config.host, config.port, config.num_workers
        )
        return NativeConnectorL2Adapter(
            native_client,
            max_capacity_gb=config.max_capacity_gb,
        )


    # Self-register -- runs automatically when the module
    # is imported by the L2 adapter auto-discovery mechanism
    register_l2_adapter_type("mybackend", MyBackendL2AdapterConfig)
    register_l2_adapter_factory("mybackend", _create_mybackend_l2_adapter)

.. note::
   The L2 adapter package uses ``pkgutil.iter_modules`` to auto-discover all modules
   in ``lmcache/v1/distributed/l2_adapters/``. Simply creating the file above is
   sufficient -- no changes to ``__init__.py`` or any other existing file are needed.

**Usage from the command line:**

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --l2-adapter '{"type": "mybackend", "host": "10.0.0.1", "port": 9000}'


How NativeConnectorL2Adapter Bridges the Gap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C++ connector has 1 eventfd and mixed completions. MP mode's ``L2AdapterInterface``
requires 3 separate eventfds and typed results. The bridge handles this transparently:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - L2 Adapter Method
     - Native Call
     - Extra Logic
   * - ``submit_store_task(keys, objs)``
     - ``submit_batch_set``
     - ObjectKey to string, MemoryObj to memoryview
   * - ``submit_lookup_and_lock_task(keys)``
     - ``submit_batch_exists``
     - + client-side lock refcount
   * - ``submit_load_task(keys, objs)``
     - ``submit_batch_get``
     - ObjectKey to string, MemoryObj to memoryview
   * - ``submit_unlock(keys)``
     - *(none)*
     - client-side lock decrement
   * - ``pop_completed_store_tasks()``
     - via ``drain_completions``
     - demux by op type
   * - ``query_lookup_and_lock_result()``
     - via ``drain_completions``
     - exists results to Bitmap, apply locks
   * - ``query_load_result()``
     - via ``drain_completions``
     - ok/fail to Bitmap

A background demux thread polls the native eventfd, calls ``drain_completions()``,
looks up each ``future_id`` to determine its operation type, routes the result to
the correct completion dict, and signals the corresponding Python eventfd.


Third-Party Native Connector Plugins (``native_plugin``)
---------------------------------------------------------

.. _native-plugin-overview:

The steps above describe how to add a native connector **inside** the LMCache source tree.
If you want to ship a connector as a **separate, pip-installable package** (e.g. a proprietary
storage backend), use the ``native_plugin`` L2 adapter type instead. It dynamically loads your
connector at runtime -- no LMCache source modifications required.

How It Works
~~~~~~~~~~~~

The ``native_plugin`` adapter type loads a third-party **connector object** (pybind-wrapped C++
or pure Python) and wraps it with the built-in ``NativeConnectorL2Adapter`` bridge. This means
you only need to implement the 6 connector methods -- the Python-side demux/lock bridging logic
is reused from LMCache.

.. list-table:: ``plugin`` vs ``native_plugin``
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - ``plugin``
     - ``native_plugin``
   * - What is loaded
     - A full ``L2AdapterInterface`` subclass
     - A **connector** object (lower level)
   * - Bridging logic
     - Provided by the plugin itself
     - Reused from ``NativeConnectorL2Adapter``
   * - Third-party effort
     - Must implement all abstract methods + 3 eventfds
     - Only 6 connector methods

Required Connector Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamically loaded connector instance must expose the following methods (identical to the
pybind ``LMCACHE_BIND_CONNECTOR_METHODS`` contract):

.. code-block:: python

    class NativeConnectorProtocol:
        def event_fd(self) -> int: ...
        def submit_batch_get(
            self,
            keys: list[str],
            memoryviews: list[memoryview],
        ) -> int: ...
        def submit_batch_set(
            self,
            keys: list[str],
            memoryviews: list[memoryview],
        ) -> int: ...
        def submit_batch_exists(
            self,
            keys: list[str],
        ) -> int: ...
        def submit_batch_delete(
            self,
            keys: list[str],
        ) -> int: ...
        def drain_completions(
            self,
        ) -> list[tuple[int, bool, str, list[bool] | None]]: ...
        def close(self) -> None: ...

The factory validates the first 6 methods at creation time and raises ``TypeError`` if
any are missing. ``submit_batch_delete`` is **optional** -- if absent, the adapter's
``delete()`` method will be a no-op (eviction will not remove keys from the backend).

Configuration
~~~~~~~~~~~~~

.. code-block:: json

    {
      "type": "native_plugin",
      "module_path": "my_ext_package",
      "class_name": "MyConnectorClient",
      "adapter_params": {
        "host": "localhost",
        "port": 1234
      }
    }

.. list-table:: ``NativePluginL2AdapterConfig`` fields
   :header-rows: 1
   :widths: 20 10 10 60

   * - Field
     - Type
     - Required
     - Description
   * - ``module_path``
     - ``str``
     - yes
     - Dotted Python import path of the module containing the connector class.
   * - ``class_name``
     - ``str``
     - yes
     - Name of the connector class inside ``module_path``.
   * - ``adapter_params``
     - ``dict``
     - no
     - Forwarded as ``**kwargs`` to the connector class constructor.
   * - ``max_capacity_gb``
     - ``float``
     - no
     - Maximum L2 storage capacity in GB for client-side usage tracking. Required for L2 eviction. Default 0 (disabled).

Loading Flow
~~~~~~~~~~~~

.. code-block:: text

    CLI / config JSON
      |
      v
    NativePluginL2AdapterConfig.from_dict(d)
      |
      v
    _create_native_plugin_l2_adapter(config, ...)
      |
      +-- importlib.import_module(config.module_path)
      +-- getattr(module, config.class_name)
      +-- connector_cls(**config.adapter_params)
      +-- validate 6 required methods
      +-- NativeConnectorL2Adapter(native_client)
              |
              v
      L2AdapterInterface instance (ready for use)

Step-by-Step: Building an External Native Connector Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create a Python package** with a C++ pybind11 extension that inherits from
   ``ConnectorBase<T>`` (same base class as built-in connectors).

   Project layout:

   .. code-block:: text

       my_ext_connector/
       +-- csrc/
       |   +-- connector.h      # C++ connector class
       |   +-- connector.cpp    # C++ implementation
       |   +-- pybind.cpp       # pybind11 bindings
       +-- src/
       |   +-- my_ext_connector/
       |       +-- __init__.py   # re-export the factory class
       |       +-- connector.py  # Python factory wrapper
       +-- pyproject.toml
       +-- setup.py              # build C++ extension

2. **Implement the C++ connector** inheriting from ``ConnectorBase<T>`` and override
   the 4 required methods (``create_connection``, ``do_single_get``, ``do_single_set``,
   ``do_single_exists``) and optionally ``do_single_delete`` for eviction support.

3. **Create pybind11 bindings** using the ``LMCACHE_BIND_CONNECTOR_METHODS`` macro:

   .. code-block:: cpp

       #include <pybind11/pybind11.h>
       #include "connector_pybind_utils.h"
       #include "connector.h"

       namespace py = pybind11;

       PYBIND11_MODULE(_native, m) {
         py::class_<MyFSConnector>(m, "MyFSConnector")
             .def(py::init<std::string, int>(),
                  py::arg("base_path"),
                  py::arg("num_workers"))
             LMCACHE_BIND_CONNECTOR_METHODS(MyFSConnector);
       }

4. **Write a Python factory class** that selects the backend and returns the native
   connector instance:

   .. code-block:: python

       from my_ext_connector._native import MyFSConnector

       class MyConnectorClient:
           def __new__(
               cls,
               base_path: str = "/tmp/my_ext",
               num_workers: int = 2,
           ):
               return MyFSConnector(base_path, num_workers)

5. **Build and install** the package:

   .. code-block:: bash

       cd my_ext_connector
       pip install -e .

6. **Configure LMCache** to use it:

   .. code-block:: bash

       --l2-adapter '{
         "type": "native_plugin",
         "module_path": "my_ext_connector",
         "class_name": "MyConnectorClient",
         "adapter_params": {
           "base_path": "/tmp/my_ext",
           "num_workers": 2
         }
       }'

Reference Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/lmc_external_native_connector/`` for a complete, pip-installable example
connector plugin that demonstrates:

- C++ pybind11-wrapped connectors inheriting from ``ConnectorBase<T>`` (same as built-in
  Redis/FS).
- Two backends: filesystem (``ExampleFSConnector``) and in-memory
  (``ExampleMemoryConnector``), both in C++.
- A thin Python factory class (``ExampleNativeConnector``) that selects the backend via a
  ``"backend"`` parameter.
- Worker thread pool with eventfd notification (inherited from ``ConnectorBase``).
- Build via ``pip install -e .`` using pybind11 + setuptools.


Checklist
---------

Use this checklist when adding a new native connector:

1. C++ connector inheriting ``ConnectorBase<T>`` with 4 required + 1 optional (``do_single_delete``) method overrides
2. Pybind module using ``LMCACHE_BIND_CONNECTOR_METHODS``
3. ``setup.py`` entry for the new ``CppExtension``
4. Python client inheriting ``ConnectorClientBase`` (non-MP mode)
5. L2 adapter module with config class + factory self-registration (MP mode)
6. Unit tests (see ``tests/v1/distributed/test_native_connector_l2_adapter.py``)
7. Rebuild with ``pip install -e .`` and verify both modes work

For **external** native connector plugins (``native_plugin``):

1. Separate pip-installable package with C++ pybind11 extension
2. Connector class exposing the 6 required methods (+ optional ``submit_batch_delete`` for eviction)
3. Python factory class for backend selection
4. ``pip install -e .`` and configure via ``--l2-adapter`` JSON
5. Unit tests (see ``examples/lmc_external_native_connector/tests/``)


Built-in Aerospike backend (optional)
-------------------------------------

LMCache ships an optional in-tree Aerospike native connector (same
``ConnectorBase`` harness as Redis). It is compiled only when
``BUILD_AEROSPIKE=1`` or ``AEROSPIKE_INCLUDE_DIR`` is set during
``pip install -e .``.

**Build:**

.. code-block:: bash

   See ``.github/workflows/aerospike_integration.yml`` for installing the C client into ``.deps/``
   source .deps/aerospike-client-c.env
   BUILD_AEROSPIKE=1 pip install -e .

The ``aerospike-client-c.env`` file simply points the build at the C client
headers and shared libraries you unpacked into ``.deps/``. Adjust the paths to
match where the client was installed. Example:

.. code-block:: bash

   # .deps/aerospike-client-c.env
   export AEROSPIKE_INCLUDE_DIR="${PWD}/.deps/aerospike-install/usr/include"
   export AEROSPIKE_LIBRARY_DIR="${PWD}/.deps/aerospike-install/usr/lib"
   # Needed at build and runtime so the loader can find libaerospike (and
   # libyaml if you built it locally):
   export LD_LIBRARY_PATH="${AEROSPIKE_LIBRARY_DIR}:${PWD}/.deps/libyaml-install/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

Setting ``AEROSPIKE_INCLUDE_DIR`` is enough to enable the extension, so
``BUILD_AEROSPIKE=1`` is optional once the env file is sourced. Multiple include
or library directories can be passed as ``;``-separated lists.

**MP mode:**

.. code-block:: bash

   --l2-adapter '{"type": "aerospike", "hosts": "127.0.0.1:3000", "namespace": "lmcache", "set_name": "kv_chunks", "num_workers": 8}'

Implementation: ``csrc/storage_backends/aerospike/``,
``lmcache/v1/distributed/l2_adapters/aerospike_l2_adapter.py``.


Additional Resources
--------------------

- Framework source: ``csrc/storage_backends/``
- ``ConnectorBase`` template: ``csrc/storage_backends/connector_base.h``
- ``IStorageConnector`` interface: ``csrc/storage_backends/connector_interface.h``
- Pybind utilities: ``csrc/storage_backends/connector_pybind_utils.h``
- Redis reference implementation: ``csrc/storage_backends/redis/``
- Aerospike implementation (optional): ``csrc/storage_backends/aerospike/``
- Architecture README: ``csrc/storage_backends/README.md``
- External native connector example:
  ``examples/lmc_external_native_connector/``
- Native plugin adapter source:
  ``lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py``
- Design document:
  ``lmcache/v1/distributed/l2_adapters/design_docs/plugin.md``
- RESP backend user guide: :doc:`RESP (Native Redis/Valkey) <../../kv_cache/storage_backends/resp>`
