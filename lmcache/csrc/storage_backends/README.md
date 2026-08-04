# Native Storage Backends

Build high-performance storage connectors in C++/Rust that work in **both**
LMCache modes:

- **Non-MP mode** (single process): via `ConnectorClientBase` (asyncio integration)
- **MP mode** (multiprocess): via `NativeConnectorL2Adapter` (L2 adapter interface)

Write the connector once, get both modes for free.

```
Non-MP mode:
  CacheEngine → RemoteBackend → ConnectorClientBase → native client (C++)
                                  (asyncio event loop)

MP mode:
  StoreController / PrefetchController
        ↓
  NativeConnectorL2Adapter (Python bridge)
    ├─ 3 eventfds (store, lookup, load)
    ├─ completion demux thread
    ├─ ObjectKey ↔ string serialization
    └─ client-side lock tracking
        ↓
  native client (C++)
    └─ 1 eventfd, worker threads, GIL-free I/O
```

## Design principles

There are two sources of overhead in a Python integration:

1. **Submission**: the submitting Python thread shouldn't block, and we
   should make as few submissions to the event loop as possible.

2. **Completion**: we shouldn't poll for completions, and we should have
   as few completions as possible.

Therefore the framework enforces:

1. GIL release at pybind layer for true concurrency between native threads
2. Batching with tiling (work for a batched request split evenly among threads)
3. eventfd-based non-polling completions (the kernel wakes up Python)
4. Non-blocking submission (submission queue / completion queue architecture)

## Files

| File | Purpose |
|------|---------|
| `connector_types.h` | `Request`, `Completion`, `BatchState`, `Op` |
| `connector_interface.h` | `IStorageConnector` — top-level abstract interface |
| `connector_base.h` | `ConnectorBase<T>` — core harness (eventfd, SQ/CQ, threading, tiling). Override 4 required + 1 optional method per backend |
| `connector_pybind_utils.h` | Pybind utilities with GIL release + `LMCACHE_BIND_CONNECTOR_METHODS` macro |
| `redis/` | Reference implementation (RESP2 protocol over TCP) |
| `aerospike/` | Optional native Aerospike backend (meta + segment sharding; `BUILD_AEROSPIKE=1`) |

## Aerospike (optional build)

The Aerospike connector is **not** built by default. Enable it when packaging or
developing:

Set ``AEROSPIKE_INCLUDE_DIR`` and ``AEROSPIKE_LIBRARY_DIR`` to a libaerospike
development install (or use ``BUILD_AEROSPIKE=1`` after placing headers/libs under
``.deps/`` as in ``.github/workflows/aerospike_integration.yml``), then:

```bash
BUILD_AEROSPIKE=1 pip install -e .
```

MP mode:

```bash
--l2-adapter '{"type": "aerospike", "hosts": "127.0.0.1:3000", "namespace": "lmcache", "set_name": "kv_chunks", "num_workers": 8}'
```

Config module: ``lmcache/v1/distributed/l2_adapters/aerospike_l2_adapter.py``.

## How to add a new native backend

There are 5 steps. The Redis connector is the reference implementation for
each step.

### Step 1: C++ connector — inherit from ConnectorBase

Create your connector directory (e.g., `csrc/storage_backends/mybackend/`)
and inherit from `ConnectorBase<YourConnectionType>`. You need to
override 4 required methods (and optionally `do_single_delete` for eviction):

```cpp
// csrc/storage_backends/mybackend/connector.h
#include "../connector_base.h"

struct MyConn {
  int fd = -1;
  // your per-thread connection state
};

class MyConnector : public lmcache::connector::ConnectorBase<MyConn> {
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

  // 2. GET: read value for key into buf (buf has chunk_size bytes)
  void do_single_get(MyConn& conn, const std::string& key,
                     void* buf, size_t len, size_t chunk_size) override {
    // send GET, recv response into buf
  }

  // 3. SET: write chunk_size bytes from buf under key
  void do_single_set(MyConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override {
    // send SET with data from buf
  }

  // 4. EXISTS: check if key exists
  bool do_single_exists(MyConn& conn, const std::string& key) override {
    // send EXISTS, return true/false
  }

  // Optional: delete a key (enables eviction support)
  bool do_single_delete(MyConn& conn, const std::string& key) override {
    // send DELETE, return true if deleted, false if not found
  }

  // Optional: clean shutdown of connections
  void shutdown_connections() override { /* close sockets */ }

 private:
  std::string host_;
  int port_;
};
```

**Reference:** `redis/connector.{h,cpp}`

What `ConnectorBase` gives you for free:
- Worker thread pool with per-thread connections
- Submission queue (lock-free enqueue) and completion queue
- Automatic tiling: batch operations are split across workers
- eventfd signaling on completion (kernel wakes Python)
- Graceful shutdown (stop flag, drain, join)

### Step 2: Pybind module

Use the `LMCACHE_BIND_CONNECTOR_METHODS` macro which binds all 6 methods
(`event_fd`, `submit_batch_get/set/exists`, `drain_completions`, `close`)
with proper GIL release and buffer protocol handling.

```cpp
// csrc/storage_backends/mybackend/pybind.cpp
#include <pybind11/pybind11.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_mybackend, m) {
  py::class_<MyConnector>(m, "LMCacheMyBackendClient")
      .def(py::init<std::string, int, int>(),
           py::arg("host"), py::arg("port"), py::arg("num_workers"))
      LMCACHE_BIND_CONNECTOR_METHODS(MyConnector);
}
```

**Reference:** `redis/pybind.cpp`

### Step 3: Build system — register in setup.py

Add your sources to `setup.py` alongside the existing Redis extension:

```python
# In _common_cpp_extensions():
mybackend_sources = [
    "csrc/storage_backends/mybackend/pybind.cpp",
    "csrc/storage_backends/mybackend/connector.cpp",
]

# Add to ext_modules list:
cpp_extension.CppExtension(
    "lmcache.lmcache_mybackend",
    sources=mybackend_sources,
    include_dirs=["csrc/storage_backends", "csrc/storage_backends/mybackend"],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
),
```

Then rebuild: `pip install -e .`

### Step 4: Python client — non-MP mode integration

Inherit from `ConnectorClientBase` which provides asyncio event loop
integration, future management, and both sync and async methods.

```python
# lmcache/v1/storage_backend/native_clients/mybackend_client.py
from .connector_client_base import ConnectorClientBase
from lmcache.lmcache_mybackend import LMCacheMyBackendClient

class MyBackendClient(ConnectorClientBase[LMCacheMyBackendClient]):
    def __init__(self, host: str, port: int, num_workers: int, loop=None):
        native = LMCacheMyBackendClient(host, port, num_workers)
        super().__init__(native, loop)
```

This gives you `async get/set/exists`, `batch_get/batch_set/batch_exists`,
and sync variants, all with automatic eventfd-driven completion handling.

**Reference:** `lmcache/v1/storage_backend/native_clients/resp_client.py`

### Step 5: MP mode integration — L2 adapter config + factory

To use your connector as an L2 adapter in MP mode, add a config class and
register it in the factory. The `NativeConnectorL2Adapter` bridge handles
all the complexity (eventfd demuxing, key serialization, locking).

**a) Add config class** in `lmcache/v1/distributed/l2_adapters/config.py`:

```python
class MyBackendL2AdapterConfig(L2AdapterConfigBase):
    def __init__(self, host: str, port: int, num_workers: int = 8):
        self.host = host
        self.port = port
        self.num_workers = num_workers

    @classmethod
    def from_dict(cls, d: dict) -> "MyBackendL2AdapterConfig":
        host = d.get("host")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        port = d.get("port")
        if not isinstance(port, int) or port <= 0:
            raise ValueError("port must be a positive integer")
        num_workers = d.get("num_workers", 8)
        return cls(host=host, port=port, num_workers=num_workers)

    @classmethod
    def help(cls) -> str:
        return (
            "MyBackend L2 adapter config fields:\n"
            "- host (str): server hostname (required)\n"
            "- port (int): server port (required)\n"
            "- num_workers (int): worker threads (default 8)"
        )

register_l2_adapter_type("mybackend", MyBackendL2AdapterConfig)
```

**b) Add factory branch** in `lmcache/v1/distributed/l2_adapters/__init__.py`:

```python
if isinstance(config, MyBackendL2AdapterConfig):
    from lmcache.lmcache_mybackend import LMCacheMyBackendClient
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )
    native_client = LMCacheMyBackendClient(
        config.host, config.port, config.num_workers
    )
    return NativeConnectorL2Adapter(native_client)
```

**c) Use it** from the command line:

```bash
# MP mode with your backend as L2 storage
--l2-adapter '{"type": "mybackend", "host": "10.0.0.1", "port": 9000}'
```

**Reference:** `RESPL2AdapterConfig` in `config.py` and `_create_resp_l2_adapter`
in `__init__.py`

## Architecture: how NativeConnectorL2Adapter bridges the gap

The C++ connector has 1 eventfd and mixed completions. MP mode's
`L2AdapterInterface` requires 3 eventfds and typed results. The bridge
handles this transparently:

| L2 Adapter method | Native connector call | Extra logic |
|---|---|---|
| `submit_store_task(keys, objs)` | `submit_batch_set` | ObjectKey→str, MemoryObj→memoryview |
| `submit_lookup_and_lock_task(keys)` | `submit_batch_exists` | + client-side lock refcount |
| `submit_load_task(keys, objs)` | `submit_batch_get` | ObjectKey→str, MemoryObj→memoryview |
| `submit_unlock(keys)` | _(none)_ | client-side lock decrement |
| `pop_completed_store_tasks()` | via `drain_completions` | demux by op type |
| `query_lookup_and_lock_result()` | via `drain_completions` | exists→Bitmap, apply locks |
| `query_load_result()` | via `drain_completions` | ok/fail→Bitmap |

The demux thread polls the native eventfd, calls `drain_completions()`,
looks up each `future_id` to determine its operation type, routes the
result to the correct completion dict, and signals the corresponding
Python eventfd.

## Checklist for a new backend

- [ ] C++ connector inheriting `ConnectorBase<T>` with 4 required + 1 optional (`do_single_delete`) method overrides
- [ ] Pybind module using `LMCACHE_BIND_CONNECTOR_METHODS`
- [ ] `setup.py` entry for the new `CppExtension`
- [ ] Python client inheriting `ConnectorClientBase` (non-MP mode)
- [ ] L2 adapter config class + factory registration (MP mode)
- [ ] Unit tests (see `tests/v1/distributed/test_native_connector_l2_adapter.py`)
- [ ] Optional: Aerospike integration (`RUN_AEROSPIKE_INTEGRATION=1`, see `tests/v1/distributed/test_aerospike_l2_integration.py`)
