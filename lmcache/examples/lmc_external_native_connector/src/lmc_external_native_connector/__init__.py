# SPDX-License-Identifier: Apache-2.0
"""
lmc_external_native_connector - Example external C++
native connector plugin for LMCache.

This package provides pybind11-wrapped C++ connectors
that implement the same interface as the built-in
LMCache connectors (event_fd, submit_batch_get/set/exists,
drain_completions, close).  It can be loaded via the
``native_plugin`` L2 adapter type.

Two storage backends are supported:
- "fs" (default): stores data as files on disk.
- "memory": stores data in a C++ unordered_map.

Low-level access to individual backends:
- ExampleFSConnector: C++ filesystem connector.
- ExampleMemoryConnector: C++ in-memory connector.
"""

# Third Party
from lmc_external_native_connector.connector import (
    ExampleNativeConnector,
)

__all__ = [
    "ExampleNativeConnector",
]
