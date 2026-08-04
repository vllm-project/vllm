# SPDX-License-Identifier: Apache-2.0
"""
Example native connector plugin for LMCache.

This module provides a factory function that creates a
C++ pybind11-wrapped connector instance.  Two storage
backends are available:

- ``"fs"`` (default): persists data as files under a
  configurable directory.
- ``"memory"``: stores data in a C++ unordered_map
  (in-process, lost on restart).

Usage via ``native_plugin`` L2 adapter type::

    --l2-adapter '{
      "type": "native_plugin",
      "module_path": "lmc_external_native_connector",
      "class_name": "ExampleNativeConnector",
      "adapter_params": {
        "backend": "fs",
        "base_path": "/tmp/lmcache_ext",
        "num_workers": 2
      }
    }'
"""

# Future
from __future__ import annotations

# Third Party
from lmc_external_native_connector._native import (
    ExampleFSConnector,
    ExampleMemoryConnector,
)


class ExampleNativeConnector:
    """Factory-style wrapper that delegates to the
    appropriate C++ connector backend.

    Constructor keyword arguments:
    - backend (str): ``"fs"`` or ``"memory"``
      (default ``"fs"``).
    - base_path (str): directory for fs backend
      (default ``"/tmp/lmcache_ext"``).
    - num_workers (int): number of C++ I/O worker
      threads (default 2).

    The returned instance is itself a native connector
    exposing the full pybind interface (event_fd,
    submit_batch_get/set/exists, drain_completions,
    close).
    """

    def __new__(
        cls,
        backend: str = "fs",
        base_path: str = "/tmp/lmcache_ext",
        num_workers: int = 2,
    ):
        if backend == "fs":
            return ExampleFSConnector(base_path, num_workers)
        if backend == "memory":
            return ExampleMemoryConnector(num_workers)
        raise ValueError("Unknown backend %r, choose 'fs' or 'memory'" % backend)
