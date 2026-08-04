# SPDX-License-Identifier: Apache-2.0
"""Shared Valkey building blocks used by both the non-MP connector and the
MP-mode L2 adapter.

The single reusable core is :class:`ValkeyWorkerPool`, which owns the
``valkey-glide`` sync clients, the worker thread pool, and the zero-copy
GET/SET/EXISTS/DELETE primitives. Mirrors the DAX layout
(``storage_backend/dax/core.py`` shared by the non-MP backend and the MP
adapter).
"""

# First Party
from lmcache.v1.storage_backend.valkey.worker_pool import ValkeyWorkerPool

__all__ = ["ValkeyWorkerPool"]
