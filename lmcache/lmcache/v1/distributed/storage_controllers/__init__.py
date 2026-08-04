# SPDX-License-Identifier: Apache-2.0
"""
Storage controllers with automatic module discovery.

All modules under this package that call ``register_store_policy`` or
``register_prefetch_policy`` at import time are discovered automatically.
To add a new policy, simply create a new module in this directory -- **no
changes to any existing file are needed**.
"""

# Standard
import importlib
import pkgutil

# First Party
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    EvictionController,
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
)

# Auto-discover and import every module in this package so
# that each policy's self-registration code runs.
_PACKAGE_PATH = __path__  # type: ignore[name-defined]
_PACKAGE_NAME = __name__
for _finder, _mod_name, _is_pkg in pkgutil.iter_modules(_PACKAGE_PATH):
    importlib.import_module(f".{_mod_name}", _PACKAGE_NAME)

__all__ = [
    "EvictionController",
    "L1EvictionController",
    "L2EvictionController",
    "L2AdapterEvictionState",
    "PrefetchController",
    "StoreController",
]
