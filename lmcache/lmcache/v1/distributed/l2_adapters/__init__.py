# SPDX-License-Identifier: Apache-2.0
"""
L2 adapter factory with lazy module loading.

Built-in adapter modules are auto-discovered via
``pkgutil`` but only imported when their adapter type
is actually requested.  To add a new built-in adapter,
simply create a new ``*_l2_adapter.py`` module in this
directory -- **no other changes are needed**.
"""

# Standard
import pkgutil

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    add_pending_module,
    create_l2_adapter_from_registry,
)

# ---------------------------------------------------------
# Auto-discover built-in adapter modules (not imported yet)
# Convention: any ``*_l2_adapter`` submodule is eligible.
# ---------------------------------------------------------
for _finder, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    if _module_name.endswith("_l2_adapter"):
        add_pending_module(f"{__name__}.{_module_name}")


def create_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """Create an L2 adapter from its config via the
    factory registry.

    Args:
        config: The adapter-specific config object.
        l1_memory_desc: Descriptor of the L1 memory buffer,
            required for adapters that register L1 memory
            with an external backend (e.g. Nixl or
            Mooncake when ``protocol == "rdma"``).

    Returns:
        L2AdapterInterface: A new adapter instance.

    Raises:
        ValueError: If no factory is registered for
            the config type.
    """
    return create_l2_adapter_from_registry(
        config,
        l1_memory_desc=l1_memory_desc,
    )


__all__ = [
    "L2AdapterInterface",
    "L2AdapterConfigBase",
    "create_l2_adapter",
]
