# SPDX-License-Identifier: Apache-2.0
"""
Factory registry for L2 adapters.

Each adapter module self-registers a factory callable via
``register_l2_adapter_factory`` at import time.  The factory
signature is
``(L2AdapterConfigBase, Optional[L1MemoryDesc]) -> L2AdapterInterface``.

Built-in adapters are discovered via ``pkgutil`` but only
imported on demand -- their modules (and third-party
dependencies) are loaded only when the adapter type is
actually requested.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Callable, Optional
import importlib

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.distributed.l2_adapters.base import (
        L2AdapterInterface,
    )
    from lmcache.v1.distributed.l2_adapters.config import (
        L2AdapterConfigBase,
    )

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Type alias for factory callables:
#   (config, l1_memory_desc) -> L2AdapterInterface
L2AdapterFactory = Callable[
    ["L2AdapterConfigBase", "Optional[L1MemoryDesc]"],
    "L2AdapterInterface",
]

# -----------------------------------------------------------------
# Registry: adapter type name -> factory callable
# -----------------------------------------------------------------

_L2_ADAPTER_FACTORY_REGISTRY: dict[str, L2AdapterFactory] = {}

# -----------------------------------------------------------------
# Pending modules: fully-qualified module paths that have
# been discovered (e.g. via pkgutil) but not yet imported.
# When a type name is not found in the eager registry we
# import these one-by-one until the name appears.
# -----------------------------------------------------------------

_PENDING_MODULES: list[str] = []


def register_l2_adapter_factory(
    name: str,
    factory: L2AdapterFactory,
) -> None:
    """Register an adapter factory for the given type
    name.

    Each adapter module should call this at import time
    **after** its config class has been registered via
    ``register_l2_adapter_type``.

    Args:
        name: Adapter type name (must match the name used
            in ``register_l2_adapter_type``).
        factory: ``(config, l1_memory_desc)``
            -> ``L2AdapterInterface``.
    """
    if name in _L2_ADAPTER_FACTORY_REGISTRY:
        raise ValueError("L2 adapter factory already registered: %s" % name)
    _L2_ADAPTER_FACTORY_REGISTRY[name] = factory


def add_pending_module(module_path: str) -> None:
    """Register a module path for deferred import.

    The module will only be imported when a type name
    lookup misses the eager registry.  Importing the
    module triggers its ``register_l2_adapter_type``
    and ``register_l2_adapter_factory`` calls.

    Args:
        module_path: Fully-qualified module path
            (e.g. ``"lmcache.v1.distributed.
            l2_adapters.fs_l2_adapter"``).
    """
    if module_path not in _PENDING_MODULES:
        _PENDING_MODULES.append(module_path)


def ensure_adapter_loaded(name: str) -> None:
    """Import pending modules until *name* appears in
    the factory registry (or all pending modules have
    been tried).

    Raises:
        ImportError: If a module fails to import due to
            a missing third-party dependency **and** it
            was the last candidate.
    """
    if name in _L2_ADAPTER_FACTORY_REGISTRY:
        return

    last_err: ImportError | None = None
    while _PENDING_MODULES:
        mod_path = _PENDING_MODULES.pop(0)
        try:
            importlib.import_module(mod_path)
        except ImportError as exc:
            logger.debug(
                "Skipping module %s (import failed: %s)",
                mod_path,
                exc,
            )
            last_err = exc
            continue
        # Check if the name is now registered
        if name in _L2_ADAPTER_FACTORY_REGISTRY:
            return

    # If we exhausted all pending modules and still
    # didn't find the name, the last ImportError (if
    # any) might be the root cause.
    if last_err is not None and name not in _L2_ADAPTER_FACTORY_REGISTRY:
        raise last_err


def load_all_adapters() -> None:
    """Force-import all pending adapter modules.

    Useful for CLI help that needs to list every
    registered type name.
    """
    while _PENDING_MODULES:
        mod_path = _PENDING_MODULES.pop(0)
        try:
            importlib.import_module(mod_path)
        except ImportError:
            logger.debug(
                "Skipping module %s during bulk load",
                mod_path,
            )


def get_all_registered_names() -> list[str]:
    """Return all known adapter type names.

    Forces import of all pending modules so that the
    returned list is complete.
    """
    load_all_adapters()
    return sorted(_L2_ADAPTER_FACTORY_REGISTRY)


def create_l2_adapter_from_registry(
    config: "L2AdapterConfigBase",
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> "L2AdapterInterface":
    """Create an L2 adapter using the factory registry.

    Looks up the type name for *config* via the config
    registry, then calls the matching factory.

    Args:
        config: An adapter config instance.
        l1_memory_desc: Optional L1 memory descriptor,
            required by adapters that register L1 memory
            with an external backend (e.g. Nixl).

    Returns:
        A new ``L2AdapterInterface`` instance.

    Raises:
        ValueError: If no factory is registered for this
            config type.
    """
    # Import here to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.config import (
        get_type_name_for_config,
    )

    name = get_type_name_for_config(config)

    # Trigger lazy import if needed
    ensure_adapter_loaded(name)

    factory = _L2_ADAPTER_FACTORY_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            "No adapter factory registered for type "
            "%s. Make sure the adapter module is "
            "imported." % name
        )
    return factory(config, l1_memory_desc)
