"""A registry for vLLM engine backends.

This module provides a mechanism for registering and discovering different
vLLM engine core implementations. The engine selection process, which happens
in `vllm.v1.engine.core.EngineCoreProc.run_engine_core`, follows a
three-tier priority system:

1. **Explicit Naming (`retrieve_engine_core_proc`)**:
   If the user specifies an `engine_mode` in their configuration (e.g., via the
   `--engine-mode` CLI flag) with a value other than "auto", vLLM will
   attempt to fetch that exact engine from the registry by its name. This
   provides a direct way for users to force a specific engine implementation.

2. **Automatic Discovery (`discover_supported_engine_core_proc`)**:
   If the `engine_mode` is "auto", vLLM will perform an automatic discovery
   by calling `discover_supported_engine_core_proc()`. This function iterates
   through all registered engines and calls their `is_supported()` static
   method. The first (and only) engine that returns `True` is selected. This
   allows external plugins to activate themselves automatically based on the
   environment (e.g., hardware, environment variables).

3. **Internal Fallback (in `core.py`)**:
   If neither an explicitly named engine nor an auto-discovered engine is
   found, the system falls back to its internal default behavior, selecting
   either `EngineCoreProc` or `DPEngineCoreProc` based on the parallel
   configuration.
"""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from typing import Optional

from vllm.v1.engine.interface import IDiscoverableEngineCoreProc

# The lazy-load map is a configuration that tells the registry how to find
# engines that are not part of the core vLLM library and may not be
# installed. It is exclusively for engines from optional, external packages
# that support non-GPU hardware.
#
# The dictionary is structured as follows:
#
# {
#   "name": {
#     "path": "full.import.path.to.DiscoverableEngineCoreProcClass",
#     "extra": "optional_dependency_name"
#   }
# }
#
# - "name" (str): The name to be used in the registry.
# - "path" (str): The full import path to the class.
# - "extra" (str): The name of the optional dependency group (e.g., 'tpu')
#   that provides this engine core proc. This is used to generate a helpful
#   error message if the import fails.
_LAZY_ENGINE_CORE_PROC_MAP: dict[str, dict[str, str]] = {}

# The engine registry is the single source of truth for all available engine
# cores at runtime. It holds the actual, loaded `type` objects for all engines
# that are ready to be instantiated.
#
# It contains both all external engines that have been successfully lazy-loaded
# from the _LAZY_ENGINE_CORE_PROC_MAP.
_engine_core_proc_registry: dict[str, type[IDiscoverableEngineCoreProc]] = {}

_RESERVED_ENGINE_CORE_PROC_NAMES = {"auto", "default"}


def register_engine_core_proc(name: str,
                              engine_class: type[IDiscoverableEngineCoreProc]):
    """Registers a new IDiscoverableEngineCoreProc class."""
    if name in _RESERVED_ENGINE_CORE_PROC_NAMES:
        raise ValueError(
            f"IDiscoverableEngineCoreProc named '{name}' is reserved and "
            "cannot be used.")
    if name in _engine_core_proc_registry:
        raise ValueError(
            f"IDiscoverableEngineCoreProc named '{name}' is already registered."
        )
    _engine_core_proc_registry[name] = engine_class


def retrieve_engine_core_proc(name: str) -> type[IDiscoverableEngineCoreProc]:
    """Retrieves an IDiscoverableEngineCoreProc class by name."""
    # 1. Check if the engine is already loaded in the main registry.
    if name not in _engine_core_proc_registry:
        # 2. If not, check if it's a known lazy-loadable engine.
        if name in _LAZY_ENGINE_CORE_PROC_MAP:
            engine_info = _LAZY_ENGINE_CORE_PROC_MAP[name]
            import_path = engine_info["path"]
            try:
                # Attempt to import the engine on-the-fly.
                module_path, class_name = import_path.rsplit(".", 1)
                module = import_module(module_path)
                engine_class = getattr(module, class_name)
                # If successful, register it for next time to speed up lookup.
                register_engine_core_proc(name, engine_class)
                return engine_class
            except (ImportError, AttributeError) as err:
                # If import fails, provide a helpful error message.
                extra = engine_info.get("extra")
                if extra:
                    install_command = f"pip install vllm[{extra}]"
                else:
                    install_command = "the necessary dependencies"
                raise ValueError(
                    f"Engine core '{name}' could not be imported from "
                    f"'{import_path}'. To install it, run `{install_command}`."
                ) from err
        # 3. If it's not in the main registry or the lazy map, it's unknown.
        raise ValueError(f"Engine core '{name}' is not registered.")

    # Engine was found in the main registry, return it directly.
    return _engine_core_proc_registry[name]


def discover_supported_engine_core_proc(
) -> Optional[type[IDiscoverableEngineCoreProc]]:
    """Iterates through registered IDiscoverableEngineCoreProc and returns the
    first one that supports the current use case."""
    supported_engines = []
    # Note: .values() is not ordered in older pythons, but for now we assume
    # either one or zero. If multiple, we raise an error.
    for engine_class in _engine_core_proc_registry.values():
        if engine_class.is_supported():
            supported_engines.append(engine_class)

    if not supported_engines:
        return None

    if len(supported_engines) > 1:
        raise RuntimeError(
            "Multiple custom engines support the current environment: "
            f"{[e.__name__ for e in supported_engines]}. "
            "Please review your configuration or environment to ensure only "
            "one custom engine's `is_supported()` method returns True.")
    return supported_engines[0]
