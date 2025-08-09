"""A registry for vLLM engine backends.

This module provides a mechanism for registering and discovering different
vLLM engine core implementations. The engine selection process, which happens
in `vllm.v1.engine.core.EngineCoreProc.run_engine_core`, follows a
three-tier priority system:

1. **Explicit Naming (`get_engine_core`)**:
   If the user specifies an `engine_mode` in their configuration (e.g., via the
   `--engine-mode` CLI flag) with a value other than "default", vLLM will
   attempt to fetch that exact engine from the registry by its name. This
   provides a direct way for users to force a specific engine implementation.

2. **Automatic Discovery (`find_supported_engine`)**:
   If the `engine_mode` is "default", vLLM will perform an automatic discovery
   by calling `find_supported_engine()`. This function iterates through all
   registered engines and calls their `is_supported()` static method. The first
   (and only) engine that returns `True` is selected. This allows external
   plugins to activate themselves automatically based on the environment (e.g.,
   hardware, environment variables).

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

from vllm.v1.engine.interface import IEngineCoreProc

# The lazy-load map is a configuration that tells the registry how to find
# engines that are not part of the core vLLM library and may not be
# installed. It is exclusively for engines from optional, external packages
# that support non GPU hardware.
#
# The dictionary is structured as follows:
#
# {
#   "engine_name": {
#     "path": "full.import.path.to.EngineCoreProcClass",
#     "extra": "optional_dependency_name"
#   }
# }
#
# - "engine_name" (str): The name to be used in the registry.
# - "path" (str): The full import path to the engine class.
# - "extra" (str): The name of the optional dependency group (e.g., 'tpu')
#   that provides this engine. This is used to generate a helpful error
#   message if the import fails.
_LAZY_ENGINE_MAP: dict[str, dict[str, str]] = {}

# The engine registry is the single source of truth for all available engine
# cores at runtime. It holds the actual, loaded `type` objects for all engines
# that are ready to be instantiated.
#
# It contains both vLLM's native engines (e.g., EngineCoreProc) and any
# external engines that have been successfully lazy-loaded from the
# _LAZY_ENGINE_MAP.
_engine_registry: dict[str, type[IEngineCoreProc]] = {}


def register_engine_core(name: str, engine_class: type[IEngineCoreProc]):
    """Registers a new engine core class."""
    if name in _engine_registry:
        raise ValueError(f"Engine core '{name}' is already registered.")
    _engine_registry[name] = engine_class


def get_engine_core(name: str) -> type[IEngineCoreProc]:
    """Retrieves an engine core class by name."""
    if name not in _engine_registry:
        if name in _LAZY_ENGINE_MAP:
            engine_info = _LAZY_ENGINE_MAP[name]
            import_path = engine_info["path"]
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = import_module(module_path)
                engine_class = getattr(module, class_name)
                register_engine_core(name, engine_class)
                return engine_class
            except (ImportError, AttributeError) as err:
                extra = engine_info.get("extra")
                if extra:
                    install_command = f"pip install vllm[{extra}]"
                else:
                    install_command = "the necessary dependencies"
                raise ValueError(
                    f"Engine core '{name}' could not be imported from "
                    f"'{import_path}'. To install it, run `{install_command}`."
                ) from err
        raise ValueError(f"Engine core '{name}' is not registered.")
    return _engine_registry[name]


def find_supported_engine() -> Optional[type[IEngineCoreProc]]:
    """Iterates through registered engines and returns the first one that
    supports the current environment."""
    supported_engines = []
    # Note: .values() is not ordered in older pythons, but for now we assume
    # either one or zero. If multiple, we raise an error.
    for engine_class in _engine_registry.values():
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
