# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module

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
