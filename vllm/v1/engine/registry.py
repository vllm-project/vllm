# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.engine.interface import IEngineCoreProc

_engine_registry: dict[str, type[IEngineCoreProc]] = {}


def register_engine_core(name: str, engine_class: type[IEngineCoreProc]):
    """Registers a new engine core class."""
    if name in _engine_registry:
        raise ValueError(f"Engine core '{name}' is already registered.")
    _engine_registry[name] = engine_class


def get_engine_core(name: str) -> type[IEngineCoreProc]:
    """Retrieves an engine core class by name."""
    if name not in _engine_registry:
        # Attempt to import and register on-demand if it's a known type
        if name == "disaggregated":
            try:
                from tpu_commons.core.core_tpu import DisaggEngineCoreProc
                register_engine_core("disaggregated", DisaggEngineCoreProc)
                return DisaggEngineCoreProc
            except ImportError as err:
                raise ValueError(
                    f"Engine core '{name}' is not registered and 'tpu_commons' "
                    "could not be imported.") from err
        raise ValueError(f"Engine core '{name}' is not registered.")
    return _engine_registry[name]
