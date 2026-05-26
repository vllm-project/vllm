# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# declaration for static analyzers
NixlWrapper: Any
nixl_agent_config: Any
nixlXferTelemetry: Any


def _get_nixl_module_name(name: str) -> str:
    package_name = "rixl" if current_platform.is_rocm() else "nixl"
    if name == "nixlXferTelemetry":
        return f"{package_name}._bindings"
    return f"{package_name}._api"


def _load_nixl_attr(name: str) -> Any:
    attr_name = {
        "NixlWrapper": "nixl_agent",
        "nixl_agent_config": "nixl_agent_config",
        "nixlXferTelemetry": "nixlXferTelemetry",
    }[name]

    try:
        module = importlib.import_module(_get_nixl_module_name(name))
    except ImportError:
        if name == "NixlWrapper":
            logger.warning_once("NIXL is not available")
        elif name == "nixl_agent_config":
            logger.warning_once("NIXL agent config is not available")
        value = None
    else:
        value = getattr(module, attr_name, None)
        if name == "NixlWrapper":
            if value is None:
                logger.warning_once("NIXL is not available")
            else:
                logger.info_once("NIXL is available")
        elif name == "nixl_agent_config" and value is None:
            logger.warning_once("NIXL agent config is not available")

    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in __all__:
        return _load_nixl_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["NixlWrapper", "nixl_agent_config", "nixlXferTelemetry"]
