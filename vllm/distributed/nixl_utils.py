# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import sys
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import cpu_supports_avx

logger = init_logger(__name__)

# declaration for static analyzers
NixlWrapper: Any
nixl_agent_config: Any
nixlXferTelemetry: Any


def _nixl_is_safe_to_load() -> bool:
    """Whether NIXL's native libraries can be loaded on this host.

    The UCX library bundled with the NIXL wheels is compiled with AVX and
    terminates the process from its native initializer on x86 CPUs that do
    not support it (https://github.com/ai-dynamo/nixl/issues/2119). The
    crash cannot be caught as a Python exception, so NIXL is treated as
    unavailable on such hosts.
    """
    if cpu_supports_avx():
        return True
    logger.warning_once(
        "Disabling NIXL: the UCX library bundled with NIXL is compiled "
        "with AVX, which this x86 CPU does not support. Loading it would "
        "terminate the process. NIXL-based features (KV transfer "
        "connectors, the NIXL EPLB communicator) will be unavailable. "
        "See https://github.com/ai-dynamo/nixl/issues/2119."
    )
    return False


def _maybe_set_ucx_rcache_limit() -> None:
    if "UCX_RCACHE_MAX_UNRELEASED" in os.environ:
        return

    if "nixl" in sys.modules or "nixl_rocm" in sys.modules:
        logger.warning_once(
            "NIXL was already imported, we can't reset "
            "UCX_RCACHE_MAX_UNRELEASED. "
            "Please set it to '1024' manually."
        )
        return

    logger.info_once(
        "Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare "
        "memory leak in UCX when using NIXL."
    )
    os.environ["UCX_RCACHE_MAX_UNRELEASED"] = "1024"


def _get_nixl_package_name() -> str:
    return "nixl_rocm" if current_platform.is_rocm() else "nixl"


def _get_nixl_module_name(name: str) -> str:
    package_name = _get_nixl_package_name()
    if name == "nixlXferTelemetry":
        return f"{package_name}._bindings"
    return f"{package_name}._api"


def _load_nixl_attr(name: str) -> Any:
    attr_name = {
        "NixlWrapper": "nixl_agent",
        "nixl_agent_config": "nixl_agent_config",
        "nixlXferTelemetry": "nixlXferTelemetry",
    }[name]

    if not _nixl_is_safe_to_load():
        globals()[name] = None
        return None

    _maybe_set_ucx_rcache_limit()
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


def is_nixl_available() -> bool:
    """Lightweight check for the platform's NIXL package without importing it.

    Returns False on x86 CPUs without AVX, where loading the UCX library
    bundled with NIXL terminates the process (see
    ``_nixl_is_safe_to_load``).
    """
    import importlib.util

    if not _nixl_is_safe_to_load():
        return False
    pkg = _get_nixl_package_name()
    return pkg in sys.modules or importlib.util.find_spec(pkg) is not None


__all__ = [
    "NixlWrapper",
    "nixl_agent_config",
    "nixlXferTelemetry",
    "is_nixl_available",
]
