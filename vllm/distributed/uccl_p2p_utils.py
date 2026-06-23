# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.util
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# declaration for static analyzers
Endpoint: Any
XferDesc: Any


def _load_uccl_p2p_attr(name: str) -> Any:
    attr_name = {
        "Endpoint": "Endpoint",
        "XferDesc": "XferDesc",
    }[name]

    try:
        module = importlib.import_module("uccl.p2p")
    except ImportError:
        logger.warning_once("uccl.p2p is not available")
        value = None
    else:
        value = getattr(module, attr_name, None)
        if value is None:
            logger.warning_once("uccl.p2p is not available")
        else:
            logger.info_once("uccl.p2p is available")

    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in __all__:
        return _load_uccl_p2p_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_uccl_p2p_available() -> bool:
    """Lightweight check for uccl.p2p package without importing it."""
    return importlib.util.find_spec("uccl.p2p") is not None


__all__ = [
    "Endpoint",
    "XferDesc",
    "is_uccl_p2p_available",
]
