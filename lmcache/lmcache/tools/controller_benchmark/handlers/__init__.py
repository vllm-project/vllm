"""Operation handlers with dynamic discovery and registration"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict

# First Party
from lmcache.v1.utils.subclass_discovery import discover_subclasses

# Local
from .base import OperationHandler

# Operation handler registry
OPERATION_HANDLERS: Dict[str, OperationHandler] = {}


def _discover_and_register_handlers():
    """Dynamically discover and register all operation handlers"""
    for cls in discover_subclasses(
        __name__,
        OperationHandler,
        module_filter=lambda name: name != "base",
        require_defined_in_module=False,
    ):
        handler = cls()
        OPERATION_HANDLERS[handler.operation_name] = handler


# Auto-discover and register handlers on import
_discover_and_register_handlers()
