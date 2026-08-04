# SPDX-License-Identifier: Apache-2.0
"""
Main RPC protocol for the LMCache core server and clients.

This module serves as the main entry point for the protocol system.
All protocol definitions are now organized in the protocols/ subdirectory:
- protocols/base.py: RequestType enum, HandlerType, ProtocolDefinition
- protocols/engine.py: Core KV cache operations (REGISTER, STORE, RETRIEVE, etc.)
- protocols/controller.py: Cache management operations (CLEAR, GET_CHUNK_SIZE)
- protocols/debug.py: Debug and testing operations (NOOP)

The protocol definitions are loaded and validated during initialization.
"""

# Standard
from typing import Any, Optional

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType

# Initialize the protocol system
# This loads all protocol definitions and validates them against the RequestType enum
_PROTOCOL_DEFINITIONS = initialize_protocols()

# Type aliases for backwards compatibility
InstanceID = int
KeyType = IPCCacheServerKey


def get_payload_classes(req_type: RequestType) -> list[Any]:
    """
    Get the expected payload classes for a request type.

    Args:
        req_type: The request type to look up

    Returns:
        List of expected payload classes in order

    Raises:
        ValueError: If the request type is not recognized
    """
    if pd := _PROTOCOL_DEFINITIONS.get(req_type, None):
        return pd.payload_classes
    else:
        raise ValueError(f"Invalid request type: {req_type}")


def get_response_class(req_type: RequestType) -> Optional[Any]:
    """
    Get the expected response class for a request type.

    Args:
        req_type: The request type to look up

    Returns:
        Expected response class, or None if no response

    Raises:
        ValueError: If the request type is not recognized
    """
    if pd := _PROTOCOL_DEFINITIONS.get(req_type, None):
        return pd.response_class
    else:
        raise ValueError(f"Invalid request type: {req_type}")


def get_handler_type(req_type: RequestType) -> HandlerType:
    """
    Get the handler type for a request type.

    Args:
        req_type: The request type to look up

    Returns:
        The handler type (SYNC, BLOCKING, or NON_BLOCKING)

    Raises:
        ValueError: If the request type is not recognized
    """
    if pd := _PROTOCOL_DEFINITIONS.get(req_type, None):
        return pd.handler_type
    else:
        raise ValueError(f"Invalid request type: {req_type}")
