# SPDX-License-Identifier: Apache-2.0
"""
Controller protocol definitions for cache management and configuration.

This module defines the protocol for:
- CLEAR: Clear all caches in the server
- GET_CHUNK_SIZE: Get the chunk size configuration from the server
- GET_EXPERIMENTAL: Get the enabled experimental intermediate tensor transfer
"""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "CLEAR",
    "GET_CHUNK_SIZE",
    "GET_EXPERIMENTAL",
    "PING",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for controller operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Clear all caches
        # Payload: None
        # Returns: None
        "CLEAR": ProtocolDefinition(
            payload_classes=[],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Get chunk size configuration
        # Payload: None
        # Returns: int - The chunk size value
        "GET_CHUNK_SIZE": ProtocolDefinition(
            payload_classes=[],
            response_class=int,
            handler_type=HandlerType.SYNC,
        ),
        # Ping
        # Payload: [instance_id] -- the sender's worker instance ID, or None
        #   for an untracked prober (the scheduler adapter).
        # Returns: bool - Always True
        # BLOCKING on the NORMAL pool: keeps PING off the MQ main loop (where a
        # slow SYNC REGISTER_KV_CACHE would stall it) and lets pool saturation
        # surface as worker degraded mode.
        "PING": ProtocolDefinition(
            payload_classes=[int | None],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # Get the enabled experimental intermediate tensor transfer types
        # Payload: None
        # Returns: list[str]: the experimental intermediate tensor transfer
        # types the server was launched with (empty when none are enabled)
        "GET_EXPERIMENTAL": ProtocolDefinition(
            payload_classes=[],
            response_class=list[str],
            handler_type=HandlerType.SYNC,
        ),
    }
