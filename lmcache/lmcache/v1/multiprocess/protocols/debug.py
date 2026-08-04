# SPDX-License-Identifier: Apache-2.0
"""
Debug protocol definitions for testing and monitoring.

This module defines the protocol for:
- NOOP: No-operation command for testing connectivity and as a heartbeat
"""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "NOOP",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for debug operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # No-operation (for testing/heartbeat)
        # Payload: None
        # Returns: str - A confirmation message
        "NOOP": ProtocolDefinition(
            payload_classes=[],
            response_class=str,
            handler_type=HandlerType.SYNC,
        ),
    }
