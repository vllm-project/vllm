# SPDX-License-Identifier: Apache-2.0
"""
P2P protocol definitions for peer lookup-and-lock operations.

This module defines the protocol for:
- P2P_LOOKUP_AND_LOCK: Look up and read-lock the locally cached keys
- P2P_QUERY_LOOKUP_RESULTS: Poll the transfer addresses of a lookup
- P2P_UNLOCK_OBJECTS: Release the read locks held by a peer
"""

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "P2P_LOOKUP_AND_LOCK",
    "P2P_QUERY_LOOKUP_RESULTS",
    "P2P_UNLOCK_OBJECTS",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for P2P operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Look up and read-lock the L1-resident subset of the given keys
        # (sparse; gaps allowed)
        # Payload:
        #   - keys: list[ObjectKey] - Object keys to look up and lock
        #   - layout_desc: MemoryLayoutDesc - Memory layout of the objects
        # Returns: int - Task id for querying the lookup status later
        "P2P_LOOKUP_AND_LOCK": ProtocolDefinition(
            payload_classes=[list[ObjectKey], MemoryLayoutDesc],
            response_class=int,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the transfer addresses for a lookup task
        # Payload:
        #   - task_id: int - Task id returned by P2P_LOOKUP_AND_LOCK
        # Returns: list[TransferChannelAddress] | None - Addresses when the
        #   lookup is complete, None if still in progress or already consumed
        "P2P_QUERY_LOOKUP_RESULTS": ProtocolDefinition(
            payload_classes=[int],
            response_class=list[TransferChannelAddress] | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Release the read locks held on the given keys
        # Payload:
        #   - keys: list[ObjectKey] - Object keys to unlock
        # Returns: None
        "P2P_UNLOCK_OBJECTS": ProtocolDefinition(
            payload_classes=[list[ObjectKey]],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
    }
