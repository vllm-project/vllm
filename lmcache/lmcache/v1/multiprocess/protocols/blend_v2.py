# SPDX-License-Identifier: Apache-2.0
"""
Blend V2 protocol definitions.

This module defines the V2 variants of blend lookup/retrieve protocols that use
CBMatchResult instead of plain (start, end) integer tuples.  CBMatchResult carries
old/cur position ranges plus the pre-computed chunk hash, enabling direct storage
key lookup without re-hashing on the server side.
"""

# First Party
from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "CB_LOOKUP_PRE_COMPUTED_V2",
    "CB_RETRIEVE_PRE_COMPUTED_V2",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for blend V2 lookup/retrieve operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Lookup pre-computed chunks (V2)
        # Payload:
        #   - key: IPCCacheServerKey - The key containing the token ids
        # Returns: List of CBMatchResult with match positions and chunk hashes
        "CB_LOOKUP_PRE_COMPUTED_V2": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey],
            response_class=list[CBMatchResult],
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve pre-computed chunks (V2)
        # Payload:
        #   - key: IPCCacheServerKey - The key containing the token ids
        #   - cb_match_result: list[CBMatchResult] - Match results returned by
        #                      CB_LOOKUP_PRE_COMPUTED_V2, with per-chunk hashes
        #                      and query positions
        #   - offset: int - The starting offset in the CB KV cache buffer
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - event_ipc_handle: bytes - IPC handle for event notification when the
        #                       retrieval is complete
        # Returns:
        #   - IPC handle bytes
        #   - boolean flag indicating if the retrieval is successful
        "CB_RETRIEVE_PRE_COMPUTED_V2": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey, list[CBMatchResult], int, int, bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
    }
