# SPDX-License-Identifier: Apache-2.0
"""
Debug protocol definitions for testing and monitoring.

This module defines the protocol for:
- NOOP: No-operation command for testing connectivity and as a heartbeat
"""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "CB_LOOKUP_PRE_COMPUTED",
    "CB_STORE_PRE_COMPUTED",
    "CB_RETRIEVE_PRE_COMPUTED",
    "CB_STORE_FINAL",
    "CB_REGISTER_KV_CACHE",
    "CB_UNREGISTER_KV_CACHE",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for debug operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Lookup pre-computed chunks
        # Payload:
        #   - key: IPCCacheServerKey - The key containing the token ids
        # Returns: List of tuples (start, end) indicating the match ranges
        "CB_LOOKUP_PRE_COMPUTED": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey],
            response_class=list[tuple[int, int]],
            handler_type=HandlerType.BLOCKING,
        ),
        # Store pre-computed chunks
        # Payload:
        #   - key: IPCCacheServerKey - The key containing the token ids
        #   - offset: int - The starting offset in the CB KV cache buffer
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - event_ipc_handle: bytes - IPC handle for event notification
        #                       when the pre-computed chunks are ready
        # Returns:
        #   - IPC handle bytes
        #   - boolean flag indicating if the store is successful
        "CB_STORE_PRE_COMPUTED": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey, int, int, bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve pre-computed chunks
        # Payload:
        #   - key: IPCCacheServerKey - The key containing the token ids
        #   - ranges: List[tuple[int, int]] - List of tuples (start, end) indicating
        #                                     the match ranges to retrieve
        #   - offset: int - The starting offset in the CB KV cache buffer
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - event_ipc_handle: bytes - IPC handle for event notification when the
        #                       retrieval is complete
        # Returns:
        #   - IPC handle bytes
        #   - boolean flag indicating if the retrieval is successful
        "CB_RETRIEVE_PRE_COMPUTED": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey, list[tuple[int, int]], int, int, bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Store final chunks after processing
        # Payload:
        #   - key: IPCCacheServerKey - The key containing the token ids
        #   - offset: int - The starting offset in the CB KV cache buffer
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - event_ipc_handle: bytes - IPC handle for event notification
        #                       when the final chunks are stored
        # Returns:
        #   - IPC handle bytes
        #   - boolean flag indicating if the store is successful
        "CB_STORE_FINAL": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey, int, int, bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Register CB KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - kv_cache: KVCache - The CB KV cache configuration
        #   - model_name: str - Name of the model associated with the engine
        #   - world_size: int - World size of the engine
        # Returns: None
        "CB_REGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int, KVCache, str, int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister CB KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "CB_UNREGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
    }
