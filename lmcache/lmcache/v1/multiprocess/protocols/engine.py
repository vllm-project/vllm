# SPDX-License-Identifier: Apache-2.0
"""
Engine protocol definitions for core KV cache operations.

This module defines the protocol for:
- REGISTER_KV_CACHE: Register a KV cache instance with the server
- UNREGISTER_KV_CACHE: Unregister a KV cache instance (GPU path)
- UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT: Unregister a non-GPU KV cache context
- STORE: Store KV cache blocks to the server
- RETRIEVE: Retrieve KV cache blocks from the server
- LOOKUP: Submit a prefix lookup and return a prefetch job ID
- QUERY_PREFETCH_STATUS: Poll a prefetch job for its result
- END_SESSION: End a session and clean up associated resources
"""

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition


@dataclass
class PrepareStoreResponse:
    """Response for PREPARE_STORE."""

    context: dict = field(
        default_factory=dict
    )  # pickle: {}, shm will put slot info here


@dataclass
class PrepareRetrieveResponse:
    """Response for PREPARE_RETRIEVE."""

    success: bool
    data: bytes = b""
    context: dict = field(
        default_factory=dict
    )  # pickle: {}, shm will put slot info here


@dataclass
class RegisterEngineDrivenContextResponse:
    """Response for REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT."""

    shm_name: str = ""
    pool_size: int = 0


# Define request names for this protocol group
REQUEST_NAMES = [
    "REGISTER_KV_CACHE",
    "UNREGISTER_KV_CACHE",
    "REGISTER_Q_CACHE",
    "UNREGISTER_Q_CACHE",
    "STORE_Q",
    "STORE",
    "RETRIEVE",
    "LOOKUP",
    "QUERY_PREFETCH_STATUS",
    "WAIT_PREFETCH_STATUS",
    "QUERY_PREFETCH_LOOKUP_HITS",
    "FREE_LOOKUP_LOCKS",
    "END_SESSION",
    "REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT",
    "UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT",
    "PREPARE_STORE",
    "COMMIT_STORE",
    "PREPARE_RETRIEVE",
    "COMMIT_RETRIEVE",
]

# Type alias for cache keys
KeyType = IPCCacheServerKey


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for engine operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Register KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the engine instance
        #   - kv_cache: KVCache - The KV cache configuration
        #   - model_name: str - Name of the model associated with the engine
        #   - world_size: int - World size of the engine
        #   - engine_type: EngineType - Which serving engine produced the
        #     caches (vLLM, SGLang, ...). Drives format detection.
        #   - layout_hints: LayoutHints - See custom_types.LayoutHints.
        #   - engine_group_infos: list[EngineGroupInfo] - Engine-neutral KV cache
        #     group metadata (msgspec-encoded by the message queue).
        # Returns: None
        "REGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[
                int,
                KVCache,
                str,
                int,
                EngineType,
                LayoutHints,
                list[EngineGroupInfo],
            ],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "UNREGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Register QRingBuffer.
        # Same as REGISTER_KV_CACHE: the Q ring reuses worker's instance_id
        # registered in REGISTER_KV_CACHE, keyed by a distinct model_name, so
        # STORE can serve it via the existing transfer path with no separate
        # instance id.
        "REGISTER_Q_CACHE": ProtocolDefinition(
            payload_classes=[
                int,
                KVCache,
                str,
                int,
                EngineType,
                LayoutHints,
                list[EngineGroupInfo],
            ],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister the paged Q ring buffer.
        # Same as UNREGISTER_KV_CACHE.
        # Returns: None
        "UNREGISTER_Q_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Store paged Q ring blocks (served by QStoreModule).
        # Same as STORE.
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "STORE_Q": ProtocolDefinition(
            payload_classes=[KeyType, int, list[list[int]], bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Store KV cache blocks
        # Payload:
        #   - key: KeyType - Cache key to store
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[list[int]] - GPU block IDs containing the
        #     data, indexed by LMCache KV group index.
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "STORE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[list[int]], bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve KV cache blocks
        # Payload:
        #   - key: KeyType - Cache key to retrieve
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[list[int]] - GPU block IDs to store
        #     retrieved data, indexed by LMCache KV group index.
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        #   - skip_first_n_tokens: int - Number of tokens to skip writing at the
        #     start of the retrieve range (to avoid overwriting APC-shared blocks)
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "RETRIEVE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[list[int]], bytes, int],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Submit a prefix lookup; job is tracked server-side by request_id
        # Payload:
        #   - key: KeyType - Cache key to look up
        #   - tp_size: int - Tensor-parallel size for
        #       MLA multi-reader locking
        # Returns: None
        "LOOKUP": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the status of a prefetch job by request_id
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        # Returns: int | None - Chunk count when done, None if still in progress
        "QUERY_PREFETCH_STATUS": ProtocolDefinition(
            payload_classes=[str],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Block until a prefetch job completes, then return its result
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        #   - timeout: float - Max seconds to wait for the prefetch to finish
        # Returns: int | None - Chunk count when done, None if the wait timed out
        "WAIT_PREFETCH_STATUS": ProtocolDefinition(
            payload_classes=[str, float],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the lookup hit chunks before the prefetch is done
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        # Returns: int | None - Chunk count if lookup is done, None if still in progress
        "QUERY_PREFETCH_LOOKUP_HITS": ProtocolDefinition(
            payload_classes=[str],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Free locks (release read locks without a full RETRIEVE)
        # Payload:
        #   - key: KeyType - Cache key whose read locks
        #       to release
        #   - tp_size: int - Tensor-parallel size for
        #       MLA multi-reader locking
        # Returns: None
        "FREE_LOOKUP_LOCKS": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # End session
        # Payload:
        #   - request_id: str - Request ID of the session to end
        # Returns: None
        "END_SESSION": ProtocolDefinition(
            payload_classes=[str],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Unregister non-GPU KV cache context
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Register non-GPU KV cache context
        # Payload:
        #   - RegisterEngineDrivenContextPayload - all metadata fields in one struct
        # Returns: RegisterEngineDrivenContextResponse
        "REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": ProtocolDefinition(
            payload_classes=[RegisterEngineDrivenContextPayload],
            response_class=RegisterEngineDrivenContextResponse,
            handler_type=HandlerType.SYNC,
        ),
        "PREPARE_STORE": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=PrepareStoreResponse,
            handler_type=HandlerType.BLOCKING,
        ),
        "COMMIT_STORE": ProtocolDefinition(
            payload_classes=[KeyType, int, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        "PREPARE_RETRIEVE": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=PrepareRetrieveResponse,
            handler_type=HandlerType.BLOCKING,
        ),
        "COMMIT_RETRIEVE": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
    }
