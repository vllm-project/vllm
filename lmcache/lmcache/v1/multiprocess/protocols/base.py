# SPDX-License-Identifier: Apache-2.0
"""
Base types and classes for the multiprocess protocol system.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Optional
import enum


class HandlerType(enum.Enum):
    """
    Defines how a protocol handler should be executed.

    - SYNC: Handler runs directly in the main loop (fast, non-blocking operations)
    - BLOCKING: Handler may block, run in a thread pool (I/O, slow operations)
    - NON_BLOCKING: Not supported yet (for future async handlers)
    """

    SYNC = enum.auto()
    BLOCKING = enum.auto()
    NON_BLOCKING = enum.auto()


class RequestType(enum.Enum):
    """
    Enum of all available request types in the protocol system.

    When adding a new request type:
    1. Add the enum member here
    2. Add the protocol definition in the appropriate protocols/*.py file
    3. The validation system will ensure they stay in sync

    Organized by category:
    - Engine operations: Core KV cache operations
    - Controller operations: Cache management and configuration
    - Debug operations: Testing and monitoring
    """

    # Engine operations
    REGISTER_KV_CACHE = enum.auto()
    UNREGISTER_KV_CACHE = enum.auto()
    REGISTER_Q_CACHE = enum.auto()
    UNREGISTER_Q_CACHE = enum.auto()
    STORE_Q = enum.auto()
    STORE = enum.auto()
    RETRIEVE = enum.auto()
    LOOKUP = enum.auto()
    QUERY_PREFETCH_STATUS = enum.auto()
    WAIT_PREFETCH_STATUS = enum.auto()
    QUERY_PREFETCH_LOOKUP_HITS = enum.auto()
    FREE_LOOKUP_LOCKS = enum.auto()
    END_SESSION = enum.auto()
    REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT = enum.auto()
    UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT = enum.auto()
    PREPARE_STORE = enum.auto()
    COMMIT_STORE = enum.auto()
    PREPARE_RETRIEVE = enum.auto()
    COMMIT_RETRIEVE = enum.auto()

    # Controller operations
    CLEAR = enum.auto()
    GET_CHUNK_SIZE = enum.auto()
    PING = enum.auto()

    # Observability operations
    REPORT_BLOCK_ALLOCATION = enum.auto()

    # Debug operations
    NOOP = enum.auto()

    # Blend operations
    CB_REGISTER_KV_CACHE = enum.auto()
    CB_UNREGISTER_KV_CACHE = enum.auto()
    CB_STORE_PRE_COMPUTED = enum.auto()
    CB_LOOKUP_PRE_COMPUTED = enum.auto()
    CB_RETRIEVE_PRE_COMPUTED = enum.auto()
    CB_STORE_FINAL = enum.auto()

    # Blend V2 operations (use CBMatchResult instead of list[tuple[int, int]])
    CB_LOOKUP_PRE_COMPUTED_V2 = enum.auto()
    CB_RETRIEVE_PRE_COMPUTED_V2 = enum.auto()

    # Blend V3 — paged-aware CB.
    CB_REGISTER_ROPE_V3 = enum.auto()
    CB_UNREGISTER_ROPE_V3 = enum.auto()
    CB_RETRIEVE_PRE_COMPUTED_V3 = enum.auto()
    CB_UNIFIED_LOOKUP = enum.auto()

    # P2P operations
    P2P_LOOKUP_AND_LOCK = enum.auto()
    P2P_QUERY_LOOKUP_RESULTS = enum.auto()
    P2P_UNLOCK_OBJECTS = enum.auto()

    # Experimental transfer intermediate tensor
    GET_EXPERIMENTAL = enum.auto()


@dataclass
class ProtocolDefinition:
    """
    Defines the structure and behavior of a protocol request.

    Attributes:
        payload_classes: List of expected payload types in order
        response_class: Expected response type, or None if no response
        handler_type: How the handler should be executed (SYNC/BLOCKING/NON_BLOCKING)
    """

    payload_classes: list[Any]
    response_class: Optional[Any]
    handler_type: HandlerType
