# Modular Protocol System

This directory contains the modular protocol definitions for the LMCache multiprocess system.

## Directory Structure

```
protocols/
├── README.md           # This file
├── __init__.py         # Protocol initialization and registration
├── base.py             # Common types (HandlerType, ProtocolDefinition, RequestType)
├── engine.py           # Engine operations (REGISTER/UNREGISTER, STORE, RETRIEVE, LOOKUP, PREPARE/COMMIT, ...)
├── controller.py       # Controller operations (CLEAR, GET_CHUNK_SIZE, PING)
├── debug.py            # Debug operations (NOOP)
├── blend.py            # CacheBlend v1 operations (CB_STORE/RETRIEVE_PRE_COMPUTED, CB_STORE_FINAL, ...)
├── blend_v2.py         # CacheBlend v2 lookup/retrieve (CB_*_V2)
├── blend_v3.py         # CacheBlend v3 rope + unified lookup (CB_*_V3, CB_UNIFIED_LOOKUP)
├── observability.py    # Observability events (REPORT_BLOCK_ALLOCATION)
└── p2p.py              # Peer-to-peer transfers (P2P_LOOKUP_AND_LOCK, P2P_QUERY_LOOKUP_RESULTS, P2P_UNLOCK_OBJECTS)
```

## Design Overview

The protocol system is designed to be modular, extensible, and IDE-friendly:

1. **Static Enum with Validation**: The `RequestType` enum is defined statically in `base.py`:
   - Provides perfect IDE autocomplete and type checking
   - All request types visible to static analysis tools
   - Validation ensures enum stays in sync with protocol definitions

2. **Protocol Modules**: Each module (engine, controller, debug, blend, blend_v2, blend_v3, observability, p2p) defines:
   - `REQUEST_NAMES`: List of request type names (for validation)
   - `get_protocol_definitions()`: Returns dict of name → ProtocolDefinition

3. **Validated Initialization**: The `__init__.py` module:
   - Collects all protocol definitions from modules
   - Validates each `RequestType` enum member has a definition
   - Validates each definition has a corresponding enum member
   - Ensures no duplicates or mismatches

4. **Main Entry Point**: The `protocol.py` file:
   - Calls `initialize_protocols()` on import
   - Provides backwards-compatible API
   - Exports `RequestType`, helper functions, and types

## Adding New Protocols

To add new protocol operations:

### Option 1: Add to Existing Module

If your operation fits an existing category (engine / controller / debug / blend / blend_v2 / blend_v3 / observability / p2p):

1. **Add to the enum** in `protocols/base.py`:
   ```python
   class RequestType(enum.Enum):
       # ... existing members ...
       YOUR_NEW_OP = enum.auto()  # Add here
   ```

2. **Edit the appropriate protocol file** (e.g., `engine.py`):
   - Add the request name to `REQUEST_NAMES`:
     ```python
     REQUEST_NAMES = [
         "EXISTING_OP",
         "YOUR_NEW_OP",  # Add here
     ]
     ```

3. **Add the protocol definition** in `get_protocol_definitions()`:
   ```python
   def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
       return {
           # ... existing definitions ...
           "YOUR_NEW_OP": ProtocolDefinition(
               payload_classes=[int, str],  # Your payload types
               response_class=bool,          # Your response type
               handler_type=HandlerType.SYNC,  # or BLOCKING
           ),
       }
   ```

4. **Done!** The validation system will verify everything matches on import.

### Option 2: Create New Protocol Module

If you're adding a new category of operations:

1. **Add enum members** in `protocols/base.py`:
   ```python
   class RequestType(enum.Enum):
       # ... existing members ...
       
       # Monitoring operations
       HEALTH_CHECK = enum.auto()
       GET_STATS = enum.auto()
   ```

2. **Create a new protocol file** (e.g., `monitoring.py`):
   ```python
   # SPDX-License-Identifier: Apache-2.0
   """
   Monitoring protocol definitions.
   
   This module defines protocols for:
   - HEALTH_CHECK: Check server health status
   - GET_STATS: Get cache statistics
   """
   from lmcache.v1.multiprocess.protocols.base import ProtocolDefinition, HandlerType

   REQUEST_NAMES = [
       "HEALTH_CHECK",
       "GET_STATS",
   ]

   def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
       return {
           "HEALTH_CHECK": ProtocolDefinition(
               payload_classes=[],
               response_class=dict,
               handler_type=HandlerType.SYNC,
           ),
           "GET_STATS": ProtocolDefinition(
               payload_classes=[],
               response_class=dict,
               handler_type=HandlerType.SYNC,
           ),
       }
   ```

3. **Register the module** in `__init__.py` by importing it and adding
   an entry to the `_PROTOCOL_MODULES` list:
   ```python
   from lmcache.v1.multiprocess.protocols import monitoring  # Import your module

   _PROTOCOL_MODULES = [
       ("engine", engine),
       ("controller", controller),
       ("debug", debug),
       ("blend", blend),
       ("blend_v2", blend_v2),
       ("blend_v3", blend_v3),
       ("observability", observability),
       ("p2p", p2p),
       ("monitoring", monitoring),  # Add here
   ]
   ```

4. **Done!** The new operations are now available as:
   - `RequestType.HEALTH_CHECK` (with full IDE autocomplete!)
   - `RequestType.GET_STATS`

## Using the Protocol System

From any module in the codebase:

```python
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    HandlerType,
    get_payload_classes,
    get_response_class,
    get_handler_type,
)

# Use request types
req_type = RequestType.STORE

# Get protocol information
payloads = get_payload_classes(req_type)
response = get_response_class(req_type)
handler = get_handler_type(req_type)
```

## Validation

The initialization system validates at startup:

1. **Enum-Definition Sync**: Every `RequestType` enum member must have a protocol definition
2. **Definition-Enum Sync**: Every protocol definition must have a corresponding enum member
3. **No duplicates**: Same request name cannot be defined in multiple modules
4. **Complete definitions**: All names in `REQUEST_NAMES` must have definitions in `get_protocol_definitions()`

If validation fails, `ProtocolInitializationError` is raised with a descriptive message pointing to the issue.

### Example Error Messages

```python
# If you add RequestType.NEW_OP but forget the definition:
ProtocolInitializationError: RequestType enum members {'NEW_OP'} have no protocol definitions.
Add definitions to the appropriate protocol module or remove from the enum.

# If you add a definition but forget the enum member:
ProtocolInitializationError: Protocol definition 'NEW_OP' in module 'engine'
has no corresponding RequestType enum member. Add 'RequestType.NEW_OP' to protocols/base.py
```

## Current Protocol Groups

The authoritative list of request names is `REQUEST_NAMES` in each module; the
list below is a snapshot as of this file's last update.

### Engine Operations (`engine.py`)
Core KV cache operations and their split-phase variants:
- `REGISTER_KV_CACHE` / `UNREGISTER_KV_CACHE`: Register / unregister the GPU KV cache
- `REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT` / `UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`: Register / unregister the non-GPU (engine-driven) KV cache context
- `STORE` / `RETRIEVE`: Fused store / retrieve
- `PREPARE_STORE` / `COMMIT_STORE` / `PREPARE_RETRIEVE` / `COMMIT_RETRIEVE`: Split-phase store / retrieve (used by the engine-driven path)
- `LOOKUP`: Submit a prefix lookup and return a prefetch job id
- `QUERY_PREFETCH_STATUS` / `WAIT_PREFETCH_STATUS`: Poll / block for a prefetch job's result
- `QUERY_PREFETCH_LOOKUP_HITS` / `FREE_LOOKUP_LOCKS`: Inspect and release the read locks a lookup took on cached chunks
- `END_SESSION`: End a session and clean up associated resources

### Controller Operations (`controller.py`)
Cache management and configuration:
- `CLEAR`: Clear all caches in the server
- `GET_CHUNK_SIZE`: Get the chunk size configuration
- `PING`: Liveness / worker probe (payload: sender's worker instance id or `None`)

### Debug Operations (`debug.py`)
Testing and monitoring:
- `NOOP`: No-operation command for testing / heartbeat

### CacheBlend v1 (`blend.py`)
First-generation CacheBlend operations (pre-computed lookup / retrieve, final store, and blend-scoped KV cache registration):
- `CB_LOOKUP_PRE_COMPUTED`
- `CB_STORE_PRE_COMPUTED`
- `CB_RETRIEVE_PRE_COMPUTED`
- `CB_STORE_FINAL`
- `CB_REGISTER_KV_CACHE`
- `CB_UNREGISTER_KV_CACHE`

### CacheBlend v2 (`blend_v2.py`)
Second-generation CacheBlend lookup + retrieve:
- `CB_LOOKUP_PRE_COMPUTED_V2`
- `CB_RETRIEVE_PRE_COMPUTED_V2`

### CacheBlend v3 (`blend_v3.py`)
Third-generation CacheBlend with rope state and unified lookup:
- `CB_REGISTER_ROPE_V3` / `CB_UNREGISTER_ROPE_V3`
- `CB_RETRIEVE_PRE_COMPUTED_V3`
- `CB_UNIFIED_LOOKUP`

### Observability (`observability.py`)
Server-side observability events:
- `REPORT_BLOCK_ALLOCATION`

### P2P (`p2p.py`)
Peer-to-peer transfer operations:
- `P2P_LOOKUP_AND_LOCK`
- `P2P_QUERY_LOOKUP_RESULTS`
- `P2P_UNLOCK_OBJECTS`

## Handler Types

- `HandlerType.SYNC`: Fast operations that run directly in the main loop
- `HandlerType.BLOCKING`: Operations that may block, run in a thread pool
- `HandlerType.NON_BLOCKING`: Not yet supported (for future async handlers)
