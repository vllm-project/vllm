# MoRIIO Connector: Transfer ID Design

**Status:** Implementation Planning
**Created:** 2026-01-23
**Related PRs:** #27987, #29665 (NIXL)

## Overview

This design document describes the solution for fixing the MoRIIO connector after PR #27987 introduced different internal request_ids for Prefill and Decode instances. The solution introduces a `transfer_id` concept to separate KV transfer coordination (between P/D instances) from internal scheduler request tracking.

## Background

### The Problem

After #27987, Prefill and Decode instances append random suffixes to frontend-provided request_ids to create unique internal request_ids:
- Frontend provides: `"user-request-123"`
- Prefill creates: `"user-request-123-abc123"` (random suffix)
- Decode creates: `"user-request-123-def456"` (different random suffix)

The MoRIIO connector broke because:
1. Decode finishes fetching KV cache blocks
2. Decode sends completion notification using its internal request_id: `"user-request-123-def456"`
3. Prefill receives the notification but can't match it to its internal request_id: `"user-request-123-abc123"`
4. KV cache blocks are never freed on Prefill

### Why MoRIIO Differs from NIXL

The NIXL connector was fixed proactively in PR #29665 by adding `remote_request_id` to `kv_transfer_params`. Prefill includes its internal request_id when sending transfer parameters to Decode, so Decode can use that exact ID in completion notifications.

MoRIIO's WRITE mode has a different architecture:
- The proxy dispatches to **both** Prefill and Decode in parallel (for performance)
- The proxy constructs `kv_transfer_params` itself before Prefill's internal request_id is available
- This prevents using the NIXL approach directly

## Solution Strategy

Introduce a **`transfer_id`** (from proxy's request UUID with "xfer-" prefix) that both P and D use for KV transfer coordination, while maintaining internal request_ids for scheduler operations.

**Key principle:** The connector acts as a translator between two domains:
- **External (P↔D coordination):** Uses `kv_transfer_params["transfer_id"]` (type: `TransferId`, format: `"xfer-{uuid}"`)
- **Internal (Connector↔Scheduler):** Uses `Request.request_id` (type: `ReqId`, format: `"cmpl-{uuid}-{suffix}"`)

## Architecture

```
Proxy (request_id="550e8400-e29b-41d4-a716-446655440000")
  |
  |-- Creates transfer_id="xfer-550e8400-e29b-41d4-a716-446655440000"
  |
  ├─> Prefill (transfer_id="xfer-550e8400-...", internal_id="cmpl-...-abc123")
  |     ↓ Writes KV cache blocks
  |     ↓ Sends completion using transfer_id
  |
  └─> Decode  (transfer_id="xfer-550e8400-...", internal_id="cmpl-...-def456")
        ↓ Receives notification using transfer_id
        ↓ Translates to internal_id for scheduler

P↔D communication: Uses transfer_id (xfer-* format)
Connector↔Scheduler: Uses internal request_id (cmpl-*-suffix format)
```

### Data Flow

```
Proxy adds transfer_id to kv_transfer_params
    ↓
Scheduler extracts and passes to connector
    ↓
Connector establishes mapping: transfer_id ↔ internal request_id
    ↓
Worker uses transfer_id for P/D coordination
    |
    ├─ WriteTask, LayerTransferPlan (identified by transfer_id)
    ├─ send_notify messages (contain transfer_id)
    └─ Remote allocation tracking (keyed by transfer_id)
    ↓
Worker returns finished transfer_ids
    ↓
Connector translates: transfer_id → internal request_id
    ↓
Scheduler receives internal request_ids
```

## Implementation Plan

### Phase 1: Proxy Changes

**File: `examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py`**

In `handle_request()`, after generating the request UUID, create a `transfer_id` and add it to both Prefill and Decode `kv_transfer_params`:

```python
# Line ~213 - Generate IDs
request_id = str(uuid.uuid4())
transfer_id = f"xfer-{request_id}"

# Line ~242-250 - Add to Prefill request
req_data_to_prefill["kv_transfer_params"]["transfer_id"] = transfer_id

# Line ~267-276 - Add to Decode request
req_data["kv_transfer_params"] = {
    "transfer_id": transfer_id,
    "do_remote_decode": False,
    "do_remote_prefill": True,
    ...
}
```

**Key points:**
- Generate `transfer_id = f"xfer-{request_id}"` once in `handle_request()`
- Both P and D receive the same `transfer_id` via `kv_transfer_params`

### Phase 2: Data Structure Changes

**File: `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py`**

1. **Add TransferId type alias:**
   ```python
   Transfer = tuple[int, float]
   EngineId = str
   ReqId = str  # Internal scheduler request ID
   TransferId = str  # NEW - KV transfer coordination ID (shared by P/D)
   ```

2. **Add transfer_id to ReqMeta:**
   ```python
   @dataclass
   class ReqMeta:
       transfer_id: TransferId  # NEW - for P/D coordination
       local_block_ids: list[int]
       remote_block_ids: list[int]
       # ... rest of fields
   ```

3. **Update MoRIIOConnectorMetadata.add_new_req():**
   ```python
   def add_new_req(
       self,
       request_id: ReqId,  # Internal request_id
       local_block_ids: list[int],
       kv_transfer_params: dict[str, Any],
       write_mode=False,
   ):
       transfer_id: TransferId = kv_transfer_params["transfer_id"]
       _req = ReqMeta(
           transfer_id=transfer_id,
           local_block_ids=local_block_ids,
           # ... rest
       )
   ```

4. **Update WriteTask and LayerTransferPlan:**
   ```python
   @dataclass
   class WriteTask:
       transfer_id: TransferId  # Renamed from request_id
       dst_engine_id: str
       # ... rest

   @dataclass
   class LayerTransferPlan:
       transfer_id: TransferId  # Renamed from request_id
       layer_name: str
       # ... rest
   ```

### Phase 3: Worker Mapping Infrastructure

**File: `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`**

Add bidirectional mapping in `MoRIIOConnectorWorker.__init__()`:

```python
def __init__(self, ...):
    # Existing fields
    self._recving_transfers: dict[TransferId, list] = {}
    self._recving_transfers_callback_addr: dict[TransferId, tuple] = {}

    # NEW: Bidirectional mapping
    self._transfer_id_to_request_id: dict[TransferId, ReqId] = {}
    self._request_id_to_transfer_id: dict[ReqId, TransferId] = {}
```

**Mapping lifecycle:**
- **Established:** When worker receives request to transfer (in `save_kv_layer()`, `start_load_kv()`)
- **Used:** When translating worker results to scheduler IDs (in `get_finished()`, `_pop_done_transfers()`)
- **Cleaned up:** When transfer completes and results are returned to scheduler

### Phase 4: External P/D Coordination

**File: `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`**

Key changes for P/D coordination (all use `transfer_id`):

1. **send_notify_block()** - Change message protocol field from `"req_id"` to `"transfer_id"`
2. **save_kv_layer()** - Establish mapping and use `transfer_id` for writes
3. **start_load_kv()** - Establish mapping and use `transfer_id` for reads
4. **schedule_write_blocks()** - Accept `transfer_id` parameter
5. **_read_blocks()**, **_write_blocks_for_req()** - Use `transfer_id` for tracking
6. **_pop_done_transfers()** - Translate `transfer_id` → internal request_id before returning
7. **get_finished()** - Translate worker results from `transfer_id` to internal request_id

Example - `get_finished()` translation:

```python
def get_finished(self) -> tuple[set[ReqId], set[ReqId]]:
    done_sending: set[ReqId] = set()
    done_recving: set[ReqId] = set()

    if self.is_producer:
        # Worker returns transfer_ids, we translate to internal
        done_transfer_ids: set[TransferId] = self.moriio_wrapper.pop_finished_transfer_ids()
        for transfer_id in done_transfer_ids:
            internal_id = self._transfer_id_to_request_id[transfer_id]
            done_sending.add(internal_id)

        # Clean up mappings
        for transfer_id in done_transfer_ids:
            internal_id = self._transfer_id_to_request_id.pop(transfer_id)
            self._request_id_to_transfer_id.pop(internal_id)

    # ... similar for done_recving

    return done_sending, done_recving
```

### Phase 5: Engine Layer Changes

**File: `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py`**

Update engine layer to use `transfer_id` semantics:

1. **Rename tracking lists:**
   ```python
   self.done_transfer_ids: list[TransferId] = []
   self.done_write_cache_transfer_ids: list[TransferId] = []
   self.done_remote_allocate_req_dict: dict[TransferId, RemoteAllocInfo] = {}
   ```

2. **Update message handling:**
   ```python
   def _handle_structured_message(self, data: dict):
       transfer_id: TransferId = data["transfer_id"]  # Changed from "req_id"
       # ... rest
   ```

3. **Rename accessor methods:**
   ```python
   def pop_finished_transfer_ids(self) -> set[TransferId]:  # Was pop_finished_req_ids
   def pop_finished_write_transfer_ids(self) -> set[TransferId]:  # Was pop_finished_write_req_ids
   ```

4. **Update all `task.request_id` → `task.transfer_id`**

### Phase 6: Scheduler-Side Connector

**File: `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`**

Scheduler-side methods remain unchanged - they continue to work with internal `request_id`:

- `_reqs_need_recv`, `_reqs_need_save`, `_reqs_need_send` stay keyed by internal request_id
- `build_connector_meta()` works with internal request_ids from scheduler
- `meta.add_new_req()` receives internal request_id as first parameter

The connector serves as the translation layer between scheduler and worker domains.

## Type System

The solution uses Python type aliases to distinguish the two ID types:

- **`ReqId`**: Internal scheduler request ID (e.g., `"cmpl-550e8400-...-abc123"`)
  - Format: `"cmpl-{uuid}-{random_suffix}"`
  - Used for scheduler↔connector interface
  - Unique per instance (P and D have different values)

- **`TransferId`**: KV transfer coordination ID (e.g., `"xfer-550e8400-..."`)
  - Format: `"xfer-{uuid}"` (generated by proxy)
  - Used for P↔D coordination
  - Same value on both P and D for a given transfer

This makes the separation explicit in function signatures and helps catch bugs at review time. The different prefixes ("xfer-" vs "cmpl-") also make it immediately obvious which ID type is being used when reading logs.

## Critical Files Modified

1. **Proxy:** `examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py`
   - Add transfer_id to kv_transfer_params

2. **Common:** `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py`
   - Add `TransferId` type alias
   - Update data structures (ReqMeta, WriteTask, LayerTransferPlan)

3. **Connector:** `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`
   - Add mapping infrastructure
   - Update worker methods to use transfer_id
   - Add translation in get_finished()

4. **Engine:** `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py`
   - Rename tracking structures
   - Update message protocol
   - Rename accessor methods

## Testing Strategy

### Unit Tests
- Update tests that create ReqMeta objects to include transfer_id
- Update tests that mock kv_transfer_params to include transfer_id
- Verify send_notify messages contain "transfer_id" field (not "req_id")
- Test mapping establishment and cleanup

### Integration Tests
- Test WRITE mode with parallel P/D dispatch
- Test READ mode (should continue working)
- Verify transfer completion notifications work end-to-end
- Test with different internal request_ids on P and D

### Manual Verification
- Run toy proxy server with Prefill/Decode instances
- Send requests and verify successful completion
- Check logs for transfer_id in notification messages
- Verify scheduler receives correct internal request_ids
- Confirm KV cache blocks are freed properly on Prefill

## Alternatives Considered

### Alternative 1: NIXL-style remote_request_id

Add Prefill's internal request_id to kv_transfer_params after Prefill starts, requiring Decode to wait.

**Rejected because:**
- Forces serialization, losing WRITE mode's parallel dispatch performance benefit
- Requires significant proxy changes to buffer/wait for Prefill response
- Goes against MoRIIO's design philosophy of parallel P/D operation

### Alternative 2: Scheduler-level request ID mapping (previous fix)

Add bidirectional mapping in scheduler to resolve external↔internal request_ids.

**Rejected because:**
- Request.external_req_id should only be used by the frontend, not core engine or connectors
- Adds 66 lines of complex mapping logic to core scheduler
- O(n) lookup complexity with fallback heuristics
- Couples scheduler to KV transfer implementation details
- Affects all schedulers, not just those using MoRIIO

### Alternative 3: Remove random suffixes from request_ids

Revert #27987 changes to make P and D use same internal request_ids.

**Rejected because:**
- #27987 fixed important correctness issues
- Random suffixes are part of v1 request ID design
- Would break other systems relying on unique internal request_ids

## Future Work

### Deferred in Initial Implementation

Variable renaming from local `req_id`/`request_id` variables to `transfer_id` throughout the codebase. The initial fix focuses on:
- Type annotations (TransferId vs ReqId)
- Function signatures
- Data structure fields
- ZMQ message protocol fields

But defers renaming loop variables and local identifiers where the meaning is clear from context. This reduces churn in the bugfix PR.

### Potential Enhancements

1. **Protocol versioning**: Add version field to ZMQ messages for future compatibility
2. **Transfer ID validation**: Add assertions to catch misuse of transfer_id vs request_id
3. **Metrics**: Track mapping table size and translation performance
4. **Documentation**: Add sequence diagrams showing full P/D coordination flow

## References

- **PR #27987**: Add random suffix to internal request IDs
- **PR #29665**: NIXL connector fix with remote_request_id
- **Issue**: MoRIIO connector broken after #27987 (internal tracking)
