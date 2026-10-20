# Orchestrator-to-Mooncake implementation path

## Delivery contract

The primary executable example follows the path requested in the latest discussion:

```text
orchestrator chooses a supported action, target, and parameters
  → existing KvHintsEnvelope / KvHintAction
  → vLLM transport and consumer dispatch
  → vLLM interprets native action payload / optional policy consumes signal
  → target resolution and common resource approval
  → vLLM connector invokes supported Mooncake Store APIs
  → transfer completion and actual residency / source reclamation
  → correlated result and events returned to the orchestrator
```

This is the G2-backed complement to a G1-only Retain example.
It uses the merged hint record, not a custom hint registration system.
The complete PR retains the broader session, retention, movement, policy-composition, and scheduler scope.
The first release uses a runnable lightweight orchestrator, as confirmed by the user.
It does not require Dynamo modifications; Dynamo is a documented follow-up.

The latest diagram also preserves:

```text
orchestrator → same envelope → transport/dispatch
  → KVCR submit_hint (payload opaque to native vLLM interpreter)
  → KVCR interpretation and existing query/deliver integration
  → backend transfer/completion
```

The second path has its own action meanings and supported lifecycle.
Request-free Mooncake control does not imply request-free KVCR control already exists.

An entire envelope goes to exactly one interpretation path in the first release.
The invocation/deployment context selects native or backend ownership without adding wire fields.
Preserve action order and payload values; do not split or broadcast an envelope.
Unsupported actions/versions are handled by the selected consumer rather than retried through the other interpreter.
Applications needing both paths submit separate envelopes.

## Who owns each decision

| Component | Responsibility |
| --- | --- |
| Orchestrator | Choose supported actions, logical targets, timing and destination engine; consume outcomes |
| Transport/dispatch | Preserve the merged records; validate wrapper structure; select the intended consumer |
| Native vLLM interpretation | Define/validate native payload meaning; resolve targets; implement explicit commands; invoke optional local policies for signals |
| KVCR interpretation | Define/validate its own payloads after forwarding; preserve existing fetch/source-selection behavior |
| vLLM resource operations | Validate current GPU blocks and references; reserve/release buffers; control safe mutation and actual readiness |
| Mooncake Store | Validate its own keys/storage/capacity and execute requested supported storage/transfer work |

Native parsing and engine approval are distinct: explicit Demote remains native semantic processing even with no optional policy enabled.
An opt-in policy decides whether a tool-wait signal should produce a Demote proposal.
Both use the same approved execution path.
No backend is granted GPU memory by accepting a hint.

## Proposed concrete implementation surface

These are proposed internal class names/methods for discussion, not implemented code or newly committed wire APIs.
They can be integrated into existing modules where that is simpler; none requires a parallel cache allocator.

| Proposed component | Proposed methods | Integration |
| --- | --- | --- |
| `KvHintDispatcher` | `dispatch(envelope, context)`, `capabilities()` | Common Python/HTTP→EngineCore ingress; routes to native or backend consumer |
| `NativeKVController` | `submit(actions, context)`, `get_operation(id)`, `take_events()` | Native payload validation; logical state; explicit operations and signal delivery |
| `SessionPrefixIndex` | `associate(...)`, `resolve(target)`, `remove_task_tree(...)`, `query(...)` | Shared engine index updated on cache hits/admission; opaque session coordinates remain distinct from hashes |
| `KVPolicy` | `propose(signals, targets, view)` | Existing agreed startup-configured local policy interface |
| `KVPlanComposer` | `compose(explicit, proposals, view)` | Coalescing and conflict handling; preserve explicit action meaning |
| `KVResourceController` | `approve(plan)`, `complete(operation, outcome)`, `cancel(operation)` | Facade over KVCacheManager/BlockPool and connector/worker holds, not a new allocator |
| `MooncakeKVAdapter` | `capabilities()`, `lookup(targets)`, `submit(approved)`, `poll()` | Adapt existing MooncakeStoreScheduler/Worker metadata and transfer paths |

The KVCR route should reuse KVCRSecondaryTierManager rather than introduce a second implementation of its policy interpreter.
When it requests GPU work via integration, it reaches resource mechanisms without a native-payload round trip.

## Current code evidence

- `vllm/v1/kv_hints/protocol.py`: merged envelope/action record fields.
- `rust/src/engine-core-client/src/protocol/kv_hints.rs`: corresponding Rust records.
- `vllm/v1/kv_offload/tiering/kvcr/manager.py`: `KVCRSecondaryTierManager.on_new_request()` calls `submit_hint(request_id=..., hints=msgspec.to_builtins(hint))`; `on_request_finished()` calls `discard_hint(request_id)`.
- `tests/v1/kv_offload/tiering/test_kvcr_tier.py`: existing forwarding, key conversion, load destination, and cleanup coverage to extend.
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`: `batch_is_exist`, `batch_put_from_multi_buffers`, and `batch_get_into_multi_buffers` implement underlying storage/transfer calls; `get_transfer_results()` distinguishes failed receives from completed work.
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`: existing request-oriented allocation/metadata and completed-save reference handling.

These establish reusable execution machinery, not a completed independent action executor or a backend-native implementation of all Retain/Demote guarantees.

## Action-to-backend lowering

| Native operation | vLLM work | Mooncake work | Success evidence |
| --- | --- | --- | --- |
| Retain in GPU | Independent declaration + eviction bias | None required | Declaration effective at reported scope; no residency guarantee |
| Retain in a cold tier | Validate requested retention capability | Only supported backend behavior | Explicit supported/unsupported result; no invented numeric priority/TTL guarantee |
| Demote | Resolve valid source, protect transfer buffers, confirm cold copy, reclaim only eligible GPU entries | Lookup/Put required objects using existing keys/layout | Cold-copy availability and GPU reclamation reported separately |
| Prefetch to GPU | Resolve content, reserve missing destination blocks, hold buffers, register after all required shards complete | Lookup/Get into approved buffers | Reusable destination state, then an actual request cache hit |
| Share | Orchestrator selects destination; distinguish Store accessibility from destination GPU preparation | Make compatible content accessible in the Store | Report exact location/readiness; preserve source |
| Evict/Release | Separate physical-location eviction from declaration cleanup | Invoke selective backend removal only when supported with correct scope | No unrelated shared data deletion; no all-store reset as a per-session eviction |

Backend storage capacity and GPU capacity are separate checks.
GPU approval does not guarantee Put succeeds.
Failure and partial results must release engine holds only when the transfer can no longer access the buffers.

## Primary integration scenario

1. Orchestrator sends inference for a known session/continuation to engine A and records the resulting logical prefix association.
2. During a tool/rollout pause, it sends an explicit Demote hint without a new inference request.
   A's native controller validates the target and derives the save/reclamation operation.
3. vLLM performs real Mooncake Put work, reports cold availability, and separately reports the portion of GPU capacity actually reclaimed.
4. Orchestrator selects B and submits Prefetch for an explicit compatible target.
   B reserves buffers and performs real Get work before its inference request.
5. The result becomes available only after the required groups/ranks complete and the prefix is registered.
   The subsequent inference demonstrates reuse.
6. Repeat with a tool-wait signal and a configured native policy to show optional engine-side decisions reaching the same backend execution boundary.
7. Repeat under capacity shortage, shared-prefix use, transfer failure, and a resume arriving while preparation is still in flight.

The trace must tie together message/action IDs, selected consumer, interpreted operation, approved scope, backend jobs, completion, and observed cache reuse.
Mocks can verify API contracts; full-path acceptance needs real Store calls.

## Dual-path compatibility checks

- Native commands reach native semantics and are not unintentionally forwarded again to KVCR through the existing request hook.
- KVCR-owned payloads preserve IDs/versions/JSON values and do not require native schema recognition.
  Opaque forwarding means semantic preservation, not a promise of identical serialized bytes.
- Forwarding success is not reported as backend completion or GPU readiness.
- Missing/ambiguous consumer selection is observable rather than implicit broadcasting, fallback reinterpretation, or double execution.
- Existing KVCR request cleanup, host-slot accounting, and source-reference callbacks continue to work.
  Internal pin callbacks are not public Pin hints.
- Future KVCR G1 directives use engine GPU resource operations only after concrete integration exists; the diagram's dashed arrow is not claimed as implemented.

## Confirmed first-release choices and follow-ups

1. A runnable lightweight orchestrator is the first-release sender.
   Real Dynamo→vLLM→Mooncake integration is subsequent work.
2. Each envelope selects one interpretation path.
   Both native and KVCR paths remain supported; mixed-owner envelopes are subsequent work.
3. Exact route-selection method/configuration and consumer capability reporting must be specified for the first release without changing the merged records.

This section records the follow-up scope after removal of the separate local notes.
Dynamo integration, mixed-owner envelopes, and concrete KVCR G1 integration remain subsequent work.
Unexecuted first-release performance and failure acceptance remains in `tasks.md`.

The user has allowed related repositories under `../` for required adaptations.
The confirmed no-Mooncake-project-modification boundary still applies.
