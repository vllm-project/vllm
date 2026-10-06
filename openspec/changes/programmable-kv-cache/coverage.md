# Programmable KV cache coverage and discussion

The delivery lets applications control KV cache behavior through the already-
merged KvHintsEnvelope/KvHintAction, with complete engine execution and runnable
examples. Retention, movement, and quota/score are optional algorithm examples.
No custom hint registration/handler framework is in scope. Names below denote
responsibilities rather than finalized action payloads or Python classes.

The primary first-release path is lightweight orchestrator → merged envelope →
native vLLM interpretation → real Mooncake calls → completion and actual reuse.
Preserve KVCR's opaque forwarding/interpretation path as well. Each entire
envelope selects one path; real Dynamo integration and mixed-owner envelopes are
subsequent work in `docs/programmable-kvc/task2do.md`.

## What the local discussions establish

| Local evidence | Design implication | Detail still requiring a choice |
| --- | --- | --- |
| `chats.md`, August 15: Yifan supports an engine-local SessionPrefixIndex; Karen describes global and per-engine indexes | Include a local index that scheduler and offload consumers can use; keep global routing outside the engine | Compact representation, bounded history, pruning |
| `chats.md`, August 18–19: new cache-hit sessions are not captured by first-writer BlockStored | Track and report association changes separately from physical storage | Confirmed: incremental association events + state queries |
| `chats.md`, August 20: retention must also accept standalone hashes/session IDs | Expose usable request-free Retain and other control operations | Concrete transport methods and target scope |
| `chats.md`, August 22: flexible/versioned envelope for different action types | Consume the merged envelope and execute the application's supported KV commands | Concrete action payloads and completion semantics |
| `notes.md`: SessionAwareManager includes session relationships, inverse lookup, TTLs, edits, and guarded KVCacheManager APIs | Provide coherent engine logical state and guarded physical operations, not scattered policy-local maps | Confirmed: explicit state commands are separate from policy signals |
| RFC #57103: shared observations, proposals, composition, approval, completion | Implement a reusable engine execution contract underneath all policies/adapters | Conflict rules and reference policy parameters |

Historical Pin mentions do not override the user's confirmed September 24
semantics: no public Pin. Broad context-editing/attention changes are not implied
by the notes' references to management edits.

## Audited coverage (2026-10-06)

Each row distinguishes implementation, executed evidence and remaining acceptance.
A source file or mock is not evidence of real movement or serving performance.
Historical Qwen/Store results below are the October 1 records in `validation.md`
and compressed artifacts in `benchmarks/programmable_kv_cache/results/`.

| Capability | Implementation and concrete evidence | Remaining acceptance |
| --- | --- | --- |
| Common hints | `api.py`, `controller.py`; `test_kv_hint_control.py`, HTTP `test_api.py`, recorded `http-control.json.gz` | Full workload validation |
| Interpretation ownership | `EngineCore.add_request`, `consume_attached`; engine `test_kv_control.py`, `test_kvcr_tier.py` opaque forwarding | KVCR independent control and G1 directives are unsupported |
| Application-directed behavior | `http_control.py` executes explicit Retain/Release/Demote/Prefetch and polls outcomes; recorded CPU/Mooncake runs | Multi-round rollout controller (7.8) |
| Session/continuation state | `state.py`, `request_observer.py`; state and engine control suites, recorded `request-associations.json.gz` | Complete branch/range workflow (7.7) |
| Shared-content membership | `test_cache_hit_request_adds_logical_membership_without_allocating_another_copy`; state reverse-membership tests | Broader serving workload |
| Request-free retention | `retention.py`, `KVRetentionPriority`; prefix-cache tests for independent expiry/eviction, direct HTTP run | Complete range and task-tree example (7.7) |
| Explicit reclamation | `BlockPool.evict_unreferenced`; prefix-cache active-reference tests; recorded 4-copy HTTP eviction | Pressure comparisons |
| Demotion lifecycle | `transfers.py` and CPU/Store adapters; queued source-reference tests; recorded real CPU/Store copies | Full failure workload |
| Proactive preparation | Transfer leases, all-rank publication; Store/CPU scheduler and worker suites; recorded 784-token reuse | Executed closed-loop resume race (7.9) |
| Cross-engine reuse | `http_control.py --destination-url`; recorded `mooncake-cross-engine.json.gz` | Other models/layouts |
| Complete native G2 path | Legacy actions and version-1.0 primitives have real Store/Qwen evidence; request-attached and opt-in native-policy G2 traces pass on rfc-hint | Broader workload and external model validation |
| Programmable local policy | `policy.py`, `policy_runtime.py`; derived primitive tests and real tool-wait/resume Mooncake execution on rfc-hint | Policy performance/pressure comparisons |
| Multi-policy composition | Composer and common transfer leases; tests for duplicate proposals, finite capacity and independent cancellation | Policy ablations and adverse workloads (7.4) |
| Request/KV cooperation | Real quota/scheduler tests check priority and maximum waiting; the October 6 Qwen quota-enabled correctness run is recorded separately | Fairness, admission-delay and overhead matrix (7.5) |
| Logical lifecycle completion | `finish_task` and independent declaration tests in state/control/cache suites | Full runnable workflow (7.7) |
| Closed-loop control | Common operation results/events and association queries; direct transfer polling | Physical residency reporting (6.3) and full recovery loop (7.9) |
| Recovery | Epoch checks, bounded replay, HTTP history-loss test; October 6 actual restart rejects the old epoch and drops local records | Full failure injection, prefetch/resume races and fallback loop (7.9) |
| Weight compatibility | Startup namespace includes weight version; `test_engine_epochs_differ_but_compatible_dp_ranks_share_namespace` | Version changes are rejected while control is enabled; live local/external invalidation is not integrated (7.2) |

Runtime restrictions: general CPU control requires one block per chunk; native
CPU Share is unsupported. Mooncake supports full primary boundaries and PCP=DCP=1.
Cold Retain/delete, SimpleCPU control, KVCR request-free actions and mixed-owner
envelopes are unsupported. Local Qwen results do not validate GPT-OSS-120B or
DeepSeek-V4-Flash. No verl Agent Loop integration or pressure workload is delivered.

## Engine responsibilities behind application control

| Responsibility | Receives | Produces | Authority |
| --- | --- | --- | --- |
| Hint ingestion/dispatch | Existing envelope and caller/request context | Envelope delivered to one selected consumer | Receipt does not approve resources; backend payload is not parsed as native |
| Native interpretation | Native-selected actions | Validated supported native commands or policy signals | No automatic fallback reinterpretation of backend actions |
| Target resolver | Selector, continuation/range, compatibility context | Explicit per-group content and known/missing copies | Cannot manufacture residency from session names |
| State manager | Explicit logical updates and observed cache activity | Associations, declarations, revisions | Does not allocate merely because content is associated |
| Policy | Signals, targets, read-only current observations | Proposals and optional scheduling guidance | No direct allocator/queue mutation |
| Composer | Explicit commands and policy proposals | Deduplicated plan with dependencies and caller attribution | Preserves explicit command meaning |
| Engine approval | Plan and current physical state | Approved scope, reservations, holds, or refusal | Owns resource commitment |
| Backend adapter | Approved operation/buffers and capability requirements | Progress, completion, failure | Uses connector keys/layout; cannot expand scope |
| Result publisher | Operation and association/residency records | Queries and events | Reports observed outcomes, not predicted success |

These are contracts, not a proposed parallel allocator or a requirement to add
eight heavyweight managers. Implementation can group responsibilities while
preserving ownership and extension boundaries.

## Additional complete example flows

1. **Direct cache controller:** no local policies; submit session-addressed
   Retain, inspect the result, Demote during a gap, Prefetch, resume, and Release.
2. **Branch-aware controller:** establish H, H+B, H+C; operate on B without losing
   C; finish a task tree while another task still associates with H.
3. **Range lifecycle:** one shared system prefix, medium-priority history, and a
   low-priority tail; independently update and expire declarations.
4. **Cross-engine handoff:** A saves, B prepares, the router waits for B's actual
   readiness; failure/restart has an explicit fallback and does not erase A.
5. **Application-directed rollout preparation:** an application selects the next
   rollout's prefix, submits supported Retain/Prefetch hints, observes results,
   and chooses when/where to resume without registering a new hint type.
6. **Composable pressure response:** retention, movement, and quota compete on a
   shared snapshot; show duplicate coalescing, clipping/deferral, and physical
   accounting independently of the algorithms' performance.
7. **Control failure and retry:** reordered/duplicate messages, unknown versions,
   partial backend results, cancellation, and stale completion publication.
8. **State reconstruction:** cache-hit associations, event gaps, queried logical
   state, engine restart, and current-request full-report fallback.
9. **Multi-round rollout benchmark:** controllable shared families, pauses and
   finite capacity, comparing router-only and optional local policies.
10. **Weight-update correctness:** hold identical token input while changing the
    model version and verify that old local/remote copies are not reused.

Examples should share a small controller/workload helper instead of each
implementing a separate protocol client. The demo IDs are not public action names.

## Validation configuration

- Local debug: **Qwen3.8-27B, TP=2**, as corrected by the user.
- Complete evaluation elsewhere: GPT-OSS-120B and DeepSeek-V4-Flash.
- Model downloads: `/home/youwei/bzh/model/download.py`.
- Dataset downloads: `/home/youwei/bzh/dataset/download_dataset.py`.
- The first delivery includes a real Mooncake cross-engine run; scripts and mocks
  alone do not establish that result.

## Remaining delivery gaps

Payloads, whole-envelope routing, revisions and snapshot/follow-growth scope are
specified in `implementation.md` and implemented in the native modules. The open
items are the incomplete acceptance rows above and the task list, including
residency reporting, real workload integration and live weight invalidation.

The `rfc-hint` implementation adds the public primitive profile and the concrete
request lifecycle described in `vllm/v1/kv_hints/README.md`. Result scope now includes
per-target transfer IDs/generations. These additions complete alignment tasks
9.1–9.6 and the previously unexecuted native-policy G2 acceptance (7.11); they do
not complete the remaining rollout/RL/residency/fairness requirements.
