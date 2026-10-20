# Programmable KV cache coverage and discussion

The delivery lets applications control KV cache behavior through the already merged KvHintsEnvelope/KvHintAction, with complete engine execution and runnable examples.
Retention, movement, and quota/score are optional algorithm examples.
No custom hint registration/handler framework is in scope.
Names below denote responsibilities rather than finalized action payloads or Python classes.

The primary first-release path is lightweight orchestrator → merged envelope → native vLLM interpretation → real Mooncake calls → completion and actual reuse.
Preserve KVCR's opaque forwarding/interpretation path as well.
Each entire envelope selects one path; real Dynamo integration and mixed-owner envelopes are subsequent work in `integration-paths.md`.

## What the local discussions establish

The discussion files below are historical sources retained in commit `ac5094f177`, preceding removal of the local notes.

| Local evidence | Design implication | Detail still requiring a choice |
| --- | --- | --- |
| `chats.md`, August 15: Yifan supports an engine-local SessionPrefixIndex; Karen describes global and per-engine indexes | Include a local index that scheduler and offload consumers can use; keep global routing outside the engine | Compact representation, bounded history, pruning |
| `chats.md`, August 18–19: new cache-hit sessions are not captured by first-writer BlockStored | Track and report association changes separately from physical storage | Confirmed: incremental association events + state queries |
| `chats.md`, August 20: retention must also accept standalone hashes/session IDs | Expose usable request-free Retain and other control operations | Concrete transport methods and target scope |
| `chats.md`, August 22: flexible/versioned envelope for different action types | Consume the merged envelope and execute the application's supported KV commands | Concrete action payloads and completion semantics |
| `notes.md`: SessionAwareManager includes session relationships, inverse lookup, TTLs, edits, and guarded KVCacheManager APIs | Provide coherent engine logical state and guarded physical operations, not scattered policy-local maps | Confirmed: explicit state commands are separate from policy signals |
| RFC #57103: shared observations, proposals, composition, approval, completion | Implement a reusable engine execution contract underneath all policies/adapters | Conflict rules and reference policy parameters |

Historical Pin mentions do not override the user's confirmed September 24 semantics: no public Pin.
Broad context-editing/attention changes are not implied by the notes' references to management edits.

## Audited coverage (updated 2026-10-19)

Each row distinguishes implementation, executed evidence and remaining acceptance.
A source file or mock is not evidence of real movement or serving performance.
Historical Qwen/Store results are located by the evidence index in `validation.md` and compressed artifacts in `benchmarks/programmable_kv_cache/results/`.

| Capability | Implementation and concrete evidence | Remaining acceptance |
| --- | --- | --- |
| Common hints | `api.py`, `controller.py`; `test_kv_hint_control.py`, HTTP `test_api.py`, recorded `http-control.json.gz` | Full workload validation |
| Interpretation ownership | `EngineCore.add_request`, `consume_attached`; engine `test_kv_control.py`, `test_kvcr_tier.py` opaque forwarding | KVCR independent control and G1 directives are unsupported |
| Application-directed behavior | `rollout.py` overlaps inference/control while draining and cleaning results; real 256-rollout/920-call deployment trace and complete policy diagnostic in the October 18 round | Repeated deployment comparisons, adverse forecasts and merged recovery completion |
| Session/continuation state | `state.py`, `request_observer.py`; engine tests and real `lifecycle.py` root/branches/external sharing | Other models/layouts |
| Shared-content membership | Cache-hit membership and reverse-index tests; bounded shared cold-checkpoint recovery after original owner cleanup; full policy diagnostic has zero backend misses | Other models/layouts and ordinary repeated movement comparisons |
| Request-free retention | Real GPU queries observe shared priority 90→20→0 across task-tree cleanup and independent expiry in `lifecycle.json.gz` | Cold-tier retention remains unsupported |
| Explicit reclamation | `BlockPool.evict_unreferenced`; prefix-cache active-reference tests; recorded 4-copy HTTP eviction | Pressure comparisons |
| Demotion lifecycle | `transfers.py` and CPU/Store adapters; queued source-reference tests; recorded real CPU/Store copies | Full failure workload |
| Proactive preparation | Transfer leases, all-rank publication; Store/CPU suites; real `lifecycle.py` duplicate COPY/resume race hits 784 tokens and cancelled preparation falls back correctly | Full outage/restart loop (7.9) |
| Cross-engine reuse | Unified `http_control.py --destination-url`; actual destination hits in `functional/` and both merged CLI rounds | Other models/layouts |
| Complete native G2 path | Legacy actions and version-1.0 primitives have real Store/Qwen evidence; request-attached and opt-in native-policy G2 traces pass on rfc-hint | Broader workload validation |
| Programmable local policy | At utilization 0.5 and 256 rollouts, quota 30/1 and 120/30 finish in 1,898.86 s and 1,369.48 s versus native 1,994.30 s (+5.0%/+45.6%, one run each); the latter has 2 versus 9 preemptions and lower tails for all 32 families | Repeated deployment cache-fit/pressure and complete forecast sensitivity |
| Multi-policy composition | Composer and shared leases; unit coverage plus direct retention/movement/composed and quota-only ablations | Oracle/adverse workloads (7.4) remain unexecuted |
| Request/KV cooperation | Quota scheduler checks preserve priority and bounded waiting; deployment runs record phases, per-family latency and zero per-request control RPCs | Repeated fairness/admission and idle/noop overhead matrix (7.5) |
| Logical lifecycle completion | Real root+B/C cleanup preserves the independently owned external task and its GPU priority | Broader pressure/failure matrix |
| Closed-loop control | Current bounded GPU residency/priority queries, association/session revisions, explicit result ACK and incarnation-checked cancellation | Full recovery loop (7.9); cold residency is not advertised |
| Recovery | `functional/` actual Store outage has no false residency, recomputation matches, restart rejects old epoch and reconstructs logical state | Unified-CLI full recovery completion after fixing early Store-segment-host shutdown; user deferred the remaining quality run |
| Weight compatibility | `functional/weight-update.json.gz` actual norm parameter update/rollback passes with async control drained; new namespace cannot reuse old KV, rollback restores reuse | Full optimizer/trainer integration (outside first-release acceptance) |

Runtime restrictions: general CPU control requires one block per chunk; native CPU Share is unsupported.
Mooncake supports full primary boundaries and PCP=DCP=1.
Cold Retain/delete, SimpleCPU control, KVCR request-free actions and mixed-owner envelopes are unsupported.
Evaluation conclusions are limited to the tested local Qwen configuration.
`verl_rollout.py` now executes the actual verl ToolAgentLoop with real model/tool calls, response masks, logprobs and rewards.
The current 256-rollout native trace and ordinary quota comparison use utilization 0.5, peak at full KV occupancy and incur real preemptions.
Repeated deployment-scale acceptance remains tracked separately and is not inferred from the earlier 32-rollout artificial-budget results.

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

These are contracts, not a proposed parallel allocator or a requirement to add eight heavyweight managers.
Implementation can group responsibilities while preserving ownership and extension boundaries.

## Additional complete example flows

1. **Direct cache controller:** no local policies; submit session-addressed Retain, inspect the result, Demote during a gap, Prefetch, resume, and Release.
2. **Branch-aware controller:** establish H, H+B, H+C; operate on B without losing C; finish a task tree while another task still associates with H.
3. **Range lifecycle:** one shared system prefix, medium-priority history, and a low-priority tail; independently update and expire declarations.
4. **Cross-engine handoff:** A saves, B prepares, the router waits for B's actual readiness; failure/restart has an explicit fallback and does not erase A.
5. **Application-directed rollout preparation:** an application selects the next rollout's prefix, submits supported Retain/Prefetch hints, observes results, and chooses when/where to resume without registering a new hint type.
6. **Composable pressure response:** retention, movement, and quota compete on a shared snapshot; show duplicate coalescing, clipping/deferral, and physical accounting independently of the algorithms' performance.
7. **Control failure and retry:** reordered/duplicate messages, unknown versions, partial backend results, cancellation, and stale completion publication.
8. **State reconstruction:** cache-hit associations, event gaps, queried logical state, engine restart, and current-request full-report fallback.
9. **Multi-round rollout benchmark:** controllable shared families, pauses and finite capacity, comparing router-only and optional local policies.
10. **Weight-update correctness:** hold identical token input while changing the model version and verify that old local/remote copies are not reused.

Examples should share a small controller/workload helper instead of each implementing a separate protocol client.
The demo IDs are not public action names.

## Validation configuration

- Local debug: **Qwen3.8-27B, TP=2**, as corrected by the user.
- Acceptance environment: GPUs 0/3 with local Qwen TP=2, including repeated pressure/fit and failure validation.
- Model downloads: `/home/youwei/bzh/model/download.py`.
- Dataset downloads: `/home/youwei/bzh/dataset/download_dataset.py`.
- The first delivery includes a real Mooncake cross-engine run; scripts and mocks alone do not establish that result.

## Remaining delivery gaps

Payloads, whole-envelope routing, revisions and snapshot/follow-growth scope are specified in `implementation.md` and implemented in the native modules.
The open items are the incomplete acceptance rows above and the task list, especially the complete performance/failure matrix.

The `rfc-hint` implementation adds the public primitive profile and the concrete request lifecycle described in `vllm/v1/kv_hints/README.md`.
Result scope now includes per-target transfer IDs/generations.
These additions complete alignment tasks 9.1–9.6 and the previously unexecuted native-policy G2 acceptance (7.11); they do not by themselves complete rollout/RL/fairness requirements.
The October 11 phase adds actual verl execution, current GPU observations, real lifecycle examples and weight transitions; consult its executed results rather than the earlier counts.
