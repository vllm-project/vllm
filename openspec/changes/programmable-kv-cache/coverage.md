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

## Capability, code boundary, and demonstration

| Capability | Required implementation boundary | Runnable demonstration and assertion |
| --- | --- | --- |
| Common hints | Existing envelope + action validation/dispatch | Submit the same supported hint through Python/HTTP and with/without inference; observe the same semantics |
| Interpretation ownership | Whole-envelope native/backend dispatch | Native payload reaches vLLM semantics; a KVCR payload remains opaque to native parsing; no envelope is executed by both |
| Application-directed behavior | Existing hint contract + common engine execution | Application chooses retention/movement targets and timing, submits hints, and reacts to actual results without a custom engine plugin |
| Session/continuation state | Engine-local index + guarded state commands | Record a parent and two branches; resolve each explicit range and shared ancestor correctly |
| Shared-content membership | Cache-hit/computation hooks + reverse association lookup | Session B hits A's prefix; both associations exist, physical capacity is counted once |
| Request-free retention | Declaration records + eviction candidate selection | Apply/update/release independent range TTLs after inference ends; expiry changes priority without deleting B's declaration |
| Explicit reclamation | KVCacheManager/BlockPool authority | Evict an inactive selected copy; referenced blocks are reported blocked/partial and remain usable |
| Demotion lifecycle | Operation manager + CPU/Store adapter | Cold save succeeds, active refs delay source reclaim, and the result exposes both milestones |
| Proactive preparation | Destination approval + request-free worker progress | Prefetch completes before resume; a racing resume shares the in-flight transfer |
| Cross-engine reuse | Router-directed A/Store/B operations | A publishes compatible content, B prefetches, actual B GPU reuse is measured, A is not implicitly deleted |
| Complete native G2 path | Lightweight orchestrator + ingress + native semantics + real Store calls | Correlate the original action with approval, Put/Get jobs, completion, and actual cache reuse |
| Programmable local policy | PolicyView + proposal interface | The same tool-wait signal produces no action without a consumer, then a bounded plan with a configured policy |
| Multi-policy composition | Replaceable composer + common resource account | Duplicate prefetch proposals share physical work; distinct targets compete for finite capacity |
| Request/KV cooperation | Scheduler admission/score extension | Quota guidance and request growth compete with movement under one capacity account; existing preemption remains |
| Logical lifecycle completion | Task-tree cleanup + declaration ownership | End one branch/task while an unrelated session retains the shared prefix |
| Closed-loop control | Correlated results + capabilities + association/residency events | A router retries a deferred command using current observations and never equates receipt with readiness |
| Recovery | Epochs/revisions + bounded replay/query + stale completion handling | Restart or event loss invalidates stale locations; unavailable KV falls back to normal recomputation |
| Weight compatibility | Local/external version invalidation | Weight update prevents reuse/publication of obsolete KV even when bytes remain in the Store |

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

## Pending interface decisions

- Concrete supported action payloads within the merged hint contract.
- Association delta encoding and recovery/query contract.
- Snapshot/follow-growth selection, ordering/retry semantics, plan partial
  effects, and exact backend/group capability coverage.
