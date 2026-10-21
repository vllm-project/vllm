# Programmable KV Cache proposal

## Why

Agentic workloads and multi-round rollouts need application-controlled KV across request boundaries.
Implement RFC #57103 as one complete, runnable contribution whose primary demonstration is orchestrator → merged hint envelope → native vLLM semantic interpretation → real Mooncake calls → observable completion, with a mechanism-only default, optional local policies, and no public Pin action.
Preserve the complementary backend-interpreted KVCR path.

## What Changes

- Reuse `KvHintsEnvelope` for request-attached hints and request-free control, with versioned action payloads, capability discovery, correlated operation results, and bounded duplicate handling.
- Separate transport/dispatch from payload interpretation: vLLM interprets native actions for the Mooncake path; KVCR-owned payloads remain opaque to vLLM's native interpreter and are forwarded to KVCR.
  GPU resource operations remain engine-controlled regardless of the interpretation owner.
- Run the primary full-path example with a lightweight orchestrator and real vLLM/Mooncake execution.
  Route each entire envelope to one interpretation path; defer Dynamo integration and mixed-owner envelopes to the follow-up scope in `integration-paths.md`.
- Implement application-controlled KV behavior using the already-merged KvHintsEnvelope/KvHintAction contract: explicit session/continuation management, session-addressable cache operations, and complete operation feedback.
  Keep explicit state commands separate from optional policy signals and publish incremental association events with state queries.
- Resolve logical hashes, sessions, continuations, and token ranges while separating logical associations, physical residency, retention declarations, and active request/transfer references.
- Implement Retain, Release, Evict, Demote, Prefetch, and Share through one engine resource-approval boundary.
  Report actual supported scope, partial outcomes, and unavailable capabilities explicitly.
- Introduce independently configurable policy, composition, and backend-adaptation interfaces.
  Plugins load at startup; changing them requires restart.
  Unconsumed lifecycle signals have no side effects, and local policies cannot rewrite explicit commands.
- Integrate policy observations, operation progress, and resource accounting with the scheduler, KVCacheManager, and BlockPool.
  Include optional quota/admission and request-score examples, applied by the scheduler while preserving its existing preemption algorithm.
- Adapt CPU offload and the vLLM Mooncake Store connector using existing backend APIs.
  The first delivery includes a real cross-engine Mooncake example; it does not modify the Mooncake project.
- Add multi-round rollout, shared-prefix, tool-wait, and memory-pressure experiments, including a router-directive baseline, policy ablations, failures, and overhead measurements.
  Performance improvements are hypotheses to measure.
- Measure rollout reuse within one weight version and independently verify invalidation after weight updates.
  Acceptance uses local Qwen3.8-27B TP=2 on the user-assigned GPUs: initially 0/3, and 0/2 for the October 20 continuation. Conclusions are scoped to each tested hardware allocation and layout; timings are not pooled across allocations.
- Provide runnable control recipes for direct commands without policies, shared-prefix association, branching and task cleanup, range retention, paused-session movement, cross-engine reuse, policy competition, and recovery.
  Three reference policy algorithms do not define the scope of this contribution.

## Capabilities

### New Capabilities

- `kv-control`: Common ingress, action validation, capabilities, operation lifecycle, correlation, queries, and event recovery.
- `kv-hint-dispatch`: Consumer ownership of hint interpretation, native-versus-KVCR routing, opaque forwarding compatibility, and prevention of duplicate semantic execution.
- `kv-logical-state`: Session/continuation associations, logical target resolution, independent retention declarations, and task-tree cleanup.
- `kv-policy-runtime`: Startup-configured policies and composer, shared observations, explicit-command preservation, and bounded scheduler cooperation.
- `kv-operation-execution`: Unified resource approval, local cache operations, request-free transfer progress, CPU/Mooncake adapters, and cross-engine availability.
- `kv-workload-validation`: Reproducible rollout and agentic workloads, correctness checks, comparative performance measurements, and disabled-path overhead.
- `kv-programmable-examples`: Executable hint/engine control workflows demonstrating the complete programmable-cache interface independently of particular optimization algorithms.

### Modified Capabilities

None.
No existing OpenSpec capability specifications are present.

## Impact

- Request-facing protocols and serving paths, engine clients and EngineCore, scheduler/configuration, KVCacheManager/BlockPool, and KV events/results.
- Existing CPU offload and Mooncake Store connector paths and their tests; reuse existing transport, allocation, and backend key/layout machinery.
- Existing KVCR hint forwarding/query/deliver/cleanup and regression tests.
  The primary new implementation is native semantics invoking Mooncake, complementary to the G1-only policy example discussed with Dynamo.
- Existing nearby unit/integration suites, benchmark scripts, runnable examples, and developer documentation.
  Model evaluation and GPU/remote integration evidence are required before claiming validated serving behavior.
- Builds on merged #53423 and #53624.
  Coordinate retention overlap with #38514 and execution overlap with #58198; neither standalone retention nor standalone CPU prefetch is the scope of this contribution.
  Align lifecycle boundaries with #51428, #48501, and Agentic API #364/#367.

## Non-goals

- Reusing the earlier TokenCake implementation, runtime replacement of plugins, a public Pin guarantee, or modifications to the Mooncake project.
- A custom hint-handler registration API or a replacement for the merged hint protocol.
  Programmability means application control of cache behavior through that protocol.
- Changing attention/model semantics through context editing or treating session identity as cache-content identity.
- Splitting this delivery into multiple PRs now.
  Decomposition can be considered after the complete implementation is reviewable.
- Treating the diagram's future KVCR G1 directives as existing functionality, or forcing KVCR-owned action meanings through vLLM's native payload interpreter.
