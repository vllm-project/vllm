# Programmable KV cache design

## Context

The confirmed scope comes from RFC #57103, the author's September 24 comment, and the October 1 discussion.
`implementation.md` records the subsequent user answers and concrete implementation contract; [testing.md](../../../docs/programmable-kvc/testing.md) defines test contracts and phase gates.
Unresolved deployment/model capabilities remain subject to actual validation.

The working tree already contains `KvHintsEnvelope` in the Python engine/request path and Rust protocol, as well as the merged KVCR secondary-tier adapter.
The OpenAI entrypoints do not currently expose the corresponding hint plumbing.
CPU offloading and Mooncake Store have request-oriented allocation, metadata, transfer, and completion paths.
`Scheduler.has_requests()` already accounts for connector pending push work; a new independent operation must participate in engine liveness rather than require a synthetic generation request.

The older TokenCake implementation will not be used.
Scheduler and KV manager changes are authorized.
One complete PR is the delivery unit.
The existing uncommitted discussion-document edit belongs to the user.

The local discussion sources were `docs/programmable-kvc/chats.md` and `notes.md`, retained in commit `ac5094f177` before removal of the local notes.
They support a broader delivery than three optimization algorithms: engine-local session indexing available to scheduler/offload consumers, request-independent session-addressable APIs, extensible hint transport, and association-event feedback.
Participant preferences in the chat are evidence for design proposals, not proof that every event or index detail is an upstream standard.
`coverage.md` maps those requirements to implementation and examples.

The latest user-provided discussion with Karen and the supplied dual-path diagram make the primary acceptance flow explicit: orchestrator → KvHintsEnvelope → vLLM-native interpretation → Mooncake.
A complementary path forwards opaque payloads to KVCR for its own interpretation.
`integration-paths.md` describes the ownership boundary, proposed concrete classes, current code anchors, and required end-to-end evidence.
This does not reinstate custom hint-handler registration.

## Goals / Non-Goals

**Goals:**

- Implement the common control, logical-state, policy, resource-approval, and result contracts across request lifetimes.
- Keep the default engine driven by explicit directives; enable semantic local decisions through opt-in policies.
- Make policy, composition, and backend adaptation independently configurable at startup and extensible without replacing the scheduler implementation.
- Exercise real CPU movement and cross-engine reuse through Mooncake Store.
- Measure correctness, serving performance, overhead, and negative cases using reproducible agentic/rollout workloads.

**Non-Goals:**

- Runtime plugin replacement or state migration between plugin implementations.
- Public Pin guarantees, changing content hashes to session IDs, attention/context editing, or modifying the Mooncake project.
- Advertising unsupported remote guarantees or claiming simulation results as real transfer/model performance.

## Decisions

### Confirmed control and policy boundary

Both request-attached and independent native control reuse the existing envelope.
Signals and explicit operations remain distinguishable.
No registered policy means no native signal-triggered operation; the result reports that signal unconsumed.
Backend-owned hints follow the selected backend consumer's contract.
Local policies cannot rewrite explicit operations.
Engine validation can still clip, defer, or reject work within the action's allowed semantics.

This follows the author's updated RFC position.
The older public-Pin design and a second `agent_hint` envelope are not implementation targets.

### Confirmed plugin lifecycle

Plugins are configured and instantiated at engine startup.
Changing the plugin set requires restart.
Policies propose, a composer combines proposals, and an engine-owned approval path controls allocation and execution.
Backend adapters use connector-owned keys, layouts, buffers, and completion mechanisms.

Candidate class/module names and configuration fields are not public commitments yet.
Existing `scheduler_cls` replacement alone would not provide the desired policy composition and common state/resource ownership.

### Confirmed application-controlled KV behavior

Programmability means that applications/orchestrators control KV cache behavior through the already-merged KvHintsEnvelope and KvHintAction.
Preserve their protocol_version/message_id/actions and action_id/action_type/action_version/ payload fields.
This change supplies engine consumption, state management, resource approval, execution, and observable results above that existing contract.

Implement the RFC's supported cache actions and explicit state-management semantics in the engine.
No custom hint registration API, user-defined handler plugin system, or replacement envelope is part of this delivery.
The existing envelope is generic; action-specific payload semantics still require concrete implementation and must not be described as already supplied by transport alone.

Request-attached and independent ingress use the same validation and execution path for native actions.
Transport validates the envelope; the selected consumer owns action-version/payload meaning and validation.
KVCR-owned payloads do not pass through native semantic validation.
The application can retain a prefix, request demotion during a pause, prefetch before reuse, observe readiness, and explicitly release state.
It does not need to install an engine policy to issue those commands.

Policies, composition, and backend adapters retain the separately confirmed startup configuration interfaces.
They are implementation extension points, not the definition of programmable cache control.
Internal transfer-buffer holds remain engine correctness mechanisms and do not reintroduce public Pin.

Track an application's command across its execution stages.
For example, cold save and GPU reclamation are distinct stages of Demote; receipt of the command does not imply either stage succeeded.

### Confirmed native and backend interpretation paths

vLLM defines and processes the native cache-operation semantics.
Mooncake Store checks its own storage state/capacity and executes supported storage/transfer calls.
Retain/Demote meanings are not new policy definitions to implement in the Mooncake project.
A remote capability weaker than the requested operation must be reported rather than silently substituted.

The existing KVCRSecondaryTierManager forwards request-scoped envelope data to `submit_hint`, then uses its query/deliver and framework-reference lifecycle.
It must continue to permit KVCR to interpret backend-owned payloads independently.
Existing source pin callbacks protect transfer buffers; they are not this RFC's removed public Pin operation.

Both interpretation paths must respect engine GPU allocation/reference/lifetime authority when requesting GPU work.
KVCR need not pass through the native hint interpreter to request such integration work.
The figure's dashed future KVCR G1 directives are not implemented by current hint forwarding alone.

The first release selects one interpretation path for each entire envelope; mixed native/backend ownership inside one envelope is not supported.
Preserve the envelope fields, action order, and payload values.
Route selection is carried by the invocation/deployment context rather than a new envelope field; its exact API spelling remains to be specified.
Do not broadcast the envelope or silently fallback to another interpreter.
A selected backend's missing capability is not permission to reinterpret its payload as a native action.

### Logical state and physical authority

The RFC requires separate records for:

1. Session/continuation relationships and logical content associations.
2. Actual copies at an engine/tier and their compatibility context.
3. Independent Retain declarations and their lifetimes.
4. Operations, dependencies, destination reservations, and transfer references.

Cache hits must establish associations as well as newly computed cache blocks.
Evicting a GPU copy must not imply task completion.
Engine-local associations must not be represented by reusable physical block IDs.
Existing KV managers and BlockPool remain responsible for physical mutations and correctness refs.

An optional engine-local index supports session addressing and policy inputs; hash-addressed operation remains possible without a complete global task index.
The concrete index layout, bounds, and pruning behavior still need design review.

### Confirmed distinction between explicit state commands and policy signals

Session/continuation association, parent-prefix boundaries, and explicit task-tree termination are explicit engine management commands.
They update logical records and the addressed declarations according to a defined contract.
They do not infer application intent or reserve GPU space merely because a relationship exists.

Tool waits, predicted resumes, importance, and similar contextual signals are policy inputs.
With no consuming policy they have no side effects, including no implicit task cleanup.
A completed inference request is never automatically a completed task tree.
This permits session-addressable explicit control even in the no-local-policy configuration.

Whether a directive binds to a snapshot of a session's current content or follows future growth must be explicit in the target/declaration contract.
The first version must not silently interpret one as the other.
Operation and logical-state revisions must make retries and stale branch updates observable; field names and ordering rules remain to be agreed.

### Confirmed association event contract

The discussion distinguishes first-writer BlockStored from a new association on an already-resident prefix.
Use separate association deltas plus a bounded state query, correlated with engine epoch and event position.
A new session cache hit emits a new association without claiming new physical storage.
Residency removal and association removal remain different changes.

Events do not carry a full session set on every association update.
Existing full-report mode re-emits prefixes hit by the current request; it is not a complete engine snapshot.
Recovery must not pretend it repairs all unobserved sessions.

### Execution and result contract

Approval revalidates current state rather than trusting a policy snapshot.
Identical physical transfer work can be coalesced while preserving each caller's correlation and declarations.
Removing a caller or proposal cannot release memory that an in-flight transfer still accesses.

Demote has separate cold-copy and GPU-reclamation milestones.
Prefetch completion requires registration of reusable destination content.
Share does not implicitly delete its source.
A resume racing a compatible prefetch must coordinate with the existing load instead of starting duplicate work.

The operation record is the source for both query and event results.
Exact status enums, retry semantics, retention limits, and transport shape remain to be finalized.

### Confirmed backend scope

The vLLM-side Mooncake connector/adapter can change; the backend project cannot.
Capabilities must describe the operations and guarantees available from existing backend APIs.
Unsupported operations return an explicit result.

The acceptance path is engine A producing/storing an explicit compatible prefix, engine B preparing that prefix before a resume request, and the resume observing actual reuse.
Store availability and B's GPU availability are separate facts.
Cross-engine control remains orchestrator-owned; neither a plugin nor Store replication implicitly proves another engine's GPU residency.

The confirmed CPU path is general OffloadingConnector/CPUOffloadingManager.
Mooncake Store is the remote adapter.
SimpleCPUOffloadConnector does not implement the new control actions in the first delivery and must report that accurately.
Exact remote action coverage still requires a backend capability matrix.

### Confirmed ingress profile

The first delivery exposes Python and HTTP control, using the common EngineCore utility channel internally.
It does not add a separate gRPC control service.
Submission, capability discovery, and operation queries share the same records.
Engine targeting must remain explicit for independent control in multi-engine deployments.
Exact method names and endpoint paths are implementation details to document with the concrete schemas.

### Confirmed scheduler cooperation

The first implementation includes optional quota/admission and request-score examples.
The scheduler consumes the guidance and retains ownership of queue operations.
Its existing preemption algorithm is preserved.
Cache policy plugins cannot directly manipulate waiting/running queues or select preemption victims.
The concrete quota formula, score combination, and starvation protection still need to be specified before implementing the examples.

### Confirmed validation environment and weight boundary

Performance scenarios reuse KV only within one weight version; weight updates receive independent old-KV invalidation correctness checks.
A full RL training loop is not required for the first performance workload.

The local four-A100 machine is for quick debugging, not comprehensive evaluation.
The user corrected the local debug model to Qwen3.8-27B with TP=2.
Use that model instead of Flash-Next.
Exact downloaded checkpoint/configuration, usable GPU IDs, and Mooncake deployment still need verification before execution.
The repository already lists `Qwen/Qwen3.8-27B` in model tests through the Qwen3.5 architecture.

Model downloads use `/home/youwei/bzh/model/download.py`; dataset downloads use `/home/youwei/bzh/dataset/download_dataset.py`.
These scripts currently execute fixed lists when run, so resource preparation must select only intended artifacts.
Any Python execution follows the repository's uv/venv instructions.
Related repositories can be cloned alongside this repository under `../` when necessary for integration/adaptation.
This does not remove the agreed boundary prohibiting changes to the Mooncake project.

Later full evaluation on other machines targets GPT-OSS-120B and DeepSeek-V4-Flash.
Group-aware resolution and multi-rank completion must therefore be part of the main implementation, not deferred behind a full-attention-only demonstration.
Exact supported layouts remain in the capability matrix.

### Proposed validation approach

The main integration test starts at the orchestrator sender and reaches real Mooncake calls and observed reuse.
Calling the adapter directly is useful unit coverage but does not meet the full-path acceptance requirement.
Demonstrate both an explicit command and an opt-in native signal-driven policy using the same execution boundary.
Preserve the G1-only baseline for comparison.

The user confirmed a runnable lightweight orchestrator for the first complete delivery.
Dynamo integration is subsequent work, described in `integration-paths.md`.
The lightweight sender must exercise real Python/HTTP ingress and Mooncake calls; it is not a mock for engine/backend work and must not be presented as an already-working Dynamo integration.

Use deterministic workload descriptions with real model execution for performance: shared prompt families, multiple branches, tool waits, variable output lengths, memory-pressure sweeps, incorrect resume predictions, and cross-engine resumes.
Use an oracle schedule only as a separately labeled comparison.

Compare native cache behavior, explicit router directives with feasibility and retry, individual local policies, and composed policies under equivalent work and resource limits.
Report whole-workload completion, TTFT/tails, decode latency, admission delay, actual hits/recomputation, transfer volume, CPU/index overhead, and failures, including regressions.

Potential external workload source: verl's Agent Loop and multi-turn tool example support vLLM.
Selection and how much framework integration belongs in this PR remain discussion items.
Existing vLLM RL examples also provide weight-update correctness patterns, but some depend on batch invariance requiring newer GPUs than the currently visible A100s.

Proposed extreme workloads, pending final configuration:

| Workload | Controlled stress | Evidence to collect |
| --- | --- | --- |
| Interleaved rollout families | Four shared-prefix families, multiple branches, alternating active groups, total working set larger than GPU cache | Whole-batch completion, recomputed tokens, retention/movement benefit |
| Partial rollout pause/resume | Staggered pauses and unequal remaining lengths within one weight version | Resume TTFT, copy/compute overlap, queue delay |
| Fork/join with external sharing | Complete one task tree while another task still uses its prefix | Correct association cleanup and independent retention |
| Incorrect resume estimates | Unused prefetched prefixes competing with live decoding | Wasted bytes, displacement, throughput/tail regressions |
| Cross-engine resume | A stores a prefix; B prepares it before the next request | All-rank completion and measured destination reuse |

These workloads test opportunities under pressure, not guaranteed cache hits.
Retain is evictable and prefetch can be deferred; neither is public Pin.

### Rollout completion and performance acceptance (2026-10-11)

The user requested implementation of the remaining rollout/runtime gaps and real performance validation.
The primary target is at least 25% higher completed-rollout throughput than native vLLM with equal GPU KV capacity and equivalent workload.
Report elapsed batch time, useful output throughput, TTFT/tails, admission delay, reward/answer correctness, preemptions, cache reuse, transfer bytes and failures.
Keep regressions and unsuccessful configurations in the report.
A speed target does not authorize changing prompts, output budgets or task quality between arms.

Use the actual verl ToolAgentLoop against real vLLM inference, real executable tools, generated continuations and multiple samples per prompt.
An HTTP client adapter may add native KV controls without replacing verl's loop/token accounting.
Separate controlled trace replay (identical model-visible inputs/output budgets) from live agent trajectories (which can diverge); report both.
Cover multiple prefix families, staggered tool waits, cold/warm pressure sweeps and misleading resume predictions.
Repeat measured arms, separate warmup, and retain raw rows, configuration, source versions and server metrics.
Local GPUs 0/3 and Qwen TP=2 remain the development environment; external model results require actual runs.

Runtime work includes an idle-engine weight-version transition with stale-transfer and local/external KV isolation, bounded current-residency queries, explicit result acknowledgement that preserves bounded deduplication, and workload-aware movement and scheduler guidance.
Admission remains engine-owned; hints never guarantee residency.
Updating model tensors must use the weight-update lifecycle, with cache invalidation before new inference is permitted.
A metadata-only version change is not evidence that real weight-update correctness has been tested.

Optimize observation/selection costs only with behavioral regressions and measured evidence.
Application recovery handles deferral, missing data, partial scope, restarts and failed transfers with bounded retries and ordinary-compute fallback.
No public Pin or changes to Mooncake are introduced.

### Control overhead and deployment-scale validation (2026-10-18)

The user resumed implementation and experiments, requesting minimum control overhead and a local commit after full relevant validation.
Primary performance runs SHALL use at least `gpu_memory_utilization=0.5`, without a small fixed KV-byte override.
Increase distinct prefix families, branches and client concurrency to create pressure; retain a cache-fit control at the same deployment memory setting.
Record actual cache capacity, source hashes, commands, repeated equal-work comparisons, request phases, control RPC counts/times and GPU samples.
Instrumented profiles SHALL be identified separately from normal performance runs.
Move optional control waiting out of the next inference request's critical path while preserving engine load publication barriers and draining all submitted control before task cleanup.
Reuse terminal submission results while still collecting asynchronous derived outcomes.
Test lifecycle, transfer races, cancellation, reset/restart, model correctness and disabled/idle paths; smoke runs do not satisfy acceptance.

## Risks / Trade-offs

- Partial blocks and hybrid groups have different reusable boundaries → resolve coverage through cache-group semantics and report unsupported/partial scope; do not equate bytes copied with a valid prefix hit.
- Optional policies add work to scheduling → bound work and measure the disabled path, policy CPU time, and index memory.
- Speculative movement displaces useful idle cache → account for displaced cache, future demand, and in-flight capacity; include adverse workloads.
- Backend completion races cancellation/reset → keep physical holds until workers can no longer access buffers and reject stale completion publication.
- Weight changes invalidate computed KV → define compatibility and invalidation across local/remote copies before running training-loop workloads.
- Open retention/prefetch PRs overlap execution pieces → document the distinct composition/control contribution and decide reuse boundaries before publication.

## Migration Plan

Keep new behavior opt-in and the existing no-plugin configuration usable.
Publish one complete implementation with configuration, capability coverage, runnable examples, and measured results.
Rollback disables the feature and restarts the engine; no cross-restart policy-state migration is promised.

## Open Questions

Confirmed in the latest discussion: reuse the merged hint protocol without a custom handler system; explicit state commands are separate from policy signals; association reporting uses incremental events and state queries.

The user explicitly rejected treating the three proposed reference policies as the complete delivery.
Full hint/engine programming coverage is required; `coverage.md` and `kv-programmable-examples` specify additional end-to-end flows.

Remaining concrete design discussion:

- Exact invocation/configuration setting selecting the single interpretation path for an entire envelope without changing the merged envelope.
- Concrete control method/endpoint schemas and routing to an explicit engine.
- The exact per-backend action/target capability matrix.
- Supported attention/cache-group profiles, range rounding/partial results, task cleanup ordering, resource limits, retries, and retention expiry behavior.
- Reference policy inputs/objectives and composer conflict rules.
- Snapshot versus follow-growth targeting, derived-plan completion, and ordering of retried lifecycle updates.
- Open-source rollout example and reproducible benchmark configurations.
- Usable GPU IDs and local Mooncake deployment before running GPU experiments.

## Follow-up work

### Application timing and admission budgets, 2026-10-19

The fixed-work trace has 664 pauses, all below 54 ms (median 10.6 ms).
The composed diagnostic nevertheless writes 370.5 GB and loads 262.2 GB;
its client submits no resume predictions because it waits for stores first.
The separately tuned quota run improves throughput by 45.6% in one run.
These observations motivate two independent changes and separate ablations:

- Include the remaining application pause estimate in the initial tool-wait
  signal. The movement policy skips a round trip when the pause does not exceed
  a configurable minimum or its observed store-plus-load duration. A skipped
  store also suppresses the associated speculative load. Missing estimates keep
  the previous pressure-only behavior; explicit COPY/MOVE remain unchanged.
  This is a transfer-amortization heuristic, not a measured recompute cost model.
- Recheck live free capacity and running growth on each quota admission.
  Several requests admitted in one scheduling iteration must not each spend the
  same observation's free capacity. Policy scores remain interval-cached, the
  scheduler owns admission, and its finite fairness bypass remains in force.
  Candidate demand uses the scheduler's actual cache-hit blocks immediately
  before allocation. This reuses its lookup and counts a newly shared prefix
  once even when several admissions occur within one observation interval.
- Optional admission-coupled COPY proposals wait while foreground requests are
  queued. When the scheduler selects a matching session/content prefix, its
  runtime starts the preparation before cache lookup and waits for terminal
  publication. Other sessions can continue. A failed load falls back to normal
  lookup/recomputation; explicit commands keep their independent execution.
  Forecast-only requests can still prepare early when there is no queued demand.
  Benchmark result collection budgets include the configured admission wait and
  forecast horizon in this mode, since intentional queue time is observable.
- Estimate live tool batches from prior completed batches of the same tool
  names. Replay without tool names uses a labeled aggregate estimator; fixed and
  oracle variants remain selectable, and every prediction/actual pair is saved.
- Built-in signals which use no request demand opt out of request scans. Such
  observations still include physical counters and current time, and do not
  prune arrival/wait history. Only signals that can change scheduler scores
  invalidate cached guidance. Custom policy subclasses receive full observations
  unless they explicitly override the opt-out method.
- A multi-block GPU preparation retains both completed destinations and already
  resident members until the whole operation finishes or its caller cancels.
  These are engine-owned operation references, not a public residency promise.
  Shared callers release only their own interest; reset releases completed holds
  while dispatched worker buffers remain protected until completion.
- Rollout finalization prevents new preparation submissions, waits for submitted
  receipts, ends the logical task to cancel unused preparation, and then collects
  and acknowledges every result. A timeout remains a failed control outcome even
  when cancellation cleanup successfully frees its result records.

Tests extend the existing prefix, scheduler and HTTP suites. Their contracts are
respectively signal-to-transfer behavior, admission-to-KV capacity behavior, and
pause-to-control ordering. Guard short-pause round trips, repeated spending of
one capacity observation, stale predictions and missing result collection with
small in-process behavioral tests before real full-workload timing.

Use the same 256 trajectories at utilization 0.5 for native, quota and composed
comparisons. Preserve all timing, control and per-family results; repeat runs,
and label long-pause perturbations and oracle estimates separately.

The V2 pressure run exposed eviction of 778 completed destination members before
their multi-block operation finished. Keep both resident and newly loaded members
referenced until the whole operation finishes, with independent interests for
shared callers. Cancellation/reset release completed interests while dispatched
worker buffers remain protected through completion. Admission-coupled preparation
checks the complete target's load slots and physical commitment before starting;
temporary shortages queue, impossible preparations and expired waiting fall back
to recomputation. Late automatic resume work is skipped for already running
sessions; explicit importance still applies. Task completion cancels unused
preparation before draining results. Failed control work stays failed even when
its result records are acknowledged, and measured cleanup includes worker drain.

The four-family fit workload exposes a separate decision failure in forced
pressure mode: 258.6 GB stored with only 2.67 GB loaded, 526 preparations already
past admission, and 503.27 s compared with about 170 s native. The same workload
needs at most about 65% KV without the policy. Projected demand alone does not
justify a short-pause round trip. In normal admission-coupled mode, a short tool
may also amortize movement when current referenced occupancy reaches the
configured pressure threshold and an already waiting request has waited longer
than max(minimum pause window, measured store plus load cost). This is a bounded
backlog heuristic, not an oracle for a particular request's remaining wait.
Long/unknown pauses retain their existing policy contract; the forced-pressure
option remains an explicit ablation. The benchmark exposes the movement pressure
threshold and uses 0.8 for the normal admission profile. Tests must distinguish
actual occupancy from projected demand and use completed transfers to establish
the cost boundary before repeating pressure, fit and live tool workloads.
Signal observations read physical counters first. An optional concrete-policy
predicate can then request per-request demand; low-occupancy short pauses retain
the counters-only path and cached scheduler age history. Unknown subclasses keep
the existing full-observation default.

`integration-paths.md` records the follow-up scope; incomplete first-release acceptance stays in `tasks.md`.
Dynamo integration and mixed-envelope dispatch are future work.
The supplied diagram's future KVCR G1 path requires its own agreed integration and is not represented as current support.
This change's native Mooncake full-path example, shared GPU mechanisms, and existing KVCR compatibility checks remain first-release work.

## References

- <https://github.com/vllm-project/vllm/issues/57103#issuecomment-5814964446>
- <https://github.com/vllm-project/vllm/issues/51428>
- <https://github.com/vllm-project/vllm/issues/48501>
- <https://github.com/vllm-project/vllm/pull/53423>
- <https://github.com/vllm-project/vllm/pull/53624>
- <https://github.com/vllm-project/vllm/pull/38514>
- <https://github.com/vllm-project/agentic-api/issues/364>
- <https://github.com/vllm-project/agentic-api/issues/367>
- <https://verl.readthedocs.io/en/latest/advance/agent_loop.html>
- <https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst>
- Historical source at `ac5094f177`: `docs/programmable-kvc/chats.md`, especially sections on SessionPrefixIndex, KVEvents, request-independent retention, and envelopes.
- Historical source at `ac5094f177`: `docs/programmable-kvc/notes.md`.

## RFC #59708 alignment, authorized 2026-10-06

The `rfc-hint` branch implements the alignment analysis without removing existing first-release requirements.
Public actions are `vllm.set_priority`, `vllm.copy_to` and `vllm.move_to`, version `1.0`; old action names/version `1` remain an explicit compatibility profile.
One dispatcher validates both profiles.
Whole-envelope native/backend ownership remains explicit; mixed ownership is not silently added.

SET_PRIORITY accepts the #59714 request_cached/local_g1 payload (claim ID, revision, signed value, required positive finite TTL, null clears) and logical selectors as an extension.
Attached priority and GPU-source movement queue until successful completion, before freeing request references.
Abort/error cancel the queued action.
Cold-source COPY executes at ingress, and the request joins the existing load barrier.
Independent submission is an extension using the same executors.
A queued lifecycle action is nonterminal and owns no transfer resources; resource DEFERRED remains terminal.
Explicit hash/session selectors capture their logical scope at submission; request_cached and bound request/range selectors are resolved at successful completion.
A finite end_token fixes the requested range; hash/session selectors retain their submission-time snapshot.

COPY/MOVE payloads contain target, src and dst.
Locations carry tier (gpu, cpu, mooncake) and optional engine_id (omitted means the explicitly addressed engine).
Only local G1 <-> configured cold tier is supported; GPU-source COPY preserves source, GPU-source MOVE reclaims eligible copies after storage succeeds.
Cold-source MOVE is unsupported because selective cold deletion is not implemented.
Cross-engine sharing uses two independent COPYs through the shared Store, each engine approving its own buffers.
Remote engine mutation is never inferred from a location string.

Copy priority records are keyed internally by physical copy lifetime, with bounded per-copy revision tombstones until reuse/reset.
They feed the existing retention queue together with legacy logical declarations; no second eviction heap is added.
Public target selectors/results contain content identities, not allocator IDs.
Per-target outcomes are bounded and identify stored/registered/reclaimed, blocked, missing and failed scope.
Existing all-rank publication and cancellation holds remain authoritative.
Optional policies lower to these same primitive contracts.
