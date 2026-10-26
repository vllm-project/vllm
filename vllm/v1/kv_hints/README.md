# Native KV hint primitives

This experimental implementation of [RFC #57103](https://github.com/vllm-project/vllm/issues/57103) adds RFC #59708 primitives to the existing `KvHintsEnvelope` (`protocol_version: "0.1"`).
Native action version **1.0** provides `vllm.set_priority`, `vllm.copy_to` and `vllm.move_to`.
It follows #59714's priority payload and successful-completion semantics, and extends execution to existing CPU/Mooncake adapters and correlated asynchronous results.
The COPY/MOVE payloads below define this branch's experimental interface; the upstream schema remains under discussion.

Enable `--kv-control-config '{"engine_id":"worker-a"}'` (Python accepts the same configuration).
Without it, existing hint forwarding is unchanged.
Local policies are optional and disabled by default.
No changes to Mooncake itself are required.

See [benchmark and validation commands](../../../benchmarks/programmable_kv_cache/README.md) for serving examples and regression checks.

## Terms

- A native action is interpreted by vLLM; a backend action is forwarded for the selected backend to interpret.
- G1 is the engine's local GPU cache; a cold tier stores copies in CPU memory or Mooncake Store.
- A transfer hold keeps a buffer allocated while a worker may access it.
- Publication registers successfully loaded data as reusable GPU cache.
- A claim is one caller's priority declaration for a physical copy; its revision orders updates and rejects stale retries.
- An operation incarnation is one submission instance identified by its operation ID.

## Ownership and entry points

`NativeKVController` checks control-record capacity and manages operation context and results.
`KvHintDispatcher` registers primitive payload schemas and selects when each action executes.
EngineCore consumes native request hints once and passes the same dispatcher to Scheduler for completion callbacks.
Scheduler uses that dispatcher without parsing the envelope again.
The configuration selects the envelope's interpreter: the `attached_consumer` startup default for request hints, or `consumer` in independent control invocation.
The existing connector hook forwards backend envelopes without interpreting their action payloads, even when an action name resembles a native action.
Mixed native/backend ownership is not supported.

Request-attached GPU-source COPY/MOVE and SET_PRIORITY validate at ingress and queue until successful stop or length-limit completion.
They execute before the scheduler releases request references.
Abort/error cancel them; preemption keeps them pending.
Cold-source COPY executes at ingress; a compatible resume waits for that load and falls back to ordinary computation if preparation fails.

COPY executes at request entry or successful completion according to its transfer direction.
Control-record limits bound pending work; resetting the cache generation invalidates it.
A transfer started at request completion retains its source buffers after the request ends.
The connector keeps the engine loop running until that work finishes.

Independent Python/HTTP submission is an extension using the same dispatcher, resource approval and adapters; it executes immediately.
`request_cached` requires an attached request.
Logical selectors can be used independently.

## Targets and locations

Targets are `"request_cached"`, or the existing tagged request/hash/session selectors:

- `request_cached` selects the actual cached copies still in the bound request's block tables.
  Detached Mamba checkpoints are not in that scope.
- `{"kind":"request","request_id":"$request","end_token":784}` selects logical token content in the bound request.
  At completion it also resolves cached checkpoints that have left the active tables.
  Explicit ranges are half-open.
- A `hashes` target carries compatibility namespace and hash/group/token spans; a `session` target selects tracked continuation/range state.
  Those targets are snapshotted at submission even when execution waits for request completion.

Applications must discover group sizes and supported boundaries; `784` is the recorded local Qwen profile, not a universal block size.
Physical block IDs are not public selectors.
Unknown/unrepresented/unaligned ranges are reported.

COPY/MOVE locations contain `tier` (`gpu`, `cpu`, `mooncake`) and optional `engine_id`.
Omission means the addressed engine/configured backend.
A different engine ID is rejected; destination GPU resources must be approved by that engine.
Capabilities list the supported `directions`, targets and adapter limitations.

## R1 MOVE / R2 COPY example

Attach to R1, to save and reclaim its prefix after successful generation:

```json
{
  "protocol_version": "0.1",
  "message_id": "round-1-save",
  "actions": [
    {
      "action_id": "move-prefix",
      "action_type": "vllm.move_to",
      "action_version": "1.0",
      "payload": {
        "target": {
          "kind": "request",
          "request_id": "$request",
          "end_token": 784
        },
        "src": {
          "tier": "gpu"
        },
        "dst": {
          "tier": "mooncake"
        }
      }
    }
  ]
}
```

R2 uses `vllm.copy_to`, the same logical range, `src={"tier":"mooncake"}` and `dst={"tier":"gpu"}`.
It waits for the compatible load and uses GPU KV after the data is registered as reusable.
This example starts COPY when R2 arrives, so it verifies execution and reuse without measuring savings from prefetch before request arrival.

GPU-to-cold COPY preserves the source.
MOVE saves first and reclaims only eligible copies.
Cold-to-GPU MOVE is unsupported because selective cold deletion is absent.
Cross-engine reuse is two COPYs (A→Store, then Store→B); Store availability never means B's GPU is ready.
General CPU uses the existing CPU offload path; SimpleCPU and KVCR do not acquire new native movement capabilities through this branch.

## SET_PRIORITY

The #59714-compatible payload is:

```json
{
  "target": "request_cached",
  "location": "local_g1",
  "claim_id": "router/task-1",
  "revision": 1,
  "value": 10,
  "ttl_seconds": 60
}
```

A signed integer value sets the eviction priority of a physical cache copy.
A finite positive TTL starts when the priority takes effect.
Copies with higher effective priority are evicted later; normal free-list/LRU ordering breaks ties.
`value: null`, no TTL and a higher revision clear only this claim.
Equal/older revisions do not renew TTL; conflicting payloads at an equal revision reject the operation before changing any selected claim.
An explicit claim ends when a higher revision clears it with null, its TTL expires, or its physical copy is invalidated.
An arbitrary claim ID does not associate the claim with a logical task, so task completion cannot infer that ownership.
Task cleanup also removes the associated local-policy claims.
The controller retains the latest revision for each claim ID, including cleared claims, until the copy is invalidated, reused or reset.
This retained revision prevents stale updates from recreating a cleared claim; each copy supports at most 16 claim IDs.
Reconstructed copies do not inherit these claims.

Logical selectors are an extension selecting currently resident copies.
Legacy Retain/Release declarations retain their independent logical-content lifetimes and can follow newly appended content when configured.
Both declaration families feed **one** `RetentionFreeBlockQueue`, using the maximum valid priority; the default 0 applies only when no declaration exists.
Priority never pins memory, allocates blocks, or changes active references.

## Outcomes, retries and limits

Queries/events use the common OperationRegistry, including message/action IDs, action type/version, engine epoch, namespace and optional originating policy.

| Status | Meaning |
| --- | --- |
| received | Admitted and correlated |
| queued | Waiting for request completion, session admission, or a compatible COPY prerequisite; nonterminal, no reservation |
| accepted / running | Approved work; not yet a guarantee of complete reusable scope |
| completed / partial | Terminal outcome with actual stages and target scope |
| deferred | Terminal resource refusal; retry intentionally with a new message ID |
| missing / unsupported / failed / cancelled | Distinct terminal negative outcomes |

`target_outcomes` identifies hash/group/token range, status, completed stages, reason and transfer ID/generation where physical work was dispatched.
Already resident content is labeled and has no transfer ID.
Missing ranges have no hash.
MOVE can report a stored copy and still-referenced source as partial; COPY only reports GPU availability after all participating ranks stop accessing buffers and successful data is registered.
Cancellation drains worker holds before reuse.
Historical completion does not guarantee continued residency.

`max_result_targets` (default 16384) rejects excessive movement scope before any allocation.
Existing envelope/action/result/index limits still apply.
Identical message/context replay returns existing results and never requeues or renews TTL.

Call `acknowledge` with a message ID after collecting terminal results for every action in that message.
This releases detailed records while keeping a bounded replay fingerprint until the original expiry.
Identical replay then returns `acknowledged`; it does not execute again.
`cancel` requires message ID, action ID and operation ID, so a delayed caller cannot cancel a newer operation instance.
Cancelling a signal does not cancel its separately identified derived operations or roll back completed effects.

`residency` resolves a bounded logical target into current GPU copies, references, effective priorities and native load progress.
The query reports current state without reserving the copies.
Cold-tier residency queries and physical residency events are not supported.
`capacity` reports record/replay occupancy and actual unique physical transfer payload bytes.
Per-operation bytes can attribute a shared transfer to more than one caller.

Weight updates require idle requests and drained transfers.
Use the existing start/update/finish lifecycle; the controller invalidates local KV, encoder and multimodal caches before transfer and rejects new inference/control submissions until a new version commits.
Commit creates a fresh epoch and namespace and isolates ordinary connector hashes without deleting another engine's Store objects.
The default version preserves legacy hashes.
A previous version may be reused only with exactly its original weights.
Tensor changes that bypass the start/update/finish lifecycle do not receive these protections.
After a failed update, the engine rejects new work until the update completes or the engine restarts.

Legacy action version `1` remains supported (`retain`, `release`, `evict`, `demote`, `prefetch`, `share`, state commands and signals).
Explicit eviction has no implicit mapping to clearing priority.
Optional retention/tool-wait policies now produce version-1.0 primitives through the same execution boundary.
A SET_PRIORITY proposal waits for overlapping GPU COPY proposals from the same signal before application; its TTL starts after the copy is ready, and failed copies yield explicit missing or partial priority outcomes.
Waiting proposals do not consume the execution budget, so a one-action-per-step limit cannot starve their prerequisites.
There is no automatic tool-call detector.
Request-free control and local policy production are explicit extensions to the initial request-attached RFC profile.

Policies may declare `consumed_signals` as an immutable set of signal names.
Signals with no configured consumer return `unconsumed` without building a request observation or invalidating scheduler guidance.
The default `None` preserves delivery of every signal to existing third-party policies.
Subclasses must explicitly redeclare the filter to opt in, so an overridden handler does not silently inherit a narrower signal set.

The built-in tool-wait policy accepts `resume_after_seconds` on the initial
`tool_wait` signal. It skips a speculative round trip when that estimate does
not exceed `min_pause_seconds` (default 0.5) or its observed store-plus-load time.
`consider_pause_duration=false` selects the pressure-only ablation. Missing
estimates keep pressure-based demotion. Skipped movement history is bounded and
cleaned with its logical owners; explicit movement commands are unaffected.
This heuristic does not estimate GPU recomputation cost or guarantee residency.
In admission-coupled mode a short tool can also use a sustained GPU backlog:
current referenced occupancy must reach `min_pressure`, and an observed waiting
request must have waited longer than the minimum window or measured store/load
round trip, whichever is larger. With low current occupancy a short pause keeps
G1 content even when projected demand is high. Observed queue age is a heuristic
for this decision, not a prediction of the target request's remaining wait.

`prefetch_on_admission=true` enables scheduler coordination for the movement
policy. Under queued demand, a matching session's GPU COPY waits until that
session is selected for admission. The scheduler starts preparation before cache
lookup and waits for terminal publication; failure permits ordinary recomputation.
Without queued demand the predicted preparation may execute early. Cancellation
and task cleanup release queued scope without acquiring transfer buffers.
Custom policies emitting `PolicyProposal.wait_for_admission` must enable
`uses_scheduler_guidance` at startup. Only GPU COPY proposals support this flag.

A concrete policy may implement `needs_request_observation(payload)` and return
false when it does not read request or session demand. The observation still
contains time and physical counters, but request/session tuples are empty and
arrival history is preserved. Alternatively, it can implement
`needs_request_observation_for_view(payload, counters)` to refine that decision
using a counters-only observation before request queries. Subclasses default to
full observations unless they explicitly implement one of these methods.
A concrete scheduler policy may also set
`signal_updates_guidance=false` if its signals cannot change request guidance.
The built-in quota importance signal continues to invalidate cached guidance.

Source references: [RFC #59708](https://github.com/vllm-project/vllm/issues/59708), [PR #59714 at 7966f717dd4e](https://github.com/vllm-project/vllm/tree/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints).
Copy claims use the existing eviction queue; the branch does not add the draft PR's separate victim-selection heap.
