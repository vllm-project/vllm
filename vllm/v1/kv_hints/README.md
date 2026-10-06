# Native KV hint primitives

The `rfc-hint` reference branch implements RFC #59708 primitives on the existing
`KvHintsEnvelope` (`protocol_version: "0.1"`). Native action version **1.0** provides
`vllm.set_priority`, `vllm.copy_to` and `vllm.move_to`. It follows #59714's priority
payload and successful-completion semantics, and extends execution to existing
CPU/Mooncake adapters and correlated asynchronous results. COPY/MOVE payloads here
are a concrete experimental profile, not a claim that an upstream schema is final.

Enable `--kv-control-config '{"engine_id":"worker-a"}'` (Python accepts the same
configuration). Without it, existing hint forwarding is unchanged. Local policies
are optional and disabled by default. No changes to Mooncake itself are required.

## Ownership and entry points

`NativeKVController` owns common admission, context and records;
`KvHintDispatcher` owns primitive schema registration and lifecycle dispatch.
EngineCore consumes native request hints once and passes the same dispatcher to
Scheduler for completion callbacks. There is no second scheduler ingress parser.
An envelope's consumer is explicit: the `attached_consumer` startup default for
request hints, or `consumer` in independent control invocation. Backend envelopes
remain opaque through the existing connector hook, even if they contain a
native-looking action name. Mixed native/backend ownership is not supported.

Request-attached GPU-source COPY/MOVE and SET_PRIORITY validate at ingress and
queue until successful stop or length-limit completion. They execute before the
scheduler releases request references. Abort/error cancel them; preemption keeps
them pending. Cold-source COPY executes at ingress; a compatible resume waits for
that load and falls back to ordinary computation if preparation fails.

The registered phase depends on direction, rather than assuming every COPY has
the same timing. Pending work is bounded by control-record admission and invalidated
by cache generation reset. Request completion can start an asynchronous transfer:
source holds then outlive the request and existing connector liveness drives it.

Independent Python/HTTP submission is an extension using the same dispatcher,
resource approval and adapters; it executes immediately. `request_cached` requires
an attached request. Logical selectors can be used independently.

## Targets and locations

Targets are `"request_cached"`, or the existing tagged request/hash/session selectors:

- `request_cached` selects the actual cached copies still in the bound request's
  block tables. Detached Mamba checkpoints are not in that scope.
- `{"kind":"request","request_id":"$request","end_token":784}` selects logical
  token content in the bound request. At completion it also resolves cached
  checkpoints that have left the active tables. Explicit ranges are half-open.
- A `hashes` target carries compatibility namespace and hash/group/token spans;
  a `session` target selects tracked continuation/range state. Those targets are
  snapshotted at submission even when execution waits for request completion.

Applications must discover group sizes and supported boundaries; `784` is the
recorded local Qwen profile, not a universal block size. Physical block IDs are
not public selectors. Unknown/unrepresented/unaligned ranges are reported.

COPY/MOVE locations contain `tier` (`gpu`, `cpu`, `mooncake`) and optional
`engine_id`. Omission means the addressed engine/configured backend. A different
engine ID is rejected; destination GPU resources must be approved by that engine.
Capabilities list the supported `directions`, targets and adapter limitations.

## R1 MOVE / R2 COPY example

Attach to R1, to save and reclaim its prefix after successful generation:

```json
{
  "protocol_version": "0.1",
  "message_id": "round-1-save",
  "actions": [{
    "action_id": "move-prefix",
    "action_type": "vllm.move_to",
    "action_version": "1.0",
    "payload": {
      "target": {"kind":"request","request_id":"$request","end_token":784},
      "src": {"tier":"gpu"},
      "dst": {"tier":"mooncake"}
    }
  }]
}
```

R2 uses `vllm.copy_to`, the same logical range, `src={"tier":"mooncake"}` and
`dst={"tier":"gpu"}`. It joins the load and uses GPU KV only after publication.
This attached profile has no pre-arrival lead time; it establishes execution and
reuse, not proactive-prefetch latency savings.

GPU-to-cold COPY preserves the source. MOVE saves first and reclaims only eligible
copies. Cold-to-GPU MOVE is unsupported because selective cold deletion is absent.
Cross-engine reuse is two COPYs (A→Store, then Store→B); Store availability never
means B's GPU is ready. General CPU uses the existing CPU offload path; SimpleCPU
and KVCR do not acquire new native movement capabilities through this branch.

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

A signed integer value sets a copy-lifetime eviction bias with a finite positive
TTL starting at application. Higher effective values survive longer, with normal
free-list/LRU ordering breaking ties. `value: null`, no TTL and a higher revision
clear only this claim. Equal/older revisions do not renew TTL; conflicting payloads
at an equal revision reject the operation before changing any selected claim.
Explicit copy claims are cleared by null/higher revision, expiry or copy lifetime;
finishing a logical task does not infer ownership from an opaque claim ID. Task
cleanup also removes the associated local-policy claims. Revision tombstones
remain until physical invalidation/reuse/reset, bounded at
16 claim IDs per copy. Reconstructed copies do not inherit these claims.

Logical selectors are an extension selecting currently resident copies. Legacy
Retain/Release keep their independent content-level lifetime and follow-growth
semantics. Both declaration families feed **one** `RetentionFreeBlockQueue`, using
the maximum valid priority; the default 0 applies only when no declaration exists.
Priority never pins memory, allocates blocks, or changes active references.

## Outcomes, retries and limits

Queries/events use the common OperationRegistry, including message/action IDs,
action type/version, engine epoch, namespace and optional originating policy.

| Status | Meaning |
| --- | --- |
| received | Admitted and correlated |
| queued | Waiting for request completion or a compatible COPY prerequisite; nonterminal, no reservation |
| accepted / running | Approved work; not yet a guarantee of complete reusable scope |
| completed / partial | Terminal outcome with actual stages and target scope |
| deferred | Terminal resource refusal; retry intentionally with a new message ID |
| missing / unsupported / failed / cancelled | Distinct terminal negative outcomes |

`target_outcomes` identifies hash/group/token range, status, completed stages,
reason and transfer ID/generation where physical work was dispatched. Already
resident content is labeled and has no transfer ID. Missing ranges have no hash.
MOVE can report a stored copy and still-referenced source as partial; COPY only
reports GPU availability after all participating ranks stop accessing buffers and
successful data is registered. Cancellation drains worker holds before reuse.
Historical completion does not guarantee continued residency.

`max_result_targets` (default 16384) rejects excessive movement scope before any
allocation. Existing envelope/action/result/index limits still apply. Identical
message/context replay returns existing results and never requeues or renews TTL.

Legacy action version `1` remains supported (`retain`, `release`, `evict`, `demote`,
`prefetch`, `share`, state commands and signals). Explicit eviction has no implicit
mapping to clearing priority. Optional retention/tool-wait policies now produce
version-1.0 primitives through the same execution boundary. A SET_PRIORITY proposal
waits for overlapping GPU COPY proposals from the same signal before application;
its TTL starts after the copy is ready, and failed copies yield explicit missing or
partial priority outcomes. Waiting proposals do not consume the execution budget,
so a one-action-per-step limit cannot starve their prerequisites. There is no automatic
tool-call detector. Request-free control and local policy production are explicit
extensions to the initial request-attached RFC profile.

Source references: [RFC #59708](https://github.com/vllm-project/vllm/issues/59708),
[PR #59714 at 7966f717dd4e](https://github.com/vllm-project/vllm/tree/7966f717dd4eb09461dd97102db1352c924bbf50/vllm/v1/kv_hints).
The independent copy-claim adaptation uses the existing eviction authority rather
than importing a second victim-selection heap from that draft PR.
