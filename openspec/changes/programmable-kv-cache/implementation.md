# Implementation contract

The October 1 implementation discussion confirms the following choices. This
document refines `design.md`; the existing envelope fields remain unchanged.

## Control and targeting

- Python and HTTP use the EngineCore utility channel. Independent calls name the
  engine and its epoch. A stale epoch fails before any action runs.
- The invocation selects `native` or `backend` for the entire envelope. Startup
  configuration supplies the request-attached default. There is no fallback,
  broadcast, or per-action owner inference.
- Ordinary operations capture resolved content at submission. Retain additionally
  accepts explicit `follow_growth` on a named continuation. Future branches are
  never implicitly included. Snapshot declarations keep their original content.
- Session/continuation mutations use expected revisions. A stale revision fails
  without effects. Transport retries with the same IDs return the original result;
  changing content under those IDs is a conflict.
- Content identity includes the existing block hash and cache group, within the
  engine's compatibility namespace. Token ranges are half-open. Resolution reports
  each group's reusable coverage; it never rounds outward without reporting it.

## Results, bounds, and lifetime

Native action version `1` uses `retain`, `release`, `evict`, `demote`, `prefetch`,
`share`, `associate`, `finish_task`, and `signal`. Their fixed payload schemas live
in `vllm/v1/kv_hints/native.py`; this is not a handler-registration API. The wrapper
protocol remains `0.1`. Backend-selected action types and versions remain opaque.

Targets use a `kind` of `hashes`, `session`, or `request`. A hash target includes
the compatibility `namespace` and `blocks`, each with a hexadecimal `block_hash`,
`group_id`, `start_token`, and `end_token`. Session targeting may select all tracked
continuations as a snapshot or name one continuation explicitly; follow-growth
retention requires the latter. Active-request targeting does not follow request-ID
reuse. Association payloads name the destination session/continuation and expected
revision; `task_parent_id` and `parent_prefix` describe distinct relationships.

Declaring these schemas does not advertise backend execution support. Runtime
capabilities must be filled from the initialized adapters and engine profile.

Operations distinguish `received`, `accepted`, `running`, `completed`, `partial`,
`deferred`, `unsupported`, `missing`, `failed`, `cancelled`, and `unconsumed`.
Only accepted/running work can own transfer resources. A terminal result does not
change after publication. Backend forwarding is a receipt, never GPU readiness.
Demote reports storage and GPU reclamation separately; Share preserves its source.

Control records are local to an engine epoch. Defaults are configurable: 4,096
retained actions, 128 actions per envelope, 1 MiB serialized envelope size, and
300 seconds of terminal result retention. Admission is atomic for record capacity:
reject an envelope rather than execute actions whose results cannot be retained.
Active operations cannot be evicted to admit control traffic. Terminal records
remain available for the advertised retention period; capacity pressure during
that period rejects new submissions. Expired IDs can execute again, so callers
must query within the retention window and use new IDs for intentional retries.

The deduplication unit is the entire ordered envelope plus interpretation/routing
context. An identical resubmission cannot refresh a Retain TTL or restart work.
Unknown versions and invalid payloads are reported before semantic execution.
Each derived step retains its parent message/action identity and its own outcome.
One failed action does not roll back earlier actions; multi-action submission is
not an application transaction.

Retention TTL starts when the declaration becomes effective, uses a monotonic
clock, and is independent of request activity. Priority is the maximum among
unexpired declarations on the same compatible content. Release removes only the
named declaration. Task completion removes that task tree's associations and
declarations, abandons its operation interests, and leaves shared or in-flight
physical references intact until the ordinary completion path releases them.

## Policy examples

The user selected retention priority, tool-wait demotion/predicted-resume prefetch,
and quota/admission plus request scoring. Explicit directives take precedence;
conflicting policy proposals receive deferred outcomes. Compatible physical work
is coalesced without merging caller lifetimes.

Admission uses configured session shares of the common capacity estimate; unused
logical shares are not extra blocks. Waiting requests bypass optional policy
deferral after a configurable maximum wait, but still require engine feasibility.
Request-score guidance uses importance plus elapsed waiting time with deterministic
arrival-order tie breaking. The scheduler applies guidance and keeps its existing
preemption algorithm. Parameters and adverse workloads must be recorded with each
evaluation; these examples do not promise universal performance gains.

## Adapter coverage

| Path | First-delivery behavior |
| --- | --- |
| GPU | Retain/Release, referenced-safe Evict, engine-approved allocation/holds |
| General CPU offload | Demote and request-free Prefetch through existing manager/worker transfers |
| Mooncake Store | Demote, Prefetch, source-preserving Share as store accessibility; separate destination prefetch |
| Cold retention/deletion | Unsupported unless the adapter proves the requested independent lifetime/scope |
| SimpleCPU | No new programmable operation capability |
| KVCR | Existing opaque request-scoped forwarding, query/deliver and cleanup; no claimed independent control |

Actual model/group support is discovered from the initialized connector layout;
the matrix does not equate a full-attention unit test with hybrid-model support.

## Execution environment

The user selected GPUs 0 and 3 for local TP=2 and authorized an isolated local
Mooncake Store. GPU 2 belongs to another running workload. The local model remains
Qwen/Qwen3.8-27B. Verify the downloaded revision/config before model execution.
Use the supplied model and dataset download functions, selecting only the intended
artifact instead of executing their unrelated top-level download lists.

The open-source workload is verl Agent Loop's multi-turn tool example. Preserve
equivalent model-visible work across baselines. External GPT-OSS-120B and
DeepSeek-V4-Flash machines will be provided later; those runs remain pending and
are not replaced by the local debugging results.
