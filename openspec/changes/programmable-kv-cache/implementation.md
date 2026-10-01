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
- A live continuation is append-only. Rewriting or truncating its prefix requires
  a new continuation ID, which may remain in the same session. Association updates
  preserve every previously observed hash/group/span, including shorter aliases
  when a tail grows; they may add observations for previously unknown groups or
  ranges. Hash-addressed association is a declaration of content, not proof of
  token ancestry. Request observation must also check the actual token prefix.
  A new branch never retargets existing snapshot operations or following Retain
  declarations, and association updates never renew their TTL.
- Request-addressed association records a token-count/digest/context checkpoint
  in the shared index, including prefixes too short for a cache block. A later
  request must preserve that token prefix and cache salt. Finite text-token
  requests are supported; multimodal/embedding, LoRA, resumable-input and diffusion
  request targeting remain unsupported. Hash-addressed operations retain their
  independently advertised group/layout coverage.
- `associate.observe_growth` defaults to false. Setting it explicitly on a request
  target subscribes to that request's committed token/hash observations until it
  finishes. It does not observe another request or branch automatically. Only one
  request may grow a continuation at a time; competing writers are deferred.
  Cleanup removes the subscription; late observations cannot attach to a recreated
  continuation. Capacity failures preserve the last valid state and emit a bounded
  `observation_failed` event. Ordinary operation/declaration snapshots remain fixed.
- Request and checkpoint-backed session resolution reports unknown/unaligned tails
  as partial coverage. Association or retention may take effect with a partial
  result; that result does not assert residency. Metadata-only checkpoint changes
  advance logical revisions/events without rebuilding the eviction-priority view.
- Content identity includes the existing block hash and cache group, within the
  engine's compatibility namespace. Token ranges are half-open. Resolution reports
  each group's reusable coverage; it never rounds outward without reporting it.

## Results, bounds, and lifetime

Startup accepts `--kv-control-config` (or `EngineArgs.kv_control_config`) with a
`KVControlConfig`; `None` disables optional state and leaves existing hint routing
in place. A configured engine gets a fresh epoch and a supplied or generated ID;
named DP engines append `:dp<rank>`. Cache compatibility includes model/cache
configuration, the initialized group layout and effective block sizes, hash
algorithm/granularity, parallel layout, and weight version.

Python exposes `get_kv_control_capabilities()` and
`kv_control(method, invocation, payload)` on LLM/AsyncLLM. HTTP exposes
`GET /v1/kv/capabilities` and `POST /v1/kv/control`, whose body contains those same
three fields. Invocation names `engine_id`, `engine_epoch`, `consumer` (default
`native`), and optional `namespace`. `submit` accepts the unchanged envelope as
payload; `operation` takes message/action IDs; `associations` takes optional
session/cursor/revision/limit; `events` takes stream (`operation` or `association`),
after and limit. Discovery can query all connected engines, while a mutation
targets exactly one engine without using DP's broadcast utility method. Results
use the same JSON shape in process, through MP utility, and over HTTP.
Native request-attached envelopes are consumed before connector admission, then
removed from the legacy hook's input. Backend-selected envelopes retain their
opaque versions and payloads through that hook. Invalid wrappers or exhausted
control-record admission abort only the affected request without forwarding it;
valid envelopes keep per-action failures in the common queryable registry.
Completion and chat-completion requests accept `kv_hints` in the existing envelope
shape and pass the merged records to AsyncLLM, including streaming responses.
This attached HTTP profile requires one prompt, one completion and no beam search;
separate branches use distinct envelopes/IDs. The capability cache profile exposes
hash algorithm/seed/granularity and each group's effective size/reusability so
hash-addressed controllers can select compatible content without allocator IDs.

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
context, including an attached request binding and cache compatibility namespace.
An identical resubmission cannot refresh a Retain TTL or restart work. Reusing IDs
for another request binding or weight/layout namespace is a conflict. These fields
belong to the invocation/result context, not the merged envelope. Native results
identify their compatibility namespace explicitly.
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

The logical index defaults to 4,096 sessions, 16,384 continuations, 262,144 content
associations, 4,096 replayable association events and 16,384 pruning tombstones.
Exhaustion rejects the logical update atomically. History is pruned explicitly
after checking active interests; unknown/pruned state is distinct from a known
empty continuation. Query cursors carry the index revision, so pagination cannot
silently combine different snapshots. Revisions and continuation incarnations are
not reused within an engine epoch. Child changes also invalidate ancestor cleanup
revisions. Parent-prefix inheritance is captured at branch creation.

Declarations default to 4,096 entries and 262,144 retained snapshot entries.
Following declarations query the shared index rather than copy its growth into
another ownership map. Snapshot accounting includes repeated branch references,
partial coverage and empty-prefix records. Expiry bookkeeping is compacted after
updates, so repeatedly replacing one TTL cannot grow an unbounded timer heap.
When task cleanup closes a continuation still followed by another owner, preserve
that owner's final scope as a snapshot with its original priority/expiry. Refuse
cleanup atomically if preserving this independent lifetime would exhaust snapshot
capacity. An old following declaration never attaches to a recreated continuation.

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

Policies receive immutable observations and return proposals, including an optional
not-before time for predicted resume. The runtime bounds pending proposals and
work per step. Each derived command keeps an `origin` containing its signal's
message/action/operation IDs and the configured policy name. A signal's completion
means its observation was consumed; movement readiness comes from the derived
command's own result. The envelope fields remain unchanged.

Scheduler guidance uses the existing request `session_id` (also exposed by HTTP).
Its occupancy and token/group demand estimates are advisory; allocation still
uses KVCacheManager. Scores affect waiting-request selection, without changing
request priorities or the existing preemption algorithm. Quota deferral has a
maximum wait and cannot manufacture physical capacity from unused logical shares.

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

Native Mooncake operations use the connector's registered Store layouts and one
engine-owned lease per group/hash/backend. First support requires primary cache
entries at full group boundaries and PCP=DCP=1. Each rank reports its identity and
generation; duplicate/stale acknowledgements cannot satisfy the all-rank barrier.
No worker result registers GPU content until every rank has stopped accessing the
buffer. Existing request lookup waits for a compatible native load.

Control keys include the model/cache/weight compatibility namespace separately
from ordinary automatic-offload keys. Share establishes Store accessibility while
preserving the source; the orchestrator must submit Prefetch to the destination.
Missing objects are distinct from backend failures. Cold retention/deletion remain
unsupported. New movement is deferred while the engine is paused or sleeping;
logical control remains available. The in-process Python client has no busy loop
and advances idle connector work cooperatively when callers submit/query control.
Live weight-version transition integration and its RL validation remain task 7.2.

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

The installed native Mooncake CUDA 13 package is `0.3.13.post1`. Local development
uses an isolated master on `127.0.0.1:19151`, metrics/admin on `127.0.0.1:19153`,
and TCP transfer. The real-buffer integration suite also supports starting and
stopping its own isolated master, so its correctness checks are reproducible
without this particular development process.
