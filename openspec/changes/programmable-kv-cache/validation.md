# Implementation validation

This is an execution record, not a claim that the full change is complete.
Local dates use Asia/Shanghai. All Python commands use the repository `.venv`.

## Append-only continuation contract

The user selected option A: a live continuation grows by adding content; rewriting
or truncating it requires a new ID within the same or a different session. The
index rejects removal or substitution of known hash/group/span associations before
changing revisions, membership or events. Short aliases remain recorded when a
tail grows. Sparse observations can add previously unknown groups and ranges.
Hash-only associations remain application declarations; token ancestry must also
be validated at request observation when the engine path is connected.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_state.py tests/v1/test_kv_hint_control.py -q
```

Result: **127 passed**, 36.65 seconds. Complete suites cover rejection atomicity,
alias growth, stale revisions, cleanup and capacity reclamation, retention scope
and unchanged expiry across branches, and correlated replay of rejected rewrites.
This change has no serving/model-path effect yet.

## Native startup and Python/HTTP serving control

Added opt-in `KVControlConfig`, an initialized model/group/hash compatibility
profile, safe utility serialization, explicit engine/epoch targeting, Python and
HTTP submission/query/events, and request-attached completion/chat/streaming
forwarding into the merged envelope. Native envelopes are removed before the
legacy connector hook; backend envelopes preserve their opaque records. Invalid
attached wrappers abort their request. Request targeting and automatic growth are
not advertised yet. CPU/Mooncake adapters and policies remain pending.

Complete regression command:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/engine/test_kv_control.py \
  tests/v1/engine/test_kv_cache_group_metadata.py \
  tests/v1/engine/test_engine_args.py tests/v1/test_request.py \
  tests/v1/test_serial_utils.py tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_deferred_block_free.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py \
  tests/entrypoints/serve/kv_control/test_api.py -q
```

Result: **641 passed**, 322.58 seconds. An earlier run caught tuple/list differences
between in-process and serialized replies; the facade now returns the same JSON
shape. After adding attached dispatch, its complete control/engine/KVCR suites
passed **134 tests** (38.90 seconds), and the final engine suite passed **12 tests**.
After adding serving fields, the complete HTTP, Responses protocol/sampling and
admission suites passed **72 tests** (21.33 seconds). DP routing tests verify a
mutation reaches only its named engine, reject ambiguous names, and reject unknown
engines before sending a mutation. Normal hooks and manual mypy 3.12 passed.
The final complete control/state/engine/HTTP/Responses/admission suites pass
**234 tests**, 88.83 seconds, after the serving and capability changes.

The enabled-controller 128-question Qwen TP=2 command adds `--enable-kv-control` to
the recorded `evaluate.py` command. The request-free utility lifecycle passes;
recompute and prefix-cache modes both score **125/128 (97.65625%)**, with zero
truncations and zero final numeric-answer mismatches. Cache hits are **0 / 99,568**;
token sequences differ on 33 questions. Timings are 356.190 / 325.700 seconds;
these concurrent-development runs are not a controlled overhead/speedup experiment.
The accuracy run preceded final HTTP/profile-reporting changes; those were tested
separately through the live server. No transfer or policy benefit is claimed.

Started the actual local Qwen HTTP server and ran `http_control.py` using the exact
commands in the benchmark README. Result: **passed**, with **0 → 784 → 0** cached
tokens across cold, warm, and physically evicted requests. Evict reports **4 copies
reclaimed**, no blocked/missing targets; all three greedy output token sequences
match. This uses actual initialized Mamba/full-attention groups, native utility
control and the real BlockPool rather than simulated residency.

The live workflow also passes completion, chat and SSE attached hints; rejects
multi-prompt/n>1/beam attached scope; checks correlated queries/events, two explicit
associations of shared content, append-only rewrite rejection, retry, task cleanup,
unknown action/version and engine/epoch/namespace errors. Movement explicitly
returns unsupported at this development stage. Raw enabled accuracy and HTTP
responses are committed as compressed JSON under the benchmark `results/` folder.
The server is stopped after validation; GPU 2's unrelated workload is untouched.

## Request association and explicit generation observation

Request targets now resolve through the initialized engine's active requests,
including an attached `$request`. Associations record token-prefix evidence even
before a reusable block exists; rewriting that prefix or changing its cache salt
requires a new continuation. `observe_growth=true` explicitly follows one request
until completion. Snapshot association does not grow implicitly. Competing writers
are deferred; bounds failures preserve the last state and emit an observable
failure event. Metadata changes do not invalidate an unchanged retention view.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/engine/test_kv_control.py tests/v1/test_kv_hint_control.py \
  tests/v1/test_kv_hint_state.py tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_deferred_block_free.py tests/v1/core/test_scheduler.py \
  tests/v1/core/test_async_scheduler.py -q
```

Result: **568 passed**, 302.80 seconds. Tests use actual Request/KVCacheManager
objects for cached-prefix membership, allocation stability, snapshot/growth,
token-prefix rewrites, original retention expiry, concurrent writers, capacity
failure reporting, final observations, inherited-prefix mismatch, and cleanup
before ID recreation. Range outcomes report missing/unaligned portions explicitly.
The short-alias branch boundary is also covered by the final state suite.
The final complete control/state/engine/HTTP suites pass **178 tests** in
48.72 seconds. Normal pre-commit hooks, manual mypy 3.12 and OpenSpec validation
pass for this phase.

Reran the real Qwen TP=2 server with the README configuration and added
`--validate-request-associations` to `http_control.py`. Result: **passed**.
The snapshot and cache-hit continuations remain at **1,302 tokens**; the live
continuation grows to **1,318**, accepts **10 tool tokens**, and reaches **1,344**
after another 16 generated tokens. All four tracked inference requests hit
**784 cached tokens**. A request rewriting the live prefix fails while the
continuation remains unchanged. Explicit task cleanup succeeds afterward.

The same run repeats the actual GPU Evict/reuse, chat/SSE, routing/version and
event/query checks from the previous phase. Cold/warm/evicted cache hits remain
**0 → 784 → 0**, with four reclaimed copies and identical greedy output token IDs.
Raw observations/responses are in
`results/qwen3.8-27b-tp2-request-associations.json.gz`. This validates request/state
and local GPU behavior, not cold-tier movement or full RL/policy performance.

## October 1: control record foundation

Implemented immutable correlated operation snapshots, explicit engine/epoch/path
validation, whole-envelope deduplication, ID/content conflicts, bounded admission,
terminal TTL, progress/terminal transitions and bounded event replay. Completions
carry an operation incarnation so a late completion cannot update reused IDs.
Added fixed native payload, target and capability records, including strict
action versions, unknown-field rejection, group-specific content validation,
and explicit follow-growth scope. Native actions are not yet connected to serving
or physical execution.

The complete relevant baseline command was:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py \
  tests/v1/test_request.py \
  tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py \
  tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py \
  tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py \
  tests/v1/kv_connector/unit/test_mooncake_store_layout.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py -q
```

Result: **412 passed, 1 skipped**, 146.83 seconds. The Mooncake suites use mocks;
this does not prove real Store transfer or cross-engine reuse. The KVCR module was
initially skipped because its optional package was absent; it was installed and
tested separately below.

After strengthening stale-completion handling, reran the full control suite:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py -q
```

Result: **36 passed**, 10.64 seconds.

After adding native payload/target validation, the same complete control suite
passed **73 tests**, 20.10 seconds.

Existing real CPU/GPU worker transfer correctness:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/kv_offload/cpu/test_gpu_worker.py -q
```

Result: **40 passed, 1 skipped**, 13.81 seconds. The skipped case is ROCm-specific.
This verifies existing transfer behavior, not new request-free execution.

Cloned the official `ai-dynamo/kvcr` repository into `../kvcr`, pinned by the
checkout at `317e62f301300c101bf9c911e020fb89b35f1583`, and installed it editable
using `.venv/bin/uv pip install --python .venv/bin/python -e ../kvcr`.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py -q -rs
```

Result: **20 passed**, 6.09 seconds; no skip. This covers the existing adapter
contract and framework-reference callbacks with its existing test doubles.

Ruff 0.14.0 formatting, normal pre-commit hooks, and mypy 3.10/3.12 checks passed for
the changed files. OpenSpec strict validation also passed. GitHub HTTPS timed out, so Git fetching uses the
existing authenticated SSH transport through per-command configuration. A slow
ShellCheck binary download required a retry; its installation subsequently
finished. The final hook run uses the normal configuration with no hook disabled.
Buildkite runs the new control suite in the existing V1 job.

## Environment and outstanding integration

Editable precompiled vLLM and all `requirements/test/cuda.in` dependencies are
installed. Python 3.12.7, uv 0.12.21, Torch 2.13.0+cu132, Transformers 5.17.0;
local A100 GPUs 0 and 3, driver 595.71.05.

Qwen/Qwen3.8-27B is pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The supplied download function is used
without executing its unrelated model list. Direct Hugging Face access timed out;
the public mirror works for metadata. Xet downloads failed; ordinary HTTP download
is being retried. Model validation requires every indexed weight shard, not just
the presence of `config.json`.

Pending: real Mooncake deployment/transfer, native control execution, model output
evaluation, full workload comparisons and the external large-model matrix.

## October 1: logical state and independent retention

Added a bounded session/continuation index with captured parent prefixes,
group-aware range resolution, reverse membership, revision-checked mutation,
task-tree cleanup, explicit history pruning, paginated queries and association
events. Known empty content remains distinguishable from unavailable history.
These records contain no reusable physical block IDs.

Added independent priority/TTL declarations, snapshot and explicit follow-growth
scope, release and coordinated logical task cleanup. Closing a followed prefix
preserves another owner's final scope and original expiry. Following declarations
cannot attach to a reused continuation identity. These modules still await engine
request/cache hooks and the common command executor.

The new state suite tests independent shared-prefix lifetimes, branch stability,
group boundaries, stale writes/cleanup, atomic capacity refusal, revision-safe
pagination, event gaps, explicit pruning and seeded declaration interleavings.
Existing full prefix-cache, KV-manager, deferred-free and scheduler suites run
alongside the control/state suites:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_deferred_block_free.py \
  tests/v1/core/test_scheduler.py -q
```

Initial result: **503 passed, 1 failed**, 354.74 seconds. The existing CPU-marked
`test_diffusion_read_deferral_keeps_a_longer_pp_wait` constructs PP=3 without
workers, but the real platform count sees only the two allocated GPUs. Corrected
that test's hardware isolation by supplying a mocked device count of three;
kept all PP scheduling assertions intact and did not expose additional GPUs.
The complete suite is rerun after this fixture correction.

Final result: **504 passed**, 324.09 seconds. Normal pre-commit hooks and mypy
3.10/3.12 checks passed for the changed state/retention and test files.

## October 1: native command interpretation and real Store baseline

`NativeKVController` consumes the merged actions through a common attached/free
submission method. Logical commands execute without local policies; signals are
unconsumed by default. Backend-owned attached envelopes remain opaque for the
existing request hook. Independent backend control reports unsupported instead
of reinterpreting the payload. Engine resource approval remains a separate
executor interface and is not yet connected to scheduler/worker execution.

Tests cover native/opaque ownership, action-specific errors, non-transactional
multi-action outcomes, TTL-safe retries, request binding, compatibility namespaces,
asynchronous approval without false completion, and partial cleanup effects.

Installed `mooncake-transfer-engine-cuda13==0.3.13.post1`. Added a real native Store
integration suite rather than extending the dict-backed mock with claims of real
transfer. It starts an isolated loopback master unless an address is supplied,
creates two Store clients, and transfers actual GPU buffers in both directions.
Its 36 cases cover three dtypes, three sizes, one/three buffers per key, and a
missing member in every batch; successful data must match exactly and the missing
destination must remain unchanged. The suite is wired into the existing two-GPU
Mooncake TCP Buildkite job.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/kv_connector/mooncake_integration/test_store_buffers.py -q -rs
```

Result: **36 passed**, 14.27 seconds. This establishes native buffer-transfer
correctness, not hint-to-worker integration, model reuse, or a serving speedup.

Combined phase gate after request-binding/namespace correlation changes:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/test_request.py tests/v1/kv_offload/tiering/test_kvcr_tier.py \
  tests/v1/kv_connector/mooncake_integration/test_store_buffers.py -q -rs
```

Result: **176 passed**, 52.36 seconds, no skipped cases. Normal pre-commit hooks
and mypy 3.10/3.12 checks passed. The real Store suite used its own master lifecycle.

## October 1: engine-owned local eviction

Added optional retention ordering to BlockPool's existing free queue. Priority
ties retain the original LRU/LIFO ordering, and retained blocks remain evictable.
Declarations do not acquire correctness references. Heap bookkeeping is bounded;
partial-prefix aliases, cache reset and transfer reuse notifications are covered.
The feature-disabled pool keeps its original queue implementation.

Added scoped eviction through the existing pool and KV manager. Active references,
unpin/reuse readers and unaddressed aliases prevent reclamation. A duplicate copy
does not produce a logical removal event while another copy is still present.
Native results expose affected physical block counts and blocked/missing targets.
This invalidates cache entries inside the existing arena; it does not claim to
free the arena's CUDA allocation. Startup/serving wiring is still pending.

The complete cache/manager/control gate was:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_deferred_block_free.py \
  tests/v1/test_kv_hint_state.py tests/v1/test_kv_hint_control.py -q
```

Result: **485 passed**, 157.73 seconds. An earlier run found the same host-GPU-count
assumption in an existing PP=4 grouping unit test; its topology check is now mocked
locally, without changing the grouping assertions or using additional GPUs.
Range reporting was subsequently refined to distinguish a missing tail from a
known but unaligned range, and to recognize an existing partial-prefix alias.
After that refinement, the complete control/state suites passed **120 tests** in
32.25 seconds. Normal pre-commit and mypy 3.10/3.12 checks passed.

The Qwen3.8-27B checkpoint is complete: 18 shards, 55,563,006,776 weight-file bytes.
A TP=2 baseline on GPUs 0/3 with 4 GiB KV per rank and eager execution initializes
three Mamba groups plus one full-attention group, all with an effective block size
of 784 tokens. Three shared-prefix questions produce identical output token IDs
before/after cache reuse and after cache reset; the warm pass hits 784 tokens per
question. This is a debugging baseline, not the full model evaluation.

GSM8K was downloaded through the supplied dataset function at revision
`740312add88f781978c0658806c59bc2815b9866`. The dedicated evaluation uses 128 seeded
questions, full few-shot rationales matching the repository's GSM8K completion
format, and separate forced-recomputation/prefix-cache passes. An initial
direct-answer chat prompt was rejected: its 621–729-token prompts did not exercise
784-token reuse boundaries and scored only 55.47%. The evaluation now asserts a
sufficient shared prefix before generation; the accuracy threshold was not lowered.

The first full-rationale run at 256 output tokens was also rejected: 48/128
recomputation samples and 49/128 cached samples hit the length limit. The limit
was increased to 1,024 without changing the 80% minimum accuracy or 1% allowed
accuracy-drop thresholds. Both modes then scored **125/128 (97.66%)**, with zero
length-limited outputs and zero differences in final numeric answers. There are
27 token-sequence differences. The cache-enabled pass recorded **99,568 cached
tokens**, versus zero when cache reads were disabled.

The reproducible script, command and compressed per-question results are in
`benchmarks/programmable_kv_cache/`. This is a feature-disabled model/cache
baseline; it does not complete the hint-to-worker, policy or cross-engine serving
evaluation tasks. The external GPT-OSS/DeepSeek evaluations remain unexecuted.

## Current implementation dependency

The user selected append-only continuation identity (option A). Request observation,
explicit DP engine selection, and Python/HTTP serving are implemented and validated
above. The next dependency is engine-owned transfer reservations and buffer holds,
followed by request-free CPU/Mooncake execution, policy composition, and workload
integration. Existing native Store Put/Get tests do not establish this control
path's readiness or subsequent model reuse.

The selected verl Agent Loop source is cloned at `../verl`, revision
`fbb4b3a8bf636f290c9c59fc346f756849e9c241`; it has not been modified or validated.

## Request-independent transfer resource foundation

Added a lazily created KVCacheManager transfer manager. It acquires existing
BlockPool source references or unpublished destination allocations, shares equal
content/direction/backend work without merging caller lifetimes, and observes the
ordinary allocation watermark. Different cold destinations retain independent
holds; competing loads from different sources cannot allocate the same target
concurrently. Cancelled dispatched work retains its buffer until worker completion.
Reset invalidates publication, retains in-flight buffers and returns false until
they drain; callers must retry reset. Lease identities prevent late duplicate
completion from affecting a newer allocation.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_deferred_block_free.py \
  tests/v1/test_kv_hint_state.py tests/v1/test_kv_hint_control.py -q
```

Result: **518 passed**, 203.94 seconds. Added group-completeness and alias tests
subsequently; the final complete `test_prefix_caching.py` run passes **192 tests**
in 59.06 seconds. An initial alias fixture incorrectly used different hash/cache
block sizes with a unitary coordinator; it was corrected to a supported hybrid
configuration. Existing request references and concurrent computed copies survive
transfer completion/failure; incomplete Mamba/attention groups cannot produce a
full prefix hit. Normal pre-commit and manual mypy 3.12 passed.

These are resource/metadata correctness tests, with explicit worker-completion
injection. Native control dispatch, backend copies and all-rank completion are
not connected by this substage; tasks 4.1 and 5.1 remain unchecked. Ordinary serving
does not instantiate the transfer manager, so no model-output change is introduced
by this foundation alone.

## Native movement lifecycle

Native physical execution now accepts initialized transfer adapters and correlates
Demote/Prefetch/Share with the manager's shared leases. Explicit task cleanup drops
only that task's interests; a reset cancels outcomes while worker-held buffers
continue draining. Backend submission failure releases undispatched reservations.
Demote reports cold storage separately from referenced-safe GPU reclamation;
Prefetch rechecks registered residency before reporting its terminal scope.
The engine's post-step path consumes progress even on a step without model tokens.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py tests/v1/test_kv_hint_control.py \
  tests/v1/engine/test_kv_control.py -q
```

Result: **328 passed**, 98.90 seconds. Manual mypy 3.12 passed. The new tests inject
an explicitly queued adapter into native control; they validate independent task
cancellation of shared loads, partial capacity, backend failure cleanup, source
references during Demote, and reset before worker acknowledgement. This phase
introduces no advertised production transfer adapter yet. It does not establish
actual byte transfer, connector liveness, all-rank readiness or model equivalence;
those are the next integration gate, and tasks 4.1/5.1 remain open.

## Native Mooncake execution and real cross-engine reuse

Added request-free connector metadata, per-rank/generation acknowledgements,
backend submission through registered Store buffer layouts, and progress through
existing empty model-runner steps. Native load reservations remain unpublished
until all ranks finish; a compatible resume waits instead of starting another
load. Store writes precede eligible GPU reclamation. Share keeps the source and
reports Store accessibility, with destination Prefetch handled by the controller.

The complete related gate was:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_deferred_block_free.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/engine/test_kv_control.py tests/v1/kv_offload/tiering/test_kvcr_tier.py \
  tests/v1/kv_connector/unit/test_mooncake_store_*.py \
  tests/entrypoints/serve/kv_control/test_api.py -q
```

Result: **991 passed**, 457.16 seconds. The Store regression before the added
control tests passed **348 tests**; the complete updated scheduler/worker suites
passed **237 tests**. In-process cooperative polling subsequently passed the
complete engine/control/HTTP suites: **142 passed**, 41.13 seconds.
After adding the paused/sleeping admission guard, complete prefix/engine/control
suites pass **331 tests**, 97.80 seconds. Logical Retain remains usable while
movement is deferred; a retry with the original IDs retains its original outcome.

The real Store suite now passes **42 tests**, including native control driving
actual GPU buffers through two rank-local layouts, missing-rank completion and
reset while worker buffers remain held. Combined with the full prefix-cache suite,
the final missing-result refinement passes **242 tests**, 82.86 seconds. The first
integration fixture omitted hash initialization; adding the same initialization
used by the existing cache suites resolved it. No simulated copy counts as GPU
transfer evidence. Get must return the expected byte count; truncated or failed
objects are never registered.

The Qwen TP=2 HTTP workflow, with real Mooncake and ordinary request lookup disabled,
passes Share, Demote, request-free Prefetch, missing content and request association
checks. Demote stores and reclaims **4 group copies**; Prefetch registers **4 copies**
and the next request hits **784 tokens** with identical greedy output token IDs.
The separate two-engine run passes: A preserves its source, B completes Prefetch
before its first inference, and B hits **784 tokens** with A's output token IDs.
Both engines use GPUs 0/3; B uses GPU utilization 0.45 and the same fixed 4 GiB KV
cache per rank. Model servers and the isolated Store are stopped after validation.
Commands and configurations are in the benchmark README.

The movement accuracy run uses the same 128 seeded GSM8K questions, eight-shot
rationales and a 1,024-token output limit in both passes. Per-question salts force
recomputation; the comparison pass explicitly demotes/prefetches the common prefix
before each eight-question batch. Both score **124/128 (96.875%)**, with **zero**
truncations and **zero** final numeric-answer mismatches. Cached tokens are
**0 / 100,352**. Durations are 593.50 / 589.27 seconds, including driver warmup and
control work; no speedup or disabled-path overhead conclusion is drawn. All rows,
control outcomes and HTTP responses are saved in the three `mooncake-*.json.gz`
artifacts under `benchmarks/programmable_kv_cache/results/`.

This verifies the fixed-weight local Qwen profile and the direct-controller path.
Live weight updates, policy-driven movement, pressure/fairness measurements and
the external GPT-OSS/DeepSeek matrix remain incomplete.

## Native CPU movement through the existing offload path

The CPU adapter shares existing CPUOffloadingManager capacity and reservations,
with namespaced cold keys and ordinary CPUOffloadingWorker copies. Each rank
acknowledges native work separately from request jobs; no fake inference request
or inference completion receipt is produced. Ready cold copies remain referenced
until all ranks finish, and reset is refused while native work owns cold buffers.
Share is unsupported on CPU. Initial support requires one GPU block per CPU chunk,
full primary boundaries and PCP=DCP=1; tiering/KVCR is not advertised as this adapter.

Native destination allocations now join the connector's existing reuse fence.
This is necessary because ordinary CPU saves can keep reading an idle GPU block
until a later step flushes them. The old store is submitted/waited before a native
load can overwrite a reused block. The existing boundary-store test now covers
request allocation, CoW and native-control reuse.

```bash
CUDA_VISIBLE_DEVICES=0,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/cpu/test_gpu_worker.py \
  tests/v1/kv_connector/unit/offloading_connector/ -q -rs
```

Result: **565 passed, 1 skipped**, 162.90 seconds. The skip is ROCm-specific.
After the reuse-fence refinement, the complete connector scheduler suite passed
**312 tests**, 89.59 seconds. New manager/worker checks cover cold capacity holds,
duplicate rank reports, failure cleanup, namespace separation and absent inference
receipts. Normal hooks and manual mypy 3.12 passed before the final fence refinement;
the final checks are recorded with the phase commit.

A broader initial run also included `unit/test_offloading_connector.py`, which
actually loads multiple models: **558 passed, 1 skipped, 11 failed**. Six Llama
cases hit gated-repository 403 responses, Gemma was unavailable from the configured
mirror/local cache, and four public-model cases failed their startup memory check
while the live validation engines occupied GPUs. This run is not counted as a
passing model matrix; the local Qwen checks below have their own actual evidence.
After releasing both Qwen instances, an offline `-k 'mamba or Falcon'` rerun passed
the Mamba-130m model test. The other three cases now reach weight loading and fail
because Falcon-H1-0.5B and Mamba-1.4B weights are absent from the local snapshots
(**1 passed, 3 failed, 7 deselected**, 139.53 seconds). Those model cases remain
unvalidated; no code failure is inferred from missing artifacts. The final normal
pre-commit hooks and manual mypy 3.12 pass for the implemented CPU phase.

The full HTTP Qwen CPU workflow passes, including request-free Demote/Prefetch,
missing content, request association and source-reference checks. Measured requests
set the existing `max_load_tokens=0` to exclude automatic CPU loading. Demote
reclaims **4 group copies**; explicit Prefetch restores them before inference,
which hits **784 tokens** and preserves the short greedy output token IDs.

The 128-question accuracy run scores **124/128 (96.875%)** for recomputation and
**125/128 (97.65625%)** after CPU movement before each eight-question batch. Both
have zero truncated answers; **one** final numeric answer differs. Cache hits are
**0 / 100,352**. Timings are 709.25 / 724.20 seconds, including driver warmup and
control work, so no serving-speed or accuracy improvement is claimed. The complete
rows and operation results are in the two `cpu-*.json.gz` artifacts. The model
server is stopped afterward; GPU 1's unrelated workload is untouched.

## Policy and scheduler extension checks (2026-10-02)

The optional policy runtime now supplies immutable cache pressure/request
observations, proposal-only retention/tool-wait/quota policies, deterministic
composer output and bounded derived operation records. Scheduler guidance is
applied only when configured; the scheduler still owns queue operations and its
existing preemption algorithm. Request-free transfer liveness includes pending
executor jobs, so an idle engine continues progress without a synthetic request.

```bash
.venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/engine/test_kv_control.py tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py -q
```

Result: **574 passed**, 14 warnings, 250.00 seconds. The policy-focused subset
passed 9 tests and the scheduler extension subset passed 2 tests. Additional
HTTP association recovery and KVCR opaque-forwarding tests passed in the
following run:

```bash
.venv/bin/python -m pytest \
  tests/entrypoints/serve/kv_control/test_api.py \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py -q
```

These were separate commands: HTTP **13 passed** (4.45 seconds) and KVCR
**21 passed** (6.89 seconds), each with 14 warnings. The original combined-command
wording did not match the execution record. File-existence checks and deterministic
plan generation from the placeholder scripts did not establish workload coverage;
see the October 6 correction below.

The real CPU/Mooncake transfer, local Qwen cross-engine, and GSM8K correctness
results above remain the acceptance evidence for those paths. Policy pressure,
fairness, external model, live weight-update and full recovery-server runs are
still unexecuted and remain open tasks.

## Duplicate-work check (2026-10-02)

The required pre-publication commands were repeated:

```bash
gh issue view 57103 --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "57103 in:body"
gh pr list --repo vllm-project/vllm --state open --search "programmable kv cache"
```

Each command stopped with GitHub CLI exit code 4 because this environment is
not authenticated. No PR was opened. The earlier public REST check recorded in
this document found request-free prefetch PR #58198 and retention PR #38514;
this change keeps those execution/reuse and standalone-retention boundaries and
adds the merged-envelope control, logical state, policy composition, and
cross-engine orchestration layers. Authentication and a final human duplicate
review are required before publication.

After the liveness and recovery-test refinements in commit `0ea7d3f6f5`, the
final targeted regression was rerun:

```bash
.venv/bin/python -m pytest tests/v1/test_kv_hint_control.py \
  tests/v1/test_kv_hint_state.py tests/v1/engine/test_kv_control.py \
  tests/entrypoints/serve/kv_control/test_api.py \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py tests/v1/core/test_scheduler.py -q
```

Result: **399 passed**, 14 warnings, 193.93 seconds. The broader 574-test
control/cache/scheduler run and the 399-test final run cover overlapping suites;
no failures were observed after the final changes.

## Quality review corrections (2026-10-06)

The October 2 completion count and workload claims above overstate the evidence.
`rollout.py` submitted empty targets and never generated real pressure/inference.
`verl_agent_loop.py` used an unverified JSONL format, sent only resume signals,
and did not invalidate weights. `coverage_check.py` tested file existence only.
These placeholders were removed; `coverage.md`, `tasks.md`, the benchmark README,
PR draft and follow-up tracker now distinguish implemented mechanisms from
unexecuted acceptance. The delivery is **40/52**, with the previously overstated
items reopened. The existing real CPU/Mooncake artifacts remain historical
results; no previous run is relabeled as policy or rollout validation.

Changes in this quality pass:

- Polling prunes terminal messages using a single bounded expiry-heap entry per
  message, keeping active siblings and duplicate TTL semantics intact.
- Guidance observations honor the configured interval; elapsed waiting still
  enforces the admission bound between refreshes. Cache counts distinguish
  physical copies from aliases and choose sparse-index/dense-pool traversal.
- Invalid composer output terminates every registered child, so records become
  reclaimable instead of remaining received forever. Pending-snapshot limits
  include the retained hash target and empty-prefix records.
- Association query pages supply epoch/namespace/event sequence for resuming
  replay after a real history gap. Event pages avoid building an unused tail.
- Real quota/scheduler tests replace fake-policy wiring tests. The HTTP test
  actually exceeds the association replay buffer, reconstructs state and reads
  the next delta. The protocol probe asserts every outcome and selects an
  explicit engine using fresh IDs; it makes no restart/residency recovery claim.

A 628-case intermediate run reported **627 passed, 1 failed** (572.83 seconds):
using the alias-only reverse map for occupancy omitted primary copies. The fix
uses the primary hash map and dense-pool fallback. Targeted tests now cover both
paths, duplicate physical copies, aliases, active references and eviction.
A full cache/scheduler run passed **415 tests** (195.33 seconds); after the final
counting refinement the complete prefix-cache suite passed **216 tests**
(74.71 seconds). These suites overlap; counts must not be summed.

The reproducible `control_overhead.py` microbenchmark was run against detached
baseline `a41f867f69` and the updated source with identical arguments. Median
microseconds per call, five samples per measurement:

| Case | Baseline | Updated |
| --- | ---: | ---: |
| Poll 64 retained results | 3.675 | 0.188 |
| Poll 4,096 retained results | 220.817 | 0.190 |
| Cached guidance, 100,000 blocks | 10,987.637 | 0.153 |
| Fresh observation, empty 100,000-block pool | 8,441.472 | 4.109 |
| Fresh observation, populated 100,000-block pool | 11,334.206 | 9,593.808 |

Raw samples, interpreter, source hashes and command are in
`benchmarks/programmable_kv_cache/results/control-overhead-2026-10-06.json.gz`.
These are isolated CPU costs with no requests and a fixed observation clock.
They do not measure serving throughput, fairness, or disabled-feature overhead.

Final regression after the runtime/counting changes:

```bash
.venv/bin/python -m pytest tests/v1/test_kv_hint_control.py \
  tests/v1/test_kv_hint_state.py tests/v1/engine/test_kv_control.py \
  tests/entrypoints/serve/kv_control/test_api.py tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py -q
```

**603 passed**, 20 warnings, 252.98 seconds. The additional warnings are from
Starlette's in-process client ignoring socket timeouts in the HTTP probe tests;
the original 14 Torch deprecation warnings remain. The earlier 628-case run also
covered the complete async and randomized priority scheduler suites; its sole
failure was the occupancy assertion fixed above. The final targeted suite covers
that correction and all newly added behaviors.

All normal pre-commit hooks on changed files and manual `mypy-3.12` passed.
The rebuilt Qwen TP=2 server also passed `recovery.py` with explicit engine ID
`quality-quota`. This validates protocol errors/replay, not actual engine restart.

### Live weight-version guard

The quality review found that the version setter could silently preserve an old
native namespace. It now rejects changed versions while programmable control is
enabled; unchanged-version calls and the disabled feature path keep their behavior.
This is an explicit unsupported boundary, not a live invalidation implementation.
Callers must restart with the intended weights/version. Out-of-band tensor changes
are not intercepted. Task 7.2 remains incomplete.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/engine/test_kv_control.py -q
```

**29 passed**, 14 warnings, 10.80 seconds. Normal hooks and manual mypy 3.12 also
pass for the guard. An additional attempt to run all of `test_engine_core.py`
was stopped after its real-model/GPU cases reported failures. A completed failure
summary was not collected before cancellation, so no cause is attributed to those
failures. That file is not a CPU-only suite; this stopped attempt is not counted
as passing validation. The real serving checks below supply the model/EngineCore evidence.

### Qwen TP=2 with quota guidance enabled

The October 6 run uses the README native CPU server configuration on GPUs **0/3**,
with the following control configuration:

```json
{"engine_id":"quality-quota","policies":[{"name":"quota","implementation":"quota"}]}
```

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python \
  benchmarks/programmable_kv_cache/http_control.py \
  --base-url http://127.0.0.1:18083 \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --validate-movement cpu --validate-request-associations \
  --evaluation-samples 128 --output /tmp/kv-quality-qwen-quota-cpu-final.json
```

Recomputation and explicit CPU preparation both score **124/128 (96.875%)**,
with **zero** truncated answers and **zero** final numeric-answer mismatches.
Cached tokens are **0 / 100,352**. Driver times are **859.53 / 883.55 seconds**;
concurrent CPU tests and component benchmarks ran during portions of the workload,
so these times are not a controlled performance comparison. This configuration
exercises guidance in serving, not quota fairness or an adverse-pressure matrix.

The full report is saved in
`benchmarks/programmable_kv_cache/results/qwen3.8-27b-tp2-quality-quota-cpu.json.gz`.
The final dense-pool traversal optimization and weight-version setter guard were
added after this long-run server started. The fresh-server HTTP validation below
covers the final source; the 128-question report identifies this process boundary.

The final source was then loaded into a fresh Qwen TP=2 server with the same CPU
and quota configuration. Running the command above without `--evaluation-samples`
passes the complete HTTP movement/request-association workflow: **four** copies
reclaimed/restored, **784** cached tokens on resume and identical short greedy
outputs. An actual stop/restart also verifies a changed engine epoch, rejection
of the captured previous epoch, absence of the last old operation ID and an empty
new association snapshot. This does not complete the full failure/race matrix.
The complete responses and final source hashes are in
`results/qwen3.8-27b-tp2-quality-head-http.json.gz` under the benchmark directory.
All model servers started for this quality review were stopped after validation.
