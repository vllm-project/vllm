# Validation plan

Each implementation phase runs the complete relevant suites before its commit.
Smoke tests are not acceptance evidence. Record commands, versions, counts,
failures and skipped capabilities; distinguish mocks from real transfers.

## Contracts and cheapest effective checks

| Module purpose | Input/output contract | Failure guarded against | Test level and existing suite |
| --- | --- | --- | --- |
| Hint validation/results | Merged envelope plus context → correlated, bounded results | Duplicate effects, conflicting retries, false completion, result loss | New `tests/v1/test_kv_hint_control.py`; existing `test_request.py` only tests request transport and has no control lifecycle suite |
| Logical index/declarations | Session/continuation updates → group-aware resolution, independent expiry | Branch confusion, cache-hit membership loss, shared-prefix cleanup, stale writes | New `tests/v1/test_kv_hint_state.py` for the new pure state module; cache hooks in existing prefix-cache suites |
| GPU resource authority | Resolved content/current refs → approved buffers or explicit refusal | Double allocation, stale registration, releasing active memory | `tests/v1/core/test_prefix_caching.py`, `test_deferred_block_free.py`, `test_single_type_kv_cache_manager.py` |
| Policy and scheduler | Read-only observations/proposals → bounded guidance | Rewriting directives, starvation, double-counted quota, changed preemption | `tests/v1/core/test_scheduler.py`, `test_async_scheduler.py`, `test_priority_scheduler_random.py` |
| CPU adapter | Approved transfer → real bytes and completion | Buffer lifetime, races, failed-load registration | `tests/v1/kv_offload/cpu/test_manager.py`, `test_gpu_worker.py`, `tests/v1/kv_connector/unit/offloading_connector/` |
| Mooncake adapter | Group/rank metadata → storage/load outcomes | Missing shards, incompatible boundaries, reset and completion races | Complete `tests/v1/kv_connector/unit/test_mooncake_store_*.py` suites, then real Store/GPU integration |
| KVCR compatibility | Opaque envelope/request → forwarding/query/deliver/cleanup | Native double consumption, payload mutation, premature readiness | `tests/v1/kv_offload/tiering/test_kvcr_tier.py` |
| Serving | Python/HTTP targeting → same engine records | Wrong engine/epoch, version errors, ingress mismatch | Existing engine and OpenAI serving suites plus control endpoint integration |

State tests use a fake monotonic clock, bounded capacities, and observable queries.
Request-free transfer approval maps a group/hash/boundary and caller interest to
an existing source reference or an unpublished destination allocation. Extend
`test_prefix_caching.py` with actual KVCacheManager instances to guard against
duplicate capacity charges, premature hits, cancelled-buffer reuse, stale reset
completion, alias boundary confusion, and changes to active request references.
Worker completion is injected only for this resource unit contract; actual copy
correctness and all-rank completion require the subsequent adapter integration.
Native movement maps a correlated action to shared transfer leases and terminal
results. Extend the same core suite with a queued adapter to check independent
task cancellation, capacity/partial failures, source preservation, and reset races
through native submission and operation queries. This fake adapter verifies the
engine contract only; Mooncake worker and live-model tests must prove real bytes.
Mooncake control metadata carries approved leases, and each worker reports its
rank/generation/result after buffer access stops. Extend the Store scheduler and
worker suites to check duplicate/stale rank reports, no-request liveness, load
coordination and byte-count/error handling. Extend the existing real-buffer suite
to run the actual control worker on registered GPU layouts and observe subsequent
KVCacheManager hits after all ranks complete. Finally exercise the real Qwen
HTTP/EngineCore path to validate scheduler and model-runner integration.
CPU control uses the existing CPUOffloadingManager chunk reservations and
CPUOffloadingWorker copies. Extend the manager/connector suites for cold capacity,
independent source/destination holds, missing objects, failed-rank acknowledgement,
reset rejection during transfer, and native jobs producing no inference receipts.
GPU worker and live-model checks must exercise both directions and actual reuse.
Policy tests extend the existing control/state and scheduler suites. Inputs are
immutable capacity/request observations and signals; outputs are native proposals
or advisory admission/scores. Guard against duplicate signal effects, lost parent
correlation, stale predicted resumes, policy mutation of explicit declarations,
unbounded proposal state, and policy-induced starvation. Use fake clocks for
prediction and aging, actual queues for admission, and the same native transfer
fixtures for conflicting proposals and shared work. Then run real policy-driven
Mooncake and CPU workflows; simulated policy outcomes do not establish speedup.
Request association consumes active request token/hash observations and produces
the shared logical index plus an optional bounded growth subscription. Guard
against rewrites below the first cache-block boundary, implicit growth of snapshot
commands, busy concurrent writers, inherited-prefix mismatch, lost alias coverage,
late observations after cleanup/recreation, and silently dropped capacity errors.
Use actual Request/KVCacheManager instances in the existing engine control suite;
assert common action outcomes, state/events and unchanged allocation/reference
counts. Metadata-only growth must preserve the retention projection and TTL.
Then exercise request-addressed associations and generated growth on real Qwen
HTTP serving, alongside the existing GPU eviction/reuse workflow.
Control utility input is a method, strict invocation context and JSON payload;
output is JSON-compatible common records or a structured error. Guard against DP
mutation broadcasts, ambiguous engine names, stale epochs, mixed pagination,
unsafe serialization and HTTP/Python disagreement. Use the existing control suite
for the facade and new model-free `engine/test_kv_control.py` for client routing:
the older engine-client module initializes remote model tokenizers at import and
cannot provide isolated control routing tests. HTTP ingress tests use an actual
control service through FastAPI, followed by a real GPU/MP serving run.
Continuation updates must preserve known hash/group/span associations. Guard
against rewrite, truncation, group substitution, and tail replacement even with a
current revision; assert unchanged queries, reverse membership and event position.
Accept growth retaining a short alias and newly observed groups. Check that a new
branch cannot inherit a named follow-growth declaration or extend its expiry.
Use seeded interleavings for shared declarations, retries, cancellation and reset;
do not replace real worker/engine integration with a simulated operation loop.

The native Store transport's contract is registered source/destination buffers →
per-key results and exact copied values. Guard against batch failure corrupting
successful members or treating Get's byte counts as errors. The cheapest effective
level is native Store integration on real GPUs: the existing accuracy scripts test
the separate PD connector and existing Store unit suites mock the backend, so
`mooncake_integration/test_store_buffers.py` supplies this missing suite.

## Real execution acceptance

1. Run existing CPU worker transfer correctness on allocated GPUs, including both
   copy directions, layouts and failure/reference cleanup.
2. Run real Mooncake Store Put/Get with actual GPU buffers and per-rank completion.
   Existing `test_mooncake_store_hma_e2e.py` uses a dict-backed mock and cannot satisfy
   this requirement despite its filename.
3. Run lightweight orchestrator → HTTP/Python → native hint semantics → Store.
   Verify no inference request is required for operation progress; only report
   prefetch ready after registration and subsequent actual prefix-cache reuse.
4. Compare deterministic outputs/logprobs with no-feature baselines; exercise TP=2,
   shared prefixes, hybrid groups, branches, partial ranges and weight invalidation.
5. Run failures: one failed rank, missing object, reset during transfer, duplicate
   commands, partial demotion, resume racing prefetch, capacity deferral and restart.
6. Run native/router-directed/individual-policy/composed-policy workloads and
   separately labeled oracle schedules. Include all branches and delayed requests.
   Save raw timings, transfer volume, displacement, admission delay, throughput,
   TTFT/tails, recomputation, index memory and CPU overhead.
7. Run the external large-model matrix when the machines become available. Keep
   unexecuted results explicitly incomplete in `tasks.md`.

## Initial environment observations

- Source baseline: `0ee724bb7a` (upstream code ancestor `30f5c01d4e`).
- New uv-managed CPython 3.12 environment; pre-commit and commit-msg hooks installed.
- Local uv upgraded inside `.venv` because the host uv cannot parse this project's
  `exclude-dependencies` configuration. No global Python environment modification.
- GPUs 0/3: A100 PCIe 80GB; driver 595.71.05. GPU 2 excluded by allocation.
- Editable precompiled installation and full test dependencies: pending completion.
- GitHub CLI is unauthenticated. Public REST checks found request-free prefetch
  PR #58198 and retention PR #38514. Preserve their execution/reuse boundaries;
  repeat required authenticated CLI checks before PR publication. No PR created.

## Quality review, 2026-10-06

- OperationRegistry maps message/action IDs to retained results. Exercise terminal
  expiry in completion order, empty messages, replay and active siblings through
  submit/get/update with a fake clock. Optimize lookup without weakening the TTL.
- Policy guidance maps an immutable observation to scheduler-owned admission and
  ordering. Replace fake-policy wiring checks with the real quota policy and real
  queues; check priority, bounded waiting, observation throttling and overflow.
  Fake clocks make deadlines deterministic; CPU scheduler tests are sufficient.
  Invalid composer output must terminate child records; snapshots must account
  for original hash targets outside the resolved range. Extend the existing
  movement fixture and verify reclamation/capacity through native submissions.
- Association queries must provide a cursor from which a consumer can resume
  replay. Force actual bounded-history loss through HTTP, reconstruct state, then
  consume a subsequent delta; a forged future cursor alone is not gap recovery.
- Example controllers must check correlated terminal outcomes on nonempty targets.
  Exercise them against the model-free HTTP control fixture for failure/replay;
  label this separately from live transfer and model reuse validation.

## RFC primitive alignment

Actions map typed payloads to queued or approved operations; test wire validation,
unsupported locations, duplicate IDs and lifecycle timing in the existing control
and engine suites. Copy priority maps actual cached copies/claims to eviction
order; extend prefix-cache tests for signed priorities, overlapping independent
claims, revisions/TTL and reuse/reset, using the real manager and fake clocks.
Scheduler hooks must run before free and preserve abort/preemption semantics;
extend sync/async scheduler cases, including source holds acquired at completion.
Transfers still use the existing queued-adapter fixtures for partial scope,
capacity, cancellation and all-rank errors. Then run real Mooncake buffers and a
request-attached R1 MOVE / R2 COPY workflow on GPUs 0/3, followed by model eval.
No fake inference request may be used to advance a completed request's transfer.
