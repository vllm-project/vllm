# Programmable KV cache tasks

## 1. Resolve remaining design questions

- [x] 1.1 Record the user's scheduler-cooperation scope and RL workload choice; update design and affected specs before implementing dependent behavior.
- [ ] 1.2 Agree on the control transport/profile, explicit engine routing, target coverage, CPU adapter, and per-backend capability matrix.
- [x] 1.3 Specify operation statuses, deduplication/retry limits, declaration timing, task cleanup ordering, and bounded index/result storage.
- [x] 1.4 Specify reference policies, composer conflict rules, and any optional admission/score behavior without allowing cache plugins to mutate queues directly.
- [x] 1.5 Identify the model, usable GPU allocation, Mooncake deployment, and open-source rollout example for reproducible validation.
- [x] 1.6 Confirm application control through the merged hints, explicit state commands separate from policy signals, and incremental association events with queries.
- [ ] 1.7 Define snapshot/follow-growth target scope, derived-plan outcomes, and stale/retried logical-state updates.
- [x] 1.8 Confirm lightweight orchestrator for first delivery, one interpretation path per entire envelope, preserved KVCR forwarding, and a separate future-work tracker.
- [ ] 1.9 Specify native/backend route selection in invocation/configuration, consumer capability reporting, and native payload validation without changing merged hint fields.

## 2. Environment and test design

- [x] 2.1 Set up the uv-managed Python environment, editable vLLM dependencies, and required pre-commit hooks.
- [x] 2.2 Map behavior checks to existing nearby suites; document module contracts, guarded failure modes, and the cheapest effective test level before adding tests.
- [x] 2.3 Record baseline versions/configurations and run complete relevant CPU and Mooncake correctness suites plus real-transfer integration in the chosen environment; smoke tests do not satisfy acceptance.

## 3. Control contracts and logical state

- [x] 3.1 Implement validated action payloads, target/capability records, and bounded correlated operation results using the existing envelope.
- [ ] 3.2 Implement the logical session/continuation index and hash/group-aware resolution, including cache-hit associations and reported coverage. Logical index/resolution implemented; engine request/cache hooks and physical group validation remain.
- [x] 3.3 Implement independent Retain declarations, expiry, Release, and explicit task-tree cleanup.
- [x] 3.4 Verify shared-prefix lifetimes, branch selection, missing content, duplicate control, and unconsumed signals.
- [ ] 3.5 Implement consumption of the merged hint contract for supported cache/state actions, with shared request-attached/request-free validation and action/version capability discovery.
- [x] 3.6 Implement the agreed explicit association/continuation/task management commands separately from optional policy signals.
- [ ] 3.7 Implement bounded logical-state queries and the agreed association events, including new membership on cache hits.
- [ ] 3.8 Verify derived-step correlation, partial effects, unknown versions, bounded/pruned index behavior, and stale logical updates.
- [ ] 3.9 Implement whole-envelope native/backend dispatch and ensure native-consumed envelopes are not also forwarded through KVCR's existing request hook.

## 4. Local execution and policy integration

- [ ] 4.1 Add engine-owned operation approval, reservation/reference accounting, cancellation, and stale-completion protection through existing KV manager paths.
- [x] 4.2 Implement retention-aware eviction and explicit local Evict while preserving active references and the feature-disabled path.
- [ ] 4.3 Add startup-configured policy/composer interfaces and a common pressure/demand snapshot.
- [ ] 4.4 Implement the agreed reference policies and scheduler-cooperation extension.
- [ ] 4.5 Verify deduplicated work, competing capacity requests, declaration independence, and pressure-induced deferral.

## 5. Independent backend execution

- [ ] 5.1 Add request-free operation dispatch/progress/completion to engine and connector liveness without synthetic inference requests.
- [ ] 5.2 Adapt the selected CPU offload path for approved Demote and Prefetch, including resume/load coordination.
- [ ] 5.3 Adapt Mooncake Store lookup, approved buffers, transfer metadata, and completion for independent operations using unchanged backend APIs.
- [ ] 5.4 Implement the agreed Share orchestration/result semantics and per-backend unsupported results.
- [ ] 5.5 Verify real transfer correctness, partial failures, cancellation, all-rank completion, reset races, and source reclamation timing.
- [ ] 5.6 Extend existing KVCR forwarding/query/deliver/cleanup regression coverage; preserve opaque payload values and prevent forwarding receipts from claiming GPU readiness.

## 6. Serving and observability

- [ ] 6.1 Wire request-attached hints through the agreed public serving paths into the common control implementation.
- [ ] 6.2 Expose independent submission, capability queries, and operation results through the agreed transport with explicit engine targeting.
- [ ] 6.3 Add association/residency/result reporting and restart/gap handling from common state records.
- [ ] 6.4 Verify serving protocol compatibility, version errors, engine routing, and agreement between events and queries.

## 7. Examples and evaluation

- [ ] 7.1 Add a reproducible multi-round shared-prefix rollout workload with controllable pressure, pauses, branches, and resume predictions.
- [ ] 7.2 Add the agreed open-source workload integration and RL weight-version correctness coverage.
- [ ] 7.3 Add and run the two-engine Mooncake example, proving actual destination cache hits after request-free preparation.
- [ ] 7.4 Run native/router-directive/local-policy/composed-policy comparisons and labeled oracle experiments; save raw measurements and commands.
- [ ] 7.5 Run relevant model evaluations and measure overhead, regressions, admission delay, fairness, transfer volume, and failure recovery.
- [ ] 7.6 Add and run a direct-hint controller without local policies, covering request-free retention, movement, query, and release.
- [ ] 7.7 Add and run session/branch/range/shared-prefix workflows, asserting logical cleanup and independent retention lifetimes.
- [ ] 7.8 Add an application-directed rollout controller using supported Retain/Prefetch hints and actual operation results, with no custom hint types or handler plugins.
- [ ] 7.9 Add and run closed-loop control/recovery examples for duplicate hints, prefetch/resume races, capacity deferral, backend errors, and engine restart.
- [ ] 7.10 Check all rows in coverage.md against concrete examples and tests; distinguish implemented, unsupported backend scope, and unexecuted validation.
- [ ] 7.11 Run the primary lightweight-orchestrator → envelope → native vLLM semantics → real Mooncake scenario; correlate each action with backend completion and actual reuse.

## 8. Complete the reviewable delivery

- [ ] 8.1 Document configuration, plugin extension contracts, operation semantics, backend coverage, examples, and measured limitations.
- [ ] 8.2 Run relevant unit/integration suites and required lint checks; record commands and results.
- [ ] 8.3 Repeat duplicate-work checks and document the concrete differences/reuse boundaries with related retention and prefetch work.
- [ ] 8.4 Prepare one complete PR description with AI disclosure, test/model-evaluation evidence, and human review/testing requirements; publication must follow repository contribution rules.
- [ ] 8.5 Reconcile delivered coverage with docs/programmable-kvc/task2do.md; keep first-release failures/incomplete validation in this task list rather than relabeling them future work.
