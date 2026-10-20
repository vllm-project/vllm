# Programmable KV cache tasks

## 1. Resolve remaining design questions

- [x] 1.1 Record the user's scheduler-cooperation scope and RL workload choice; update design and affected specs before implementing dependent behavior.
- [x] 1.2 Agree on the control transport/profile, explicit engine routing, target coverage, CPU adapter, and per-backend capability matrix.
- [x] 1.3 Specify operation statuses, deduplication/retry limits, declaration timing, task cleanup ordering, and bounded index/result storage.
- [x] 1.4 Specify reference policies, composer conflict rules, and any optional admission/score behavior without allowing cache plugins to mutate queues directly.
- [x] 1.5 Identify the model, usable GPU allocation, Mooncake deployment, and open-source rollout example for reproducible validation.
- [x] 1.6 Confirm application control through the merged hints, explicit state commands separate from policy signals, and incremental association events with queries.
- [x] 1.7 Define snapshot/follow-growth target scope, derived-plan outcomes, and stale/retried logical-state updates.
- [x] 1.8 Confirm lightweight orchestrator for first delivery, one interpretation path per entire envelope, preserved KVCR forwarding, and a separate future-work tracker.
- [x] 1.9 Specify native/backend route selection in invocation/configuration, consumer capability reporting, and native payload validation without changing merged hint fields.

## 2. Environment and test design

- [x] 2.1 Set up the uv-managed Python environment, editable vLLM dependencies, and required pre-commit hooks.
- [x] 2.2 Map behavior checks to existing nearby suites; document module contracts, guarded failure modes, and the cheapest effective test level before adding tests.
- [x] 2.3 Record baseline versions/configurations and run complete relevant CPU and Mooncake correctness suites plus real-transfer integration in the chosen environment; smoke tests do not satisfy acceptance.

## 3. Control contracts and logical state

- [x] 3.1 Implement validated action payloads, target/capability records, and bounded correlated operation results using the existing envelope.
- [x] 3.2 Implement the logical session/continuation index and hash/group-aware resolution, including cache-hit associations and reported coverage.
- [x] 3.3 Implement independent Retain declarations, expiry, Release, and explicit task-tree cleanup.
- [x] 3.4 Verify shared-prefix lifetimes, branch selection, missing content, duplicate control, and unconsumed signals.
- [x] 3.5 Implement consumption of the merged hint contract for supported cache/state actions, with shared request-attached/request-free validation and action/version capability discovery.
- [x] 3.6 Implement the agreed explicit association/continuation/task management commands separately from optional policy signals.
- [x] 3.7 Implement bounded logical-state queries and the agreed association events, including new membership on cache hits.
- [x] 3.8 Verify derived-step correlation, partial effects, unknown versions, bounded/pruned index behavior, and stale logical updates.
- [x] 3.9 Implement whole-envelope native/backend dispatch and ensure native-consumed envelopes are not also forwarded through KVCR's existing request hook.

## 4. Local execution and policy integration

- [x] 4.1 Add engine-owned operation approval, reservation/reference accounting, cancellation, and stale-completion protection through existing KV manager paths.
- [x] 4.2 Implement retention-aware eviction and explicit local Evict while preserving active references and the feature-disabled path.
- [x] 4.3 Add startup-configured policy/composer interfaces and a common pressure/demand snapshot.
- [x] 4.4 Implement the agreed reference policies and scheduler-cooperation extension.
- [x] 4.5 Verify deduplicated work, competing capacity requests, declaration independence, and pressure-induced deferral.

## 5. Independent backend execution

- [x] 5.1 Add request-free operation dispatch/progress/completion to engine and connector liveness without synthetic inference requests.
- [x] 5.2 Adapt the selected CPU offload path for approved Demote and Prefetch, including resume/load coordination.
- [x] 5.3 Adapt Mooncake Store lookup, approved buffers, transfer metadata, and completion for independent operations using unchanged backend APIs.
- [x] 5.4 Implement the agreed Share orchestration/result semantics and per-backend unsupported results.
- [x] 5.5 Verify real transfer correctness, partial failures, cancellation, all-rank completion, reset races, and source reclamation timing.
- [x] 5.6 Extend existing KVCR forwarding/query/deliver/cleanup regression coverage; preserve opaque payload values and prevent forwarding receipts from claiming GPU readiness.

## 6. Serving and observability

- [x] 6.1 Wire request-attached hints through the agreed public serving paths into the common control implementation.
- [x] 6.2 Expose independent submission, capability queries, and operation results through the agreed transport with explicit engine targeting.
- [x] 6.3 Add association/residency/result reporting and restart/gap handling from common state records.
- [x] 6.4 Verify serving protocol compatibility, version errors, engine routing, and agreement between events and queries.

## 7. Examples and evaluation

- [x] 7.1 Add a reproducible multi-round shared-prefix rollout workload with controllable pressure, pauses, branches, and resume predictions.
- [x] 7.2 Add the agreed open-source workload integration and RL weight-version correctness coverage.
- [x] 7.3 Add and run the two-engine Mooncake example, proving actual destination cache hits after request-free preparation.
- [ ] 7.4 Run native/router-directive/local-policy/composed-policy comparisons and labeled oracle experiments; save raw measurements and commands.
- [ ] 7.5 Run relevant model evaluations and measure overhead, regressions, admission delay, fairness, transfer volume, and failure recovery.
- [x] 7.6 Add and run a direct-hint controller without local policies, covering request-free retention, movement, query, and release.
- [x] 7.7 Add and run session/branch/range/shared-prefix workflows, asserting logical cleanup and independent retention lifetimes.
- [x] 7.8 Add an application-directed rollout controller using supported Retain/Prefetch hints and actual operation results, with no custom hint types or handler plugins.
- [ ] 7.9 Add and run closed-loop control/recovery examples for duplicate hints, prefetch/resume races, capacity deferral, backend errors, and engine restart.
- [x] 7.10 Check all rows in coverage.md against concrete examples and tests; distinguish implemented, unsupported backend scope, and unexecuted validation.
- [x] 7.11 Run the primary lightweight-orchestrator → envelope → native vLLM semantics → real Mooncake scenario; correlate each action with backend completion and actual reuse.

## 8. Complete the reviewable delivery

- [x] 8.1 Document configuration, plugin extension contracts, operation semantics, backend coverage, examples, and measured limitations.
- [x] 8.2 Run relevant unit/integration suites and required lint checks; record commands and results.
- [x] 8.3 Repeat duplicate-work checks and document the concrete differences/reuse boundaries with related retention and prefetch work.
- [ ] 8.4 Prepare one complete PR description with AI disclosure, test/model-evaluation evidence, and human review/testing requirements; publication must follow repository contribution rules.
- [x] 8.5 Reconcile delivered coverage with the follow-up scope in integration-paths.md; keep first-release failures/incomplete validation in this task list rather than relabeling them future work.

## Evidence corrections, 2026-10-06

The October 2 count of 48/52 included unsupported completion claims.
Reopened:

- 6.3: association/result queries exist, but live residency reporting is absent.
- 7.1/7.8: the removed rollout script operated on empty targets and never ran inference, applied pressure or consumed resume predictions.
- 7.7: existing state/unit checks and HTTP branches do not cover the complete requested range/task-tree/shared-retention workflow as an executed example.
- 7.11: explicit Mooncake/cross-engine runs exist; optional native-policy G2 execution and its correlated real-transfer/reuse trace are still unexecuted.
- 8.2: full delivery validation, including the remaining serving/model gates, remains incomplete; passing phase checks are recorded separately.
- 8.3: all three CLI checks failed authentication; they did not establish absence of duplicates.
- 8.4: the PR description is a draft, pending duplicate and complete validation evidence.
  It must not claim runnable rollout or verl integration.

7.2/7.4/7.5/7.9 remain incomplete.
The removed verl script consumed an invented JSONL format and sent only a resume signal; it did not integrate Agent Loop or invalidate KV on a weight update.
File presence is not coverage evidence; the coverage inventory is now audited manually against concrete tests and results.

## 9. RFC #59708 alignment on rfc-hint

- [x] 9.1 Specify primitive payloads, lifecycle phases, ownership, compatibility and the test contract.
- [x] 9.2 Add one typed dispatcher for public primitives and legacy actions with correlated queued/completion results.
- [x] 9.3 Integrate copy-scoped revisioned priority into the existing eviction authority and successful request completion.
- [x] 9.4 Execute COPY/MOVE through existing resource leases and backend adapters, reporting per-target partial outcomes.
- [x] 9.5 Make optional policies produce the same primitives; preserve legacy directives and opaque backend routing.
- [x] 9.6 Add and run the request-attached Mooncake reference workflow and request-free/cross-engine extensions, relevant regression/model checks and documentation.

## 10. Real rollout performance and runtime completion

Stopped at the user's requested boundary on October 12 after quota confirmation and a real live rollout.
See the [experiment report](../../../docs/programmable-kvc/rollout-report-2026-10-12.md) and [test guide](../../../docs/programmable-kvc/testing.md).
Pressure/fit and ablations ran; forecast sensitivity, the new fault/restart/128-question harness and final broad checks did not.
Their encompassing tasks remain unchecked.
At that stop point progress was 56/64 and no experiment or regression queue remained active.

- [x] 10.1 Implement guarded weight-update lifecycle and namespace transition; verify real tensor updates cannot reuse old local or remote KV.
- [x] 10.2 Add bounded current-residency queries and control-result acknowledgement with preserved duplicate semantics and observable capacity.
- [x] 10.3 Improve policy demand/movement budgets and hot-path observation costs with behavioral coverage.
- [x] 10.4 Integrate the real verl tool loop and executable tools with an application-directed controller, generated branches, fallback and cleanup.
- [ ] 10.5 Run repeated fixed-work and live rollout comparisons across cache pressure, policy ablations and adverse predictions; evaluate the 25% throughput target and preserve all results.
- [ ] 10.6 Run relevant regression/lint/model checks and reconcile runtime documentation, acceptance gaps and the final performance report.

## 11. Control overhead and deployment-scale experiments

Resumed on October 18 at the user's request. The October 12 stop boundary no longer applies.
On October 19 the user requested incremental tested commits and prioritized speed over detailed output checks.
Completed `quota-wait120-aging30` and stopped all experiment launchers at the user's requested boundary.
Ordinary native/quota 30/1/quota 120/30 runs take 1,994.30/1,898.86/1,369.48 seconds at utilization 0.5 and equal work; the latter exceeds the 25% target in this single run.
Repeated cache-fit/pressure, remaining ablations and external models stay incomplete; detailed output validation was deferred by the user.

- [x] 11.1 Remove avoidable control waits and queries while preserving result collection, resource publication and cleanup ordering; add behavioral regressions.
- [ ] 11.2 Run repeated cache-fit and pressure workloads at gpu_memory_utilization >= 0.5, with larger rollout counts and prefix working sets.
- [ ] 11.3 Analyze request phases, control RPCs, CPU profiles and GPU logs; retain adverse prediction and regression results.
- [ ] 11.4 Complete relevant unit/adapter/GPU, lifecycle, fault/restart and model-quality checks on the final implementation.
- [x] 11.5 Record commands, raw evidence locations, measured conclusions and remaining external coverage; pass required code checks and commit locally.

## 12. Application timing and live admission, 2026-10-19

Resumed by the user's request to analyze experiment evidence and improve core
performance. The previous requested stop boundary is superseded for this work.

- [x] 12.1 Audit raw pause, control, transfer and queue data; document the optimization and behavioral test contracts.
- [x] 12.2 Use application pause estimates to suppress unamortized round trips and verify resume/cleanup behavior.
- [x] 12.3 Revalidate quota capacity across admissions within one observation interval; preserve bounded waiting and shared-prefix behavior.
- [x] 12.6 Couple optional preparation with selected-session admission, verifying publication, failure fallback and cancellation.
- [x] 12.7 Avoid request scans and guidance invalidation for signals that need neither; preserve custom-policy observations and scheduler age history.
- [x] 12.8 Reuse the scheduler's actual cache hit for live admission demand, so a preceding admission cannot cause a shared prefix to be charged twice.
- [x] 12.9 Keep completed/resident prefix members protected until multi-block preparation finishes, preserving shared cancellation and reset safety.
- [x] 12.10 Cancel unused preparations before final result drain and acknowledge timed-out operations without hiding their failure; retain failure correlation.
- [x] 12.11 Check complete preparation capacity before starting; preserve admission fairness and skip automatic resume work after admission without suppressing explicit importance.
- [x] 12.12 Amortize short-pause movement against current KV occupancy and sustained admission backlog; preserve ample-capacity reuse and retain forced-pressure regression evidence.
- [x] 12.13 Evaluate GPU-source COPY backups against the validated MOVE policy and retain the measured selection without unused policy options.
- [ ] 12.4 Run repeated native/quota/cooperative full-work comparisons and relevant negative cases, recording workload fingerprints and per-family tails.
- [ ] 12.5 Run relevant regression and lint checks, document measured limits, and commit complete tested changes locally.
