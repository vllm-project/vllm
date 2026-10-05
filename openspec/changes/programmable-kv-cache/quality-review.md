# Implementation quality review, 2026-10-06

Baseline: `a41f867f69`. Scope: the programmable-KV control implementation,
optional policies/scheduler integration, examples, tests and delivery claims.

## Fixed findings

| Finding | Effect | Correction and regression evidence |
| --- | --- | --- |
| Every operation lookup scanned all retained messages for expiry | Polling cost grew with all 4,096 records, including in policy reap loops | One bounded heap entry per terminal message; TTL, active-sibling, completion-order and ID-reuse tests |
| Scheduler guidance ignored the configured observation interval | Each scheduling step scanned requests and the full block pool | Cache guidance at the configured cadence; refresh after signals/reset generation; maximum-wait checks use current time |
| Cache count traversal cost depended on all allocated blocks | Empty/sparse caches paid for untouched capacity | Traverse primary cache entries when sparse, scan the pool when dense; count aliases once and shared physical copies separately |
| Invalid composer return types left registered child operations live forever | Active record capacity could be permanently exhausted | Validate within the cleanup boundary; terminate every child and prove a subsequent command can reuse capacity after TTL |
| Pending snapshot limits omitted original hash-target entries and empty prefixes | Excluded ranges could still retain content beyond the advertised snapshot budget | Charge retained targets and resolved prefixes before admission; verify deferral without dispatch |
| Live weight-version metadata could change while control kept the old namespace | Callers could mistake a new version stamp for KV invalidation | Reject version changes when programmable control is enabled; preserve same-version retries and the disabled path |
| Association snapshots omitted the event position | Consumers could query state after a gap but lacked a reliable replay cursor | Include epoch, namespace and event sequence; force a real HTTP history gap, query state, then read the next membership delta |
| Scheduler tests replaced the policy with hand-written selection stubs | They passed without verifying actual quota, score, priority or aging behavior | Use the startup-configured quota policy with real sync/async scheduler queues and fake clocks |
| Recovery script ignored several outcomes and reused fixed message IDs | Incorrect version/gap responses could still produce a passing report | Explicit engine selection, fresh IDs, exact replay/query checks, error-status assertions and negative response tests |
| Three workload/coverage scripts were placeholders | Empty targets/file existence were presented as rollout, verl and delivery evidence | Remove the placeholders, audit every coverage row, reopen unsupported completion claims; progress is 40/52 |

## Boundaries still requiring implementation or validation

- Live weight changes with programmable control are now explicitly rejected.
  Restart with the intended weights/version until namespace transition, transfer
  draining and local/external invalidation are integrated (task 7.2). Arbitrary
  out-of-band weight mutation is not protected by this version-setter guard.
- Physical residency reporting is incomplete. Association state and historical
  operation completion are not current residency guarantees (6.3).
- Real rollout/verl workloads, branch/range/task-tree examples, policy-driven
  Mooncake reuse and full restart/failure recovery remain incomplete. The existing
  direct CPU/Mooncake artifacts demonstrate their documented narrower scenarios.
- Waiting-request selection still scans queues; full-cache observations still
  scan cached metadata. The local component benchmark is not evidence of serving
  throughput, fairness, disabled-path cost or external model coverage.
- Successful duplicate-work checks and a complete PR description remain required
  before publication. An unauthenticated CLI attempt is not a duplicate check.

Concrete coverage is in [coverage.md](coverage.md); commands, raw measurements,
intermediate failures and final results are in [validation.md](validation.md).
