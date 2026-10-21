# KV workload validation requirements

## ADDED Requirements

### Requirement: Reproducible rollout and agentic workloads

Validation SHALL include multi-round rollout, shared prefixes and branches, pause/resume, and memory pressure.
Workload definitions SHALL expose repeatable seeds/configuration and SHALL preserve equivalent model-visible work across comparisons.

#### Scenario: Shared-prefix rollout under pressure

- **WHEN** several rollout branches compete for a cache smaller than their total working set
- **THEN** measurements capture whole-workload completion and cache behavior for all branches, including those delayed or displaced by the policy

### Requirement: Compare mechanism and policy contributions

Performance validation SHALL compare the native baseline, explicit router directives with engine feasibility and retry, individual policy components, and composed policies.
Any oracle schedule SHALL be labeled separately.

#### Scenario: Local policy improves one branch but harms others

- **WHEN** a policy reduces one branch's resume latency and increases others' waiting
- **THEN** the report includes both effects and aggregate completion rather than only the selected branch's gain

### Requirement: Serving correctness and overhead evidence

Validation SHALL cover output correctness appropriate to the model/configuration, transfer failures, cancellation, shared declarations, stale state, and no-feature overhead.
Reports SHALL include versions, commands, actual results, TTFT/tails, throughput, admission delay, reuse/recomputation, movement volume, and CPU/index cost where applicable.

#### Scenario: Simulation without real transfer

- **WHEN** a unit test or simulation exercises resource accounting
- **THEN** its result is labeled as correctness/simulation evidence rather than a measured end-to-end serving speedup

### Requirement: Real cross-engine acceptance

The first complete delivery SHALL run the Mooncake cross-engine example and record actual results.
An unexecuted script SHALL NOT satisfy this acceptance requirement.

#### Scenario: Remote environment unavailable

- **WHEN** the example cannot run because the backend/model/GPU environment is unavailable
- **THEN** the validation remains explicitly incomplete

### Requirement: Weight-version correctness

Rollout performance comparisons SHALL operate within one weight version.
Separate correctness validation SHALL establish that local and external KV from an obsolete weight version cannot be reused as current content after weight updates.

#### Scenario: Weight update with cached remote content

- **WHEN** a new weight version becomes active while a prefix from the previous version remains in a remote store
- **THEN** the old copy cannot satisfy a new-version request or be published by a stale control completion

### Requirement: Record the validated model and hardware scope

Acceptance SHALL use Qwen3.8-27B TP=2 on the user-assigned local GPU pair.
The initial allocation was GPUs 0 and 3; the October 20 continuation uses GPUs 0 and 2.
Reports SHALL keep the two allocations' measurements separate and SHALL NOT calculate speedups across them.
Reports SHALL record the actual cache-group/layout and hardware coverage and SHALL NOT generalize measured results to untested configurations.

#### Scenario: Local evaluation succeeds

- **WHEN** the local comparison and correctness checks pass
- **THEN** the report records the tested model, hardware, layouts and workload as the scope of its conclusions

### Requirement: Deployment-scale overhead comparisons

Primary performance comparisons SHALL use `gpu_memory_utilization` of at least 0.5 without a small fixed KV-byte override.
Validation SHALL increase workload pressure through distinct prefixes, rollout branches and concurrency, and retain a cache-fit comparison at the same memory setting.
Reports SHALL separate instrumented profiling from normal timing and record equal-work fingerprints, control RPC cost and request phase breakdowns.

#### Scenario: Cache fits the deployment memory budget

- **WHEN** the workload fits and optional control provides no extra reuse
- **THEN** the comparison still measures its control and scheduler overhead against native serving and preserves any regression
