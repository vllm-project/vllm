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

### Requirement: Distinguish local debugging from full evaluation

Local Qwen3.8-27B TP=2 debugging SHALL NOT be reported as comprehensive performance validation.
The full evaluation matrix SHALL include GPT-OSS-120B and DeepSeek-V4-Flash on the designated external machines and record their actual cache-group/layout coverage.

#### Scenario: Local debug succeeds

- **WHEN** the local debug scenario passes
- **THEN** the report records that result while leaving unexecuted external model evaluations incomplete
