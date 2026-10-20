# Representative KV Workload Requirements

## ADDED Requirements

### Requirement: Bounded local workload selection

The experiment system SHALL support three workloads: SWE-bench Verified with
mini-swe-agent, long-doc-qa, and coding agents mixed with short chat. Before
execution, it SHALL record task/document/chat IDs, hashes, selection rules, seeds,
tuning/held-out roles, and execution budgets. Defaults are 4 SWE tuning tasks and
8 held-out tasks, expandable to at most 20 distinct tasks; at most 32 documents;
and 128 short chats in the mixed workload. This change MUST NOT automatically
launch full-dataset runs.

#### Scenario: A small benchmark is prepared

- **WHEN** an operator creates a local experiment protocol
- **THEN** its manifest fixes all three input sets, finite scale candidates, and
  tuning/held-out groups, and a dry run exposes the expanded matrix without
  starting services

#### Scenario: An input exceeds context limits

- **WHEN** complete input plus output budget exceeds the model context limit
- **THEN** the system reports the input and budget, revises the protocol uniformly
  or retains the failure, and does not silently truncate, drop history, or assign
  different context limits by policy

### Requirement: Real coding-agent and tool evidence

The system SHALL use an explicitly versioned mini-swe-agent with this branch's
model to execute real tasks and tools. It SHALL preserve model messages, commands,
exit codes, tool results, timestamps, patches, and terminal task outcomes. It
SHALL reuse available local tasks and container resources under comparable
resource limits. Token IDs, routing-capture performance, and historical model
speeds from the adjacent project MUST NOT replace measurements on this branch.

#### Scenario: A tool or task fails

- **WHEN** a command fails, a step/time budget is exhausted, model output has an
  invalid format, or a task remains unresolved
- **THEN** the task stays in the plan and results, task outcomes remain distinct
  from infrastructure errors, and failed tasks remain in the denominator

#### Scenario: A submitted patch is evaluated

- **WHEN** a real small-sample run produces a patch
- **THEN** the system records actual test results on the frozen scoring subset,
  reporting submitted, resolved, failed, and unscored states separately

### Requirement: Long-document reuse with verifiable content

long-doc-qa SHALL use local natural text with provenance and hashes, four fixed
questions per document, and at least one verifiable answer per document. Questions
for the same document SHALL share identical document-prefix token IDs. The system
SHALL support interleaved independent documents, fixed ordering seeds, and bounded
concurrency, and record actual reuse distance. Synthetic text and artificial waits
MUST be labeled separately.

#### Scenario: Questions revisit an evicted document

- **WHEN** a question revisits a document after interleaving other documents
- **THEN** the report distinguishes GPU hits, cold-tier loads, and recomputation,
  while retaining answer correctness, waiting, and actual completion times

### Requirement: Mixed-workload arrival and fairness evidence

The mixed workload SHALL run coding sessions together with independent short
chats. Chats SHALL follow a fixed open-loop arrival table, and their latency
SHALL start at scheduled readiness, including client-capacity waiting. The system
SHALL report offered/completed/failed counts, waiting, and tails by class, and
continue processing and draining every planned request after arrivals stop.

#### Scenario: Chat requests wait behind coding agents

- **WHEN** coding resource usage prevents chats from being sent or admitted
  immediately
- **THEN** waiting contributes to chat end-to-end latency and SLO statistics, and
  every planned request remains in completion and failure reporting

### Requirement: Fixed-work replay preserves the full workload

The system SHALL convert validated real traces into fixed complete request
inputs, output token counts, dependencies, initial arrivals, tool intervals, and
session/class metadata. Each replay SHALL execute the real model while retaining
recorded later inputs and without rerunning tools. Model, tokenizer, template,
weight version, and work SHALL have comparable fingerprints. Replay outputs MUST
NOT be reported as new task scores.

#### Scenario: Fresh generation differs from the recording

- **WHEN** a replay generates text different from the recording
- **THEN** later requests retain frozen complete inputs; changes to output token
  counts, request lists, or dependencies disqualify that run from equal-work
  throughput comparisons

#### Scenario: A recorded request cannot reach model execution

- **WHEN** a trace contains a context rejection or another pre-execution error
- **THEN** the original error is retained separately without counting as
  successful inference, and any executable replay subset is frozen separately
  before comparisons with its exclusions disclosed

### Requirement: Baseline-calibrated pressure and cache-fit controls

The system SHALL fix the deployment GPU KV budget and increase independent
working sets or concurrency on Native using preregistered finite candidates,
selecting the smallest valid pressure point and a cache-fit point. Pressure SHALL
have working-set/effective-capacity evidence plus cache-turnover recomputation or
loading, capacity deferral, preemption, or sustained queueing. A single occupancy
peak is insufficient. Selection MUST NOT depend on cooperation speedup rankings.

#### Scenario: The bounded workload still fits

- **WHEN** the local scale cap is reached without establishing required pressure
- **THEN** the report records unestablished pressure and limiting conditions,
  pressure acceptance remains incomplete, and a small fixed KV-byte override
  cannot substitute for primary deployment pressure

### Requirement: Comparable arms and complete timing

The system SHALL provide N Native APC, B enablement without hints, D explicit
directives without policies, Q quota-only, and C full cooperation. Each of the
three pressure points SHALL run N/D/Q/C at least three times; cache-fit points
SHALL run N/B/Q/C at least three times. Arms SHALL interleave run order, restart
independently, verify consistent initial cache states, and record all settings.
Ordinary timing SHALL include actual workload preparation, control, retries,
task cleanup, and device-transfer draining.

#### Scenario: A transfer continues after the last response

- **WHEN** model responses finish while operations remain in flight or records
  still require cleanup
- **THEN** timing continues through completion of actual device access and related
  terminal results/cleanup, and timeouts retain failure outcomes

#### Scenario: A connector adds host cache capacity

- **WHEN** D/C use cold-tier capacity absent from Native
- **THEN** the report discloses extra CPU/Store resources and includes D/C with
  identical cold-tier resources and Q/C with identical quota parameters

### Requirement: Local resources provenance and measurement isolation

Primary performance experiments SHALL use local Qwen3.8-27B BF16, TP=2, GPUs 0/3,
APC, PCP=DCP=1, and `gpu_memory_utilization=0.5`, recording actual cache groups and
KV capacity. Experiments SHALL fix CPU/container/Store resources and record source,
dependencies, input hashes, and process ownership. Profiling, regressions, and
ordinary timing SHALL occupy separate windows. Every Python and dependency
operation SHALL follow the uv/venv convention.

#### Scenario: Measurement conditions change during a run

- **WHEN** source changes, extra GPU compute processes, overlapping
  profiling/regressions, or mismatched work are detected
- **THEN** the original run and reason are retained, the comparison is marked
  invalid and repeated after conditions recover, and fastest samples cannot be
  selected silently
