# Requirements for Experiment-Driven KV Optimization

## ADDED Requirements

### Requirement: Preserve the programmable-cache authority boundary

Optimization SHALL reuse `KvHintsEnvelope`, existing explicit primitives, and
correlated results, keeping the mechanism-only default separate from optional
local policies. Policies SHALL propose through shared resource approval; the
scheduler SHALL retain queue/preemption ownership and existing adapters SHALL
execute movement. Optimization MUST preserve opaque KVCR forwarding, valid
references, shared-capacity deduplication, publication after all ranks/cache groups
complete, and failure recomputation. It MUST NOT add public Pin, replace the
scheduler, change the Mooncake project, or alter model context semantics.

#### Scenario: An explicit operation conflicts with policy advice

- **WHEN** an explicit application directive conflicts with local advice or
  resources are insufficient
- **THEN** the engine executes, defers, rejects, or reports actual partial scope
  under the existing contract, and policies cannot silently rewrite the command

#### Scenario: A multi-group preparation is incomplete

- **WHEN** any group or rank remains incomplete, or target checkpoints cannot
  form a compatible prefix
- **THEN** the system does not claim complete reuse availability, protects
  in-flight buffers, and coordinates waiting, cancellation, or recomputation
  according to operation outcomes

### Requirement: Policies use only available application information

Normal optimization SHALL use only tool types, lifecycle facts, historical
completed costs, and current engine state available at decision time. Predictions
SHALL record their inputs and availability times. Recorded future tool durations,
output lengths, admission times, and completion times MUST NOT become future hints
for normal policies. Oracle results SHALL be separate.

#### Scenario: Replay contains the actual future pause

- **WHEN** the replay driver knows the actual duration of a tool batch
- **THEN** it uses that duration only to control simulated readiness, while normal
  policies receive predictions based on past samples; only explicitly labeled
  oracle diagnostics can use the future value

### Requirement: Attribute remaining bottlenecks before core changes

Before implementing an optimization, the experiment process SHALL retain evidence
of the limiting mechanism, a falsifiable hypothesis, expected effects, and minimal
failing behavior. Analysis SHALL cover tools, waiting after readiness, control,
prefill/decode, non-overlapped transfers, complete-group preparation, and cleanup,
separating cumulative durations from wall-clock intervals. Each round SHALL
focus on one main mechanism. Parameter search or increased pressure alone cannot
establish completion of core optimization.

#### Scenario: Copies complete but requests recompute

- **WHEN** completed preparation does not translate into actual cache hits
- **THEN** analysis correlates publication, admission, coverage gaps, and eviction
  evidence by content/cache group/operation/session, retains unknown causes, and
  uses the resulting evidence to select a change

#### Scenario: Saving does not free additional allocatable blocks

- **WHEN** saved content was already unreferenced reclaimable cache
- **THEN** the system records physical references, reclaimability, and mapping
  changes separately and cannot claim more allocatable capacity from saved bytes
  alone

### Requirement: Demonstrate the contribution of application cooperation

Experiments SHALL compare explicit directives, quota-only, and full cooperation.
D/C SHALL share available application information, the predictor, and cold-tier
resources; Q/C SHALL fix identical quota parameters. Diagnostics SHALL include
individual policies, removal of prediction or admission cooperation, simple
prefetch at pause, and targeted prediction-error, wrong-target/cancellation, and
control-delay cases. These conditions and oracle SHALL be reported separately
from primary performance results.

#### Scenario: Full cooperation only matches quota

- **WHEN** quota explains all full-cooperation gains over Native
- **THEN** the report states that incremental application-information/movement
  benefits remain unestablished and continues diagnosing cost and reuse, without
  using the Native difference as evidence of composition benefits

### Requirement: Bounded diagnostics and actionable transfer accounting

Diagnostics SHALL record actual saved/loaded ranges and bytes, complete available
coverage, duplicate/unconsumed operations, actual hits, and admission intervals,
retaining correlation through engine epochs and operation/transfer IDs.
Diagnostics SHALL be bounded and support disabling, avoiding full scans at each
scheduler step. The system MUST distinguish ordinary GPU hits, loaded content
subsequently used, and hits without attributable provenance.

#### Scenario: Existing GPU content already satisfies a request

- **WHEN** a request hits existing GPU content without evidence linking it to this
  round's load
- **THEN** that hit is not fully counted as movement contribution, and bytes,
  operations, and reuse retain their distinct accounting definitions

### Requirement: Verify core gains on preserved samples and frozen versions

The system SHALL preserve the source and configuration of C0, full cooperation
before this round's core changes. Core candidates SHALL be selected using tuning
samples and, after freezing, compared as C0/C on held-out samples with equal work.
If held-out results guide further tuning, the system SHALL reclassify that set as
tuning data and allocate new preregistered held-out samples. Shared execution-path
changes SHALL trigger affected-baseline reruns or explicit equivalence checks.

#### Scenario: A candidate improves tuning tasks only

- **WHEN** a candidate improves throughput on tuning tasks without net gains on
  held-out tasks
- **THEN** optimization acceptance remains incomplete, regression results and
  causes are retained, and tuning results alone cannot establish core gains

### Requirement: Separate validity correctness and performance acceptance

The system SHALL report experiment validity, contract checks, and performance
acceptance separately. At one or more held-out pressure points, performance
acceptance SHALL require median fixed-work throughput gains for C of at least
25% over N, 10% over Q, and 5% over C0, each with at least three ordinary-timing
repetitions and gains in the same direction in at least two repetitions. All valid
runs and other scenarios SHALL be reported together. Performance acceptance MUST
remain incomplete while these targets are unmet.

#### Scenario: Existing cooperation already beats Native

- **WHEN** C0 already improves on Native but this round's change misses the
  incremental target against C0
- **THEN** the report can establish existing behavior on the new workload while
  this round's core-optimization target remains incomplete

#### Scenario: A fast run changes outputs or omits work

- **WHEN** gains come from fewer requests/output tokens, hidden failures,
  incomplete draining, or changed comparison conditions
- **THEN** that run cannot satisfy performance targets and retains its
  invalidation reason

### Requirement: Protect low-contention cost and short-request latency

Performance acceptance SHALL also require cache-fit median elapsed-time
regression for C against N within 5%, and mixed short-chat P95
readiness-to-completion latency at most 1.10 times N in that same workload. The
system SHALL process identical planned requests, report completions, failures,
maximum waiting, and throughput within preregistered SLOs by class, and introduce
no incorrect reuse, early publication, resource leaks, or errors outside task
outcomes themselves.

#### Scenario: Aggregate throughput rises while chat tails regress

- **WHEN** fixed-work throughput improves but short-chat tail latency exceeds its
  threshold
- **THEN** the report records waiting shifts and class metrics, performance
  acceptance remains incomplete, and optimization continues to address
  interference from scheduling and capacity held by preparation

### Requirement: Reviewable optimization evidence and regression delivery

Delivery SHALL retain experiment settings, input/source hashes, raw request/task
data, commands, analysis, and positive/negative conclusions. Core changes SHALL
add minimal behavioral regressions in existing nearby suites and run affected
complete tests, necessary real GPU/model checks, pre-commit, mypy, and strict
OpenSpec validation. New evidence SHALL link to the original change without
replacing unexecuted acceptance. This change SHALL NOT create a separate upstream
PR.

#### Scenario: A candidate has no confirmed net benefit

- **WHEN** a change has only a single-run advantage, unreproducible gains, or
  unresolved regressions
- **THEN** it is not marked as validated optimization; unsupported final code
  options are withdrawn while raw experiments and the next concrete limitation
  remain documented
