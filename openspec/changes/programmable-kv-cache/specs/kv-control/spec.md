# KV control requirements

## ADDED Requirements

### Requirement: Application control through the merged hint contract

The implementation SHALL use the already-merged KvHintsEnvelope/KvHintAction records for applications to control supported cache behavior.
It SHALL preserve their wire fields and SHALL NOT require a custom hint-handler registration API or a replacement envelope.
Configurable engine policies SHALL NOT be prerequisites for executing explicit supported commands.

#### Scenario: Application changes retention and placement

- **WHEN** an application submits supported Retain, Demote, or Prefetch actions using the merged envelope with no local policies configured
- **THEN** the engine validates, resolves, approves or refuses, executes, and reports each action through the common control path

#### Scenario: Existing envelope encoding

- **WHEN** a caller sends protocol_version, message_id, and actions containing action_id, action_type, action_version, and payload
- **THEN** the engine consumes those existing records without requiring a new custom protocol or plugin registration

### Requirement: Common envelope across request lifetimes

The native control path SHALL accept the existing versioned KvHintsEnvelope with an inference request or independently of inference and SHALL correlate action results with message_id and action_id.
Backend-interpreted forwarding SHALL advertise and preserve the selected backend's supported lifecycle rather than imply support for every native control operation.

#### Scenario: Control during a tool pause

- **WHEN** a supported explicit operation arrives with no live inference request
- **THEN** the engine processes it and exposes its result without creating a synthetic inference request

### Requirement: Python and HTTP control profile

The first delivery SHALL expose Python and HTTP APIs for independent submission, capability discovery, and operation queries, using the common EngineCore utility path internally.
Independent control SHALL identify the intended engine.
A new gRPC control service is outside the first delivery.

#### Scenario: HTTP operation followed by Python query

- **WHEN** a caller submits through HTTP and queries the same operation through Python on the same engine
- **THEN** both surfaces observe the same correlated operation state

### Requirement: Truthful operation results

The engine SHALL distinguish receipt, resource approval, effective completion, partial scope, deferral, unsupported capability, missing targets, failure, and intentional cancellation.
Queries and events SHALL derive from the same operation record.

#### Scenario: Transfer approved but unfinished

- **WHEN** destination capacity is reserved but a transfer has not completed
- **THEN** the result does not claim reusable destination KV

#### Scenario: Backend error

- **WHEN** a transfer fails because of a backend error
- **THEN** its result reports failure rather than intentional cancellation or completion

### Requirement: Capability and version validation

Transport SHALL validate the envelope contract.
The selected interpretation owner SHALL validate action versions and action-specific payloads before semantic execution; vLLM SHALL perform that validation for native actions without applying native schemas to KVCR-owned payloads.
Capability reporting SHALL identify the selected path's supported targets, tiers, operations, and limitations.
The native public action set SHALL NOT include Pin; existing backend/internal reference callbacks are not public native Pin actions.

#### Scenario: Unsupported remote operation

- **WHEN** an action requests a guarantee the configured adapter cannot provide
- **THEN** the engine reports unsupported without silently substituting a weaker guarantee

### Requirement: Bounded correlation and recovery

The implementation SHALL bound control records, handle duplicate submissions without duplicating effective work within its documented deduplication scope, and identify publisher restarts and unrecoverable event gaps.

#### Scenario: Engine restart

- **WHEN** an engine restarts and loses local runtime state
- **THEN** consumers can invalidate that engine's old operation/residency observations without inferring deletion of other engines' copies

### Requirement: Explicit result acknowledgement and cancellation

The engine SHALL support acknowledgement of a fully terminal message after callers collect its sibling results.
It SHALL release result capacity while retaining a bounded replay guard until the original expiry.
Cancellation SHALL identify the operation incarnation and preserve buffers still accessed by workers.

#### Scenario: Completed results consume the detailed record limit

- **WHEN** the caller acknowledges a terminal message and replays its envelope
- **THEN** detailed capacity is available for new work and the replay cannot execute again

#### Scenario: Cancellation arrives after message-ID reuse

- **WHEN** cancellation names an earlier operation ID under reused message/action IDs
- **THEN** it fails without cancelling the new operation

### Requirement: Current bounded GPU observations

The control API SHALL report current copies, references, eviction priorities and load progress for supported bounded targets, scoped by namespace and engine epoch.
It SHALL distinguish this observation from a residency guarantee and from a historical operation result.

#### Scenario: Completed copy was subsequently evicted

- **WHEN** the caller queries its current GPU residency
- **THEN** absent copies are reported even though the historical copy result remains completed
