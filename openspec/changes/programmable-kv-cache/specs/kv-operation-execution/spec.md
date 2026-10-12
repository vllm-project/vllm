# KV operation execution requirements

## ADDED Requirements

### Requirement: First adapter coverage

The first delivery SHALL support the new operation contract through the general OffloadingConnector/CPUOffloadingManager and Mooncake Store adapters.
SimpleCPUOffloadConnector SHALL NOT advertise new programmable-control operations it does not implement.

#### Scenario: SimpleCPU control request

- **WHEN** independent control requests an operation that the configured SimpleCPU adapter does not implement
- **THEN** capability discovery and the operation result consistently report that lack of support

### Requirement: Engine-owned approval and resource accounting

The engine SHALL revalidate current content, references, and resources before execution.
Physical allocations and destination reservations SHALL be charged once.
Cancellation SHALL NOT release resources still accessed by workers.

#### Scenario: Capacity changed after observation

- **WHEN** an operation's earlier policy view shows capacity that is no longer available
- **THEN** approval defers, clips within permitted semantics, or rejects rather than over-allocating

#### Scenario: Shared transfer loses one consumer

- **WHEN** one caller cancels while another still requires the same transfer
- **THEN** the remaining caller and in-flight memory remain valid

### Requirement: Local retention and eviction operations

The engine SHALL implement Retain, Release, and Evict with the declared logical scope.
Evict SHALL respect active request and transfer references.
Retain SHALL remain an eviction bias under memory pressure.

#### Scenario: Eviction conflicts with an active request

- **WHEN** an explicit eviction targets referenced KV
- **THEN** the engine reports the blocked or partial scope without invalidating the request's memory

### Requirement: Demotion reports cold copy and GPU reclamation separately

Demote SHALL establish a recoverable cold copy before invalidating a reclaimable GPU copy.
Completion reporting SHALL distinguish cold storage success from actual GPU reclamation.

#### Scenario: Store succeeds while a request still references the source

- **WHEN** the cold copy completes but source references remain
- **THEN** the engine reports cold availability without claiming GPU space was freed

### Requirement: Request-free prefetch and resume coordination

Prefetch SHALL reserve missing destination capacity, perform a real transfer, and register reusable content before reporting availability.
It SHALL progress without live inference.
A compatible resume SHALL coordinate with an in-flight prefetch rather than duplicate its load.

#### Scenario: Resume races prefetch

- **WHEN** a request needs content already being prefetched
- **THEN** compatible work is shared and the request proceeds only after its required content is valid or the normal failure/fallback path is selected

### Requirement: Cross-engine sharing through existing Mooncake APIs

The change SHALL include a vLLM-side Mooncake Store adapter and a real cross-engine example using unchanged backend APIs.
Share SHALL preserve the source unless a separate operation authorizes its removal.
Store availability SHALL NOT be reported as destination GPU availability.

#### Scenario: Resume on engine B after production on engine A

- **WHEN** A has made a compatible prefix available in the Store and B receives its prefetch directive before inference
- **THEN** B reserves and loads the missing KV, reports availability after registration, and the subsequent request demonstrates actual prefix reuse

### Requirement: Reset and stale completion handling

Operation execution SHALL preserve correctness across cancellation, cache reset, and engine shutdown.
A completion from invalidated work SHALL NOT republish stale KV as usable.

#### Scenario: Reset overlaps transfer

- **WHEN** cache reset invalidates the generation of an in-flight operation
- **THEN** its eventual completion cannot restore invalidated content into the reusable cache
