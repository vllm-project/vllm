# Programmable KV examples requirements

## ADDED Requirements

### Requirement: Complete native orchestrator-to-Mooncake path

The primary example SHALL use a runnable lightweight orchestrator to send the merged hint envelope through real Python/HTTP ingress, native semantic interpretation, target/resource approval, and actual Mooncake calls.
It SHALL observe correlated completion and actual subsequent cache reuse.
Dynamo integration is a documented follow-up rather than first-release acceptance.

#### Scenario: Application demotes and prepares a paused prefix

- **WHEN** the orchestrator requests Demote during a pause and Prefetch before resume
- **THEN** the trace connects the original hints to native interpretation, real backend work, cold/destination availability, eligible GPU reclamation, and actual inference reuse

#### Scenario: Optional native policy uses G2

- **WHEN** the orchestrator sends a signal consumed by an enabled native policy
- **THEN** the example shows the policy's resulting operation reaching the same resource-approval and real Mooncake execution path

### Requirement: Demonstrate the control contract beyond policy algorithms

The delivery SHALL include runnable hint/engine workflows for explicit commands, logical state, target resolution, independent operation progress, result feedback, and recovery.
Retention, movement, and quota policy examples alone SHALL NOT satisfy this requirement.

#### Scenario: No local policy configured

- **WHEN** the direct-controller example runs with no local policies
- **THEN** supported explicit cache directives still resolve, execute, and report results across inference request lifetimes

### Requirement: Exercise session and continuation addressing

Examples SHALL demonstrate shared prefixes, explicit branches/ranges, independent declarations, and task cleanup through supported logical selectors without exposing physical block IDs to the application.

#### Scenario: Release one branch of shared content

- **WHEN** a controller releases B's declarations for H+B while C or another task still uses H
- **THEN** the remaining associations/declarations and active references remain valid

### Requirement: Distinguish association from storage events

Examples SHALL demonstrate a new session association on a cache hit and SHALL NOT interpret first-writer BlockStored as an exhaustive ownership record.
Recovery SHALL preserve the distinction between current-request full reporting and a complete engine-state query.

#### Scenario: Session B reuses A's stored prefix

- **WHEN** B hits a prefix without allocating a new physical copy
- **THEN** association observation establishes B's membership without requiring a fictitious new-copy event

### Requirement: Demonstrate asynchronous outcomes and shared work

Examples SHALL cover no-request operation progress, a resume racing a compatible prefetch, deduplicated proposals, finite-capacity competition, cancellation, partial failure, and retry behavior according to the documented control contract.

#### Scenario: Two hint actions request the same content

- **WHEN** both target the same compatible destination prefix concurrently
- **THEN** physical work is charged once while results remain attributable to both callers

### Requirement: Demonstrate cross-engine and restart behavior

The cross-engine workflow SHALL show source preservation, actual destination readiness, and failure/restart fallback.
Successful backend storage alone SHALL NOT be presented as a successful GPU prefetch on another engine.

#### Scenario: Destination restarts before resume

- **WHEN** B restarts after a controller observed a previous availability result
- **THEN** the old result is not treated as proof that B's new instance has a warm GPU cache

### Requirement: Shared example infrastructure

Examples SHALL reuse common hint submission, result polling, and workload helpers.
They SHALL record reproducible configuration and assert observable outcomes rather than only print submitted actions.

#### Scenario: Complete direct-controller workflow

- **WHEN** the example submits a transfer operation
- **THEN** it checks actual terminal/partial outcomes and subsequent reuse before claiming success

### Requirement: Application-directed rollout preparation

The rollout controller example SHALL express its cache decisions through supported actions in the merged hint envelope, without installing new hint types or handler plugins.
It SHALL show the application choosing targets/timing and responding to actual engine results.

#### Scenario: Prepare the next rollout's prefix

- **WHEN** an application knows the next rollout will reuse an explicit prefix
- **THEN** it submits supported retention/prefetch hints, observes the accepted scope and completion, and handles deferral or missing content before resume
