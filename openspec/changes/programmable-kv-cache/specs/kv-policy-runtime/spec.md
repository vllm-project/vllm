# KV policy runtime requirements

## ADDED Requirements

### Requirement: Startup-configured independent plugins

Policy, composition, and backend-adaptation implementations SHALL be independently configurable at startup.
Plugin replacement SHALL require restart; runtime hot replacement is outside this change.

#### Scenario: Alternate local policy

- **WHEN** an operator configures a compatible alternate policy
- **THEN** the engine loads it through the policy interface without requiring a replacement scheduler implementation

### Requirement: Mechanism-only default

Without a registered local policy, the engine SHALL resolve, validate, reserve, execute, and report explicit directives.
Lifecycle signals SHALL have no operational side effects and SHALL be reported unconsumed.

#### Scenario: Unconsumed tool-wait signal

- **WHEN** a tool-wait signal arrives with no policy registered to consume it
- **THEN** it does not implicitly retain, demote, prefetch, or allocate KV

### Requirement: Shared observations and proposal-only policies

Policies SHALL receive common observations that distinguish physical occupancy, reclaimable capacity, in-flight reservations, demand estimates, and policy commitments.
Policies SHALL propose work rather than mutate allocators, start transfers, reorder queues, or select preemption victims directly.

#### Scenario: Two policies observe the same free capacity

- **WHEN** two policies propose competing allocations against one view
- **THEN** neither proposal independently reserves that capacity and both proceed through common composition and engine approval

### Requirement: Preserve explicit command meaning

Composition SHALL preserve explicit command semantics, coalesce compatible physical work, and retain independent caller correlation and retention lifetimes.
Local policies SHALL NOT rewrite explicit directives.

#### Scenario: Policy prefers a different destination

- **WHEN** a local policy prefers CPU residency while an explicit command requests GPU prefetch
- **THEN** the command is evaluated with its requested destination rather than silently rewritten

### Requirement: Disabled-path compatibility

When the new feature is disabled, the engine SHALL retain existing serving behavior and SHALL NOT maintain optional session/policy state solely for this feature.
Validation SHALL measure disabled-path overhead.

#### Scenario: Unconfigured deployment

- **WHEN** serving starts with the existing configuration and no programmable-cache feature enabled
- **THEN** ordinary requests use existing cache and scheduling semantics

### Requirement: Optional scheduler guidance

The first delivery SHALL include optional quota/admission and request-score examples.
The scheduler SHALL apply the guidance and retain its existing preemption algorithm.
Logical quota shares SHALL NOT be counted as additional physical capacity.

#### Scenario: Quota guidance competes with a transfer

- **WHEN** admission and prefetch both require the same remaining physical capacity
- **THEN** engine accounting prevents double commitment even when a policy's logical quota has unused allowance

#### Scenario: Policy proposes a request score

- **WHEN** the optional score example produces guidance
- **THEN** the scheduler applies it through the agreed extension without giving the KV policy direct queue or victim-selection access

### Requirement: Observe completion without resource authority

The runtime SHALL allow optional policies to observe immutable outcomes and elapsed time for their own proposals.
Such feedback SHALL arrive before an acknowledged result is discarded.
Stored-target hints SHALL remain bounded and shall not assert current backend residency.

#### Scenario: Partial store followed by a predicted resume

- **WHEN** movement stores only part of the proposed scope
- **THEN** the reference movement policy can prepare the confirmed stored subset and still checks actual backend outcomes

#### Scenario: Actual resume overtakes a running prediction

- **WHEN** both proposals request the same compatible COPY
- **THEN** the callers share in-flight work without cancelling the original publication
