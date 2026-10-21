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

### Requirement: Account for application pause duration

The reference movement policy SHALL consider an optional pause estimate on the
initial tool-wait signal before proposing a store. It SHALL avoid speculative
store/load round trips whose estimated duration exceeds the application pause.
With pause-aware movement enabled, a configured pressure threshold SHALL also
apply to current referenced GPU blocks. Projected future demand and a long pause
estimate alone SHALL NOT authorize MOVE below that threshold. This check SHALL
avoid building a request-demand observation when movement is already ineligible.
The explicitly selected pressure-only diagnostic mode retains its projected-demand behavior.
Skipped movement state SHALL be bounded and released with its logical owners.
Explicit movement commands SHALL keep their original meaning.

#### Scenario: Short tool under cache pressure

- **WHEN** a predicted tool wait is shorter than the configured minimum pause
- **THEN** pressure alone does not cause a store followed by a speculative load

#### Scenario: Overestimated pause while the cache fits

- **WHEN** a long predicted pause and projected request growth exceed the pressure threshold but current referenced occupancy remains below it
- **THEN** automatic movement preserves the resident content without store/load work or a request-demand scan

### Requirement: Revalidate admission headroom

Quota guidance that reserves running growth SHALL check current physical free
capacity before each admission, including growth of requests admitted since the
last policy observation. The finite waiting fallback SHALL remain available.

#### Scenario: Two requests fit individually but not together

- **WHEN** two candidates each fit one observation's available capacity
- **THEN** admitting the first does not permit the second to spend the same
  capacity reserved for the first request's growth

### Requirement: Coordinate optional prefetch with admission

Admission-coupled policy COPY proposals SHALL wait under queued foreground
demand until the scheduler selects a session needing their content. The request
SHALL wait for terminal publication or fall back after a failed preparation.
Preparation SHALL retain its original bounded scope and cancellation semantics.

#### Scenario: Forecast arrives well before admission

- **WHEN** a session predicts readiness but other requests still occupy the queue
- **THEN** its admission-coupled preparation does not occupy GPU capacity early
  merely because the tool is ready

#### Scenario: Preparation fails

- **WHEN** the selected request's preparation terminates with missing content
- **THEN** the request proceeds through ordinary lookup and recomputation
