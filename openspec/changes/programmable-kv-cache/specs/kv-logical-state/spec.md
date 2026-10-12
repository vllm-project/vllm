# Logical KV state

## ADDED Requirements

### Requirement: Independent logical identity and physical residency

The engine SHALL distinguish logical content, session/continuation associations, residency, retention declarations, and active references.
Session identity SHALL NOT replace the existing content hash or establish cache compatibility.

#### Scenario: Shared prefix cache hit

- **WHEN** a second tracked session hits content first computed by another session
- **THEN** its logical association is recorded without creating exclusive ownership or duplicating the physical copy

#### Scenario: GPU copy evicted

- **WHEN** one GPU copy disappears
- **THEN** the engine updates that residency without inferring task completion or deleting other known copies

### Requirement: Logical target resolution

Control SHALL support logical hashes, tracked sessions, explicit continuations and token ranges, and active request ranges within advertised capabilities.
Resolution SHALL report actual group-specific coverage and missing or unsupported portions.

#### Scenario: Historical branch selection

- **WHEN** an action selects an explicit continuation after another branch advances
- **THEN** resolution uses the selected continuation's content rather than an implicit latest branch

#### Scenario: Unsupported reusable boundary

- **WHEN** a requested token range cannot form reusable cache state for a configured group
- **THEN** the result reports its coverage limitation and does not claim full reuse

### Requirement: Append-only continuation identity

A live continuation SHALL preserve its previously associated content while it grows.
Rewriting or truncating existing content SHALL require a new continuation ID; the session ID may remain unchanged.
Adding previously unobserved content or compatible shorter-prefix aliases SHALL NOT be treated as rewriting.
Logical association alone SHALL NOT establish token ancestry or cache compatibility.

#### Scenario: Rewrite rejected with a current revision

- **WHEN** an update has the current revision but replaces or removes known content
- **THEN** it fails without changing associations, revision, or association events

#### Scenario: New branch does not inherit retention implicitly

- **WHEN** an application creates another continuation in the same session
- **THEN** an existing follow-growth Retain remains on its named continuation with its original expiry, and existing snapshot operations keep their original content

#### Scenario: Tail growth preserves an observed shorter alias

- **WHEN** a continuation adds a longer compatible tail and retains its previously observed shorter alias
- **THEN** both associations remain queryable without losing the earlier content

### Requirement: Independent retention lifetimes

Retain SHALL bias eviction priority without guaranteeing residency.
Shared content SHALL use the maximum valid declared priority while preserving each declaration's own lifetime.
A cache hit SHALL NOT implicitly renew a declaration.
Release SHALL remove only the addressed declaration.

#### Scenario: High priority expires before a lower priority

- **WHEN** A's high-priority declaration expires while B's lower-priority declaration remains valid
- **THEN** B's declaration continues at its own priority and expiry

### Requirement: Explicit task lifecycle cleanup

Request completion SHALL NOT imply task completion.
Explicit task-tree completion SHALL release the relevant logical associations and declarations while preserving unrelated shared state and active references.

#### Scenario: Parent finishes while another task shares an ancestor

- **WHEN** the parent task and its descendants are completed
- **THEN** their state is cleaned up without releasing the other task's association or memory still referenced by requests/transfers

### Requirement: Explicit state management independent of local policies

Establishing/updating session and continuation associations and explicitly ending a task tree SHALL be supported management commands without requiring an opt-in local policy.
Tool-wait, resume, and importance signals SHALL remain separate policy inputs.
Association alone SHALL NOT allocate or guarantee resident KV.

#### Scenario: Session-addressed control in default mode

- **WHEN** an application explicitly establishes a valid tracked association with no local policies configured
- **THEN** supported session-addressed commands can resolve that association without interpreting tool-wait signals or implicitly protecting the content

### Requirement: Incremental association events and state queries

The engine SHALL publish incremental association changes and expose queryable association state.
Each association event SHALL NOT require a complete session set.
Association changes SHALL remain distinguishable from physical residency changes.
Existing full-report behavior SHALL retain its current-request prefix-reporting meaning.

#### Scenario: Shared prefix gains a second session

- **WHEN** session B establishes an association by hitting content already associated with A
- **THEN** a delta reports B's new association and queries can discover both associations without claiming a new physical copy

#### Scenario: Recover associations after an event gap

- **WHEN** a consumer cannot recover all missing association events
- **THEN** it can query supported association state and does not treat current-request full reporting as a complete engine snapshot

### Requirement: Shared index for engine consumers

Session-addressable execution SHALL use an engine-local logical index available to scheduler observations and operation resolution.
Policies and adapters SHALL NOT each require a separate authoritative session-to-physical-block ownership map.

#### Scenario: Retention and movement target the same continuation

- **WHEN** two consumers resolve the same tracked continuation against the same logical state
- **THEN** they use the common content associations while independently revalidating physical feasibility at approval

### Requirement: Bounded index behavior is observable

Logical-state tracking SHALL have documented limits and reclamation behavior.
Queries and resolution SHALL distinguish unknown or pruned associations from known empty targets and actual missing residency.

#### Scenario: Historical association is no longer tracked

- **WHEN** a controller addresses a continuation outside the index's retained history
- **THEN** the engine reports the unavailable association instead of resolving it to a reused physical block or a different branch
