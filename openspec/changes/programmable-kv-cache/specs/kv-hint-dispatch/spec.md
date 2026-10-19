# KV hint dispatch

## ADDED Requirements

### Requirement: Preserve two interpretation owners

The implementation SHALL support native vLLM interpretation for its Mooncake control path and opaque forwarding for the existing KVCR path.
Both SHALL use the merged KvHintsEnvelope/KvHintAction fields.
Native interpretation SHALL NOT be a prerequisite for forwarding a backend-owned payload.

#### Scenario: Native Mooncake operation

- **WHEN** an envelope selects the native path and requests a supported Demote
- **THEN** vLLM interprets its meaning and invokes approved Mooncake storage/transfer operations rather than asking Mooncake to interpret the envelope

#### Scenario: Backend-owned payload

- **WHEN** an envelope selects KVCR interpretation
- **THEN** the integration preserves its IDs, versions, action order, and JSON payload values when forwarding without applying native action-specific validation

### Requirement: One path per envelope

The first delivery SHALL select one interpretation path for each entire envelope using invocation/deployment context without adding a consumer field to the merged records.
It SHALL NOT split or broadcast an envelope across owners.
Unsupported selected-consumer actions SHALL NOT be silently reinterpreted by another consumer.

#### Scenario: Native consumer does not support an action

- **WHEN** the selected native consumer cannot execute an action
- **THEN** it reports the unsupported result without automatically forwarding that action to KVCR

#### Scenario: Caller requires both consumers

- **WHEN** an application intends work on both interpretation paths
- **THEN** it submits separate envelopes to the selected paths; a single envelope is not executed by both

### Requirement: Preserve existing KVCR lifecycle

Changes to native hint consumption SHALL preserve the existing KVCR request-scoped submit_hint, query/deliver, source-reference, and discard_hint behavior.
Native-consumed envelopes SHALL NOT also reach the legacy forwarding hook unintentionally.

#### Scenario: Request cleanup after backend forwarding

- **WHEN** an existing KVCR request completes
- **THEN** the integration performs its existing hint cleanup and does not leak its request-scoped state

### Requirement: Shared GPU resource authority

Any GPU work requested by either interpretation path SHALL respect engine-controlled capacity, references, buffer lifetime, and readiness.
Backend interpretation SHALL NOT grant direct allocator authority.
KVCR integration SHALL NOT be required to reparse its payload through the native interpreter to use engine resource mechanisms.

#### Scenario: Backend needs engine GPU buffers

- **WHEN** a supported backend integration requests GPU resource operations
- **THEN** the engine checks and controls those resources independently of where the payload was interpreted

### Requirement: Do not overstate forwarding or future support

Forwarding receipt SHALL NOT be reported as completed backend execution or GPU readiness.
Existing KVCR forwarding SHALL NOT be advertised as support for arbitrary future KVCR G1 directives or request-free actions.

#### Scenario: Backend accepts a hint

- **WHEN** submit_hint returns before the related transfer has completed
- **THEN** the integration does not claim that destination GPU KV is ready

### Requirement: RFC primitive dispatch and lifecycle

The native consumer SHALL register COPY_TO, MOVE_TO and SET_PRIORITY at version 1.0 alongside the legacy profile in one dispatcher.
Attached GPU-source movement and priority SHALL be validated at ingress and executed before successful request reference release.
Abort/error SHALL discard queued actions; preemption SHALL preserve them.
Cold-source COPY SHALL join the existing request load coordination.

#### Scenario: Successful completion starts a move

- **WHEN** a request with a valid MOVE_TO completes successfully
- **THEN** the engine resolves its cached source and acquires transfer references before releasing request references, and reports storage and reclamation separately

#### Scenario: Backend-selected public-looking action

- **WHEN** an entire envelope selects the backend consumer
- **THEN** the native primitive dispatcher does not interpret any of its actions
