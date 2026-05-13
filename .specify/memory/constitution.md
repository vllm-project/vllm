# vLLM Constitution

## Core Principles

### I. Correctness First

Inference output must match the model's intended behavior. No optimization may change the semantics of generated tokens, KV cache contents, or request state. Regressions in correctness (including determinism, stop conditions, and structured output) are non-negotiable and must be caught by tests before merge.

### II. Scheduler as Single Source of Truth

Per-request state that affects the next scheduling step lives in the scheduler (on the Request object). Workers are stateless across steps with respect to request content: they receive inputs via SchedulerOutput and return outputs (e.g., sampled tokens, draft tokens, or dLLM step results) that the scheduler applies to the Request. This keeps preemption, multi-worker, and resumption correct without duplicating state.

### III. Test-Driven Development

When adding or refactoring behavior, write tests first where feasible. Tests must be deterministic and isolated. Existing tests are regression guards: any change that breaks them must either fix the code or update the test with explicit justification. New features must include unit tests for new contracts and integration or e2e tests for critical paths.

### IV. One Step, One Forward

Each scheduler step corresponds to at most one forward pass of the model per batch. Scheduling is expressed as "how many tokens to process per request this step." Features that need multiple tokens per step (e.g., speculative decoding, dLLM blocks) still map to one forward; the step output may contain variable-length results (e.g., committed tokens, next-step input) that the scheduler applies before the next schedule.

### V. Backward Compatibility and Opt-In Behavior

New inference modes (e.g., dLLM, new spec decode methods) must not change behavior for existing models or configs. Enable new behavior via explicit config or request-level flags. Defaults must preserve current behavior so existing deployments and tests do not regress.

### VI. Performance and Batching

Continuous batching and existing optimizations (KV cache, prefix caching, chunked prefill, attention backends) must continue to apply unless a feature explicitly opts out with justification. New paths should integrate with the same batching and scheduling budget so throughput and latency characteristics remain predictable.

## Additional Constraints

- **Prefix caching**: When a feature changes how many positions are "committed" per step (e.g., dLLM commits 0..LOOKAHEAD_SIZE), prefix cache validity must be defined and documented (e.g., cacheable only up to committed length).
- **Attention and masks**: Causal attention is the default. Custom attention masks (e.g., block-wise for dLLM) require a clear contract and backend support; they must not break existing causal/sliding-window paths.
- **Serialization and multi-process**: ModelRunnerOutput and scheduler outputs may cross process boundaries; prefer lists and simple types for payloads that are sent between engine and worker.

## Development Workflow

- Follow the existing v1 layout: core (scheduler, KV cache), worker (model runner, input batch), engine (step loop, post_step).
- When extending outputs (e.g., adding DllmStepOutput), ensure all constructors and make_empty() include new fields with safe defaults so existing call sites do not break.
- Document the contract for new request/scheduler/worker contracts in code (docstrings) or in design docs so future changes preserve the paradigm.

## Governance

This constitution guides all feature work in vLLM. PRs that introduce new inference modes or change scheduler/worker contracts must verify compliance with these principles. Amendments require documentation and should be reflected in this file.

**Version**: 1.0.0 | **Ratified**: 2025-03-01 | **Last Amended**: 2025-03-01
