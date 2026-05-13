# Feature Specification: dLLM (Blocked Masked Diffusion LLM) Integration

**Feature Branch**: `001-dllm-integration`  
**Created**: 2025-03-01  
**Status**: Draft  
**Input**: Integrate blocked masked diffusion LLMs (dLLMs) into vLLM—including support for SDAR, LLaDA2.0, LLaDA2.1, Fast-dLLMv2, WeDLM, and similar architectures—so that each diffusion step is treated as one worker iteration and existing continuous batching and optimizations apply automatically.

## Context and Paradigm

dLLMs share a common abstraction:

- **Per step**, the model consumes a **lookahead buffer** of fixed size `LOOKAHEAD_SIZE` (e.g., 32 tokens). Some positions may be `<MASK>`.
- **After the step**, model-specific logic uses logits for all block positions to decide:
  - **Committed output**: 0 to `LOOKAHEAD_SIZE` tokens to append to the sequence and to cache in the KV cache.
  - **Next-step input**: Exactly `LOOKAHEAD_SIZE` tokens for the next forward (including `<MASK>` as needed).

Each architecture defines its own attention mask; prefix caching remains valid only up to committed length (e.g., at most `PREFIX_LEN - LOOKAHEAD_SIZE` in the worst case).

**Design paradigm (aligned with constitution):**

- The **model runner (worker)** returns two values per dLLM request: (1) a variable-length list of committed token IDs (and implies that many KV positions are cached), and (2) a fixed-length list of exactly `LOOKAHEAD_SIZE` input token IDs for the next step.
- The **scheduler** is the single source of truth: it applies committed tokens to the Request (append to output, advance `num_computed_tokens`) and stores the next-step input on the Request. On the next `schedule()`, it sends that next-step input to the worker via SchedulerOutput so the worker can build the input batch without holding durable state.

**Instance and batch scope:** vLLM does not support multiple different models in the same instance or batch. One instance loads one model; every request in a batch is for that model. When a dLLM model is loaded, all requests in that instance are dLLM requests. Prefill and decode (different phases) can still be scheduled in the same step for the same model.

## Clarifications

### Session 2025-03-01

- Q: Does vLLM support mixing dLLM and non-dLLM requests (or different models) in the same instance or batch? → A: No. vLLM uses one model per instance; one batch serves only that model. When a dLLM model is loaded, all requests are dLLM. No mixing of model types in one instance or batch.
- Q: First decode step when prompt+output length < LOOKAHEAD_SIZE: pad with MASK at start (left) or at end (right)? → A: Pad at end (right). In decode, dLLMs are trained to fill masked positions; the block should have known context on the left and positions to decode (MASK) on the right. So the block is [prompt+output, MASK, …, MASK]. Left-padding may still apply for prefill if needed; decode uses right-side masking.
- Q: Where is dLLM mode determined—model-level, request-level, or both? → A: Model-level only. dLLM is inferred from the loaded model (e.g. architecture or model config). All requests in that instance are dLLM; no per-request toggle.
- Q: Who validates worker dLLM output (committed_token_ids and next_step_input_token_ids lengths)? → A: Worker validates. The worker MUST ensure committed_token_ids and next_step_input_token_ids have the correct lengths (0..LOOKAHEAD_SIZE and exactly LOOKAHEAD_SIZE per request) before returning; scheduler trusts the shape.
- Q: Require dLLM-specific metrics or reuse existing? → A: Reuse existing metrics. No new observability requirement; existing step/request metrics (e.g. IterationStats.num_generation_tokens per step) suffice. TPF (tokens per forward) for dLLM equals generation tokens that iteration. Implementation may add dLLM-specific metrics optionally.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run dLLM inference with one request (Priority: P1)

As a user, I want to run inference with a dLLM model (e.g., LLaDA2.0) so that I get generated text where each "step" corresponds to one forward over a lookahead block and the engine advances by 0 to LOOKAHEAD_SIZE tokens per step according to the model's commit rule.

**Why this priority**: This is the minimal viable behavior; without it, no dLLM model can be served.

**Independent Test**: Run a small dLLM model (or stub) for one request: prefill, then several decode steps. Assert that the total output length equals prompt length plus the sum of committed tokens per step, and that each step's forward receives the correct input (first step: last LOOKAHEAD_SIZE of prompt or prompt+output; later steps: previous step's next-step input).

**Acceptance Scenarios**:

1. **Given** a dLLM model and one request with a short prompt, **When** prefill runs then three decode steps with a stub that commits 2 tokens per step, **Then** the request's output has length prompt_len + 6 and the request finishes or continues according to stop conditions.
2. **Given** a dLLM request in decode phase, **When** the worker returns committed_token_ids of length k and next_step_input_token_ids of length LOOKAHEAD_SIZE, **Then** the scheduler appends the k tokens to the request, updates num_computed_tokens by k, and stores the next-step input on the request for the next schedule.
3. **Given** a dLLM request, **When** the next schedule runs, **Then** the worker receives the stored next-step input (exactly LOOKAHEAD_SIZE tokens) in SchedulerOutput and uses it to build the input batch for the forward.

---

### User Story 2 - Continuous batching of dLLM requests (Priority: P2)

As a user, I want multiple dLLM requests to be batched in the same step so that throughput and GPU utilization benefit from existing continuous batching and scheduling.

**Why this priority**: Batching is core to vLLM's value; dLLM must not bypass it.

**Independent Test**: Schedule two or more dLLM requests in the same step; assert that the model runs one forward with all of them and that each request receives the correct committed tokens and next-step input in the output.

**Acceptance Scenarios**:

1. **Given** multiple dLLM requests in the running set (same instance, same dLLM model), **When** schedule() runs, **Then** each request gets num_scheduled_tokens = LOOKAHEAD_SIZE and (if set) its next_step_input_token_ids appear in the output sent to the worker.
2. **Given** a batch that includes both prefill and decode phases for the same dLLM model, **When** the step executes, **Then** the worker runs one forward; prefill requests and dLLM decode requests are batched according to existing scheduling (same model, different phases).

---

### User Story 3 - KV cache and prefix caching correctness (Priority: P2)

As a user, I want KV cache to grow only by the number of committed tokens per step and any unused lookahead slots to be freed so that memory use is correct and prefix caching (where applicable) is valid up to committed length.

**Why this priority**: Prevents cache bloat and wrong reuse of prefix cache.

**Independent Test**: After a dLLM step that commits k < LOOKAHEAD_SIZE tokens, assert that the request's logical length (num_computed_tokens) increased by k and that blocks for (LOOKAHEAD_SIZE - k) tokens were freed or never committed.

**Acceptance Scenarios**:

1. **Given** a dLLM step that commits k tokens, **When** update_from_output runs, **Then** the request's output and num_computed_tokens grow by k and the KV cache manager frees or does not retain slots for the remaining LOOKAHEAD_SIZE - k positions.
2. **Given** prefix caching is enabled, **Then** documentation or contract states that prefix match is valid only up to committed length (e.g., PREFIX_LEN - LOOKAHEAD_SIZE worst case).

---

### User Story 4 - No regression for non-dLLM (Priority: P1)

As a user and developer, I want existing causal decode, speculative decoding, and other modes to behave exactly as before when dLLM is not enabled.

**Why this priority**: Backward compatibility is non-negotiable.

**Independent Test**: Run the existing v1 test suite (scheduler, worker, engine, e2e, spec_decode) without enabling dLLM; all tests pass. No new code path is exercised for requests that are not dLLM.

**Acceptance Scenarios**:

1. **Given** ModelRunnerOutput without dllm_step_output, **When** the scheduler processes it, **Then** behavior is unchanged (sampled_token_ids path only).
2. **Given** SchedulerOutput without scheduled_dllm_input_tokens, **When** the worker builds the input batch, **Then** input is derived from prompt + output + spec_token_ids as today.

---

### Edge Cases

- What happens when a dLLM step commits 0 tokens? Next-step input is still provided; the request does not advance in output length but the next forward uses the new input (e.g., re-masked block).
- What happens when one of the committed tokens is EOS (or triggers stop)? Request is marked finished; remaining committed tokens may be trimmed; next-step input is not needed for that request.
- What happens on the first decode step after prefill when next_dllm_input_token_ids is not yet set? The worker uses the last LOOKAHEAD_SIZE tokens of (prompt + output) when the sequence is long enough; when shorter than LOOKAHEAD_SIZE, the worker right-pads with MASK so the block is [prompt+output, MASK, …, MASK] (context left, positions to decode right). The scheduler does not include that request in scheduled_dllm_input_tokens for that step.
- How does preemption interact with dLLM? Request state (including next_dllm_input_token_ids) lives on the Request in the scheduler; after preemption and resumption, the scheduler still has the correct state to send to the worker.
- What if LOOKAHEAD_SIZE differs per architecture? LOOKAHEAD_SIZE is a constant per model/request (e.g., from config or request); the scheduler and worker use the value associated with that request.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST determine dLLM mode at model level (e.g., from model architecture or model config). When a dLLM model is loaded, all requests in that instance are treated as dLLM; the scheduler and worker MUST use LOOKAHEAD_SIZE (from model config) for scheduled tokens and next-step input length. No per-request toggle.
- **FR-002**: The model runner MUST be able to return, for dLLM requests, a variable-length list of committed token IDs (0..LOOKAHEAD_SIZE) and a fixed-length list of next-step input token IDs (exactly LOOKAHEAD_SIZE) per request. The worker MUST validate these lengths before returning (scheduler trusts the shape).
- **FR-003**: The scheduler MUST apply committed token IDs to the Request (append to output, update num_computed_tokens, respect stop conditions) and MUST store the next-step input token IDs on the Request (e.g., next_dllm_input_token_ids).
- **FR-004**: The scheduler MUST include the stored next-step input (when present) in the output sent to the worker on the next schedule (e.g., scheduled_dllm_input_tokens) so the worker can build the input batch without holding durable request state.
- **FR-005**: The worker MUST build the input batch for dLLM requests from scheduled_dllm_input_tokens when present; when absent (e.g., first decode step), the worker MUST use the last LOOKAHEAD_SIZE tokens of prompt+output, or when the sequence is shorter than LOOKAHEAD_SIZE, right-pad with MASK so the block is [prompt+output, MASK, …, MASK].
- **FR-006**: The system MUST free or not retain KV cache slots for positions beyond the committed count each step (i.e., LOOKAHEAD_SIZE - k slots for a step that commits k tokens).
- **FR-007**: The system MUST NOT use the draft-token (spec decode) path for updating next-step input for dLLM requests; next-step input comes only from the dLLM step output stored by the scheduler.
- **FR-008**: Existing code paths (causal decode, speculative decoding, chunked prefill) MUST remain unchanged when dLLM is not enabled; new fields (e.g., dllm_step_output, scheduled_dllm_input_tokens) MUST have safe defaults and be optional.

### Key Entities

- **Request**: Holds prompt, output_token_ids, num_computed_tokens, and (for dLLM) next_dllm_input_token_ids. Single source of truth for "what to run next" after the scheduler applies step output.
- **ModelRunnerOutput**: Carries sampled_token_ids (existing) and optionally dllm_step_output (DllmStepOutput: committed_token_ids, next_step_input_token_ids per request).
- **SchedulerOutput**: Carries num_scheduled_tokens, scheduled_spec_decode_tokens (existing), and optionally scheduled_dllm_input_tokens (req_id -> list of LOOKAHEAD_SIZE token IDs) for the worker to use as input for the current step.
- **DllmStepOutput**: Per-step result from the worker for dLLM: committed_token_ids (list of list of int, variable length per request), next_step_input_token_ids (list of list of int, exactly LOOKAHEAD_SIZE per request).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single dLLM request can be run from prefill through multiple decode steps with correct output length (prompt + sum of committed tokens per step) and correct next-step input flow (worker receives stored next-step input on the next schedule).
- **SC-002**: Multiple dLLM requests can be batched in one step; each receives the correct committed tokens and next-step input in the output.
- **SC-003**: KV cache growth per step equals the number of committed tokens (not LOOKAHEAD_SIZE); unused lookahead slots are freed.
- **SC-004**: Full v1 test suite (scheduler, worker, engine, e2e, spec_decode) passes without enabling dLLM; no behavioral change for non-dLLM requests.
- **SC-005**: Documentation or in-code contract describes prefix caching validity for dLLM (e.g., up to committed length or PREFIX_LEN - LOOKAHEAD_SIZE).

**Observability:** No new metrics are required. Existing iteration stats (e.g. `num_generation_tokens` per step, which is the sum of `len(new_token_ids)` over outputs) already support TPF (tokens per forward) for dLLM: committed tokens are reported as `new_token_ids`, so TPF = generation tokens that step. Implementation may add optional dLLM-specific metrics.

## Out of Scope (This Spec)

- Concrete implementations of SDAR, LLaDA2.0, LLaDA2.1, Fast-dLLMv2, WeDLM (those are model-side logic that produce committed and next-step input from logits).
- Custom attention mask implementations for each dLLM architecture (required for full support but specified separately).
- Training or fine-tuning of dLLM models.
