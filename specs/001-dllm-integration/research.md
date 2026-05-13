# Research: dLLM Integration

**Branch**: `001-dllm-integration` | **Date**: 2025-03-01

All decisions below are resolved from the feature spec and clarifications. No open NEEDS CLARIFICATION items.

---

## 1. Where to store next-step input (scheduler vs worker)

**Decision**: Store next-step input on the Request in the scheduler; send to the worker via SchedulerOutput (`scheduled_dllm_input_tokens`).

**Rationale**: vLLM constitution (Scheduler as Single Source of Truth) and existing pattern: speculative decoding stores draft tokens on the Request and sends them in SchedulerOutput. Same pattern keeps preemption and multi-worker correct without duplicating state.

**Alternatives considered**: Keeping next-step input only in the worker was rejected because it would split request state and complicate preemption/resumption.

---

## 2. First decode step when sequence length < LOOKAHEAD_SIZE

**Decision**: Right-pad with MASK so the block is `[prompt+output, MASK, …, MASK]`. Context on the left, positions to decode (masked) on the right.

**Rationale**: dLLMs are trained to fill masked positions; the model must see positions to predict on the right. Left-padding with MASK would put “to decode” on the left, which does not match training.

**Alternatives considered**: Left-pad was rejected for decode (see spec clarifications). Requiring minimum prompt length was rejected to allow short prompts.

---

## 3. Where dLLM mode is determined

**Decision**: Model-level only. When a dLLM model is loaded (by architecture or model config), all requests in that instance are dLLM. No per-request toggle.

**Rationale**: vLLM does not support multiple models per instance; one instance = one model. So “all requests are dLLM” when that model is dLLM. Simplifies engine and avoids branching on per-request flags.

**Alternatives considered**: Request-level or both (model + request opt-in) were rejected to keep scope and implementation simple.

---

## 4. Who validates worker dLLM output lengths

**Decision**: Worker validates. The worker MUST ensure `committed_token_ids` and `next_step_input_token_ids` have correct lengths (0..LOOKAHEAD_SIZE and exactly LOOKAHEAD_SIZE per request) before returning; scheduler trusts the shape.

**Rationale**: Validation at the producer (worker) keeps scheduler simple and avoids duplicate logic. Aligns with “worker returns two values” contract.

**Alternatives considered**: Scheduler validation or both were rejected to avoid duplication and keep a single responsibility.

---

## 5. Observability and TPF (tokens per forward)

**Decision**: Reuse existing metrics. No new observability requirement. TPF for dLLM = `IterationStats.num_generation_tokens` for that step (committed tokens are reported as `new_token_ids`).

**Rationale**: Existing output pipeline already aggregates `len(new_token_ids)` per step. Feeding committed tokens into `EngineCoreOutput.new_token_ids` makes TPF derivable without new metrics.

**Alternatives considered**: Requiring dLLM-specific metrics was rejected to limit scope; optional dLLM metrics may be added later.

---

## 6. KV cache: freeing unused lookahead slots

**Decision**: After each dLLM step, free or do not retain KV cache slots for positions beyond the committed count (i.e. LOOKAHEAD_SIZE - k slots when k tokens are committed).

**Rationale**: Prevents cache bloat. Allocation is for LOOKAHEAD_SIZE tokens per step; only k are committed to the logical sequence; the rest must be freed so block tables and ref counts stay correct.

**Alternatives considered**: Keeping all LOOKAHEAD_SIZE slots would grow cache incorrectly. Freeing tail blocks (or equivalent API) is required; exact API (e.g. `free_tail_blocks(request, num_tokens)`) is an implementation detail.

---

## 7. Integration pattern (spec-decode as reference)

**Decision**: Follow the same data flow as speculative decoding for “tokens that determine next step input”: worker produces output → scheduler updates Request → next schedule sends that data to the worker via SchedulerOutput. For dLLM, the “output” is DllmStepOutput; the “tokens for next step” are next_step_input_token_ids stored on Request and sent as scheduled_dllm_input_tokens.

**Rationale**: Minimizes new concepts; reuses scheduler/worker contract and engine post_step handling (skip draft update when dLLM).

**Alternatives considered**: A separate “dLLM state” channel was rejected in favor of extending existing outputs and SchedulerOutput.
