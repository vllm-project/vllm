# Data Model: dLLM Integration

**Branch**: `001-dllm-integration` | **Date**: 2025-03-01

Entities and state relevant to the dLLM feature. Validation rules and state transitions are derived from the spec and FRs.

---

## 1. Request (extended)

**Source**: `vllm/v1/request.py`

| Field | Type | Description |
|-------|------|-------------|
| *(existing)* | … | prompt_token_ids, _output_token_ids, _all_token_ids, num_computed_tokens, spec_token_ids, etc. |
| next_dllm_input_token_ids | list[int] \| None | When set, length is exactly LOOKAHEAD_SIZE. Exact input block for the next dLLM decode step. Cleared/overwritten when scheduler consumes it and worker returns new value. |

**Validation**: When non-None, `len(next_dllm_input_token_ids) == LOOKAHEAD_SIZE` (worker validates before setting; scheduler trusts).

**Lifecycle**: Set by scheduler in `update_from_output` from `dllm_step_output.next_step_input_token_ids[i]`. Read by scheduler in `schedule()` and sent as `scheduled_dllm_input_tokens[req_id]`. Not part of `num_tokens` or `num_tokens_with_spec`.

---

## 2. DllmStepOutput

**Source**: New type in `vllm/v1/outputs.py`.

| Field | Type | Description |
|-------|------|-------------|
| req_ids | list[str] | Order matches other per-request lists. |
| committed_token_ids | list[list[int]] | Per-request list of token IDs to append to output and cache in KV. Length per request in [0, LOOKAHEAD_SIZE]. |
| next_step_input_token_ids | list[list[int]] | Per-request list of token IDs for the next forward. Length per request exactly LOOKAHEAD_SIZE. |

**Validation**: Worker MUST validate before returning: for each i, `0 <= len(committed_token_ids[i]) <= LOOKAHEAD_SIZE` and `len(next_step_input_token_ids[i]) == LOOKAHEAD_SIZE`. Scheduler trusts the shape.

---

## 3. ModelRunnerOutput (extended)

**Source**: `vllm/v1/outputs.py`

| Field | Type | Description |
|-------|------|-------------|
| *(existing)* | … | req_ids, req_id_to_index, sampled_token_ids, logprobs, etc. |
| dllm_step_output | DllmStepOutput \| None | When present, scheduler uses this (and not sampled_token_ids for append/commit) for those requests. Default None. |

**Validation**: When dllm_step_output is not None, req_ids in DllmStepOutput must match the requests in the step. Existing call sites must handle None (safe default).

---

## 4. SchedulerOutput (extended)

**Source**: `vllm/v1/core/sched/output.py`

| Field | Type | Description |
|-------|------|-------------|
| *(existing)* | … | num_scheduled_tokens, scheduled_spec_decode_tokens, etc. |
| scheduled_dllm_input_tokens | dict[str, list[int]] | req_id -> list of length LOOKAHEAD_SIZE. Input block for dLLM decode for the current step. Empty when no dLLM or first decode step (worker uses first-step convention). |

**Validation**: When present, each value has length LOOKAHEAD_SIZE. Scheduler fills from `request.next_dllm_input_token_ids` when non-None.

---

## 5. CachedRequestState / InputBatch (worker state)

**Source**: `vllm/v1/worker/gpu_input_batch.py`

For dLLM decode, the worker builds the input block for each request from either:

- `scheduler_output.scheduled_dllm_input_tokens[req_id]` when present (subsequent decode steps), or  
- First-step convention: last LOOKAHEAD_SIZE tokens of (prompt + output), or right-pad with MASK when shorter: `[prompt+output, MASK, …, MASK]`.

No new persistent fields required; the worker uses scheduler_output and request state (prompt, output_token_ids, num_computed_tokens) already present in the batch.

---

## 6. KV cache and block tables

**Logical length**: After a dLLM step that commits k tokens, request’s logical length (num_computed_tokens) increases by k. Blocks for (LOOKAHEAD_SIZE - k) positions must be freed (or never committed).

**Prefix caching**: Valid only up to committed length; document as PREFIX_LEN - LOOKAHEAD_SIZE worst case.

---

## State transitions (scheduler, per dLLM request)

1. **After schedule()**: If request has next_dllm_input_token_ids, scheduler puts it in scheduled_dllm_input_tokens and schedules num_scheduled_tokens = LOOKAHEAD_SIZE.
2. **After update_from_output() with dllm_step_output**: Append committed_token_ids to request output; update num_computed_tokens by len(committed); set request.next_dllm_input_token_ids = next_step_input_token_ids; free (LOOKAHEAD_SIZE - k) tail slots; emit EngineCoreOutput with new_token_ids = committed for metrics.
3. **First decode step**: request.next_dllm_input_token_ids is None; scheduler does not add to scheduled_dllm_input_tokens; worker builds block from prompt+output (or right-pad with MASK).
