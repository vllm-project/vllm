# Internal Contract: dLLM Step (Engine ↔ Worker)

**Branch**: `001-dllm-integration` | **Date**: 2025-03-01

This document describes the in-process / cross-process contract between the scheduler (engine) and the worker for dLLM decode steps. No external API surface.

---

## 1. Scheduler → Worker (per step)

**Channel**: `SchedulerOutput` passed to the worker with the batch.

| Contract | Type | Requirement |
|----------|------|-------------|
| num_scheduled_tokens[req_id] | int | LOOKAHEAD_SIZE for each dLLM request in decode. |
| scheduled_dllm_input_tokens[req_id] | list[int] | Length exactly LOOKAHEAD_SIZE. Present when request has next_dllm_input_token_ids set (i.e. not first decode step). Worker uses this as the input block for the forward. |

When `scheduled_dllm_input_tokens` does not contain req_id (e.g. first decode step), worker builds block from last LOOKAHEAD_SIZE of (prompt + output) or right-pads with MASK.

---

## 2. Worker → Scheduler (after step)

**Channel**: `ModelRunnerOutput` returned to the engine.

| Contract | Type | Requirement |
|----------|------|-------------|
| dllm_step_output | DllmStepOutput \| None | When dLLM model and decode step: not None. |
| dllm_step_output.req_ids | list[str] | Same order as requests in the step. |
| dllm_step_output.committed_token_ids[i] | list[int] | Length in [0, LOOKAHEAD_SIZE]. Worker MUST validate before return. |
| dllm_step_output.next_step_input_token_ids[i] | list[int] | Length exactly LOOKAHEAD_SIZE. Worker MUST validate before return. |

Scheduler uses committed_token_ids as new_token_ids for EngineCoreOutput (append to request, update num_computed_tokens, stop checks). Scheduler sets request.next_dllm_input_token_ids = next_step_input_token_ids[i]. Scheduler does not use draft-token path for these requests; post_step skips update_draft_token_ids when dLLM step output was present.

---

## 3. Serialization

Payloads (lists of int, list of list of int) are process-safe. No new binary formats. Existing ModelRunnerOutput and SchedulerOutput serialization paths apply.
