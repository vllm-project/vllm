# Quickstart: dLLM Integration (After Implementation)

**Branch**: `001-dllm-integration` | **Date**: 2025-03-01

This document describes how to run and test the dLLM feature once the implementation is complete. It does not prescribe the implementation order.

---

## Prerequisites

- vLLM built/installed with dLLM support.
- A dLLM model (or stub) that the engine loads as a dLLM model (model config or architecture indicates dLLM and LOOKAHEAD_SIZE).

---

## Running a dLLM model

1. **Start the engine** with a dLLM model. dLLM mode is inferred from the loaded model (e.g. architecture or config); no extra CLI flag required if the model is recognized as dLLM.

   ```bash
   # Example (exact command depends on how dLLM models are registered)
   python -m vllm.entrypoints.openai.api_server --model <dllm-model-path-or-name>
   ```

2. **Send a request** as usual (e.g. OpenAI-compatible API). All requests in that instance are dLLM (one model per instance).

3. **Observe**: Output length grows by the sum of committed tokens per step. TPF (tokens per forward) = generation tokens that step (existing iteration stats).

---

## Running tests

- **Unit tests**: Scheduler dLLM path, worker output validation, first-step padding, KV free tail.
  ```bash
  pytest tests/v1/core/test_scheduler.py -k dllm
  pytest tests/v1/test_outputs.py -k dllm
  ```

- **Regression**: Full v1 suite without dLLM must pass.
  ```bash
  pytest tests/v1/
  ```
  (Exclude or skip dLLM-only tests when no dLLM model is configured.)

- **E2E (optional)**: Stub dLLM model, one request, prefill + several decode steps; assert output length and next-step input flow.
  ```bash
  pytest tests/v1/e2e/test_dllm_stub_e2e.py
  ```

---

## Verifying the contract

- With a dLLM model loaded, after each decode step:
  - Request’s output length increases by the number of committed tokens for that request.
  - Request’s next_dllm_input_token_ids is set (until consumed next schedule).
  - SchedulerOutput for the next step includes scheduled_dllm_input_tokens for that request (when not first decode step).
- KV cache: after a step that commits k < LOOKAHEAD_SIZE, logical length grows by k; tail slots freed.

---

## Configuration (implementation-defined)

- LOOKAHEAD_SIZE and MASK token ID may come from model config or a small dLLM config block. See implementation and model registration for the exact location.
