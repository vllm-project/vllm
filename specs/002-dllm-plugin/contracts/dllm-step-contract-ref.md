# Reference: dLLM Step Contract (Engine ↔ Worker)

**Branch**: `002-dllm-plugin` | **Date**: 2025-03-01

The internal contract between the scheduler and the worker for dLLM decode steps is **unchanged** from the core dLLM integration design. It is implemented in vLLM core when a dLLM model (including plugin-registered) is loaded.

**Full contract**: [001-dllm-integration/contracts/dllm-step-contract.md](../../001-dllm-integration/contracts/dllm-step-contract.md)

**Summary**:
- **Scheduler → Worker**: `SchedulerOutput.scheduled_dllm_input_tokens[req_id]` (length LOOKAHEAD_SIZE) when not first decode step; worker uses first-step convention when missing.
- **Worker → Scheduler**: `ModelRunnerOutput.dllm_step_output` with `committed_token_ids` (0..LOOKAHEAD_SIZE per request) and `next_step_input_token_ids` (exactly LOOKAHEAD_SIZE per request). Worker validates lengths before return.

Plugin model classes do not produce DllmStepOutput directly; the worker in core builds it from the model’s forward output and validates it.
