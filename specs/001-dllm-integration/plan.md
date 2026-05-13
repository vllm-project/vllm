# Implementation Plan: dLLM (Blocked Masked Diffusion LLM) Integration

**Branch**: `001-dllm-integration` | **Date**: 2025-03-01 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/001-dllm-integration/spec.md`

## Summary

Integrate blocked masked diffusion LLMs (dLLMs) into vLLM so that each diffusion step is one worker iteration: the model consumes a lookahead buffer of fixed size LOOKAHEAD_SIZE, and the worker returns (1) variable-length committed token IDs (0..LOOKAHEAD_SIZE) and (2) fixed-length next-step input token IDs (exactly LOOKAHEAD_SIZE). The scheduler applies committed tokens to the Request and stores next-step input on the Request, then sends it to the worker via SchedulerOutput. One model per instance; dLLM mode is determined at model level. Existing continuous batching, KV cache, and metrics (TPF = generation tokens per step) apply without new observability requirements.

## Technical Context

**Language/Version**: Python 3.10+ (vLLM existing)  
**Primary Dependencies**: PyTorch, vLLM v1 stack (scheduler, worker, engine, outputs)  
**Storage**: N/A (in-memory Request state, KV cache; no new persistence)  
**Testing**: pytest; existing v1 tests in `tests/v1/` (core, worker, engine, e2e, spec_decode)  
**Target Platform**: Same as vLLM (Linux/CUDA, etc.)  
**Project Type**: Inference engine extension (library-style integration into vLLM)  
**Performance Goals**: Preserve continuous batching; one forward per step; TPF derivable from existing IterationStats  
**Constraints**: No behavioral change when dLLM not enabled; new fields optional with safe defaults; worker validates dLLM output lengths  
**Scale/Scope**: v1 code paths only; scheduler, worker, engine, outputs, KV cache manager

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check |
|-----------|--------|
| I. Correctness First | dLLM output (committed + next-step input) must match model semantics; stop conditions and KV growth validated by tests. |
| II. Scheduler as Single Source of Truth | Next-step input stored on Request; worker receives via SchedulerOutput; no durable worker state for request content. PASS. |
| III. Test-Driven Development | Plan includes unit tests for new contracts (DllmStepOutput, scheduler/worker branches) and integration/e2e for critical path. |
| IV. One Step, One Forward | One scheduler step = one forward; step output is variable-length committed + fixed-length next input. PASS. |
| V. Backward Compatibility | dLLM enabled only when dLLM model loaded; new fields (dllm_step_output, scheduled_dllm_input_tokens) optional with defaults. PASS. |
| VI. Performance and Batching | Same batching/scheduling budget; KV cache freed for unused lookahead slots. PASS. |
| Prefix caching | Document prefix cache validity (committed length / PREFIX_LEN - LOOKAHEAD_SIZE). |
| Serialization | DllmStepOutput and scheduled_dllm_input_tokens use lists/simple types for engine–worker payload. PASS. |
| Development Workflow | Follow v1 layout (core, worker, engine); extend outputs with safe defaults; document contracts. PASS. |

**Gate result**: PASS (no unjustified violations).

## Project Structure

### Documentation (this feature)

```text
specs/001-dllm-integration/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 (decisions, rationale)
├── data-model.md        # Phase 1 (entities, state)
├── quickstart.md        # Phase 1 (how to run/test)
├── contracts/           # Phase 1 (internal contracts)
└── tasks.md             # Phase 2 (/speckit.tasks – not created by plan)
```

### Source Code (repository root)

```text
vllm/
├── v1/
│   ├── outputs.py                    # DllmStepOutput; ModelRunnerOutput.dllm_step_output
│   ├── request.py                    # Request.next_dllm_input_token_ids
│   ├── core/
│   │   ├── sched/
│   │   │   ├── output.py             # SchedulerOutput.scheduled_dllm_input_tokens
│   │   │   └── scheduler.py          # schedule() dLLM path; update_from_output() dLLM path; free tail slots
│   │   └── kv_cache_manager.py       # free_tail_blocks (or equivalent) for unused lookahead
│   ├── worker/
│   │   ├── gpu_input_batch.py        # dLLM input build from scheduled_dllm_input_tokens / first-step convention
│   │   └── gpu_model_runner.py       # dLLM step logic; validate lengths; produce DllmStepOutput
│   └── engine/
│       └── core.py                   # post_step: skip draft update for dLLM
├── config/                            # Model/config: dLLM flag, LOOKAHEAD_SIZE (if needed)
tests/
└── v1/
    ├── core/test_scheduler.py        # dLLM scheduler tests
    ├── worker/                       # worker/input batch dLLM tests
    ├── engine/                       # post_step regression
    ├── test_outputs.py               # DllmStepOutput, ModelRunnerOutput
    └── e2e/                          # dLLM stub e2e (optional)
```

**Structure Decision**: Integration is confined to vLLM v1: outputs, request, scheduler, KV cache manager, worker (input batch + model runner), and engine core. Tests live alongside existing v1 tests. No new top-level packages.

## Complexity Tracking

No constitution violations requiring justification. This section is empty.
