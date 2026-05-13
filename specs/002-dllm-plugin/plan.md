# Implementation Plan: dLLM (Blocked Masked Diffusion LLM) Plugin

**Branch**: `002-dllm-plugin` | **Date**: 2025-03-01 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/002-dllm-plugin/spec.md`

## Summary

Deliver block-based dLLM support as a **plugin**: users install a plugin package to run dLLM models (e.g. LLaDA2.0, WeDLM) without modifying vLLM core. The engine must support the dLLM step contract (one step = one forward, committed tokens, next-block input) when a plugin-registered dLLM model is loaded; the plugin registers one or more dLLM model architectures via `vllm.general_plugins` and `ModelRegistry.register_model`, following the [bart-plugin](https://github.com/vllm-project/bart-plugin) pattern. Core changes (scheduler/worker/outputs) align with the 001-dllm-integration design; the plugin package is a separate installable (e.g. `vllm-dllm-plugin`) with registration and model implementations.

## Technical Context

**Language/Version**: Python 3.10+ (vLLM existing)  
**Primary Dependencies**: PyTorch, vLLM (with dLLM execution path), transformers (for plugin model impl)  
**Storage**: N/A (in-memory request state, KV cache; no new persistence)  
**Testing**: pytest; vLLM tests in `tests/` (plugins_tests, v1); plugin package can ship its own tests (e.g. `tests/` in plugin repo)  
**Target Platform**: Same as vLLM (Linux/CUDA, etc.)  
**Project Type**: Inference engine extension (plugin package) + core dLLM execution path in vLLM  
**Performance Goals**: Preserve continuous batching; one forward per step; TPF derivable from existing metrics  
**Constraints**: Plugin must not require vLLM source edits to add new dLLM architectures; re-entrant registration; VLLM_PLUGINS filter support  
**Scale/Scope**: (1) vLLM repo: dLLM step contract in v1 (scheduler, worker, outputs); (2) plugin repo: one package, one or more model architectures

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check |
|-----------|--------|
| I. Correctness First | dLLM output (committed + next-step input) must match model semantics; stop conditions and KV growth validated by tests. |
| II. Scheduler as Single Source of Truth | Next-step input stored on Request; worker receives via SchedulerOutput; no durable worker state for request content. PASS. |
| III. Test-Driven Development | Plan includes unit tests for new contracts and plugin registration; integration/e2e for dLLM plugin load and inference. |
| IV. One Step, One Forward | One scheduler step = one forward; step output is variable-length committed + fixed-length next input. PASS. |
| V. Backward Compatibility | dLLM enabled only when a dLLM model (from plugin) is loaded; new fields optional with safe defaults. PASS. |
| VI. Performance and Batching | Same batching/scheduling budget; KV cache freed for unused lookahead slots. PASS. |
| Prefix caching | Document prefix cache validity (committed length); disable for bidirectional prefill models. |
| Serialization | DllmStepOutput and scheduled_dllm_input_tokens use lists/simple types for engine–worker payload. PASS. |
| Development Workflow | Follow v1 layout for core; plugin follows bart-plugin layout (entry_points, register function, model module). PASS. |

**Gate result**: PASS (no unjustified violations). *Post Phase 1 design*: Re-checked; data model and contracts align with scheduler as source of truth, one step one forward, and plugin-only model registration. No changes.

## Project Structure

### Documentation (this feature)

```text
specs/002-dllm-plugin/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 (decisions, bart-plugin patterns)
├── data-model.md        # Phase 1 (entities, plugin + core state)
├── quickstart.md        # Phase 1 (how to run and test plugin)
├── contracts/           # Phase 1 (plugin contract, dLLM step contract)
└── tasks.md             # Phase 2 (/speckit.tasks – not created by plan)
```

### Source Code

**In vLLM repository** (dLLM execution path; reference 001-dllm-integration design):

```text
vllm/
├── v1/
│   ├── outputs.py                    # DllmStepOutput; ModelRunnerOutput.dllm_step_output
│   ├── request.py                    # Request.next_dllm_input_token_ids
│   ├── core/sched/
│   │   ├── output.py                 # SchedulerOutput.scheduled_dllm_input_tokens
│   │   └── scheduler.py               # schedule(), update_from_output() for dLLM
│   ├── worker/
│   │   ├── gpu_model_runner.py        # dLLM step logic, validation
│   │   └── gpu_input_batch.py        # build input from scheduled_dllm_input_tokens / first-step
│   └── engine/core.py                # post_step applies dllm_step_output
```

**Plugin package** (separate repository or tree, bart-plugin–style):

```text
vllm-dllm-plugin/                    # or dllm-plugin/
├── vllm_dllm_plugin/
│   ├── __init__.py                  # register_dllm_model() -> ModelRegistry.register_model(...)
│   ├── llada.py                     # LLaDA2 / LLaDA2.1 model class (example)
│   └── stub.py                     # Optional: minimal stub model for tests
├── setup.py  # or pyproject.toml   # entry_points: vllm.general_plugins -> dllm = vllm_dllm_plugin:register_dllm_model
├── README.md
├── verify_plugin.py                 # Optional: verify plugin loads and model is available
└── tests/
    └── test_plugin.py               # Test registration and basic inference
```

**Structure Decision**: Two-part delivery: (1) vLLM core gains the dLLM step contract (same as 001) so that when the loaded model declares dLLM and returns DllmStepOutput, the scheduler/engine use it; (2) a separate plugin package (inspired by [bart-plugin](https://github.com/vllm-project/bart-plugin)) provides the model registration and one or more dLLM model implementations. Plugin discovery uses existing `load_general_plugins()`; model resolution uses existing `ModelRegistry`.

## Complexity Tracking

None; no constitution violations requiring justification.
