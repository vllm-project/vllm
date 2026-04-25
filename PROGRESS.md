# PROGRESS: MoE CPU Offload Expert Paging

## Current Branch

```text
moe
```

This branch is for the new MoE CPU offload expert paging feature.

The high-level design is documented in:

```text
FEATURE.md
```

The staged implementation plan is documented in:

```text
WORK.md
```

## Core Design Summary

This feature treats sparse-MoE inference as a virtual-memory/page-management problem.

```text
pinned CPU memory          = source of truth for all model weights
GPU memory                 = active expert execution cache
router                     = GPU-resident routing path
expert model               = page
missing expert bucket      = page fault
active expert model list   = GPU expert page table
missing expert model list  = expert page-fault queue
background pager           = hot expert preload / cold expert eviction manager
```

The goal is to run large sparse-MoE models on smaller GPUs by using the GPU as an active-expert accelerator instead of requiring the whole model to fit in GPU memory.

## Completed Work

## Stage 0: Design Docs

Completed.

Added:

```text
FEATURE.md
WORK.md
```

`FEATURE.md` describes the architecture:

- CPU pinned memory as source of truth,
- GPU memory as active expert cache,
- router-first execution,
- expert buckets,
- missing bucket pipeline,
- background expert pager,
- active expert model list,
- cold expert eviction,
- staged active expert execution for OOM survival.

`WORK.md` defines the three-stage development roadmap:

```text
Stage 1: Add new flags and expose them in usage
Stage 2: CPU offload for all weights and synchronous active expert loading
Stage 3: Background hot expert preload and page management
```

## Stage 1: Add New Flags and Expose Them in Usage

Completed.

Stage 1 added the user-facing CLI/config surface only. It does not change vLLM runtime execution behavior.

Added the following flags:

```bash
--moe-cpu-offload
--moe-gpu-limit 0.4
--moe-active-expert-budget 2
--moe-max-pipeline-depth 4
```

### Files Added

```text
vllm/config/moe.py
vllm/engine/moe_offload_cli.py
```

### Files Modified

```text
vllm/config/__init__.py
vllm/engine/__init__.py
tests/v1/engine/test_engine_args.py
```

### Implemented Details

`vllm/config/moe.py` adds:

```python
class MoEOffloadConfig:
    enabled: bool = False
    gpu_limit: float = 0.4
    active_expert_budget: int = 2
    max_pipeline_depth: int = 4
```

`vllm/config/__init__.py` exports `MoEOffloadConfig`.

`vllm/engine/moe_offload_cli.py` adds Stage-1 CLI/config wiring and attaches:

```python
vllm_config.moe_offload_config
```

The current Stage-1 implementation uses a small import-time `EngineArgs` extension module to avoid large direct edits to `vllm/engine/arg_utils.py` through the GitHub file API.

This is acceptable for branch-stage development. For upstream-quality cleanup later, the shim can be folded directly into `arg_utils.py`.

`vllm/engine/__init__.py` imports the Stage-1 CLI extension.

`tests/v1/engine/test_engine_args.py` adds tests for:

- flag visibility,
- default values,
- explicit value parsing,
- `MoEOffloadConfig` attachment to `VllmConfig`,
- validation of value ranges,
- offload disabled by default.

### Stage 1 Acceptance Status

Done:

- `--moe-cpu-offload` appears in CLI wiring.
- `--moe-gpu-limit` defaults to `0.4`.
- `--moe-active-expert-budget` defaults to `2`.
- `--moe-max-pipeline-depth` defaults to `4`.
- Offload mode is disabled by default.
- No model execution behavior is changed yet.

## Important Rule for Next Work

Do not change default vLLM behavior.

All runtime behavior changes must be gated by:

```python
vllm_config.moe_offload_config.enabled
```

or equivalent access to the Stage-1 config.

If `--moe-cpu-offload` is not set, stock vLLM execution should remain unchanged.

## Next Work: Stage 2

Stage 2 is the next coding target.

## Stage 2: CPU Offload for All Weights and Synchronous Active Expert Loading

### Goal

Implement the first working CPU-offload path.

In Stage 2:

- all model weights are kept in pinned CPU memory as the source of truth,
- GPU memory is used as an execution cache,
- router model path is copied to GPU and kept resident,
- active expert weights are copied from CPU to GPU on demand,
- missing expert buckets trigger synchronous expert loading,
- no background pager yet.

Stage 2 should be simple and synchronous.

## Stage 2 Expected Runtime Behavior

When `--moe-cpu-offload` is enabled:

1. Load all model weights into pinned CPU memory.
2. Keep pinned CPU memory as the canonical source of truth.
3. Copy the router model path to GPU.
4. Optionally preload a small number of initial hot experts to GPU.
5. Run the router on GPU.
6. Build token-to-expert buckets.
7. For each expert bucket:
   - if the expert is already resident on GPU, execute it;
   - if the expert is missing, synchronously load it from CPU to GPU;
   - then execute the bucket.
8. Respect `--moe-active-expert-budget`.
9. If GPU expert cache is full, evict a cold expert before loading the missing hot expert.
10. CPU source-of-truth weights must remain unchanged.

## Stage 2 First Implementation Strategy

Start with one real MoE model family.

Recommended first target:

```text
Mixtral-style MoE
```

Reasons:

- simpler and widely supported,
- clear router/gate and expert modules,
- good first target before Qwen/DeepSeek-style variants.

Stage 2 should first identify:

```text
router module
expert module structure
expert weight tensor names
MoE forward path
where routed expert ids are produced
where expert execution is dispatched
```

## Stage 2 Work Items for Codex

1. Inspect existing vLLM MoE model and layer implementation.
2. Locate Mixtral-style MoE modules and fused MoE execution path.
3. Identify where router logits/top-k expert ids are computed.
4. Identify where expert weights are loaded and stored today.
5. Add a small MoE offload runtime module for Stage 2.
6. Add a CPU source-of-truth weight manager.
7. Add a GPU active expert cache/resident table.
8. Implement synchronous CPU-to-GPU active expert load.
9. Implement simple cold expert eviction.
10. Gate everything behind `moe_offload_config.enabled`.
11. Add tests with mock tensors first if full model tests are too heavy.
12. Add one functional smoke path for a real MoE model only after mock logic is stable.

## Stage 2 Minimum Data Structures

Suggested initial structures:

```python
class ActiveExpertEntry:
    layer_id: int
    expert_id: int
    state: Literal["loading", "resident", "executing", "evicting"]
    weight_bytes: int
    loaded_step: int
    last_used_step: int
    recent_token_count: int
```

```python
class ExpertCache:
    cpu_source_weights: dict[tuple[int, int], object]
    gpu_resident_experts: dict[tuple[int, int], ActiveExpertEntry]
```

For Stage 2, the cache can be synchronous and single-threaded.

Do not implement the background pager yet.

## Stage 2 Hotness and Eviction

For Stage 2, hotness can be simple:

```text
hotness = current bucket token count
```

Because loading is synchronous, the minimum required behavior is:

```text
if bucket expert is missing:
    load that expert from CPU to GPU
    compute bucket
```

First eviction policy:

```text
evict resident expert with:
    lowest recent_token_count
    then oldest last_used_step
    excluding currently executing experts
```

## Stage 2 Non-Goals

Do not implement these in Stage 2:

- background pager thread,
- `missing_expert_model_list`,
- async expert preloading,
- same-group fallback,
- approximate expert substitution,
- multi-GPU expert paging,
- expert prediction,
- dense model changes.

## Stage 3 Preview

Stage 3 will add the true page-management architecture.

It introduces two shared structures:

```text
active_expert_model_list
missing_expert_model_list
```

Ownership:

```text
active_expert_model_list:
    writer = background expert pager
    reader = foreground main loop

missing_expert_model_list:
    writer = foreground main loop
    reader = background expert pager
```

Stage 3 is not the next immediate coding target. Build Stage 2 first.

## Suggested Codex Instruction

Use this instruction for the next coding run:

```text
Read FEATURE.md, WORK.md, and PROGRESS.md.
Implement Stage 2 only.
Do not implement the background pager yet.
Keep default vLLM behavior unchanged when --moe-cpu-offload is not set.
Start by inspecting the existing Mixtral/MoE execution path.
Add synchronous CPU source-of-truth expert offload and active expert load behind moe_offload_config.enabled.
Add mock/unit tests before attempting a full real-model smoke test.
```
