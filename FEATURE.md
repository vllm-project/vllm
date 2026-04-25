# FEATURE: Simple MoE CPU Offload Mode for vLLM

## Summary

Add an optional MoE CPU offload mode to vLLM.

This feature treats the GPU as an active-expert accelerator, not as the full model container.

In normal inference, a large model often fails if the full model weights cannot fit into GPU memory. For sparse MoE models, this is wasteful because only a small subset of experts is active for a given token batch.

When `--moe-cpu-offload` is enabled:

- pinned CPU memory stores the full model and remains the source of truth for all model weights,
- GPU memory stores only execution-time working copies,
- the router model path is copied to GPU and kept resident,
- active expert weights are copied to GPU on demand,
- inactive experts remain in pinned CPU memory,
- routed tokens are bucketed by expert,
- hot experts are preloaded progressively,
- cold experts are evicted when GPU memory pressure increases,
- active experts may also be executed stage by stage to avoid OOM.

The goal is to let large sparse-MoE models run on smaller GPUs by using the GPU as a dynamic accelerator for the currently active expert working set.

## Hard Rule

If `--moe-cpu-offload` is not set, vLLM behavior must remain unchanged.

That means:

- no scheduling behavior change,
- no model loading behavior change,
- no expert placement change,
- no MoE execution semantic change,
- no performance regression for the default path.

## Core Goal

When `--moe-cpu-offload` is enabled, pinned CPU memory is the canonical source of truth for all model weights.

That means:

- all model weights are loaded and kept in pinned CPU memory,
- CPU memory remains the persistent source of truth throughout runtime,
- GPU memory only holds temporary working copies needed for execution,
- router weights are copied from pinned CPU memory to GPU memory and kept resident,
- active expert weights are copied from pinned CPU memory to GPU memory on demand,
- inactive expert weights remain only in pinned CPU memory,
- when an active expert is evicted from GPU memory, the canonical CPU copy remains unchanged.

The GPU copy is only a runtime cache. The pinned CPU copy is the persistent source of truth for computation.

## Flags

### `--moe-cpu-offload`

Master flag.

Default: disabled.

When disabled, vLLM runs normally.

When enabled:

- all model weights are loaded into pinned CPU memory,
- pinned CPU memory is the canonical source of truth for all model weights,
- GPU memory is used only as an execution cache,
- router weights are copied to GPU and kept resident,
- currently active expert weights are copied to GPU on demand,
- inactive expert weights remain in pinned CPU memory,
- evicted experts are removed only from GPU memory, never from the CPU source-of-truth copy.

### `--moe-gpu-limit`

Default: `0.4`

Meaning: maximum fraction of total GPU memory that the MoE offload runtime should use for the offload-managed working set.

This budget should include:

- KV cache,
- router weights,
- active expert weights on GPU,
- expert staging buffers,
- activations,
- temporary MoE workspace,
- token pipeline buffers related to MoE offload.

The first version can use conservative accounting. The important rule is that the offload path must not blindly fill GPU memory with experts.

### `--moe-active-expert-budget`

Default: `2`

Meaning: maximum number of active MoE experts kept on GPU at one time.

For the first implementation, this should be treated as a hard cap.

Example:

```bash
--moe-cpu-offload --moe-active-expert-budget 2
```

This means only two experts should be resident in the offload-managed expert cache at a time.

### `--moe-max-pipeline-depth`

Default: `4`

Meaning: maximum number of MoE execution waves that may be planned or staged ahead to improve active expert reuse.

This is the key latency-throughput tradeoff knob.

A larger value allows the runtime to wait for more routed tokens and reuse loaded experts more efficiently.

A smaller value keeps behavior closer to normal vLLM latency-oriented execution.

## Runtime Behavior

When `--moe-cpu-offload` is enabled, the runtime follows a router-first, bucketed-expert execution model.

Pinned CPU memory remains the source of truth for all model weights. GPU memory is used as an execution cache for the router, active experts, KV cache, activations, and temporary runtime buffers.

### Execution Flow

1. Batch incoming tokens using the normal vLLM batching path as much as possible.
2. Run the router model first on GPU.
3. Collect the routed expert id for each token.
4. Aggregate routed tokens into expert buckets.
5. Build an execution pipeline from these expert buckets.
6. Bound the pipeline by `--moe-max-pipeline-depth`.
7. Rank expert buckets by hotness.
8. Progressively preload the hottest required expert weights from pinned CPU memory to GPU memory.
9. Execute token buckets whose expert weights are resident on GPU.
10. Reuse resident expert weights across as many queued bucket waves as possible.
11. Track total GPU memory usage, including KV cache, router weights, active expert weights, activations, staging buffers, and temporary MoE workspace.
12. When projected GPU memory usage exceeds the configured limit, retire cold experts from GPU memory.
13. Cold expert retirement only removes the GPU cached copy. The pinned CPU source-of-truth copy remains unchanged.
14. Continue loading hot experts, executing their buckets, and retiring cold experts until the batch or pipeline is complete.

Initial hotness can be simple:

```text
hotness = number of routed tokens for this expert
```

Later, hotness can include recent reuse, queue age, and transfer cost.

## Expert Cache Policy

The GPU expert cache is controlled by two limits:

```text
--moe-active-expert-budget
--moe-gpu-limit
```

`--moe-active-expert-budget` limits how many expert models may be resident on GPU.

`--moe-gpu-limit` limits how much total GPU memory the offload mode may consume.

When a new expert must be loaded and there is not enough GPU memory, the runtime evicts cold experts.

A cold expert is an expert that:

- is not needed by the current executing bucket,
- has low recent token count,
- has not been reused recently,
- is not among the hottest experts in the current pipeline window.

The first version can use a simple LRU plus token-count policy:

```text
eviction_score = recent_token_count, then last_used_step
```

Evict the expert with the lowest recent token count and oldest last-used step.

## Staged Active Expert Execution

Even for active experts, the runtime should not assume the full active expert working set must be loaded into GPU memory at once.

When `--moe-cpu-offload` is enabled, active expert execution may be staged.

This means:

- the full model remains in pinned CPU memory,
- active experts are selected after router execution,
- active expert weights are copied to GPU progressively,
- expert buckets are executed stage by stage,
- GPU memory pressure is checked before each stage,
- cold experts or completed expert stages are evicted before loading the next stage,
- the runtime should prefer forward progress over failing with OOM.

The goal is not only to fit large sparse-MoE models on smaller GPUs, but also to survive tight GPU-memory situations during execution.

### OOM-Survival Principle

The offload runtime should avoid treating GPU memory as a binary load-or-fail constraint.

If a full active expert set does not fit, the runtime should reduce the execution granularity:

```text
full active expert set
-> one active expert
-> one expert stage
-> smaller token bucket
```

The runtime should attempt smaller stages before failing with OOM.

This makes the GPU a progressive execution accelerator rather than a fixed full-model memory container.

## Target Behavior

When `--moe-cpu-offload` is enabled:

1. Load all model weights into pinned CPU memory.
2. Keep pinned CPU memory as the canonical source of truth for all model weights.
3. Copy the router model path to GPU and keep it resident.
4. Keep inactive experts in pinned CPU memory.
5. Batch tokens and run the router first.
6. Aggregate routed tokens into expert buckets.
7. Rank expert buckets by hotness.
8. Build a bounded execution pipeline using `--moe-max-pipeline-depth`.
9. Preload hot expert weights from pinned CPU memory to GPU progressively.
10. Execute buckets for resident active experts.
11. Reuse active experts across the pipeline window.
12. Keep at most `--moe-active-expert-budget` active experts resident on GPU.
13. Respect `--moe-gpu-limit` when allocating router, KV cache, active experts, staging buffers, activations, and MoE runtime workspace.
14. Evict cold active experts when a new expert must be loaded.
15. If needed, execute active experts stage by stage to avoid OOM.

## Initial Implementation Scope

The first implementation should be intentionally narrow.

Focus on:

- one GPU,
- one sparse-MoE model family first,
- exact expert execution only,
- no same-group fallback,
- no approximation of routed experts,
- no multi-node behavior,
- no expert parallel changes.

Correctness comes before performance.

## Non-Goals for First Version

Do not implement these in the first version:

- same-group fallback,
- approximate expert substitution,
- learned expert prediction,
- multi-GPU expert scheduling,
- distributed expert parallel offload,
- changes to dense models,
- changes to default vLLM behavior.

## Suggested Internal Components

### Config

Add a small config object, for example:

```python
class MoEOffloadConfig:
    enabled: bool = False
    gpu_limit: float = 0.4
    active_expert_budget: int = 2
    max_pipeline_depth: int = 4
```

Wire it from CLI flags into `VllmConfig`.

### Expert Residency Manager

Responsible for:

- tracking which experts are currently on GPU,
- tracking which experts are stored in pinned CPU memory,
- loading a requested expert to GPU,
- evicting old experts when the active budget is full,
- accounting for approximate GPU memory usage,
- keeping CPU memory as the source of truth.

### Pipeline Planner

Responsible for:

- grouping routed tokens by expert,
- deciding which experts should be active next,
- ranking experts by bucket hotness,
- respecting `moe_active_expert_budget`,
- respecting `moe_max_pipeline_depth`,
- improving reuse of already-loaded experts.

## Basic Runtime Flow

```text
if not moe_cpu_offload:
    run normal vLLM path
else:
    keep all model weights in pinned CPU memory
    keep router on GPU
    batch tokens
    run router first
    collect routed expert ids
    bucket tokens by expert
    rank buckets by hotness
    choose active experts within budget
    progressively load hot active experts from CPU to GPU
    execute resident expert buckets
    evict cold experts when GPU memory pressure appears
    use staged expert execution if needed to avoid OOM
```

## Correctness Requirements

In exact mode, the output should match normal routed-expert execution as closely as standard runtime nondeterminism allows.

The offload mode should not replace a routed expert with another expert.

If the router selects expert `e`, the runtime must execute expert `e`. If `e` is not resident, load `e` from pinned CPU memory to GPU before execution.

## Metrics for First Version

Add simple metrics or logs first:

- whether MoE CPU offload is enabled,
- number of experts currently resident on GPU,
- number of expert loads from CPU to GPU,
- number of expert evictions,
- number of staged expert executions,
- bytes loaded from CPU to GPU if available,
- tokens served per expert load if available,
- expert bytes moved per generated token if available.

## Testing Plan

### Unit Tests

Test:

- flag parsing and config defaults,
- `--moe-cpu-offload` disabled by default,
- default `moe_gpu_limit == 0.4`,
- default `moe_active_expert_budget == 2`,
- default `moe_max_pipeline_depth == 4`,
- expert residency cap behavior,
- cold expert eviction behavior,
- CPU source-of-truth state remains valid after GPU eviction.

### Functional Test

For one supported MoE model:

1. Run normal vLLM.
2. Run with `--moe-cpu-offload`.
3. Use the same prompt and seed.
4. Compare outputs or logits within acceptable tolerance.

### Performance Smoke Test

Measure:

- GPU memory usage,
- expert load count,
- expert eviction count,
- tokens per second,
- tokens per expert load,
- expert bytes moved per token.

The first goal is not maximum speed. The first goal is to prove the architecture works.

## Acceptance Criteria

The first version is successful when:

1. vLLM default behavior is unchanged.
2. The four flags are available and wired into config.
3. `--moe-cpu-offload` enables a separate MoE offload path.
4. Pinned CPU memory is the source of truth for all model weights.
5. GPU memory is used as an execution cache for router, active experts, KV cache, activations, and runtime buffers.
6. The active expert GPU budget is enforced.
7. Experts can be loaded from pinned CPU memory to GPU on demand.
8. Cold experts can be evicted from GPU while remaining valid in CPU memory.
9. Exact routed expert semantics are preserved.
10. The runtime can reduce execution granularity when needed to avoid OOM.
11. A basic MoE model can run with this mode on one GPU.

## Recommended Development Order

1. Add config and CLI flags.
2. Add tests for config defaults and parsing.
3. Add expert residency state object.
4. Add unit tests for resident expert budget and eviction.
5. Add pinned CPU source-of-truth weight handling.
6. Identify expert weights for one model family.
7. Keep expert weights in pinned CPU memory.
8. Load active experts to GPU on demand.
9. Add hotness-based expert bucket planning.
10. Add cold expert eviction under GPU memory pressure.
11. Add staged active expert execution if needed.
12. Connect to the actual MoE execution path.
13. Add correctness comparison against normal vLLM.
14. Add simple metrics.
