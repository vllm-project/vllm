# FEATURE: MoE CPU Offload as Expert Paging for vLLM

## Summary

Add an optional MoE CPU offload mode to vLLM.

This feature treats the GPU as an active-expert accelerator, not as the full model container.

The design is equivalent to virtual memory and page management for sparse MoE expert models:

```text
pinned CPU memory      = backing store / source of truth for all model weights
GPU memory             = limited expert execution cache
expert model           = page
missing expert bucket  = page fault
background pager       = page fault handler / prefetcher
resident expert table  = page table
cold expert eviction   = page replacement
hot expert preload     = prefetch
```

In normal inference, a large model often fails if all model weights cannot fit into GPU memory. For sparse MoE models, this is wasteful because only a small subset of experts is active for a routed token batch.

When `--moe-cpu-offload` is enabled:

- pinned CPU memory stores the full model and remains the source of truth for all model weights,
- GPU memory stores only execution-time working copies,
- the router model path is copied to GPU and kept resident,
- active expert weights are copied to GPU on demand,
- inactive experts remain in pinned CPU memory,
- routed tokens are bucketed by expert,
- missing expert buckets are queued on CPU,
- a background expert pager loads hot missing experts into GPU memory,
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

A larger value allows the runtime to hold more routed expert buckets in the CPU miss-bucket pipeline. This gives the expert pager a larger future-demand window and improves active expert reuse.

A smaller value keeps behavior closer to normal vLLM latency-oriented execution.

## Virtual-Memory Analogy

This feature treats MoE expert execution as a virtual-memory and page-management problem.

Pinned CPU memory is the backing store and source of truth for all model weights.

GPU memory is a limited expert page cache.

Each expert model is treated like a page.

When routed tokens require an expert that is not resident on GPU, the runtime records a missing expert bucket. This is equivalent to a page fault.

A background expert pager reads the missing bucket queue, aggregates future demand, selects the hottest missing experts, and copies them from pinned CPU memory to GPU memory.

The resident expert table acts like a page table. It records which expert pages are currently resident on GPU, which are loading, which are executing, and which can be evicted.

Cold expert eviction is equivalent to page replacement. Eviction removes only the GPU cached copy; the pinned CPU source-of-truth copy remains valid.

The CPU can hold a large missing bucket pipeline. This gives the pager visibility into future expert demand and enables hot-expert prefetch instead of purely reactive expert loading.

The key idea is:

```text
The CPU miss-bucket pipeline is the future-demand window for GPU expert page management.
```

## Runtime Architecture: Router Loop + Expert Paging Loop

When `--moe-cpu-offload` is enabled, the runtime uses two cooperating loops:

1. a foreground routing/execution loop,
2. a background expert paging loop.

Pinned CPU memory is the source of truth for all model weights. GPU memory is the execution cache for the router, active experts, KV cache, activations, and runtime buffers.

## Startup

At startup:

1. Load all model weights into pinned CPU memory.
2. Copy the router model path to GPU.
3. Preload a small number of initially hot expert models to GPU.
4. Initialize the resident expert table.
5. Initialize the missing bucket queue.
6. Start the background expert paging thread.

The number of preloaded experts is bounded by:

```text
--moe-active-expert-budget
```

The total GPU memory usage is bounded by:

```text
--moe-gpu-limit
```

## Foreground Loop: Router and Resident Expert Execution

The foreground loop is responsible for normal inference progress.

```text
loop:
    batch tokens
    run router on GPU
    build token-to-expert buckets
    sync compact bucket metadata to CPU
    dispatch buckets whose experts are already resident on GPU
    enqueue missing expert buckets into the CPU miss-bucket queue
    execute resident expert buckets on GPU
    update expert usage stats
```

### Router Compute

The router model runs on GPU.

It produces compact routing metadata:

```text
token_index -> expert_id(s), routing_score(s)
```

The token hidden states stay on GPU.

Only routing metadata and bucket metadata are synchronized back to CPU. The system must not move full hidden-state tensors back to CPU for scheduling.

### Bucket Construction

The runtime groups routed tokens by expert:

```text
expert_id -> token_indices, routing_scores
```

The CPU-side bucket is a schedule, not a tensor container.

The actual token hidden states remain on GPU and are later indexed by token id when the resident expert executes.

### Resident Expert Dispatch

The CPU checks the resident expert table.

If an expert is already resident on GPU, its bucket can be dispatched immediately:

```text
if expert_id in resident_experts:
    dispatch bucket to GPU executor
else:
    enqueue bucket into missing_bucket_queue
```

### Missing Bucket Queue

Buckets whose experts are not resident on GPU are placed into a CPU-side missing bucket queue.

The missing bucket queue is the demand signal for the background paging loop.

Each missing bucket should contain metadata such as:

```python
class MissingExpertBucket:
    layer_id: int
    expert_id: int
    token_indices: list[int]
    routing_scores: list[float]
    token_count: int
    enqueue_step: int
    enqueue_time: float
```

The queue should aggregate repeated misses for the same expert when possible.

## Background Loop: Expert Paging Thread

A separate background thread manages CPU-to-GPU expert movement.

```text
loop:
    inspect missing_bucket_queue
    aggregate demand by expert
    rank missing experts by hotness
    choose the hottest expert to load next
    check GPU memory pressure
    evict cold resident experts if needed
    copy hot expert weights from pinned CPU memory to GPU
    update resident_expert_table
    notify foreground loop that new buckets are executable
```

This loop is similar to page management.

Pinned CPU memory is like backing storage. GPU memory is like a limited page cache. Experts are the pages.

## Hot Expert Selection

The paging thread chooses which missing expert to load based on queue demand.

The first version can use:

```text
hotness(expert) = number of queued tokens waiting for this expert
```

A better version can use:

```text
hotness(expert) =
    queued_token_count
  + queue_age_bonus
  + recent_reuse_bonus
  - load_cost_penalty
```

Where:

- `queued_token_count` means how many tokens are blocked waiting for this expert,
- `queue_age_bonus` prevents starvation,
- `recent_reuse_bonus` keeps recently hot experts resident longer,
- `load_cost_penalty` discourages loading a large expert for a tiny bucket.

The background pager should prioritize experts that maximize:

```text
tokens computed per expert load
```

and minimize:

```text
expert bytes moved per generated token
```

## Resident Expert Table

The runtime maintains a CPU-side table describing GPU expert residency:

```python
class ResidentExpertEntry:
    layer_id: int
    expert_id: int
    gpu_slot_id: int
    weight_bytes: int
    loaded_step: int
    last_used_step: int
    recent_token_count: int
    state: Literal["loading", "resident", "executing", "evicting"]
```

The foreground loop reads this table to decide whether a bucket can run immediately.

The paging loop updates this table when experts are loaded or evicted.

This table must be thread-safe.

## Expert Cache Policy

The GPU expert cache is controlled by two limits:

```text
--moe-active-expert-budget
--moe-gpu-limit
```

`--moe-active-expert-budget` limits how many expert models may be resident on GPU.

`--moe-gpu-limit` limits how much total GPU memory the offload mode may consume.

When a new expert must be loaded and there is not enough GPU memory, the runtime evicts cold experts.

A cold expert is one that:

- is resident on GPU,
- is not currently executing,
- is not needed by a high-priority queued bucket,
- has low recent token reuse,
- has old `last_used_step`.

Simple first version:

```text
evict expert with:
    lowest recent_token_count
    then oldest last_used_step
```

Eviction removes only the GPU cached copy. The pinned CPU source-of-truth copy remains unchanged.

## Missing Bucket Pipeline

The missing bucket queue is not just a waiting list. It is the demand signal used by the expert pager.

The CPU may hold a large missing bucket pipeline across multiple routed token waves.

The pager aggregates queued buckets by expert:

```text
expert_hotness = total queued tokens waiting for that expert
```

This lets the pager preload the hottest missing experts first.

A large missing bucket pipeline allows the runtime to trade latency for better expert reuse and fewer expert loads.

This changes the behavior from naive offload:

```text
token needs expert -> load expert -> compute -> unload
```

to expert paging:

```text
router generates many future expert demands
CPU accumulates missing buckets
CPU ranks hot missing experts
background pager preloads experts
GPU keeps computing resident buckets
cold experts are retired only when needed
```

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

## Threading and Synchronization Rules

There should be two logical threads:

```text
Thread 1: foreground inference / router / dispatch loop
Thread 2: background expert paging / preload / eviction loop
```

They communicate through:

```text
resident_expert_table
missing_bucket_queue
expert_load_events
gpu_memory_accounting
```

Important synchronization rules:

1. Do not evict an expert while a GPU kernel is using it.
2. Mark expert state as `loading` before copying CPU to GPU.
3. Mark expert state as `resident` only after the copy completes.
4. Mark expert state as `executing` while a dispatched bucket is using it.
5. Mark expert state as `evicting` before releasing a GPU slot.
6. A bucket can execute only if its expert state is `resident`.
7. Hidden states stay on GPU; only bucket metadata is synchronized to CPU.
8. CPU source-of-truth weights are read-only during inference.
9. The paging thread must never modify the CPU source-of-truth weights.

## Target Behavior

When `--moe-cpu-offload` is enabled:

1. Load all model weights into pinned CPU memory.
2. Keep pinned CPU memory as the canonical source of truth for all model weights.
3. Copy the router model path to GPU and keep it resident.
4. Preload a small number of hot experts to GPU.
5. Keep inactive experts in pinned CPU memory.
6. Batch tokens and run the router first.
7. Aggregate routed tokens into expert buckets.
8. Sync compact bucket metadata to CPU.
9. Dispatch buckets whose experts are already resident on GPU.
10. Put missing buckets into the CPU miss-bucket pipeline.
11. Let the background pager rank missing experts by hotness.
12. Progressively preload hot missing expert weights from pinned CPU memory to GPU.
13. Reuse active experts across the pipeline window.
14. Keep at most `--moe-active-expert-budget` active experts resident on GPU.
15. Respect `--moe-gpu-limit` when allocating router, KV cache, active experts, staging buffers, activations, and MoE runtime workspace.
16. Evict cold active experts when a new expert must be loaded.
17. If needed, execute active experts stage by stage to avoid OOM.

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
- maintaining the missing bucket pipeline,
- deciding which resident buckets should execute immediately,
- ranking missing experts by bucket hotness,
- respecting `moe_active_expert_budget`,
- respecting `moe_max_pipeline_depth`,
- improving reuse of already-loaded experts.

### Background Expert Pager

Responsible for:

- reading the missing bucket queue,
- aggregating missing demand by expert,
- selecting hot experts to preload,
- evicting cold experts under memory pressure,
- copying expert weights from pinned CPU memory to GPU,
- updating the resident expert table.

## Basic Runtime Flow

```text
startup:
    load all weights into pinned CPU memory
    copy router to GPU
    preload initial hot experts to GPU
    initialize resident_expert_table
    initialize missing_bucket_queue
    start background expert pager

foreground loop:
    batch tokens
    run router on GPU
    build expert buckets
    sync compact bucket metadata to CPU
    for each bucket:
        if expert is resident on GPU:
            dispatch bucket to GPU
        else:
            enqueue bucket into missing_bucket_queue
    execute resident buckets
    update expert usage stats

background paging loop:
    read missing_bucket_queue
    aggregate queued demand by expert
    select hottest missing expert
    if GPU memory is full:
        evict cold resident expert
    copy selected expert from pinned CPU memory to GPU
    update resident_expert_table
    wake foreground loop if needed
```

## Correctness Requirements

In exact mode, the output should match normal routed-expert execution as closely as standard runtime nondeterminism allows.

The offload mode should not replace a routed expert with another expert.

If the router selects expert `e`, the runtime must execute expert `e`. If `e` is not resident, load `e` from pinned CPU memory to GPU before execution.

## Metrics for First Version

Add simple metrics or logs first:

- whether MoE CPU offload is enabled,
- number of experts currently resident on GPU,
- number of queued missing buckets,
- number of tokens waiting in missing buckets,
- expert cache hit ratio,
- expert page fault count,
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
- missing bucket queue aggregation,
- hot expert selection from missing buckets,
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
- expert cache hit ratio,
- expert page fault count,
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
8. Missing buckets are queued on CPU and aggregated by expert demand.
9. A background expert pager can preload hot missing experts.
10. Cold experts can be evicted from GPU while remaining valid in CPU memory.
11. Exact routed expert semantics are preserved.
12. The runtime can reduce execution granularity when needed to avoid OOM.
13. A basic MoE model can run with this mode on one GPU.

## Recommended Development Order

1. Add config and CLI flags.
2. Add tests for config defaults and parsing.
3. Add expert residency state object.
4. Add missing bucket queue data structures.
5. Add unit tests for resident expert budget and eviction.
6. Add unit tests for missing bucket aggregation and hot expert selection.
7. Add pinned CPU source-of-truth weight handling.
8. Identify expert weights for one model family.
9. Keep expert weights in pinned CPU memory.
10. Load active experts to GPU on demand.
11. Add foreground router/bucket/dispatch loop hooks.
12. Add background expert pager loop.
13. Add hotness-based expert bucket planning.
14. Add cold expert eviction under GPU memory pressure.
15. Add staged active expert execution if needed.
16. Connect to the actual MoE execution path.
17. Add correctness comparison against normal vLLM.
18. Add simple metrics.
