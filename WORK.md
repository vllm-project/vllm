# WORK: MoE CPU Offload Expert Paging Roadmap

## Goal

Implement `--moe-cpu-offload` as a sparse-MoE expert paging system.

The design treats:

```text
pinned CPU memory          = source of truth for all model weights
GPU memory                 = active expert execution cache
router                     = GPU-resident routing path
hot expert                 = expert model currently useful for routed token buckets
missing expert bucket      = routed token bucket whose expert is not resident on GPU
active expert model list   = GPU expert page table
missing expert model list  = expert page-fault queue
background pager           = hot expert preload and cold expert eviction manager
```

The work is divided into three stages.

## Stage 1: Add New Flags and Expose Them in Usage

### Feature Goal

Add the new MoE CPU offload flags into vLLM and make them visible in help and usage output.

### Flags

```bash
--moe-cpu-offload
--moe-gpu-limit 0.4
--moe-active-expert-budget 2
--moe-max-pipeline-depth 4
```

### Expected Behavior

When users run vLLM help, the new flags should appear.

When `--moe-cpu-offload` is not set, vLLM behavior must remain unchanged.

### Work Items

1. Add `MoEOffloadConfig`.
2. Add `moe_offload_config` into `VllmConfig`.
3. Add CLI fields into `EngineArgs`.
4. Wire CLI values into runtime config.
5. Add validation for defaults and value ranges.
6. Add tests for flag parsing and defaults.

### Done When

- `--moe-cpu-offload` appears in CLI help.
- `--moe-gpu-limit` defaults to `0.4`.
- `--moe-active-expert-budget` defaults to `2`.
- `--moe-max-pipeline-depth` defaults to `4`.
- Offload mode is disabled by default.
- Existing vLLM behavior is unchanged when the master flag is not set.

## Stage 2: CPU Offload for All Weights and Synchronous Active Expert Loading

### Feature Goal

Implement the first working CPU-offload path.

In this stage, all model weights are kept in pinned CPU memory as the source of truth. GPU memory is used as an execution cache.

When the router produces a bucket for an expert that is not resident on GPU, the runtime loads that active hot expert from CPU to GPU first, then computes the bucket.

This stage is synchronous and simple. There is no background pager yet.

### Expected Behavior

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

### Work Items

1. Implement pinned CPU source-of-truth weight handling.
2. Identify router weights and expert weights for the first target MoE model.
3. Keep router path GPU-resident.
4. Implement GPU expert cache with resident expert table.
5. Implement synchronous expert load from CPU to GPU.
6. Implement simple cold expert eviction.
7. Implement router-first bucket construction.
8. Dispatch resident expert buckets.
9. On missing expert bucket, load expert synchronously and then compute.
10. Add correctness tests against normal vLLM execution.

### First Hotness Policy

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

### First Eviction Policy

Use a simple deterministic policy:

```text
evict resident expert with:
    lowest recent_token_count
    then oldest last_used_step
    excluding currently executing experts
```

### Done When

- `--moe-cpu-offload` can run one target MoE model.
- All model weights have a pinned CPU source-of-truth copy.
- Router path runs on GPU.
- Active expert weights can be loaded from CPU to GPU.
- Missing expert buckets trigger synchronous expert load.
- Resident expert budget is enforced.
- Cold expert eviction works.
- Exact routed expert semantics are preserved.
- Existing vLLM behavior remains unchanged when offload is disabled.

## Stage 3: Background Hot Expert Preload and Page Management

### Feature Goal

Introduce background management for hot expert models.

In Stage 2, missing experts are loaded synchronously. In Stage 3, the CPU maintains missing expert demand and a background expert pager preloads hot expert models into GPU memory.

This turns the system into virtual memory and page management for MoE experts.

### Two Shared Structures

Stage 3 uses two shared structures between the foreground main loop and the background pager.

```text
active_expert_model_list
missing_expert_model_list
```

### Shared Structure 1: `active_expert_model_list`

This list is the GPU expert page table.

It records which expert models are currently usable, loading, executing, or evicting on GPU.

Writer:

```text
background expert pager
```

Reader:

```text
foreground main loop
```

The foreground loop reads this list to decide whether an expert bucket can execute immediately.

Each entry should contain:

```python
class ActiveExpertEntry:
    layer_id: int
    expert_id: int
    gpu_slot_id: int
    state: Literal["loading", "resident", "executing", "evicting"]
    weight_bytes: int
    loaded_step: int
    last_used_step: int
    recent_token_count: int
```

### Shared Structure 2: `missing_expert_model_list`

This list is the expert page-fault queue.

It records expert buckets whose expert models are not currently resident on GPU.

Writer:

```text
foreground main loop
```

Reader:

```text
background expert pager
```

The background pager reads this list to decide which missing expert models should be loaded next.

Each entry should contain:

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

In code, this may be named `MissingExpertQueue` or `missing_bucket_queue`, because the structure stores work waiting for an expert model, not only an expert id.

### Expected Behavior

When `--moe-cpu-offload` is enabled:

1. Foreground loop runs router on GPU.
2. Foreground loop builds expert buckets.
3. Foreground loop reads `active_expert_model_list`.
4. Buckets whose experts are marked `resident` execute immediately.
5. Buckets whose experts are missing or still `loading` are pushed into `missing_expert_model_list`.
6. Background pager watches `missing_expert_model_list`.
7. Background pager aggregates missing demand by expert.
8. Background pager selects the hottest missing expert.
9. If GPU memory or expert budget is full, background pager evicts cold experts.
10. Background pager preloads hot expert weights from pinned CPU memory to GPU.
11. Background pager updates `active_expert_model_list`.
12. Foreground loop sees the updated `active_expert_model_list` and dispatches newly executable buckets.

### Foreground Main Loop

```text
loop:
    run router on GPU
    build expert buckets
    for each expert bucket:
        read active_expert_model_list
        if expert state is resident:
            mark expert as executing
            dispatch bucket to GPU
            update expert usage stats
            mark expert back to resident
        else:
            append bucket to missing_expert_model_list
```

### Background Expert Pager

```text
loop:
    read missing_expert_model_list
    aggregate missing demand by expert
    choose hottest missing expert
    if GPU memory or expert budget is full:
        evict cold expert from active_expert_model_list
    mark selected expert as loading
    copy expert weights from pinned CPU memory to GPU
    mark selected expert as resident in active_expert_model_list
    remove or mark satisfied buckets from missing_expert_model_list
```

### Hotness Policy

Stage 3 hotness should be based on the missing bucket pipeline:

```text
hotness(expert) = total queued tokens waiting for this expert
```

Later, it can become:

```text
hotness(expert) =
    queued_token_count
  + queue_age_bonus
  + recent_reuse_bonus
  - load_cost_penalty
```

### Thread-Safety Rules

1. `active_expert_model_list` must be thread-safe.
2. `missing_expert_model_list` must be thread-safe.
3. The foreground loop only dispatches buckets whose expert state is `resident`.
4. The background pager may write `loading`, `resident`, and `evicting` states.
5. The foreground loop may temporarily mark `resident -> executing -> resident`.
6. The background pager must not evict an expert in `executing` state.
7. The missing expert list should aggregate repeated misses for the same `(layer_id, expert_id)`.
8. The CPU pinned source-of-truth weights are never removed or modified.
9. Hidden states stay on GPU; only compact routing or bucket metadata is synchronized to CPU.

### Metrics Added in Stage 3

- active expert count,
- missing expert bucket count,
- tokens waiting in missing buckets,
- expert cache hit ratio,
- expert page fault count,
- expert load count,
- expert eviction count,
- tokens per expert load,
- expert bytes moved per token.

### Done When

- Missing expert buckets are queued instead of always loading synchronously.
- Background pager preloads hot missing experts.
- `active_expert_model_list` is maintained by the background pager and read by the foreground main loop.
- `missing_expert_model_list` is maintained by the foreground main loop and read by the background pager.
- GPU expert cache behaves like a page cache.
- Executing experts are not evicted.
- Cache hit/miss metrics are visible.
- Tokens per expert load can be measured.
- A large miss bucket pipeline can improve expert reuse.

## Final Architecture

After Stage 3, the runtime should behave like this:

```text
main loop -> writes misses -> missing_expert_model_list
background pager -> reads misses -> loads hot experts
background pager -> writes active experts -> active_expert_model_list
main loop -> reads active experts -> dispatches executable buckets
```

The final design is:

```text
router decides expert demand
main loop records missing expert buckets
CPU miss-bucket pipeline exposes future demand
background pager preloads hot expert models
GPU accelerates resident active experts
cold experts are evicted like pages
```
