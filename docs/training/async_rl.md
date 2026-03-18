# Async Reinforcement Learning

## Overview

In a standard RL training loop, generation and training happen sequentially: the policy generates rollouts, then training runs on those rollouts, and the cycle repeats. During generation the training accelerators sit idle, and vice versa.

The **one-off pipelining** approach separates the generation and training phases into two parallel coroutines, allowing the model to generate new samples while simultaneously training on previously generated data. This can lead to better GPU utilization and greater training throughput.

However, this overlap introduces a complication: weights must be updated in the inference engine mid-flight, while requests may still be in progress.

## The Pause and Resume API

To safely update weights while the inference engine is running, vLLM provides `pause_generation` and `resume_generation` methods. These let the trainer coordinate a clean window for weight synchronization without losing in-flight work.

### pause_generation

```python
await engine.pause_generation(mode="keep", clear_cache=True)
```

The `mode` parameter controls how in-flight requests are handled:

| Mode | Behavior |
| ---- | -------- |
| `"abort"` | Abort all in-flight requests immediately and return partial results (default) |
| `"wait"` | Wait for all in-flight requests to finish before pausing |
| `"keep"` | Freeze requests in the queue; they resume when `resume_generation` is called |

The `clear_cache` parameter controls whether to clear the KV cache and prefix cache after pausing.

### resume_generation

```python
await engine.resume_generation()
```

Resumes the scheduler after a pause. Any requests frozen with `mode="keep"` will continue generating.

### HTTP Endpoints

When using the vLLM HTTP server, the same functionality is available via:

- `POST /pause?mode=keep` - Pause generation
- `POST /resume` - Resume generation

!!! note "Data Parallelism"
    When using data parallelism with vLLM's **internal load balancer** (i.e. `data_parallel_backend="ray"`), pause and resume are handled automatically across all DP ranks -- a single call is sufficient. When using an **external load balancer** (i.e. multiple independent vLLM instances behind a proxy), you must send pause and resume requests to **every** engine instance individually before and after the weight update.

## Typical Async RL Flow

A typical async RL loop with weight syncing looks like this:

1. Start generating rollouts from the current policy
2. Once trainer has new weights to update to, pause generation with `mode="keep"`
3. Sync the updated weights from the trainer to the inference engine (see [Weight Transfer](weight_transfer/README.md))
4. Resume generation -- in-flight requests continue with the new weights
5. Repeat

The key insight is that requests paused with `mode="keep"` will produce tokens from the **old** weights before the pause and tokens from the **new** weights after resume. The `clear_cache` parameter controls whether the KV cache is invalidated during the pause. When `clear_cache=True`, previously cached key-value entries are discarded, so all tokens generated after resume will be computed entirely with the new weights. When `clear_cache=False`, existing KV cache entries are retained, meaning some tokens in context may still reflect the old weights (stale KV cache).

## Example

The [async RLHF example](../examples/rl/rlhf_async_new_apis.md) demonstrates this pattern with `vllm.AsyncLLMEngine`, NCCL weight transfer, and mid-flight pause/resume with validation.
