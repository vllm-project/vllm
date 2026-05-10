# Sleep Mode

vLLM's Sleep Mode allows you to temporarily release most GPU memory used by a model, including model weights and KV cache, without stopping the server or unloading the Docker container. This is especially useful for RLHF, training, or cost-saving scenarios where GPU resources need to be freed between inference workloads.

Key benefits:

- **Frees GPU memory**: Offloads model weights to CPU RAM and discards KV cache, releasing up to 90%+ of GPU memory for other tasks.
- **Fast resume**: Quickly wake up the engine and resume inference without full model reload.
- **API endpoints**: Control sleep/wake_up state via HTTP endpoints or Python API.
- **Supports distributed workloads**: Works with tensor parallelism, pipeline parallelism, etc.
- **Fine-grained control**: Optionally wake up only model weights or KV cache to avoid OOM during weight updates.
- **Tag-wise selective offload**: Use `offload_tags=["weights"]` (or `["kv_cache"]`) on `sleep()` to release one tagged pool while keeping the other live on GPU. This enables hybrid co-location with partial rollout — the engine can yield weights' GPU memory to a co-located trainer without losing its KV cache, so an in-flight generation can be resumed without a re-prefill.

!!! note
    This feature is now supported on CUDA and ROCm platform.

!!! note
    For more information, see this [Blog Post](https://blog.vllm.ai/2025/10/26/sleep-mode.html).

## Tag-wise selective sleep (hybrid co-location with partial rollout)

In addition to the integer `level`, `LLM.sleep` accepts an `offload_tags`
argument that names which of the engine's tagged GPU memory pools to
release. The known tags are `"weights"` and `"kv_cache"`. Tags that
appear in `offload_tags` are backed up to CPU and unmapped from GPU;
tags that are *not* in `offload_tags` are left fully mapped on GPU and
untouched — no CPU copy, no unmap.

This is what enables hybrid co-location with partial rollout in RL
training. When a trainer and an inference engine share the same GPUs:

1. Inference produces a partial rollout (the prompt + already-generated
   prefix lives in the KV cache).
2. The trainer needs the GPU for a step. The engine releases its
   weights' GPU memory but *keeps* the KV cache mapped, freezing any
   in-flight requests in place rather than aborting them:
   `llm.sleep(offload_tags=["weights"], mode="keep")`.
3. After the training step, the engine reloads its weights:
   `llm.wake_up(tags=["weights"])`. Because the KV cache survived and
   the in-flight requests were frozen, prefill for the partial prompt
   does not need to be repeated and the rollout resumes from where it
   left off.

Important behavior to keep in mind:

- The default `mode="abort"` cancels any in-flight generation; if you
  want partial rollouts to *resume* after wake_up, you must pass
  `mode="keep"` (or `"wait"`) to retain the requests through the
  sleep.
- The engine's prefix cache is cleared only when `"kv_cache"` is in the
  offloaded set. Under tag-wise sleep that preserves the KV cache, the
  prefix cache is preserved as well.
- `wake_up(tags=[t])` will warn and refuse if `t` was never offloaded
  (it's not in the executor's `sleeping_tags`). Wake the tag you slept.
- `offload_tags=["weights", "kv_cache"]` is equivalent in memory effect
  to `level=1` but is more explicit; it does *not* save model buffers
  separately the way `level=2` does.
- `offload_tags=[]` is a pure pause: no GPU memory is offloaded and
  the executor stays awake. It is functionally equivalent to
  `level=0`; use `wake_up(tags=["scheduling"])` to resume.

```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
out = llm.generate("Hello", SamplingParams(max_tokens=8))

# Yield weights' GPU memory to a co-located trainer; keep KV cache and
# any in-flight requests frozen so they can resume after wake_up.
llm.sleep(offload_tags=["weights"], mode="keep")

# ... trainer runs a step on the freed GPU memory ...

# Restore weights. KV cache was never asleep, so we don't include it.
llm.wake_up(tags=["weights"])
out2 = llm.generate("Hello", SamplingParams(max_tokens=8))
```

The same control is available on the HTTP server: pass `offload_tags`
as a repeated query parameter, e.g.
`POST /sleep?offload_tags=weights`. Omitting `offload_tags` falls back
to the legacy `level`-based behavior.

## Sleep levels

Level 1 sleep will offload the model weights and discard the KV cache. The content of KV cache is forgotten. Level 1 sleep is good for sleeping and waking up the engine to run the same model again. The model weights are backed up in CPU memory. Please make sure there's enough CPU memory to store the model weights. Level 2 sleep will discard both the model weights and the KV cache (while the model's buffers are kept in CPU, like rope scaling tensors). The content of both the model weights and KV cache is forgotten. Level 2 sleep is good for sleeping and waking up the engine to run a different model or update the model, where previous model weights are not needed, e.g. RLHF weight update.

## Usage

### Offline inference

Enable sleep mode by passing `enable_sleep_mode=True` to the `LLM` class.

```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

#### Python API

```python
# Sleep level 1
# Put the engine to sleep (level=1: offload weights to CPU RAM, discard KV cache)
llm.sleep(level=1)

# Wake up the engine (restore weights)
llm.wake_up()
```

```python
# Sleep level 2
# Put the engine to sleep (level=2: discard both weights and KV cache)
llm.sleep(level=2)

# Reallocate weights memory only
llm.wake_up(tags=["weights"])

# Load weights in-place
llm.collective_rpc("reload_weights")

# Reallocate KV cache
llm.wake_up(tags=["kv_cache"])
```

#### RLHF weight updates

During RLHF training, vLLM allows you to selectively wake up only the model weights or the KV cache using the tags argument in wake_up(). This fine-grained control is especially useful when updating model weights: by waking up just the weights (e.g., llm.wake_up(tags=["weights"])), you avoid allocating memory for the KV cache until after the weight update is complete. This approach helps prevent GPU out-of-memory (OOM) errors, particularly with large models, by minimizing peak memory usage during weight synchronization and update operations.

Use `tags=["weights"]` or `tags=["kv_cache"]` to control which resources are restored, useful for RLHF and weight updates. **Note** that `is_sleeping` will report `true` until all components are awake.

```python
# Put engine to deep sleep (level=2)
llm.sleep(level=2)
# ... Get the new weights
# Wake up only weights to avoid OOM
llm.wake_up(tags=["weights"])
# ... Update the weights
# wake up KV cache after weights are updated
llm.wake_up(tags=["kv_cache"])
```

### Online Serving

To enable sleep mode in a vLLM server you need to initialize it with the flag `VLLM_SERVER_DEV_MODE=1` and pass `--enable-sleep-mode` to the vLLM server.

#### Server in development mode

When using the flag `VLLM_SERVER_DEV_MODE=1` you enable development endpoints, and these endpoints should not be exposed to users.

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

Below is an example of how to sleep and wake up a model in level 1.

```bash
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

And this is an example of how to sleep and wake up a model in level 2.

```bash
curl -X POST 'http://localhost:8000/sleep?level=2'
# Reallocate weights memory only
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
# Load weights in-place
curl -X POST 'http://localhost:8000/collective_rpc' -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
# Reallocate KV cache
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

#### HTTP endpoints

- `POST /sleep?level=1` — Put the model to sleep (`level=1`).
- `POST /wake_up` — Wake up the model. Supports optional `tags` query parameters for partial wake-up (e.g., `?tags=weights`).
- `POST /collective_rpc` — Perform a collective remote procedure call (RPC).
- `GET /is_sleeping` — Check if the model is sleeping.

!!! note
    These endpoints are only available when passing `VLLM_SERVER_DEV_MODE=1`.

## Limitation

On ROCm, the virtual memory allocation on ROCm is done through chunked memory allocation. You can control the chunk size through `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` (in MB). The default value is set at 256MB. The larger the chunk size the faster the performance. However, setting it too large will cause OOM. So if you encounter OOM when using sleep mode. Try reducing the chunk size. It is recommended to define the chunk size as a power of 2.
