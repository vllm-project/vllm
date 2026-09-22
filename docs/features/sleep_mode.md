# Sleep Mode

vLLM's Sleep Mode allows you to temporarily release most GPU memory used by a model, including model weights and KV cache, without stopping the server or unloading the Docker container. This is especially useful for RLHF, training, or cost-saving scenarios where GPU resources need to be freed between inference workloads.

Key benefits:

- **Frees GPU memory**: Offloads model weights to CPU RAM and discards KV cache, releasing up to 90%+ of GPU memory for other tasks.
- **Fast resume**: Quickly wake up the engine and resume inference without full model reload.
- **API endpoints**: Control sleep/wake_up state via HTTP endpoints or Python API.
- **Supports distributed workloads**: Works with tensor parallelism, pipeline parallelism, etc.
- **Fine-grained control**: Optionally wake up only model weights or KV cache to avoid OOM during weight updates.

!!! note
    This feature is now supported on CUDA and ROCm platform.

!!! note
    For more information, see this [Blog Post](https://blog.vllm.ai/2025/10/26/sleep-mode.html).

## Sleep levels

Level 1 sleep will offload the model weights and discard the KV cache. The content of KV cache is forgotten. Level 1 sleep is good for sleeping and waking up the engine to run the same model again. The model weights are backed up in CPU memory. Please make sure there's enough CPU memory to store the model weights. Level 2 sleep will discard both the model weights and the KV cache (while the model's buffers are kept in CPU, like rope scaling tensors). The content of both the model weights and KV cache is forgotten. Level 2 sleep is good for sleeping and waking up the engine to run a different model or update the model, where previous model weights are not needed, e.g. RLHF weight update. Level 2 sleep is also useful when there is not enough CPU memory to back up the model weights, e.g. when a colocated trainer already uses CPU memory for offloading its own state; since nothing is backed up, restore the weights after waking up with `collective_rpc("reload_weights")`.

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

#### Release only KV cache memory

`LLM.release_kv_cache_memory()` discards KV cache physical memory while keeping model weights resident. It requires a completed pause and all executor memory to be resident: full sleep, partial wake-up, and repeated release without restoring memory are rejected. Requests retained with `mode="keep"` are recomputed after wake-up.

Use the default `cumem` backend with managed allocations: `enable_cumem_allocator=True` on CUDA/ROCm (also enabled by `enable_sleep_mode=True`), or `enable_sleep_mode=True` on XPU. Other backends must implement selective discard; CPU and unmanaged allocations are unsupported.

```python
llm.sleep(level=0, mode="keep")  # Wait for the pause to complete.
llm.release_kv_cache_memory()
llm.wake_up(tags=["kv_cache"])  # Reallocate KV cache and resume scheduling.
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

To release only KV cache memory, wait for the level 0 pause to complete before calling the release endpoint. The same resident-memory and backend requirements as the Python API apply.

```bash
curl -X POST 'http://localhost:8000/sleep?level=0&mode=keep'
curl -X POST 'http://localhost:8000/release_kv_cache_memory'
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

#### HTTP endpoints

- `POST /sleep?level=1` — Put the model to sleep (`level=1`).
- `POST /release_kv_cache_memory` — Discard KV cache memory after a completed pause, while all executor memory is resident.
- `POST /wake_up` — Wake up the model. Supports optional `tags` query parameters for partial wake-up (e.g., `?tags=weights`).
- `POST /collective_rpc` — Perform a collective remote procedure call (RPC).
- `GET /is_sleeping` — Check if the model is sleeping.

!!! note
    These endpoints are only available when passing `VLLM_SERVER_DEV_MODE=1`.

## Limitation

### Experimental graph discard and lazy recapture

`VLLM_SLEEP_DISCARD_GRAPHS=1` destroys the model's CUDA graph executables
before level-1 memory suspension. A complete wake resumes inference eagerly;
after the first request completes, the engine captures at most one pending
graph per idle iteration. Captured shapes become usable immediately. Other
shapes continue eagerly. Ordinary sleep behavior is unchanged by default.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SLEEP_DISCARD_GRAPHS=1 \
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode --dtype float16 --max-model-len 256 \
  --max-num-seqs 8 --max-num-batched-tokens 256 --no-async-scheduling \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8]}'
```

The initial prototype requires Linux/CUDA, the V2 runner in a local EngineCore
process, one GPU, the `uni` executor and `cumem` backend. Only unquantized
`Qwen3ForCausalLM` and `Qwen3MoeForCausalLM` models with FULL graphs and no
`torch.compile` are accepted. Additional weight offloaders, KV transfer,
microbatching, LoRA, speculative decoding, live weight updates and communicator
suspension are excluded. Level 0 retains its existing behavior. Level 2,
selective KV discard and another sleep after a partial wake are rejected.
Partial wake does not permit inference or recapture until all memory is awake.

Graph destruction does not guarantee that the CUDA driver returns its cached
physical memory. vLLM does not request driver cache reclamation or explicitly
clear global cuBLAS workspaces. PyTorch may clean up capture-stream workspaces
when destroying a graph. This option does not release the entire CUDA context or vLLM's
persistent workspace.
The regression records free memory before allocator suspension, while weights
and KV remain mapped. This interval includes graph outputs and allocator cache
cleanup, so its entire change must not be attributed to graph executables
(GraphExec). Physical release must be measured on the target device and driver;
neither full GraphExec memory release nor a smaller complete-wake footprint is
guaranteed. Snapshots after wake are retained separately from the
before-suspension measurement.

Recapture never overlaps an inference step, but a request or sleep arriving
during capture waits for the current descriptor to finish. Continuous traffic
can defer completion indefinitely. A recoverable capture allocation failure
leaves the engine eager; unknown CUDA/capture failures terminate the engine so
requests receive an error instead of waiting indefinitely. A failed destructive
sleep or wake refuses further resume operations and requires restarting the
engine. The main loop reports these failures through the EngineDead path,
including failures delivered by deferred sleep callbacks, so queued requests
do not remain waiting. Repeated completed sleep/wake calls remain supported.

Recapture can also increase the cost of subsequent sleeps: PyTorch graph
capture may clear cached pinned host allocations, requiring weight backup
buffers to be allocated again. Measure repeated sleep latency as well as
wake and inference latency for the installed PyTorch version.

Eager and graph execution can have different floating-point results even
without sleep. This option does not promise cross-mode or batch-invariant
token equality. Validate the intended model, dtype and batches before use.
The GPU regression is `tests/v1/cudagraph/test_sleep_graphs.py`; it checks exact
tokens for its controlled recipe, first-request eager execution and actual
replay after idle recapture. The prototype does not implement model switching.

### ROCm

On ROCm, the virtual memory allocation on ROCm is done through chunked memory allocation. You can control the chunk size through `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` (in MB). The default value is set at 256MB. The larger the chunk size the faster the performance. However, setting it too large will cause OOM. So if you encounter OOM when using sleep mode. Try reducing the chunk size. It is recommended to define the chunk size as a power of 2.
