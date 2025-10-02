# Sleep Mode

vLLM's Sleep Mode allows you to temporarily release most GPU memory used by a model, including model weights and KV cache, without stopping the server or unloading the Docker container. This is especially useful for RLHF, training, or cost-saving scenarios where GPU resources need to be freed between inference workloads.

Key benefits:

- **Frees GPU memory**: Offloads model weights to CPU RAM and discards KV cache, releasing up to 90%+ of GPU memory for other tasks.
- **Fast resume**: Quickly wake up the engine and resume inference without full model reload.
- **API endpoints**: Control sleep/wake_up state via HTTP endpoints or Python API.
- **Supports distributed workloads**: Works with tensor parallelism, pipeline parallelism, etc.
- **Fine-grained control**: Optionally wake up only model weights or KV cache to avoid OOM during weight updates.

!!! note
    This feature is only supported on CUDA platform.

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
# Put the engine to sleep (level=1: offload weights to CPU RAM, discard KV cache)
llm.sleep(level=1)

# Wake up the engine (restore weights)
llm.wake_up()
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
VLLM_SERVER_DEV_MODE=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

#### HTTP endpoints

- `POST /sleep?level=1` — Put the model to sleep (`level=1`).
- `POST /wake_up` — Wake up the model. Supports optional `tags` query parameters for partial wake-up (e.g., `?tags=weights`).
- `GET /is_sleeping` — Check if the model is sleeping.

!!! note
    These endpoints are only available when passing `VLLM_SERVER_DEV_MODE=1`.
