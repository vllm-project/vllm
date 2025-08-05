# Sleep Mode

vLLM's **Sleep Mode** allows you to temporarily release most GPU memory used by a model, including model weights and KV cache, without stopping the server or unloading the Docker container. This is especially useful for RLHF, training, or cost-saving scenarios where GPU resources need to be freed between inference workloads.

## **Benefits**

- **Frees GPU memory**: Offloads model weights to CPU RAM and discards KV cache, releasing up to 90%+ of GPU memory for other tasks.
- **Fast resume**: Quickly wake up the engine and resume inference without full model reload.
- **API endpoints**: Control sleep/wake state via HTTP endpoints or Python API.
- **Supports distributed workloads**: Works with tensor parallelism.

## **Usage**

### Offline inference

Enable sleep mode by passing `enable_sleep_mode=True` to the `LLM` class.

```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

### **Python API**

```python
# Put the engine to sleep (level=1: offload weights to CPU RAM, discard KV cache)
llm.sleep(level=1)

# Wake up the engine (restore weights)
llm.wake_up()# or llm.wake_up(tags=["weights"]) for fine-grained control
```

---

### Online inference

To Enable sleep mode in a vLLM server you need to initialize it with the flag `VLLM_SERVER_DEV_MODE=1` and pass `--enable-sleep-mode` to the vLLM server 

```bash
VLLM_SERVER_DEV_MODE=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

### **HTTP Endpoints**

- `POST /sleep?level=1` — Put the model to sleep.
- `POST /wake_up` — Wake up the model.
- `GET /is_sleeping` — Check if the model is sleeping.

These endpoints are only available when passing `VLLM_SERVER_DEV_MODE=1`

### Notes

- **Sleep levels**: Level 1 sleep will offload the model weights and discard the kv cache. The content of kv cache is forgotten. Level 1 sleep is good for sleeping and waking up the engine to run the same model again. The model weights are backed up in CPU memory. Please make sure there's enough CPU memory to store the model weights. Level 2 sleep will discard both the model weights and the kv cache. The content of both the model weights and kv cache is forgotten. Level 2 sleep is good for sleeping and waking up the engine to run a different model or update the model, where previous model weights are not needed.

- **Platform support**: Supported on CUDA platform.