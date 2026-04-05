# IPC Engine

The IPC weight transfer engine uses **CUDA IPC** (Inter-Process Communication) handles to share GPU memory directly between the trainer and inference workers on the **same GPU**. This avoids any data copying, making it the most efficient option when colocating training and inference. Multi-GPU setups are supported — weights are all gathered by each GPU and are extracted by the correct colocated process.

## When to Use IPC

- Training and inference share the **same GPU(s)** (colocated)

## How It Works

1. The trainer creates CUDA tensors for each weight and generates IPC handles using `torch.multiprocessing.reductions.reduce_tensor`. In multi-GPU setups (e.g. FSDP), each trainer rank must all-gather the full tensor for each layer onto its own GPU before generating the IPC handle.
2. IPC handles for each gpu are sent to the inference engine via **Ray**, **HTTP**, or a **custom callable**. Each rank only reads the handle corresponding to its own GPU.
3. The inference worker reconstructs the tensors from the handles using `rebuild_cuda_tensor`, reading directly from the trainer's GPU memory.

!!! warning
    IPC handles involve sending serialized Python objects. When using HTTP transport, you must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on both the server and client. This is because IPC handles are pickled and base64-encoded for HTTP transmission.

## Packed (Chunked) Transfer

By default, all weights are sent in a single API call. For large models, this requires the full model to reside in GPU memory on both sides simultaneously. Setting `packed=True` enables **chunked transfer** with bounded GPU memory:

- Weights are concatenated into fixed-size packed buffers (controlled by `packed_buffer_size_bytes`).
- Each chunk is sent as a separate API call with `first_chunk` and `last_chunk` flags so that vLLM only initializes/finalizes the reload pass on the first/last chunk.
- After each chunk is consumed, the GPU memory for that chunk can be reclaimed.

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="ray",
    llm_handle=llm_actor_handle,
    packed=True,
    packed_buffer_size_bytes=256 * 1024 * 1024,  # 256 MB chunks
)
```

## Initialization

The IPC backend requires no initialization on either side. The `init_transfer_engine` call is a no-op for IPC.

## Sending Weights

IPC supports three transport modes for delivering the handles:

### Ray Mode

Used when vLLM is running as a Ray actor:

```python
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="ray",
    llm_handle=llm_actor_handle,
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

In Ray mode, the engine calls `llm_handle.update_weights.remote(...)` directly, passing the IPC handles via Ray's serialization.

### HTTP Mode

Used when vLLM is running as an HTTP server:

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="http",
    url="http://localhost:8000",
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

In HTTP mode, IPC handles are pickled, base64-encoded, and sent as JSON to the `/update_weights` endpoint. The pickled payload contains only `rebuild_cuda_tensor` argument tuples (ints, bytes, strings) — no arbitrary callables — so no special environment flags are required.

### Custom Callable Mode

For custom transport mechanisms, pass a callable as `send_mode`. The callable receives an `IPCWeightTransferUpdateInfo` object and is responsible for delivering it to the inference engine. Both sync and async callables are supported — use `async_trainer_send_weights` for async callables:

```python
# Sync callable — use trainer_send_weights
def my_custom_sender(update_info: IPCWeightTransferUpdateInfo):
    # Custom logic to deliver update_info to vLLM
    ...

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode=my_custom_sender,
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)

# Async callable — use async_trainer_send_weights
async def my_async_sender(update_info: IPCWeightTransferUpdateInfo):
    # Custom async logic to deliver update_info to vLLM
    ...

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode=my_async_sender,
)

await IPCWeightTransferEngine.async_trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

See [`IPCTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/ipc_engine.py) for the full list of configurable fields.

## Examples

- [RLHF with IPC weight syncing (offline, Ray)](../../examples/rl/rlhf_ipc.md) - Colocated training and inference on a single GPU using Ray placement groups and CUDA IPC handles
- [RLHF with IPC weight syncing (online serving, HTTP)](../../examples/rl/rlhf_http_ipc.md) - Weight transfer with a vLLM HTTP server where both server and trainer share the same GPU
