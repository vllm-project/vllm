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
- Each chunk is sent as a separate `update_weights` call within a single `start_weight_update` / `finish_weight_update` bracket, so the layerwise reload pass is initialized once at the start and finalized once at the end regardless of chunk count.
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

IPC supports two transport modes for delivering the handles:

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
# start
ray.get(llm_actor_handle.start_weight_update.remote(is_checkpoint_format=True))
# send weights
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
# finish
ray.get(llm_actor_handle.finish_weight_update.remote())
```

In Ray mode, the engine calls `llm_handle.update_weights.remote(...)` directly, passing the IPC handles via Ray's serialization.

### HTTP Mode

Used when vLLM is running as an HTTP server:

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="http",
    url="http://localhost:8000",
)

# start
base_url = "http://localhost:8000"
url = f"{base_url}/start_weight_update"
response = requests.post(url, json={"is_checkpoint_format": True}, timeout=60)
response.raise_for_status()
# send weights
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
# finish
url = f"{base_url}/finish_weight_update"
response = requests.post(url, json={}, timeout=60)
response.raise_for_status()
```

In HTTP mode, IPC handles are pickled, base64-encoded, and sent as JSON to the `/update_weights` endpoint. Because the worker deserializes the payload via `pickle.loads`, the vLLM server must be started with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`.

```python
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
```

See [`IPCTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/ipc_engine.py) for the full list of configurable fields.

## Examples

- [RLHF with IPC weight syncing (offline, Ray)](../../../examples/rl/rlhf_ipc.py) - Colocated training and inference on a single GPU using Ray placement groups and CUDA IPC handles
- [RLHF with IPC weight syncing (online serving, HTTP)](../../../examples/rl/rlhf_http_ipc.py) - Weight transfer with a vLLM HTTP server where both server and trainer share the same GPU
