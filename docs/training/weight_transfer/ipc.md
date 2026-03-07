# IPC Engine

The IPC weight transfer engine uses **CUDA IPC** (Inter-Process Communication) handles to share GPU memory directly between the trainer and inference workers on the **same node and same GPU**. This avoids any data copying, making it a efficient option when colocating training and inference.

## When to Use IPC

- Training and inference on the **same GPU** (colocated)
- You want to minimize memory overhead by sharing tensors in-place

## How It Works

1. The trainer creates CUDA tensors for each weight and generates IPC handles using `torch.multiprocessing.reductions.reduce_tensor`.
2. IPC handles are sent to the inference engine via **Ray RPC** or **HTTP POST**.
3. The inference worker reconstructs the tensors from the handles, reading directly from the trainer's GPU memory.

!!! warning
    IPC handles involve sending serialized Python objects. When using HTTP transport, you must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on both the server and client. This is because IPC handles are pickled and base64-encoded for HTTP transmission.

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
    mode="ray",
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
    mode="http",
    url="http://localhost:8000",
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

In HTTP mode, IPC handles are pickled, base64-encoded, and sent as JSON to the `/update_weights` endpoint.

See [`IPCTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/ipc_engine.py) for the full list of configurable fields.

## Examples

- [RLHF with IPC weight syncing (offline, Ray)](../../examples/offline_inference/weight_syncing.md#rlhf-ipc) - Colocated training and inference on a single GPU using Ray placement groups and CUDA IPC handles
- [RLHF with IPC weight syncing (online serving, HTTP)](../../examples/online_serving/weight_syncing.md#rlhf-http-ipc) - Weight transfer with a vLLM HTTP server where both server and trainer share the same GPU
