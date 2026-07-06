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

By default, all weights are sent in a single API call. For large models, this requires the full model to reside in GPU memory on both sides simultaneously. Setting `packed=True` on `IPCWeightTransferConfig` enables **chunked transfer** with bounded GPU memory:

- Weights are concatenated into fixed-size packed buffers (controlled by `packed_buffer_size_bytes`).
- Each chunk is sent as a separate `update_weights` call within a single `start_weight_update` / `finish_weight_update` bracket, so the layerwise reload pass is initialized once at the start and finalized once at the end regardless of chunk count.
- After each chunk is consumed, the GPU memory for that chunk can be reclaimed.

```python
from vllm.config import IPCWeightTransferConfig

config = IPCWeightTransferConfig(
    packed=True,
    packed_buffer_size_bytes=256 * 1024 * 1024,  # 256 MB chunks
)
```

`packed` and `packed_buffer_size_bytes` are static wire params on the config, constructed
identically at the trainer and inference sides.

## Initialization

The IPC backend requires no data-plane rendezvous. `trainer_init` still calls the inference
side's (no-op) `init_weight_transfer_engine` via the client.

## Sending Weights

The transport is chosen by the `VLLMWeightSyncClient` passed at `trainer_init`; the trainer
code is identical across transports.

```python
from vllm.config import IPCWeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

# Ray (vLLM running as a Ray actor):
client = RayVLLMWeightSyncClient(llm_actor_handle)
# ...or HTTP (vLLM running as a server):
client = HTTPVLLMWeightSyncClient("http://localhost:8000")

engine = WeightTransferTrainerFactory.trainer_init(
    backend="ipc",
    config=IPCWeightTransferConfig(packed=False),
    init_info=IPCTrainerInitInfo(rank=0),  # single-GPU trainer = sender
    client=client,
    source=ModuleSource(model),
)

# Drives start_weight_update / update_weights / finish_weight_update.
engine.send_weights()
```

In **Ray** mode the IPC handles pass through Ray's serialization natively. In **HTTP** mode
the `HTTPVLLMWeightSyncClient` pickles and base64-encodes the handles into JSON, so the vLLM
server must be started with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`.

### Custom transports

The control plane is a structural `VLLMWeightSyncClient` protocol — any object with
`init_weight_transfer_engine`, `start_weight_update`, `update_weights`, and
`finish_weight_update` (all taking/returning the documented dicts) works as a `client`, with
no import or subclassing required.

### Multi-rank (FSDP) trainers

Every rank builds the engine via `trainer_init` and calls `send_weights()` concurrently. All
ranks join the IPC handle all-gather (and any FSDP `full_tensor()` gather); only rank 0 (the
sender) ships the merged handles and drives the inference side — non-sender ranks guard those
steps on `is_sender` internally. Pass each rank's index via the init info
(`IPCTrainerInitInfo(rank=...)`). See the FSDP example below.

## Examples

- [RLHF with IPC weight syncing (offline, Ray)](../../../examples/rl/rlhf_ipc.py) - Colocated training and inference on a single GPU using Ray placement groups and CUDA IPC handles
- [RLHF with IPC weight syncing (online serving, HTTP)](../../../examples/rl/rlhf_http_ipc.py) - Weight transfer with a vLLM HTTP server where both server and trainer share the same GPU
