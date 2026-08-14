# IPC Engine

The IPC weight transfer engine uses **CUDA IPC** (Inter-Process Communication) handles to share GPU memory directly between the trainer and inference workers on the **same GPU**. This avoids any data copying, making it the most efficient option when colocating training and inference. Multi-GPU setups are supported — weights are all gathered by each GPU and are extracted by the correct colocated process.

## When to Use IPC

- Training and inference share the **same GPU(s)** (colocated)

## How It Works

1. The trainer creates CUDA tensors for each weight and generates IPC handles using `torch.multiprocessing.reductions.reduce_tensor`. In multi-GPU setups (e.g. FSDP), each trainer rank materializes the full tensor for each parameter onto its own GPU before generating the handle — which `ModuleSource` does for you.
2. Every trainer rank contributes its handles to an all-gather; the sender merges them so each payload maps every GPU UUID to its args, and ships the merged handles to the inference engine through the client. Each worker reads only the handle for its own GPU.
3. The inference worker reconstructs the tensors from the handles using `rebuild_cuda_tensor`, reading directly from the trainer's GPU memory.

Unlike NCCL, IPC transfer is straight-line: `update_weights` *is* the transfer,
and it rides the client, so there is no concurrent broadcast to overlap with.

!!! note
    The handle all-gather in step 2 runs over the **default** process group, so it
    assumes that group is exactly the set of colocated trainer ranks and that the
    sender is a member. It is a no-op when no distributed group exists.

!!! warning
    IPC handles involve sending serialized Python objects. When using HTTP transport, you must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on both the server and client. This is because IPC handles are pickled and base64-encoded for HTTP transmission.

## Inference Side

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(model="my-model", weight_transfer_config=WeightTransferConfig(backend="ipc"))
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "ipc"}'
```

IPC needs no data-plane rendezvous, so `init_transfer_engine` opens no channel —
it only records the `packed` flag the trainer ships at the handshake, which
`receive_weights` then reads. Whether a transfer is packed is therefore never
something you configure on the inference side.

## Trainer Side

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=IPCTrainerInitInfo(rank=0, packed=False),   # rank 0 is the sender
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
    source=ModuleSource(model),
)

engine.send_weights()   # once per sync
```

`send_weights()` drives `start_weight_update`, the transfer itself, and
`finish_weight_update`, and holds strong references to the IPC-shared copies
until after the post-send barrier — the consumer's views would otherwise dangle.

Any [`VLLMWeightSyncClient`](base.md#vllmweightsyncclient) works here — the
built-in HTTP and Ray clients, or an adapter for your own stack. The engine is
identical either way.

### `IPCTrainerInitInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `rank` | — | Keyword-only. This trainer process's rank; **0 is the sender** |
| `packed` | `False` | Chunked, bounded-memory transfer (see below) |
| `packed_buffer_size_bytes` | 1 GiB | Chunk size when `packed=True` |

`packed` is a must-agree wire param: `trainer_init` ships it to the worker, which
records it and decodes accordingly. It is not a `WeightTransferConfig` field and
not a per-round `update_weights` field. `packed_buffer_size_bytes` is
producer-only — the consumer rebuilds from the IPC handle plus the per-chunk
`tensor_sizes`, so it never needs the buffer size.

## Packed (Chunked) Transfer

By default, all weights are sent in a single `update_weights` call. For large
models, this requires the full model to reside in GPU memory on both sides
simultaneously. Setting `packed=True` enables **chunked transfer** with bounded
GPU memory:

- Weights are concatenated into fixed-size packed buffers (`packed_buffer_size_bytes`).
- Each chunk is sent as a separate `update_weights` call within a single `start_weight_update` / `finish_weight_update` bracket, so the layerwise reload pass is initialized once at the start and finalized once at the end regardless of chunk count.
- After each chunk is consumed, the GPU memory for that chunk can be reclaimed.

```python
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=IPCTrainerInitInfo(
        rank=0,
        packed=True,
        packed_buffer_size_bytes=256 * 1024 * 1024,  # 256 MB chunks
    ),
    client=client,
    source=ModuleSource(model),
)
```

On a multi-rank trainer the producer reuses one buffer across chunks, so packed
mode carries a per-chunk barrier across ranks: without it, a rank could overwrite
its buffer while its colocated worker is still reading the current chunk. This is
handled inside `send_weights()`.

## Examples

- [RLHF with IPC weight syncing (`vllm serve`, HTTP)](../../../examples/rl/rlhf_http_ipc.py) - **Start here.** Server and training model share a single GPU; HTTP control plane, CUDA IPC data plane. Launches and tears down its own server
- [RLHF with IPC + FSDP2 and expert parallelism](../../../examples/rl/rlhf_ipc_fsdp_ep.py) - Multi-rank trainer colocated with a `--data-parallel-size 4` server on the same 4 GPUs: every FSDP rank builds an engine and joins the handle all-gather, with packed chunking and sleep/wake around the transfer
