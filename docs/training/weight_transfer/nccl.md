# NCCL Engine

The NCCL weight transfer engine uses [NCCL](https://developer.nvidia.com/nccl) broadcast operations to transfer weights from the trainer to inference workers. It supports **multi-node** and **multi-GPU** setups where the trainer and inference engine run on separate GPUs.

## When to Use NCCL

- Training and inference on **separate GPUs** (possibly across nodes)
- **Tensor-parallel** inference with multiple workers that all need the updated weights
- You need high-bandwidth, low-latency weight transfer over NVLink or InfiniBand

## How It Works

1. The trainer and all inference workers join a shared NCCL process group using `StatelessProcessGroup` (vLLM's torch.distributed-independent group abstraction). The trainer is rank 0; workers start at `rank_offset` (1).
2. The trainer broadcasts weights to all workers simultaneously. Each worker receives and loads the weights.
3. Optionally, **packed tensor broadcasting** batches multiple small tensors into larger buffers with double/triple buffering and CUDA stream overlap for higher throughput. This implementation is based on [NeMo-RL's packed tensor](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/utils/packed_tensor.py).

The workers' `update_weights` and the trainer's broadcast run at the same time —
both sides rendezvous inside the same NCCL calls. The trainer engine
owns that concurrency internally.

## Inference Side

The inference side takes a plain backend selector. The rendezvous parameters and
the packing wire params arrive from the trainer at the init handshake.

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(model="my-model", weight_transfer_config=WeightTransferConfig(backend="nccl"))
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "nccl"}'
```

Nothing else is required: `init_weight_transfer_engine`, `start_weight_update`,
`update_weights`, and `finish_weight_update` are all driven remotely by the
trainer engine.

## Trainer Side

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=NCCLTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,   # trainer + all inference workers
        rank=0,                  # this trainer rank; rank 0 is the sender
        packed=True,
    ),
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),  # or RayVLLMWeightSyncClient(llm)
    source=ModuleSource(model),
)

engine.send_weights()   # once per sync
```

`trainer_init` drives the whole handshake: it kicks off the inference side's
`init_weight_transfer_engine` (through the client) on a side thread while opening
the trainer's own rank-0 endpoint, since both ends must rendezvous together. It
builds the worker's init info itself, with `rank_offset=1` and the same packed
params, so the two sides cannot disagree.

`send_weights()` then drives `start_weight_update`, `update_weights` — run
concurrently with the broadcast — and `finish_weight_update`, and returns only
once every transfer has drained.

### `NCCLTrainerInitInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `master_address` | — | Rendezvous host |
| `master_port` | — | Rendezvous port |
| `world_size` | — | Full trainer + worker NCCL group size |
| `rank` | — | Keyword-only. This trainer process's rank; **0 is the sender** |
| `packed` | `True` | Use packed broadcasting |
| `packed_buffer_size_bytes` | 1 GiB | Packed buffer size |
| `packed_num_buffers` | 2 | Number of rotating buffers (double/triple buffering) |

`packed` defaults to `True` here. The worker-side default is `False`, but it only
applies when no trainer ships a value — which, on this path, never happens.

### Packed Tensor Broadcasting

When `packed=True`, weight tensors are packed into large contiguous buffers
before broadcasting. This cuts the number of NCCL operations and uses
double/triple buffering with dedicated CUDA streams to overlap packing,
broadcasting, and unpacking.

You set `packed`, `packed_buffer_size_bytes`, and `packed_num_buffers` **only on
`NCCLTrainerInitInfo`**. The trainer propagates them to the worker inside
`trainer_init`, the worker records them at the handshake, and `receive_weights`
decodes with exactly the values the trainer encoded with. They are not per-round
`update_weights` fields.

!!! note "Memory"
    The rotating buffers are live for the whole transfer:
    `packed_buffer_size_bytes * packed_num_buffers` on each side (2 GiB at the
    defaults). Lower `packed_buffer_size_bytes` if that is too much headroom.

### The two `WeightSource` channels must agree

Dense NCCL is the backend that reads *both*
[`WeightSource`](base.md#weightsource) channels, so it is the one where a
disagreement between them is fatal. The engine builds the per-round update info
from `metadata()` and ships it ahead of the bytes; the worker sizes its receive
buffers from that info, and in packed mode cuts its chunk boundaries from it. The
bytes themselves come from iterating the source.

If iteration disagrees with what `metadata()` declared — a reordered, omitted, or
re-dtyped parameter — the two sides split the same byte stream differently. The
transfer then either hangs in NCCL waiting for a length that never arrives, or
loads garbage into the model.

The sender therefore checks each pair against the declared metadata as it goes,
one comparison per parameter, and raises naming the first divergent parameter
rather than letting it reach the wire. `ModuleSource` satisfies this by
construction. If you write a custom source — a Megatron export, an MoE re-fusing
pass — this is the invariant to test first.

!!! note
    IPC does not read `metadata()` at all: it derives the update info from
    iteration as it goes, so it cannot observe a divergence.

## Sparse NCCL

Sparse, flat-index weight patches use
`WeightTransferConfig(backend="sparse_nccl")`. Names, full shapes, and flat
indices are interpreted in **checkpoint/Hugging Face coordinates**. Every
inference rank receives the same checkpoint-global patch, then the model's native
`load_weights()` maps it to rank-local TP/EP and packed runtime parameters.

Tensor parallelism is supported and does not have to match the trainer's layout.
Pipeline-parallel behavior is not covered by the current sparse NCCL GPU test.

Sparse is a **delta** backend: each call supplies replacement patches rather
than a stable stream of the model's parameters. The engine therefore takes no
`WeightSource`; patches go straight to `send_weights(patches)`, and an empty
patch list is a no-op.

```python
from vllm.distributed.weight_transfer import (
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseWeightPatch,
)

client = RayVLLMWeightSyncClient(llm)
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=SparseNCCLTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
        rank=0,
    ),
    client=client,
)

patches = [
    SparseWeightPatch(
        name="model.layers.0.mlp.down_proj.weight",
        indices=flat_indices,           # int32, 1-D
        values=new_values,              # same length as indices
        full_shape=tuple(param.shape),  # required when sending via the engine
    )
]
engine.send_weights(patches)
```

Each `send_weights(patches)` call owns a complete one-shot `start` / `update` / `finish` lifecycle.

RL infrastructure that owns the generic worker lifecycle can keep one logical update open across bounded chunks without adding trainer-side session state:

```python
client.start_weight_update()
for patches in patch_chunks:
    engine.send_weight_chunk(patches)
client.finish_weight_update()
```

Sparse NCCL sends only `O(nnz)` indices and values. Checkpoint application still uses `O(N)` staging through the native loader, and the caller owns export/diff state and restart or reseed after a partial failure.

[`rlhf_sparse_nccl.py`](../../../examples/rl/rlhf_sparse_nccl.py) demonstrates per-expert checkpoint patches with a Qwen3 MoE TP2/EP2 engine.

## Examples

- [RLHF with NCCL weight syncing (`vllm serve`, HTTP)](../../../examples/rl/rlhf_http_nccl.py) - **Start here.** Trainer on one GPU, 2x tensor-parallel fp8 server on two others; HTTP control plane, NCCL data plane. Launches and tears down its own server
- [RLHF with NCCL + FSDP2 and expert parallelism](../../../examples/rl/rlhf_nccl_fsdp_ep.py) - Multi-rank trainer: every FSDP rank builds an engine and joins the `full_tensor()` gather while only rank 0 touches the wire
- [RLHF with sparse NCCL weight syncing (offline, Ray)](../../../examples/rl/rlhf_sparse_nccl.py) - Qwen3 MoE per-expert checkpoint updates with one trainer GPU and two TP2/EP2 inference GPUs
- [RLHF with async weight syncing (offline, Ray)](../../../examples/rl/rlhf_async_new_apis.py) - Async generation with mid-flight pause, weight sync, resume, and validation against a fresh model; uses an in-process `AsyncLLMEngine` with `RayVLLMWeightSyncClient`
