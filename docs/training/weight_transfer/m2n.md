# NCCL M2N Engine

The `nccl_m2n` weight transfer engine uses [NCCL M2N](https://github.com/NVIDIA/nccl-extensions) to **reshard** weights between two disjoint meshes of GPUs — the trainer and the inference workers — in a single collective per parameter. Unlike the broadcast-based [NCCL engine](nccl.md), the trainer sends its local shard as-is: no full tensor is ever materialized on the training side, whatever its parallelism layout.

## When to Use NCCL M2N

- The trainer and the inference engine use **different parallelism layouts** (e.g. trainer FSDP/EP, inference TP) and gathering full tensors on the trainer is expensive
- Training and inference are on **separate GPUs**, possibly across nodes
- You want weight sync off the critical path of an async RL loop

## Requirements

This backend depends on the `nccl-extensions` Python package (and its `nccl4py` dependency), which vLLM does not install. Install both against a compatible NCCL 2.30.5 or newer runtime. The backend creates its data-plane communicator through `nccl4py` and passes that communicator directly to `nccl.m2n`.

The backend imports the runtime lazily, so vLLM is unaffected unless you select it.

## How It Works

1. A `StatelessProcessGroup` exchanges metadata and the NCCL unique ID. The trainer ranks and all inference workers then create **one** `nccl.core.Communicator`: trainer ranks occupy `[0, T)` and workers `[T, T + N)`. The communicator is passed directly to each M2N collective.
2. At the init handshake the trainer ships the full transfer plan: the session-wide destination mesh, then each parameter's name, dtype, shape, source mesh, and source placements. The workers resolve the matching **destination placements** and publish them over the data-plane communicator; the meshes themselves are declared, not negotiated.
3. Each round, both sides issue one `reshard` per parameter in the same order. Every rank in the communicator participates in every reshard, so — as with the NCCL engine — the worker must be inside `update_weights` while the trainer sends. The trainer engine handles that concurrency itself.

### Destination resolution

A parameter is resharded **directly into the live vLLM parameter** when its checkpoint name maps 1:1 onto a model parameter whose shape is either identical to the checkpoint shape (replicated) or differs on exactly one dim by the tensor-parallel factor. That rank then only ever receives its own shard.

Anything else falls back to receiving the full tensor and letting `load_weights` do the sharding, exactly as the NCCL engine does for every parameter. The fallback covers:

- fused parameters (`qkv_proj`, `gate_up_proj`, MoE `w13`/`w2`), whose checkpoint names do not name a model parameter
- quantized models, where the weight loader does more than a sharded copy
- pipeline-parallel deployments

Correctness never depends on a parameter resolving. The engine logs how many parameters took each path at init.

## Configuration

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl_m2n"),
)
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "nccl_m2n"}'
```

## Trainer Side

The trainer uses the stateful trainer engine. m2n plans each reshard from the parameter's source layout, which a [`WeightSource`](base.md#parammeta-ownership-and-layout) declares on its `ParamMeta` (`owners`, `mesh_shape`, `placements`), and sends what `iter_local()` yields: each rank's **local shard**, never a `full_tensor()`. `ModuleSource` covers the common case: it reads each parameter's device mesh and placements.

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

# Called on every trainer rank; rank 0 is the sender.
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=M2NTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=num_trainer_ranks + num_inference_workers,
        num_trainer_ranks=num_trainer_ranks,
        rank=my_rank,
    ),
    client=RayVLLMWeightSyncClient(llm_handle),
    source=ModuleSource(model),
)

# Each round, on every trainer rank:
engine.send_weights()
```

`send_weights()` drives the whole round — `start_weight_update`, `update_weights`, `finish_weight_update`. Every rank in the communicator must be inside the same reshard, so `update_weights` has to be in flight while the trainer sends; the engine runs it on a helper thread so the trainer loop does not have to open-code that.

Trainers with a custom producer (a Megatron export, MoE re-fusing) subclass
`WeightSource`, declare each layout on `ParamMeta` and implement `iter_local()`.
A 1-D mesh is described over m2n's 2-D mesh with a size-1 second axis. Each
yielded tensor must match its declared dtype and local shape, and is kept alive
in stream order until its reshard completes.

See [`examples/rl/rlhf_m2n.py`](../../../examples/rl/rlhf_m2n.py) for a runnable FSDP → TP example.

## Limitations

- vLLM validates tensor-rank and dtype support while agreeing on the complete
  plan; the installed M2N runtime validates them again for each reshard.
- Every parameter must be owned by all trainer ranks, in rank order (`owners` unset or `(0, ..., T-1)`), with a 1-D or 2-D mesh. Partitioned owners, such as one group per pipeline stage, are rejected at init.
- `Partial` placements cannot be sent as shards. `ModuleSource` declares such a parameter replicated and gathers it in `iter_local()`.
- Models with known model-level checkpoint preprocessing use the full-tensor
  fallback for the affected module.
- A reshard cannot be issued from inside a CUDA graph capture (weight updates run outside capture, so this only matters for custom callers).
- vLLM checks every parameter's shard overlap against conservative 16-source
  and 64-target capacities before committing the plan. The installed M2N
  runtime remains authoritative and validates each reshard again.
- Each communicator holds a cached staging pool (~256 MiB by default: 4 channels x 64 MiB).
