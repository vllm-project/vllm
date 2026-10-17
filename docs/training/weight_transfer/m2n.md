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

1. A `StatelessProcessGroup` exchanges metadata and the NCCL unique ID. During this metadata-only phase, trainers agree on each parameter's source layout and workers resolve their rank-local destinations. Every participant agrees on the complete ordered source and destination plans before entering NCCL initialization.
2. After plan agreement, the trainer ranks and all inference workers create **one** `nccl.core.Communicator`: trainer ranks occupy `[0, T)` and workers `[T, T + N)`. The communicator is passed directly to each M2N collective.
3. Each round, both sides issue one `reshard` per parameter in the same order. Every rank in the communicator participates in every reshard, so — as with the NCCL engine — the worker must be inside `update_weights` while the trainer sends. The trainer engine handles that concurrency itself.

### Destination resolution

A parameter uses one collectively agreed destination mode:

- **In place:** M2N writes a validated exact-name live vLLM parameter.
- **Rank-local staging:** a model module declares the local shard it owns and a callback that installs it into model-native storage. M2N transfers only that shard to the worker.
- **Full fallback:** when the producer explicitly permits it, M2N reconstructs the complete logical tensor on each worker and passes it through `load_weights`.

The first model-native integration is unquantized `RoutedExperts`. It receives only locally owned expert shards and uses the existing loading path to place logical gate/up and down projections into fused `w13_weight` and `w2_weight` storage. This is an example of the generic destination contract, not a model-name rule in the transport engine.

Other fused parameters, quantized models, pipeline-parallel deployments, or modules without a semantic target use full fallback only when the source permits it. Otherwise initialization fails before the data-plane communicator is created.

`plan_summary()` reports how many parameters and logical bytes use each mode, plus the aggregate receiver payload implied by the agreed plan.

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

The trainer uses the stateful trainer engine. m2n needs each parameter's source layout, which the base `WeightSource` does not carry, so it takes an `M2NWeightSource`. `DTensorModuleSource` covers the common case: it reads each parameter's device mesh and placements and yields **local shards** — it never calls `full_tensor()`.

```python
from vllm.distributed.weight_transfer import (
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_source import DTensorModuleSource
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
    source=DTensorModuleSource(model, num_trainer_ranks),
)

# Each round, on every trainer rank:
engine.send_weights()
```

`send_weights()` drives the whole round — `start_weight_update`, `update_weights`, `finish_weight_update`. Every rank in the communicator must be inside the same reshard, so `update_weights` has to be in flight while the trainer sends; the engine runs it on a helper thread so the trainer loop does not have to open-code that.

Trainers with a custom producer (a Megatron export, MoE re-fusing) subclass
`M2NWeightSource` and supply layouts explicitly. A source may yield any
nccl4py-compatible CUDA buffer: a torch tensor, a CUDA Array Interface object,
or a raw device pointer. vLLM keeps buffer owners alive through stream
completion, using allocator stream tracking when the buffer provides it.
Sources must provide buffers on the communicator's current device; nccl4py
resolves their pointers but does not verify device ownership. Raw-pointer
sources are additionally responsible for pointer lifetime and matching their
declared dtype and local shape.

See [`examples/rl/rlhf_m2n.py`](../../../examples/rl/rlhf_m2n.py) for a runnable FSDP → TP example.

## Limitations

- vLLM validates tensor-rank and dtype support while agreeing on the complete
  plan; the installed M2N runtime validates them again for each reshard.
- Each parameter's source mesh must cover the same contiguous trainer-rank interval starting at 0, and be 1-D or 2-D.
- `Partial` placements are not supported.
- Models with known model-level checkpoint preprocessing use the full-tensor
  fallback for the affected module.
- Model-native rank-local targets currently require an unquantized PP1
  deployment. The first production consumer additionally requires TP1,
  expert parallelism, and uniform contiguous expert ownership.
- A reshard cannot be issued from inside a CUDA graph capture (weight updates run outside capture, so this only matters for custom callers).
- vLLM checks every parameter's shard overlap against conservative 16-source
  and 64-target capacities before committing the plan. The installed M2N
  runtime remains authoritative and validates each reshard again.
- Each communicator holds a cached staging pool (~256 MiB by default: 4 channels x 64 MiB).
