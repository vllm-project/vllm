# NCCL M2N Engine

The `nccl_m2n` weight transfer engine uses [NCCL M2N](https://github.com/NVIDIA/nccl-extensions) to **reshard** weights between two disjoint meshes of GPUs — the trainer and the inference workers — in a single collective per parameter. Unlike the broadcast-based [NCCL engine](nccl.md), the trainer sends its local shard as-is: no full tensor is ever materialized on the training side, whatever its parallelism layout.

## When to Use NCCL M2N

- The trainer and the inference engine use **different parallelism layouts** (e.g. trainer FSDP/EP, inference TP) and gathering full tensors on the trainer is expensive
- Training and inference are on **separate GPUs**, possibly across nodes
- You want weight sync off the critical path of an async RL loop

## Requirements

This backend depends on the `nccl-extensions` Python package (and its `nccl4py` dependency), which vLLM does not install. It needs NCCL 2.30.5 or newer, and `libnccl_m2n.so` is linked against a specific NCCL build, so vLLM must load the same library:

```bash
export VLLM_NCCL_SO_PATH=/path/to/the/same/libnccl.so
```

The backend imports the runtime lazily, so vLLM is unaffected unless you select it.

## How It Works

1. The trainer ranks and all inference workers join **one** NCCL communicator via `StatelessProcessGroup`. Trainer ranks occupy `[0, T)` and workers `[T, T + N)` — m2n meshes are contiguous rank intervals, which is exactly what vLLM's existing `rank_offset` convention produces.
2. At the init handshake the trainer ships the full transfer plan: both meshes — one per side, shared by every parameter — then each parameter's name, dtype, shape and placement. Declaring the meshes rather than inferring them is what lets the two sides describe every reshard identically.
3. Each round, both sides issue one `reshard` per parameter in the same order. Every rank in the communicator participates in every reshard, so — as with the NCCL engine — the worker must be inside `update_weights` while the trainer sends. The trainer engine handles that concurrency itself.

Each worker receives the full tensor and hands it to `load_weights`, which shards it — the same thing the NCCL engine does. The gain is entirely on the trainer side: an FSDP or expert-parallel trainer sends its local shards, where broadcast would force an all-gather per parameter first. Resharding directly into each worker's own shard is a follow-up.

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

Trainers with a custom producer (a Megatron export, MoE re-fusing) subclass `M2NWeightSource` and supply layouts explicitly.

See [`examples/rl/rlhf_m2n.py`](../../../examples/rl/rlhf_m2n.py) for a runnable FSDP → TP example.

## Limitations

- Tensor rank 1..3; 4-D parameters are rejected.
- dtypes: int8/uint8, fp8 e4m3/e5m2, fp16, bf16, int32/uint32, fp32, int64/uint64, fp64. No fp4.
- The trainer's device mesh must cover a contiguous rank interval starting at 0, and be 1-D or 2-D.
- `Partial` placements are not supported.
- A reshard cannot be issued from inside a CUDA graph capture (weight updates run outside capture, so this only matters for custom callers).
- The reshard plan is capped per *shard*, not per mesh: at most 16 source shards may feed one destination shard, and one source shard may feed at most 64 destination shards (`MAX_SOURCES` / `MAX_TARGETS` in the m2n build; raising either needs a rebuild).
- Because every worker here receives the whole tensor, the destination is a single shard fed by every source shard — so a trainer that **sharded** its parameters is limited to 16 ranks. A replicated trainer is unaffected, and the cap loosens once the destination is sharded too.
- Each communicator holds a cached staging pool (~256 MiB by default: 4 channels x 64 MiB).
