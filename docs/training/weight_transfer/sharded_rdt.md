# Sharded RDT Engine

The sharded RDT weight transfer engine moves weights point-to-point over [NIXL](https://github.com/ai-dynamo/nixl) using Ray Direct Transport (RDT), Ray's tensor transport between actors. It is **pull-based**: each inference worker reads from the trainer only the parts of each checkpoint tensor it holds under tensor and expert parallelism. A large MoE model therefore moves roughly `total_bytes / num_workers` per worker instead of the `total_bytes` a broadcast costs.

## When to Use Sharded RDT

- Very large models where broadcasting whole parameters is the bottleneck, typically an MoE served with expert parallelism, where each worker holds a small fraction of the experts
- Trainer and inference on **separate GPUs**, over a fabric NIXL supports (InfiniBand, RoCE, EFA)
- Trainers that are themselves sharded, including **pipeline-parallel** ones where a rank holds only part of the model

Requirements:

- [`VLLM_RELOAD_MODE=direct`](../direct_reload.md) on the inference server; the engine refuses to start under layerwise reload
- `distributed_executor_backend="ray"`: the workers must be Ray actors
- Ray `>= 2.60.0` on both the trainer and the workers (the NIXL sender memory pool)
- `nixl` installed in the environment shared by trainer and workers

## How It Works

### The load plan says what each worker needs

At the first `start_weight_update`, each worker builds the load plan (`plan_loads`) for the trainer's metadata, on the tensors [direct reload](../direct_reload.md) has just viewed back to their checkpoint layout. For every checkpoint name, the plan gives the boxes of the tensor this worker needs (`get_boxes`) and the live tensors they are written to (`get_dest`). A name the loaders transform rather than copy is a *fallback*: the worker needs it in full, through `load_weights`. A name another worker owns (another expert, another pipeline stage) needs nothing.

Each box is a *piece*. The worker routes every piece to one trainer rank that owns the name, and sorts the pieces into two kinds of pulls:

- **Direct**: the destination is contiguous with the box's shape and dtype. NIXL writes the bytes straight into model memory, so the destination storages are registered with NIXL once, at plan time.
- **Staged**: everything else. Destinations that need a strided copy (a transposed or interleaved layout, two boxes of one tensor under expert TP) and fallbacks. Ray receives these into a buffer it allocates, and the worker copies them into place or passes them to `load_weights`.

The plan is reused by every later sync. Re-initializing the engine (a new trainer, new metadata) drops it, and the next `start_weight_update` plans again.

### Gather units and buckets

The trainer usually cannot serve its parameters as they sit: FSDP shards them, and an EP trainer holds only its experts. So each sync still gathers, a **gather unit** at a time. The metadata, in order, is cut into consecutive runs of at most `buffer_bytes` full-tensor bytes; a larger tensor is a unit of its own. Units are the same on every rank and every worker, so ranks that gather collectively reach the same collectives in the same order.

A worker cuts its pieces into **buckets** (one pull each) of at most `buffer_bytes`, per trainer rank and per unit, so a pull never needs two units at once. At plan time it sends each trainer rank the list of buckets it will pull from it, in order; afterwards a pull carries only the bucket's index.

### A sync

```text
trainer rank (each)                     inference worker (each)
-------------------                     -----------------------
begin_sync                              start_weight_update: start_reload
for unit in its units:                  update_weights:
    wait while > lookahead unreleased       keep num_buffers pulls in flight:
    gather (iter_full), publish                 bucket(i) -> NIXL read
                                                direct: lands in the model
bucket(i): wait for the unit,                   staged: copy_ / load_weights
    return the boxes; Ray copies                drop the ref -> pool block freed
    them into the NIXL pool
release a unit once every live worker   finish_weight_update: finish_reload
    has freed every bucket reading it
end_sync
```

A trainer rank runs at most `gather_lookahead` units ahead of the units its workers are still reading. Workers pull in unit order within each group of co-owning trainer ranks, and pull from independent groups (pipeline stages) side by side, so no rank waits on a unit its workers will pull later.

Each trainer rank serves out of one NIXL memory pool registered at startup, sized for `num_buffers` buckets for every worker that may pull from it, plus one bucket of slack. A rank admits a pull only while the buckets it has served and Ray has not yet freed fit in the pool, so a full pool slows pulls down rather than failing them.

**Every inference deployment must call `start_weight_update` before any calls `update_weights`.** The first `start_weight_update` registers the worker's buckets with every trainer rank, and a rank releases a unit only once every live worker has registered.

### Ownership

A trainer rank need not hold the whole model. The source declares each parameter's owners in [`ParamMeta.owners`](base.md#parammeta-ownership-and-layout), the same on every rank, so the engine builds the ownership table from metadata alone. Each rank gathers only the names it owns, and the workers route each piece to a rank that owns the name. Pipeline stages, expert parallelism, and combinations of the two are all the same declaration. Workers spread their pulls across a name's owners (rotating per unit), so no single trainer NIC becomes the bottleneck. The server-name all-gather at `trainer_init` also carries a digest of each rank's metadata and `buffer_bytes`, so a rank that disagrees fails at init.

## Inference Side

```bash
VLLM_RELOAD_MODE=direct vllm serve my-model \
  --distributed-executor-backend ray \
  --weight-transfer-config '{"backend": "sharded_rdt"}'
```

Everything else (the producers, the metadata, the ownership table, `buffer_bytes` and `num_buffers`) arrives from the trainer at the init handshake.

!!! warning "Leave headroom for staged pulls"
    Staged pulls land in buffers Ray allocates on the worker's GPU: up to `num_buffers` buckets of at most `buffer_bytes` (or one tensor, if a fallback is larger) at a time. They do not count against `gpu_memory_utilization`, so leave room for them (see [GPU Memory Utilization](../../configuration/optimization.md#gpu-memory-utilization)). Direct pulls need no buffer.

## Trainer Side

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=ShardedRDTTrainerInitInfo(
        rank=rank,                                # rank 0 is the sender
        num_consumers=8,                          # inference workers, fleet-wide
        trainer_actor_namespace="my_namespace",   # must be visible to the workers
    ),
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
    source=ModuleSource(model),
)

engine.send_weights()   # once per sync, on every trainer rank
```

`trainer_init` and `send_weights` run on **every** trainer rank: each one owns a producer actor and takes part in the gathers, while only rank 0 drives the inference-side calls. Any [`VLLMWeightSyncClient`](base.md#vllmweightsyncclient) works.

To adapt a trainer that is not a plain `nn.Module` (a Megatron export, a raw sharded checkpoint), subclass [`WeightSource`](base.md#weightsource).

### `ShardedRDTTrainerInitInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `rank` | — | Keyword-only. This trainer rank; **0 is the sender**. Must equal the default process group rank, since `owners` are matched against it |
| `num_consumers` | — | Inference workers across the whole fleet (TP × PP × DP) |
| `workers_per_replica` | 0 | Workers per inference deployment, when several independent deployments share the trainer; 0 means one |
| `trainer_actor_namespace` | `None` | Ray namespace for the producer actors; the workers resolve them by name there |
| `buffer_bytes` | 1 GiB | Most bytes one pull moves, and the size of a gather unit |
| `num_buffers` | 2 | Pulls each worker keeps in flight |
| `gather_lookahead` | 1 | Units the gather loop runs ahead of the workers; at most `gather_lookahead + 1` units are resident on a trainer rank |
| `stall_timeout_s` | 300.0 | Seconds without progress before the sync fails. A liveness backstop for a worker that dies mid-sync, not a latency target |

Each trainer rank's GPU holds its NIXL pool, `(num_buffers × workers pulling from it + 1) × max(buffer_bytes, largest tensor)` bytes, and up to `gather_lookahead + 1` gathered units.

## Limitations

- **Separate fleets only.** The trainer and the inference workers run on different GPUs; colocated setups need [IPC](ipc.md).
- **Direct reload only**, with its [refused configurations](../direct_reload.md#refused-configurations); the load plan also refuses EPLB and online quantization.
- **GPU staging only.** Staged pulls land in GPU memory.
- Ray's RDT APIs used here are alpha. With many trainer ranks, keep Ray's NIXL remote-agent cache (`RAY_NIXL_REMOTE_AGENT_CACHE_MAXSIZE`, default 1000) at least the number of trainer ranks.

## Examples

- [Small MoE on 4 GPUs](../../../examples/rl/rlhf_sharded_rdt_small_ep.py): 2 FSDP2 trainer ranks to 2 vLLM DP ranks with expert parallelism, one node. It checks that the sync moved the weights and that a second sync leaves generation unchanged, so it runs unattended in CI. `RDT_BUFFER_BYTES` and `RDT_NUM_BUFFERS` set the transfer settings.

For a full RL trainer, SkyRL integrates this backend with Megatron and FSDP: [NovaSky-AI/SkyRL#1753](https://github.com/NovaSky-AI/SkyRL/pull/1753).
