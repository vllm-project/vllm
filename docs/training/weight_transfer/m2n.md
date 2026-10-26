# NCCL M2N Engine

The `nccl_m2n` weight transfer engine uses [NCCL M2N](https://github.com/NVIDIA/nccl-extensions) to **reshard** weights between two disjoint meshes of GPUs — the trainer and the inference workers — in a single collective per parameter. Unlike the broadcast-based [NCCL engine](nccl.md), the trainer sends its local shard as-is: no full tensor is ever materialized on the training side, whatever its parallelism layout.

## When to Use NCCL M2N

- The trainer and the inference engine use **different parallelism layouts** (e.g. trainer FSDP/EP, inference TP) and gathering full tensors on the trainer is expensive
- Training and inference are on **separate GPUs**, possibly across nodes
- You want weight sync off the critical path of an async RL loop

## Requirements

This backend depends on the `nccl-extensions` Python package (and its `nccl4py` dependency), which vLLM does not install. Install both against a compatible NCCL 2.30.5 or newer runtime. The backend creates its data-plane communicator through `nccl4py` and passes that communicator directly to `nccl.m2n`.

The backend imports the runtime lazily, so vLLM is unaffected unless you select it.

For multiple trainer source groups, set `NCCL_LAUNCH_ORDER_IMPLICIT=1` before
starting every trainer and inference process. vLLM checks this collectively
before creating communicators. A single source group does not require it.

## How It Works

1. All trainers and inference workers join one `StatelessProcessGroup` for
   metadata. Trainers declare the ordered parameter manifest, each parameter's
   source layout, and its owning source group. Workers resolve direct or
   full-fallback destinations. All ranks agree on those placements and loading
   choices before creating any NCCL communicator.
2. vLLM creates one `nccl.core.Communicator` per source group. Its local ranks
   contain the ordered owning trainers followed by every inference worker.
   Trainers outside that group do not join its communicator. Every metadata
   rank processes the fixed communicator list in order, then one combined
   agreement reports initialization success or failure.
3. Each round uses one M2N reshard per parameter on its owning group's transfer
   route. Separate routes use separate CUDA streams and a deterministic bounded
   schedule. Direct destinations need no loader dependency; fallback tensors
   retain global manifest loading order. The worker must be inside
   `update_weights` while the trainers send; the trainer engine handles that
   concurrency.

The schedule can submit work from multiple source groups before waiting, while
retaining at most one pending full-fallback buffer per route. It preserves
parameter order within each route. It does not insert a barrier after every
parameter: fallback buffers are synchronized before `load_weights` reads them,
and route streams are drained at the end of the transfer.

### Destination resolution

A parameter is resharded **directly into the live vLLM parameter** when its checkpoint name maps 1:1 onto a model parameter whose shape is either identical to the checkpoint shape (replicated) or differs on exactly one dim by the tensor-parallel factor. That rank then only ever receives its own shard.

Anything else falls back to receiving the full tensor and letting `load_weights` do the sharding, exactly as the NCCL engine does for every parameter. The fallback covers:

- fused parameters (`qkv_proj`, `gate_up_proj`, MoE `w13`/`w2`), whose checkpoint names do not name a model parameter
- quantized models, where the weight loader does more than a sharded copy
- pipeline-parallel inference deployments (trainer pipeline parallelism is supported)

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

`send_weights()` drives one complete round — `start_weight_update`,
`update_weights`, `finish_weight_update`. Trainer rank zero coordinates worker
RPCs and distributes the dispatch and outcome to every trainer rank. Only the
trainers in a parameter's canonical source group send it. The update RPC runs
on a helper thread so worker receives and trainer sends can proceed together.

Trainers with a custom producer (a Megatron export, MoE re-fusing) subclass
`M2NWeightSource` and supply layouts explicitly. A source may yield any
nccl4py-compatible CUDA buffer: a torch tensor, a CUDA Array Interface object,
or a raw device pointer. vLLM keeps buffer owners alive through stream
completion, using allocator stream tracking when the buffer provides it.
Sources must provide buffers on the communicator's current device; nccl4py
resolves their pointers but does not verify device ownership. Raw-pointer
sources are additionally responsible for pointer lifetime and matching their
declared dtype and local shape.

### Partitioned trainer sources

A custom producer overrides `source_topology()` and `iter_group()` when
different trainer groups own different parameters. All trainer ranks report the
same complete metadata manifest, even when they hold only part of it.

```python
from vllm.distributed.weight_transfer.m2n_source import (
    M2NSourceTopology,
    M2NWeightSource,
)


class PipelineSource(M2NWeightSource):
    def __init__(self, metadata, local_shards):
        self._metadata = metadata
        self._local_shards = local_shards

    def metadata(self):
        # Same four entries on every trainer rank, each with its own layout.
        return list(self._metadata)

    def source_topology(self):
        return M2NSourceTopology(
            groups=((0, 1), (2, 3)),
            param_groups=(0, 0, 1, 1),
        )

    def held_names(self):
        return self._local_shards.keys()

    def __iter__(self):
        raise RuntimeError("use iter_group for a partitioned source")

    def iter_group(self, group_index):
        topology = self.source_topology()
        for meta, owner in zip(self._metadata, topology.param_groups):
            if owner == group_index:
                yield meta.name, self._local_shards[meta.name]
```

The contract requires non-empty, pairwise-disjoint source groups. A trainer rank
belongs to at most one group; a rank holding no transferable parameters may
belong to none. Every parameter selects exactly one canonical group, and every
declared group owns at least one parameter. Tied or replicated copies outside
the canonical group are not transferred.

Each parameter's mesh starts at communicator-local rank zero and covers its
group's ordered ranks. Different parameters within that group can use different
mesh factorizations and placements. For example, a two-rank group can use
`(1, 2)` for one parameter and `(2, 1)` for another. Group ranks need not be
contiguous globally. vLLM does not infer trainer ownership from names or from
the inference pipeline topology; the producer supplies it.

Returning `None` from `source_topology()` preserves the existing behavior:
every trainer belongs to one common group, and `iter_group(0)` delegates to the
source's normal iterator. Multiple routes can use pre-shared NCCL unique IDs
through `nccl_unique_ids_b64`, in route order; the singular field is valid only
for a single route.

`plan_summary()` reports communicator membership, the transfer schedule,
destination choices, and planned receiver tensor bytes. Those byte counts are
not measured network traffic.

### Update failures

Failures known to occur before trainer dispatch preserve communicators and
reset update-local loader state. Failures after dispatch or during transfer
abort all joined communicators without waiting on CUDA streams; callers must
reinitialize transport before sending another update. Some live parameters may
already have changed, so serving must remain paused until a complete update
succeeds.

A loader/finalizer failure after transfer completion resets loader state while
preserving transport. The trainer coordinates one replay of the complete update
from `start_weight_update()`. Healthy shutdown synchronizes outstanding work,
destroys the M2N handle, and aborts the route communicators.

See [`examples/rl/rlhf_m2n.py`](../../../examples/rl/rlhf_m2n.py) for a runnable FSDP → TP example.

## Limitations

- vLLM validates tensor-rank and dtype support while agreeing on the complete
  plan; the installed M2N runtime validates them again for each reshard.
- Each parameter's source mesh must be 1-D or 2-D and cover its assigned ordered
  source group, using communicator-local ranks starting at zero.
- All parameters currently use one configured inference-worker mesh. Per-parameter
  destination meshes over the same worker subset remain future work.
- `Partial` placements are not supported.
- Models with known model-level checkpoint preprocessing use the full-tensor
  fallback for the affected module.
- A reshard cannot be issued from inside a CUDA graph capture (weight updates run outside capture, so this only matters for custom callers).
- vLLM checks every parameter's shard overlap against conservative 16-source
  and 64-target capacities before committing the plan. The installed M2N
  runtime remains authoritative and validates each reshard again.
- Each communicator holds an M2N runtime staging pool (~256 MiB by default:
  4 channels x 64 MiB). Multiple routes increase this runtime allocation on
  workers joining more than one communicator.
