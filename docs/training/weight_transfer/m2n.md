# NCCL M2N Engine

The `nccl_m2n` weight transfer engine uses [NCCL M2N](https://github.com/NVIDIA/nccl-extensions) to **reshard** weights between two disjoint meshes of GPUs — the trainer and the inference workers — in a single collective per parameter. Unlike the broadcast-based [NCCL engine](nccl.md), the trainer sends its local shard as-is: no full tensor is ever materialized on the training side, whatever its parallelism layout.

## When to Use NCCL M2N

- The trainer and the inference engine use **different parallelism layouts** (e.g. trainer FSDP/EP, inference TP) and gathering full tensors on the trainer is expensive
- Training and inference are on **separate GPUs**, possibly across nodes
- You want weight sync off the critical path of an async RL loop

## Requirements

This backend depends on the `nccl-extensions` Python package (and its `nccl4py` dependency), which vLLM does not install. Install both against a compatible NCCL 2.30.5 or newer runtime. The backend creates its data-plane communicator through `nccl4py` and passes that communicator directly to `nccl.m2n`.

The backend imports the runtime lazily, so vLLM is unaffected unless you select it.

When a source topology needs more than one transfer communicator, set
`NCCL_LAUNCH_ORDER_IMPLICIT=1` before starting every trainer and inference
process. The communicators use independent CUDA streams; NCCL requires this
mode for deterministic cross-communicator launch ordering. vLLM checks the
setting collectively before communicator initialization. The legacy
single-source-group transfer uses one communicator and does not require it.

## How It Works

1. A `StatelessProcessGroup` exchanges metadata and the NCCL unique IDs, one
   per transfer communicator. During this metadata-only phase, trainers agree
   on each parameter's source layout and workers resolve their rank-local
   destinations. Every participant agrees on the complete ordered source and
   destination plans before entering NCCL initialization. The final plan hash
   covers parameter ownership, source and destination layouts, rank-local
   loading decisions, and ordered transfer-communicator membership.
2. After plan agreement, vLLM creates one transfer communicator per source
   group. Each communicator contains that ordered trainer group followed by all
   inference workers. A parameter is sent on exactly one communicator; trainer
   ranks that do not own that parameter never join its data-plane communicator.
   Every metadata rank processes the fixed communicator list in order. Only
   listed members initialize each NCCL communicator; nonmembers still process
   its UID bootstrap. All ranks then exchange one combined
   startup result.
3. Each round, both sides replay the same bounded transfer schedule. Parameters
   keep their manifest order within a communicator, while independent
   communicators use separate CUDA streams. Every member enters its reshards in
   the same order, so the worker must be inside `update_weights` while the
   trainer sends. The trainer engine handles that concurrency itself.

   Each launch wave submits the next parameter from every unblocked source
   group before waiting. Staged load callbacks retain only their declared
   semantic order, while full-fallback inputs retain global manifest order.
   This lets pipeline source groups advance together without allocating one
   receive buffer per parameter. Before a completed semantic group releases
   its staging buffers, every transfer stream is ordered after the model
   loader's current CUDA stream so asynchronous loads cannot race with reuse.

### Destination resolution

A parameter uses one collectively agreed destination mode:

- **In place:** M2N writes a validated exact-name live vLLM parameter.
- **Rank-local staging:** a model module declares the local shard it owns and a callback that installs it into model-native storage. M2N transfers only that shard to the worker.
- **Full fallback:** when the producer explicitly permits it, M2N reconstructs the complete logical tensor on each worker and passes it through `load_weights`.

The first model-native integration is unquantized `RoutedExperts`. It receives only locally owned expert shards and uses the existing loading path to place logical gate/up and down projections into fused `w13_weight` and `w2_weight` storage. This is an example of the generic destination contract, not a model-name rule in the transport engine.

Other fused parameters, quantized models, inference pipeline-parallel
deployments, or modules without a semantic target use full fallback only when
the source permits it. Otherwise initialization fails before the data-plane
communicator is created.

`plan_summary()` reports the agreed communicator membership and ordered transfer
waves using source-manifest parameter indices. It also reports how many
parameters and logical bytes use each destination mode, and the aggregate
receiver payload implied by the plan.

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

# Called on every trainer rank; rank 0 coordinates the round.
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

`send_weights()` drives the whole round — `start_weight_update`,
`update_weights`, `finish_weight_update`. Rank 0 coordinates the worker RPCs
and trainer round control, while every source-group member submits its local
shard. Every rank in a transfer communicator must be inside the same reshard, so
`update_weights` has to be in flight while those trainer ranks send; the engine
runs it on a helper thread so the trainer loop does not have to open-code that.

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

`M2NWeightSource.source_topology()` separates parameter ownership from each
parameter's mesh factorization. It returns an `M2NSourceTopology` with:

- `groups`: ordered tuples of trainer rank IDs from the global metadata group.
  Their order defines the source-mesh coordinates and becomes the
  communicator-local source-rank order.
- `param_groups`: one group index for every parameter in the global metadata
  manifest.

Every trainer rank still reports the same complete metadata manifest. A trainer
rank assigned to a source group uses `iter_group(group_index)` to yield only
that group's parameters, in their subsequence of global manifest order. A
parameter belongs to exactly one canonical group. Source groups are non-empty
and pairwise disjoint, so a trainer rank belongs to at most one group. Each
parameter may use its own mesh and placements over its assigned group's ordered
ranks, and the source-mesh size must equal that group size. Every declared group
must own at least one parameter.
vLLM does not infer trainer ownership from parameter names or from the
inference pipeline topology.

For example, a two-stage trainer whose source meshes each contain two ranks can
declare:

```python
from vllm.distributed.weight_transfer.m2n_source import M2NSourceTopology

topology = M2NSourceTopology(
    groups=((0, 1), (2, 3)),
    # One entry per parameter in metadata() order.
    param_groups=(0, 0, 1, 1),
)
```

This creates two data-plane communicators: `(0, 1, workers...)` and
`(2, 3, workers...)`. It does not gather either stage onto trainer rank zero.
Shared or tied parameters choose one canonical exporting group; extra local
copies on other stages are not transferred.

Returning `None` from `source_topology()` preserves the legacy
single-source-group behavior: all trainer ranks form one source group, and
`iter_group(0)` delegates to the source's normal iterator.

See [`examples/rl/rlhf_m2n.py`](../../../examples/rl/rlhf_m2n.py) for a runnable FSDP → TP example.

## Limitations

- All parameters currently target the same inference-worker mesh. The source
  and transfer-route contracts preserve room for per-parameter destination
  meshes over the same worker subset, but per-parameter destination-mesh
  selection is not implemented yet.
- vLLM validates tensor-rank and dtype support while agreeing on the complete
  plan; the installed M2N runtime validates them again for each reshard.
- Each source mesh is 1-D or 2-D and uses communicator-local ranks starting at
  zero. Its size must match its parameter's ordered source group; source groups
  need not be contiguous in the global trainer-rank namespace.
- `Partial` placements are not supported.
- Models with known model-level checkpoint preprocessing use the full-tensor
  fallback for the affected module.
- Model-native rank-local targets currently require an unquantized inference
  deployment with PP=1. The first production consumer additionally requires TP1,
  expert parallelism, and uniform contiguous expert ownership.
- A reshard cannot be issued from inside a CUDA graph capture (weight updates run outside capture, so this only matters for custom callers).
- vLLM checks every parameter's shard overlap against conservative 16-source
  and 64-target capacities before committing the plan. The installed M2N
  runtime remains authoritative and validates each reshard again.
- Each data-plane communicator holds a cached staging pool (~256 MiB by
  default: 4 channels x 64 MiB), so trainer partitioning increases aggregate
  M2N staging by the number of source-group communicators a process joins.
- A failure known to occur before trainer dispatch preserves the transport and
  discards only update-local staging. A failure after dispatch or during
  transfer aborts every joined transfer communicator without first waiting on
  CUDA streams; a subset of in-place destinations may already have changed and
  must not be served. A loader/finalizer failure after transfer completion
  preserves the transport, resets the worker's layerwise-loader and staging
  state, and makes every trainer rank replay the complete update once from
  `start_weight_update()`.
- Collective preflight reports communicator-init calls that return an error,
  but no control-plane protocol can recover a rank that is terminated or stuck
  inside NCCL initialization. Deployment-level timeouts remain required.
