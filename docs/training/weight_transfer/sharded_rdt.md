# Sharded RDT Engine

The sharded RDT weight transfer engine moves weights point-to-point over [NIXL](https://github.com/ai-dynamo/nixl) using Ray Direct Transport (RDT), Ray's zero-copy tensor transport between actors. It is **pull-based**: the inference workers initiate every transfer and each one asks only for the *slice* it consumes under tensor and expert parallelism. A large MoE model therefore moves roughly `total_bytes / num_workers` per worker instead of the `total_bytes` a broadcast costs.

## When to Use Sharded RDT

- Very large models where broadcasting whole parameters is the bottleneck — typically an MoE served with expert parallelism, where each worker owns a small fraction of the experts
- Trainer and inference on **separate GPUs**, over a fabric NIXL supports (InfiniBand, RoCE, EFA)
- Trainers that are themselves sharded, including **pipeline-parallel** ones where a rank holds only part of the model

Requirements:

- `distributed_executor_backend="ray"` — the workers must be Ray actors
- Ray `>= 2.56.0` on both the trainer and the workers
- `nixl` installed in the environment shared by trainer and workers
- Weight loaders that stay inside the supported op set (below)
- EPLB (`enable_eplb=true`) is rejected: it rearranges experts at runtime, which invalidates the recorded plan

## How It Works

### Slices are tracked through vLLM's own weight loaders

A weight loader normally receives a full HF-format tensor and slices out the part this worker needs. Instead of sending it one, the engine hands it a `FakeRDTTensor`: a zero-storage tensor that answers `.shape` / `.dtype` / `.size()` but holds no data. Every view or slice op the loader calls returns a new fake with that op appended to a recorded chain, and `copy_` is the sink that ends it.

That chain is the wire format. `("model.layers.0.w", (("narrow", (0, 512, 512), ()), ("t", (), ())))` tells the trainer: take this tensor, `narrow` it, transpose it, send the result. The trainer replays it with `getattr(tensor, op)(*args, **kwargs)`.

Discovery is expensive, so it happens **once**, at `init_transfer_engine`, as a dry run over `model.load_weights` with every parameter on meta. Nothing is transferred; the engine just records, per leaf module, which slice feeds which destination region. Every later sync is pure replay.

Anything a loader does that needs real data — arithmetic, `.to()`, `.float()`, `.item()`, `.data`, bool-mask indexing — falls outside the allowlist and raises at init. That is deliberate: failing loudly during setup beats silently transferring the wrong bytes. `SUPPORTED_OPS` in `sharded_rdt_common.py` is the single table both sides derive from, so the recorder and the replayer cannot drift.

### Received slices land directly in the layerwise reload buffers

The engine drives [layerwise reload](../layerwise.md) itself, in `start_weight_update` / `finish_weight_update`. Because the dry run already recorded each destination as an `as_strided` region of its parameter, an arriving slice is copied straight into the layer being reloaded — no full HF tensor is ever materialized on the worker, and no second pass over `load_weights` runs. Each layer is quantized and copied into its persistent kernel storage as soon as its last slice lands.

### Gathers and pulls are pipelined, which is what `gather_lookahead` bounds

The trainer usually cannot serve its parameters as they sit: FSDP shards them, and even an EP-split trainer has to assemble a whole expert. So each sync still runs gather collectives — but a layer at a time, not a model at a time.

**A gather group is one decoder layer.** The parameter list is keyed on the outermost index segment of each name, which leaves runs of un-indexed names — the embeddings before the first layer, the final norm and `lm_head` after the last — as groups of their own:

```text
group 0     model.embed_tokens.weight
group 1     model.layers.0.*          <- one decoder layer
group 2     model.layers.1.*
...
group N+1   model.norm.weight, lm_head.weight
```

The layer is the unit of everything that follows: the trainer gathers a layer, publishes it (immediately pullable), and moves on to the next while the consumers pull the one it just published. Once every consumer has signalled that it is done with a layer, the trainer drops it and gains a credit to gather another. `gather_lookahead` is how far ahead of the consumers that loop may run, so at most `gather_lookahead + 1` layers are resident on the trainer at a time. The default of 1 keeps the next layer gathered and pullable while the current one is being pulled — enough to hide the handoff without doubling trainer memory. Raise it only if one layer's gather is slower than its pulls.

Because a layer is also the unit the consumers free and the unit the receive buffers are sized against, it is what keeps memory bounded on both sides: without it the whole model would be one transfer, and both sides would have to hold their full share of it at once. Keying on the index rather than a fixed `model.layers.` prefix is what makes that hold across naming conventions — a VLM's `model.language_model.layers.`, GPT-2's `transformer.h.`, a vision tower's `visual.blocks.`. Sources can control the partition; see [gather groups](base.md#gather-groups).

### Ownership

A trainer rank need not hold the whole model. Each one declares what it holds through [`WeightSource.held_names()`](base.md#held_names-partial-ownership), the fleet all-gathers those declarations at `trainer_init`, and the consumers route each pull to a rank that actually holds the name. Pipeline stages, expert parallelism, and combinations of the two are all the same declaration. Consumers spread their pulls across the ranks that hold a name, so no single trainer NIC becomes the bottleneck.

## Inference Side

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="sharded_rdt"),
    distributed_executor_backend="ray",
)
```

```bash
vllm serve my-model \
  --distributed-executor-backend ray \
  --weight-transfer-config '{"backend": "sharded_rdt"}'
```

Everything else — which producers exist, how the model splits into layer groups, the ownership table — arrives from the trainer at the init handshake.

!!! warning "Size the receive buffers before choosing `gpu_memory_utilization`"
    Each worker holds `num_rdt_buffers` receive buffers, each large enough for the biggest single slice batch it pulls. Like NCCL and NIXL internals, they do **not** count against `gpu_memory_utilization`, so a fraction that leaves no headroom OOMs at the first sync even though the engine came up healthy. The buffer size is driven by the largest atomic slice — for an untied vocab matrix on a worker that holds it unsliced, that is the whole embedding.

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

`trainer_init` and `send_weights` run on **every** trainer rank: each one owns a serve actor and takes part in the gathers, while only rank 0 drives the inference-side handshake. Any [`VLLMWeightSyncClient`](base.md#vllmweightsyncclient) works.

To adapt a trainer that is not a plain `nn.Module` — a Megatron export, a raw sharded checkpoint — subclass [`WeightSource`](base.md#weightsource).

### `ShardedRDTTrainerInitInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `rank` | — | Keyword-only. This trainer rank; **0 is the sender** |
| `num_consumers` | — | Inference workers across the whole fleet (TP × DP) |
| `trainer_actor_namespace` | `None` | Ray namespace for the serve actors; the workers resolve them by name here |
| `num_rdt_buffers` | 2 | Ring depth on both sides |
| `buffer_presize_gb` | 0.0 | Pre-size each buffer slot, in GiB. Set it to cover the largest atomic slice |
| `gather_lookahead` | 1 | Gathered-but-unfreed layers the gather loop runs ahead by |
| `stall_timeout_s` | 300.0 | Seconds without progress before the sync fails. A liveness backstop for a consumer that dies mid-sync, not a latency target |

## Examples

- [Small MoE on 4 GPUs](../../../examples/rl/rlhf_sharded_rdt_small_ep.py) — 2 FSDP2 trainer ranks → 2 vLLM DP ranks with expert parallelism, one node. It pairs a trainer fleet with a separate inference fleet, the only arrangement this backend supports, and asserts that the sync moved the weights and that a second sync leaves generation unchanged — so it runs unattended in CI

It keeps the trainer deliberately small — just enough FSDP2 to make the weights real — so the file stays about the weight sync rather than about the trainer. For a full RL trainer, SkyRL integrates this backend with Megatron (PP-local gathering and expert-stack fusion for MoE) alongside FSDP: [NovaSky-AI/SkyRL#1753](https://github.com/NovaSky-AI/SkyRL/pull/1753).
