# Weight Transfer

vLLM provides a pluggable weight transfer system for synchronizing model weights from a training process to the inference engine during reinforcement learning (RL) workflows. This is essential for RLHF, GRPO, and other online RL methods where the policy model is iteratively updated during training and the updated weights must be reflected in the inference engine for rollout generation.

## Architecture

Weight transfer has **two engines, one per process**, and they are symmetric:

| | Trainer process | Inference workers |
| --- | --- | --- |
| Class | `TrainerWeightTransferEngine` | `WeightTransferEngine` |
| Built by | `WeightTransferTrainerFactory.trainer_init(...)` | vLLM, from `WeightTransferConfig` |
| Driven by | `send_weights()` | the four-phase protocol below |
| Holds | communicators, transfer plans, wire params | communicators, the target model |

The trainer engine is stateful: it owns its communicators and wire params, pulls
weights from a [`WeightSource`](base.md#weightsource), and drives the inference
side through a [`VLLMWeightSyncClient`](base.md#vllmweightsyncclient). Trainer
code never has to know the transport, and never has to thread transfer state
back in on every round — one `send_weights()` call per sync.

Under the hood every round is the same **four-phase protocol**, which the trainer
engine drives on your behalf:

1. **Initialization** (`init_weight_transfer_engine`): establishes the communication channel between the trainer and inference workers. Called once, from `trainer_init`, before the training loop begins.
2. **Start** (`start_weight_update`): prepares the inference engine for a weight update.
3. **Weight Update** (`update_weights`): transfers updated weights. May be called one or more times (e.g. for chunked transfers).
4. **Finish** (`finish_weight_update`): finalizes the update (e.g. runs post-processing for checkpoint-format weights). Called once after all weights have been transferred.

## Available Backends

The backend picks the transport; the protocol and the API stay the same.

| Backend | Transport | Use Case | What a sync moves |
| ------- | --------- | -------- | ----------------- |
| [NCCL](nccl.md) | NCCL broadcast | Separate GPUs for training and inference | One broadcast over NVLink or InfiniBand; `packed=True` batches small tensors |
| [IPC](ipc.md) | CUDA IPC handles | Colocated training and inference on same GPU | Nothing over a data plane: workers read the trainer's tensors through the handles |
| [sparse_nccl](nccl.md#sparse-nccl) | NCCL broadcast | Checkpoint-coordinate sparse weight patches | Only the changed coordinates |
| [sharded_rdt](sharded_rdt.md) | NIXL / Ray Direct Transport (pull-based) | Very large models where each worker needs only its own slice (MoE with expert parallelism) | Each worker fetches only its own slices |
| [nccl_m2n](m2n.md) | NCCL M2N reshard | Trainer and inference use different sharding layouts | Reshard on the wire, without materializing full tensors |
| [ModelExpress](modelexpress.md) | ModelExpress version-based transfer | Install immutable published weight versions, including object-storage checkpoints and deltas | Immutable versions installed by id, optionally as deltas |

For IPC, NCCL, sparse_nccl, sharded_rdt, and nccl_m2n a sync costs one transfer
of the updated weights over the chosen link plus the in-place load on each
worker; there is no checkpoint write or read, no server restart, and no CUDA
graph recapture. Checkpoint-format engines (such as IPC and NCCL) run the
[layerwise reload](../layerwise.md) path on the inference side, which fuses,
(online-)quantizes, and shards each layer as it lands. When training and
inference are colocated, IPC is the cheapest option because no bytes cross a
device boundary; NCCL is the default once they are split across GPUs.

## Quickstart

### Inference Side

The inference side takes only a backend name. Everything else about the transfer
is decided by the trainer and shipped over at the init handshake.

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    # Other backends: "ipc", "sparse_nccl", "sharded_rdt", "nccl_m2n".
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)
```

Or, for online serving:

```bash
vllm serve my-model \
    --weight-transfer-config '{"backend": "nccl"}'
```

### Trainer Side

Build the engine once, then call `send_weights()` once per sync:

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

# Once, before the training loop.
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=NCCLTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,   # trainer + all inference workers
        rank=0,                  # this trainer rank; rank 0 is the sender
        packed=True,
    ),
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
    source=ModuleSource(model),
)

# Once per weight sync.
for step in range(num_steps):
    train_one_step(model)
    engine.send_weights()
```

`send_weights()` drives start → update → finish on the inference side *and* the
data-plane transfer, including any concurrency the backend needs (NCCL, for
example, must run the workers' `update_weights` at the same time as the
trainer's broadcast, since both rendezvous inside the same NCCL calls).

There is no `backend=` argument on the trainer side: each `TrainerInitInfo`
subclass declares its own `backend`, and the factory dispatches on it.

### Reference Examples

- [RLHF with IPC weight syncing (`vllm serve`, HTTP)](../../../examples/rl/rlhf_http_ipc.py) — start here when training and inference share a GPU
- [RLHF with NCCL weight syncing (`vllm serve`, HTTP)](../../../examples/rl/rlhf_http_nccl.py) — trainer on one GPU, tensor-parallel fp8 server on others
- [Async RLHF with mid-flight pause and resume](../../../examples/rl/rlhf_async_new_apis.py) — generation overlaps training, with weight sync between rollouts

Each `vllm serve` example launches and tears down its own server; the async
example runs the engine in-process. The backend pages link the remaining
examples (FSDP/EP, multi-source, sparse, sharded RDT, M2N).

## Updating Weights While Serving

Weight transfer updates weights **in place**: every worker loads the new tensors
into the storage the model already owns (see [Layerwise Reloading](../layerwise.md)),
so compiled artifacts are reused rather than rebuilt. CUDA graphs are replayed
against the new values and are not recaptured, and no startup-time work (initial
weight load, graph capture) is repeated. A sync therefore costs the transfer
plus the in-place load, not a restart and warmup. When the new weights are
already a checkpoint on disk, the mid-level `reload_weights` API uses the same
in-place path; weight transfer exists to move tensors straight from the training
process, with no checkpoint round trip.

The protocol does not pause the engine for you. When requests may be in flight:

- **Pause first.** Call `pause_generation()` before `/start_weight_update` and
  `resume_generation()` after `/finish_weight_update`. `mode="keep"` freezes
  in-flight requests so they continue on the new weights; `"wait"` and `"abort"`
  make the update a generation boundary. See
  [The Pause and Resume API](../async_rl.md#the-pause-and-resume-api).
- **Invalidate stale KV.** Entries computed under the old weights are stale
  after the update. `pause_generation(clear_cache=True)` clears the KV, prefix,
  and connector caches. Block hashes do not cover weights, so if you keep the
  connector cache, give requests a per-version `cache_salt`.
- **Label the update.** Pass `weight_version` to `finish_weight_update` (or call
  `/update_weight_version`) so outputs carry `weight_versions` spans for tokens
  sampled before and after the update. See
  [Weight Versions in Outputs](../async_rl.md#weight-versions-in-outputs).
- **Data parallelism.** With the internal load balancer one pause/resume covers
  all DP ranks; behind an external load balancer, pause, label, and resume every
  instance. See [the DP note in the Async RL guide](../async_rl.md).

The [Async RL](../async_rl.md) guide covers the full async loop that combines
these pieces.

## Where Each Setting Lives

The trainer is the single source of truth for anything the two sides must agree
on. This is the main rule to internalize:

| Setting | Lives on | Notes |
| ------- | -------- | ----- |
| Backend, inference side | `WeightTransferConfig(backend=...)` | A plain string selector — that is all this config holds |
| Backend, trainer side | `init_info.backend` | A `ClassVar` on each init info; never passed by hand |
| Wire params (`packed`, buffer sizes) | the backend's `TrainerInitInfo` | The trainer propagates them to the worker during `trainer_init`, so the two sides *cannot* disagree |
| Weights to send | a [`WeightSource`](base.md#weightsource) | Usually written for your framework; `ModuleSource(model)` covers modules every rank holds whole (plain, FSDP, DTensor) |
| Per-round payload | the backend's update info | Built by the engine (from `source.metadata`, or from a delta backend's per-round payload); you never construct it |

!!! note
    Wire params are deliberately *not* on `WeightTransferConfig` and *not* on the
    per-round update info. Because the trainer ships them at init and the worker
    reads what it was told, a mismatched `packed` flag between the two sides is
    unrepresentable rather than merely discouraged.

## Multi-Rank Trainers

For sharded trainers (FSDP, TP/PP/EP), **every trainer rank builds an engine and
calls `send_weights()`**. Rank 0 is the sender: only it holds a communicator,
talks to the client, and puts bytes on the wire. Non-sender ranks still iterate
the `WeightSource` over the parameters they own, because materializing a
parameter is usually itself a collective among its owners (an FSDP
`full_tensor()` all-gather, a Megatron export) that would deadlock if some of
them skipped it. Which ranks own which parameter is declared in
[`ParamMeta.owners`](base.md#parammeta-ownership-and-layout).
Only [sharded RDT](sharded_rdt.md) and [NCCL](nccl.md#broadcasting-ranks)
accept ranks that own different parameters, such as pipeline stages.
Pipeline-parallel support for IPC is in progress; until then it, like NCCL M2N,
needs every parameter owned by rank 0 or by every rank (see each backend's
Limitations).

Pass each process's own `rank` on the init info. It is explicit rather than read
from a global process group, which is ambiguous once several groups (FSDP / TP /
PP / EP) exist.

```python
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=NCCLTrainerInitInfo(..., rank=torch.distributed.get_rank()),
    client=client,
    source=ModuleSource(model),
)
engine.send_weights()  # called on every rank
```

## Verifying a New Model Without a Trainer

To check that weight updates work for a new model before wiring up an RL
framework, the [checkpoint publisher and weight-update verifier](https://github.com/vllm-project/vllm/tree/vllm-state-transition-verifier/tools/state_transition_verifier) on the
`vllm-state-transition-verifier` branch pushes a local Hugging Face checkpoint
into a running `vllm serve` over NCCL, without a trainer, and compares
fixed-token outputs and GSM8K accuracy before and after the update. Get it with:

```bash
git fetch origin vllm-state-transition-verifier
git checkout FETCH_HEAD -- tools/state_transition_verifier
```

and start from `tools/state_transition_verifier/README.md`.

## API Endpoints

When running vLLM as an HTTP server, the following endpoints are available for weight transfer. `HTTPVLLMWeightSyncClient` speaks the first five for you. A trainer may also implement the protocol directly without importing vLLM: see the wire contracts in [NCCL](nccl.md#trainer-without-vllm-over-http) and [IPC](ipc.md#trainer-without-vllm-over-http). This keeps the trainer decoupled from vLLM: it needs no vLLM install or version pin, only this documented HTTP and wire contract, so trainer and server can upgrade independently (non-PyTorch trainers such as JAX can use it too).

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/init_weight_transfer_engine` | POST | Initialize the weight transfer engine with backend-specific info |
| `/start_weight_update` | POST | Start a weight update |
| `/update_weights` | POST | Transfer a batch of weights with backend-specific metadata |
| `/finish_weight_update` | POST | Finish the update and optionally commit its `weight_version` |
| `/shutdown_weight_transfer_engine` | POST | Release the transfer channel without stopping inference |
| `/update_weight_version` | POST | Update `weight_version` without changing model weights |
| `/weight_info` | GET | Get the latest committed weight version |
| `/pause` | POST | Pause generation before weight sync to handle inflight requests |
| `/resume` | POST | Resume generation after weight sync |
| `/get_world_size` | GET | Get the number of inference workers (useful for NCCL world size calculation) |

!!! note
    The HTTP weight transfer endpoints require `VLLM_SERVER_DEV_MODE=1` to be set.

!!! warning
    If `/update_weights` fails, the model is left mid-reload while `/health`
    stays 200, and `/finish_weight_update` is rejected. Before generating again,
    resend a full `/start_weight_update` → `/update_weights` →
    `/finish_weight_update` (see [#57854](https://github.com/vllm-project/vllm/pull/57854)).

The Rust frontend's optional gRPC `Control` service exposes the same pause, sleep, weight-transfer, and weight-version lifecycle for trusted sidecars. The `ServerInfo.rl_capabilities` response reports whether weight transfer and sleep mode were configured. Backend-specific `init_info` and `update_info` remain JSON metadata; model tensors continue to move over the configured NCCL, IPC, sparse-NCCL, or sharded-RDT transport.

## Extending the System

Every piece of the system is replaceable: the weights you send (`WeightSource`),
the control-plane transport (`VLLMWeightSyncClient`), and the transports
themselves (both engine ABCs, each with its own factory registry). See
[Base Classes](base.md).
