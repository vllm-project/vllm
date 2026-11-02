# Connex Engine

The `connex` weight transfer engine pulls each published weight version with
one-sided RDMA reads (GPUDirect when available) from the trainer or from
inference engines that already hold that version. It is maintained in
[Connex](https://github.com/HeterInfra/connex) and re-exported by vLLM, like
the [ModelExpress](modelexpress.md) backend.

## When to Use Connex

- The rollout pool is elastic: inference engines join, leave or fail between
  or during weight updates, and a membership change should not require
  re-forming a communicator group.
- One trainer feeds many inference engines across nodes, and the trainer's
  NIC should not serve every engine: engines that finished pulling serve the
  next ones (tree fan-out).
- An engine that dies while others are pulling from it must not stall the
  update: readers switch to another source and finish.

For a fixed set of engines on a healthy network, [NCCL](nccl.md) broadcast
remains the simplest choice.

## How It Works

1. The trainer packs its weights into one registered buffer, publishes it as
   the next version and records the version in the Connex weight registry
   (a small HTTP service that tracks members, versions and who holds each
   version).
2. `update_weights` on each worker asks the registry for a source of that
   version and reads the buffer in chunks with one-sided reads. A read that
   fails (source died or left) is retried from another source; reads are
   idempotent byte ranges.
3. The worker loads the received tensors with `model.load_weights` inside the
   layerwise reload lifecycle (`start_weight_update` / `finish_weight_update`),
   then registers itself as a holder of the version so later pullers can read
   from it.

vLLM's `connex_engine.py` re-exports the engine and its init/update
dataclasses from `connex.rl.vllm_backend`.

## Inference Side

Install Connex and its RDMA data plane, connex-rs, on every worker (see the
Connex README for building connex-rs against the host's UCX):

```bash
uv pip install "connex-ai[rs]"
```

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve my-model \
    --weight-transfer-config '{"backend": "connex"}'
```

### `ConnexWeightTransferInitInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `registry_url` | required | URL of the Connex weight registry. |
| `member` | host and pid | Logical id prefix of this engine in the registry. |
| `transport` | `"connex-rs"` | Data plane; `"local"` is an in-process transport for tests. |
| `chunk_bytes` | 64 MiB | Size of one read. |

### `ConnexWeightTransferUpdateInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `version` | latest | Version to install. Older versions are not kept; an engine asked for an older version installs the latest. |

## Trainer Side

```python
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.connex_engine import ConnexTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=ConnexTrainerInitInfo(rank=0, registry_url="http://registry:29700"),
    client=HTTPVLLMWeightSyncClient("http://inference:8000"),
    source=ModuleSource(model),
)
for step in range(num_steps):
    train_step(model)
    engine.send_weights()
```

The Connex package also provides the registry service, a client that drives
many engines and drops the ones that fail instead of failing the round, and a
rollout router that never mixes weight versions within a request; see its
documentation.
