# ModelExpress Engine

The `modelexpress` weight transfer engine installs immutable weight versions
published through [ModelExpress](https://github.com/ai-dynamo/modelexpress).
Each update identifies the exact version to install, while ModelExpress handles
source selection, transfer, and installation.

## When to Use ModelExpress

- Your workflow manages immutable weight versions through the ModelExpress
  Refit API.
- Inference workers need to pull a specific published version through
  ModelExpress, using GPU peer transfer or object storage.
- You want to load full checkpoints or replay checkpoint deltas from S3.

## How It Works

1. A publisher creates a weight-version resource through the ModelExpress Refit
   API, publishes its payload, and marks the version READY.
2. The orchestrator pauses generation and calls `start_weight_update` on the
   inference engine.
3. The orchestrator calls `update_weights` with the target `version_id`.
   Each worker's `ModelExpressGeneratorClient` resolves and stages that version,
   then applies it to the live vLLM model.
4. After a successful update, `finish_weight_update` releases the staged handle.
   The orchestrator can then resume inference.

`ModelExpressTrainerClient` is an optional publisher. Custom integrations can
manage weight-version resources directly through the ModelExpress server's
Refit API. For S3, create a STAGING version with the correct model, payload
format, lineage, and index URI; upload the index and referenced shards in the
format expected by `ModelExpressGeneratorClient`; then mark the version READY.
See the [S3 Delta Weight Refit guide](https://github.com/ai-dynamo/modelexpress/blob/main/docs/S3_DELTA_WEIGHT_REFIT.md)
for the publication sequence and generator artifact requirements.

vLLM's `modelexpress_engine.py` re-exports the engine and its init/update
dataclasses from
`modelexpress_rl.inference.engines.vllm.weight_transfer_engine`.
The implementation and configuration schema live in ModelExpress. Upgrading
the installed ModelExpress package and restarting workers picks up compatible
changes without a vLLM code change. The package must continue to implement
vLLM's `WeightTransferEngine` interface.

## Inference Side

Install the ModelExpress Python package with `modelexpress_rl` support on every
worker. The backend is registered natively and loaded only when selected;
`VLLM_PLUGINS=modelexpress` is not required for weight transfer. If the plugin
is enabled, it preserves the native registration and uses the same engine
implementation.

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(model="my-model", weight_transfer_config=WeightTransferConfig(backend="modelexpress"))
```

```bash
vllm serve /models/launch \
    --load-format safetensors \
    --weight-transfer-config '{"backend":"modelexpress"}'
```

### Initialize the engine

Initialize once by passing `init_info` to `init_weight_transfer_engine`.
For HTTP administration, set `VLLM_SERVER_DEV_MODE=1` before starting vLLM and
restrict these endpoints to trusted orchestrators.

For S3 updates, send the following body to `POST /init_weight_transfer_engine`:

```json
{
  "init_info": {
    "model_name": "policy",
    "server_url": "modelexpress:8001",
    "object_storage_type": "S3",
    "initial_base_version_id": "policy-v0",
    "seed_checkpoint_path": "/models/launch",
    "refit_checkpoint_dir": "/mxdelta/receiver",
    "refit_checkpoint_max_size_gb": 200,
    "object_storage_region_name": "us-west-2"
  }
}
```

### `ModelExpressWeightTransferInitInfo`

These fields belong inside `init_info`, rather than
`--weight-transfer-config`. The installed ModelExpress package defines the
schema and defaults.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `model_name` | vLLM model name | Logical model identity used for published ModelExpress versions. Set this when it differs from the vLLM model path or name. |
| `server_url` | ModelExpress configuration | ModelExpress server address, such as `modelexpress:8001`. If omitted, uses ModelExpress's environment settings, then `localhost:8001`. |
| `initial_serving_version_id` | `None` | Version already loaded by the inference worker. If omitted, ModelExpress uses the initial version supplied by its runtime, when available. |
| `registration_ttl_seconds` | Three MX heartbeat intervals | Worker registration lifetime in seconds, renewed by ModelExpress. |
| `lease_ttl_seconds` | Registration lifetime | Weight-version lease lifetime in seconds, renewed while held. |
| `max_transfer_attempts` | `3` | Maximum source-discovery and transfer attempts for one staged update. |
| `max_replay_chain_length` | `64` | Maximum number of payload revisions replayed to reach the requested version. |
| `rpc_timeout_seconds` | `30.0` | Deadline in seconds for each control-plane or manifest RPC. |

### Object storage settings

To enable object storage, provide `object_storage_type`,
`initial_base_version_id`, `seed_checkpoint_path`, and `refit_checkpoint_dir`
inside `init_info`. Endpoint and region settings also require these four fields.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `object_storage_type` | `None` | Object storage provider. Use `"S3"`; the generator currently supports S3. |
| `initial_base_version_id` | `None` | Registered base version matching the seed checkpoint. Required for object storage. |
| `seed_checkpoint_path` | `None` | Local seed checkpoint path accessible to each worker. Required for object storage. |
| `refit_checkpoint_dir` | `None` | Local directory for cached checkpoints and delta replay. Required for object storage. |
| `refit_checkpoint_max_size_gb` | ModelExpress default | Checkpoint cache quota in decimal GB. The example sets an explicit 200 GB quota; it does not reserve disk space. |
| `object_storage_endpoint_url` | `None` | Optional endpoint override for an S3-compatible service. |
| `object_storage_region_name` | `None` | Optional S3 region, such as `us-west-2`. |

The seed checkpoint must match `initial_base_version_id`. Credentials use
ModelExpress's normal credential chain. Omit the object storage settings when
using only GPU peer transfer.

### Apply a weight update

After the target version is READY, pause generation and call
`POST /start_weight_update`. Then send the exact immutable version ID to
`POST /update_weights`:

```json
{"update_info": {"version_id": "policy-v1"}}
```

Call `POST /finish_weight_update` only after the update succeeds, then resume
inference. The staged handle is retained until finish or shutdown. Staging and
installation failures propagate; do not resume a worker with uncertain weights.

See the ModelExpress documentation for publication, source selection, object
storage, and deployment requirements.
