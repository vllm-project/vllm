# Weight Transfer

vLLM provides a pluggable weight transfer system for synchronizing model weights from a training process to the inference engine during reinforcement learning (RL) workflows. This is essential for RLHF, GRPO, and other online RL methods where the policy model is iteratively updated during training and the updated weights must be reflected in the inference engine for rollout generation.

## Architecture

The weight transfer system follows a **four-phase protocol** with a pluggable backend design:

1. **Initialization** (`init_weight_transfer_engine`): Establishes the communication channel between the trainer and inference workers. Called once before the training loop begins.
2. **Start** (`start_weight_update`): Prepares the inference engine for a weight update.
3. **Weight Update** (`update_weights`): Transfers updated weights from the trainer to the inference engine. May be called one or more times (e.g., for chunked transfers).
4. **Finish** (`finish_weight_update`): Finalizes the weight update (e.g., runs post-processing for checkpoint-format weights). Called once after all weights have been transferred.

## Available Backends

| Backend | Transport | Use Case |
| ------- | --------- | -------- |
| [NCCL](nccl.md) | NCCL broadcast | Separate GPUs for training and inference |
| [IPC](ipc.md) | CUDA IPC handles | Colocated training and inference on same GPU |
| [sparse_nccl](nccl.md#sparse-nccl) | NCCL broadcast | Sparse flat-index weight patches (TP=1/PP=1) |

## Configuration

Specify the weight transfer backend through `WeightTransferConfig`. The backend determines which engine handles the weight synchronization.

### Programmatic (Offline Inference)

```python
from vllm import LLM
from vllm.config import NCCLWeightTransferConfig  # or IPCWeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=NCCLWeightTransferConfig(packed=True),
)
```

When passing the config as a dict (e.g. via the CLI), the right backend subclass is selected
automatically from the `backend` field.

### CLI (Online Serving)

```bash
vllm serve my-model \
    --weight-transfer-config '{"backend": "nccl"}'
```

The `backend` field accepts `"nccl"` (default), `"ipc"`, or `"sparse_nccl"`.

## API Endpoints

When running vLLM as an HTTP server, the following endpoints are available for weight transfer:

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/init_weight_transfer_engine` | POST | Initialize the weight transfer engine with backend-specific info |
| `/start_weight_update` | POST | Start a weight update |
| `/update_weights` | POST | Transfer a batch of weights with backend-specific metadata |
| `/finish_weight_update` | POST | Finish the weight update and run post-processing |
| `/pause` | POST | Pause generation before weight sync to handle inflight requests |
| `/resume` | POST | Resume generation after weight sync |
| `/get_world_size` | GET | Get the number of inference workers (useful for NCCL world size calculation) |

!!! note
    The HTTP weight transfer endpoints require `VLLM_SERVER_DEV_MODE=1` to be set.

## Trainer-Side API

The trainer side mirrors the worker side: a stateful `TrainerWeightTransferEngine`
constructed via the `WeightTransferTrainerFactory.trainer_init` factory, then driven by a
parameter-free `send_weights()`. The engine owns the full handshake and the four-phase
protocol — it talks to the inference side through a `VLLMWeightSyncClient` (built-in HTTP and
Ray implementations are provided; any object with the four control-plane methods works, since
the protocol is structural).

```python
from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    ModuleSource,
    RayVLLMWeightSyncClient,        # or HTTPVLLMWeightSyncClient(base_url)
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo

# 1. Build the engine and rendezvous with the inference side. `trainer_init`
#    drives init_weight_transfer_engine internally (concurrent with the
#    trainer-side rendezvous when the backend needs it).
engine = WeightTransferTrainerFactory.trainer_init(
    backend="nccl",
    config=NCCLWeightTransferConfig(packed=True),
    init_info=NCCLTrainerInitInfo(
        master_address=addr, master_port=port, world_size=ws, rank=0
    ),
    client=RayVLLMWeightSyncClient(llm_handle),
    source=ModuleSource(model),  # re-iterable (name, tensor) source
)

# 2. Push weights. One call drives start_weight_update / update_weights /
#    finish_weight_update on the inference side (and the data-plane transfer).
engine.send_weights()
```

`packed` and the buffer sizes are static "must-agree" wire params and live on the
backend-specific `WeightTransferConfig` subclass, so the trainer and inference sides cannot
drift. See the [NCCL](nccl.md) and [IPC](ipc.md) pages for backend-specific details and full
examples.

## Extending the System

The weight transfer system is designed to be extensible. You can implement custom backends by subclassing `WeightTransferEngine` and registering them with the factory. See the [Base Class](base.md) page for details.
