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

## Configuration

Specify the weight transfer backend through `WeightTransferConfig`. The backend determines which engine handles the weight synchronization.

### Programmatic (Offline Inference)

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),  # or "ipc"
)
```

### CLI (Online Serving)

```bash
vllm serve my-model \
    --weight-transfer-config '{"backend": "nccl"}'
```

The `backend` field accepts `"nccl"` (default) or `"ipc"`.

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

Both backends provide static methods that the trainer calls to send weights. The general pattern is:

```python
# 1. Initialize the transfer engine (backend-specific)
EngineClass.trainer_init(init_info)

# 2. Start weight update on inference side
llm.start_weight_update(is_checkpoint_format=True)

# 3. Send weights to inference workers
EngineClass.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=backend_specific_args,
)

# 4. Finish weight update on inference side
llm.finish_weight_update()
```

See the [NCCL](nccl.md) and [IPC](ipc.md) pages for backend-specific trainer APIs and full examples.

## Extending the System

The weight transfer system is designed to be extensible. You can implement custom backends by subclassing `WeightTransferEngine` and registering them with the factory. See the [Base Class](base.md) page for details.
