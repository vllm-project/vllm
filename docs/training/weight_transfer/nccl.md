# NCCL Engine

The NCCL weight transfer engine uses [NCCL](https://developer.nvidia.com/nccl) broadcast operations to transfer weights from the trainer to inference workers. It supports **multi-node** and **multi-GPU** setups where the trainer and inference engine run on separate GPUs.

## When to Use NCCL

- Training and inference on **separate GPUs** (possibly across nodes)
- **Tensor-parallel** inference with multiple workers that all need the updated weights
- You need high-bandwidth, low-latency weight transfer over NVLink or InfiniBand

## How It Works

1. The trainer and all inference workers join a shared NCCL process group using `StatelessProcessGroup` (vLLM's torch.distributed-independent group abstraction).
2. The trainer broadcasts weights to all workers simultaneously. Each worker receives and loads weights incrementally.
3. Optionally, **packed tensor broadcasting** batches multiple small tensors into larger buffers with double/triple buffering and CUDA stream overlap for higher throughput. This implementation is based on [NeMo-RL's packed tensor](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/utils/packed_tensor.py).

## Initialization

NCCL requires explicit process group setup. The trainer and inference workers must agree on a master address, port, and world size.

### Inference Side

```python
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# rank_offset accounts for the trainer occupying rank 0
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,  # trainer + all inference workers
        )
    )
)
```

### Trainer Side

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)

group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

!!! note
    `trainer_init` always assigns the trainer to rank 0. Inference workers start at `rank_offset` (typically 1).

## Sending Weights

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

trainer_args = NCCLTrainerSendWeightsArgs(
    group=group,
    packed=True,  # use packed broadcasting for efficiency
)

NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

See [`NCCLTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/nccl_engine.py) for the full list of configurable fields.

### Packed Tensor Broadcasting

When `packed=True`, multiple weight tensors are packed into large contiguous buffers before broadcasting. This reduces the number of NCCL operations and uses double/triple buffering with dedicated CUDA streams for overlap between packing, broadcasting, and unpacking.

Both the trainer (`NCCLTrainerSendWeightsArgs`) and inference side (`NCCLWeightTransferUpdateInfo`) must use matching `packed_buffer_size_bytes` and `packed_num_buffers` values.

## Receiving Weights (Inference Side)

The inference side triggers weight reception by calling `update_weights`:

```python
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)
```

The `names`, `dtype_names`, and `shapes` lists describe each parameter. These must match the order in which the trainer iterates over its parameters.

## Examples

- [RLHF with NCCL weight syncing (offline, Ray)](../../examples/offline_inference/weight_syncing.md#rlhf-nccl) - Trainer on one GPU, 2x tensor-parallel vLLM engine on two others, with packed NCCL weight broadcast
- [RLHF with async weight syncing (offline, Ray)](../../examples/offline_inference/weight_syncing.md#rlhf-async-new-apis) - Async generation with mid-flight pause, weight sync, resume, and validation against a fresh model
- [RLHF with NCCL weight syncing (online serving, HTTP)](../../examples/online_serving/weight_syncing.md#rlhf-http-nccl) - Weight transfer with a running vLLM HTTP server using HTTP control plane and NCCL data plane
