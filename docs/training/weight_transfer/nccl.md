# NCCL Engine

The NCCL weight transfer engine uses [NCCL](https://developer.nvidia.com/nccl) broadcast operations to transfer weights from the trainer to inference workers. It supports **multi-node** and **multi-GPU** setups where the trainer and inference engine run on separate GPUs.

## When to Use NCCL

- Training and inference on **separate GPUs** (possibly across nodes)
- **Tensor-parallel** inference with multiple workers that all need the updated weights
- You need high-bandwidth, low-latency weight transfer over NVLink or InfiniBand

## How It Works

1. The trainer and all inference workers join a shared NCCL process group using `StatelessProcessGroup` (vLLM's torch.distributed-independent group abstraction).
2. The trainer broadcasts weights to all workers simultaneously. Each worker receives and loads the weights.
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

The trainer builds a `NCCLTrainerWeightTransferEngine` via the factory. `trainer_init` opens
the trainer's rank-0 endpoint *and* drives the inference side's
`init_weight_transfer_engine` (the inference-side init shown above is performed for you by the
engine's `VLLMWeightSyncClient`).

```python
from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    ModuleSource,
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    backend="nccl",
    config=NCCLWeightTransferConfig(packed=True),  # packed broadcasting on
    init_info=NCCLTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,  # trainer + all inference workers
        rank=0,  # this trainer process's rank; rank 0 is the sender
    ),
    client=RayVLLMWeightSyncClient(llm_handle),  # or HTTPVLLMWeightSyncClient(url)
    source=ModuleSource(model),
)
```

!!! note
    The trainer is always rank 0; inference workers start at `rank_offset` (1). The trainer
    engine derives the worker-side `rank_offset` for you.

## Sending Weights

```python
engine.send_weights()
```

This single call drives `start_weight_update`, `update_weights` (run concurrently with the
trainer-side NCCL broadcast — both rendezvous inside the same NCCL calls), and
`finish_weight_update` on the inference side.

### Packed Tensor Broadcasting

When `packed=True`, multiple weight tensors are packed into large contiguous buffers before broadcasting. This reduces the number of NCCL operations and uses double/triple buffering with dedicated CUDA streams for overlap between packing, broadcasting, and unpacking.

`packed` and the buffer sizes (`packed_buffer_size_bytes`, `packed_num_buffers`) are static
wire params on `NCCLWeightTransferConfig`. The **same config object** is constructed at both
the trainer and inference sides, so these values cannot drift.

## Receiving Weights (Inference Side)

The inference side triggers weight reception using the four-phase protocol:
`init_weight_transfer_engine`, `start_weight_update`, `update_weights`,
`finish_weight_update`. The init phase is shown [above](#initialization). The
remaining three steps are:

```python
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

# 1. Start the weight update
llm.start_weight_update()

# 2. Receive weights (can be called multiple times for chunked transfers)
llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
        )
    )
)

# 3. Finish the weight update
llm.finish_weight_update()
```

The `names`, `dtype_names`, and `shapes` lists describe each parameter. These
must match the order in which the trainer iterates over its parameters.

`start_weight_update` must be called before `update_weights`, and
`finish_weight_update` must be called after all weight chunks have been
transferred. The NCCL engine receives checkpoint-format weights and applies
layerwise reload processing automatically inside `start_weight_update` /
`finish_weight_update`.

## Sparse NCCL

Sparse, flat-index weight patches use a separate backend,
`WeightTransferConfig(backend="sparse_nccl")`, implemented by
`SparseNCCLWeightTransferEngine`. It shares only NCCL process-group
initialization with the dense engine; patches are applied directly in place to
existing parameters (no layerwise reload). The current sparse MVP requires
`TP=1` and `PP=1`. See the example below.

## Examples

- [RLHF with NCCL weight syncing (offline, Ray)](../../../examples/rl/rlhf_nccl.py) - Trainer on one GPU, 2x tensor-parallel vLLM engine on two others, with packed NCCL weight broadcast
- [RLHF with sparse NCCL weight syncing (offline, Ray)](../../../examples/rl/rlhf_sparse_nccl.py) - Dense-vs-sparse equivalence demo with a real model on a 2-GPU trainer/inference setup; sparse patches use `backend="sparse_nccl"` and currently require `TP=1` and `PP=1`
- [RLHF with async weight syncing (offline, Ray)](../../../examples/rl/rlhf_async_new_apis.py) - Async generation with mid-flight pause, weight sync, resume, and validation against a fresh model
- [RLHF with NCCL weight syncing (online serving, HTTP)](../../../examples/rl/rlhf_http_nccl.py) - Weight transfer with a running vLLM HTTP server using HTTP control plane and NCCL data plane
