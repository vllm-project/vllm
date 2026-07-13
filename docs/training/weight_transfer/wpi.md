# WPI Engine

The WPI (Weight Propagation Interface) weight transfer engine uses the [Weight Propagation Interface](https://github.com/llm-d-incubation/weight-propagation-interface) to enable high-throughput, cross-node zero-copy weight updates.

Unlike the [NCCL](nccl.md) or [IPC](ipc.md) engines where vLLM itself manages communication channels and CPU/GPU memory allocation, the WPI backend relies on a persistent VRAM buffer and an internal NCCL communicator managed externally by a node-level WPI driver.

## When to Use WPI

- Training and inference run on **separate GPUs** across one or more nodes.
- You have large models (e.g. 70B+ parameters) and need to avoid the GPU memory allocation overhead of creating duplicate parameter tensors during updates.
- You want to utilize high-bandwidth hardware fabrics (like InfiniBand or RoCE) at near-line rate.
- You are using Kubernetes and have the WPI operator and driver daemonset deployed in your cluster.

## How It Works

1. **Persistent VRAM Buffer**: The WPI driver manages a pre-allocated VRAM buffer on each GPU node. This buffer is persistent and reused across training/sync iterations (avoiding allocation/deallocation overhead).
2. **Zero-Copy Memory Sharing**: The local WPI driver passes the CUDA memory file descriptor (FD) to the containerized trainer/worker processes using UNIX domain sockets and SCM_RIGHTS.
3. **CUDA Import**: Both trainer and inference processes import the shareable handle and map the memory space into their own GPU memory, wrapping the pointer as a flat PyTorch tensor.
4. **Hardware Propagation**: The trainer copies weights into the flat buffer and triggers propagation. The WPI driver executes the NCCL broadcast or sharded scatter over InfiniBand/RDMA fabric.
5. **Worker Sync**: The inference worker waits for a READY signal from the local notify socket, unpacks the parameters from the mapped VRAM buffer using offset metadata, and swaps them into the active model.

## Installation

The WPI engine requires the WPI client Python package to be installed in both the trainer and vLLM environments.

```bash
# From the WPI repository subdirectory
pip install weight-propagation-interface/consumer/wpi_client/
```

## Initialization

The WPI backend requires explicit setup of the persistent buffer and drivers on both the trainer and inference worker processes.

### Inference Side

Call `init_weight_transfer_engine` with `WPIWeightTransferInitInfo`:

```python
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# Initialize WPI weight transfer engine
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
        init_info=dict(
            buffer_id="vllm-weights",
            buffer_size_bytes=20 * 1024**3,  # 20 GiB (must hold all weights)
            socket_dir="/run/wpi/sockets",
            driver_port=50051,
            # Optional sharding config for tensor parallel:
            # shard_index=-1 (defaults to tp_rank if total_shards > 0)
            # total_shards=0
        )
    )
)
```

### Trainer Side

Call `WPIWeightTransferEngine.trainer_init` to stage the source buffer and map the trainer GPU's memory:

```python
from vllm.distributed.weight_transfer.wpi_engine import (
    WPIWeightTransferEngine,
)

ctx = WPIWeightTransferEngine.trainer_init(
    init_info=dict(
        buffer_id="vllm-weights",
        buffer_size_bytes=20 * 1024**3,
        socket_dir="/run/wpi/sockets",
        driver_port=50051,
    ),
    target_node_ids=["10.0.0.2", "10.0.0.3"],  # Inference node IPs
)
```

The returned context contains references to the WPI client and the flat mapped VRAM buffer.

## Sending Weights

To perform a weight update, the trainer packs model parameters into the mapped WPI buffer, triggers the WPI driver's NCCL propagation, and sends the layout metadata to vLLM workers (via HTTP or Ray control plane):

```python
from vllm.distributed.weight_transfer.wpi_engine import (
    WPIWeightTransferEngine,
    WPITrainerSendWeightsArgs,
)

# Start weight update on inference side
llm.start_weight_update()

# Prepare send arguments with the trainer context
trainer_args = WPITrainerSendWeightsArgs(
    send_mode="http",  # 'http' or 'ray'
    url="http://localhost:8000",
    trainer_ctx=ctx,
)

# Pack weights, propagate via WPI NCCL, and deliver layout metadata
WPIWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)

# Finalize weight update on inference side
llm.finish_weight_update()
```

See [`WPITrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/wpi_engine.py) for the full list of configurable fields.

## Comparison with Other Backends

| Feature | IPC Backend | NCCL Backend | WPI Backend |
| :--- | :--- | :--- | :--- |
| **Colocation** | Must be on same GPU/node | Can be separate nodes | Can be separate nodes |
| **VRAM Allocation** | Zero-copy (shares trainer's memory) | Allocates temporary duplicate buffers | Zero-copy (persistent driver-managed buffer) |
| **Network Fabric** | CUDA IPC (NVLink/PCIe) | NCCL broadcast (NVLink/InfiniBand) | NCCL broadcast/scatter (InfiniBand/RoCE) |
| **Connection Mgmt** | Ray / HTTP control plane | vLLM `StatelessProcessGroup` | External WPI DaemonSet |
| **Orchestration** | Python process orchestration | PyTorch distributed group | Kubernetes Custom Resources (DRA) |

## Performance Benchmarks

The WPI engine achieves high fabric utilization by avoiding CPU staging and runtime memory allocation overheads. Below are baseline performance metrics:

- **Cross-Node (8-GPU Scatter, A4)**: **251 GB/s aggregate throughput** for a 600 GB tensor-parallel sharded scatter over InfiniBand (GPUDirect RDMA enabled).
- **Cross-Node (A3 Ultra InfiniBand)**: **36.57 GB/s** for a 75 GB model broadcast, achieving over 73% of maximum physical fabric bandwidth (a 565% improvement over standard socket routing).
