# IPC Engine

The IPC weight transfer engine uses **CUDA IPC** (Inter-Process Communication) handles to share GPU memory directly between the trainer and inference workers on the **same node and same GPU**. This avoids any data copying, making it a efficient option when colocating training and inference.

## When to Use IPC

- Training and inference on the **same GPU** (colocated)
- You want to minimize memory overhead by sharing tensors in-place

## How It Works

1. The trainer creates CUDA tensors for each weight and generates IPC handles using `torch.multiprocessing.reductions.reduce_tensor`.
2. IPC handles are sent to the inference engine via **Ray.remote()** or **HTTP POST**.
3. The inference worker reconstructs the tensors from the handles, reading directly from the trainer's GPU memory.

!!! warning
    IPC handles involve sending serialized Python objects. When using HTTP transport, you must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on both the server and client. This is because IPC handles are pickled and base64-encoded for HTTP transmission.

## Initialization

The IPC backend requires no initialization on either side. The `init_transfer_engine` call is a no-op for IPC.

## Sending Weights

IPC supports two transport modes for delivering the handles:

### Ray Mode

Used when vLLM is running as a Ray actor:

```python
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

trainer_args = IPCTrainerSendWeightsArgs(
    mode="ray",
    llm_handle=llm_actor_handle,
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

In Ray mode, the engine calls `llm_handle.update_weights.remote(...)` directly, passing the IPC handles via Ray's serialization.

### HTTP Mode

Used when vLLM is running as an HTTP server:

```python
trainer_args = IPCTrainerSendWeightsArgs(
    mode="http",
    url="http://localhost:8000",
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

In HTTP mode, IPC handles are pickled, base64-encoded, and sent as JSON to the `/update_weights` endpoint.

See [`IPCTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/ipc_engine.py) for the full list of configurable fields.

## EP-sharded MoE routed-expert load

For large MoE models (where a single fused `gate_up_proj` weight can reach 8–21 GB and force `cuda_ipc_buffer_size ≥ max_weight_size`), the trainer can ship only the local-EP-rank's slice of routed experts:

```python
# Trainer: query the rollout layout once at startup
layouts = llm.collective_rpc("get_moe_routed_ep_layout")
# layouts[rank][layer_name] = {
#   "ep_rank", "ep_size",
#   "local_num_routed_experts", "global_num_routed_experts",
#   "num_fused_shared_experts",
#   "local_to_global_routed": [<global_id>, ...],   # length local_num_routed_experts
# }

# Trainer: stack only this rank's routed experts in ascending global-id order
# w13_local has shape [local_num_routed_experts, 2 * intermediate_per_partition, hidden]

trainer_args = IPCTrainerSendWeightsArgs(
    mode="ray",
    llm_handle=llm_actor_handle,
    moe_routed_expert_global_ids={
        "model.layers.0.mlp.experts.gate_up_proj":
            layouts[rank]["model.layers.0.mlp.experts"]["local_to_global_routed"],
        "model.layers.0.mlp.experts.down_proj":
            layouts[rank]["model.layers.0.mlp.experts"]["local_to_global_routed"],
        # ... per layer ...
    },
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=trainer_iter_routed_sharded(),  # yields (name, [local_routed_E, ...])
    trainer_args=trainer_args,
)
```

The sender validates `len(ids) == tensor.shape[0]` before creating IPC handles. The receiver dispatches each routed weight through `FusedMoE.load_routed_expert_weights`, which uses the global-id list (not heuristic shape comparisons) to write into the correct local expert slots; this works under both contiguous and round-robin EP partitions and is unaffected by `num_fused_shared_experts` inflation of `local_num_experts`.

Weights not in `moe_routed_expert_global_ids` (shared experts, global scales, attention, etc.) fall through to `model.load_weights` unchanged. `Worker.update_weights` preserves the original arrival order so order-sensitive flows such as GGUF `is_gguf_weight_type`-before-data still work.

## Examples

- [RLHF with IPC weight syncing (offline, Ray)](../../examples/rl/rlhf_ipc.md) - Colocated training and inference on a single GPU using Ray placement groups and CUDA IPC handles
- [RLHF with IPC weight syncing (online serving, HTTP)](../../examples/rl/rlhf_http_ipc.md) - Weight transfer with a vLLM HTTP server where both server and trainer share the same GPU
