# Base Class and Custom Engines

The weight transfer system is built on an abstract base class that defines the contract between vLLM's worker infrastructure and the transport backend. You can implement custom backends by subclassing `WeightTransferEngine` and registering them with the `WeightTransferEngineFactory`.

## WeightTransferEngine

The `WeightTransferEngine` is a generic abstract class parameterized by two dataclass types:

- **`TInitInfo`** (extends `WeightTransferInitInfo`): Backend-specific initialization parameters.
- **`TUpdateInfo`** (extends `WeightTransferUpdateInfo`): Backend-specific weight update metadata.

### Abstract Methods

Subclasses must implement these four methods:

| Method | Side | Description |
| ------ | ---- | ----------- |
| `init_transfer_engine(init_info)` | Inference | Initialize the communication channel on each inference worker |
| `receive_weights(update_info, load_weights)` | Inference | Receive weights and call `load_weights` incrementally |
| `shutdown()` | Inference | Clean up resources |
| `trainer_send_weights(iterator, trainer_args)` | Trainer | Static method to send weights from the trainer process |

### Request Classes

The API-level request classes provide backend-agnostic serialization using plain dictionaries. The engine's `parse_init_info` and `parse_update_info` methods convert these dictionaries into typed dataclasses.

```python
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)

# Init request (dict is converted to backend-specific TInitInfo)
init_request = WeightTransferInitRequest(
    init_info={"master_address": "10.0.0.1", "master_port": 29500, ...}
)

# Update request (dict is converted to backend-specific TUpdateInfo)
update_request = WeightTransferUpdateRequest(
    update_info={"names": [...], "dtype_names": [...], "shapes": [...]}
)
```

### WeightTransferUpdateInfo

The base `WeightTransferUpdateInfo` includes an `is_checkpoint_format` flag:

```python
@dataclass
class WeightTransferUpdateInfo(ABC):
    is_checkpoint_format: bool = True
```

When `is_checkpoint_format=True` (the default), vLLM applies layerwise weight processing (repacking, renaming, etc.) on the received weights before loading them. Set to `False` if the trainer has already converted weights to the kernel format expected by the model.

### `moe_routed_expert_global_ids` (EP-sharded MoE)

For RL workflows where the trainer wants to send only the local-EP-rank's slice of routed experts (instead of the full expert list), set the optional `moe_routed_expert_global_ids` field on `WeightTransferUpdateInfo`:

```python
@dataclass
class WeightTransferUpdateInfo(ABC):
    is_checkpoint_format: bool = True
    moe_routed_expert_global_ids: dict[str, list[int]] | None = None
```

Semantics:

- Keys are full weight names (matching what the trainer ships) of MoE routed-expert tensors only — typically `model.layers.<i>.mlp.experts.gate_up_proj` and `.down_proj`.
- Values are `list[int]` of *global* routed-expert ids, one per leading-dim slice of the tensor. For a 3D tensor `[N, ...]`, `len(ids) == N`. For a non-3D single-expert tensor, `len(ids) == 1`.
- Weights present in this map are dispatched through `FusedMoE.load_routed_expert_weights`, which writes only the local-rank's expert slots and silently skips experts mapped elsewhere.
- Weights not present in this map (shared experts, global scales, attention, embeddings, etc.) fall through to `model.load_weights` unchanged.

The trainer side queries the routed-expert layout once at startup:

```python
# Trainer: query layout per FusedMoE layer
layout = llm.collective_rpc("get_moe_routed_ep_layout")[rank]
# layout = {layer_name: {"ep_rank", "ep_size", "local_to_global_routed", ...}}
```

and uses `local_to_global_routed` to decide which global expert ids each rollout rank should receive, packing them into 3D `[local_routed_E, ...]` tensors.

This path is supported on the IPC backend (per-tensor application semantics required); the NCCL backend raises `NotImplementedError` if `moe_routed_expert_global_ids` is non-`None` because packed broadcast cannot dispatch per-tensor routing.

## Implementing a Custom Engine

To create a custom weight transfer backend:

### 1. Define Info Dataclasses

```python
from dataclasses import dataclass
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

@dataclass
class MyInitInfo(WeightTransferInitInfo):
    endpoint: str
    token: str

@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    # Add custom fields as needed
```

### 2. Implement the Engine

```python
from collections.abc import Callable, Iterator
from typing import Any
import torch

class MyWeightTransferEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo) -> None:
        # Set up connection to trainer using init_info.endpoint, etc.
        ...

    def receive_weights(
        self,
        update_info: MyUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        # Receive each weight and call load_weights incrementally
        for name, dtype_name, shape in zip(
            update_info.names, update_info.dtype_names, update_info.shapes
        ):
            dtype = getattr(torch, dtype_name)
            weight = self._fetch_weight(name, shape, dtype)
            load_weights([(name, weight)])

    def shutdown(self) -> None:
        # Clean up resources
        ...

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any],
    ) -> None:
        # Send weights from the trainer process
        for name, tensor in iterator:
            # Send tensor via custom transport
            ...
```

!!! important
    The `load_weights` callable passed to `receive_weights` should be called **incrementally** (one or a few weights at a time) rather than accumulating all weights first. This avoids GPU out-of-memory errors with large models.

### 3. Register with the Factory

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# Option 1: Lazy loading (recommended for built-in engines)
WeightTransferEngineFactory.register_engine(
    "my_backend",
    "my_package.my_module",
    "MyWeightTransferEngine",
)

# Option 2: Direct class registration
WeightTransferEngineFactory.register_engine(
    "my_backend",
    MyWeightTransferEngine,
)
```

Once registered, users can select your backend via `WeightTransferConfig(backend="my_backend")`.

## WeightTransferEngineFactory

The factory uses a registry pattern with lazy loading. Built-in engines (`nccl` and `ipc`) are registered at import time but their modules are only loaded when the backend is actually requested. This avoids importing heavy dependencies (like NCCL communicators) when they aren't needed.

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# Create an engine from config
engine = WeightTransferEngineFactory.create_engine(
    config=weight_transfer_config,
    parallel_config=parallel_config,
)
```
