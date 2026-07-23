# Base Class and Custom Engines

The weight transfer system is built on an abstract base class that defines the contract between vLLM's worker infrastructure and the transport backend. You can implement custom backends by subclassing `WeightTransferEngine` and registering them with the `WeightTransferEngineFactory`.

## WeightTransferEngine

The `WeightTransferEngine` is a generic abstract class parameterized by two dataclass types:

- **`TInitInfo`** (extends `WeightTransferInitInfo`): Backend-specific initialization parameters.
- **`TUpdateInfo`** (extends `WeightTransferUpdateInfo`): Backend-specific weight update metadata.

### Abstract Methods

Subclasses must implement these methods:

| Method | Side | Description |
| ------ | ---- | ----------- |
| `init_transfer_engine(init_info)` | Inference | Initialize the communication channel on each inference worker |
| `start_weight_update()` | Inference | Prepare for an update (e.g. begin layerwise reload); no-op for in-place engines |
| `finish_weight_update()` | Inference | Finalize the update (e.g. finalize layerwise reload); no-op for in-place engines |
| `receive_weights(update_info)` | Inference | Receive weights and load them into `self.model` |
| `shutdown()` | Inference | Clean up resources |
| `trainer_send_weights(iterator, trainer_args)` | Trainer | Static method to send weights from the trainer process |

The base class provides two methods:

1. `__init__` : Engines receive `config` (`WeightTransferConfig`),  `vllm_config` (`VllmConfig`), `device` (`torch.device`) and  `model` (`nn.Module`)  
2. `update_weights(update_info_dict)`:  Thin wrapper for `receive_weights`: parses
the dict into user-specified data type, calls `receive_weights`, and synchronizes the device. Subclasses implement `receive_weights`.

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

At the LLM/API layer, call `start_draft_weight_update()` instead of
`start_weight_update()` to target the speculative draft model;
`update_weights` / `finish_weight_update` are unchanged.

### WeightTransferUpdateInfo

The base `WeightTransferUpdateInfo` is a marker class for backend-specific update info:

```python
@dataclass
class WeightTransferUpdateInfo(ABC):
    pass
```

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
from collections.abc import Iterator
from typing import Any
import torch

class MyWeightTransferEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo) -> None:
        # Set up connection to trainer using init_info.endpoint, etc.
        ...

    def start_weight_update(self) -> None:
        # Checkpoint-format engines: run initialize_layerwise_reload(self.model).
        # In-place engines: no-op
        ...

    def finish_weight_update(self) -> None:
        # Checkpoint-format engines: run finalize_layerwise_reload(...).
        # In-place engines: no-op
        ...

    def receive_weights(self, update_info: MyUpdateInfo) -> None:
        weights = []
        for name, dtype_name, shape in zip(
            update_info.names, update_info.dtype_names, update_info.shapes
        ):
            dtype = getattr(torch, dtype_name)
            weight = self._fetch_weight(name, shape, dtype)
            weights.append((name, weight))
        self.model.load_weights(weights)

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

The factory uses a registry pattern with lazy loading. Built-in engines (`nccl`, `ipc`, and `sparse_nccl`) are registered at import time but their modules are only loaded when the backend is actually requested. This avoids importing heavy dependencies (like NCCL communicators) when they aren't needed.

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# Create an engine from config
engine = WeightTransferEngineFactory.create_engine(
    config=weight_transfer_config,
    vllm_config=vllm_config,
    device=device,
    model=model,
)
```
