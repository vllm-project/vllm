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

The base class provides two methods:

1. `__init__` : Engines receive `config` (`WeightTransferConfig`),  `vllm_config` (`VllmConfig`), `device` (`torch.device`) and  `model` (`nn.Module`)  
2. `update_weights(update_info_dict)`:  Thin wrapper for `receive_weights`: parses
the dict into user-specified data type, calls `receive_weights`, and synchronizes the device. Subclasses implement `receive_weights`.

`WeightTransferEngine` is purely worker-side. The trainer side is a separate
`TrainerWeightTransferEngine` ABC (see below).

## TrainerWeightTransferEngine (trainer side)

The trainer-side engine is symmetric to the worker side: it is stateful,
constructed via a `trainer_init` factory classmethod, and driven by a
parameter-free `send_weights()`. It talks to the inference side through a
`VLLMWeightSyncClient` and is created via `WeightTransferTrainerFactory`
(a separate registry from the worker-side factory).

| Method | Description |
| ------ | ----------- |
| `trainer_init(config, init_info, *, client, weight_iterator=None)` | Classmethod factory: rendezvous with the inference side (driving `client.init_weight_transfer_engine`) and return a ready instance |
| `send_weights(weight_iterator=None)` | Push weights and drive `start_weight_update` / `update_weights` / `finish_weight_update` via the client |
| `shutdown()` | Tear down communicators / process groups (default no-op) |

`weight_iterator` is a **factory** (`Callable[[], Iterator[tuple[str, Tensor]]]`),
not a bare iterator, because `model.named_parameters()` is single-use and each
send round must re-iterate. `materialize_full_tensor(tensor)` (in `base.py`) is
the shared helper that gathers FSDP shards (`full_tensor()`) at send time.

### VLLMWeightSyncClient (control plane)

`VLLMWeightSyncClient` is a structural `Protocol` abstracting the inference-side
weight-sync RPCs. Implementations adapt it to a transport; built-ins are
`HTTPVLLMWeightSyncClient` and `RayVLLMWeightSyncClient`. Any object with the four
methods works — no import or subclassing required.

```python
class VLLMWeightSyncClient(Protocol):
    def init_weight_transfer_engine(self, init_info: dict) -> None: ...
    def start_weight_update(self) -> None: ...
    def update_weights(self, update_info: dict) -> None: ...
    def finish_weight_update(self) -> None: ...
```

All methods are synchronous; backend-specific concurrency (e.g. NCCL running
`update_weights` concurrently with the trainer broadcast) lives inside the engine,
not the client.

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
```

The trainer side is a separate `TrainerWeightTransferEngine`, registered with
`WeightTransferTrainerFactory`:

```python
from typing import Self
from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightIterator,
)

class MyTrainerEngine(TrainerWeightTransferEngine[MyConfig, MyTrainerInitInfo]):
    init_info_cls = MyTrainerInitInfo
    config_cls = MyConfig

    @classmethod
    def trainer_init(cls, config, init_info, *, client, weight_iterator=None) -> Self:
        engine = cls(config, client=client, weight_iterator=weight_iterator)
        # Build the worker-side init info and hand it to the inference side; open
        # the trainer endpoint (concurrently if the backend rendezvous blocks).
        client.init_weight_transfer_engine(worker_init_info_dict)
        return engine

    def send_weights(self, weight_iterator: WeightIterator | None = None) -> None:
        factory = self._resolve_iterator(weight_iterator)
        update_info = build_update_info(factory())  # per-round metadata
        self.client.start_weight_update()
        self.client.update_weights(update_info)      # + data-plane transfer
        self.client.finish_weight_update()
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
