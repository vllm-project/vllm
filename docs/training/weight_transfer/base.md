# Base Classes and Custom Engines

The weight transfer system is built from four abstractions, each independently
replaceable:

| Abstraction | Side | Answers |
| ----------- | ---- | ------- |
| [`WeightSource`](#weightsource) | Trainer | *What* weights to send |
| [`VLLMWeightSyncClient`](#vllmweightsyncclient) | Trainer | *How to reach* the inference engine — the adapter for your RL stack's own vLLM wrapper |
| [`TrainerWeightTransferEngine`](#trainerweighttransferengine) | Trainer | *How to transmit* the bytes |
| [`WeightTransferEngine`](#weighttransferengine) | Inference | *How to receive* them and load them |

The two engines are registered in two separate factories,
[`WeightTransferTrainerFactory`](#weighttransfertrainerfactory) and
[`WeightTransferEngineFactory`](#weighttransferenginefactory). They share backend
names by convention, but a trainer process never instantiates a worker engine or
vice versa, so the registries stay independent.

## Trainer Side

### WeightSource

A `WeightSource` is a **re-iterable** source of the trainer's weights with two
channels:

- **`metadata() -> list[ParamMeta]`** — the name, wire dtype, and full shape of
  every parameter, *without transferring anything*. Cheap when shapes are known
  locally (an FSDP `DTensor` knows its global shape); may be expensive on the
  first call for producers that must materialize to learn shapes (a
  Megatron-Bridge export), in which case it should cache.
- **iteration** — yields fully-materialized `(name, tensor)` pairs, one at a
  time.

```python
@dataclass(frozen=True)
class ParamMeta:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
```

!!! warning "The two channels must agree, element for element"
    `metadata()` must declare exactly what iteration will yield: the same
    parameters, in the same order, with the same dtypes and shapes. This is an
    invariant of the ABC, not of any one backend — a source that reorders, omits,
    or re-dtypes a parameter between the two channels is broken even if the
    backend you happen to test against never notices. Backends are free to read
    both channels and to trust that they match; dense NCCL does, and
    [enforces it](nccl.md#the-two-weightsource-channels-must-agree).

Materializing is typically a collective, so **every trainer rank must iterate the
same source in the same order, in lockstep**, or ranks deadlock. `metadata()` can
itself be a collective for custom producers, so it too runs on every rank — only
the sender ships the result. Under pipeline parallelism a rank may not own a
parameter at all; iterating still drives the collective, and the yielded tensor
is only meaningful on the sender.

`iter(source)` must yield a *fresh* pass each round.

#### ModuleSource

`ModuleSource(module)` is the common case, over `module.named_parameters()`. It
handles plain and FSDP-sharded modules with no special casing: iteration
all-gathers each `DTensor` via `full_tensor()`, while `metadata()` reads the
*global* `.shape` / `.dtype` and so never triggers a gather.

```python
from vllm.distributed.weight_transfer import ModuleSource

source = ModuleSource(model)
```

#### Custom Sources

Subclass `WeightSource` when the weights you want to send require additional processing to convert to a HF compatible format.

```python
from vllm.distributed.weight_transfer import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.base import materialize_full_tensor


class MyExportSource(WeightSource):
    def __init__(self, model):
        self._model = model
        self._meta: list[ParamMeta] | None = None

    def metadata(self) -> list[ParamMeta]:
        # Cache: for producers that must materialize to learn shapes, this is
        # the expensive channel. Runs on every rank (it may be a collective).
        if self._meta is None:
            self._meta = [
                ParamMeta(name, t.dtype, tuple(t.shape)) for name, t in self._export()
            ]
        return self._meta

    def __iter__(self):
        # Must yield exactly what metadata() declared, in the same order.
        for name, tensor in self._export():
            yield name, materialize_full_tensor(tensor)
```

### VLLMWeightSyncClient

**This is the adapter for however your RL stack reaches vLLM.** Many RL frameworks wrap
inference engines in their own abstractions, and each reaches vLLM its own way.
`VLLMWeightSyncClient` is the single seam where that bespoke shape is adapted, so
weight sync engines remain control plane agnostic.

The contract is only this: **however the wrapper is shaped, it must bottom out in
the same four calls** — `init_weight_transfer_engine` once at setup, then
`start_weight_update` → one or more `update_weights` → `finish_weight_update` per
round. Everything a trainer engine needs from the inference side goes through them.

```python
class VLLMWeightSyncClient(Protocol):
    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None: ...
    def start_weight_update(self) -> None: ...
    def update_weights(self, update_info: dict[str, Any]) -> None: ...
    def finish_weight_update(self, weight_version: str | None = None) -> None: ...
```

It is a `@runtime_checkable` structural `Protocol` (PEP 544), which is what makes
adapting cheap: **any object with those four methods already satisfies it**. An existing
wrapper in your framework can usually become a client by gaining four forwarding
methods.

Two implementations ship with vLLM:

| Client | Talks to |
| ------ | -------- |
| `RayVLLMWeightSyncClient(handle)` | One or more `AsyncLLM`/`LLM` Ray actors. Accepts a list and fans each call out to every handle, blocking on all of them, so a multi-actor (e.g. multi-DP) deployment is driven as one unit |
| `HTTPVLLMWeightSyncClient(base_url, timeout=300)` | A vLLM server over the RLHF HTTP routes |

Custom weight sync clients can be implement like so:

```python
class MyFrameworkWeightSyncClient:
    """Adapts one RL framework's rollout pool to the four weight-sync calls."""

    def __init__(self, rollout_pool):
        self.pool = rollout_pool          # whatever your stack already has

    def init_weight_transfer_engine(self, init_info):
        # Fan out to every replica and block: all of them receive weights.
        self.pool.broadcast_rpc("init_weight_transfer_engine", init_info=init_info)

    def start_weight_update(self):
        self.pool.broadcast_rpc("start_weight_update")

    def update_weights(self, update_info):
        self.pool.broadcast_rpc("update_weights", update_info=update_info)

    def finish_weight_update(self, weight_version=None):
        self.pool.broadcast_rpc("finish_weight_update")
        if weight_version is not None:
            self.pool.broadcast_rpc("update_weight_version", weight_version)
```

Two things to get right in any adapter:

- **Reach every replica, and block until all of them are done.** A weight update
  is not a load-balanced request: every worker holding a copy of the model must
  receive it. Returning before they all finish lets the trainer race ahead of
  workers still loading. (Both built-in clients do this — Ray by fanning out over
  its handles, HTTP because the server's DP client broadcasts internally.)
- **Raise on failure.** Trainer engines rely on exceptions to surface
  inference-side errors; a client that swallows them turns a failed sync into
  silently stale weights, or into a hang for backends whose transfer rendezvouses
  with the worker.

!!! note
    HTTP cannot carry raw CUDA IPC handles, so `HTTPVLLMWeightSyncClient` pickles
    and base64-encodes them into an `ipc_handles_pickled` field. The worker
    deserializes it only when `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Backends
    whose payloads are JSON-native (NCCL) pass through untouched.

### TrainerWeightTransferEngine

The trainer-side engine: it holds the transport state (NCCL communicators, IPC
device info, transfer plans), pulls weights from a `WeightSource`, and drives the
inference side through a `VLLMWeightSyncClient`. It is generic over its init info
type, constructed by the `trainer_init` classmethod factory, and driven by `send_weights()`.

| Method | Description |
| ------ | ----------- |
| `trainer_init(init_info, *, client, source=None)` | Classmethod. Rendezvous with the inference side and return a ready instance |
| `send_weights()` | Push weights and drive the full update round trip |
| `shutdown()` | Tear down communicators / process groups. Default no-op |

Both `trainer_init` and `send_weights` are called on **every** trainer rank.
`is_sender` is resolved once, at `trainer_init`, from `init_info.rank`. Each
engine holds the real client on every rank but guards the control-plane RPCs and
the transmit on `self.is_sender`, so only the sender touches the wire; non-sender
ranks still run every collective so the group stays aligned.

The trainer side takes **no `WeightTransferConfig`**. The backend comes from the
init info's `backend` `ClassVar`, and the wire params ride the init info too.

#### TrainerInitInfo

The `init_info` passed to `trainer_init` above. It is how a caller configures a
transfer: it selects the backend, says which rank this process is, and carries the
wire params. Each backend subclasses it; the base class holds the one field every
backend needs.

```python
@dataclass
class TrainerInitInfo:
    backend: ClassVar[str]        # factory dispatch key
    rank: int = field(kw_only=True)

    @property
    def is_sender(self) -> bool:
        return self.rank == 0
```

- **`rank`** is this trainer process's rank, supplied **explicitly**. The engine
  does not read it from a global process group, which is ambiguous once several
  groups (FSDP / TP / PP / EP) exist. **Rank 0 is always the sender** — this is
  what `trainer_init` resolves into `is_sender`. It is keyword-only, so backend
  subclasses can add positional fields freely.
- **`backend`** is a `ClassVar`, not an `__init__` field: it is a fixed
  per-backend constant that the factory reads to dispatch, which is why callers
  never pass a `backend=` argument. Every subclass must set it —
  `__init_subclass__` raises otherwise.

Subclasses also carry the transfer's **wire params** (`packed`, buffer sizes).
The sender propagates them to the worker inside `trainer_init`, so the two sides
cannot disagree. See [`NCCLTrainerInitInfo`](nccl.md#nccltrainerinitinfo) and
[`IPCTrainerInitInfo`](ipc.md#ipctrainerinitinfo) for the concrete fields.

#### Full-Resync vs. Delta Backends

`source` is optional, which splits the backends into two shapes:

- **Full resync** (NCCL, IPC) — a stable `WeightSource` is fixed at
  `trainer_init` and re-iterated each round; `send_weights()` takes no
  arguments. These backends validate that `source` is non-null themselves.
- **Delta** (sparse NCCL) — the payload differs every round, so there is no
  stable source. The engine takes no `source` and each round's payload is passed
  straight to `send_weights(patches)`.

#### Implementing a Custom Trainer Engine

```python
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import Self

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)


@dataclass
class MyTrainerInitInfo(TrainerInitInfo):
    backend: ClassVar[str] = "my_backend"

    endpoint: str
    chunk_size_bytes: int = 256 * 1024 * 1024   # a wire param: shipped to the worker


class MyTrainerWeightTransferEngine(TrainerWeightTransferEngine[MyTrainerInitInfo]):
    init_info_cls = MyTrainerInitInfo

    def __init__(self, *, client, source, is_sender=True, chunk_size_bytes=0):
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.chunk_size_bytes = chunk_size_bytes

    @classmethod
    def trainer_init(
        cls,
        init_info: MyTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("my_backend requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            chunk_size_bytes=init_info.chunk_size_bytes,
        )
        if engine.is_sender:
            # Ship the must-agree wire params so the worker decodes exactly as
            # this trainer encodes, then open the trainer-side endpoint.
            engine.client.init_weight_transfer_engine(
                {"chunk_size_bytes": init_info.chunk_size_bytes}
            )
        return engine

    def send_weights(self) -> None:
        assert self.source is not None
        meta = self.source.metadata()      # every rank: may be a collective
        if not self.is_sender:
            for _ in self.source:          # stay in the trainer-side collective
                pass
            return

        self.client.start_weight_update()
        self.client.update_weights(
            {
                "names": [m.name for m in meta],
                "dtype_names": [str(m.dtype).split(".")[-1] for m in meta],
                "shapes": [list(m.shape) for m in meta],
            }
        )
        for name, tensor in self.source:
            ...                            # transmit
        self.client.finish_weight_update()
```

Two things to get right, both of which have bitten the built-in backends:

- **Drain before returning.** `send_weights` must not return with transfers
  still in flight. Anything keeping a send buffer alive dies with the frame, and
  the inference side's `finish_weight_update` post-processing can otherwise
  finalize weights that have not landed.
- **Never join a control-plane thread on the error path.** If you run
  `update_weights` on a side thread concurrently with a transmit (as NCCL does)
  and the transmit raises, the worker is still blocked in the matching
  collective and will never return. Shut the executor down without waiting, so
  the real exception surfaces instead of hanging.

### WeightTransferTrainerFactory

```python
from vllm.distributed.weight_transfer import WeightTransferTrainerFactory

# Lazy loading (recommended): the module is imported only when the backend is used
WeightTransferTrainerFactory.register_engine(
    "my_backend",
    "my_package.my_module",
    "MyTrainerWeightTransferEngine",
)

# Or register the class directly
WeightTransferTrainerFactory.register_engine("my_backend", MyTrainerWeightTransferEngine)

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=MyTrainerInitInfo(rank=0, endpoint="..."),  # `backend` selects the engine
    client=client,
    source=source,
)
```

## Inference Side

### WeightTransferEngine

A generic abstract class parameterized by two dataclass types:

- **`TInitInfo`** (extends `WeightTransferInitInfo`): backend-specific initialization parameters.
- **`TUpdateInfo`** (extends `WeightTransferUpdateInfo`): backend-specific weight update metadata.

Subclasses must implement five methods:

| Method | Description |
| ------ | ----------- |
| `init_transfer_engine(init_info)` | Initialize the communication channel on each inference worker, and record the trainer-supplied wire params |
| `start_weight_update()` | Prepare for an update (e.g. begin layerwise reload); no-op for in-place engines |
| `finish_weight_update()` | Finalize the update (e.g. finalize layerwise reload); no-op for in-place engines |
| `receive_weights(update_info)` | Receive weights and load them into `self.model` |
| `shutdown()` | Clean up resources |

The base class provides:

1. `__init__`, taking `config` (`WeightTransferConfig`), `vllm_config` (`VllmConfig`), `device` (`torch.device`), and `model` (`nn.Module`).
2. `update_weights(update_info_dict)`, a thin wrapper for `receive_weights`: it parses the dict into the typed dataclass, calls `receive_weights`, and synchronizes the device.
3. `parse_init_info` / `parse_update_info`, which convert API-level dicts into the typed dataclasses and raise `ValueError` on a bad payload.
4. `set_weight_update_target` / `reset_weight_update_target`, used to retarget an update at the speculative draft model.

!!! note "Read wire params from the handshake, not the payload"
    Anything the two sides must agree on — `packed`, buffer geometry — arrives
    on the **init info** and should be stored on `self` in
    `init_transfer_engine`, then read from `self` in `receive_weights`. Per-round
    update info carries only per-round metadata. This is what makes a
    trainer/worker mismatch unrepresentable.

### Request Classes

The API-level request classes provide backend-agnostic serialization using plain dictionaries.

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

Using a built-in client, you never construct these by hand — `RayVLLMWeightSyncClient`
wraps the dicts for you, and `HTTPVLLMWeightSyncClient` posts them as JSON.

At the LLM/API layer, call `start_draft_weight_update()` instead of
`start_weight_update()` to target the speculative draft model;
`update_weights` / `finish_weight_update` are unchanged. Engines that cannot
support this set `supports_draft_weight_update = False`.

### Implementing a Custom Engine

#### 1. Define Info Dataclasses

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
    chunk_size_bytes: int = 256 * 1024 * 1024   # must-agree wire param

@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    # Per-round metadata only.
```

#### 2. Implement the Engine

```python
class MyWeightTransferEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo) -> None:
        # Record the trainer's wire params, then set up the connection.
        self.chunk_size_bytes = init_info.chunk_size_bytes
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

#### 3. Register with the Factory

```python
from vllm.distributed.weight_transfer import WeightTransferEngineFactory

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

Once registered, users select your backend via `WeightTransferConfig(backend="my_backend")`.

### WeightTransferEngineFactory

The factory uses a registry pattern with lazy loading. Built-in engines (`nccl`, `ipc`, and `sparse_nccl`) are registered at import time but their modules are only loaded when the backend is actually requested. This avoids importing heavy dependencies (like NCCL communicators) when they aren't needed.

```python
from vllm.distributed.weight_transfer import WeightTransferEngineFactory

# Create an engine from config
engine = WeightTransferEngineFactory.create_engine(
    config=weight_transfer_config,
    vllm_config=vllm_config,
    device=device,
    model=model,
)
```

vLLM calls this for you during worker startup; you only need it directly when
embedding the engine in your own worker.
