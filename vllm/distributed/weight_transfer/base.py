# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig

TInitInfo = TypeVar("TInitInfo", bound="WeightTransferInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo", bound="WeightTransferUpdateInfo")
TTrainerInitInfo = TypeVar("TTrainerInitInfo", bound="TrainerInitInfo")

# A trainer supplies its parameters as a `WeightSource` (defined below): a
# re-iterable stream of materialized `(name, tensor)` pairs plus a `metadata()`
# channel. The built-in `ModuleSource` uses `materialize_full_tensor`.


def materialize_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a full, locally-materialized tensor ready to send.

    FSDP shards (DTensors) expose `full_tensor()`, a collective all-gather;
    regular tensors do not and are returned unchanged. Trainer engines call
    this at send time so the (potentially expensive) gather happens exactly
    once — reading `.shape`/`.dtype` for metadata does not trigger it.
    """
    full_tensor = getattr(tensor, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else tensor


@dataclass(frozen=True)
class ParamMeta:
    """Name / wire dtype / full (HF) shape for one output parameter."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


class WeightSource(ABC):
    """A re-iterable source of the trainer's weights, handed to a trainer engine.

    Two channels:

    * `metadata()` — `(name, wire dtype, full shape)` for every parameter,
      *without* transferring. Cheap when shapes are known locally (FSDP
      `DTensor` global shape); may be expensive on first call for backends that
      must materialize to learn shapes (e.g. a Megatron-Bridge export), in which
      case it should cache.
    * iteration — yields fully-materialized `(name, tensor)` pairs, one at a
      time. Materializing is typically a collective (FSDP `full_tensor()`, a
      Megatron export), so every trainer rank must iterate the same source in the
      same order in lockstep, or ranks deadlock. Under pipeline parallelism a
      rank may not own a parameter at all — iterating still drives the collective
      and the yielded tensor is only meaningful on the sender.

    `iter(source)` must yield a *fresh* pass each round. Backends with custom
    producer logic (Megatron export, RDT plans, MoE re-fusing) subclass this.
    """

    @abstractmethod
    def metadata(self) -> list[ParamMeta]:
        """Declare what iteration will yield, without transferring anything.

        Must agree with iteration element for element: the same parameters, in
        the same order, with the same dtypes and shapes. Backends may read both
        channels and trust that they match (dense NCCL sizes the worker's receive
        buffers and its packed chunk boundaries from this, then sends the bytes
        from iteration), so a source that disagrees between the two splits the
        stream differently on each side.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        raise NotImplementedError


class ModuleSource(WeightSource):
    """`WeightSource` over `module.named_parameters()` — the common case.

    Handles both plain dense modules and FSDP-sharded ones with no special
    casing: iteration all-gathers each `DTensor` via `full_tensor()` (a
    collective) and passes regular tensors through. `metadata()` reads the
    *global* `.shape` / `.dtype`, so it never triggers a gather.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(name, p.dtype, tuple(p.shape))
            for name, p in self._module.named_parameters()
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, param in self._module.named_parameters():
            yield name, materialize_full_tensor(param)

    def iter_names(self, names: frozenset[str]) -> Iterator[tuple[str, torch.Tensor]]:
        """Materialize only the named parameters required by one worker."""
        parameters = dict(self._module.named_parameters())
        missing = names - parameters.keys()
        if missing:
            raise ValueError(
                "Rank-local sharding manifest references weights absent from "
                f"the trainer source iteration: {sorted(missing)[:20]}"
            )
        for name, param in parameters.items():
            if name in names:
                yield name, materialize_full_tensor(param)


class RankLocalWeightSource(WeightSource):
    """Restrict a source to checkpoint names consumed by one inference rank."""

    def __init__(self, source: WeightSource, source_names: set[str]) -> None:
        self._source = source
        self._source_names = frozenset(source_names)

    @classmethod
    def from_manifest(
        cls, source: WeightSource, manifest: Any
    ) -> "RankLocalWeightSource":
        if getattr(manifest, "state", "unavailable") != "exact":
            raise ValueError(
                "Cannot construct a rank-local source from an inexact sharding "
                f"manifest: {getattr(manifest, 'reason', None)}"
            )
        return cls(source, set(manifest.source_names))

    def metadata(self) -> list[ParamMeta]:
        metadata = [
            item for item in self._source.metadata() if item.name in self._source_names
        ]
        found = {item.name for item in metadata}
        missing = self._source_names - found
        if missing:
            raise ValueError(
                "Rank-local sharding manifest references weights absent from "
                f"the trainer source: {sorted(missing)[:20]}"
            )
        return metadata

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        if isinstance(self._source, ModuleSource):
            yield from self._source.iter_names(self._source_names)
            return

        # Custom sources may require every trainer rank to drive all producer
        # collectives in lockstep. Consume their full iterator but only expose
        # this inference rank's source names to the transport.
        expected = set(self._source_names)
        for name, tensor in self._source:
            if name in expected:
                expected.remove(name)
                yield name, tensor
        if expected:
            raise ValueError(
                "Rank-local sharding manifest references weights absent from "
                f"the trainer source iteration: {sorted(expected)[:20]}"
            )


# Base protocols for backend-specific dataclasses
@dataclass
class WeightTransferInitInfo(ABC):  # noqa: B024
    """Base class for backend-specific initialization info."""

    pass


@dataclass
class TrainerInitInfo:
    """Base trainer-side init info: which trainer rank drives the transfer.

    `rank` is this trainer process's rank, provided **explicitly** by the
    caller — the engine does not read it from a global process group, which is
    ambiguous once several groups (FSDP / TP / PP / EP) exist. Rank 0 is always
    the sender: only it opens the endpoint and drives the inference-side RPCs,
    while every rank still runs the trainer-side collectives. Backend subclasses
    add their own (positional) fields; `rank` is keyword-only so that ordering
    never conflicts.

    Every concrete subclass sets a class-level `backend` string (the same key it
    registers under in `WeightTransferTrainerFactory`). The factory reads it to
    dispatch, so callers pass only the init info/ It is a `ClassVar`
    (a fixed per-backend constant), so it is not an ``__init__`` field.
    """

    backend: ClassVar[str]

    rank: int = field(kw_only=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "backend", None):
            raise TypeError(
                f"{cls.__name__} must set a class-level `backend` string "
                "(the WeightTransferTrainerFactory registry key)."
            )

    @property
    def is_sender(self) -> bool:
        return self.rank == 0


@dataclass
class WeightTransferUpdateInfo(ABC):  # noqa: B024
    """Base class for backend-specific weight update info."""

    pass


# API-level request classes (accept dicts for backend-agnostic serialization)
@dataclass
class WeightTransferInitRequest:
    """API-level weight transfer initialization request."""

    init_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightTransferUpdateRequest:
    """API-level weight update request."""

    update_info: dict[str, Any] = field(default_factory=dict)


class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    """
    Base class for weight transfer engines that handle transport of model weights
    from a trainer to inference workers.

    This abstraction separates weight transfer transport logic from the worker
    implementation, allowing different backends (NCCL, CUDA IPC, RDMA[TODO]) to be
    plugged in.

    Each engine owns its full weight-update lifecycle: `start_weight_update`,
    `update_weights`, and `finish_weight_update`. Checkpoint-format engines use
    this explicit boundary to defer model-wide post-load processing until the
    trainer declares the update complete. Runtime-format engines copy already
    processed tensors directly and use the boundary only for protocol state.

    Subclasses should define:
        init_info_cls: Type of backend-specific initialization info
        update_info_cls: Type of backend-specific update info
    """

    # Subclasses should override these class attributes
    init_info_cls: type[TInitInfo]
    update_info_cls: type[TUpdateInfo]

    supports_draft_weight_update: bool = True

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            vllm_config: The full vLLM config (provides parallel/model config)
            device: The device this worker's model lives on
            model: The local model instance which will receive the weights
        """
        self.config = config
        self.vllm_config = vllm_config
        self.parallel_config: ParallelConfig = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.model = model
        self._default_model_config = self.model_config
        self._default_model = model

    def set_weight_update_target(
        self,
        model: torch.nn.Module,
        model_config: Any,
    ) -> None:
        """Set the model that will receive the active weight update."""
        self.model = model
        self.model_config = model_config

    def reset_weight_update_target(self) -> None:
        """Restore weight updates to the engine's default target model."""
        self.model = self._default_model
        self.model_config = self._default_model_config

    def parse_init_info(self, init_dict: dict[str, Any]) -> TInitInfo:
        """
        Construct typed init info from dict with validation.

        Args:
            init_dict: Dictionary containing backend-specific initialization parameters

        Returns:
            Typed backend-specific init info dataclass

        Raises:
            ValueError: If init_dict is invalid for this backend
        """
        try:
            return self.init_info_cls(**init_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid init_info for {self.__class__.__name__}: {e}"
            ) from e

    def parse_update_info(self, update_dict: dict[str, Any]) -> TUpdateInfo:
        """
        Construct typed update info from dict with validation.

        Args:
            update_dict: Dictionary containing backend-specific update parameters

        Returns:
            Typed backend-specific update info dataclass

        Raises:
            ValueError: If update_dict is invalid for this backend
        """
        try:
            return self.update_info_cls(**update_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid update_info for {self.__class__.__name__}: {e}"
            ) from e

    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.

        Args:
            init_info: Backend-specific initialization info
        """
        raise NotImplementedError

    @abstractmethod
    def start_weight_update(self) -> None:
        """
        Prepare the engine for a new weight update.

        Checkpoint-format engines open a modelwise reload transaction. Runtime-
        format engines open a direct in-place update session.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        Checkpoint-format engines run modelwise finalization here. Runtime-
        format engines only close their protocol session; values were already
        copied in place.
        """
        raise NotImplementedError

    def abort_weight_update(self) -> None:
        """Abort an active update and restore the serving model if needed."""

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """
        Receive one weight update chunk and load it into the model.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        typed_update_info = self.parse_update_info(update_info)
        self.receive_weights(typed_update_info)
        # NCCL broadcast / IPC paths may be asynchronous. Synchronize here so the
        # next step uses the new weights.
        torch.accelerator.synchronize()

    def rank_local_checkpoint_names(self) -> frozenset[str] | None:
        """Return checkpoint sources consumed by this worker's rank."""
        if self.config.weight_format != "checkpoint":
            return None
        from vllm.model_executor.model_loader.reload import (
            get_rank_sharding_manifest,
        )

        manifest = get_rank_sharding_manifest(self.model)
        if manifest.state == "legacy":
            return None
        if manifest.state != "exact":
            raise ValueError(
                f"Worker has no exact rank-local sharding manifest: {manifest.reason}"
            )
        return frozenset(manifest.source_names)

    def filter_rank_local_weights(
        self, weights: Iterator[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        allowed = self.rank_local_checkpoint_names()
        if allowed is None:
            yield from weights
            return
        for name, tensor in weights:
            if name in allowed:
                yield name, tensor

    @abstractmethod
    def receive_weights(self, update_info: TUpdateInfo) -> None:
        """
        Receive weights from the trainer and load them into the model.

        Args:
            update_info: Backend-specific update info containing parameter metadata
                        and any backend-specific data
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the weight transfer engine.
        This should be called when the worker is shutting down.
        """
        raise NotImplementedError


@runtime_checkable
class VLLMWeightSyncClient(Protocol):
    """Trainer-side stub for the inference engine's weight-sync control plane.

    Mirrors the weight-sync methods that the inference engine exposes
    (`EngineClient` / the HTTP RLHF routes / Ray actors). A
    `TrainerWeightTransferEngine` drives the full handshake through this
    protocol so trainer code never has to know the transport.

    All methods are synchronous and accept plain dicts (matching what the
    inference side already accepts). Concurrency that some backends need
    (e.g. NCCL must run `update_weights` concurrently with the trainer-side
    broadcast) is the engine's responsibility, not the client's, so the
    protocol stays a flat four-method surface that any wrapper can implement.

    The protocol is structural (PEP 544), so user implementations need only
    define these four methods — no import or subclassing required.
    """

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None: ...

    def start_weight_update(self) -> None: ...

    def update_weights(self, update_info: dict[str, Any]) -> None: ...

    def finish_weight_update(self, weight_version: str | None = None) -> None: ...


class TrainerWeightTransferEngine(ABC, Generic[TTrainerInitInfo]):
    """Trainer-side weight transfer engine.

    Symmetric to `WeightTransferEngine` but lives in the training process.
    Constructed via the `trainer_init` factory classmethod; carries any
    backend-specific state (NCCL communicators, IPC device info, transfer
    plans) on `self`.

    Unlike the worker engine, the trainer side does not take a
    `WeightTransferConfig`: the backend is selected from the init info's
    `backend` `ClassVar` (so callers pass only the init info), and the static
    wire params (packed, buffer sizes) ride the backend-specific
    `TrainerInitInfo`, which the sender also propagates to the worker at the init
    handshake.

    Multi-rank trainers: `trainer_init` and `send_weights` are
    called on *every* trainer rank. Rank 0 is the sender, resolved once at
    `trainer_init` into `is_sender`. Non-sender ranks still run every
    collective (iterating the source, metadata export, IPC handle all-gather) so
    the group stays aligned, but each engine explicitly guards the control-plane
    RPCs and the transmit on `self.is_sender`, so only the sender touches the
    client.

    Subclasses should define:
        init_info_cls: Type of backend-specific trainer init info
    """

    # Subclasses should override this class attribute
    init_info_cls: type[TTrainerInitInfo]

    def __init__(
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
        is_sender: bool = True,
    ) -> None:
        self.is_sender = is_sender
        # The real client is held on every rank; each engine only *calls* it when
        # `is_sender`, so non-sender ranks never touch the wire.
        self.client = client
        self.source = source
        self._rank_local_source: WeightSource | None = None

    def rank_local_source(self) -> WeightSource:
        """Filter the trainer source by the union of worker load manifests."""
        if self.source is None:
            raise RuntimeError("Weight transfer has no trainer source")
        if self._rank_local_source is not None:
            return self._rank_local_source

        names: set[str] | None = None
        getter = getattr(self.client, "get_rank_sharding_manifests", None)
        if self.is_sender and callable(getter):
            manifests = getter()
            names = set()
            for manifest in manifests:
                state = (
                    manifest.get("state")
                    if isinstance(manifest, dict)
                    else getattr(manifest, "state", None)
                )
                if state != "exact":
                    reason = (
                        manifest.get("reason")
                        if isinstance(manifest, dict)
                        else getattr(manifest, "reason", None)
                    )
                    raise ValueError(
                        "Inference worker has no exact rank-local sharding "
                        f"manifest: {reason}"
                    )
                source_names = (
                    manifest.get("source_names", ())
                    if isinstance(manifest, dict)
                    else manifest.source_names
                )
                names.update(source_names)

        if torch.distributed.is_initialized():
            payload = [names]
            torch.distributed.broadcast_object_list(payload, src=0)
            names = payload[0]

        self._rank_local_source = (
            self.source if names is None else RankLocalWeightSource(self.source, names)
        )
        return self._rank_local_source

    @classmethod
    @abstractmethod
    def trainer_init(
        cls,
        init_info: TTrainerInitInfo,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
    ) -> Self:
        """Rendezvous with the inference side and return a ready instance.

        Called on every trainer rank. The sender drives the full handshake via
        `client` (build the worker-side init info, call
        `client.init_weight_transfer_engine`, open the trainer-side endpoint);
        non-sender ranks skip the rendezvous and the RPC.
        """
        raise NotImplementedError

    @abstractmethod
    def send_weights(self) -> None:
        """Push weights to inference workers and drive the full update round
        trip: `start_weight_update`, `update_weights` (run concurrently with the
        trainer-side broadcast when the backend requires it), then
        `finish_weight_update`. Called on every trainer rank.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Tear down communicators / process groups. Default no-op."""
