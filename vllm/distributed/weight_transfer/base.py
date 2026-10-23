# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for weight transfer engines."""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig

TInitInfo = TypeVar("TInitInfo", bound="WeightTransferInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo", bound="WeightTransferUpdateInfo")
TTrainerInitInfo = TypeVar("TTrainerInitInfo", bound="TrainerInitInfo")
WeightTransferUpdatePayload = dict[str, Any] | list[dict[str, Any]]


def _stack_key(name: str) -> tuple[str, int] | None:
    """``(prefix, index)`` of the OUTERMOST integer segment, or None if there is
    none.

    Outermost is what keeps a MoE layer whole:
    ``model.layers.3.mlp.experts.7.w1`` keys on the layer, not the expert.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[:i]), int(part)
    return None


def layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition flat parameter names into one group per decoder layer, keyed on
    the outermost index segment of each name.

    This defines what a *group index* means for backends that transfer a layer
    at a time: index *g* names the same group on every trainer rank and every
    consumer, because it is derived from `metadata()` order, which every rank
    shares.

    Keying on the index rather than a literal prefix needs no per-architecture
    naming table: ``model.layers.0.``, ``model.language_model.layers.0.``,
    ``transformer.h.0.``, ``backbone.layers.0.`` and a vision tower's
    ``visual.blocks.0.`` all partition alike. Matching one fixed prefix does not,
    and its failure is silent — every name lands in a single group holding the
    whole model, which defeats the per-layer bound below.

    Un-indexed names split by POSITION relative to the first indexed one: the pre
    block (embeddings) and the post block (the final norm, `lm_head`, and any
    inter-stack projector). Post lands last however early it arrived, which is
    what a pipeline-parallel source needs — Megatron-Bridge streams the last
    stage's output block *before* its layers.

    Stacks come out in first-appearance order of their prefix and ascending index
    within it, whatever order the source yielded them, so a source can normalize
    an arbitrary export order by flattening this partition.

    Backends that gather and free per group (sharded RDT) also use it as the unit
    of transfer, which bounds their buffer sizes: without it a whole model becomes
    one chunk.
    """
    pre: list[str] = []
    post: list[str] = []
    stacks: dict[tuple[str, int], list[str]] = {}
    order: list[tuple[str, int]] = []
    for name in names:
        key = _stack_key(name)
        if key is None:
            (post if order else pre).append(name)
            continue
        if key not in stacks:
            stacks[key] = []
            order.append(key)
        stacks[key].append(name)

    prefix_rank: dict[str, int] = {}
    for key in order:
        prefix_rank.setdefault(key[0], len(prefix_rank))
    order.sort(key=lambda key: (prefix_rank[key[0]], key[1]))

    groups: list[list[str]] = [pre] if pre else []
    groups += [stacks[key] for key in order]
    if post:
        groups.append(post)
    return groups


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
class Replicate:
    """Every coordinate along this mesh axis holds the whole extent."""


@dataclass(frozen=True)
class Shard:
    """This mesh axis splits tensor dim `dim` the way `torch.chunk` does."""

    dim: int


Placement = Replicate | Shard


@dataclass(frozen=True)
class ParamMeta:
    """One output parameter: its name, wire dtype and full (HF) shape, plus
    which trainer ranks own it and how it is laid out across them.

    Metadata is identical on every trainer rank, ownership included, so every
    rank and every engine derives the same plan from it without a runtime
    exchange.

    Attributes:
        owners: Global trainer ranks (`TrainerInitInfo.rank`) that own this
            parameter, in mesh order. `None` means every rank.
        mesh_shape: Row-major shape of the mesh over `owners`. Defaults to
            `(len(owners),)`.
        placements: One `Replicate` or `Shard` per mesh axis, describing what
            `WeightSource.iter_local` yields on each owner. `None` means
            replicated; a source without `iter_local` leaves it unused.

    """

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    owners: tuple[int, ...] | None = field(default=None, kw_only=True)
    mesh_shape: tuple[int, ...] | None = field(default=None, kw_only=True)
    placements: tuple[Placement, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.owners is None:
            if self.mesh_shape is not None or self.placements is not None:
                raise ValueError(
                    f"parameter {self.name!r} declares a mesh or placements "
                    "without owners"
                )
            return
        owners = tuple(self.owners)
        if (
            not owners
            or any(not isinstance(r, int) or isinstance(r, bool) for r in owners)
            or min(owners) < 0
            or len(set(owners)) != len(owners)
        ):
            raise ValueError(
                f"parameter {self.name!r} owners must be distinct non-negative "
                f"ranks, got {self.owners}"
            )
        mesh_shape = (len(owners),) if self.mesh_shape is None else self.mesh_shape
        mesh_shape = tuple(mesh_shape)
        if any(d <= 0 for d in mesh_shape) or math.prod(mesh_shape) != len(owners):
            raise ValueError(
                f"parameter {self.name!r} mesh_shape {mesh_shape} does not cover "
                f"its {len(owners)} owners"
            )
        placements = self.placements
        if placements is not None:
            placements = tuple(placements)
            if len(placements) != len(mesh_shape):
                raise ValueError(
                    f"parameter {self.name!r} has {len(placements)} placements "
                    f"for a {len(mesh_shape)}-D mesh"
                )
            for p in placements:
                if isinstance(p, Shard):
                    if not 0 <= p.dim < len(self.shape):
                        raise ValueError(
                            f"parameter {self.name!r} shards dim {p.dim} of a "
                            f"{len(self.shape)}-D tensor"
                        )
                elif not isinstance(p, Replicate):
                    raise TypeError(
                        f"parameter {self.name!r} placement {p!r} is neither "
                        "Replicate nor Shard"
                    )
            if all(isinstance(p, Replicate) for p in placements):
                placements = None
        object.__setattr__(self, "owners", owners)
        object.__setattr__(self, "mesh_shape", mesh_shape)
        object.__setattr__(self, "placements", placements)


def owned_names(metas: Sequence[ParamMeta], rank: int) -> list[str]:
    """The names `rank` owns, in metadata order: exactly what it iterates."""
    return [m.name for m in metas if m.owners is None or rank in m.owners]


def _mesh_coords(meta: ParamMeta, rank: int) -> tuple[int, ...]:
    """`rank`'s coordinate in `meta`'s owner mesh."""
    if meta.owners is None or meta.mesh_shape is None:
        raise ValueError(f"parameter {meta.name!r} declares no owner mesh")
    if rank not in meta.owners:
        raise ValueError(f"rank {rank} does not own parameter {meta.name!r}")
    index = meta.owners.index(rank)
    coords: list[int] = []
    for size in reversed(meta.mesh_shape):
        index, coord = divmod(index, size)
        coords.append(coord)
    return tuple(reversed(coords))


def _chunk_len(length: int, chunks: int, index: int) -> int:
    size = -(-length // chunks)
    return max(0, min(size, length - index * size))


def local_shape(meta: ParamMeta, rank: int) -> tuple[int, ...]:
    """The shape of `rank`'s piece of `meta`, as `iter_local` yields it.

    Shards apply in mesh-axis order, so two axes sharding one dim split it
    outer axis first.
    """
    if meta.placements is None:
        return meta.shape
    assert meta.mesh_shape is not None
    coords = _mesh_coords(meta, rank)
    shape = list(meta.shape)
    for axis, p in enumerate(meta.placements):
        if isinstance(p, Shard):
            shape[p.dim] = _chunk_len(shape[p.dim], meta.mesh_shape[axis], coords[axis])
    return tuple(shape)


WeightBatch = list[tuple[str, torch.Tensor]]


class WeightSource(ABC):
    """The trainer's weights, as metadata plus one or two iteration modes.

    * `metadata()` declares every parameter of the whole model, on every rank:
      name, wire dtype, full shape, and which ranks own it in what layout (see
      `ParamMeta`). It transfers nothing.
    * `iter_full()` yields each owned parameter as a full tensor.
    * `iter_local()` yields each owned parameter as this rank's piece, shaped
      by `ParamMeta.placements`.

    A source implements one mode or both, and nothing is derived from one mode
    to the other. Each engine consumes the mode it needs: broadcast, IPC and
    RDT consume `iter_full`, NCCL M2N consumes `iter_local`.

    Both modes yield batches of `(name, tensor)` pairs chosen by the source:
    one tensor, one layer, one export task. Joined together, the batches are
    exactly `owned_names(metadata(), rank)` in metadata order. A batch is
    resident all at once, so its size is the source's memory trade-off; engines
    re-chunk the stream for transfer themselves.

    `iter_full` may gather, but only among the owners of the parameter being
    gathered, and every owner iterates in metadata order so the gathers line up.
    `iter_local` runs no collectives, yields `torch.Tensor`s only, and enqueues
    any work on the caller's current stream.

    Each call to a mode must start a fresh pass.
    """

    @abstractmethod
    def metadata(self) -> list[ParamMeta]:
        """Declare every parameter, without transferring anything.

        Must be identical on every trainer rank, including ownership: engines
        plan from it without exchanging it. May be expensive on first call for
        producers that must materialize to learn shapes, in which case cache.
        """
        raise NotImplementedError

    def iter_full(self) -> Iterator[WeightBatch]:
        """Yield batches of full tensors for the names this rank owns."""
        raise NotImplementedError(f"{type(self).__name__} does not implement iter_full")

    def iter_local(self) -> Iterator[WeightBatch]:
        """Yield batches of this rank's pieces for the names it owns."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement iter_local"
        )


def _dtensor_layout(
    param: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[Placement, ...] | None] | None:
    """`(owners, mesh_shape, placements)` of a DTensor, or None for a plain
    tensor. Placements are None when the layout is not plain `Shard` /
    `Replicate` (e.g. `Partial`, or FSDP+TP's strided shards)."""
    device_mesh = getattr(param, "device_mesh", None)
    if device_mesh is None:
        return None
    from torch.distributed.tensor import Replicate as TorchReplicate
    from torch.distributed.tensor import Shard as TorchShard

    grid = device_mesh.mesh
    placements: list[Placement] = []
    for p in param.placements:  # type: ignore[attr-defined]
        if type(p) is TorchShard:
            placements.append(Shard(p.dim))
        elif type(p) is TorchReplicate:
            placements.append(Replicate())
        else:
            return tuple(grid.flatten().tolist()), tuple(grid.shape), None
    return tuple(grid.flatten().tolist()), tuple(grid.shape), tuple(placements)


def _is_member(param: torch.Tensor) -> bool:
    device_mesh = getattr(param, "device_mesh", None)
    return device_mesh is None or device_mesh.get_coordinate() is not None


class ModuleSource(WeightSource):
    """`WeightSource` over `module.named_parameters()`: plain, FSDP or DTensor.

    A plain tensor is owned by every rank. A `DTensor` is owned by the global
    ranks of its device mesh, with the mesh's shape and the DTensor's
    placements. `iter_full` gathers each DTensor with `full_tensor()` (a
    collective among its owners); `iter_local` yields `to_local()`. A DTensor
    whose placements are not plain `Shard` / `Replicate` is declared replicated,
    and `iter_local` gathers it. `metadata()` reads global shapes and never
    gathers.

    The mesh ranks are taken as trainer ranks, so the default process group's
    ranks must be the `TrainerInitInfo.rank`s.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def metadata(self) -> list[ParamMeta]:
        metas = []
        for name, p in self._module.named_parameters():
            layout = _dtensor_layout(p)
            if layout is None:
                metas.append(ParamMeta(name, p.dtype, tuple(p.shape)))
                continue
            owners, mesh_shape, placements = layout
            metas.append(
                ParamMeta(
                    name,
                    p.dtype,
                    tuple(p.shape),
                    owners=owners,
                    mesh_shape=mesh_shape,
                    placements=placements,
                )
            )
        return metas

    def iter_full(self) -> Iterator[WeightBatch]:
        for name, param in self._module.named_parameters():
            if _is_member(param):
                yield [(name, materialize_full_tensor(param))]

    def iter_local(self) -> Iterator[WeightBatch]:
        for name, param in self._module.named_parameters():
            if not _is_member(param):
                continue
            layout = _dtensor_layout(param)
            if layout is None:
                yield [(name, param)]
            elif layout[2] is None:
                yield [(name, materialize_full_tensor(param))]
            else:
                yield [(name, param.to_local())]  # type: ignore[attr-defined]


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

    update_info: WeightTransferUpdatePayload = field(default_factory=dict)


class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    """Base class for weight transfer engines that handle transport of model weights
    from a trainer to inference workers.

    This abstraction separates weight transfer transport logic from the worker
    implementation, allowing different backends (NCCL, CUDA IPC, RDMA[TODO]) to be
    plugged in.

    Each engine owns its full weight-update lifecycle: `start_weight_update`,
    `update_weights`, and `finish_weight_update`. Layerwise reloading (used by
    checkpoint-format engines) is opted into per engine by running it inside
    `start_weight_update`/`finish_weight_update`. Engines that apply weights in
    place (e.g. sparse patches) leave those methods as no-ops.

    Subclasses should define:
        init_info_cls: Type of backend-specific initialization info
        update_info_cls: Type of backend-specific update info
    """

    # Subclasses should override these class attributes
    init_info_cls: type[TInitInfo]
    update_info_cls: type[TUpdateInfo]

    supports_draft_weight_update: bool = True

    defers_processing: bool = False
    """Whether `update_weights` returns before the weights are on the device.

    An engine that pipelines its GPU post-processing onto background threads
    cannot let `update_weights` synchronize the device — that would block on those
    threads and serialize the pipeline. Such an engine sets this True, omits the
    per-update sync, and guarantees completion in `finish_weight_update` instead.

    Callers that go through `finish_weight_update` need do nothing: the engine
    drains there. A caller that instead drives the tail itself — running its own
    `finalize_layerwise_reload`, say — must read this flag and call
    `drain_pending()` first, because with it set a returned `update_weights` means
    "queued", not "applied".
    """

    def drain_pending(self) -> None:
        """Block until every deferred update has been applied to the model.

        The companion to `defers_processing`: a caller that has taken over the
        update tail calls this to re-establish the guarantee that
        `finish_weight_update` would otherwise have given it. Idempotent, and a
        no-op by default — an engine that processes synchronously has nothing to
        drain, so this is always safe to call.
        """

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        """Initialize the weight transfer engine.

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
        """Construct typed init info from dict with validation.

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
        """Construct typed update info from dict with validation.

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
        """Initialize the weight transfer mechanism.
        This is called once at the beginning of training.

        Args:
            init_info: Backend-specific initialization info

        """
        raise NotImplementedError

    @abstractmethod
    def start_weight_update(self) -> None:
        """Prepare the engine for a new weight update.

        Engines that receive weights in checkpoint format initialize layerwise reloading
        here, else this is typically a no-op.
        See: https://docs.vllm.ai/en/latest/training/layerwise/ for more details.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_weight_update(self) -> None:
        """Finalize the current weight update.

        Checkpoint-format engines finalize layerwise reloading here; engines
        that apply weights in place leave this as a no-op.
        """
        raise NotImplementedError

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Receive one weight update chunk and load it into the model.

        Args:
            update_info: Dictionary containing backend-specific update info

        """
        typed_update_info = self.parse_update_info(update_info)
        self.receive_weights(typed_update_info)
        # NCCL broadcast / IPC paths may be asynchronous. Synchronize here so the
        # next step uses the new weights.
        torch.accelerator.synchronize()

    @abstractmethod
    def receive_weights(self, update_info: TUpdateInfo) -> None:
        """Receive weights from the trainer and load them into the model.

        Args:
            update_info: Backend-specific update info containing parameter metadata
                        and any backend-specific data

        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release transfer resources without shutting down the inference worker."""
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

    def update_weights(self, update_info: WeightTransferUpdatePayload) -> None: ...

    def finish_weight_update(self, weight_version: str | None = None) -> None: ...


class TrainerWeightTransferEngine(ABC, Generic[TTrainerInitInfo]):
    """Trainer-side weight transfer engine.

    Symmetric to `WeightTransferEngine` but lives in the training process.
    Constructed via the `trainer_init` factory classmethod; carries any
    backend-specific state (NCCL communicators, IPC device info, transfer
    plans) on `self`. Full-resync backends (NCCL, IPC) take a `WeightSource` at
    `trainer_init` and replay it each round via the no-argument
    `send_weights()`. Backends that push per-round deltas instead (e.g. sparse
    patches) leave `source` as `None` and take their payload as a `send_weights`
    argument.

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
        rank: int = 0,
    ) -> None:
        self.is_sender = is_sender
        self.rank = rank
        # The real client is held on every rank; each engine only *calls* it when
        # `is_sender`, so non-sender ranks never touch the wire.
        self.client = client
        self.source = source

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
