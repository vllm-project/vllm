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


def layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition flat parameter names into the pre block, one group per decoder
    layer, then the post block (keys on ``model.layers.<N>.``).

    This defines what a *group index* means for `WeightSource.owned_groups` and
    `WeightSource.iter_groups`: index *g* names the same group on every trainer
    rank and every consumer, because it is derived from one rank's `metadata()`
    order. The split is by POSITION relative to the first layer name, not by name
    class, so flattening a partition always reproduces the input order.

    Backends that gather and free per group (sharded RDT) also use it as the unit
    of transfer, which bounds their arena sizes: without it a whole model becomes
    one chunk.
    """
    pre: list[str] = []
    layers: dict[int, list[str]] = {}
    post: list[str] = []
    seen = False
    for n in names:
        if n.startswith("model.layers."):
            seen = True
            idx = int(n[len("model.layers.") :].split(".", 1)[0])
            layers.setdefault(idx, []).append(n)
        elif not seen:
            pre.append(n)
        else:
            post.append(n)
    groups: list[list[str]] = []
    if pre:
        groups.append(pre)
    for i in sorted(layers):
        groups.append(layers[i])
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
      Megatron export), so the ranks that share a parameter must iterate it in
      the same order in lockstep, or they deadlock.
    * `owned_groups()` — which gather groups this rank holds, for producers that
      are split so each rank holds only part of the model. Defaults to all.
    * `iter_groups()` — the same stream batched per gather group (see
      `layerwise_groups`). Defaults to batching `__iter__`; override to
      materialize a whole group in one step.

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

    def owned_groups(self) -> list[int] | None:
        """Indices of the gather groups this rank holds, or None for all of them.

        Groups partition `metadata()`'s name order into the pre block, one group
        per decoder layer, then the post block (`layerwise_groups`), so index *g*
        means the same group on every rank and on every consumer. Override when
        producers are split so each holds only part of the model — pipeline
        parallelism, where a layer's gather spans only the ranks that hold it.
        The default owns everything, which is the gather-to-all layout.

        Two requirements come with overriding it:

        * `metadata()` must still describe the WHOLE model on every rank. The
          group partition, the iteration checks and the consumers' pull plans are
          all built from one rank's metadata, so a rank that reported only its own
          share would leave the rest of the model silently un-transferred. The
          sharded-RDT engine cross-checks this across ranks at init.
        * iteration must yield ONLY the owned groups, in metadata order. A rank
          that does not hold a group must not iterate it: the group's gather is a
          collective among its owners alone, so a non-owner joining in has
          nothing to contribute and nothing to receive.

        Returns:
            Sorted group indices, or None to own every group.
        """
        return None

    def expert_ownership(self) -> "tuple[list[int], int] | None":
        """Name-level ownership WITHIN owned groups, or None for none.

        Sibling of `owned_groups`, one grain finer: `owned_groups` says which
        gather groups this rank takes part in at all; this says which NAMES
        inside those groups it actually holds. The only realistic name-level
        sharding is expert parallelism — a rank holds whole experts, its EP
        peers hold the others — hence the concrete name.

        Returns ``(name_ep_rank, my_ep_rank)``:

        * ``name_ep_rank[i]`` stamps ``metadata()[i]``: the producer EP rank
          holding that name, or ``-1`` for names replicated across EP
          (attention, norms, router/gate, embeddings, dense layers). Like
          ``metadata()``, it must describe the WHOLE model identically on
          every rank — the consumers' pull routing is built from the sender's
          copy alone, and the engine digest-checks it across ranks.
        * ``my_ep_rank`` is THIS rank's own EP coordinate. The engine
          all-gathers the coordinates (the mirror of the ``owned_groups`` ->
          ``group_owners`` split), so the source never sees the fleet.

        Three invariants come with declaring it (sources enforce their own
        preconditions):

        * Stamps must be truthful against iteration: within an owned group, a
          name stamped ``-1`` or ``my_ep_rank`` yields a real tensor; any other
          stamp yields ``None`` (the name still appears, in metadata order, so
          the group-order check holds — only the data is absent).
        * All ranks must derive the identical ``name_ep_rank`` list.
        * Ranks declaring the same coordinate must hold identical name sets for
          every group they co-own — routing spreads consumers across a
          coordinate's owner set, so its members must be interchangeable.

        The default (None) means every name is held by every owner of its
        group — equivalent to all ``-1`` — and is correct for any source whose
        iteration yields real tensors for everything it owns.
        """
        return None

    def groups(self) -> list[list[str]]:
        """This rank's gather groups, in metadata order: `layerwise_groups` over
        `metadata()`, restricted to `owned_groups()` when that is overridden.

        The indices are normalized (sorted, de-duplicated) exactly as a trainer
        engine normalizes them, so the two agree by construction: a source whose
        `owned_groups()` came back unsorted would otherwise pair each group with
        the wrong batch from this stream.
        """
        groups = layerwise_groups([m.name for m in self.metadata()])
        owned = self.owned_groups()
        if owned is None:
            return groups
        return [groups[g] for g in sorted({int(g) for g in owned})]

    def iter_groups(self) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
        """Yield one `(names, tensors)` batch per group from `groups()`.

        The default drives `__iter__` and batches its output, checking as it goes
        that the names arrive in metadata order — ranks sharing a parameter
        materialize it with a collective, so a rank that iterates out of order
        deadlocks its peers rather than returning wrong data.

        Override when a backend can produce a whole group at once. Materializing
        is usually a collective, and driving it per group instead of per tensor
        turns ~37k generator resumes into ~95 on a per-expert MoE model (worth
        ~0.9s per sync there). An override must yield the same batches in the same
        order as this default.
        """
        it = iter(self)
        for group in self.groups():
            names: list[str] = []
            tensors: list[torch.Tensor] = []
            for expected in group:
                name, tensor = next(it)
                if name != expected:
                    raise RuntimeError(
                        f"WeightSource yielded {name!r} but expected "
                        f"{expected!r}; iteration order must match metadata()."
                    )
                names.append(name)
                tensors.append(tensor)
            yield names, tensors


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

        Engines that receive weights in checkpoint format initialize layerwise reloading
        here, else this is typically a no-op.
        See: https://docs.vllm.ai/en/latest/training/layerwise/ for more details.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        Checkpoint-format engines finalize layerwise reloading here; engines
        that apply weights in place leave this as a no-op.
        """
        raise NotImplementedError

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
    ) -> None:
        self.is_sender = is_sender
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
