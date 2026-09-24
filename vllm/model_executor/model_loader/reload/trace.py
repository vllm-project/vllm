# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Track checkpoint arrivals and finish registered reload units independently.

There are three different identifiers in this module:

* ``ReloadState.key`` identifies a reload unit, usually a module's full path.
* A ``role`` identifies a loadable parameter within that unit, e.g. ``w13_weight``.
* ``SlotKey`` identifies one loader call's sharding coordinates within a role,
  e.g. ``w13_weight`` with ``expert_id=3`` and ``shard_id="w1"``.

Cold loading records parameter metadata and ordinary layers' expected slots.
After cold-load processing, binding records the runtime tensors that later rounds
must preserve. Reloading redirects existing weight loaders to policy-selected
destinations; each successful arrival can immediately finish a ready layer.
RoutedExperts use a fresh placement-dependent plan instead of observed slots.

Inference and EPLB must be quiescent throughout a reload round. GPU arrivals and
conversions must run on an ordered stream, with transport adapters protecting
borrowed receive buffers. This module neither coordinates ranks nor synchronizes
CUDA work itself. It is opt-in; the legacy layerwise helper remains separate.
"""

import inspect
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm.logger import init_logger

from .meta import to_meta_tensor

if TYPE_CHECKING:
    from .moe import RoutedExpertsReloadPlan

logger = init_logger(__name__)


class ReloadError(RuntimeError):
    """A reload contract was violated, such as a missing or duplicate slot.

    An error does not imply that runtime weights are unchanged. Inspect
    ``runtime_modified`` when diagnosing failures, but do not treat it as a
    rollback guarantee. A poisoned tracer must not be reused.
    """


@dataclass(frozen=True)
class SlotKey:
    """Immutable identity of one expected parameter-shard arrival.

    Attributes:
        role: Parameter name within a ReloadState, not a model-wide path.
        arguments: Hashable loader arguments identifying the shard. Tensor
            objects and tensor values are deliberately excluded.

    Example:
        A fused QKV parameter can have three slots with the same role::

            SlotKey("weight", (("shard_id", "q"),))
            SlotKey("weight", (("shard_id", "k"),))
            SlotKey("weight", (("shard_id", "v"),))

        A slot describes a loader invocation, not an element count or byte range.
        Different keys are not checked for overlapping destination slices.
    """

    role: str
    arguments: tuple[tuple[str, Any], ...]


def _argument_key(value: Any) -> Any:
    """Normalize supported shard identifiers into hashable values.

    Lists and tuples are normalized recursively. Arbitrary objects, including
    tensors, are rejected rather than keyed by a transient identity.
    """
    if isinstance(value, (tuple, list)):
        return tuple(_argument_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_argument_key(key), _argument_key(item)) for key, item in value.items()
            )
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ReloadError(f"Unsupported loader argument in arrival key: {type(value)}")


def _slot_key(role: str, bound: inspect.BoundArguments) -> SlotKey:
    """Build an ordinary layer's slot from normalized loader arguments.

    Callers bind the original signature and apply defaults first, so positional
    and keyword forms produce the same key. The parameter, incoming tensor,
    checkpoint name, and return-value control do not identify a shard.
    RoutedExperts construct their expert/shard keys in their own reload plan.
    """
    arguments = tuple(
        (name, _argument_key(value))
        for name, value in bound.arguments.items()
        if name not in ("param", "loaded_weight", "weight_name", "return_success")
    )
    return SlotKey(role, arguments)


@dataclass
class SlotTable:
    """Expected and successfully loaded slots for one reload unit.

    Ordinary units retain their cold-observed ``expected`` set across rounds.
    Expert units rebuild it from current placement. ``arrived`` is reset each
    round and updated only after the original loader returns successfully.
    """

    expected: set[SlotKey] = field(default_factory=set)
    arrived: set[SlotKey] = field(default_factory=set)

    def validate(self, key: SlotKey) -> None:
        """Reject an unknown or already-arrived slot before writing any data.

        This is a check, not a reservation: the caller must record the arrival
        after loading succeeds. Concurrent loader calls are not supported.
        """
        if key not in self.expected:
            raise ReloadError(f"Unknown reload slot: {key}")
        if key in self.arrived:
            raise ReloadError(f"Duplicate reload slot: {key}")

    def missing(self) -> list[SlotKey]:
        """Return expected slots not yet loaded, in unspecified order."""
        return [key for key in self.expected if key not in self.arrived]


def _layout(tensor: torch.Tensor) -> tuple:
    """Snapshot address and layout invariants, excluding tensor contents.

    Values may change during reload; the address, shape, strides, offset, dtype,
    and device must remain unchanged. Object identity is checked separately.
    """
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


@dataclass
class ReloadTarget:
    """A runtime tensor whose object and layout must survive reload.

    Attributes:
        tensor: Original runtime object receiving converted values.
        resolve: Getter for its current location, typically
            ``lambda: layer.weight``. This detects attribute replacement.
        layout: Invariants captured at binding, after cold-load processing.
    """

    tensor: torch.Tensor
    resolve: Callable[[], torch.Tensor]
    layout: tuple = field(init=False)

    def __post_init__(self) -> None:
        """Capture the bound tensor's layout without copying its values."""
        self.layout = _layout(self.tensor)

    def validate(self) -> None:
        """Require the getter to resolve to the same object and layout.

        Raises:
            ReloadError: A parameter was replaced, moved, resized, or otherwise
                changed in a way that invalidates the bound runtime target.
        """
        if self.resolve() is not self.tensor or _layout(self.tensor) != self.layout:
            raise ReloadError("Runtime tensor identity or layout changed")

    @torch.no_grad()
    def copy_(self, source: torch.Tensor) -> None:
        """Copy converted values into the existing runtime object.

        Shape and dtype must match exactly; broadcasting and implicit dtype
        conversion are not accepted. An identical source object needs no copy.
        This method does not mark ReloadState.runtime_modified; policies should
        normally use ReloadState.copy_ instead.
        """
        self.validate()
        if source.shape != self.tensor.shape or source.dtype != self.tensor.dtype:
            raise ReloadError("Converted tensor does not match runtime shape/dtype")
        if source is not self.tensor:
            self.tensor.copy_(source)


class ReloadPolicy(Protocol):
    """Backend-specific destination selection, validation, and conversion.

    The tracer owns arrival accounting and scheduling. A policy owns knowledge
    of checkpoint versus kernel layout, including whether in-place loading is
    safe and which weights/scales must be processed together.
    """

    def bind(self, state: "ReloadState") -> None:
        """Capture backend invariants after cold processing, once per state.

        Roles with runtime counterparts are already bound. A policy may add derived
        targets and record kernel/config identities for later validation.
        """
        ...

    def validate(self, state: "ReloadState") -> None:
        """Check that the backend still satisfies this state's reload contract.

        Called at round start, before policy finish, and at global finish.
        Implementations must not rely on validation being called only once.
        """
        ...

    def destination(
        self, state: "ReloadState", role: str, bound: inspect.BoundArguments
    ) -> torch.Tensor:
        """Select a loader-compatible destination for one incoming shard.

        Args:
            state: Unit owning the destination and checkpoint storage.
            role: Loadable parameter receiving this shard.
            bound: Original loader arguments with defaults applied. Policies
                may adjust arguments, e.g. swap w1/w3 for a runtime layout.

        Returns:
            Tensor to pass as the original loader's ``param``. It can alias
            runtime storage or use separate checkpoint-format storage.
        """
        ...

    def finish(self, state: "ReloadState") -> None:
        """Convert a ready unit and write its results to bound runtime targets.

        All required slots and dependencies are complete before this call.
        Use state.work() before destructive conversion when checkpoint values
        must be preserved, and state.copy_() for converted outputs. The tracer
        marks completion and releases unpreserved checkpoint storage afterward.
        """
        ...


@dataclass
class ReloadState:
    """Persistent registration plus per-round data for one reload unit.

    A unit usually corresponds to a layer, but can also represent a dependent
    alias or derived-only operation. It is not the incoming checkpoint tensor.

    Attributes:
        key: Unique identifier in the tracer, usually a full module path.
        module: Live module containing the runtime parameters.
        roles: Parameter attribute names loaded through wrapped weight loaders.
            One role can contain many expert or TP shard slots.
        policy: Backend-specific loading destination and conversion logic.
        dependencies: Other state keys that must finish before this unit.
        expert_plan: Optional per-round expert placement and slot planner.
        slots: Expected and successful arrivals for this unit.
        targets: Bound runtime tensors, including any policy-derived outputs
            that are not directly loaded and therefore need not be in roles.
        metadata: Cold parameter meta tensors retaining shape, subclass, and
            loader attributes, rather than checkpoint numerical payloads.
        loaders: Original loader callable for each role.
        ignored: Ordinary cold-load slots whose loader returned False.
        checkpoint: Lazy per-role loading destinations in checkpoint layout.
            These can alias runtime storage; they are not necessarily allocations.
        complete: Whether policy.finish succeeded for the current round.
        preserve_checkpoint: Keep loaded checkpoint-layout values intact after
            conversion. These are local loader outputs, not a full checkpoint
            archive; the next round or abort clears them.
        runtime_modified: Conservative marker that runtime writes may have begun.
            It is set before writes and is neither a byte counter nor rollback.
        runtime_names: Optional checkpoint-role to runtime-attribute mapping.
            Each key is a cold-load parameter attribute name from ``roles``,
            relative to ``module``, not a full checkpoint file key.
            Each string value is the attribute name on the same module after
            cold PWAL; bind_runtime resolves it with getattr(module, value)
            and stores the target under the original role key.
            An omitted key defaults to the same attribute name as the role.
            For example, ``{"weight_scale_inv": "weight_scale"}`` binds the
            checkpoint role ``weight_scale_inv`` to ``module.weight_scale``.
            Renamed parameters are exposed under their checkpoint names only
            during a reload round; the canonical runtime objects never change.
            A None value disables automatic target binding for that input role,
            such as a discarded activation scale. If the input produces
            multiple outputs, policy.bind must explicitly bind those targets.

    Example:
        An FP8 linear unit can use ``roles=("weight", "weight_scale_inv")``.
        Its policy combines both loaded roles when converting to kernel layout.
    """

    key: str
    module: torch.nn.Module
    roles: tuple[str, ...]
    policy: ReloadPolicy
    dependencies: tuple[str, ...] = ()
    expert_plan: "RoutedExpertsReloadPlan | None" = None
    slots: SlotTable = field(default_factory=SlotTable)
    targets: dict[str, ReloadTarget] = field(default_factory=dict)
    metadata: dict[str, torch.Tensor] = field(default_factory=dict)
    loaders: dict[str, Callable] = field(default_factory=dict)
    ignored: set[SlotKey] = field(default_factory=set)
    checkpoint: dict[str, torch.Tensor] = field(default_factory=dict)
    complete: bool = False
    preserve_checkpoint: bool = False
    runtime_modified: bool = False
    runtime_names: dict[str, str | None] = field(default_factory=dict)

    def bind_target(self, role: str, resolve: Callable[[], torch.Tensor]) -> None:
        """Record a runtime target and a getter used to detect replacement.

        Called after cold processing. ``role`` can also name a derived output
        absent from self.roles, such as a backend-generated scale.

        Example:
            ``state.bind_target("weight", lambda: layer.weight)``
        """
        self.targets[role] = ReloadTarget(resolve(), resolve)

    def prepare_sources(self, *, reuse_roles: tuple[str, ...]) -> None:
        """Prepare this round's canonical inputs without changing live tensors.

        Policies explicitly opt roles into storage reuse after auditing their
        conversion. Old values are disposable; every expected shard must arrive
        before finish. Only dense storage can provide a compact loading view.
        Dtype changes, insufficient capacity and incompatible layouts fall back
        to staging. This never reinterprets encoded scale bytes as another dtype.
        """
        for role in self.roles:
            alias = role in reuse_roles
            view = None
            target = self.targets.get(role)
            if alias and target is not None and not self.preserve_checkpoint:
                runtime = target.tensor
                meta = self.metadata[role]
                if (
                    runtime.dtype == meta.dtype
                    and runtime.numel() >= meta.numel()
                    and meta.is_contiguous()
                ):
                    # Sorting dimensions exposes physical order for dense
                    # transposes without allocating or changing runtime strides.
                    dims = sorted(
                        range(runtime.ndim),
                        key=lambda d: runtime.stride(d),
                        reverse=True,
                    )
                    physical = runtime.detach().permute(dims)
                    if physical.is_contiguous():
                        view = physical.view(-1)[: meta.numel()].view(meta.shape)
            self.source(role, alias_runtime=view is not None, loading_view=view)

    def source(
        self,
        role: str,
        *,
        alias_runtime: bool = False,
        loading_view: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get the loading destination that will be a source for conversion.

        The name is relative to policy conversion: the returned tensor is the
        *destination* of the original weight loader. It is never the incoming
        NCCL/IPC tensor. Multiple shards of a role accumulate in this object.

        Args:
            role: Loadable role with captured metadata. Input-only roles use
                the device of the state's first runtime target.
            alias_runtime: Policy permission to reuse runtime storage. Reuse
                also requires preservation to be disabled and shape, dtype,
                and strides to match. The policy must ensure the layout is
                semantically suitable; matching dimensions alone is not enough.
            loading_view: Optional canonical-layout view of the bound runtime
                storage, prepared by the policy without changing the live tensor.
                It must share the target's storage and dtype. The same layout
                and checkpoint-preservation checks still apply.

        Returns:
            A cached loader-compatible proxy with the cold parameter's subclass
            and attributes. On first access it either shares runtime storage
            through detach() or owns a zero-initialized allocation. It does not
            replace the module's Parameter.

        The first call for a role fixes its storage choice until cleared.
        ``preserve_checkpoint=False`` permits aliasing but does not guarantee
        it. Aliasing means shared storage, not Python object identity.
        """
        if role not in self.checkpoint:
            meta = self.metadata[role]
            target = self.targets.get(role)
            runtime = target.tensor if target is not None else None
            if loading_view is not None:
                if (
                    not alias_runtime
                    or runtime is None
                    or loading_view.device != runtime.device
                    or loading_view.dtype != runtime.dtype
                    or loading_view.untyped_storage().data_ptr()
                    != runtime.untyped_storage().data_ptr()
                ):
                    raise ReloadError(
                        f"{self.key}/{role}: invalid runtime loading view"
                    )
                runtime = loading_view
            can_alias = (
                alias_runtime
                and runtime is not None
                and not self.preserve_checkpoint
                and meta.shape == runtime.shape
                and meta.dtype == runtime.dtype
                and meta.stride() == runtime.stride()
            )
            data = (
                runtime.detach()
                if can_alias and runtime is not None
                else torch.empty_strided(
                    meta.shape,
                    meta.stride(),
                    dtype=meta.dtype,
                    device=(
                        runtime.device
                        if runtime is not None
                        else next(iter(self.targets.values())).tensor.device
                    ),
                ).zero_()
            )
            # Preserve vLLM Parameter subclasses and TP/loader metadata without
            # replacing the module's runtime Parameter.
            data.__class__ = meta.__class__
            data.__dict__ = meta.__dict__.copy()
            self.checkpoint[role] = data
        return self.checkpoint[role]

    def work(self, role: str) -> torch.Tensor:
        """Get values that a policy may modify during conversion.

        The role must already have been loaded into self.checkpoint. When
        preservation is enabled, each call returns a fresh clone, not a cached
        workspace. Otherwise it returns the loading destination itself, which
        may share runtime storage.
        """
        source = self.checkpoint[role]
        return source.clone() if self.preserve_checkpoint else source

    def copy_(self, role: str, source: torch.Tensor) -> None:
        """Write a converted result to a bound target without replacing it.

        Set runtime_modified before validation/copy so failure reporting is
        conservative. A True flag does not prove that a write succeeded.
        """
        self.runtime_modified = True
        self.targets[role].copy_(source)


class ModelReloadTracer:
    """Coordinate a fixed scope of states, not necessarily an entire model.

    Lifecycle: register -> observe cold load -> cold processing -> bind runtime
    -> repeated reload rounds. Ordinary slots come from observation; expert
    slots come from a fresh plan each round. Layers finish as their arrivals and
    dependencies complete, not when global finish() is called.

    The caller must quiesce inference/EPLB and order GPU arrivals and conversions.
    No lock, CUDA synchronization, cross-rank commit, or rollback is provided.
    A failed round poisons the tracer; recover the model before reconstructing it.

    Attributes:
        states: Registered units indexed by ReloadState.key.
        observed: A cold-load observation completed successfully.
        bound: Runtime targets and dependency relationships have been bound.
        active: Inside cold observation or an open reload round.
        failed: Terminal failure marker; further loading is rejected.

    Example:
        Schematic manual lifecycle for a supported, already-created FP8 layer::

            tracer = ModelReloadTracer()
            tracer.register_fp8("model.layers.0.mlp", layer)
            with tracer.observe():
                model.load_weights(cold_weights)
            # Run the model's normal cold-load processing here.
            tracer.bind_runtime()

            # Quiesce inference and EPLB before entering this block.
            with tracer.round(preserve_checkpoint=False):
                model.load_weights(new_weights)

        Every registered unit must receive its required slots in a nonempty
        round. Unregistered modules are outside this tracer's guarantees.
    """

    def __init__(self) -> None:
        """Create an empty tracer with no installed loaders or runtime targets."""
        self.states: dict[str, ReloadState] = {}
        self.observed = False
        self.bound = False
        self.active = False
        self.failed = False
        self._touched = False
        self._finished = False
        # Restore exact pre-wrap attributes in reverse installation order.
        self._wrappers: list[tuple[torch.Tensor, Any]] = []
        self._aliases: list[tuple[torch.nn.Module, str]] = []
        # Reverse dependency edges wake units when their prerequisites finish.
        self._dependents: dict[str, list[str]] = {}

    @property
    def runtime_modified(self) -> bool:
        """Whether any registered unit may have written runtime in this round.

        This aggregates conservative per-state markers. It does not inspect
        tensor contents or writes made outside the registered policies/loaders.
        """
        return any(state.runtime_modified for state in self.states.values())

    def register_state(self, state: ReloadState) -> None:
        """Add one reload unit before observation starts.

        State keys must be unique; roles and dependencies cannot repeat within
        a state. Dependency existence and cycles are checked later at binding,
        allowing registration in any order.
        """
        if (
            self.observed
            or self.active
            or self.failed
            or state.key in self.states
            or len(set(state.roles)) != len(state.roles)
            or len(set(state.dependencies)) != len(state.dependencies)
        ):
            raise ReloadError(f"Cannot register reload state: {state.key}")
        self.states[state.key] = state

    def register_fp8(self, key: str, module: torch.nn.Module) -> None:
        """Build and register an FP8 state through its quantization method.

        Args:
            key: Unique unit identifier, usually the full module path.
            module: FP8 linear or RoutedExperts created before cold loading.

        Raises:
            NotImplementedError: The quantization method has no state builder,
                or its builder rejects the specific backend/configuration.
        """
        builder = getattr(
            getattr(module, "quant_method", None), "create_reload_state", None
        )
        if builder is None:
            raise NotImplementedError(
                f"{key}: quant method has no reload state builder"
            )
        self.register_state(builder(module, key))

    @contextmanager
    def observe(self) -> Iterator[None]:
        """Capture cold metadata/loaders and learn ordinary arrival slots once.

        Wrap the initial model.load_weights() call, before cold processing can
        replace or reshape parameters. Ordinary loaders execute normally while
        their successful shard keys are recorded. A literal False return marks
        an ignored slot; None is a valid successful loader return.

        Expert units capture metadata and loaders only. Their expected slots
        must not be learned from initial placement, since EPLB can change it.
        No incoming weight payload is buffered here.

        On successful exit, every ordinary role must have at least one expected
        slot. Wrappers are always removed; an exception poisons the tracer.
        """
        if self.observed or self.active or self.failed:
            raise ReloadError("Cold load must be observed exactly once")
        self.active = True
        try:
            for state in self.states.values():
                for role in state.roles:
                    param = getattr(state.module, role)
                    state.metadata[role] = to_meta_tensor(param)
                    loader = getattr(param, "weight_loader", None)
                    if loader is None:
                        from vllm.model_executor.model_loader.weight_utils import (
                            default_weight_loader,
                        )

                        loader = default_weight_loader
                    if loader.__name__ == "online_process_loader":
                        raise ReloadError("Online/layerwise loaders cannot be traced")
                    state.loaders[role] = loader
                    if state.expert_plan is None:
                        self._observe_loader(state, role, param)
            yield
            for state in self.states.values():
                if state.expert_plan is not None:
                    continue
                for role in state.roles:
                    if not any(key.role == role for key in state.slots.expected):
                        raise ReloadError(
                            f"{state.key}: no cold-load arrivals for {role}"
                        )
            self.observed = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False
            self._unwrap()

    def bind_runtime(self) -> None:
        """Bind the final runtime layout once, after cold-load processing.

        Validate the dependency graph, capture each loadable role's current
        tensor, and let policies bind derived outputs/backend invariants.
        Shared storage among nonempty loadable roles is rejected, even for
        distinct views. Shared parameters require an explicit single owner.

        The topological pass validates the graph; actual reload finish order is
        driven by arrivals and reverse dependency edges, not registration order.
        """
        if not self.observed or self.bound or self.active or self.failed:
            raise ReloadError("Bind runtime once after observed cold-load processing")
        order: list[str] = []
        pending = set(self.states)
        while pending:
            ready = [
                key
                for key in self.states
                if key in pending
                and all(dep in order for dep in self.states[key].dependencies)
            ]
            if not ready:
                missing = {
                    key: tuple(
                        dep
                        for dep in self.states[key].dependencies
                        if dep not in self.states
                    )
                    for key in pending
                    if any(
                        dep not in self.states for dep in self.states[key].dependencies
                    )
                }
                logger.error(
                    "Reload dependency validation failed: missing=%s states=%s",
                    missing,
                    tuple(self.states),
                )
                raise ReloadError("Missing dependency or cycle in reload states")
            order.extend(ready)
            pending.difference_update(ready)
        storage_owners: dict[tuple, str] = {}
        for state in self.states.values():
            if set(state.runtime_names) - set(state.roles):
                raise ReloadError(f"{state.key}: runtime mapping has unknown roles")
            for role in state.roles:
                name = state.runtime_names.get(role, role)
                if name is not None and name != role and hasattr(state.module, role):
                    raise ReloadError(f"{state.key}: checkpoint alias already exists")
                if name is not None:
                    state.bind_target(role, partial(getattr, state.module, name))
            state.policy.bind(state)
            if state.roles and not state.targets:
                raise ReloadError(f"{state.key}: reload inputs have no runtime targets")
            # Shared targets need explicit ownership; do not silently write twice.
            for role in state.roles:
                if role not in state.targets:
                    continue
                tensor = state.targets[role].tensor
                if not tensor.numel():
                    continue
                storage = (tensor.device, tensor.untyped_storage().data_ptr())
                if storage in storage_owners:
                    raise ReloadError(
                        f"Aliased reload targets: {storage_owners[storage]}, "
                        f"{state.key}/{role}"
                    )
                storage_owners[storage] = f"{state.key}/{role}"
        self._dependents = {key: [] for key in self.states}
        for key, state in self.states.items():
            for dependency in state.dependencies:
                self._dependents[dependency].append(key)
        self.bound = True

    def begin_round(self, *, preserve_checkpoint: bool = False) -> None:
        """Validate runtime state and install loaders for a new reload round.

        Rebuild expert slots from current placement, then clear prior arrivals,
        checkpoint references, completion flags, and mutation markers. Ordinary
        expected slots remain those captured during cold observation.

        Args:
            preserve_checkpoint: Retain checkpoint-layout loader outputs through
                successful finish. Policies must use separate work storage for
                destructive conversion. Retention ends at the next begin/abort.

        This does not receive weights or allocate all staging buffers upfront.
        Direct lifecycle callers must finish or abort after opening the round;
        round() supplies that cleanup for a context-managed caller.
        """
        if not self.bound or self.active or self.failed:
            raise ReloadError("Reload tracer is unbound, active, or failed")
        for state in self.states.values():
            for target in state.targets.values():
                target.validate()
            state.policy.validate(state)
            for role, name in state.runtime_names.items():
                if name is not None and name != role and hasattr(state.module, role):
                    raise ReloadError(f"{state.key}: checkpoint alias already exists")
            if state.expert_plan is not None:
                state.slots = state.expert_plan.build(state)
        self.active = True
        self._touched = False
        self._finished = False
        try:
            for state in self.states.values():
                state.slots.arrived.clear()
                state.checkpoint.clear()
                state.complete = False
                state.runtime_modified = False
                state.preserve_checkpoint = preserve_checkpoint
                for role in state.roles:
                    role_target = state.targets.get(role)
                    if role_target is None:
                        meta = state.metadata[role]
                        # Only the public proxy is needed here; payload storage
                        # is allocated by source() and released per layer.
                        param = torch.empty(
                            (),
                            dtype=meta.dtype,
                            device=next(iter(state.targets.values())).tensor.device,
                        ).expand(meta.shape)
                    else:
                        param = role_target.tensor
                    name = state.runtime_names.get(role, role)
                    if name != role:
                        # A distinct object prevents named_parameters() from
                        # deduplicating the checkpoint name against the live one.
                        param = torch.nn.Parameter(param.detach(), requires_grad=False)
                        setattr(state.module, role, param)
                        self._aliases.append((state.module, role))
                    self._wrap(state, role, param)
        except BaseException:
            self.abort()
            raise

    def _observe_loader(
        self, state: ReloadState, role: str, param: torch.Tensor
    ) -> None:
        """Install a cold-load recorder for one ordinary parameter.

        Normalize arguments, verify the parameter object, call the original
        loader unchanged, and record its shard key. Successful duplicate slots
        and slots that change from ignored to successful are rejected.
        RoutedExperts do not use this wrapper.
        """
        loader = state.loaders[role]
        signature = inspect.signature(loader)

        @wraps(loader)
        @torch.no_grad()
        def observed_loader(*args, **kwargs):
            """Execute one cold-load call and record its non-payload identity."""
            if self.failed or not self.active:
                raise ReloadError("Arrival outside cold-load observation")
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if bound.arguments["param"] is not param:
                    raise ReloadError(f"{state.key}/{role}: wrong loader parameter")
                key = _slot_key(role, bound)
                if key in state.slots.expected:
                    raise ReloadError(f"Duplicate cold-load slot: {key}")
                result = loader(*args, **kwargs)
                if result is False:
                    state.ignored.add(key)
                else:
                    if key in state.ignored:
                        raise ReloadError(f"Cold-load slot changed locality: {key}")
                    state.slots.expected.add(key)
                return result
            except BaseException:
                self.failed = True
                raise

        self._install_wrapper(param, observed_loader)

    def _wrap(self, state: ReloadState, role: str, param: torch.Tensor) -> None:
        """Install a reload interceptor on a runtime parameter or input proxy.

        The model still invokes its usual weight loader. The interceptor checks
        the arrival, obtains a policy destination, substitutes only this call's
        ``param`` argument, and delegates sharding/copying to the original loader.
        It then records success and attempts immediate per-unit completion.

        This wraps param.weight_loader, not model.load_weights or copy_. It does
        not count tensor elements, retain transport input tensors, or perform
        CUDA synchronization. The module's runtime Parameter is not replaced.

        Args:
            state: Reload unit owning the parameter and expected slots.
            role: Parameter role within that unit.
            param: Runtime object or temporary checkpoint-name proxy whose
                loader is temporarily replaced.
        """
        loader = state.loaders[role]
        signature = inspect.signature(loader)

        @wraps(loader)
        @torch.no_grad()
        def traced_loader(*args, **kwargs):
            """Load one expected shard, then finish any newly ready units."""
            if self.failed or not self.active:
                raise ReloadError("Arrival outside an active reload round")
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if bound.arguments["param"] is not param:
                    raise ReloadError(f"{state.key}/{role}: wrong loader parameter")
                # Placement affects both which arrivals are required and where
                # they belong. Never accept a different placement mid-round.
                if state.expert_plan is not None:
                    state.expert_plan.validate(state)
                    key = state.expert_plan.slot_key(role, bound)
                    if not state.expert_plan.is_local(key):
                        return False if bound.arguments["return_success"] else None
                else:
                    key = _slot_key(role, bound)
                    if key in state.ignored:
                        return False
                state.slots.validate(key)
                for target in state.targets.values():
                    target.validate()
                # Keep the public runtime object stable; redirect the loader
                # through its argument to a checkpoint-compatible proxy.
                destination = state.policy.destination(state, role, bound)
                bound.arguments["param"] = destination
                role_target = state.targets.get(role)
                if (
                    role_target is not None
                    and destination.untyped_storage().data_ptr()
                    == role_target.tensor.untyped_storage().data_ptr()
                ):
                    state.runtime_modified = True
                first_arrival = not self._touched
                self._touched = True
                return_success = bound.arguments.get("return_success", True)
                if state.expert_plan is not None:
                    # Detect a rejected expert write even when the caller did
                    # not request a boolean, then restore its return convention.
                    bound.arguments["return_success"] = True
                result = loader(*bound.args, **bound.kwargs)
                if result is False:
                    raise ReloadError(f"Loader rejected expected slot: {key}")
                state.slots.arrived.add(key)
                if first_arrival:
                    # Defer zero-input roots until actual data arrives so an
                    # entirely untouched round remains a no-op.
                    for candidate in self.states.values():
                        if not candidate.roles and not candidate.dependencies:
                            self._finish_ready(candidate.key)
                self._finish_ready(state.key)
                return result if return_success else None
            except BaseException:
                self.failed = True
                raise

        self._install_wrapper(param, traced_loader)

    def _install_wrapper(self, param: torch.Tensor, loader: Callable) -> None:
        """Save the existing loader attribute and install a temporary one.

        Saving absence as None lets _unwrap restore parameters that originally
        relied on a default loader without leaving a new attribute behind.
        """
        self._wrappers.append((param, getattr(param, "weight_loader", None)))
        param.weight_loader = loader

    def _finish_ready(self, key: str) -> None:
        """Finish a ready unit and propagate readiness to its dependents.

        Args:
            key: A ReloadState.key, not a SlotKey.

        A unit is ready only when all expected slots and dependencies are done.
        Validate placement/backend invariants, run policy.finish once, and
        verify runtime identity/layout before marking it complete. Drop
        unpreserved checkpoint references immediately; allocator caching and
        outstanding CUDA work can delay physical memory reuse.

        No incoming payload is passed here: the policy consumes state.checkpoint.
        Dependencies express conversion ordering, not cross-rank synchronization.
        """
        queue = deque([key])
        while queue:
            state = self.states[queue.popleft()]
            if state.complete:
                continue
            if len(state.slots.arrived) != len(state.slots.expected):
                continue
            unfinished = tuple(
                dep for dep in state.dependencies if not self.states[dep].complete
            )
            if unfinished:
                logger.info(
                    "Reload state %s waiting for dependencies: %s",
                    state.key,
                    unfinished,
                )
                continue
            logger.info(
                "Reload state %s finishing after dependencies: %s",
                state.key,
                state.dependencies,
            )
            if state.expert_plan is not None:
                state.expert_plan.validate(state)
            state.policy.validate(state)
            state.policy.finish(state)
            for target in state.targets.values():
                target.validate()
            state.complete = True
            logger.info("Reload state %s finished", state.key)
            if not state.preserve_checkpoint:
                state.checkpoint.clear()
            queue.extend(self._dependents[state.key])

    def _unwrap(self) -> None:
        """Restore saved loader attributes in reverse order and clear the stack.

        This only removes interceptors; it does not undo tensor writes. Calling
        it again after cleanup is harmless because the stack is already empty.
        """
        for param, loader in reversed(self._wrappers):
            if loader is None:
                delattr(param, "weight_loader")
            else:
                param.weight_loader = loader
        self._wrappers.clear()
        for module, role in reversed(self._aliases):
            delattr(module, role)
        self._aliases.clear()

    def missing(self) -> list[str]:
        """Format missing slots with their owning state keys for diagnostics.

        Ordering within a state is unspecified. No state is changed, and this
        reports slot coverage only, not unfinished dependencies.
        """
        return [
            f"{state.key}: {key}"
            for state in self.states.values()
            for key in state.slots.missing()
        ]

    @torch.no_grad()
    def finish(self) -> bool:
        """Validate and close a round after all transport chunks have arrived.

        Policy conversions normally already ran in _finish_ready. This method
        does not flush incomplete layers or rerun their conversion. It checks
        target/backend/placement invariants, missing slots, and completion.

        Returns:
            True for a completed round that reached a managed loader write.
            False for an untouched round, after invariant validation. Repeated
            calls after successful finish return the same result.

        Raises:
            ReloadError: The lifecycle is invalid, a prior load failed, required
                slots are missing, or a state has not finished.

        Wrappers are removed when closing the round. Validation failures inside
        the closing phase abort and poison it; successful finish retains
        checkpoint values only when preservation was requested. Neither path
        synchronizes CUDA or rolls back runtime writes.
        """
        if self.failed:
            self.abort()
            raise ReloadError("Reload failed; recover the model before reuse")
        if self._finished:
            return self._touched
        if not self.active:
            raise ReloadError("No active reload round")
        try:
            for state in self.states.values():
                for target in state.targets.values():
                    target.validate()
                state.policy.validate(state)
                if state.expert_plan is not None:
                    state.expert_plan.validate(state)
            if not self._touched:
                self._finished = True
                return False
            missing = self.missing()
            if missing:
                raise ReloadError("Missing reload slots:\n" + "\n".join(missing))
            if any(not state.complete for state in self.states.values()):
                raise ReloadError("Unfinished reload states")
            self._finished = True
            return True
        except BaseException:
            self.abort()
            raise
        finally:
            self.active = False
            self._unwrap()

    def abort(self) -> None:
        """Poison the tracer, remove wrappers, and release checkpoint references.

        This is cleanup, not rollback: finished layers and in-place arrivals may
        already have changed runtime values. Mutation markers remain available
        for diagnosis. A failed tracer cannot begin another round.
        """
        self.failed = True
        self.active = False
        self._unwrap()
        for state in self.states.values():
            state.checkpoint.clear()

    @contextmanager
    def round(self, *, preserve_checkpoint: bool = False) -> Iterator[None]:
        """Bracket weight loading with begin/finish and abort on body failure.

        The caller supplies all weights inside the context. Successful exit
        validates completion; an exception from loading or finish poisons the
        tracer and propagates to the caller.

        Example:
            Load several transport chunks as one logical round::

                with tracer.round(preserve_checkpoint=True):
                    for chunk in chunks:
                        model.load_weights(chunk)

            After exit, state.checkpoint retains local checkpoint-layout values.
            It does not retain the sender's chunk buffers or a version history.
        """
        self.begin_round(preserve_checkpoint=preserve_checkpoint)
        try:
            yield
            self.finish()
        except BaseException:
            self.abort()
            raise
