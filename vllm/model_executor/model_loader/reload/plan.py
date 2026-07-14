# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed model refit contract for weight reload."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import torch

TensorResolver = Callable[[], torch.Tensor | None]
StateCallback = Callable[[], None]
StateValidator = Callable[[], bool | None]
DerivedCallback = Callable[[], Mapping[str, torch.Tensor]]

_BINDINGS_ATTR = "_vllm_reload_plan_bindings"


class ReloadCapability(Enum):
    """The strongest reload mode certified by a plan."""

    UNSUPPORTED = auto()
    EAGER_ONLY = auto()
    GRAPH_SAFE_V1 = auto()


class ReloadStatePolicy(Enum):
    """Required action for state that crosses model versions."""

    REFRESH = auto()
    INVALIDATE = auto()
    PRESERVE = auto()


class ReloadTransactionState(Enum):
    LOADING = auto()
    FINALIZING = auto()
    VALIDATING = auto()
    PREPARED = auto()
    COMMITTED = auto()
    FAILED = auto()


class ReloadPlanError(RuntimeError):
    """Base error for reload contract violations."""


class ReloadPlanCompileError(ReloadPlanError):
    """Raised when a reload plan is incomplete or inconsistent."""


class ReloadCapabilityError(ReloadPlanError):
    """Raised before mutation when a model is not graph-safe reloadable."""


class ReloadLifecycleError(ReloadPlanError):
    """Raised when transaction operations occur out of order."""


class ReloadInputError(ReloadPlanError):
    """Raised when reload input metadata violates its prototype."""


class ReloadStorageIdentityError(ReloadPlanError):
    """Raised when a stable tensor slot changes storage or layout."""


class StaleDerivedSlotError(ReloadPlanError):
    """Raised when a required derived node did not refresh."""


class ReloadStatePolicyError(ReloadPlanError):
    """Raised when reload-sensitive state does not satisfy its policy."""


@dataclass(frozen=True)
class TensorFingerprint:
    """Storage and layout identity captured when the plan is sealed."""

    storage_cdata: int
    data_ptr: int
    storage_nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorFingerprint:
        storage = tensor.untyped_storage()
        return cls(
            storage_cdata=storage._cdata,
            data_ptr=tensor.data_ptr(),
            storage_nbytes=storage.nbytes(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            device=tensor.device,
        )


@dataclass(frozen=True)
class InputPrototype:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> InputPrototype:
        return cls(shape=tuple(tensor.shape), dtype=tensor.dtype)


@dataclass
class _StableTensorSlot:
    name: str
    resolver: TensorResolver
    tensor: torch.Tensor
    fingerprint: TensorFingerprint
    last_written_epoch: int = 0

    def current(self) -> torch.Tensor:
        tensor = self.resolver()
        if tensor is None:
            raise ReloadStorageIdentityError(
                f"Stable reload slot {self.name!r} disappeared"
            )
        if not isinstance(tensor, torch.Tensor):
            raise ReloadStorageIdentityError(
                f"Stable reload slot {self.name!r} resolved to "
                f"{type(tensor).__name__}, expected Tensor"
            )
        return tensor

    def validate(self) -> None:
        current = self.current()
        actual = TensorFingerprint.from_tensor(current)
        if actual != self.fingerprint:
            raise ReloadStorageIdentityError(
                f"Stable reload slot {self.name!r} changed storage or layout: "
                f"expected={self.fingerprint}, actual={actual}"
            )

    def copy_from(self, value: torch.Tensor, epoch: int) -> None:
        expected = self.fingerprint
        if tuple(value.shape) != expected.shape:
            raise ReloadInputError(
                f"Reload value for {self.name!r} has shape {tuple(value.shape)}, "
                f"expected {expected.shape}"
            )
        if value.dtype != expected.dtype:
            raise ReloadInputError(
                f"Reload value for {self.name!r} has dtype {value.dtype}, "
                f"expected {expected.dtype}"
            )
        self.tensor.copy_(value)
        self.last_written_epoch = epoch


@dataclass(frozen=True)
class _DirectInput:
    name: str
    prototype: InputPrototype
    slot: _StableTensorSlot


@dataclass
class _DerivedNode:
    name: str
    owner: object
    owner_key: str
    depends_on: tuple[str, ...]
    outputs: dict[str, _StableTensorSlot]
    refresh: DerivedCallback | None
    always_refresh: bool
    last_executed_epoch: int = 0


@dataclass
class _StateNode:
    name: str
    policy: ReloadStatePolicy
    action: StateCallback | None
    validate: StateValidator | None
    last_executed_epoch: int = 0


@dataclass(frozen=True)
class _DerivedBinding:
    plan: ReloadPlan
    node_name: str


class ReloadPlanBuilder:
    """Builds an immutable reload contract before the first mutation."""

    def __init__(self) -> None:
        self._slots: dict[str, _StableTensorSlot] = {}
        self._inputs: dict[str, _DirectInput] = {}
        self._derived: dict[str, _DerivedNode] = {}
        self._states: dict[str, _StateNode] = {}
        self._unsupported_reasons: list[str] = []
        self._eager_only_reasons: list[str] = []
        self._sealed = False

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        *,
        prefix: str = "model",
    ) -> ReloadPlanBuilder:
        builder = cls()
        builder.add_module_tensors(model, prefix=prefix)
        for module_name, module in model.named_modules():
            participant = getattr(module, "build_reload_plan", None)
            if participant is None:
                continue
            full_name = _join_name(prefix, module_name)
            participant(builder, full_name)
        return builder

    def add_module_tensors(
        self,
        module: torch.nn.Module,
        *,
        prefix: str = "model",
    ) -> None:
        self._ensure_mutable()
        for name, tensor in module.named_parameters(remove_duplicate=False):
            full_name = _join_name(prefix, name)
            self.direct_input(
                full_name,
                tensor,
                resolver=_module_parameter_resolver(module, name),
            )
        for name, tensor in module.named_buffers(remove_duplicate=False):
            full_name = _join_name(prefix, name)
            self.direct_input(
                full_name,
                tensor,
                resolver=_module_buffer_resolver(module, name),
            )

    def direct_input(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        resolver: TensorResolver,
    ) -> None:
        self._ensure_mutable()
        slot = self._add_slot(name, tensor, resolver)
        if name in self._inputs:
            raise ReloadPlanCompileError(f"Reload input {name!r} is duplicated")
        self._inputs[name] = _DirectInput(
            name=name,
            prototype=InputPrototype.from_tensor(tensor),
            slot=slot,
        )

    def derived(
        self,
        name: str,
        *,
        owner: object,
        owner_key: str,
        outputs: Iterable[str],
        depends_on: Iterable[str] = (),
        refresh: DerivedCallback | None = None,
        always_refresh: bool = True,
    ) -> None:
        self._ensure_mutable()
        if name in self._derived:
            raise ReloadPlanCompileError(f"Derived reload node {name!r} is duplicated")

        output_slots: dict[str, _StableTensorSlot] = {}
        for attr_name in outputs:
            value = getattr(owner, attr_name, None)
            if not isinstance(value, torch.Tensor):
                raise ReloadPlanCompileError(
                    f"Derived output {name}.{attr_name} is not a Tensor"
                )
            slot_name = f"{name}.{attr_name}"
            output_slots[attr_name] = self._add_slot(
                slot_name,
                value,
                resolver=_attribute_resolver(owner, attr_name),
            )

        self._derived[name] = _DerivedNode(
            name=name,
            owner=owner,
            owner_key=owner_key,
            depends_on=tuple(depends_on),
            outputs=output_slots,
            refresh=refresh,
            always_refresh=always_refresh,
        )

    def input_names(self, *, prefix: str | None = None) -> tuple[str, ...]:
        """Return declared input names, optionally limited to a module prefix."""
        if prefix is None:
            return tuple(self._inputs)
        dotted_prefix = f"{prefix}." if prefix else ""
        return tuple(
            name
            for name in self._inputs
            if name == prefix or name.startswith(dotted_prefix)
        )

    def inputs_for_tensor(self, tensor: torch.Tensor) -> tuple[str, ...]:
        """Return logical inputs that alias a participant's source tensor."""
        return tuple(
            name
            for name, direct in self._inputs.items()
            if _same_storage_and_layout(direct.slot.tensor, tensor)
        )

    def state(
        self,
        name: str,
        *,
        policy: ReloadStatePolicy,
        action: StateCallback | None = None,
        validate: StateValidator | None = None,
    ) -> None:
        self._ensure_mutable()
        if name in self._states:
            raise ReloadPlanCompileError(f"Reload state {name!r} is duplicated")
        if policy in (ReloadStatePolicy.REFRESH, ReloadStatePolicy.INVALIDATE):
            if action is None:
                raise ReloadPlanCompileError(
                    f"Reload state {name!r} with policy {policy.name} "
                    "requires an action"
                )
        elif validate is None:
            raise ReloadPlanCompileError(
                f"Reload state {name!r} with policy PRESERVE requires a validator"
            )
        self._states[name] = _StateNode(
            name=name,
            policy=policy,
            action=action,
            validate=validate,
        )

    def unsupported(self, reason: str) -> None:
        self._ensure_mutable()
        if reason and reason not in self._unsupported_reasons:
            self._unsupported_reasons.append(reason)

    def eager_only(self, reason: str) -> None:
        self._ensure_mutable()
        if reason and reason not in self._eager_only_reasons:
            self._eager_only_reasons.append(reason)

    def seal(self) -> ReloadPlan:
        self._ensure_mutable()
        self._validate_dependencies()
        self._sealed = True
        plan = ReloadPlan(
            slots=self._slots,
            inputs=self._inputs,
            derived=self._derived,
            states=self._states,
            unsupported_reasons=tuple(self._unsupported_reasons),
            eager_only_reasons=tuple(self._eager_only_reasons),
        )
        plan._install_bindings()
        return plan

    def _add_slot(
        self,
        name: str,
        tensor: torch.Tensor,
        resolver: TensorResolver,
    ) -> _StableTensorSlot:
        if not name:
            raise ReloadPlanCompileError("Reload slot name must not be empty")
        if name in self._slots:
            raise ReloadPlanCompileError(f"Stable reload slot {name!r} is duplicated")
        current = resolver()
        if current is None or not _same_storage_and_layout(current, tensor):
            raise ReloadPlanCompileError(
                f"Resolver for stable reload slot {name!r} does not resolve "
                "to the declared tensor"
            )
        slot = _StableTensorSlot(
            name=name,
            resolver=resolver,
            tensor=tensor,
            fingerprint=TensorFingerprint.from_tensor(tensor),
        )
        self._slots[name] = slot
        return slot

    def _validate_dependencies(self) -> None:
        known = set(self._inputs) | set(self._derived)
        for node in self._derived.values():
            missing = set(node.depends_on) - known
            if missing:
                raise ReloadPlanCompileError(
                    f"Derived reload node {node.name!r} has unknown dependencies: "
                    f"{sorted(missing)}"
                )
        _topological_order(self._derived)

    def _ensure_mutable(self) -> None:
        if self._sealed:
            raise ReloadPlanCompileError("Reload plan builder is already sealed")


class ReloadPlan:
    """A sealed model refit plan shared by all reload transports."""

    def __init__(
        self,
        *,
        slots: dict[str, _StableTensorSlot],
        inputs: dict[str, _DirectInput],
        derived: dict[str, _DerivedNode],
        states: dict[str, _StateNode],
        unsupported_reasons: tuple[str, ...],
        eager_only_reasons: tuple[str, ...],
    ) -> None:
        self._slots = dict(slots)
        self._inputs = dict(inputs)
        self._derived = dict(derived)
        self._states = dict(states)
        self._topological_order = _topological_order(self._derived)
        self.unsupported_reasons = unsupported_reasons
        self.eager_only_reasons = eager_only_reasons
        if unsupported_reasons:
            self.capability = ReloadCapability.UNSUPPORTED
        elif eager_only_reasons:
            self.capability = ReloadCapability.EAGER_ONLY
        else:
            self.capability = ReloadCapability.GRAPH_SAFE_V1
        self.committed_epoch = 0
        self._active: ReloadTransaction | None = None
        self._failed = False

    @property
    def active_transaction(self) -> ReloadTransaction | None:
        return self._active

    def begin(
        self,
        *,
        expected_inputs: Iterable[str] | None = None,
        require_graph_safe: bool = True,
    ) -> ReloadTransaction:
        if self._failed:
            raise ReloadLifecycleError(
                "Reload plan is failed; fully reload or restart the worker"
            )
        if self._active is not None:
            raise ReloadLifecycleError("A reload transaction is already active")
        if self.capability is ReloadCapability.UNSUPPORTED:
            reasons = "; ".join(self.unsupported_reasons)
            raise ReloadCapabilityError(
                f"The model does not support warm reload: {reasons}"
            )
        if require_graph_safe and self.capability is ReloadCapability.EAGER_ONLY:
            reasons = "; ".join(self.eager_only_reasons)
            raise ReloadCapabilityError(
                "The model is not certified for warm reload with a live graph: "
                f"{reasons}"
            )
        transaction = ReloadTransaction(
            plan=self,
            epoch=self.committed_epoch + 1,
            expected_inputs=(
                frozenset(expected_inputs) if expected_inputs is not None else None
            ),
        )
        self._active = transaction
        return transaction

    def _install_bindings(self) -> None:
        pending_keys: set[tuple[int, str]] = set()
        for node in self._derived.values():
            existing_bindings = node.owner.__dict__.get(_BINDINGS_ATTR, {})
            key = (id(node.owner), node.owner_key)
            if node.owner_key in existing_bindings or key in pending_keys:
                raise ReloadPlanCompileError(
                    f"Reload participant key {node.owner_key!r} is duplicated on "
                    f"{type(node.owner).__name__}"
                )
            pending_keys.add(key)

        for node in self._derived.values():
            bindings: dict[str, _DerivedBinding] = node.owner.__dict__.setdefault(
                _BINDINGS_ATTR, {}
            )
            bindings[node.owner_key] = _DerivedBinding(self, node.name)

    def _refresh_derived(
        self,
        node_name: str,
        values: Mapping[str, torch.Tensor],
    ) -> None:
        transaction = self._active
        if transaction is None:
            raise ReloadLifecycleError(
                f"Derived node {node_name!r} refreshed outside a reload transaction"
            )
        transaction.refresh_derived(node_name, values)

    def _commit(self, transaction: ReloadTransaction) -> None:
        if self._active is not transaction:
            raise ReloadLifecycleError("Cannot commit an inactive reload transaction")
        self.committed_epoch = transaction.epoch
        self._active = None

    def _abort(self, transaction: ReloadTransaction) -> None:
        if self._active is transaction:
            self._active = None
        self._failed = True


class ReloadTransaction:
    """One fail-closed application of a sealed reload plan."""

    def __init__(
        self,
        *,
        plan: ReloadPlan,
        epoch: int,
        expected_inputs: frozenset[str] | None,
    ) -> None:
        self.plan = plan
        self.epoch = epoch
        self.expected_inputs = expected_inputs
        self.state = ReloadTransactionState.LOADING
        self._written_inputs: set[str] = set()
        self._required_nodes = self._compute_required_nodes()

    def __enter__(self) -> ReloadTransaction:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        if exc_value is not None:
            self.abort()
        elif self.state is not ReloadTransactionState.COMMITTED:
            self.abort()
            raise ReloadLifecycleError(
                f"Reload transaction epoch {self.epoch} exited without commit"
            )
        return False

    def record_input(
        self,
        name: str,
        tensor: torch.Tensor | None = None,
        *,
        allow_unknown: bool = False,
        allow_duplicate: bool = False,
    ) -> None:
        self._require_state(ReloadTransactionState.LOADING)
        if name in self._written_inputs:
            if allow_duplicate:
                return
            raise ReloadInputError(f"Reload input {name!r} was written twice")
        direct = self.plan._inputs.get(name)
        if direct is None:
            if allow_unknown:
                self._written_inputs.add(name)
                return
            raise ReloadInputError(f"Unknown reload input {name!r}")
        if tensor is not None:
            actual = InputPrototype.from_tensor(tensor)
            if actual != direct.prototype:
                raise ReloadInputError(
                    f"Reload input {name!r} has prototype {actual}, "
                    f"expected {direct.prototype}"
                )
        self._written_inputs.add(name)

    @torch.no_grad()
    def write(self, name: str, tensor: torch.Tensor) -> None:
        self.record_input(name, tensor)
        direct = self.plan._inputs[name]
        direct.slot.copy_from(tensor, self.epoch)

    def start_finalizing(self) -> None:
        self._require_state(ReloadTransactionState.LOADING)
        self.state = ReloadTransactionState.FINALIZING

    @torch.no_grad()
    def refresh_derived(
        self,
        node_name: str,
        values: Mapping[str, torch.Tensor],
    ) -> None:
        if self.state not in (
            ReloadTransactionState.LOADING,
            ReloadTransactionState.FINALIZING,
        ):
            raise ReloadLifecycleError(
                f"Cannot refresh derived node {node_name!r} while transaction "
                f"is {self.state.name}"
            )
        node = self.plan._derived.get(node_name)
        if node is None:
            raise ReloadPlanCompileError(f"Unknown derived reload node {node_name!r}")
        if node.last_executed_epoch == self.epoch:
            raise ReloadLifecycleError(
                f"Derived reload node {node_name!r} executed twice in epoch "
                f"{self.epoch}"
            )
        if set(values) != set(node.outputs):
            raise ReloadInputError(
                f"Derived reload node {node_name!r} produced outputs "
                f"{sorted(values)}, expected {sorted(node.outputs)}"
            )
        for output_name, slot in node.outputs.items():
            slot.copy_from(values[output_name], self.epoch)
            setattr(node.owner, output_name, slot.tensor)
        node.last_executed_epoch = self.epoch

    @torch.no_grad()
    def prepare_or_raise(self) -> None:
        if self.state is ReloadTransactionState.LOADING:
            self.start_finalizing()
        self._require_state(ReloadTransactionState.FINALIZING)
        try:
            self._run_derived_callbacks()
            self._validate_inputs()
            self._validate_derived_nodes()
            self._run_state_nodes()
            self.state = ReloadTransactionState.VALIDATING
            for slot in self.plan._slots.values():
                slot.validate()
        except BaseException:
            self.abort()
            raise
        self.state = ReloadTransactionState.PREPARED

    def commit_prepared(self) -> None:
        self._require_state(ReloadTransactionState.PREPARED)
        self.state = ReloadTransactionState.COMMITTED
        self.plan._commit(self)

    def commit_or_raise(self) -> None:
        self.prepare_or_raise()
        self.commit_prepared()

    def abort(self) -> None:
        if self.state in (
            ReloadTransactionState.COMMITTED,
            ReloadTransactionState.FAILED,
        ):
            return
        self.state = ReloadTransactionState.FAILED
        self.plan._abort(self)

    def _compute_required_nodes(self) -> set[str]:
        if self.expected_inputs is None:
            return {
                node.name for node in self.plan._derived.values() if node.always_refresh
            }
        changed: set[str] = set(self.expected_inputs)
        required: set[str] = set()
        for node_name in self.plan._topological_order:
            node = self.plan._derived[node_name]
            if node.always_refresh or changed.intersection(node.depends_on):
                required.add(node_name)
                changed.add(node_name)
        return required

    def _run_derived_callbacks(self) -> None:
        for node_name in self.plan._topological_order:
            if node_name not in self._required_nodes:
                continue
            node = self.plan._derived[node_name]
            if node.last_executed_epoch == self.epoch or node.refresh is None:
                continue
            self.refresh_derived(node_name, node.refresh())

    def _validate_inputs(self) -> None:
        if self.expected_inputs is None:
            return
        unknown = self.expected_inputs - self.plan._inputs.keys()
        if unknown:
            raise ReloadInputError(f"Unknown expected reload inputs: {sorted(unknown)}")
        missing = set(self.expected_inputs - self._written_inputs)
        for node_name in self._required_nodes:
            node = self.plan._derived[node_name]
            missing.update(
                dependency
                for dependency in node.depends_on
                if dependency in self.plan._inputs
                and dependency not in self._written_inputs
            )
        if missing:
            raise ReloadInputError(f"Missing reload inputs: {sorted(missing)}")

    def _validate_derived_nodes(self) -> None:
        stale = [
            node_name
            for node_name in self._required_nodes
            if self.plan._derived[node_name].last_executed_epoch != self.epoch
        ]
        if stale:
            raise StaleDerivedSlotError(
                f"Derived reload nodes were not refreshed in epoch {self.epoch}: "
                f"{sorted(stale)}"
            )

    def _run_state_nodes(self) -> None:
        for node in self.plan._states.values():
            if node.action is not None:
                node.action()
            if node.validate is not None and node.validate() is False:
                raise ReloadStatePolicyError(
                    f"Reload state {node.name!r} failed {node.policy.name} validation"
                )
            node.last_executed_epoch = self.epoch

    def _require_state(self, expected: ReloadTransactionState) -> None:
        if self.state is not expected:
            raise ReloadLifecycleError(
                f"Reload transaction epoch {self.epoch} is {self.state.name}, "
                f"expected {expected.name}"
            )


def refresh_reload_derived(
    owner: object,
    owner_key: str,
    values: Mapping[str, torch.Tensor],
) -> None:
    """Publish newly derived values into stable plan-owned output slots.

    Before a plan is sealed this performs the initial attribute binding. During
    reload it writes into the original tensors and records the current epoch.
    """
    bindings: dict[str, _DerivedBinding] | None = getattr(owner, _BINDINGS_ATTR, None)
    if bindings is None or owner_key not in bindings:
        for name, tensor in values.items():
            setattr(owner, name, tensor)
        return
    binding = bindings[owner_key]
    binding.plan._refresh_derived(binding.node_name, values)


def _attribute_resolver(owner: object, name: str) -> TensorResolver:
    def resolve() -> torch.Tensor | None:
        value = getattr(owner, name, None)
        return value if isinstance(value, torch.Tensor) else None

    return resolve


def _module_parameter_resolver(module: torch.nn.Module, name: str) -> TensorResolver:
    def resolve() -> torch.Tensor | None:
        try:
            return module.get_parameter(name)
        except AttributeError:
            return None

    return resolve


def _module_buffer_resolver(module: torch.nn.Module, name: str) -> TensorResolver:
    def resolve() -> torch.Tensor | None:
        try:
            return module.get_buffer(name)
        except AttributeError:
            return None

    return resolve


def _same_storage_and_layout(left: torch.Tensor, right: torch.Tensor) -> bool:
    return TensorFingerprint.from_tensor(left) == TensorFingerprint.from_tensor(right)


def _join_name(prefix: str, name: str) -> str:
    if prefix and name:
        return f"{prefix}.{name}"
    return prefix or name


def _topological_order(nodes: Mapping[str, _DerivedNode]) -> tuple[str, ...]:
    incoming = {
        name: {dependency for dependency in node.depends_on if dependency in nodes}
        for name, node in nodes.items()
    }
    ready = sorted(name for name, dependencies in incoming.items() if not dependencies)
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for candidate, dependencies in incoming.items():
            if name not in dependencies:
                continue
            dependencies.remove(name)
            if not dependencies and candidate not in order and candidate not in ready:
                ready.append(candidate)
                ready.sort()
    if len(order) != len(nodes):
        cyclic = sorted(set(nodes) - set(order))
        raise ReloadPlanCompileError(
            f"Reload plan contains a derived dependency cycle: {cyclic}"
        )
    return tuple(order)


__all__ = [
    "InputPrototype",
    "ReloadCapability",
    "ReloadCapabilityError",
    "ReloadInputError",
    "ReloadLifecycleError",
    "ReloadPlan",
    "ReloadPlanBuilder",
    "ReloadPlanCompileError",
    "ReloadPlanError",
    "ReloadStatePolicy",
    "ReloadStatePolicyError",
    "ReloadStorageIdentityError",
    "ReloadTransaction",
    "ReloadTransactionState",
    "StaleDerivedSlotError",
    "TensorFingerprint",
    "refresh_reload_derived",
]
