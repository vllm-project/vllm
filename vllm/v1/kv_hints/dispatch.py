# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One native dispatcher for RFC primitives and the legacy semantic profile."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import msgspec

from vllm.v1.kv_hints.actions import (
    PRIMITIVE_VERSION,
    PRIMITIVES,
    CopyTo,
    MoveTo,
    Primitive,
    SetPriority,
)
from vllm.v1.kv_hints.control import ControlError, OperationResult, OperationStatus
from vllm.v1.kv_hints.native import (
    ActionCapability,
    DemotePayload,
    HashTarget,
    NativePayload,
    PrefetchPayload,
    RequestTarget,
    StorePayload,
    Target,
    parse_native_action,
    validate_target,
)
from vllm.v1.kv_hints.protocol import KvHintAction
from vllm.v1.kv_hints.state import LogicalResolution
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.kv_hints.controller import NativeKVController


@dataclass
class _Consumer:
    payload_type: type[Primitive]
    when: Callable[[Primitive], Literal["ingress", "successful_completion"]]


def _phase(payload: Primitive) -> Literal["ingress", "successful_completion"]:
    return (
        "successful_completion"
        if isinstance(payload, SetPriority) or payload.src.tier == "gpu"
        else "ingress"
    )


@dataclass
class _DeferredAction:
    operation: OperationResult
    payload: Primitive
    scope: LogicalResolution | None
    request: Request
    generation: int


class KvHintDispatcher:
    def __init__(self, controller: "NativeKVController"):
        self.controller = controller
        self._consumers: dict[tuple[str, str], _Consumer] = {}
        self._pending: dict[str, list[_DeferredAction]] = {}
        for name, payload_type in PRIMITIVES.items():
            self.register(name, PRIMITIVE_VERSION, payload_type, when=_phase)

    def register(
        self,
        name: str,
        version: str,
        payload_type: type[Primitive],
        *,
        when: Callable[[Primitive], Literal["ingress", "successful_completion"]],
    ) -> None:
        key = name, version
        if key in self._consumers:
            raise ValueError(f"KV hint consumer already registered: {key}")
        self._consumers[key] = _Consumer(payload_type, when)

    def parse(self, action: KvHintAction) -> NativePayload | Primitive:
        if not action.action_type.startswith("vllm."):
            return parse_native_action(action)
        consumer = self._consumers.get((action.action_type, action.action_version))
        if consumer is None:
            raise ControlError("unsupported", "Unsupported native primitive or version")
        try:
            payload = msgspec.convert(
                action.payload, type=consumer.payload_type, strict=True
            )
        except (msgspec.ValidationError, ValueError) as error:
            raise ControlError("invalid_payload", str(error)) from error
        if not isinstance(payload.target, str):
            validate_target(payload.target)
        return payload

    def lower(
        self, payload: CopyTo | MoveTo, target: Target | str
    ) -> DemotePayload | StorePayload | PrefetchPayload:
        if isinstance(target, str):
            target = HashTarget(self.controller.index.namespace, ())
        local = self.controller.registry.engine_id
        if any(
            location.engine_id not in (None, local)
            for location in (payload.src, payload.dst)
        ):
            raise ControlError(
                "unsupported", "Remote engines must approve their own COPY"
            )
        src, dst = payload.src.tier, payload.dst.tier
        if src == "gpu" and dst in ("cpu", "mooncake"):
            cls = DemotePayload if isinstance(payload, MoveTo) else StorePayload
            return cls(target, cast(Literal["cpu", "mooncake"], dst))
        if src in ("cpu", "mooncake") and dst == "gpu" and isinstance(payload, CopyTo):
            return PrefetchPayload(target, cast(Literal["cpu", "mooncake"], src))
        raise ControlError(
            "unsupported",
            "Requires local G1/cold COPY or G1-to-cold MOVE; "
            "cold deletion is unsupported",
        )

    def execute(
        self,
        operation: OperationResult,
        payload: Primitive,
        *,
        request_id: str | None,
        resolution: LogicalResolution | None,
        attached: bool,
        owners: tuple[str, ...] = (),
    ) -> None:
        executor = self.controller.executor
        if executor is None:
            raise ControlError("unsupported", "No primitive executor configured")
        consumer = self._consumers[(operation.action_type, operation.action_version)]
        if isinstance(payload, (CopyTo, MoveTo)):
            lowered = self.lower(payload, payload.target)
            kind, tier = (
                ("prefetch", lowered.source)
                if isinstance(lowered, PrefetchPayload)
                else ("demote", lowered.destination)
            )
            if not any(
                c.action_type == kind and tier in c.tiers
                for c in executor.capabilities()
            ):
                raise ControlError(
                    "unsupported",
                    "No adapter supports the requested primitive direction",
                )
        if attached and consumer.when(payload) == "successful_completion":
            observer = self.controller.request_observer
            if observer is None:
                raise ControlError(
                    "unsupported", "Request lifecycle binding is unavailable"
                )
            request = observer.request(RequestTarget("$request"), request_id)
            pending = self._pending.get(request.request_id)
            if pending and pending[0].request is not request:
                raise ControlError(
                    "busy", "Request ID is already bound to another lifecycle"
                )
            if isinstance(
                payload.target, RequestTarget
            ) and payload.target.request_id not in ("$request", request.request_id):
                raise ControlError(
                    "unsupported",
                    "Deferred request selectors must name the bound request",
                )
            scope = resolution
            if scope is None and not isinstance(payload.target, (str, RequestTarget)):
                scope = self.controller.index.resolve(payload.target)
                executor.validate_scope(scope)
            # Pending actions share the bounded registry admission.
            work = _DeferredAction(
                operation,
                payload,
                scope,
                request,
                executor.manager.get_transfer_manager().generation,
            )
            self._pending.setdefault(request.request_id, []).append(work)
            self.controller.registry.update(
                operation,
                OperationStatus.QUEUED,
                reason="Awaiting successful request completion",
            )
            return
        self._execute_now(
            operation, payload, resolution, request_id=request_id, owners=owners
        )

    def _execute_now(
        self,
        operation: OperationResult,
        payload: Primitive,
        scope: LogicalResolution | None,
        *,
        request_id: str | None = None,
        request: Request | None = None,
        owners: tuple[str, ...] = (),
    ) -> None:
        controller = self.controller
        executor = controller.executor
        assert executor is not None
        if scope is None:
            if request is not None and isinstance(payload.target, RequestTarget):
                observer = controller.request_observer
                assert observer is not None
                scope = controller.index.resolve(
                    observer.resolve_finished(request, payload.target)
                )
            elif request is not None and isinstance(payload.target, str):
                scope = executor.request_scope(request, payload.target)
            elif payload.target == "request_cached":
                if request_id is None or controller.request_observer is None:
                    raise ControlError(
                        "unsupported", "request_cached requires a bound request"
                    )
                request = controller.request_observer.request(
                    RequestTarget("$request"), request_id
                )
                scope = executor.request_scope(request, "request_cached")
            else:
                assert not isinstance(payload.target, str)
                scope = controller.index.resolve(
                    controller._target(payload.target, request_id)
                )
        executor.validate_scope(scope)
        if payload.target == "request_cached" and not any(
            p.blocks for p in scope.prefixes
        ):
            raise ControlError("missing", "No cached copies held by the bound request")
        if isinstance(payload, SetPriority):
            if (
                controller.policies is not None
                and operation.origin is None
                and payload.value is not None
            ):
                controller.policies.before_explicit(payload, scope)
            executor.apply_priority(operation, payload, scope, request, owners)
            return
        # The logical snapshot is authoritative; lowering does not resolve it again.
        target = HashTarget(
            scope.namespace,
            tuple(b for p in scope.prefixes for b in p.blocks),
            scope.start_token,
            scope.end_token,
        )
        lowered = self.lower(payload, target)
        if controller.policies is not None and operation.origin is None:
            controller.policies.before_explicit(lowered, scope)
        executor.submit(operation, lowered, scope)

    def finish(self, request: Request) -> None:
        pending = self._pending.pop(request.request_id, ())
        for work in pending:
            try:
                if work.request is not request or request.status not in (
                    RequestStatus.FINISHED_STOPPED,
                    RequestStatus.FINISHED_LENGTH_CAPPED,
                ):
                    self.controller.registry.update(
                        work.operation,
                        OperationStatus.CANCELLED,
                        reason="Request did not finish successfully",
                    )
                    continue
                executor = self.controller.executor
                assert executor is not None
                if (
                    executor.manager.get_transfer_manager().generation
                    != work.generation
                ):
                    self.controller.registry.update(
                        work.operation,
                        OperationStatus.CANCELLED,
                        reason="Cache reset invalidated the queued action",
                    )
                    continue
                self._execute_now(
                    work.operation, work.payload, work.scope, request=request
                )
            except ControlError as error:
                self.controller.record_error(work.operation, error)
            except Exception as error:
                self.controller._record_failure(
                    work.operation, OperationStatus.FAILED, type(error).__name__
                )

    def cancel(self, operation_id: int) -> None:
        for request_id, works in tuple(self._pending.items()):
            remaining = [w for w in works if w.operation.operation_id != operation_id]
            if remaining:
                self._pending[request_id] = remaining
            else:
                del self._pending[request_id]

    def capabilities(self) -> tuple[ActionCapability, ...]:
        executor = self.controller.executor
        if executor is None:
            return ()
        targets: tuple[
            Literal["hashes", "session", "request", "request_cached"], ...
        ] = (
            ("hashes", "session", "request", "request_cached")
            if self.controller.request_observer is not None
            else ("hashes", "session")
        )
        result = [
            ActionCapability(
                "vllm.set_priority",
                PRIMITIVE_VERSION,
                targets,
                ("gpu",),
                True,
                (
                    "Copy-lifetime claims; attached actions apply at "
                    "successful completion",
                    "request_cached requires a request binding",
                ),
            )
        ]
        for kind in ("vllm.copy_to", "vllm.move_to"):
            selected = [
                c
                for c in executor.capabilities()
                if c.action_type == "demote"
                or (kind == "vllm.copy_to" and c.action_type == "prefetch")
            ]
            if not selected:
                continue
            tiers = tuple(dict.fromkeys(t for c in selected for t in c.tiers))
            directions = tuple(
                dict.fromkeys(
                    (f"{tier}->gpu" if c.action_type == "prefetch" else f"gpu->{tier}")
                    for c in selected
                    for tier in c.tiers
                    if tier != "gpu"
                )
            )
            limitations = tuple(
                dict.fromkeys(note for c in selected for note in c.limitations)
            ) + (
                "Attached GPU sources execute at successful completion; "
                "cold COPY at ingress",
                "No remote G1 allocation or cold-tier deletion",
            )
            result.append(
                ActionCapability(
                    kind,
                    PRIMITIVE_VERSION,
                    targets,
                    tiers,
                    True,
                    limitations,
                    directions,
                )
            )
        return tuple(result)
