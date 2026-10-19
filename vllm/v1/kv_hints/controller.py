# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native command interpretation, kept separate from physical resource approval."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import msgspec

from vllm.logger import init_logger
from vllm.v1.kv_hints.actions import CopyTo, MoveTo, Primitive, SetPriority
from vllm.v1.kv_hints.control import (
    ControlContext,
    ControlError,
    OperationRegistry,
    OperationResult,
    OperationStatus,
    Submission,
    envelope_fingerprint,
)
from vllm.v1.kv_hints.native import (
    NATIVE_ACTION_VERSION,
    NATIVE_PROTOCOL_VERSION,
    ActionCapability,
    AssociatePayload,
    ConsumerCapabilities,
    ContentSpan,
    DemotePayload,
    EvictPayload,
    FinishTaskPayload,
    HashTarget,
    NativePayload,
    PrefetchPayload,
    ReleasePayload,
    RequestTarget,
    RetainPayload,
    SessionTarget,
    SharePayload,
    SignalPayload,
    StorePayload,
    Target,
)
from vllm.v1.kv_hints.protocol import KvHintsEnvelope
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import LogicalResolution, SessionPrefixIndex

if TYPE_CHECKING:
    from vllm.v1.kv_hints.policy_runtime import KVPolicyRuntime
    from vllm.v1.request import Request

PhysicalPayload: TypeAlias = (
    EvictPayload | DemotePayload | PrefetchPayload | SharePayload | StorePayload
)
RequestResolver: TypeAlias = Callable[[RequestTarget, str | None], HashTarget]
logger = init_logger(__name__)


class KVOperationExecutor(Protocol):
    """Engine resource authority; adapters cannot replace this approval boundary."""

    def capabilities(self) -> tuple[ActionCapability, ...]: ...

    manager: Any

    def request_scope(
        self, request: "Request", target: RequestTarget | str
    ) -> LogicalResolution: ...

    def apply_priority(
        self,
        operation: OperationResult,
        action: SetPriority,
        scope: LogicalResolution,
        request: "Request | None",
        owners: tuple[str, ...] = (),
    ) -> None: ...

    def validate_scope(self, target: LogicalResolution) -> None: ...

    def residency(self, target: LogicalResolution, limit: int) -> dict[str, Any]: ...

    def submit(
        self,
        operation: OperationResult,
        payload: PhysicalPayload,
        target: LogicalResolution,
    ) -> None: ...

    def cancel_owners(self, owners: tuple[str, ...]) -> None: ...

    def progress(self) -> None: ...

    def cancel_operations(self, operation_ids: set[int], reason: str) -> None: ...

    def conflicts_with_explicit(
        self, payload: NativePayload | Primitive, scope: LogicalResolution
    ) -> bool: ...


class KVRequestObserver(Protocol):
    def request(self, target: RequestTarget, request_id: str | None) -> "Request": ...
    def attach(self, request: "Request") -> AbstractContextManager[None]: ...

    def resolve(self, target: RequestTarget, request_id: str | None) -> HashTarget: ...

    def resolve_finished(
        self, request: "Request", target: RequestTarget
    ) -> HashTarget: ...

    def associate(
        self,
        payload: AssociatePayload,
        request_id: str | None,
        blocks: tuple[ContentSpan, ...],
    ) -> None: ...

    def observe(self) -> None: ...

    def session_for(
        self, target: RequestTarget, request_id: str | None
    ) -> str | None: ...


class NativeKVController:
    def __init__(
        self,
        registry: OperationRegistry,
        index: SessionPrefixIndex,
        declarations: RetentionDeclarations,
        *,
        executor: KVOperationExecutor | None = None,
        request_resolver: RequestResolver | None = None,
        cache_profile: dict[str, Any] | None = None,
        request_observer: KVRequestObserver | None = None,
    ):
        if (
            index.engine_epoch != registry.engine_epoch
            or declarations.index is not index
        ):
            raise ValueError("Control consumers must share one index and engine epoch")
        self.registry = registry
        self.index = index
        self.declarations = declarations
        self.executor = executor
        self.request_resolver = request_resolver
        self.cache_profile = cache_profile
        self.request_observer = request_observer
        self.policies: KVPolicyRuntime | None = None
        from vllm.v1.kv_hints.dispatch import KvHintDispatcher

        self.dispatcher = KvHintDispatcher(self)

    def submit(
        self,
        envelope: KvHintsEnvelope,
        context: ControlContext,
        *,
        request_id: str | None = None,
    ) -> Submission:
        """Interpret a whole native envelope without changing the merged records."""
        if context.consumer == "native":
            if (
                context.namespace is not None
                and context.namespace != self.index.namespace
            ):
                raise ControlError(
                    "incompatible", "Invocation targets another cache namespace"
                )
            context = replace(context, namespace=self.index.namespace)
        if request_id is not None:
            if context.binding_id is not None and context.binding_id != request_id:
                raise ControlError("conflict", "Invocation has another request binding")
            context = replace(context, binding_id=request_id)
        receipt = self.registry.submit(envelope, context)
        if receipt.replayed:
            return receipt
        if context.consumer != "native":
            for operation in receipt.results:
                self.registry.update(
                    operation,
                    OperationStatus.UNSUPPORTED,
                    reason="Backend control requires its existing request-scoped path",
                )
            return self._submission(receipt)
        if envelope.protocol_version != NATIVE_PROTOCOL_VERSION:
            for operation in receipt.results:
                self.registry.update(
                    operation,
                    OperationStatus.UNSUPPORTED,
                    reason="Unsupported native envelope version",
                )
            return self._submission(receipt)

        # Validate every native payload before executing any of the actions.
        parsed: list[NativePayload | Primitive | ControlError] = []
        for action in envelope.actions:
            try:
                parsed.append(self.dispatcher.parse(action))
            except ControlError as error:
                parsed.append(error)
        for operation, payload in zip(receipt.results, parsed):
            self.execute_registered(
                operation,
                payload,
                request_id=request_id,
                attached=request_id is not None,
            )
        return self._submission(receipt)

    def execute_registered(
        self,
        operation: OperationResult,
        payload: NativePayload | Primitive | ControlError,
        *,
        request_id: str | None = None,
        resolution: LogicalResolution | None = None,
        attached: bool = False,
        policy_owners: tuple[str, ...] = (),
    ) -> None:
        try:
            if isinstance(payload, ControlError):
                raise payload
            if isinstance(payload, (SetPriority, CopyTo, MoveTo)):
                self.dispatcher.execute(
                    operation,
                    payload,
                    request_id=request_id,
                    resolution=resolution,
                    attached=attached,
                    owners=policy_owners,
                )
            else:
                self._execute(operation, payload, request_id, resolution)
        except ControlError as error:
            self.record_error(operation, error)
        except Exception as error:
            logger.exception(
                "KV control failed for message %r action %r",
                operation.message_id,
                operation.action_id,
            )
            self._record_failure(
                operation, OperationStatus.FAILED, type(error).__name__
            )

    def consume_attached(
        self,
        envelope: KvHintsEnvelope,
        context: ControlContext,
        *,
        request_id: str,
    ) -> Submission | None:
        """Return None for backend ownership, preserving its existing request hook."""
        self.registry.validate_context(context)
        if context.consumer == "backend":
            envelope_fingerprint(
                envelope,
                context,
                max_actions=self.registry.max_actions,
                max_bytes=self.registry.max_envelope_bytes,
            )
            return None
        return self.submit(envelope, context, request_id=request_id)

    def capabilities(self) -> ConsumerCapabilities:
        targets = (
            ("hashes", "session", "request")
            if self.request_resolver
            else ("hashes", "session")
        )
        actions = [
            ActionCapability(
                "retain",
                NATIVE_ACTION_VERSION,
                targets,
                ("gpu",),
                True,
                ("Eviction bias only; no residency guarantee",),
            ),
            ActionCapability("release", NATIVE_ACTION_VERSION, (), ("gpu",), True),
            ActionCapability(
                "associate",
                NATIVE_ACTION_VERSION,
                ("hashes", "request") if self.request_resolver else ("hashes",),
                (),
                True,
            ),
            ActionCapability("finish_task", NATIVE_ACTION_VERSION, (), (), True),
            ActionCapability(
                "signal",
                NATIVE_ACTION_VERSION,
                targets,
                (),
                True,
                ("Unconsumed without an enabled local policy",),
            ),
        ]
        if self.executor is not None:
            for capability in self.executor.capabilities():
                if self.request_resolver is not None and "hashes" in capability.targets:
                    capability = msgspec.structs.replace(
                        capability,
                        targets=tuple(dict.fromkeys((*capability.targets, "request"))),
                    )
                actions.append(capability)
        actions.extend(self.dispatcher.capabilities())
        return ConsumerCapabilities(
            self.registry.engine_id,
            self.registry.engine_epoch,
            "native",
            (NATIVE_PROTOCOL_VERSION,),
            tuple(actions),
            self.index.namespace,
            self.registry.result_ttl,
            self.registry.max_actions,
            self.registry.max_envelope_bytes,
            self.cache_profile,
        )

    def cancel(
        self, message_id: str, action_id: str, operation_id: int
    ) -> OperationResult:
        current = self.registry.get(message_id, action_id)
        if current is None:
            raise ControlError("missing", "Operation is unknown or expired")
        if current.operation_id != operation_id:
            raise ControlError(
                "stale_operation", "Cancellation targets an earlier action"
            )
        if current.status.terminal:
            return current
        self.dispatcher.cancel(operation_id)
        if self.policies is not None:
            self.policies.cancel_operation(operation_id)
        if self.executor is not None:
            self.executor.cancel_operations(
                {operation_id}, "Caller cancelled the operation"
            )
        current = self.registry.get(message_id, action_id)
        assert current is not None
        if not current.status.terminal:
            current = self.registry.update(
                current,
                OperationStatus.CANCELLED,
                reason="Caller cancelled the operation",
            )
        return current

    def _target(
        self, target: Target, request_id: str | None
    ) -> HashTarget | SessionTarget:
        if isinstance(target, RequestTarget):
            if self.request_resolver is None:
                raise ControlError(
                    "unsupported", "Active-request targeting is unavailable"
                )
            return self.request_resolver(target, request_id)
        return target

    def _execute(
        self,
        operation: OperationResult,
        payload: NativePayload,
        request_id: str | None,
        resolution: LogicalResolution | None = None,
    ) -> None:
        steps: tuple[str, ...]
        status = OperationStatus.COMPLETED
        missing_scope = 0
        if isinstance(payload, SignalPayload):
            if self.policies is not None:
                self.policies.on_signal(operation, payload, request_id)
                return
            self.registry.update(
                operation,
                OperationStatus.UNCONSUMED,
                reason="No local policy consumes this signal",
            )
            return
        if isinstance(payload, RetainPayload):
            target = (
                payload.target
                if resolution is not None
                else self._target(payload.target, request_id)
            )
            assert not isinstance(target, RequestTarget)
            scope = resolution if resolution is not None else self.index.resolve(target)
            if self.executor is not None:
                self.executor.validate_scope(scope)
            self._check_policy_declaration(
                operation, payload.owner_id, payload.declaration_id
            )
            declaration = self.declarations.retain(
                msgspec.structs.replace(payload, target=target),
                policy_name=operation.origin.policy_name if operation.origin else None,
                resolution=scope,
            )
            if self.policies is not None and operation.origin is None:
                self.policies.before_explicit(payload, scope)
            scope = self.declarations.coverage(
                declaration.owner_id, declaration.declaration_id
            )
            status = (
                OperationStatus.PARTIAL
                if any(p.partial_blocks or p.missing_ranges for p in scope.prefixes)
                else OperationStatus.COMPLETED
            )
            self.registry.update(
                operation,
                status,
                completed_steps=("declaration_effective",),
                missing_targets=sum(
                    len(p.partial_blocks) + len(p.missing_ranges)
                    for p in scope.prefixes
                ),
                reason="Range includes unrepresented or unaligned content"
                if status == OperationStatus.PARTIAL
                else "",
            )
            return
        if isinstance(payload, ReleasePayload):
            self._check_policy_declaration(
                operation, payload.owner_id, payload.declaration_id
            )
            if not self.declarations.release(payload.owner_id, payload.declaration_id):
                raise ControlError("missing", "Declaration is unknown or expired")
            steps = ("declaration_released",)
        elif isinstance(payload, AssociatePayload):
            if payload.observe_growth and self.request_observer is None:
                raise ControlError(
                    "unsupported", "Request growth observation is not configured"
                )
            target = self._target(payload.target, request_id)
            assert isinstance(target, HashTarget)
            scope = self.index.resolve(target)
            if self.executor is not None:
                self.executor.validate_scope(scope)
            if any(p.partial_blocks for p in scope.prefixes):
                raise ControlError(
                    "invalid_payload", "Association range splits content"
                )
            blocks = tuple(b for p in scope.prefixes for b in p.blocks)
            if self.request_observer is not None:
                self.request_observer.associate(payload, request_id, blocks)
            else:
                self.index.associate(
                    payload.session_id,
                    payload.continuation_id,
                    blocks,
                    expected_revision=payload.expected_revision,
                    task_parent_id=payload.task_parent_id,
                    parent_prefix=payload.parent_prefix,
                )
            if any(p.missing_ranges for p in scope.prefixes):
                status = OperationStatus.PARTIAL
                missing_scope = sum(len(p.missing_ranges) for p in scope.prefixes)
            steps = (
                ("association_updated", "request_observation_bound")
                if payload.observe_growth
                else ("association_updated",)
            )
        elif isinstance(payload, FinishTaskPayload):
            owners = self.declarations.finish_task(
                payload.session_id, expected_revision=payload.expected_revision
            )
            steps = ("declarations_released", "associations_removed")
            if self.executor is not None or self.request_observer is not None:
                self.registry.update(
                    operation, OperationStatus.ACCEPTED, completed_steps=steps
                )
            if self.request_observer is not None:
                self.request_observer.observe()
            if self.executor is not None:
                self.executor.cancel_owners(owners)
            if self.policies is not None:
                self.policies.cleanup(owners)
        else:
            assert isinstance(
                payload,
                (
                    EvictPayload,
                    DemotePayload,
                    PrefetchPayload,
                    SharePayload,
                    StorePayload,
                ),
            )
            if self.executor is None:
                raise ControlError(
                    "unsupported", "No physical operation executor configured"
                )
            if resolution is None:
                target = self._target(payload.target, request_id)
                assert not isinstance(target, RequestTarget)
                resolution = self.index.resolve(target)
            if self.policies is not None and operation.origin is None:
                self.executor.validate_scope(resolution)
                self.policies.before_explicit(payload, resolution)
            self.executor.submit(operation, payload, resolution)
            return
        self.registry.update(
            operation,
            status,
            completed_steps=steps,
            missing_targets=missing_scope,
            reason="Range includes unrepresented content" if missing_scope else "",
        )

    def _check_policy_declaration(
        self,
        operation: OperationResult,
        owner_id: str,
        declaration_id: str,
    ) -> None:
        if operation.origin is None:
            return
        existing = self.declarations.get(owner_id, declaration_id)
        if (
            existing is not None
            and existing.policy_name != operation.origin.policy_name
        ):
            raise ControlError(
                "busy", "A policy cannot replace another caller's declaration"
            )

    def _submission(self, receipt: Submission) -> Submission:
        results = []
        for operation in receipt.results:
            result = self.registry.get(operation.message_id, operation.action_id)
            assert result is not None
            results.append(result)
        return Submission(receipt.replayed, tuple(results))

    def _record_failure(
        self, operation: OperationResult, status: OperationStatus, reason: str
    ) -> None:
        current = self.registry.get(operation.message_id, operation.action_id)
        assert current is not None
        if not current.status.terminal:
            if current.completed_steps:
                status = OperationStatus.PARTIAL
            self.registry.update(operation, status, reason=reason)

    def record_error(self, operation: OperationResult, error: ControlError) -> None:
        status = {
            "unsupported": OperationStatus.UNSUPPORTED,
            "unsupported_action": OperationStatus.UNSUPPORTED,
            "unsupported_version": OperationStatus.UNSUPPORTED,
            "missing": OperationStatus.MISSING,
            "missing_association": OperationStatus.MISSING,
            "capacity": OperationStatus.DEFERRED,
            "busy": OperationStatus.DEFERRED,
        }.get(error.code, OperationStatus.FAILED)
        self._record_failure(operation, status, f"{error.code}: {error}"[:4096])
