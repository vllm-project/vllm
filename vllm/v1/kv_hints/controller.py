# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native command interpretation, kept separate from physical resource approval."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol, TypeAlias

import msgspec

from vllm.logger import init_logger
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
    Target,
    parse_native_action,
)
from vllm.v1.kv_hints.protocol import KvHintsEnvelope
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import LogicalResolution, SessionPrefixIndex

PhysicalPayload: TypeAlias = (
    EvictPayload | DemotePayload | PrefetchPayload | SharePayload
)
RequestResolver: TypeAlias = Callable[[RequestTarget, str | None], HashTarget]
logger = init_logger(__name__)


class KVOperationExecutor(Protocol):
    """Engine resource authority; adapters cannot replace this approval boundary."""

    def capabilities(self) -> tuple[ActionCapability, ...]: ...

    def validate_scope(self, target: LogicalResolution) -> None: ...

    def submit(
        self,
        operation: OperationResult,
        payload: PhysicalPayload,
        target: LogicalResolution,
    ) -> None: ...

    def cancel_owners(self, owners: tuple[str, ...]) -> None: ...


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
        parsed: list[NativePayload | ControlError] = []
        for action in envelope.actions:
            try:
                parsed.append(parse_native_action(action))
            except ControlError as error:
                parsed.append(error)
        for operation, payload in zip(receipt.results, parsed):
            try:
                if isinstance(payload, ControlError):
                    raise payload
                self._execute(operation, payload, request_id)
            except ControlError as error:
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
            except Exception as error:
                logger.exception(
                    "KV control failed for message %r action %r",
                    operation.message_id,
                    operation.action_id,
                )
                self._record_failure(
                    operation, OperationStatus.FAILED, type(error).__name__
                )
        return self._submission(receipt)

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
        self, operation: OperationResult, payload: NativePayload, request_id: str | None
    ) -> None:
        steps: tuple[str, ...]
        if isinstance(payload, SignalPayload):
            self.registry.update(
                operation,
                OperationStatus.UNCONSUMED,
                reason="No local policy consumes this signal",
            )
            return
        if isinstance(payload, RetainPayload):
            target = self._target(payload.target, request_id)
            if self.executor is not None:
                self.executor.validate_scope(self.index.resolve(target))
            declaration = self.declarations.retain(
                msgspec.structs.replace(payload, target=target)
            )
            scope = self.declarations.coverage(
                declaration.owner_id, declaration.declaration_id
            )
            status = (
                OperationStatus.PARTIAL
                if any(p.partial_blocks or p.missing_ranges for p in scope.prefixes)
                else OperationStatus.COMPLETED
            )
            self.registry.update(
                operation, status, completed_steps=("declaration_effective",)
            )
            return
        if isinstance(payload, ReleasePayload):
            if not self.declarations.release(payload.owner_id, payload.declaration_id):
                raise ControlError("missing", "Declaration is unknown or expired")
            steps = ("declaration_released",)
        elif isinstance(payload, AssociatePayload):
            target = self._target(payload.target, request_id)
            assert isinstance(target, HashTarget)
            scope = self.index.resolve(target)
            if self.executor is not None:
                self.executor.validate_scope(scope)
            if any(p.partial_blocks for p in scope.prefixes):
                raise ControlError(
                    "invalid_payload", "Association range splits content"
                )
            self.index.associate(
                payload.session_id,
                payload.continuation_id,
                tuple(b for p in scope.prefixes for b in p.blocks),
                expected_revision=payload.expected_revision,
                task_parent_id=payload.task_parent_id,
                parent_prefix=payload.parent_prefix,
            )
            steps = ("association_updated",)
        elif isinstance(payload, FinishTaskPayload):
            owners = self.declarations.finish_task(
                payload.session_id, expected_revision=payload.expected_revision
            )
            steps = ("declarations_released", "associations_removed")
            if self.executor is not None:
                self.registry.update(
                    operation, OperationStatus.ACCEPTED, completed_steps=steps
                )
                self.executor.cancel_owners(owners)
        else:
            assert isinstance(
                payload, (EvictPayload, DemotePayload, PrefetchPayload, SharePayload)
            )
            if self.executor is None:
                raise ControlError(
                    "unsupported", "No physical operation executor configured"
                )
            target = self._target(payload.target, request_id)
            assert not isinstance(target, RequestTarget)
            self.executor.submit(operation, payload, self.index.resolve(target))
            return
        self.registry.update(
            operation, OperationStatus.COMPLETED, completed_steps=steps
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
