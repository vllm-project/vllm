# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, engine-owned correlation for programmable cache operations."""

import hashlib
import heapq
import json
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from itertools import islice
from typing import Literal

from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope


class ControlError(ValueError):
    """A control request was rejected before changing operation state."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OperationStatus(str, Enum):
    RECEIVED = "received"
    QUEUED = "queued"
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    DEFERRED = "deferred"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCONSUMED = "unconsumed"

    @property
    def terminal(self) -> bool:
        return self not in (
            OperationStatus.RECEIVED,
            OperationStatus.QUEUED,
            OperationStatus.ACCEPTED,
            OperationStatus.RUNNING,
        )


@dataclass(frozen=True)
class OperationReference:
    message_id: str
    action_id: str
    operation_id: int


@dataclass(frozen=True)
class OperationOrigin(OperationReference):
    policy_name: str


@dataclass(frozen=True)
class ControlContext:
    engine_id: str
    engine_epoch: str
    consumer: Literal["native", "backend"]
    binding_id: str | None = None
    namespace: str | None = None
    origin: OperationOrigin | None = None


@dataclass(frozen=True)
class TargetOutcome:
    block_hash: str | None
    group_id: int | None
    start_token: int
    end_token: int
    status: OperationStatus
    completed_steps: tuple[str, ...] = ()
    reason: str = ""
    transfer_id: int | None = None
    generation: int | None = None


@dataclass(frozen=True)
class OperationResult:
    engine_id: str
    engine_epoch: str
    message_id: str
    action_id: str
    consumer: Literal["native", "backend"]
    operation_id: int
    status: OperationStatus
    sequence: int
    reason: str = ""
    completed_steps: tuple[str, ...] = ()
    binding_id: str | None = None
    namespace: str | None = None
    affected_blocks: int = 0
    blocked_targets: int = 0
    missing_targets: int = 0
    displaced_blocks: int = 0
    origin: OperationOrigin | None = None
    derived_operations: tuple[OperationReference, ...] = ()
    target_outcomes: tuple[TargetOutcome, ...] = ()
    action_type: str = ""
    action_version: str = ""
    transferred_bytes: int = 0


@dataclass(frozen=True)
class Submission:
    replayed: bool
    results: tuple[OperationResult, ...]


@dataclass
class _Message:
    fingerprint: bytes
    results: dict[str, OperationResult]
    acknowledged: bool = False

    @property
    def size(self) -> int:
        return 0 if self.acknowledged else max(1, len(self.results))


def _validate_id(value: object, name: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ControlError("invalid_envelope", f"{name} must have 1 to 256 characters")


def _validate_json(value: object, depth: int = 0) -> None:
    if depth > 32:
        raise ControlError("invalid_envelope", "Payload nesting exceeds 32 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item, depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ControlError(
                    "invalid_envelope", "JSON object keys must be strings"
                )
            _validate_json(item, depth + 1)
        return
    raise ControlError("invalid_envelope", "Payload must contain finite JSON values")


def envelope_fingerprint(
    envelope: KvHintsEnvelope,
    context: ControlContext,
    *,
    max_actions: int,
    max_bytes: int,
) -> bytes:
    """Validate the wrapper and opaque JSON, without interpreting action payloads."""
    if not isinstance(envelope, KvHintsEnvelope):
        raise ControlError("invalid_envelope", "Expected KvHintsEnvelope")
    for name in ("protocol_version", "message_id"):
        _validate_id(getattr(envelope, name), name)
    if not isinstance(envelope.actions, list) or len(envelope.actions) > max_actions:
        raise ControlError("invalid_envelope", "Envelope action limit exceeded")
    actions = []
    seen: set[str] = set()
    for action in envelope.actions:
        if not isinstance(action, KvHintAction):
            raise ControlError("invalid_envelope", "Expected KvHintAction")
        for name in ("action_id", "action_type", "action_version"):
            _validate_id(getattr(action, name), name)
        if action.action_id in seen:
            raise ControlError("invalid_envelope", "Duplicate action_id in envelope")
        seen.add(action.action_id)
        if type(action.payload) is not dict:
            raise ControlError("invalid_envelope", "Action payload must be an object")
        _validate_json(action.payload)
        actions.append(
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "action_version": action.action_version,
                "payload": action.payload,
            }
        )
    body = {
        "protocol_version": envelope.protocol_version,
        "message_id": envelope.message_id,
        "actions": actions,
    }
    try:
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, OverflowError) as error:
        raise ControlError("invalid_envelope", "Envelope cannot be encoded") from error
    if len(encoded) > max_bytes:
        raise ControlError("invalid_envelope", "Envelope byte limit exceeded")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            (
                context.consumer,
                context.binding_id,
                context.namespace,
                None
                if context.origin is None
                else (
                    context.origin.message_id,
                    context.origin.action_id,
                    context.origin.operation_id,
                    context.origin.policy_name,
                ),
            )
        ).encode("utf-8")
    )
    digest.update(encoded)
    return digest.digest()


class OperationRegistry:
    """Scheduler-thread records; terminal results survive their advertised TTL.

    This class owns correlation, not transfer resources. Executors must drain
    worker references even when a caller cancels. A restart creates a new epoch.
    """

    def __init__(
        self,
        engine_id: str,
        *,
        engine_epoch: str | None = None,
        max_records: int = 4096,
        max_acknowledged_messages: int = 65536,
        max_actions: int = 128,
        max_envelope_bytes: int = 1 << 20,
        max_events: int = 4096,
        max_result_targets: int = 16384,
        result_ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        _validate_id(engine_id, "engine_id")
        if any(
            type(n) is not int or n < 1
            for n in (
                max_records,
                max_acknowledged_messages,
                max_actions,
                max_envelope_bytes,
                max_events,
                max_result_targets,
            )
        ):
            raise ValueError("Control limits must be positive integers")
        if not math.isfinite(result_ttl) or result_ttl <= 0:
            raise ValueError("result_ttl must be positive and finite")
        self.engine_id = engine_id
        self.engine_epoch = (
            engine_epoch if engine_epoch is not None else uuid.uuid4().hex
        )
        _validate_id(self.engine_epoch, "engine_epoch")
        self.max_records = max_records
        self.max_acknowledged_messages = max_acknowledged_messages
        self._acknowledged_count = 0
        self.max_actions = max_actions
        self.max_envelope_bytes = max_envelope_bytes
        self.result_ttl = result_ttl
        self.max_result_targets = max_result_targets
        self._clock = clock
        self._messages: dict[str, _Message] = {}
        self._expiry: list[tuple[float, str]] = []
        self._record_count = 0
        self._sequence = 0
        self._events: deque[OperationResult] = deque(maxlen=max_events)

    def validate_context(self, context: ControlContext) -> None:
        if context.engine_id != self.engine_id:
            raise ControlError("wrong_engine", "Control targets another engine")
        if context.engine_epoch != self.engine_epoch:
            raise ControlError("stale_epoch", "Control targets an earlier engine epoch")
        if context.consumer not in ("native", "backend"):
            raise ControlError("unsupported_consumer", "Unknown interpretation path")
        if context.binding_id is not None:
            _validate_id(context.binding_id, "binding_id")
        if context.namespace is not None:
            _validate_id(context.namespace, "namespace")
        if context.origin is not None:
            for value in (
                context.origin.message_id,
                context.origin.action_id,
                context.origin.policy_name,
            ):
                _validate_id(value, "origin")
            if (
                type(context.origin.operation_id) is not int
                or context.origin.operation_id < 1
            ):
                raise ControlError("invalid_payload", "Invalid originating operation")

    def submit(self, envelope: KvHintsEnvelope, context: ControlContext) -> Submission:
        self.validate_context(context)
        fingerprint = envelope_fingerprint(
            envelope,
            context,
            max_actions=self.max_actions,
            max_bytes=self.max_envelope_bytes,
        )
        self.prune()
        previous = self._messages.get(envelope.message_id)
        if previous is not None:
            if previous.fingerprint != fingerprint:
                raise ControlError(
                    "conflict", "Message ID already has different content"
                )
            if previous.acknowledged:
                raise ControlError(
                    "acknowledged",
                    "Message completed and its results were acknowledged",
                )
            return Submission(True, tuple(previous.results.values()))
        size = max(1, len(envelope.actions))
        if self._record_count + size > self.max_records:
            raise ControlError("capacity", "Control result capacity exhausted")
        results: dict[str, OperationResult] = {}
        for action in envelope.actions:
            self._sequence += 1
            result = OperationResult(
                self.engine_id,
                self.engine_epoch,
                envelope.message_id,
                action.action_id,
                context.consumer,
                self._sequence,
                OperationStatus.RECEIVED,
                self._sequence,
                binding_id=context.binding_id,
                namespace=context.namespace,
                origin=context.origin,
                action_type=action.action_type,
                action_version=action.action_version,
            )
            results[action.action_id] = result
            self._events.append(result)
        self._messages[envelope.message_id] = _Message(fingerprint, results)
        if not results:
            self._schedule_expiry(envelope.message_id)
        self._record_count += size
        return Submission(False, tuple(results.values()))

    def update(
        self,
        operation: OperationResult,
        status: OperationStatus,
        *,
        reason: str = "",
        completed_steps: tuple[str, ...] | None = None,
        affected_blocks: int | None = None,
        blocked_targets: int | None = None,
        missing_targets: int | None = None,
        displaced_blocks: int | None = None,
        transferred_bytes: int | None = None,
        derived_operations: tuple[OperationReference, ...] | None = None,
        target_outcomes: tuple[TargetOutcome, ...] | None = None,
    ) -> OperationResult:
        """Publish progress after the executor has validated the physical outcome."""
        self.validate_context(
            ControlContext(
                operation.engine_id, operation.engine_epoch, operation.consumer
            )
        )
        current = self.get(operation.message_id, operation.action_id)
        if current is None:
            raise ControlError("missing", "Unknown or expired operation")
        if (
            operation.operation_id != current.operation_id
            or operation.binding_id != current.binding_id
            or operation.consumer != current.consumer
            or operation.namespace != current.namespace
            or operation.origin != current.origin
            or operation.action_type != current.action_type
            or operation.action_version != current.action_version
        ):
            raise ControlError(
                "stale_operation", "Completion refers to an expired action"
            )
        if not isinstance(status, OperationStatus):
            raise ValueError("Expected OperationStatus")
        counts = {
            name: getattr(current, name) if value is None else value
            for name, value in (
                ("affected_blocks", affected_blocks),
                ("blocked_targets", blocked_targets),
                ("missing_targets", missing_targets),
                ("displaced_blocks", displaced_blocks),
                ("transferred_bytes", transferred_bytes),
            )
        }
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("Operation counts must be nonnegative integers")
        if len(reason) > 4096:
            raise ValueError("Operation reason exceeds 4096 characters")
        steps = current.completed_steps if completed_steps is None else completed_steps
        outcomes = (
            current.target_outcomes if target_outcomes is None else target_outcomes
        )
        if (
            not isinstance(outcomes, tuple)
            or len(outcomes) > self.max_result_targets
            or any(not isinstance(outcome, TargetOutcome) for outcome in outcomes)
        ):
            raise ValueError("Target outcomes exceed the supported bounded scope")
        derived = (
            current.derived_operations
            if derived_operations is None
            else derived_operations
        )
        if (
            not isinstance(derived, tuple)
            or len(derived) > 128
            or derived[: len(current.derived_operations)] != current.derived_operations
            or any(not isinstance(ref, OperationReference) for ref in derived)
        ):
            raise ValueError(
                "Derived operations must be bounded append-only references"
            )
        for ref in derived:
            _validate_id(ref.message_id, "derived message_id")
            _validate_id(ref.action_id, "derived action_id")
            if type(ref.operation_id) is not int or ref.operation_id < 1:
                raise ValueError("Invalid derived operation incarnation")
        if len(set(derived)) != len(derived):
            raise ValueError("Duplicate derived operation reference")
        if (
            not isinstance(steps, tuple)
            or len(steps) > 128
            or any(
                not isinstance(step, str) or not 1 <= len(step) <= 256 for step in steps
            )
            or len(set(steps)) != len(steps)
            or steps[: len(current.completed_steps)] != current.completed_steps
        ):
            raise ValueError("Completed steps must be a bounded append-only tuple")
        if (
            status == current.status
            and reason == current.reason
            and steps == current.completed_steps
            and derived == current.derived_operations
            and outcomes == current.target_outcomes
            and all(getattr(current, name) == value for name, value in counts.items())
        ):
            return current
        if current.status.terminal:
            raise ControlError("terminal", "An operation outcome is already terminal")
        if status == OperationStatus.RECEIVED or (
            status == OperationStatus.RUNNING
            and current.status in (OperationStatus.RECEIVED, OperationStatus.QUEUED)
        ):
            raise ControlError("transition", "Operation has not been approved")
        if (
            status == OperationStatus.QUEUED
            and current.status != OperationStatus.RECEIVED
        ):
            raise ControlError(
                "transition", "Only received operations can wait for lifecycle dispatch"
            )
        if (
            status == OperationStatus.ACCEPTED
            and current.status == OperationStatus.RUNNING
        ):
            raise ControlError(
                "transition", "Running operation cannot return to approval"
            )
        self._sequence += 1
        result = replace(
            current,
            status=status,
            sequence=self._sequence,
            reason=reason,
            completed_steps=steps,
            affected_blocks=counts["affected_blocks"],
            blocked_targets=counts["blocked_targets"],
            missing_targets=counts["missing_targets"],
            displaced_blocks=counts["displaced_blocks"],
            transferred_bytes=counts["transferred_bytes"],
            derived_operations=derived,
            target_outcomes=outcomes,
        )
        message = self._messages[operation.message_id]
        message.results[operation.action_id] = result
        if all(item.status.terminal for item in message.results.values()):
            self._schedule_expiry(operation.message_id)
        self._events.append(result)
        return result

    def get(self, message_id: str, action_id: str) -> OperationResult | None:
        self.prune()
        message = self._messages.get(message_id)
        return message.results.get(action_id) if message is not None else None

    def acknowledge(self, message_id: str) -> None:
        """Release terminal result details while keeping the original replay TTL."""
        self.prune()
        message = self._messages.get(message_id)
        if message is None:
            raise ControlError("missing", "Message is unknown or expired")
        if message.acknowledged:
            return
        if any(not result.status.terminal for result in message.results.values()):
            raise ControlError("busy", "Message still has active operations")
        if self._acknowledged_count >= self.max_acknowledged_messages:
            raise ControlError("capacity", "Acknowledged message capacity exhausted")
        self._record_count -= message.size
        message.results.clear()
        message.acknowledged = True
        self._acknowledged_count += 1

    def capacity(self) -> dict[str, int | float]:
        """Report both independently bounded result and replay storage."""
        self.prune()
        return {
            "records": self._record_count,
            "max_records": self.max_records,
            "acknowledged_messages": self._acknowledged_count,
            "max_acknowledged_messages": self.max_acknowledged_messages,
            "result_ttl": self.result_ttl,
        }

    def events(
        self, after: int, *, epoch: str, limit: int = 256
    ) -> tuple[OperationResult, ...]:
        if epoch != self.engine_epoch:
            raise ControlError("stale_epoch", "Events belong to another engine epoch")
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("Event query limit must be between 1 and 4096")
        if type(after) is not int or not 0 <= after <= self._sequence:
            raise ControlError("event_gap", "Invalid event position")
        if self._events and after < self._events[0].sequence - 1:
            raise ControlError(
                "event_gap", "Event history has been pruned; query state"
            )
        return tuple(islice((e for e in self._events if e.sequence > after), limit))

    def _schedule_expiry(self, message_id: str) -> None:
        # Terminal messages are immutable: exactly one expiry per message.
        heapq.heappush(self._expiry, (self._clock() + self.result_ttl, message_id))

    def prune(self) -> None:
        now = self._clock()
        while self._expiry and self._expiry[0][0] <= now:
            _, key = heapq.heappop(self._expiry)
            message = self._messages.pop(key)
            self._record_count -= message.size
            self._acknowledged_count -= message.acknowledged
