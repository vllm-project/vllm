# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, engine-owned correlation for programmable cache operations."""

import hashlib
import json
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope


class ControlError(ValueError):
    """A control request was rejected before changing operation state."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OperationStatus(str, Enum):
    RECEIVED = "received"
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
            OperationStatus.ACCEPTED,
            OperationStatus.RUNNING,
        )


@dataclass(frozen=True)
class ControlContext:
    engine_id: str
    engine_epoch: str
    consumer: Literal["native", "backend"]
    binding_id: str | None = None
    namespace: str | None = None


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


@dataclass(frozen=True)
class Submission:
    replayed: bool
    results: tuple[OperationResult, ...]


@dataclass
class _Message:
    fingerprint: bytes
    results: dict[str, OperationResult]
    expires_at: float | None = None

    @property
    def size(self) -> int:
        return max(1, len(self.results))


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
        json.dumps((context.consumer, context.binding_id, context.namespace)).encode(
            "utf-8"
        )
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
        max_actions: int = 128,
        max_envelope_bytes: int = 1 << 20,
        max_events: int = 4096,
        result_ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        _validate_id(engine_id, "engine_id")
        if any(
            type(n) is not int or n < 1
            for n in (max_records, max_actions, max_envelope_bytes, max_events)
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
        self.max_actions = max_actions
        self.max_envelope_bytes = max_envelope_bytes
        self.result_ttl = result_ttl
        self._clock = clock
        self._messages: dict[str, _Message] = {}
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
            )
            results[action.action_id] = result
            self._events.append(result)
        self._messages[envelope.message_id] = _Message(
            fingerprint,
            results,
            None if results else self._clock() + self.result_ttl,
        )
        self._record_count += size
        return Submission(False, tuple(results.values()))

    def update(
        self,
        operation: OperationResult,
        status: OperationStatus,
        *,
        reason: str = "",
        completed_steps: tuple[str, ...] | None = None,
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
        ):
            raise ControlError(
                "stale_operation", "Completion refers to an expired action"
            )
        if not isinstance(status, OperationStatus):
            raise ValueError("Expected OperationStatus")
        if len(reason) > 4096:
            raise ValueError("Operation reason exceeds 4096 characters")
        steps = current.completed_steps if completed_steps is None else completed_steps
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
        ):
            return current
        if current.status.terminal:
            raise ControlError("terminal", "An operation outcome is already terminal")
        if status == OperationStatus.RECEIVED or (
            status == OperationStatus.RUNNING
            and current.status == OperationStatus.RECEIVED
        ):
            raise ControlError("transition", "Operation has not been approved")
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
        )
        message = self._messages[operation.message_id]
        message.results[operation.action_id] = result
        if all(item.status.terminal for item in message.results.values()):
            message.expires_at = self._clock() + self.result_ttl
        self._events.append(result)
        return result

    def get(self, message_id: str, action_id: str) -> OperationResult | None:
        self.prune()
        message = self._messages.get(message_id)
        return message.results.get(action_id) if message is not None else None

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
        return tuple(event for event in self._events if event.sequence > after)[:limit]

    def prune(self) -> None:
        now = self._clock()
        expired = [
            key
            for key, message in self._messages.items()
            if message.expires_at is not None and message.expires_at <= now
        ]
        for key in expired:
            self._record_count -= self._messages.pop(key).size
