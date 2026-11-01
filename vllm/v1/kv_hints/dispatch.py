# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import msgspec

from vllm.logger import init_logger
from vllm.v1.kv_hints.actions import KvHintResult
from vllm.v1.kv_hints.protocol import KvHintAction

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)
Payload = TypeVar("Payload")


@dataclass(frozen=True)
class _Consumer(Generic[Payload]):
    payload_type: type[Payload]
    execute: Callable[[Payload, "Request"], KvHintResult]
    when: Literal["ingress", "successful_completion"]


@dataclass(frozen=True)
class _DeferredAction:
    message_id: str
    action: KvHintAction
    execute: Callable[[], KvHintResult]


def _log_rejection(message_id: str, action: KvHintAction, result: KvHintResult) -> None:
    if result.status == "rejected":
        logger.warning(
            "Rejected KV hint %s/%s: %s", message_id, action.action_id, result.reason
        )


class KvHintDispatcher:
    """Decode native actions at ingress and dispatch at their registered phase."""

    def __init__(self) -> None:
        self._consumers: dict[tuple[str, str], _Consumer[Any]] = {}
        self._pending: dict[str, list[_DeferredAction]] = {}

    def register(
        self,
        action_type: str,
        action_version: str,
        payload_type: type[Payload],
        execute: Callable[[Payload, "Request"], KvHintResult],
        *,
        when: Literal["ingress", "successful_completion"],
    ) -> None:
        key = (action_type, action_version)
        if key in self._consumers:
            raise ValueError(f"KV hint consumer already registered: {key}")
        self._consumers[key] = _Consumer(payload_type, execute, when)

    def dispatch(self, request: "Request") -> list[tuple[KvHintAction, KvHintResult]]:
        """Route native hints while leaving backend-owned actions untouched."""
        envelope = request.kv_hints
        if envelope is None:
            return []
        results = []
        for action in envelope.actions:
            if not action.action_type.startswith("vllm."):
                continue
            consumer = self._consumers.get((action.action_type, action.action_version))
            if envelope.protocol_version != "0.1" or consumer is None:
                result = KvHintResult("unsupported")
            else:
                try:
                    payload = msgspec.convert(
                        action.payload, type=consumer.payload_type, strict=True
                    )
                except (msgspec.ValidationError, ValueError) as exc:
                    result = KvHintResult("rejected", reason=str(exc))
                else:
                    if consumer.when == "ingress":
                        result = consumer.execute(payload, request)
                    else:
                        self._pending.setdefault(request.request_id, []).append(
                            _DeferredAction(
                                envelope.message_id,
                                action,
                                partial(consumer.execute, payload, request),
                            )
                        )
                        result = KvHintResult("deferred")
            _log_rejection(envelope.message_id, action, result)
            results.append((action, result))
        return results

    def finish(
        self, request: "Request", *, successful: bool
    ) -> list[tuple[KvHintAction, KvHintResult]]:
        """Release deferred actions on every terminal path; apply only on success."""
        pending = self._pending.pop(request.request_id, ())
        results = []
        if successful:
            for item in pending:
                result = item.execute()
                _log_rejection(item.message_id, item.action, result)
                results.append((item.action, result))
        return results
