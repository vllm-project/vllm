# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable utility calls shared by Python and HTTP control clients."""

from typing import Annotated, Any, Literal

import msgspec

from vllm.v1.kv_hints.control import ControlContext, ControlError
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.native import Identifier, TokenPosition
from vllm.v1.kv_hints.protocol import KvHintsEnvelope


class ControlInvocation(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    engine_id: Identifier
    engine_epoch: Identifier
    consumer: Literal["native", "backend"] = "native"
    namespace: Identifier | None = None


class OperationQuery(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    message_id: Identifier
    action_id: Identifier


class AssociationQuery(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    session_id: Identifier | None = None
    after: tuple[Identifier, Identifier] | None = None
    revision: TokenPosition | None = None
    limit: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 256


class EventQuery(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    stream: Literal["operation", "association"]
    after: TokenPosition = 0
    limit: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 256


def unwrap_control_reply(reply: dict[str, Any]) -> Any:
    """Restore structured control errors after safe utility serialization."""
    if not reply["ok"]:
        error = reply["error"]
        raise ControlError(error["code"], error["message"])
    return reply["result"]


class KVControlService:
    """Scheduler-thread utility facade over the common native control records."""

    def __init__(self, controller: NativeKVController):
        self.controller = controller

    def call(
        self, method: str, invocation: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            if (
                len(msgspec.json.encode(payload))
                > self.controller.registry.max_envelope_bytes
            ):
                raise ControlError(
                    "invalid_payload", "Control payload byte limit exceeded"
                )
            route = msgspec.convert(invocation, type=ControlInvocation)
            context = ControlContext(
                route.engine_id,
                route.engine_epoch,
                route.consumer,
                namespace=route.namespace,
            )
            self.controller.registry.validate_context(context)
            if (
                route.namespace is not None
                and route.namespace != self.controller.index.namespace
            ):
                raise ControlError(
                    "incompatible", "Invocation targets another cache namespace"
                )
            if method == "submit":
                envelope = msgspec.convert(payload, type=KvHintsEnvelope)
                result: Any = self.controller.submit(envelope, context)
            elif method == "operation":
                query = msgspec.convert(payload, type=OperationQuery)
                result = self.controller.registry.get(query.message_id, query.action_id)
                if result is None:
                    raise ControlError("missing", "Operation is unknown or expired")
            elif route.consumer != "native":
                raise ControlError(
                    "unsupported", "Backend path has no native state queries"
                )
            elif method == "associations":
                association_query = msgspec.convert(payload, type=AssociationQuery)
                result = self.controller.index.query(
                    session_id=association_query.session_id,
                    after=association_query.after,
                    revision=association_query.revision,
                    limit=association_query.limit,
                )
            elif method == "events":
                event_query = msgspec.convert(payload, type=EventQuery)
                source = (
                    self.controller.registry
                    if event_query.stream == "operation"
                    else self.controller.index
                )
                result = source.events(
                    event_query.after, epoch=route.engine_epoch, limit=event_query.limit
                )
            else:
                raise ControlError("unsupported", "Unknown control method")
            return {
                "ok": True,
                "result": msgspec.json.decode(msgspec.json.encode(result)),
            }
        except ControlError as error:
            return {"ok": False, "error": {"code": error.code, "message": str(error)}}
        except (msgspec.ValidationError, TypeError, ValueError) as error:
            return {
                "ok": False,
                "error": {"code": "invalid_payload", "message": str(error)[:4096]},
            }
