# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import msgspec

from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope


class _WireKvHintAction(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Strict wire mirror of `KvHintAction` (which allows unknown fields)."""

    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, Any]


class _WireKvHintsEnvelope(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Strict wire mirror of `KvHintsEnvelope` (which allows unknown fields)."""

    protocol_version: str
    message_id: str
    actions: list[_WireKvHintAction]


def kv_hints_from_wire(payload: dict[str, Any] | None) -> KvHintsEnvelope | None:
    """Decode an OpenAI request body `kv_hints` object into an envelope.

    Raises:
        ValueError: if the payload does not match the envelope wire schema.

    """
    if payload is None:
        return None
    try:
        wire = msgspec.convert(payload, type=_WireKvHintsEnvelope, strict=True)
    except (msgspec.ValidationError, TypeError) as exc:
        raise ValueError(f"Invalid kv_hints payload: {exc}") from exc
    return KvHintsEnvelope(
        protocol_version=wire.protocol_version,
        message_id=wire.message_id,
        actions=[
            KvHintAction(a.action_id, a.action_type, a.action_version, a.payload)
            for a in wire.actions
        ],
    )


def kv_hints_to_wire(envelope: KvHintsEnvelope | None) -> dict[str, Any] | None:
    """Encode an envelope into the OpenAI request body `kv_hints` object form."""
    if envelope is None:
        return None
    return msgspec.to_builtins(envelope)
