# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import msgspec


class KvHintAction(msgspec.Struct, frozen=True):
    """A versioned orchestrator-provided KV hint action."""

    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, Any]


class KvHintsEnvelope(msgspec.Struct, frozen=True):
    """Orchestrator-provided KV hints optionally attached to one inference request.
    A KV hint envelope is a collection of KV hint action(s)."""

    protocol_version: str
    message_id: str
    actions: list[KvHintAction]
