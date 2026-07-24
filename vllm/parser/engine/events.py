# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Semantic event types emitted by the streaming parser engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    TEXT_CHUNK = auto()
    REASONING_START = auto()
    REASONING_CHUNK = auto()
    REASONING_END = auto()
    TOOL_CALL_START = auto()
    TOOL_NAME = auto()
    ARG_VALUE_CHUNK = auto()
    TOOL_CALL_END = auto()


@dataclass(slots=True)
class SemanticEvent:
    type: EventType
    value: str = ""
    tool_index: int = -1
    truncated: bool = False
