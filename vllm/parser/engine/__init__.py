# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Streaming parser engine framework for tool call and reasoning extraction.

Instead of hand-rolling a parser for every model's tool-call / reasoning
format, each format is declared as a ParserEngineConfig (terminals,
states, and transitions) and a shared incremental engine handles
streaming, ambiguity buffering, token-ID mapping, and delta computation.
"""

from vllm.parser.engine.events import EventType, SemanticEvent

__all__ = [
    "EventType",
    "SemanticEvent",
]
