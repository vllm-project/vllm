# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite parser for JSON-array tool calls.

Granite 3.0/3.1 emit tool calls as a JSON array following a marker::

    <|tool_call|> [{"name": "get_weather", "arguments": {"city": "SF"}}]   # 3.0
    <tool_call> [{"name": "get_weather", "arguments": {"city": "SF"}}]      # 3.1

``<|tool_call|>`` is a single special token; ``<tool_call>`` is plain text.
The array may hold several calls, has no closing marker (the ``]`` ends it),
and any surrounding prose is content. The engine's ``tool_call_body_array``
mode splits the array into one call per element and extracts each name via the
name-from-args path; ``_granite_arg_converter`` carves the ``arguments`` object
out of each ``{"name":..,"arguments":{..}}`` element. Granite has no reasoning.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.utils import extract_json_member_object

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

TOOL_TOKEN = "<|tool_call|>"  # granite 3.0 (single special token)
TOOL_STRING = "<tool_call>"  # granite 3.1 (plain text)


def _granite_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the ``arguments`` object out of a ``{"name":..,"arguments":{..}}``
    array element (see :func:`extract_json_member_object`)."""
    span = extract_json_member_object(raw_args, "arguments")
    if span is None:
        return "" if partial else "{}"
    return span


@functools.cache
def granite_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="granite",
        initial_state=ParserState.CONTENT,
        terminals={
            "TOOL_TOKEN": TOOL_TOKEN,
            "TOOL_STRING": TOOL_STRING,
        },
        token_id_terminals={
            "TOOL_TOKEN": TOOL_TOKEN,
        },
        transitions={
            # The marker only switches into the tool region; the array splitter
            # emits TOOL_CALL_START per element (empty events here).
            (ParserState.CONTENT, "TOOL_TOKEN"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.CONTENT, "TOOL_STRING"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
        },
        arg_converter=_granite_arg_converter,
        tool_call_body_array=True,
        stream_arg_deltas=True,
    )


class GraniteParser(ParserEngine):
    CONFIG_NAME = "granite"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", granite_config())
        super().__init__(tokenizer, tools, **kwargs)
