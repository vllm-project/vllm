# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermes 2 Pro tool calls, one JSON object per tag pair:

<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.logger import init_logger
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.utils import extract_json_member_object
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def _hermes_arg_converter(raw_args: str, partial: bool) -> str:
    # Without a converter the engine streams the whole wrapper and skips
    # schema type coercion, so streaming and non-streaming would disagree.
    span = extract_json_member_object(raw_args, "arguments")
    if span is None:
        return "" if partial else "{}"
    return span


def _validate_tool_call_bodies(tool_call_regex: re.Pattern, text: str) -> bool | None:
    # Like the legacy parser, non-streaming drops every tool call and returns
    # the text as content if any body is not a JSON object with a string name.
    found = False
    for match in tool_call_regex.finditer(text):
        found = True
        body = match.group(1) if match.group(1) is not None else match.group(2)
        try:
            call = json.loads(body)
        except json.JSONDecodeError:
            return False
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return False
    return True if found else None


@functools.cache
def hermes_config(
    name: str = "hermes",
    tool_call_start: str = TOOL_CALL_START,
    tool_call_end: str = TOOL_CALL_END,
) -> ParserEngineConfig:
    start, end = re.escape(tool_call_start), re.escape(tool_call_end)
    tool_call_regex = re.compile(f"{start}(.*?){end}|{start}(.*)", re.DOTALL)
    return ParserEngineConfig(
        name=name,
        terminals={
            "TOOL_START": tool_call_start,
            "TOOL_END": tool_call_end,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        arg_converter=_hermes_arg_converter,
        complete_tool_call_validator=functools.partial(
            _validate_tool_call_bodies, tool_call_regex
        ),
        strip_content_whitespace_with_tools=False,
    )


class HermesParser(ParserEngine):
    # Class attributes so variants with other tags can subclass (Longcat).
    CONFIG_NAME: str = "hermes"
    TOOL_CALL_START: str = TOOL_CALL_START
    TOOL_CALL_END: str = TOOL_CALL_END

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            tokenizer = tokenizer.tokenizer
        kwargs.setdefault(
            "parser_engine_config",
            hermes_config(self.CONFIG_NAME, self.TOOL_CALL_START, self.TOOL_CALL_END),
        )
        super().__init__(tokenizer, tools, **kwargs)
