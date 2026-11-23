# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermes 2 Pro tool-call parser, backed by the declarative parser engine.

Tool calls repeat as back-to-back pairs:

    <tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>

The body is a JSON wrapper, so the name comes from the engine's
name-from-args path and `_hermes_arg_converter` carves out the
`arguments` value -- the same shape as Inkling, with the key spelled
`arguments`.
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
    """Carve the `arguments` object out of the tool-call JSON wrapper.

    The engine treats the whole `TOOL_ARGS` span as the arguments, and
    skips `_fix_arg_types` when there is no converter, so omitting one
    makes streaming and non-streaming disagree on argument types.
    """
    span = extract_json_member_object(raw_args, "arguments")
    if span is None:
        return "" if partial else "{}"
    return span


def _validate_tool_call_bodies(tool_call_regex: re.Pattern, text: str) -> bool | None:
    """Reject the whole response when any tool-call body is not valid JSON.

    Matches the legacy parser: non-streaming is all-or-nothing, while
    streaming keeps the engine's best-effort behaviour. Returns `None`
    when there is no tool call to judge.
    """
    found = False
    for match in tool_call_regex.finditer(text):
        found = True
        body = match.group(1) if match.group(1) is not None else match.group(2)
        try:
            json.loads(body)
        except (json.JSONDecodeError, ValueError):
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
        initial_state=ParserState.CONTENT,
        terminals={
            "TOOL_START": tool_call_start,
            "TOOL_END": tool_call_end,
        },
        # Text-matched only, like the legacy parser: `ParserEngine.adjust_request`
        # keeps special tokens in the detokenized text, so the tags always
        # reach the lexer as text.
        token_id_terminals={},
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
        stream_arg_deltas=True,
        tool_args_json=True,
        strip_trailing_reasoning_whitespace=True,
        drop_whitespace_only_content_before_tools=True,
        strip_content_whitespace_with_tools=False,
        validate_tool_names=False,
    )


class HermesParser(ParserEngine):
    """Hermes 2 Pro parser backed by the declarative parser engine.

    Delimiters are class attributes so subclasses can reuse this grammar
    with different tags; see `LongcatParser`.
    """

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
