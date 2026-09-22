# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M2 parser for XML-style tool calls.

MiniMax M2 tool call format::

    <minimax:tool_call><invoke name="get_weather">
    <parameter name="city">Seattle</parameter>
    </invoke></minimax:tool_call>

Each ``<invoke>`` block becomes one tool call. The argument body consists
of ``<parameter name="...">...</parameter>`` tags.
"""

from __future__ import annotations

import functools
import json

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

TOOL_CALL_START = "<minimax:tool_call>"
TOOL_CALL_END = "</minimax:tool_call>"
THINK_START = "<think>"
THINK_END = "</think>"
INVOKE_PREFIX_DQ = '<invoke name="'
INVOKE_PREFIX_SQ = "<invoke name='"
INVOKE_PREFIX_UNQUOTED = "<invoke name="
INVOKE_END = "</invoke>"
NAME_END_DQ = '">'
NAME_END_SQ = "'>"
NAME_END_UNQUOTED = ">"
PARAM_START = "<parameter name="
PARAM_END = "</parameter>"
_PARAM_RE = re.compile(
    r"<\s*parameter\s+name\s*=\s*"
    r"(?:\"(?P<dq_name>[^\"]*)\"|'(?P<sq_name>[^']*)'|(?P<bare_name>[^>\s]+))"
    r"\s*>"
    r"(?P<value>.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s+name\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    r"<\s*parameter\s+name\s*=\s*"
    r"(?:\"(?P<dq_name>[^\"]*)\"|'(?P<sq_name>[^']*)'|(?P<bare_name>[^>\s]+))"
    r"\s*>"
    r"(?P<value>.*)$",
    re.DOTALL,
)


def _minimax_m2_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = (
            match.group("dq_name")
            or match.group("sq_name")
            or match.group("bare_name")
            or ""
        ).strip()
        if not name:
            continue
        # Keep the value verbatim (like the glm47 converter): it is inline
        # between `>` and `</parameter>`, so surrounding whitespace is data.
        params[name] = match.group("value")

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        match = _PARTIAL_PARAM_RE.search(remaining)
        if match:
            name = (
                match.group("dq_name")
                or match.group("sq_name")
                or match.group("bare_name")
                or ""
            ).strip()
            if name:
                # Verbatim, same as the complete-match loop above.
                params[name] = match.group("value")

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def minimax_m2_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="minimax_m2",
        initial_state=ParserState.REASONING,
        terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
            "TOOL_START": TOOL_CALL_START,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "TOOL_END": TOOL_CALL_END,
            "INVOKE_PREFIX_DQ": INVOKE_PREFIX_DQ,
            "INVOKE_PREFIX_SQ": INVOKE_PREFIX_SQ,
            "INVOKE_PREFIX_UNQUOTED": INVOKE_PREFIX_UNQUOTED,
            "INVOKE_END": INVOKE_END,
            "NAME_END_DQ": NAME_END_DQ,
            "NAME_END_SQ": NAME_END_SQ,
            "NAME_END_UNQUOTED": NAME_END_UNQUOTED,
        },
        token_id_terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.CONTENT, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            **{
                (state, terminal): Transition(
                    ParserState.TOOL_NAME,
                    (EventType.TOOL_CALL_START,),
                )
                for state in (
                    ParserState.CONTENT,
                    ParserState.TOOL_PREAMBLE,
                    ParserState.TOOL_BETWEEN,
                )
                for terminal in (
                    "INVOKE_PREFIX_DQ",
                    "INVOKE_PREFIX_SQ",
                    "INVOKE_PREFIX_UNQUOTED",
                )
            },
            **{
                (ParserState.TOOL_NAME, terminal): Transition(
                    ParserState.TOOL_ARGS,
                    (),
                )
                for terminal in (
                    "NAME_END_DQ",
                    "NAME_END_SQ",
                    "NAME_END_UNQUOTED",
                )
            },
        },
        arg_converter=_minimax_m2_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


class MinimaxM2Parser(ParserEngine):
    """MiniMax M2 parser backed by the declarative parser engine."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        kwargs.setdefault("parser_engine_config", minimax_m2_config())
        super().__init__(tokenizer, tools, **kwargs)
        self._think_end_token_id = self.vocab.get(THINK_END)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_id = self._think_end_token_id
        if end_id is None:
            return []
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == end_id:
                return input_ids[i + 1 :]
        return []
