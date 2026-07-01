# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 parser: DSML tool calls with ``function_calls`` wrapper.

DeepSeek V3.2 output format::

    <｜DSML｜function_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

This is identical to DeepSeek V4 except for the outer wrapper
(``function_calls`` instead of ``tool_calls``) and the absence of
``<think>``/``</think>`` reasoning tags.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
    _dsml_arg_converter,
)
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_FUNC_START = f"<{_DSML}function_calls>"
DSML_FUNC_END = f"</{_DSML}function_calls>"


@functools.cache
def deepseek_v32_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v32",
        initial_state=ParserState.CONTENT,
        terminals={
            "TOOL_START": DSML_FUNC_START,
            "TOOL_END": DSML_FUNC_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "TOOL_START": DSML_FUNC_START,
            "TOOL_END": DSML_FUNC_END,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV32Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("chat_template_kwargs", None)
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v32_config(),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return self._unwrap_wrapper_args(result, self._tools, func_name)

    @staticmethod
    def _unwrap_wrapper_args(
        args_json: str,
        tools: list[Tool] | None,
        func_name: str | None,
    ) -> str:
        if not tools or not func_name:
            return args_json
        try:
            args = json.loads(args_json)
        except (json.JSONDecodeError, ValueError):
            return args_json
        if not isinstance(args, dict):
            return args_json
        properties = find_tool_properties(tools, func_name)
        if not properties:
            return args_json
        allowed = set(properties.keys())
        for wrapper in ("arguments", "input"):
            if set(args.keys()) != {wrapper} or wrapper in allowed:
                continue
            inner = args[wrapper]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return args_json
            if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
                return json.dumps(inner, ensure_ascii=False)
        return args_json
