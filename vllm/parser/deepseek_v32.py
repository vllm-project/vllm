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
from typing import TYPE_CHECKING

from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
    _dsml_arg_converter,
    _unwrap_wrapper_args,
)
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.parser.engine.parser_engine import ToolCallSlot
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

    def _convert_slot_args(self, slot: ToolCallSlot, partial: bool) -> str:
        result = _dsml_arg_converter(slot.args, partial)
        if not self._tools:
            return result
        return _unwrap_wrapper_args(result, self._tools, slot.name)
