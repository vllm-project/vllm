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
    _PARAM_RE,
    _PARTIAL_PARAM_RE,
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
    DSML_PARAM_START,
    _dsml_arg_converter,
    _has_pending_tag,
    _is_open_json_string,
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
            "PARAM_START": DSML_PARAM_START,
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
        return _unwrap_wrapper_args(result, self._tools, func_name)

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        slot = self._tool_slots[idx]

        # Fast path: if currently streaming an open, schema-stable string parameter
        # and raw_delta contains no tag delimiters ('<'), directly append the
        # JSON-escaped chunk without full re-conversion.
        if (
            slot.active_string_param is not None
            and slot.in_open_string
            and "<" not in raw_delta
        ):
            escaped_delta = json.dumps(raw_delta, ensure_ascii=False)[1:-1]
            slot.streamed_json += escaped_delta
            return escaped_delta

        diff = super()._compute_arg_delta(idx, raw_delta)

        # Update active_string_param and in_open_string state
        last_end = 0
        for m in _PARAM_RE.finditer(slot.args):
            last_end = m.end()
        pm = _PARTIAL_PARAM_RE.search(slot.args, last_end)
        if (
            pm
            and pm.group(2) == "true"
            and not _has_pending_tag(slot.args[pm.start(3) :])
        ):
            slot.active_string_param = pm.group(1)
            is_schema_stable = (
                slot.string_keys is None or slot.active_string_param in slot.string_keys
            )
            slot.in_open_string = is_schema_stable and _is_open_json_string(
                slot.streamed_json
            )
        else:
            slot.active_string_param = None
            slot.in_open_string = False

        return diff

    def _flush_arg_converter(self, idx: int) -> str | None:
        if idx < len(self._tool_slots):
            self._tool_slots[idx].active_string_param = None
            self._tool_slots[idx].in_open_string = False
        return super()._flush_arg_converter(idx)
