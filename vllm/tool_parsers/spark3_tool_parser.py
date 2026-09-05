# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import regex as re
from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import (
    find_tool_name,
    find_tool_properties,
    partial_tag_overlap,
)

logger = init_logger(__name__)

TOOL_CALL_BEGIN = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_BEGIN = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_BEGIN = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

ARG_PAIR_PATTERN = re.compile(
    rf"{re.escape(ARG_KEY_BEGIN)}(.*?){re.escape(ARG_KEY_END)}"
    rf"{re.escape(ARG_VALUE_BEGIN)}(.*?){re.escape(ARG_VALUE_END)}",
    re.DOTALL,
)


@dataclass(frozen=True)
class _Spark3ToolCall:
    name: str
    arguments: dict[str, Any]

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":"))


def _get_param_type(
    tools: list[Tool] | None,
    function_name: str,
    param_name: str,
) -> str:
    """Return the declared JSON Schema type for one tool parameter."""
    properties = find_tool_properties(tools, function_name)
    definition = properties.get(param_name)
    if isinstance(definition, dict):
        param_type = definition.get("type")
        if isinstance(param_type, str):
            return param_type
    return "string"


def _convert_value(value: str, param_type: str) -> Any:
    """Convert Spark3 XML text into a Python value."""
    if value.lower() == "null":
        return None

    normalized_type = param_type.lower()
    try:
        if normalized_type in {"string", "str", "text"}:
            return value
        if normalized_type in {"integer", "int"}:
            return int(value)
        if normalized_type in {"number", "float"}:
            number = float(value)
            return int(number) if number.is_integer() else number
        if normalized_type in {"boolean", "bool"}:
            normalized_value = value.strip().lower()
            if normalized_value not in {"true", "1", "false", "0"}:
                raise ValueError(f"invalid boolean: {value}")
            return normalized_value in {"true", "1"}
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value


def _parse_tool_call_xml(
    tool_xml: str,
    tools: list[Tool] | None,
) -> _Spark3ToolCall | None:
    if not tool_xml.startswith(TOOL_CALL_BEGIN) or not tool_xml.endswith(
        TOOL_CALL_END
    ):
        return None

    body = tool_xml[len(TOOL_CALL_BEGIN) : -len(TOOL_CALL_END)]
    first_arg = body.find(ARG_KEY_BEGIN)
    function_name = (body if first_arg < 0 else body[:first_arg]).strip()
    if not function_name:
        return None
    if tools and not find_tool_name(tools, function_name):
        return None

    arguments: dict[str, Any] = {}
    for match in ARG_PAIR_PATTERN.finditer(body):
        key, raw_value = match.group(1), match.group(2)
        if not key:
            continue
        arguments[key] = _convert_value(
            raw_value,
            _get_param_type(tools, function_name, key),
        )
    return _Spark3ToolCall(name=function_name, arguments=arguments)


def _has_unknown_tool_name(tool_xml: str, tools: list[Tool] | None) -> bool:
    if not tools:
        return False
    if not tool_xml.startswith(TOOL_CALL_BEGIN) or not tool_xml.endswith(
        TOOL_CALL_END
    ):
        return False
    body = tool_xml[len(TOOL_CALL_BEGIN) : -len(TOOL_CALL_END)]
    first_arg = body.find(ARG_KEY_BEGIN)
    function_name = (body if first_arg < 0 else body[:first_arg]).strip()
    return bool(function_name) and not find_tool_name(tools, function_name)


class Spark3ToolParser(ToolParser):
    """Tool parser for Spark3's XML-KV function-call format."""

    supports_required_and_named = False
    engine_based_streaming = True

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.tool_call_start_token = TOOL_CALL_BEGIN
        self.tool_call_end_token = TOOL_CALL_END
        self._buffer = ""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tool_choice = request.tool_choice
            if tool_choice == "required" or isinstance(
                tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            ):
                request.skip_special_tokens = False
                return request

        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _append_tool_call(
        self,
        parsed: _Spark3ToolCall,
        tool_index: int,
        calls: list[DeltaToolCall] | list[ToolCall],
        *,
        is_streaming: bool,
    ) -> None:
        arguments_json = parsed.arguments_json()
        if is_streaming:
            self.prev_tool_call_arr.append(
                {"name": parsed.name, "arguments": parsed.arguments}
            )
            self.streamed_args_for_tool.append(arguments_json)
            calls.append(
                DeltaToolCall(
                    index=tool_index,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=parsed.name,
                        arguments=arguments_json,
                    ),
                )
            )
            return

        calls.append(
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name=parsed.name,
                    arguments=arguments_json,
                ),
            )
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        del request

        calls: list[ToolCall] = []
        normal_parts: list[str] = []
        cursor = 0

        while cursor < len(model_output):
            start = model_output.find(TOOL_CALL_BEGIN, cursor)
            if start < 0:
                normal_parts.append(model_output[cursor:])
                break

            normal_parts.append(model_output[cursor:start])
            end = model_output.find(TOOL_CALL_END, start + len(TOOL_CALL_BEGIN))
            if end < 0:
                normal_parts.append(model_output[start:])
                break

            end += len(TOOL_CALL_END)
            raw_tool_call = model_output[start:end]
            parsed = _parse_tool_call_xml(raw_tool_call, self.tools)
            if parsed is None:
                if not _has_unknown_tool_name(raw_tool_call, self.tools):
                    normal_parts.append(raw_tool_call)
            else:
                self._append_tool_call(
                    parsed,
                    len(calls),
                    calls,
                    is_streaming=False,
                )
            cursor = end

        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content="".join(normal_parts) if normal_parts else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        del (
            previous_text,
            current_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

        self._buffer += delta_text
        normal_parts: list[str] = []
        tool_calls: list[DeltaToolCall] = []

        while self._buffer:
            start = self._buffer.find(TOOL_CALL_BEGIN)
            if start < 0:
                keep = partial_tag_overlap(self._buffer, TOOL_CALL_BEGIN)
                if keep:
                    normal_parts.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                else:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                break

            if start > 0:
                normal_parts.append(self._buffer[:start])
                self._buffer = self._buffer[start:]

            end = self._buffer.find(TOOL_CALL_END, len(TOOL_CALL_BEGIN))
            if end < 0:
                break

            end += len(TOOL_CALL_END)
            raw_tool_call = self._buffer[:end]
            self._buffer = self._buffer[end:]
            parsed = _parse_tool_call_xml(raw_tool_call, self.tools)
            if parsed is None:
                if not _has_unknown_tool_name(raw_tool_call, self.tools):
                    normal_parts.append(raw_tool_call)
                continue

            self.current_tool_id += 1
            self._append_tool_call(
                parsed,
                self.current_tool_id,
                tool_calls,
                is_streaming=True,
            )

        content = "".join(normal_parts)
        if not content and not tool_calls:
            return None
        return DeltaMessage(content=content or None, tool_calls=tool_calls)

    def finish_streaming(self) -> DeltaMessage | None:
        pending = self._buffer
        self._buffer = ""
        if TOOL_CALL_BEGIN in pending:
            pending = pending[: pending.find(TOOL_CALL_BEGIN)]
        if not pending:
            return None
        return DeltaMessage(content=pending)
