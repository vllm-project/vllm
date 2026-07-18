# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool parser for the iQuest Coder V2 tokenizer template."""

import json
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser


class IquestCoderV2ToolParser(ToolParser):
    """Parse iQuest Coder V2 XML tool calls."""

    _CALL_START = "<iquestcoder_tool_call>"
    _CALL_END = "</iquestcoder_tool_call>"
    _KEY_START = "<arg_key>"
    _KEY_END = "</arg_key>"
    _VALUE_START = "<arg_value>"
    _VALUE_END = "</arg_value>"

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        if not tokenizer:
            raise ValueError("A tokenizer is required for iQuest Coder tool parsing")

        missing_tokens = [
            token
            for token in (self._CALL_START, self._CALL_END)
            if token not in self.vocab
        ]
        if missing_tokens:
            raise ValueError(
                "Tokenizer is missing required iQuest Coder tokens: "
                + ", ".join(missing_tokens)
            )

        self._stream_buffer = ""
        self._inside_tool_call = False
        self._next_tool_index = 0

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        return bool(request.tools) and request.tool_choice != "none"

    @staticmethod
    def _decode_value(raw_value: str) -> Any:
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value

    @staticmethod
    def _parameter_is_string(
        request: ChatCompletionRequest,
        tool_name: str,
        parameter_name: str,
    ) -> bool:
        for tool in request.tools or []:
            function = tool.function
            if function.name != tool_name or not function.parameters:
                continue
            properties = function.parameters.get("properties", {})
            parameter = properties.get(parameter_name, {})
            return parameter.get("type") == "string"
        return False

    def _parse_call_body(
        self,
        body: str,
        request: ChatCompletionRequest,
    ) -> tuple[str, dict[str, Any]] | None:
        first_key = body.find(self._KEY_START)
        if first_key == -1:
            tool_name = body.strip()
            return (tool_name, {}) if tool_name else None

        tool_name = body[:first_key].strip()
        if not tool_name:
            return None

        arguments: dict[str, Any] = {}
        cursor = first_key
        while cursor < len(body):
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if body[cursor:].strip() == "":
                break
            if not body.startswith(self._KEY_START, cursor):
                return None

            key_start = cursor + len(self._KEY_START)
            key_end = body.find(self._KEY_END, key_start)
            if key_end == -1:
                return None
            parameter_name = body[key_start:key_end].strip()
            if not parameter_name:
                return None

            cursor = key_end + len(self._KEY_END)
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if not body.startswith(self._VALUE_START, cursor):
                return None

            value_start = cursor + len(self._VALUE_START)
            value_end = body.find(self._VALUE_END, value_start)
            if value_end == -1:
                return None
            raw_value = body[value_start:value_end]
            if self._parameter_is_string(request, tool_name, parameter_name):
                arguments[parameter_name] = raw_value
            else:
                arguments[parameter_name] = self._decode_value(raw_value.strip())
            cursor = value_end + len(self._VALUE_END)

        return tool_name, arguments

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tool_calls: list[ToolCall] = []
        first_valid_call = -1
        cursor = 0

        while True:
            call_start = model_output.find(self._CALL_START, cursor)
            if call_start == -1:
                break
            body_start = call_start + len(self._CALL_START)
            call_end = model_output.find(self._CALL_END, body_start)
            if call_end == -1:
                break

            parsed = self._parse_call_body(model_output[body_start:call_end], request)
            if parsed is not None:
                tool_name, arguments = parsed
                tool_calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=tool_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        )
                    )
                )
                if first_valid_call == -1:
                    first_valid_call = call_start
            cursor = call_end + len(self._CALL_END)

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=model_output[:first_valid_call],
        )

    @classmethod
    def _partial_start_length(cls, text: str) -> int:
        limit = min(len(text), len(cls._CALL_START) - 1)
        for length in range(limit, 0, -1):
            if text.endswith(cls._CALL_START[:length]):
                return length
        return 0

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_text, current_text, previous_token_ids
        del current_token_ids, delta_token_ids
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        self._stream_buffer += delta_text
        content_parts: list[str] = []
        tool_call_deltas: list[DeltaToolCall] = []

        while True:
            if not self._inside_tool_call:
                call_start = self._stream_buffer.find(self._CALL_START)
                if call_start == -1:
                    keep = self._partial_start_length(self._stream_buffer)
                    emit_end = len(self._stream_buffer) - keep
                    if emit_end:
                        content_parts.append(self._stream_buffer[:emit_end])
                        self._stream_buffer = self._stream_buffer[emit_end:]
                    break

                content_parts.append(self._stream_buffer[:call_start])
                self._stream_buffer = self._stream_buffer[
                    call_start + len(self._CALL_START) :
                ]
                self._inside_tool_call = True

            call_end = self._stream_buffer.find(self._CALL_END)
            if call_end == -1:
                break

            body = self._stream_buffer[:call_end]
            self._stream_buffer = self._stream_buffer[call_end + len(self._CALL_END) :]
            self._inside_tool_call = False
            parsed = self._parse_call_body(body, request)
            if parsed is None:
                content_parts.append(self._CALL_START + body + self._CALL_END)
                continue

            tool_name, arguments = parsed
            tool_call_deltas.append(
                DeltaToolCall(
                    index=self._next_tool_index,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
            self._next_tool_index += 1

        content = "".join(content_parts)
        if not content and not tool_call_deltas:
            return None
        return DeltaMessage(
            content=content or None,
            tool_calls=tool_call_deltas,
        )


__all__ = ["IquestCoderV2ToolParser"]
