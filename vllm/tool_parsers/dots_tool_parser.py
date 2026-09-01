# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import is_complete_json, partial_tag_overlap

logger = init_logger(__name__)


class DotsToolParser(ToolParser):
    """Parse Dots tool calls in their XML wrapper format.

    The canonical body contains one or more ``invoke`` elements::

        <dots_function_call>
        <invoke name="search">
        <parameter name="query">weather in Shanghai</parameter>
        </invoke>
        </dots_function_call>

    A JSON object with ``name`` and ``arguments`` is also accepted as a
    fallback. Multiple wrapper blocks and multiple invokes per block are
    supported.
    """

    supports_required_and_named = False

    tool_call_start_token = "<dots_function_call>"
    tool_call_end_token = "</dots_function_call>"

    _block_regex = re.compile(
        rf"{re.escape(tool_call_start_token)}\s*(.*?)\s*"
        rf"{re.escape(tool_call_end_token)}",
        re.DOTALL,
    )
    _invoke_regex = re.compile(
        r"<invoke\s+name\s*=\s*(?P<name>[^>]+)>(?P<body>.*?)</invoke>",
        re.DOTALL,
    )
    _parameter_regex = re.compile(
        r"<parameter\s+name\s*=\s*(?P<name>[^>]+)>(?P<value>.*?)</parameter>",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        self._buffer = ""

    @staticmethod
    def _extract_name(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    @staticmethod
    def _convert_param_value(value: str, param_type: Any) -> Any:
        if value.lower() == "null":
            return None

        if isinstance(param_type, list):
            param_type = next((item for item in param_type if item != "null"), "string")
        if not isinstance(param_type, str):
            param_type = str(param_type)
        param_type = param_type.lower()

        if param_type in {"string", "str", "text"}:
            return value
        if param_type in {"integer", "int"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if param_type in {"number", "float"}:
            try:
                number = float(value)
                return int(number) if number.is_integer() else number
            except (TypeError, ValueError):
                return value
        if param_type in {"boolean", "bool"}:
            return value.lower() in {"true", "1"}

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return value

    def _resolve_param_type(
        self,
        schema: Any,
        defs: dict[str, Any],
        depth: int = 0,
    ) -> Any | None:
        if not isinstance(schema, dict) or depth > 10:
            return None
        if "type" in schema:
            return schema["type"]

        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return self._resolve_param_type(
                defs.get(ref.rsplit("/", 1)[-1]), defs, depth + 1
            )

        for keyword in ("anyOf", "oneOf", "allOf"):
            alternatives = schema.get(keyword)
            if not isinstance(alternatives, list):
                continue
            for alternative in alternatives:
                if isinstance(alternative, dict) and alternative.get("type") == "null":
                    continue
                resolved = self._resolve_param_type(alternative, defs, depth + 1)
                if resolved is not None:
                    return resolved
        return None

    @staticmethod
    def _tool_schema(
        name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for tool in tools or []:
            if tool.function.name != name:
                continue
            schema = tool.function.parameters
            if not isinstance(schema, dict):
                break
            properties = schema.get("properties", {})
            defs = schema.get("$defs", {})
            return (
                properties if isinstance(properties, dict) else {},
                defs if isinstance(defs, dict) else {},
            )
        return {}, {}

    def _parse_xml_invoke(
        self,
        match: re.Match[str],
        tools: list[ChatCompletionToolsParam] | None,
    ) -> dict[str, Any]:
        name = self._extract_name(match.group("name"))
        properties, defs = self._tool_schema(name, tools)
        arguments: dict[str, Any] = {}

        for parameter in self._parameter_regex.finditer(match.group("body")):
            param_name = self._extract_name(parameter.group("name"))
            value = parameter.group("value").strip()
            param_type: Any = "string"
            if param_name in properties:
                param_type = (
                    self._resolve_param_type(properties[param_name], defs) or "string"
                )
            arguments[param_name] = self._convert_param_value(value, param_type)

        return {"name": name, "arguments": arguments}

    def _parse_block(
        self,
        content: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> list[dict[str, Any]]:
        content = content.strip()
        if content.startswith("<invoke"):
            return [
                self._parse_xml_invoke(match, tools)
                for match in self._invoke_regex.finditer(content)
            ]

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise TypeError("Dots JSON tool call must be an object")
        return [parsed]

    @staticmethod
    def _known_tool_names(
        tools: list[ChatCompletionToolsParam] | None,
    ) -> set[str]:
        return {tool.function.name for tool in tools or []}

    def _validated_call(
        self,
        parsed: dict[str, Any],
        tools: list[ChatCompletionToolsParam] | None,
    ) -> tuple[str, dict[str, Any]] | None:
        name = parsed.get("name")
        if not isinstance(name, str) or name not in self._known_tool_names(tools):
            return None
        arguments = parsed.get("arguments", parsed.get("parameters", {})) or {}
        if not isinstance(arguments, dict):
            return None
        return name, arguments

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        marker_index = model_output.find(self.tool_call_start_token)
        if marker_index == -1:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for block in self._block_regex.finditer(model_output):
            try:
                for parsed in self._parse_block(block.group(1), request.tools):
                    validated = self._validated_call(parsed, request.tools)
                    if validated is None:
                        continue
                    name, arguments = validated
                    tool_calls.append(
                        ToolCall(
                            function=FunctionCall(
                                name=name,
                                arguments=json.dumps(arguments, ensure_ascii=False),
                            )
                        )
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to parse Dots tool call: %s", exc)

        normal_text = model_output[:marker_index].strip()
        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=normal_text or None,
        )

    def _append_complete_stream_call(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_calls: list[DeltaToolCall],
    ) -> None:
        self.current_tool_id += 1
        serialized = json.dumps(arguments, ensure_ascii=False)
        self.prev_tool_call_arr.append({"name": name, "arguments": arguments})
        self.streamed_args_for_tool.append(serialized)
        tool_calls.append(
            DeltaToolCall(
                index=self.current_tool_id,
                id=make_tool_call_id(),
                type="function",
                function=DeltaFunctionCall(name=name, arguments=serialized),
            )
        )

    def _stream_complete_json_body(
        self,
        tools: list[ChatCompletionToolsParam] | None,
        tool_calls: list[DeltaToolCall],
    ) -> None:
        content = self._buffer[len(self.tool_call_start_token) :].strip()
        if not content or not is_complete_json(content):
            return

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(parsed, dict):
            return
        validated = self._validated_call(parsed, tools)
        if validated is None:
            return

        name, arguments = validated
        serialized = json.dumps(arguments, ensure_ascii=False)
        if not self.current_tool_name_sent:
            self.current_tool_id += 1
            tool_calls.append(
                DeltaToolCall(
                    index=self.current_tool_id,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=""),
                )
            )
            self.prev_tool_call_arr.append({"name": name, "arguments": arguments})
            self.streamed_args_for_tool.append("")
            self.current_tool_name_sent = True

        streamed = self.streamed_args_for_tool[self.current_tool_id]
        if serialized.startswith(streamed):
            argument_diff = serialized[len(streamed) :]
            if argument_diff:
                tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=argument_diff),
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] += argument_diff

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
        del current_text, previous_token_ids, current_token_ids, delta_token_ids
        if not previous_text:
            self._buffer = ""
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []

        self._buffer += delta_text
        normal_parts: list[str] = []
        tool_calls: list[DeltaToolCall] = []

        while self._buffer:
            marker_index = self._buffer.find(self.tool_call_start_token)
            if marker_index == -1:
                partial_len = partial_tag_overlap(
                    self._buffer, self.tool_call_start_token
                )
                if partial_len:
                    normal_parts.append(self._buffer[:-partial_len])
                    self._buffer = self._buffer[-partial_len:]
                else:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                normal_parts = [
                    part.replace(self.tool_call_end_token, "") for part in normal_parts
                ]
                break

            if marker_index > 0:
                normal_parts.append(self._buffer[:marker_index])
                self._buffer = self._buffer[marker_index:]

            end_index = self._buffer.find(
                self.tool_call_end_token, len(self.tool_call_start_token)
            )
            if end_index == -1:
                self._stream_complete_json_body(request.tools, tool_calls)
                break

            content = self._buffer[len(self.tool_call_start_token) : end_index]
            self._buffer = self._buffer[end_index + len(self.tool_call_end_token) :]
            try:
                parsed_calls = self._parse_block(content, request.tools)
                if not parsed_calls:
                    raise ValueError("Dots tool-call block contains no invoke")

                block_had_streamed_call = self.current_tool_name_sent
                valid_calls = [
                    validated
                    for parsed in parsed_calls
                    if (validated := self._validated_call(parsed, request.tools))
                    is not None
                ]
                if self.current_tool_name_sent and valid_calls:
                    name, arguments = valid_calls.pop(0)
                    serialized = json.dumps(arguments, ensure_ascii=False)
                    streamed = self.streamed_args_for_tool[self.current_tool_id]
                    if serialized.startswith(streamed):
                        remaining = serialized[len(streamed) :]
                        if remaining:
                            tool_calls.append(
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=remaining),
                                )
                            )
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": name,
                        "arguments": arguments,
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = serialized

                for name, arguments in valid_calls:
                    self._append_complete_stream_call(name, arguments, tool_calls)

                if not valid_calls and not block_had_streamed_call:
                    normal_parts.append(content.strip())
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to parse streamed Dots tool call: %s", exc)
                normal_parts.append(content.strip())

            self.current_tool_name_sent = False

        content_delta = "".join(normal_parts)
        if not content_delta and not tool_calls:
            return None
        return DeltaMessage(content=content_delta or None, tool_calls=tool_calls)

    def flush_pending_normal_text(self) -> str:
        """Return a partial opening marker as text when generation ends."""
        if not self._buffer or self.tool_call_start_token in self._buffer:
            return ""
        normal_text = self._buffer.replace(self.tool_call_end_token, "")
        self._buffer = ""
        return normal_text
