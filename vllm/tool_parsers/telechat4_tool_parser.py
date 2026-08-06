# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)

TOOL_START_TOKEN = "<tool_call>"
TOOL_END_TOKEN = "</tool_call>"
PARAM_KEY_START_TOKEN = "<param_key>"

TOOL_CALL_REGEX = re.compile(
    rf"{re.escape(TOOL_START_TOKEN)}(.*?){re.escape(TOOL_END_TOKEN)}",
    re.DOTALL,
)
PARAM_REGEX = re.compile(
    r"<param_key>(.*?)</param_key>\s*<param_value>(.*?)</param_value>",
    re.DOTALL,
)


def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _tool_function(tool: ChatCompletionToolsParam | Mapping[str, Any]) -> Any:
    return _get_attr_or_item(tool, "function")


def _tool_name(tool: ChatCompletionToolsParam | Mapping[str, Any]) -> str | None:
    function = _tool_function(tool)
    name = _get_attr_or_item(function, "name")
    return str(name) if name else None


def _tool_parameters(
    tool: ChatCompletionToolsParam | Mapping[str, Any]) -> Mapping[str, Any]:
    function = _tool_function(tool)
    return _as_mapping(_get_attr_or_item(function, "parameters"))


def _iter_tool_names(
    tools: Sequence[ChatCompletionToolsParam | Mapping[str, Any]] | None,
) -> list[str]:
    if tools is None:
        return []
    names = [_tool_name(tool) for tool in tools]
    return sorted((name for name in names if name), key=len, reverse=True)


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: Sequence[ChatCompletionToolsParam | Mapping[str, Any]] | None,
) -> bool:
    if tools is None:
        return False

    for tool in tools:
        if _tool_name(tool) != tool_name:
            continue

        parameters = _tool_parameters(tool)
        properties = _as_mapping(parameters.get("properties"))
        arg_schema = _as_mapping(properties.get(arg_name))
        arg_type = arg_schema.get("type")
        if isinstance(arg_type, str):
            return arg_type == "string"
        if isinstance(arg_type, Sequence) and not isinstance(arg_type, str):
            return "string" in arg_type
        return False

    logger.debug("No tool named '%s'.", tool_name)
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        pass

    return value


def _json_arguments(value: str) -> dict[str, Any]:
    parsed = _deserialize(value)
    if not isinstance(parsed, Mapping):
        return {}

    arguments = parsed.get("arguments", parsed.get("parameters", parsed))
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return {}


def _split_payload(
    payload: str,
    tools: Sequence[ChatCompletionToolsParam | Mapping[str, Any]] | None,
) -> tuple[str, str, str]:
    payload = payload.strip()
    param_pos = payload.find(PARAM_KEY_START_TOKEN)
    if param_pos != -1:
        return payload[:param_pos].strip(), payload[param_pos:], ""

    for tool_name in _iter_tool_names(tools):
        if payload == tool_name:
            return tool_name, "", ""
        if payload.startswith(tool_name):
            rest = payload[len(tool_name):].strip()
            if rest.startswith("{"):
                return tool_name, "", rest

    return payload, "", ""


def _parse_payload(
    payload: str,
    tools: Sequence[ChatCompletionToolsParam | Mapping[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    tool_name, params_text, json_text = _split_payload(payload, tools)
    arguments = _json_arguments(json_text) if json_text else {}

    for key, value in PARAM_REGEX.findall(params_text):
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(tool_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arguments[arg_key] = arg_val

    return tool_name, arguments


def _partial_suffix_len(text: str, token: str) -> int:
    max_len = min(len(text), len(token) - 1)
    for size in range(max_len, 0, -1):
        if token.startswith(text[-size:]):
            return size
    return 0


class TeleChat4ToolParser(ToolParser):
    """Tool call parser for TeleChat4 models.

    Used when ``--enable-auto-tool-choice --tool-call-parser telechat4``
    is specified.

    Supports two tool-call formats:
      1. JSON: ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``
      2. Tag-based:
         ``<tool_call>func<param_key>k1</param_key><param_value>v1
         </param_value>...</tool_call>``
    """

    def __init__(self, tokenizer: TokenizerLike,
                 tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_id: int = -1
        self.tool_start_token = TOOL_START_TOKEN
        self.tool_end_token = TOOL_END_TOKEN
        self._buffer = ""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        logger.debug("model_output: %s", model_output)

        tool_calls: list[ToolCall] = []
        try:
            for match in TOOL_CALL_REGEX.finditer(model_output):
                tool_name, arguments = _parse_payload(
                    match.group(1), request.tools)
                if not tool_name:
                    continue

                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_name,
                            arguments=json.dumps(
                                arguments, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if tool_calls:
            content = model_output[:model_output.find(self.tool_start_token)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    def _make_delta(
        self,
        content: str,
        tool_calls: list[DeltaToolCall],
    ) -> DeltaMessage | None:
        if not content and not tool_calls:
            return None
        return DeltaMessage(
            content=content or None,
            tool_calls=tool_calls,
        )

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
        self._buffer += delta_text
        content = ""
        delta_tool_calls: list[DeltaToolCall] = []

        while True:
            start_idx = self._buffer.find(self.tool_start_token)
            if start_idx == -1:
                partial_len = _partial_suffix_len(
                    self._buffer, self.tool_start_token)
                if partial_len:
                    content += self._buffer[:-partial_len]
                    self._buffer = self._buffer[-partial_len:]
                else:
                    content += self._buffer
                    self._buffer = ""
                return self._make_delta(content, delta_tool_calls)

            content += self._buffer[:start_idx]
            self._buffer = self._buffer[start_idx:]
            end_idx = self._buffer.find(self.tool_end_token)
            if end_idx == -1:
                return self._make_delta(content, delta_tool_calls)

            end_pos = end_idx + len(self.tool_end_token)
            tool_text = self._buffer[:end_pos]
            extracted_tool_calls = self.extract_tool_calls(tool_text, request)

            if not extracted_tool_calls.tool_calls:
                logger.warning(
                    "Failed to extract any tool calls from %r.", tool_text)
                content += tool_text
            else:
                for tool_call in extracted_tool_calls.tool_calls:
                    self.current_tool_id += 1
                    delta_tool_calls.append(
                        DeltaToolCall(
                            index=self.current_tool_id,
                            id=make_tool_call_id(),
                            type=tool_call.type,
                            function=DeltaFunctionCall(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                    )

            self._buffer = self._buffer[end_pos:]
