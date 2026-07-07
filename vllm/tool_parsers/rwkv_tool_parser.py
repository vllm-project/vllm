# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re

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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


class RWKVToolParser(ToolParser):
    """Parser for RWKV XML-style tool calls.

    The RWKV chat template asks the model to emit calls as
    ``<tool_call><invoke name="fn"><parameter name="k">v</parameter>...``.
    Markers are parsed as text so the parser still works when the tokenizer
    splits them across multiple token IDs.
    """

    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)

        self.current_tool_index = 0
        self.emitted_content_len = 0
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r"<invoke\s+name\s*=\s*(['\"])(.*?)\1\s*>(.*?)</invoke>",
            re.DOTALL,
        )
        self.parameter_complete_regex = re.compile(
            r"<parameter\s+name\s*=\s*(['\"])(.*?)\1\s*>(.*?)</parameter>",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def _generate_tool_call_id(self, function_name: str, idx: int) -> str:
        return make_tool_call_id(
            id_type="random",
            func_name=function_name,
            idx=idx,
        )

    @staticmethod
    def _extract_types_from_schema(schema: Any) -> list[str]:
        if not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()
        type_value = schema.get("type")
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            types.update(t for t in type_value if isinstance(t, str))

        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            for value in enum_values:
                if value is None:
                    types.add("null")
                elif isinstance(value, bool):
                    types.add("boolean")
                elif isinstance(value, int):
                    types.add("integer")
                elif isinstance(value, float):
                    types.add("number")
                elif isinstance(value, str):
                    types.add("string")
                elif isinstance(value, list):
                    types.add("array")
                elif isinstance(value, dict):
                    types.add("object")

        for key in ("anyOf", "oneOf", "allOf"):
            choices = schema.get(key)
            if isinstance(choices, list):
                for choice in choices:
                    types.update(RWKVToolParser._extract_types_from_schema(choice))

        return sorted(types) if types else ["string"]

    @classmethod
    def _get_param_types(
        cls,
        function_name: str,
        param_name: str,
        tools: list[Any] | None,
    ) -> list[str]:
        if not tools:
            return ["string"]

        for tool in tools:
            function = getattr(tool, "function", None)
            if function is None or getattr(function, "name", None) != function_name:
                continue

            parameters = getattr(function, "parameters", None)
            if not isinstance(parameters, dict):
                return ["string"]

            properties = parameters.get("properties", {})
            if not isinstance(properties, dict):
                return ["string"]

            return cls._extract_types_from_schema(properties.get(param_name))

        return ["string"]

    @staticmethod
    def _convert_param_value(value: str, param_types: list[str]) -> Any:
        value = value.strip()
        normalized = {param_type.lower() for param_type in param_types}

        if value.lower() in {"null", "none", "nil"} and "null" in normalized:
            return None

        if normalized & {"integer", "int"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

        if normalized & {"number", "float"}:
            try:
                numeric = float(value)
                return int(numeric) if numeric.is_integer() else numeric
            except (TypeError, ValueError):
                pass

        if normalized & {"boolean", "bool"}:
            if value.lower() in {"true", "1", "yes", "on"}:
                return True
            if value.lower() in {"false", "0", "no", "off"}:
                return False

        if normalized & {"object", "array"}:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse_single_invoke(
        self,
        invoke_match: re.Match[str],
        tools: list[Any] | None,
    ) -> ToolCall | None:
        function_name = invoke_match.group(2).strip()
        invoke_body = invoke_match.group(3)
        if not function_name:
            return None

        arguments: dict[str, Any] = {}
        for parameter_match in self.parameter_complete_regex.finditer(invoke_body):
            param_name = parameter_match.group(2).strip()
            if not param_name:
                continue
            raw_value = parameter_match.group(3)
            param_types = self._get_param_types(function_name, param_name, tools)
            arguments[param_name] = self._convert_param_value(raw_value, param_types)

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _parse_tool_calls(
        self,
        text: str,
        request: ChatCompletionRequest | None,
    ) -> list[ToolCall]:
        tools = request.tools if request is not None else None
        tool_calls: list[ToolCall] = []
        for tool_call_match in self.tool_call_complete_regex.finditer(text):
            block = tool_call_match.group(1)
            for invoke_match in self.invoke_complete_regex.finditer(block):
                tool_call = self._parse_single_invoke(invoke_match, tools)
                if tool_call is not None:
                    tool_calls.append(tool_call)
        return tool_calls

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest,
    ) -> list[DeltaToolCall]:
        complete_invokes = list(self.invoke_complete_regex.finditer(current_text))
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_match = complete_invokes[self.current_tool_index]
            tool_call = self._parse_single_invoke(invoke_match, request.tools)
            if tool_call is None:
                self.current_tool_index += 1
                continue

            index = self.current_tool_index
            self.current_tool_index += 1
            self.prev_tool_call_arr.append(
                {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )
            self.streamed_args_for_tool.append(tool_call.function.arguments)
            delta_tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=self._generate_tool_call_id(tool_call.function.name, index),
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                    type="function",
                )
            )

        return delta_tool_calls

    def _partial_tool_start_suffix_len(self, text: str) -> int:
        max_len = min(len(text), len(self.tool_call_start_token) - 1)
        for suffix_len in range(max_len, 0, -1):
            if self.tool_call_start_token.startswith(text[-suffix_len:]):
                return suffix_len
        return 0

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            tool_calls = self._parse_tool_calls(model_output, request)
        except Exception:
            logger.exception("Error extracting RWKV tool calls")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        self.prev_tool_call_arr = [
            {"name": call.function.name, "arguments": call.function.arguments}
            for call in tool_calls
        ]
        first_tool_idx = model_output.find(self.tool_call_start_token)
        content = model_output[:first_tool_idx] if first_tool_idx > 0 else None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
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
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids

        start_idx = current_text.find(self.tool_call_start_token)
        if not previous_text:
            self.current_tool_index = 0
            self.emitted_content_len = 0
            self.prev_tool_call_arr.clear()
            self.streamed_args_for_tool.clear()

        if start_idx < 0:
            safe_end = len(current_text) - self._partial_tool_start_suffix_len(
                current_text
            )
            if safe_end <= self.emitted_content_len:
                return None
            content = current_text[self.emitted_content_len : safe_end]
            self.emitted_content_len = safe_end
            return DeltaMessage(content=content) if content else None

        content_before = None
        previous_start_idx = previous_text.find(self.tool_call_start_token)
        if previous_start_idx < 0:
            before = current_text[self.emitted_content_len : start_idx]
            self.emitted_content_len = start_idx
            content_before = before if before else None

        delta_tool_calls = self._extract_delta_tool_calls(current_text, request)
        if content_before or delta_tool_calls:
            return DeltaMessage(
                content=content_before,
                tool_calls=delta_tool_calls,
            )

        return None
