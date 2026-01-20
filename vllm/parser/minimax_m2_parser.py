# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MiniMax M2 Parser - A unified parser for MiniMax M2 models.

This parser combines reasoning extraction (content before </think>) and
tool call extraction (content in <minimax:tool_call> tags) into a single
unified interface.
"""

import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
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
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser, ParserManager
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


@ParserManager.register_module("minimax_m2")
class MiniMaxM2Parser(Parser):
    """
    Unified parser for MiniMax M2 models that handles both reasoning
    extraction and tool call parsing.

    MiniMax M2 models have two special behaviors:
    1. Reasoning: They don't generate <think> start token, only </think> end
       token. All content before </think> is reasoning, content after is the
       actual response.
    2. Tool Calls: They use <minimax:tool_call>...</minimax:tool_call> tags
       with <invoke name="...">...</invoke> and <parameter name="...">...</parameter>
       syntax.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # ========== Reasoning tokens ==========
        self.think_start_token: str = "<think>"
        self.think_end_token: str = "</think>"

        # ========== Tool call tokens ==========
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.invoke_start_prefix: str = "<invoke name="
        self.invoke_end_token: str = "</invoke>"
        self.parameter_prefix: str = "<parameter name="
        self.parameter_end_token: str = "</parameter>"

        # ========== Get token IDs ==========
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the Parser "
                "constructor during construction."
            )

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.think_end_token_id is None:
            raise RuntimeError(
                "MiniMax M2 parser could not locate </think> token in the tokenizer!"
            )

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "MiniMax M2 parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        # ========== Tool parsing state ==========
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: str | None = None
        self.streamed_args_for_tool: list[str] = []

        # Streaming state variables
        self.current_tool_index: int = 0
        self.invoke_index: int = 0
        self.header_sent: bool = False
        self.current_function_name: str | None = None
        self.current_param_name: str | None = None
        self.current_param_value: str = ""
        self.param_count: int = 0
        self.in_param: bool = False
        self.in_function: bool = False
        self.accumulated_text: str = ""
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}
        self.streaming_request: ChatCompletionRequest | None = None
        self.is_tool_call_started: bool = False

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r"<invoke name=(.*?)</invoke>", re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r"<parameter name=(.*?)</parameter>", re.DOTALL
        )

        logger.debug(
            "vLLM Successfully initialized parser %s!", self.__class__.__name__
        )

    # ========== Reasoning Parser Methods ==========

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Check if reasoning has ended by looking for </think> token."""
        end_token_id = self.think_end_token_id
        return any(input_id == end_token_id for input_id in reversed(input_ids))

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """Check if reasoning ends in the current delta."""
        return self.think_end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract content token IDs (everything after </think>)."""
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.think_end_token_id) + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from complete model output.

        MiniMax M2 doesn't generate <think> start token, so all content
        before </think> is reasoning.
        """
        # Remove <think> if present (for consistency)
        model_output_parts = model_output.partition(self.think_start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.think_end_token not in model_output:
            return model_output, None
        else:
            reasoning, _, content = model_output.partition(self.think_end_token)
            return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from streaming delta.

        MiniMax M2 doesn't generate <think> start token, so we assume
        all content is reasoning until we encounter </think>.
        """
        # Skip single end token
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None

        # Check if end token has already appeared in previous tokens
        if self.think_end_token_id in previous_token_ids:
            # We're past the reasoning phase, this is content
            return DeltaMessage(content=delta_text)

        # Check if end token is in delta tokens
        if self.think_end_token_id in delta_token_ids:
            # End token in delta, split reasoning and content
            end_index = delta_text.find(self.think_end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # No end token yet, all content is reasoning
        return DeltaMessage(reasoning=delta_text)

    # ========== Tool Parser Methods ==========

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.invoke_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self.streaming_request = None
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _extract_name(self, name_str: str) -> str:
        """Extract name from quoted string."""
        name_str = name_str.strip()
        if (
            name_str.startswith('"')
            and name_str.endswith('"')
            or name_str.startswith("'")
            and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """
        Extract all possible types from a JSON schema definition.
        Handles anyOf, oneOf, allOf, type arrays, and enum fields.
        """
        if schema is None:
            return ["string"]

        if not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        # Handle direct "type" field
        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

        # Handle enum - infer types from enum values
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            for value in schema["enum"]:
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

        # Handle anyOf, oneOf, allOf
        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    extracted = self._extract_types_from_schema(choice)
                    types.update(extracted)

        if not types:
            return ["string"]

        return list(types)

    def _convert_param_value_with_types(
        self, value: str, param_types: list[str]
    ) -> Any:
        """Convert parameter value to the correct type."""
        # Check if the VALUE itself indicates null
        if value.lower() in ("null", "none", "nil"):
            return None

        normalized_types = [t.lower() for t in param_types]

        # Priority: integer > number > boolean > object > array > string
        type_priority = [
            "integer",
            "int",
            "number",
            "float",
            "boolean",
            "bool",
            "object",
            "array",
            "string",
            "str",
            "text",
        ]

        for param_type in type_priority:
            if param_type not in normalized_types:
                continue

            if param_type in ["string", "str", "text"]:
                return value
            elif param_type in ["integer", "int"]:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["number", "float"]:
                try:
                    val = float(value)
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["boolean", "bool"]:
                lower_val = value.lower().strip()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                elif lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            elif param_type in ["object", "array"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        # Fallback: try JSON parse, then return as string
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_param_types_from_config(
        self, param_name: str, param_config: dict
    ) -> list[str]:
        """Get parameter types from parameter configuration."""
        if param_name not in param_config:
            return ["string"]

        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]

        return self._extract_types_from_schema(param_schema)

    def _parse_single_invoke(
        self, invoke_str: str, tools: list | None
    ) -> ToolCall | None:
        """Parse a single <invoke> block."""
        # Extract function name
        name_match = re.search(r"^([^>]+)", invoke_str)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))

        # Get parameter configuration
        param_config = {}
        if tools:
            for tool in tools:
                if (
                    hasattr(tool, "function")
                    and tool.function.name == function_name
                    and hasattr(tool.function, "parameters")
                ):
                    params = tool.function.parameters
                    if isinstance(params, dict) and "properties" in params:
                        param_config = params["properties"]
                    break

        # Extract parameters
        param_dict = {}
        for match in self.parameter_complete_regex.findall(invoke_str):
            param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
            if param_match:
                param_name = self._extract_name(param_match.group(1))
                param_value = param_match.group(2).strip()
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                param_type = self._get_param_types_from_config(param_name, param_config)
                param_dict[param_name] = self._convert_param_value_with_types(
                    param_value, param_type
                )

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(param_dict, ensure_ascii=False),
            ),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output."""
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []

            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                for invoke_match in self.invoke_complete_regex.findall(tool_call_match):
                    tool_call = self._parse_single_invoke(
                        invoke_match, request.tools if request else None
                    )
                    if tool_call:
                        tool_calls.append(tool_call)

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                self.prev_tool_call_arr.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        """Extract tool calls from streaming model output."""

        # Store request for type conversion
        if not previous_text or self.tool_call_start_token in delta_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    return DeltaMessage(content="")
            return None

        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            invoke_ends = current_text.count(self.invoke_end_token)
            if invoke_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.in_function = False
                self.accumulated_params = {}
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                self.is_tool_call_started = True
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    return None
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls
        invoke_starts_count = current_text.count(self.invoke_start_prefix)
        if self.current_tool_index >= invoke_starts_count:
            return None

        # Find the current tool call portion
        invoke_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.invoke_start_prefix, idx)
            if idx == -1:
                break
            invoke_start_positions.append(idx)
            idx += len(self.invoke_start_prefix)

        if self.current_tool_index >= len(invoke_start_positions):
            return None

        invoke_start_idx = invoke_start_positions[self.current_tool_index]
        invoke_end_idx = current_text.find(self.invoke_end_token, invoke_start_idx)
        if invoke_end_idx == -1:
            tool_text = current_text[invoke_start_idx:]
        else:
            tool_text = current_text[
                invoke_start_idx : invoke_end_idx + len(self.invoke_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.invoke_start_prefix in tool_text:
                func_start = tool_text.find(self.invoke_start_prefix) + len(
                    self.invoke_start_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    function_name_raw = tool_text[func_start:func_end]
                    self.current_function_name = self._extract_name(function_name_raw)
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    if len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append(
                            {
                                "name": self.current_function_name,
                                "arguments": {},
                            }
                        )
                        if len(self.streamed_args_for_tool) <= self.current_tool_index:
                            self.streamed_args_for_tool.append("")

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name, arguments=""
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # Handle function body
        if self.in_function:
            if self.in_function and not self.json_started:
                self.json_started = True
                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += "{"
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            if not self.json_started:
                self.json_started = True

            # Check for function end
            if not self.json_closed and self.invoke_end_token in tool_text:
                total_param_count = tool_text.count(self.parameter_prefix)

                if self.param_count >= total_param_count:
                    self.json_closed = True

                    invoke_start = tool_text.find(self.invoke_start_prefix) + len(
                        self.invoke_start_prefix
                    )
                    invoke_content_end = tool_text.find(
                        self.invoke_end_token, invoke_start
                    )
                    if invoke_content_end != -1:
                        invoke_content = tool_text[invoke_start:invoke_content_end]
                        try:
                            parsed_tool = self._parse_single_invoke(
                                invoke_content,
                                self.streaming_request.tools
                                if self.streaming_request
                                else None,
                            )
                            if parsed_tool and self.current_tool_index < len(
                                self.prev_tool_call_arr
                            ):
                                args = parsed_tool.function.arguments
                                self.prev_tool_call_arr[self.current_tool_index][
                                    "arguments"
                                ] = json.loads(args)
                        except Exception:
                            pass

                    result = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments="}"),
                            )
                        ]
                    )
                    if self.current_tool_index < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[self.current_tool_index] += "}"

                    self.json_closed = True
                    self.in_function = False
                    self.accumulated_params = {}

                    logger.debug("[M2_STREAMING] Tool call completed")
                    return result
                else:
                    return None

            # Look for parameters
            param_starts = []
            idx = 0
            while True:
                idx = tool_text.find(self.parameter_prefix, idx)
                if idx == -1:
                    break
                param_starts.append(idx)
                idx += len(self.parameter_prefix)

            if (
                not self.in_param
                and self.param_count < len(param_starts)
                and len(param_starts) > self.param_count
            ):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" in remaining:
                    name_end = remaining.find(">")
                    param_name_raw = remaining[:name_end]
                    self.current_param_name = self._extract_name(param_name_raw)

                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    param_end_idx = value_text.find(self.parameter_end_token)
                    if param_end_idx == -1:
                        next_param_idx = value_text.find(self.parameter_prefix)
                        func_end_idx = value_text.find(self.invoke_end_token)

                        if next_param_idx != -1 and (
                            func_end_idx == -1 or next_param_idx < func_end_idx
                        ):
                            param_end_idx = next_param_idx
                        elif func_end_idx != -1:
                            param_end_idx = func_end_idx
                        else:
                            if self.invoke_end_token in tool_text:
                                param_end_idx = len(value_text)
                            else:
                                return None

                    if param_end_idx != -1:
                        param_value = value_text[:param_end_idx]
                        if param_value.endswith("\n"):
                            param_value = param_value[:-1]

                        self.accumulated_params[self.current_param_name] = param_value

                        param_config = {}
                        if self.streaming_request and self.streaming_request.tools:
                            for tool in self.streaming_request.tools:
                                if (
                                    hasattr(tool, "function")
                                    and tool.function.name == self.current_function_name
                                    and hasattr(tool.function, "parameters")
                                ):
                                    params = tool.function.parameters
                                    if (
                                        isinstance(params, dict)
                                        and "properties" in params
                                    ):
                                        param_config = params["properties"]
                                    break

                        param_type = self._get_param_types_from_config(
                            self.current_param_name, param_config
                        )

                        converted_value = self._convert_param_value_with_types(
                            param_value, param_type
                        )

                        serialized_value = json.dumps(
                            converted_value, ensure_ascii=False
                        )

                        if self.param_count == 0:
                            json_fragment = (
                                f'"{self.current_param_name}": {serialized_value}'
                            )
                        else:
                            json_fragment = (
                                f', "{self.current_param_name}": {serialized_value}'
                            )

                        self.param_count += 1
                        if self.current_tool_index < len(self.streamed_args_for_tool):
                            self.streamed_args_for_tool[self.current_tool_index] += (
                                json_fragment
                            )
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    function=DeltaFunctionCall(arguments=json_fragment),
                                )
                            ]
                        )

        return None
