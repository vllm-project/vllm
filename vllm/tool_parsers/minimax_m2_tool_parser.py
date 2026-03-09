# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class PartialInvoke:
    """Represents a partially parsed invoke that can be incrementally built."""

    def __init__(
        self,
        name: str | None = None,
        parameters: list[tuple[str, str]] | None = None,
        invoke_completed: bool = False,
    ):
        self.name = name
        self.parameters = parameters if parameters is not None else []
        self.invoke_completed = invoke_completed

    def __repr__(self):
        return (
            f"PartialInvoke(name={self.name!r}, parameters={self.parameters}"
            f", invoke_completed={self.invoke_completed})"
        )


class MinimaxM2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.invoke_start_prefix: str = "<invoke name="
        self.invoke_end_token: str = "</invoke>"
        self.parameter_prefix: str = "<parameter name="
        self.parameter_end_token: str = "</parameter>"

        # Streaming state variables
        self.current_tool_name_sent: bool = False
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Initialize streaming state variables
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

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

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

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "MiniMax M2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

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
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None
        # Clear previous tool call history to avoid state pollution
        self.prev_tool_call_arr.clear()
        # Reset streamed args tracking
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

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type (legacy single-type version)."""
        return self._convert_param_value_with_types(value, [param_type])

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """
        Extract all possible types from a JSON schema definition.
        Handles anyOf, oneOf, allOf, type arrays, and enum fields.

        Args:
            schema: The JSON schema definition for a parameter

        Returns:
            List of type strings (e.g., ["string", "integer", "null"])
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

        # Handle anyOf, oneOf, allOf - recursively extract types
        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    extracted = self._extract_types_from_schema(choice)
                    types.update(extracted)

        # If no types found, default to string
        if not types:
            return ["string"]

        return list(types)

    def _convert_param_value_with_types(
        self, value: str, param_types: list[str]
    ) -> Any:
        """
        Convert parameter value to the correct type based on a list of possible types.
        Tries each type in order until one succeeds.

        Args:
            value: The string value to convert
            param_types: List of possible type strings

        Returns:
            The converted value
        """
        # Check if the VALUE itself indicates null (not just if null is allowed)
        if value.lower() in ("null", "none", "nil"):
            return None

        # Normalize types
        normalized_types = [t.lower() for t in param_types]

        # Try each type in order of preference (most specific first, string as fallback)
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
        """
        Get parameter types from parameter configuration.
        Handles anyOf, oneOf, allOf, and direct type definitions.

        Args:
            param_name: The name of the parameter
            param_config: The properties dict from the tool schema

        Returns:
            List of type strings
        """
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

                # Get parameter types (supports anyOf/oneOf/allOf)
                param_type = self._get_param_types_from_config(param_name, param_config)

                # Convert value
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

    def _parse_partial_invoke(
        self, invoke_str: str, tools: list | None = None
    ) -> PartialInvoke:
        """
        Parse a potentially incomplete <invoke> block.

        Only includes:
        - name: if the function name is completely parsed
        - parameters: only those with both opening and closing tags
        - invoke_completed: True if </invoke> closing tag is present

        Returns a PartialInvoke with ordered parameters as list of tuples.
        """
        result = PartialInvoke()

        # Extract function name - handle both single and double quotes
        # <invoke name="X"> or <invoke name='X'>
        name_pattern = rf'{re.escape(self.invoke_start_prefix)}["\']([^"\']+)["\']>'
        name_match = re.search(name_pattern, invoke_str)
        if name_match:
            result.name = name_match.group(1).strip()

        # Check if invoke block is complete (has closing tag)
        result.invoke_completed = self.invoke_end_token in invoke_str

        # Get parameter configuration for type conversion
        param_config = {}
        if tools and result.name:
            for tool in tools:
                if (
                    hasattr(tool, "function")
                    and tool.function.name == result.name
                    and hasattr(tool.function, "parameters")
                ):
                    params = tool.function.parameters
                    if isinstance(params, dict) and "properties" in params:
                        param_config = params["properties"]
                    break

        # Extract only COMPLETE parameters (both opening and closing tags)
        # Handle both single and double quotes
        # <parameter name="X">...</parameter> or <parameter name='X'>...</parameter>
        complete_param_pattern = (
            rf'{re.escape(self.parameter_prefix)}["\']([^"\']+)["\']>'
            rf"(.*?){re.escape(self.parameter_end_token)}"
        )

        for match in re.finditer(complete_param_pattern, invoke_str, re.DOTALL):
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()

            # Clean up leading/trailing newlines
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            # Get parameter types and convert
            if param_config:
                param_type = self._get_param_types_from_config(param_name, param_config)
                converted_value = self._convert_param_value_with_types(
                    param_value, param_type
                )
            else:
                # Without config, keep as string
                converted_value = param_value

            # Append as tuple to maintain order
            result.parameters.append((param_name, converted_value))

        return result

    def _find_all_indices(self, string, substring):
        indices = []
        start = 0
        while (index := string.find(substring, start)) != -1:
            indices.append(index)
            start = index + 1
        return indices

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        # Quick check
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []

            # Find all complete tool_call blocks
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                # Find all invokes within this tool_call
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

            # Update prev_tool_call_arr
            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                self.prev_tool_call_arr.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

            # Extract content before first tool call
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

    # Note: If tool_call_start_token appears in delta_text, it must be the last token
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
        current_token_ids: Sequence[int],  # pylint: disable=unused-argument
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
            # Check if this is an EOS token after all tool calls are complete
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                # Check if we're between tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        invoke_start_indices = self._find_all_indices(
            current_text, self.invoke_start_prefix
        )
        invoke_stop_indices = self._find_all_indices(
            current_text, self.invoke_end_token
        )
        invokes = []
        for i in range(self.current_tool_index, len(invoke_start_indices)):
            invoke_start_idx = invoke_start_indices[i]
            if i < len(invoke_stop_indices):
                invoke_stop_idx = invoke_stop_indices[i] + len(self.invoke_end_token)
                invoke_content = current_text[invoke_start_idx:invoke_stop_idx]
            else:
                invoke_content = current_text[invoke_start_idx:]

            parsed_invoke = self._parse_partial_invoke(
                invoke_content,
                self.streaming_request.tools if self.streaming_request else None,
            )
            # Overwrite existing entry in prev_tool_call_arr
            # this doesn't need to be incrementally built
            if i < len(self.prev_tool_call_arr):
                self.prev_tool_call_arr[i] = {
                    "name": parsed_invoke.name,
                    "arguments": dict(parsed_invoke.parameters),
                }
            else:
                self.prev_tool_call_arr.append(
                    {
                        "name": parsed_invoke.name,
                        "arguments": dict(parsed_invoke.parameters),
                    }
                )
            invokes.append(parsed_invoke)

        delta_tool_calls = []
        for i in range(len(invokes)):
            invoke = invokes[i]
            arguments_json = ""

            if not self.header_sent:
                if invoke.name is None:
                    # Still waiting for function name
                    break

                self.header_sent = True
                self.current_tool_id = self._generate_tool_call_id()

                for j in range(len(invoke.parameters)):
                    param_json = json.dumps(
                        dict([invoke.parameters[j]]), ensure_ascii=False
                    )
                    param_json = param_json[1:-1]  # Strip {}
                    json_fragment = "{" + param_json if j == 0 else f", {param_json}"
                    arguments_json += json_fragment
                    self.param_count += 1
                if invoke.invoke_completed:
                    if len(invoke.parameters) == 0:
                        arguments_json += "{"
                    arguments_json += "}"

                delta_tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(
                            name=invoke.name,
                            arguments=arguments_json,
                        ),
                        type="function",
                    )
                )
            else:
                for j in range(self.param_count, len(invoke.parameters)):
                    param_json = json.dumps(
                        dict([invoke.parameters[j]]), ensure_ascii=False
                    )
                    param_json = param_json[1:-1]  # Strip {}
                    json_fragment = "{" + param_json if j == 0 else f", {param_json}"
                    arguments_json += json_fragment
                    self.param_count += 1
                if invoke.invoke_completed:
                    if len(invoke.parameters) == 0:
                        arguments_json += "{"
                    arguments_json += "}"

                delta_tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_index,
                        function=DeltaFunctionCall(arguments=arguments_json),
                    )
                )

            if self.current_tool_index < len(self.streamed_args_for_tool):
                self.streamed_args_for_tool[self.current_tool_index] += arguments_json
            else:
                self.streamed_args_for_tool.append(arguments_json)

            if invoke.invoke_completed:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0

        if delta_tool_calls:
            return DeltaMessage(tool_calls=delta_tool_calls)
        else:
            return None
