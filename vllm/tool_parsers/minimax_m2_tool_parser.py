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
from vllm.tool_parsers.utils import build_partial_args_json

logger = init_logger(__name__)


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
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False

        # Initialize streaming state variables
        self.current_tool_index: int = 0
        self.header_sent: bool = False
        self.current_function_name: str | None = None
        self.in_function: bool = False
        self.json_closed: bool = False
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
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.in_function = False
        self.json_closed = False
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

    def _compute_current_args_json(
        self, tool_text: str, request: ChatCompletionRequest | None
    ) -> str:
        """Build the incremental JSON arguments string from available XML content.

        Extracts all complete ``<parameter>`` pairs from *tool_text*,
        type-converts them using the request schema, and delegates JSON
        construction to :func:`build_partial_args_json`.
        """
        is_complete = self.invoke_end_token in tool_text

        # Resolve parameter type information from the request schema.
        param_config: dict = {}
        if request and request.tools and self.current_function_name:
            for tool in request.tools:
                if (
                    hasattr(tool, "function")
                    and tool.function.name == self.current_function_name
                    and hasattr(tool.function, "parameters")
                ):
                    schema = tool.function.parameters
                    if isinstance(schema, dict) and "properties" in schema:
                        param_config = schema["properties"]
                    break

        # Collect all *complete* parameter pairs in document order.
        param_pairs: list[tuple[str, Any]] = []
        for match in self.parameter_complete_regex.findall(tool_text):
            m = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
            if m:
                name = self._extract_name(m.group(1))
                raw = m.group(2).strip()
                types = self._get_param_types_from_config(name, param_config)
                param_pairs.append(
                    (name, self._convert_param_value_with_types(raw, types))
                )

        return build_partial_args_json(param_pairs, is_complete)

    def _finalize_tool_call(self, complete_args_json: str) -> None:
        """Mark the current tool call as complete and update prev_tool_call_arr."""
        if self.current_tool_index < len(self.prev_tool_call_arr):
            try:
                self.prev_tool_call_arr[self.current_tool_index]["arguments"] = (
                    json.loads(complete_args_json)
                )
            except json.JSONDecodeError:
                logger.warning(
                    "[M2_STREAMING] Failed to parse complete args JSON: %s",
                    complete_args_json,
                )
        self.json_closed = True
        self.in_function = False

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

        # If no delta text, check whether there is still work to do.
        if not delta_text:
            # If we're mid-tool-call, fall through to the main processing
            # logic — current_text may contain content (parameters, closing
            # tags) from prior deltas that hasn't been fully diffed yet.
            # This is critical for stream_interval > 1 where the EOS token
            # arrives before all argument fragments have been emitted.
            if self.is_tool_call_started and not self.json_closed:
                pass  # fall through
            elif delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # EOS after tool calls are complete
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
            else:
                return None

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            invoke_ends = current_text.count(self.invoke_end_token)
            if invoke_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.json_closed = False
                self.in_function = False

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

        # Check if we're between tool calls (waiting for next one)
        invoke_starts_count = current_text.count(self.invoke_start_prefix)
        if self.current_tool_index >= invoke_starts_count:
            # We're past all tool calls, shouldn't be here
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
            # No more tool calls to process yet
            return None

        invoke_start_idx = invoke_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
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
                # Find the end of the function name
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    function_name_raw = tool_text[func_start:func_end]
                    self.current_function_name = self._extract_name(function_name_raw)
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    if len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append(
                            {
                                "name": self.current_function_name,
                                "arguments": {},  # updated when complete
                            }
                        )
                    if len(self.streamed_args_for_tool) <= self.current_tool_index:
                        self.streamed_args_for_tool.append("")

                    # Compute args already visible in current_text right now
                    initial_args = self._compute_current_args_json(tool_text, request)
                    self.streamed_args_for_tool[self.current_tool_index] = initial_args

                    if self.invoke_end_token in tool_text and initial_args.endswith(
                        "}"
                    ):
                        self._finalize_tool_call(initial_args)

                    # Send header + any args visible so far as initial delta
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name,
                                    arguments=initial_args,
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # Header already sent. Compute the full args JSON from current_text in
        # one pass and emit only the new portion as a delta.
        current_args = self._compute_current_args_json(tool_text, request)
        already_sent = (
            self.streamed_args_for_tool[self.current_tool_index]
            if self.current_tool_index < len(self.streamed_args_for_tool)
            else ""
        )
        args_delta = current_args[len(already_sent) :]

        if not args_delta:
            return None

        if self.current_tool_index < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[self.current_tool_index] = current_args

        if self.invoke_end_token in tool_text and current_args.endswith("}"):
            self._finalize_tool_call(current_args)
            logger.debug("[M2_STREAMING] Tool call completed")

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_index,
                    function=DeltaFunctionCall(arguments=args_delta),
                )
            ]
        )
