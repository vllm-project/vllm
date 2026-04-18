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
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import extract_intermediate_diff

logger = init_logger(__name__)


class MinimaxM2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"

        # Streaming state
        self.is_tool_call_started: bool = False
        self.current_tool_index: int = 0
        self._tool_call_ids: list[str] = []
        self._tool_name_sent: list[bool] = []

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

    def _extract_name(self, name_str: str) -> str:
        """Extract name from quoted string."""
        name_str = name_str.strip()
        if (name_str.startswith('"') and name_str.endswith('"')) or (
            name_str.startswith("'") and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

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

    def _reset_streaming_state(self, tool_call_started: bool = False) -> None:
        self.current_tool_index = 0
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self._tool_call_ids.clear()
        self._tool_name_sent.clear()
        self.is_tool_call_started = tool_call_started

    def _ensure_streaming_slots(self, tool_count: int) -> None:
        while len(self.streamed_args_for_tool) < tool_count:
            self.streamed_args_for_tool.append("")
        while len(self._tool_call_ids) < tool_count:
            self._tool_call_ids.append(self._generate_tool_call_id())
        while len(self._tool_name_sent) < tool_count:
            self._tool_name_sent.append(False)

    def _get_param_config(self, function_name: str) -> dict[str, Any]:
        if not self.tools:
            return {}

        for tool in self.tools:
            if (
                hasattr(tool, "function")
                and tool.function.name == function_name
                and hasattr(tool.function, "parameters")
            ):
                params = tool.function.parameters
                if isinstance(params, dict):
                    return params.get("properties", {})
                break

        return {}

    def _serialize_partial_param_value(
        self,
        value: str,
        param_types: list[str],
        *,
        is_complete: bool,
    ) -> str:
        value = value.strip()
        if is_complete:
            converted = self._convert_param_value_with_types(value, param_types)
            return json.dumps(converted, ensure_ascii=False)

        if not value:
            return ""

        normalized_types = {t.lower() for t in param_types}
        string_types = {"string", "str", "text"}

        if (
            "null" in normalized_types
            and not (normalized_types & string_types)
            and "null".startswith(value.lower())
        ):
            return value.lower()

        if {"boolean", "bool"} & normalized_types:
            lower_value = value.lower()
            if any(
                candidate.startswith(lower_value) for candidate in ("true", "false")
            ):
                return lower_value

        if {"integer", "int", "number", "float"} & normalized_types:
            return value

        if {"object", "array"} & normalized_types and value[:1] in "{[":
            return value

        # For strings, emit an open JSON string so later deltas can append
        # without rewriting already-streamed content.
        return json.dumps(value, ensure_ascii=False)[:-1]

    def _build_partial_arguments(
        self,
        invoke_body: str,
        *,
        invoke_complete: bool,
        param_config: dict[str, Any],
    ) -> str:
        args_parts: list[str] = []
        search_pos = 0

        while True:
            param_start = invoke_body.find("<parameter name=", search_pos)
            if param_start == -1:
                break

            name_start = param_start + len("<parameter name=")
            name_end = invoke_body.find(">", name_start)
            if name_end == -1:
                break

            param_name = self._extract_name(invoke_body[name_start:name_end])
            value_start = name_end + 1
            value_end = invoke_body.find("</parameter>", value_start)
            param_complete = value_end != -1
            if param_complete:
                param_value = invoke_body[value_start:value_end]
                search_pos = value_end + len("</parameter>")
            else:
                param_value = invoke_body[value_start:]
                search_pos = len(invoke_body)

            if not param_complete and not param_value.strip():
                break

            param_types = self._get_param_types_from_config(param_name, param_config)
            serialized_value = self._serialize_partial_param_value(
                param_value,
                param_types,
                is_complete=param_complete,
            )
            if not serialized_value:
                break

            args_parts.append(
                f"{json.dumps(param_name, ensure_ascii=False)}:{serialized_value}"
            )

            if not param_complete:
                break

        if not args_parts:
            return "{}" if invoke_complete else ""

        args_json = "{" + ",".join(args_parts)
        if invoke_complete:
            args_json += "}"
        return args_json

    def _get_invoke_states(self, current_text: str) -> list[dict[str, Any]]:
        tool_start = current_text.find(self.tool_call_start_token)
        if tool_start == -1:
            return []

        tool_payload = current_text[tool_start + len(self.tool_call_start_token) :]
        tool_end = tool_payload.find(self.tool_call_end_token)
        if tool_end != -1:
            tool_payload = tool_payload[:tool_end]

        invoke_states: list[dict[str, Any]] = []
        search_pos = 0
        while True:
            invoke_start = tool_payload.find("<invoke name=", search_pos)
            if invoke_start == -1:
                break

            invoke_content_start = invoke_start + len("<invoke name=")
            invoke_end = tool_payload.find("</invoke>", invoke_content_start)
            invoke_complete = invoke_end != -1

            if invoke_complete:
                invoke_str = tool_payload[invoke_content_start:invoke_end]
                search_pos = invoke_end + len("</invoke>")
            else:
                invoke_str = tool_payload[invoke_content_start:]
                search_pos = len(tool_payload)

            name_end = invoke_str.find(">")
            if name_end == -1:
                break

            function_name = self._extract_name(invoke_str[:name_end])
            param_config = self._get_param_config(function_name)
            invoke_body = invoke_str[name_end + 1 :]
            partial_args = self._build_partial_arguments(
                invoke_body,
                invoke_complete=invoke_complete,
                param_config=param_config,
            )

            tool_call = (
                self._parse_single_invoke(invoke_str, self.tools)
                if invoke_complete
                else None
            )
            invoke_states.append(
                {
                    "name": function_name,
                    "arguments": partial_args,
                    "complete": invoke_complete,
                    "tool_call": tool_call,
                }
            )

            if not invoke_complete:
                break

        return invoke_states

    def _finalize_completed_tool_call(
        self,
        idx: int,
        invoke_state: dict[str, Any],
    ) -> None:
        if not invoke_state["complete"] or len(self.prev_tool_call_arr) > idx:
            return

        tool_call = invoke_state["tool_call"]
        if tool_call is None:
            return

        self.prev_tool_call_arr.append(
            {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
        )

    def _extract_delta_tool_call(
        self,
        current_text: str,
    ) -> DeltaToolCall | None:
        invoke_states = self._get_invoke_states(current_text)
        if not invoke_states:
            return None

        self._ensure_streaming_slots(len(invoke_states))

        for idx, invoke_state in enumerate(invoke_states):
            args_json = invoke_state["arguments"]
            sent_args = self.streamed_args_for_tool[idx]
            name_sent = self._tool_name_sent[idx]

            if not name_sent:
                self._tool_name_sent[idx] = True
                self.current_tool_index = idx
                if args_json:
                    self.streamed_args_for_tool[idx] = args_json
                self._finalize_completed_tool_call(idx, invoke_state)
                return DeltaToolCall(
                    index=idx,
                    id=self._tool_call_ids[idx],
                    type="function",
                    function=DeltaFunctionCall(
                        name=invoke_state["name"],
                        arguments=args_json or None,
                    ),
                )

            if args_json and args_json != sent_args:
                if sent_args and args_json.startswith(sent_args):
                    args_delta = args_json[len(sent_args) :]
                else:
                    args_delta = extract_intermediate_diff(args_json, sent_args)

                if args_delta:
                    self.streamed_args_for_tool[idx] = args_json
                    self.current_tool_index = idx
                    self._finalize_completed_tool_call(idx, invoke_state)
                    return DeltaToolCall(
                        index=idx,
                        function=DeltaFunctionCall(arguments=args_delta),
                    )

            self._finalize_completed_tool_call(idx, invoke_state)

        return None

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
                    tool_call = self._parse_single_invoke(invoke_match, self.tools)
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
        """Extract tool calls from streaming model output.

        Uses a buffer-until-complete-invoke strategy: tokens are buffered
        until a complete ``<invoke>...</invoke>`` block is available, then
        parsed and emitted in one shot.
        """

        start_in_text = self.tool_call_start_token in delta_text
        start_in_ids = self.tool_call_start_token_id in delta_token_ids
        tool_call_starting = start_in_text or start_in_ids
        # Reset state on new request (parser is reused) or new tool-call block.
        if not previous_text or tool_call_starting:
            self._reset_streaming_state(tool_call_started=tool_call_starting)

        # Pass through content before any tool call.
        if not self.is_tool_call_started:
            return DeltaMessage(content=delta_text) if delta_text else None

        # Capture content before the start token.
        content_before = None
        if start_in_text:
            before = delta_text[: delta_text.index(self.tool_call_start_token)]
            content_before = before or None

        # Emit the next pending name/arguments fragment for the current invoke.
        delta_tool_call = self._extract_delta_tool_call(current_text)

        if delta_tool_call or content_before:
            return DeltaMessage(
                content=content_before,
                tool_calls=[delta_tool_call] if delta_tool_call else None,
            )

        # EOS and </minimax:tool_call> both arrive as special tokens with
        # no decoded text. Return non-None for EOS so the serving framework
        # reaches the finish-reason handling path instead of skipping.
        if (
            not delta_text
            and delta_token_ids
            and self.prev_tool_call_arr
            and self.tool_call_end_token_id not in delta_token_ids
        ):
            return DeltaMessage(content="")

        return None
