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

logger = init_logger(__name__)


class MinimaxM2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"

        # Streaming state for incremental parsing
        self.is_tool_call_started: bool = False
        self.current_tool_index: int = 0
        self.current_tool_name_sent: bool = False
        self._buffer: str = ""
        self._current_tool_name: str | None = None
        self._current_tool_id: int = -1
        self._args_started: list[bool] = [False]
        self._args_closed: list[bool] = [False]
        self._pending_key: str | None = None
        self._streaming_string_value: bool = False
        self._tool_call_ids: list[str] = []

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

        # Incremental parsing patterns
        self.invoke_name_regex = re.compile(
            r'<invoke name=(["\'])(.*?)\1', re.DOTALL
        )
        self.parameter_name_regex = re.compile(
            r'<parameter name=(["\'])(.*?)\1', re.DOTALL
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

    def _ensure_tool_state(self) -> None:
        while len(self.streamed_args_for_tool) <= self._current_tool_id:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= self._current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self._args_started) <= self._current_tool_id:
            self._args_started.append(False)
        while len(self._args_closed) <= self._current_tool_id:
            self._args_closed.append(False)

    def _begin_tool_call(self) -> None:
        if self._current_tool_id == -1:
            self._current_tool_id = 0
        else:
            self._current_tool_id += 1
        self._ensure_tool_state()
        self._current_tool_name = None
        self.current_tool_name_sent = False
        self._args_started[self._current_tool_id] = False
        self._args_closed[self._current_tool_id] = False
        self._pending_key = None
        self._streaming_string_value = False
        self.streamed_args_for_tool[self._current_tool_id] = ""

    def _finish_tool_call(self) -> None:
        self.current_tool_index += 1
        self._current_tool_id += 1
        self._current_tool_name = None
        self.current_tool_name_sent = False
        self._args_started = [False]
        self._args_closed = [False]
        self._pending_key = None
        self._streaming_string_value = False
        self._buffer = ""

    def _json_escape_string_content(self, s: str) -> str:
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    def _is_string_type(self, tool_name: str, arg_name: str) -> bool:
        if self.tools is None:
            return False
        for tool in self.tools:
            if tool.function.name != tool_name:
                continue
            if tool.function.parameters is None:
                return False
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type", None)
            )
            return arg_type == "string"
        return False

    def _emit_tool_name_delta(self, name: str) -> DeltaMessage:
        tool_call_id = self._generate_tool_call_id()
        while len(self._tool_call_ids) <= self._current_tool_id:
            self._tool_call_ids.append(tool_call_id)
        return DeltaMessage(
            content=None,
            tool_calls=[
                DeltaToolCall(
                    index=self._current_tool_id,
                    id=tool_call_id,
                    function=DeltaFunctionCall(
                        name=name,
                        arguments="",
                    ),
                    type="function",
                )
            ],
        )

    def _emit_tool_args_delta(self, args_fragment: str) -> DeltaMessage:
        return DeltaMessage(
            content=None,
            tool_calls=[
                DeltaToolCall(
                    index=self._current_tool_id,
                    id=self._tool_call_ids[self._current_tool_id],
                    function=DeltaFunctionCall(
                        name=None,
                        arguments=args_fragment,
                    ),
                    type="function",
                )
            ],
        )

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

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest | None,
    ) -> list[DeltaToolCall]:
        """Extract DeltaToolCalls from newly completed <invoke> blocks.

        Tracks progress via ``current_tool_index`` so each block is
        extracted exactly once across successive streaming calls.
        """
        complete_invokes = self.invoke_complete_regex.findall(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_str = complete_invokes[self.current_tool_index]
            tool_call = self._parse_single_invoke(
                invoke_str,
                self.tools,
            )
            if not tool_call:
                self.current_tool_index += 1
                continue

            args_json = tool_call.function.arguments
            idx = self.current_tool_index
            self.current_tool_index += 1

            self.prev_tool_call_arr.append(
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(args_json),
                }
            )
            self.streamed_args_for_tool.append(args_json)
            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=self._generate_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=args_json,
                    ),
                    type="function",
                )
            )

        return delta_tool_calls

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
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output.

        Uses incremental parsing to emit tool names and arguments as soon
        as they are parsed, rather than waiting for complete <invoke> blocks.
        """
        self._buffer += delta_text

        if not self.is_tool_call_started:
            start_idx = self._buffer.find(self.tool_call_start_token)
            if start_idx == -1:
                for i in range(1, len(self.tool_call_start_token)):
                    if self._buffer.endswith(self.tool_call_start_token[:i]):
                        out = self._buffer[: -i]
                        self._buffer = self._buffer[-i:]
                        return DeltaMessage(content=out) if out else None
                out = self._buffer
                self._buffer = ""
                return DeltaMessage(content=out) if out else None

            if start_idx > 0:
                out = self._buffer[:start_idx]
                self._buffer = self._buffer[start_idx:]
                return DeltaMessage(content=out) if out else None

            self._buffer = self._buffer[len(self.tool_call_start_token) :]
            self.is_tool_call_started = True
            self._begin_tool_call()

        while True:
            if not self.current_tool_name_sent:
                invoke_name_match = self.invoke_name_regex.search(self._buffer)
                if invoke_name_match:
                    self._current_tool_name = self._extract_name(
                        invoke_name_match.group(2)
                    )
                    self._buffer = self._buffer[invoke_name_match.end() :]
                    self.current_tool_name_sent = True
                    self._begin_tool_call()
                    return self._emit_tool_name_delta(self._current_tool_name)

                end_pos = self._buffer.find(self.tool_call_end_token)
                if end_pos == -1:
                    return None

                if self._buffer.strip():
                    self._buffer = self._buffer[end_pos + len(self.tool_call_end_token) :]
                    self._finish_tool_call()
                    continue
                return None

            if self._streaming_string_value:
                val_end = self._buffer.find("</parameter>")
                if val_end != -1:
                    raw_content = self._buffer[:val_end]
                    self._buffer = self._buffer[val_end + len("</parameter>") :]
                    self._streaming_string_value = False

                    escaped = self._json_escape_string_content(raw_content)
                    frag = escaped + '"'
                    self.streamed_args_for_tool[self._current_tool_id] += frag
                    self._pending_key = None
                    return self._emit_tool_args_delta(frag)
                else:
                    safe_len = len(self._buffer)
                    for i in range(1, len("</parameter>")):
                        if self._buffer.endswith("</parameter>"[:i]):
                            safe_len = len(self._buffer) - i
                            break
                    if safe_len > 0:
                        to_emit = self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                        escaped = self._json_escape_string_content(to_emit)
                        if escaped:
                            self.streamed_args_for_tool[self._current_tool_id] += escaped
                            return self._emit_tool_args_delta(escaped)
                    return None

            if self._pending_key is not None:
                val_pos = self._buffer.find(">")
                if val_pos == -1:
                    return None

                self._buffer = self._buffer[val_pos + 1 :]

                key = self._pending_key.strip()
                is_string = self._is_string_type(
                    self._current_tool_name or "", key
                )

                if is_string:
                    if not self._args_started[self._current_tool_id]:
                        key_json = json.dumps(key, ensure_ascii=False)
                        frag = "{" + key_json + ': "'
                        self._args_started[self._current_tool_id] = True
                    else:
                        key_json = json.dumps(key, ensure_ascii=False)
                        frag = ", " + key_json + ': "'
                    self.streamed_args_for_tool[self._current_tool_id] += frag
                    self._streaming_string_value = True
                    return self._emit_tool_args_delta(frag)
                else:
                    val_end = self._buffer.find("</parameter>")
                    if val_end == -1:
                        return None
                    raw_val = self._buffer[:val_end].strip()
                    self._buffer = self._buffer[val_end + len("</parameter>") :]
                    self._pending_key = None

                    if not self._args_started[self._current_tool_id]:
                        key_json = json.dumps(key, ensure_ascii=False)
                        frag = "{" + key_json + ": " + raw_val
                        self._args_started[self._current_tool_id] = True
                    else:
                        key_json = json.dumps(key, ensure_ascii=False)
                        frag = ", " + key_json + ": " + raw_val
                    self.streamed_args_for_tool[self._current_tool_id] += frag
                    return self._emit_tool_args_delta(frag)

            end_pos = self._buffer.find(self.tool_call_end_token)
            param_start = self._buffer.find("<parameter name=")
            if end_pos != -1 and (param_start == -1 or end_pos < param_start):
                self._buffer = self._buffer[end_pos + len(self.tool_call_end_token) :]
                frag_or_none = None
                if self._args_started[self._current_tool_id] and not self._args_closed[
                    self._current_tool_id
                ]:
                    self.streamed_args_for_tool[
                        self._current_tool_id
                    ] += "}"
                    self._args_closed[self._current_tool_id] = True
                    frag_or_none = "}"
                if self._current_tool_name and self.streamed_args_for_tool[
                    self._current_tool_id
                ]:
                    try:
                        args_dict = json.loads(
                            self.streamed_args_for_tool[self._current_tool_id]
                        )
                        self.prev_tool_call_arr.append({
                            "name": self._current_tool_name,
                            "arguments": args_dict,
                        })
                    except json.JSONDecodeError:
                        pass
                self._finish_tool_call()
                return (
                    self._emit_tool_args_delta(frag_or_none) if frag_or_none else None
                )

            if param_start == -1:
                return None
            if param_start > 0:
                self._buffer = self._buffer[param_start:]

            param_match = self.parameter_name_regex.search(self._buffer)
            if not param_match:
                return None

            param_name = self._extract_name(param_match.group(2))
            self._buffer = self._buffer[param_match.end() :]
            self._pending_key = param_name
            continue
