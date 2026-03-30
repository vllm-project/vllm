# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
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


class Qwen3CoderToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        # Match parameters using structural boundaries instead of </parameter> tag
        # This allows </parameter> to appear in parameter content
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
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
                "Qwen3 XML Tool parser could not locate tool call start/end "
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
        # Incremental string streaming state (ported from GLM4 parser)
        self._streaming_string_value = False
        self._value_buffer = ""

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming."""
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _strip_param_delimiter_prefix(value: str) -> str:
        """Strip the single delimiter newline right after a parameter tag."""
        if value.startswith("\r\n"):
            return value[2:]
        if value.startswith("\n"):
            return value[1:]
        return value

    @staticmethod
    def _strip_param_delimiter_suffix(value: str) -> str:
        """Strip the single delimiter newline right before a close tag."""
        if value.endswith("\r\n"):
            return value[:-2]
        if value.endswith("\n"):
            return value[:-1]
        if value.endswith("\r"):
            return value[:-1]
        return value

    def _ensure_streaming_tool_state(self, tool_index: int) -> None:
        """Ensure per-tool arrays are allocated for the given index."""
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= tool_index:
            self.prev_tool_call_arr.append({"name": "", "arguments": "{}"})

    def _emit_tool_args_delta(self, fragment: str) -> DeltaMessage | None:
        """Emit a tool args fragment and keep streamed state synchronized."""
        if not fragment:
            return None

        self._ensure_streaming_tool_state(self.current_tool_index)
        self.streamed_args_for_tool[self.current_tool_index] += fragment
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_index,
                    function=DeltaFunctionCall(arguments=fragment),
                )
            ]
        )

    def _sync_current_tool_call_arguments(
        self,
        tool_text: str,
        request: ChatCompletionRequest,
    ) -> None:
        """Sync finalized function args into prev_tool_call_arr when parsable."""
        if self.current_tool_index >= len(self.prev_tool_call_arr):
            return
        if (
            self.tool_call_prefix not in tool_text
            or self.function_end_token not in tool_text
        ):
            return

        start = tool_text.find(self.tool_call_prefix) + len(self.tool_call_prefix)
        end = tool_text.find(self.function_end_token, start)
        if end == -1:
            return

        function_call_str = tool_text[start:end]
        parsed = self._parse_xml_function_call(function_call_str, request.tools)
        if parsed is None:
            return

        self.prev_tool_call_arr[self.current_tool_index] = {
            "name": parsed.function.name,
            "arguments": parsed.function.arguments,
        }

    def _is_string_type(self, param_name: str) -> bool:
        """Check if a parameter is string type based on tool schema."""
        if not self.streaming_request or not self.streaming_request.tools:
            return True  # Default to string for unknown params
        param_config = self._get_arguments_config(
            self.current_function_name or "",
            self.streaming_request.tools,
        )
        if param_name not in param_config:
            return True  # Default to string for unknown params
        param_type = (
            str(param_config[param_name].get("type", "string")).strip().lower()
            if isinstance(param_config.get(param_name), dict)
            else "string"
        )
        return param_type in [
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
        ]

    def _get_arguments_config(self, func_name: str, tools: list[Tool] | None) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (
                hasattr(config, "function") and hasattr(config.function, "name")
            ):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name,
                    func_name,
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        elif (
            isinstance(param_config[param_name], dict)
            and "anyOf" in param_config[param_name]
        ):
            # anyOf has no top-level "type"; treat as object to trigger json.loads.
            param_type = "object"
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an "
                    "integer in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean "
                    "(`true` or `false`) in tool '%s', degenerating to "
                    "false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Parsed value '%s' of parameter '%s' cannot be "
                        "parsed with json.loads in tool '%s', will try "
                        "other methods to parse it.",
                        param_value,
                        param_name,
                        func_name,
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "converted via Python `ast.literal_eval()` in tool "
                    "'%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[Tool] | None
    ) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        # If there's no ">" character, this is not a valid xml function call
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            param_value = self._strip_param_delimiter_prefix(param_value)
            param_value = self._strip_param_delimiter_suffix(param_value)

            # Strip trailing </parameter> tag if present
            # (since we use structural boundaries)
            param_value = re.sub(r"\s*</parameter>\s*$", "", param_value)

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str, request.tools)
                for function_call_str in function_calls
            ]
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()
            valid_tool_calls = [tc for tc in tool_calls if tc is not None]
            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
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
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
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

        # Update accumulated text
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
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
                # Check if we're between/after tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Also skip whitespace-only content if any tool calls
                # have been completed (prevents trailing \n after last
                # tool call from being emitted as content)
                if (
                    delta_text.strip() == ""
                    and self.tool_call_end_token in current_text
                ):
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            # No more tool calls to process yet
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # Always append — each tool call is a separate
                    # invocation even if the function name is the same
                    # (e.g. two consecutive "read" calls).
                    self._ensure_streaming_tool_state(self.current_tool_index)
                    self.prev_tool_call_arr[self.current_tool_index] = {
                        "name": self.current_function_name,
                        "arguments": "{}",
                    }

                    # Initialize streamed args tracking for this tool.
                    # The serving layer reads streamed_args_for_tool to
                    # compute remaining arguments at stream end. Without
                    # this, IndexError occurs when the serving layer
                    # accesses streamed_args_for_tool[index].
                    self.streamed_args_for_tool[self.current_tool_index] = ""

                    # Send header with function info
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

        # We've sent header, now handle function body
        if self.in_function:
            # Always send opening brace first, regardless of whether
            # parameter_prefix is in the current delta. With speculative
            # decoding, a single delta may contain both the opening brace
            # and parameter data; skipping "{" here would desync
            # json_started from what was actually streamed.
            if not self.json_started:
                self.json_started = True
                return self._emit_tool_args_delta("{")

            # -------------------------------------------------------
            # Handle incremental string value streaming
            # -------------------------------------------------------
            if self._streaming_string_value:
                # We're in the middle of streaming a string parameter
                # value incrementally. Extract the current value from
                # the accumulated tool_text.
                param_tag = f"{self.parameter_prefix}{self.current_param_name}>"
                param_tag_pos = tool_text.find(param_tag)
                if param_tag_pos == -1:
                    return None

                value_start_pos = param_tag_pos + len(param_tag)
                value_text = tool_text[value_start_pos:]
                # Strip leading newline (Qwen3 format puts \n after >)
                value_text = self._strip_param_delimiter_prefix(value_text)

                # Check if parameter value is complete. Function/tool close can
                # also terminate a parameter in malformed/fragmented streams.
                boundary_candidates: list[tuple[int, str]] = []
                for token, kind in (
                    (self.parameter_end_token, "parameter"),
                    (self.parameter_prefix, "next_param"),
                    (self.function_end_token, "function"),
                    (self.tool_call_end_token, "tool"),
                ):
                    pos = value_text.find(token)
                    if pos != -1:
                        boundary_candidates.append((pos, kind))

                if boundary_candidates:
                    val_end, boundary_kind = min(
                        boundary_candidates, key=lambda x: x[0]
                    )
                    # Parameter complete - emit remaining content and
                    # close the JSON string quote
                    remaining_value = value_text[:val_end]
                    remaining_value = self._strip_param_delimiter_suffix(
                        remaining_value
                    )
                    new_content = remaining_value[len(self._value_buffer) :]
                    self._value_buffer = ""
                    self._streaming_string_value = False
                    self.param_count += 1

                    escaped = self._json_escape_string_content(new_content)
                    frag = escaped + '"'

                    if boundary_kind in {"function", "tool"}:
                        self._sync_current_tool_call_arguments(tool_text, request)
                        self.in_function = False
                        self.json_closed = True
                        frag += "}"

                    return self._emit_tool_args_delta(frag)
                else:
                    # Parameter still streaming - emit safe content
                    # Hold back trailing \n and partial </parameter> tags.
                    # In Qwen3-Coder format, \n precedes </parameter> as a
                    # delimiter and should not be part of the value.
                    current_value = value_text

                    # Find content safe to emit (not part of closing tag)
                    # Check for partial </parameter> at end of buffer
                    safe_len = len(current_value)

                    # Hold back trailing \n (delimiter before </parameter>)
                    if current_value.endswith("\n"):
                        safe_len = len(current_value) - 1

                    # Hold back partial boundary suffixes (close tags + next
                    # parameter start token). Covers cases like "\n<",
                    # "\n</f", and "\n<param".
                    for close_token in (
                        self.parameter_end_token,
                        self.parameter_prefix,
                        self.function_end_token,
                        self.tool_call_end_token,
                    ):
                        for i in range(1, len(close_token)):
                            if current_value.endswith(close_token[:i]):
                                candidate_len = len(current_value) - i
                                # Also hold back the delimiter newline that
                                # commonly precedes closing tags.
                                if (
                                    candidate_len > 0
                                    and current_value[candidate_len - 1] == "\n"
                                ):
                                    candidate_len -= 1
                                if (
                                    candidate_len > 0
                                    and current_value[candidate_len - 1] == "\r"
                                ):
                                    candidate_len -= 1
                                safe_len = min(safe_len, candidate_len)
                                break

                    safe_value = current_value[: max(safe_len, 0)]
                    new_content = safe_value[len(self._value_buffer) :]
                    if new_content:
                        self._value_buffer = safe_value
                        escaped = self._json_escape_string_content(new_content)
                        if escaped:
                            return self._emit_tool_args_delta(escaped)
                    return None

            # -------------------------------------------------------
            # Look for parameters
            # -------------------------------------------------------
            param_starts = []
            idx = 0
            while True:
                idx = tool_text.find(self.parameter_prefix, idx)
                if idx == -1:
                    break
                param_starts.append(idx)
                idx += len(self.parameter_prefix)

            # Check if we should start a new parameter
            if not self.in_param and len(param_starts) > self.param_count:
                # Process the next parameter
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" in remaining:
                    # We have the complete parameter name
                    name_end = remaining.find(">")
                    self.current_param_name = remaining[:name_end]

                    # Check if this is a string type parameter
                    is_string = self._is_string_type(self.current_param_name)

                    # Find the parameter value start
                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    value_text = self._strip_param_delimiter_prefix(value_text)

                    if is_string:
                        # -----------------------------------------
                        # String type: use incremental streaming
                        # -----------------------------------------
                        # Emit the key and opening quote immediately
                        if self.param_count == 0:
                            key_frag = f'"{self.current_param_name}": "'
                        else:
                            key_frag = f', "{self.current_param_name}": "'

                        self._streaming_string_value = True
                        self._value_buffer = ""

                        return self._emit_tool_args_delta(key_frag)
                    else:
                        # -----------------------------------------
                        # Non-string type: wait for complete value
                        # -----------------------------------------
                        param_end_idx = value_text.find(self.parameter_end_token)
                        boundary_kind = "parameter"
                        if param_end_idx == -1:
                            # No closing tag yet, look for boundaries
                            next_param_idx = value_text.find(self.parameter_prefix)
                            func_end_idx = value_text.find(self.function_end_token)
                            tool_end_idx = value_text.find(self.tool_call_end_token)

                            boundary_candidates = []
                            if next_param_idx != -1:
                                boundary_candidates.append(
                                    (next_param_idx, "next_param")
                                )
                            if func_end_idx != -1:
                                boundary_candidates.append((func_end_idx, "function"))
                            if tool_end_idx != -1:
                                boundary_candidates.append((tool_end_idx, "tool"))

                            if boundary_candidates:
                                param_end_idx, boundary_kind = min(
                                    boundary_candidates, key=lambda x: x[0]
                                )
                            else:
                                return None

                        if param_end_idx != -1:
                            param_value = value_text[:param_end_idx]
                            param_value = self._strip_param_delimiter_suffix(
                                param_value
                            )

                            self.accumulated_params[self.current_param_name] = (
                                param_value
                            )

                            param_config = self._get_arguments_config(
                                self.current_function_name or "",
                                self.streaming_request.tools
                                if self.streaming_request
                                else None,
                            )

                            converted_value = self._convert_param_value(
                                param_value,
                                self.current_param_name,
                                param_config,
                                self.current_function_name or "",
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

                            if boundary_kind in {"function", "tool"}:
                                self._sync_current_tool_call_arguments(
                                    tool_text, request
                                )
                                self.in_function = False
                                self.json_closed = True
                                json_fragment += "}"

                            return self._emit_tool_args_delta(json_fragment)

            # Function ended and all started parameters have been flushed:
            # emit closing brace exactly once.
            if (
                not self.json_closed
                and not self._streaming_string_value
                and self.function_end_token in tool_text
                and self.param_count >= tool_text.count(self.parameter_prefix)
            ):
                self._sync_current_tool_call_arguments(tool_text, request)
                self.in_function = False
                self.json_closed = True
                return self._emit_tool_args_delta("}")

        return None
