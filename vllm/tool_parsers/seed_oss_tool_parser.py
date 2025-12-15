# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from qwen3coder xml parser, All rights reserved.
# ruff: noqa: E501

import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class SeedOssToolParser(ToolParser):
    TOOL_CALL_START = "<seed:tool_call>"
    TOOL_CALL_END = "</seed:tool_call>"

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # --- streaming state ---
        self._reset_streaming_state()
        self.prev_tool_call_arr: list[dict] = []

        self.tool_call_start_token: str = self.TOOL_CALL_START
        self.tool_call_end_token: str = self.TOOL_CALL_END
        # Sentinel tokens for streaming mode
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.think_start_token: str = "<seed:think>"
        self.think_end_token: str = "</seed:think>"
        self.is_tool_call_started: bool = False
        self.is_thinking_end: bool = False
        self.failed_count: int = 0
        self._reset_streaming_state()

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Seed_Oss XML parser: tokenizer did not include "
                "<seed:tool_call> or its closing tag."
            )

        tool_start_re = re.escape(self.tool_call_start_token)
        tool_end_re = re.escape(self.tool_call_end_token)

        self.tool_call_complete_regex = re.compile(
            rf"{tool_start_re}(.*?){tool_end_re}", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            rf"{tool_start_re}(.*?){tool_end_re}|{tool_start_re}(.*?)$", re.DOTALL
        )

        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

        logger.info(
            "vLLM Seed-Oss XML tool parser loaded (%s).", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = -1
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ToolCall | None:
        def get_arguments_config(func_name: str) -> dict:
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
            logger.warning("Tool '%s' is not defined in the tools list.", func_name)
            return {}

        def convert_param_value(
            param_value: str, param_name: str, param_config: dict, func_name: str
        ) -> Any:
            # Handle null value for any type
            if param_value.lower() == "null":
                return None

            if param_name not in param_config:
                if param_config != {}:
                    logger.warning(
                        "Parsed parameter '%s' is not defined in "
                        "the tool parameters for tool '%s', "
                        "directly returning the string value.",
                        param_name,
                        func_name,
                    )
                return param_value

            if (
                isinstance(param_config[param_name], dict)
                and "type" in param_config[param_name]
            ):
                param_type = str(param_config[param_name]["type"]).strip().lower()
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
                    param_value = int(param_value)  # type: ignore
                except (ValueError, TypeError):
                    logger.warning(
                        "Parsed value '%s' of parameter '%s' is not an integer in tool "
                        "'%s', degenerating to string.",
                        param_value,
                        param_name,
                        func_name,
                    )
                return param_value
            elif param_type.startswith("num") or param_type.startswith("float"):
                try:
                    float_param_value = float(param_value)
                    param_value = (
                        float_param_value  # type: ignore
                        if float_param_value - int(float_param_value) != 0
                        else int(float_param_value)  # type: ignore
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Parsed value '%s' of parameter '%s' is not a float in tool "
                        "'%s', degenerating to string.",
                        param_value,
                        param_name,
                        func_name,
                    )
                return param_value
            elif param_type in ["boolean", "bool", "binary"]:
                param_value = param_value.lower()
                if param_value not in ["true", "false"]:
                    logger.warning(
                        "Parsed value '%s' of parameter '%s' is not a boolean "
                        "(`true` of `false`) in tool '%s', degenerating to false.",
                        param_value,
                        param_name,
                        func_name,
                    )
                return param_value == "true"
            else:
                if param_type == "object" or param_type.startswith("dict"):
                    try:
                        param_value = json.loads(param_value)
                        return param_value
                    except (ValueError, TypeError, json.JSONDecodeError):
                        logger.warning(
                            "Parsed value '%s' of parameter '%s' is not a valid JSON "
                            "object in tool '%s', will try other methods to parse it.",
                            param_value,
                            param_name,
                            func_name,
                        )
                try:
                    param_value = ast.literal_eval(param_value)
                except (ValueError, SyntaxError):
                    logger.warning(
                        "Parsed value '%s' of parameter '%s' cannot be converted via "
                        "Python `ast.literal_eval()` in tool '%s', degenerating to string.",
                        param_value,
                        param_name,
                        func_name,
                    )
                return param_value

        # Extract function name
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = get_arguments_config(function_name)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match in self.tool_call_parameter_regex.findall(parameters):
            match_text = match[0] if match[0] else match[1]
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = convert_param_value(
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

        # Check if both think start and end tokens are present
        if (
            self.think_start_token in model_output
            and self.think_end_token in model_output
        ):
            # Find the position of think end token
            think_end_index = model_output.find(self.think_end_token) + len(
                self.think_end_token
            )
            # Extract content after think end token
            result_content = model_output[think_end_index:]
            thinking_content = model_output[:think_end_index]
        else:
            thinking_content = ""
            result_content = model_output

        try:
            function_calls = self._get_function_calls(result_content)
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
            tool_call_start_index = result_content.find(self.tool_call_start_token)
            tool_call_start_index = (
                tool_call_start_index
                if tool_call_start_index >= 0
                else result_content.find(self.tool_call_prefix)
            )
            content = thinking_content + result_content[:tool_call_start_index]

            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
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
        # If no delta text, return None unless
        # it's an EOS token after tool calls
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # We check for tool calls in the text even if is_tool_call_started
            # is False because it might have been reset after processing all tools
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
                        # Return empty delta message to allow finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Check if this is the first call (reset state if needed)
        if not previous_text:
            self._reset_streaming_state()

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

                # Check if there are more tool calls
                if self.current_tool_index >= current_text.count(
                    self.tool_call_start_token
                ):
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Check if end thinking
        if not self.is_thinking_end and (
            self.think_end_token_id in delta_token_ids
            or self.think_end_token in delta_text
        ):
            self.is_thinking_end = True

        # If thinking hasn't ended yet, don't process any tool calls
        if not self.is_thinking_end:
            return DeltaMessage(content=delta_text)

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

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        # Only process tool calls after think_end_token
        think_end_index = (
            current_text.find(self.think_end_token) + len(self.think_end_token)
            if self.think_end_token in current_text
            else 0
        )
        tool_starts: list[int] = []
        idx = think_end_index
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_starts.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_starts):
            # No more tool calls to process yet
            return None

        tool_start_idx = tool_starts[self.current_tool_index]
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
                    self.current_tool_id = self._generate_tool_call_id()  # type: ignore
                    self.header_sent = True
                    self.in_function = True

                    # IMPORTANT: Add to prev_tool_call_arr immediately when we detect a tool call
                    # This ensures finish_reason="tool_calls" even if parsing isn't complete
                    already_added = any(
                        tool.get("name") == self.current_function_name
                        for tool in self.prev_tool_call_arr
                    )
                    if not already_added:
                        self.prev_tool_call_arr.append(
                            {
                                "name": self.current_function_name,
                                "arguments": "{}",  # Placeholder, will be updated later
                            }
                        )

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
            # Send opening brace if not sent yet
            if not self.json_started and self.parameter_prefix not in delta_text:
                self.json_started = True
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            # Make sure json_started is set if we're processing parameters
            if not self.json_started:
                self.json_started = True

            # Check for function end in accumulated text
            if not self.json_closed and self.function_end_token in tool_text:
                # Close JSON
                self.json_closed = True

                # Extract the complete tool call to update prev_tool_call_arr with final arguments
                # Find the function content
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    # Parse to get the complete arguments
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content, request.tools if request else None
                        )
                        if parsed_tool:
                            # Update existing entry in prev_tool_call_arr with complete arguments
                            for i, tool in enumerate(self.prev_tool_call_arr):
                                if tool.get("name") == parsed_tool.function.name:
                                    self.prev_tool_call_arr[i]["arguments"] = (
                                        parsed_tool.function.arguments
                                    )
                                    break
                    except Exception:
                        logger.warning(
                            "Failed to parse tool arguments during streaming.",
                            exc_info=True,
                        )

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )

                # Reset state for next tool
                self.in_function = False
                self.json_closed = True

                return result

            # Look for parameters
            # Count how many complete parameters we have processed
            complete_params = tool_text.count(self.parameter_end_token)

            # Check if we should start a new parameter
            if not self.in_param and self.param_count < complete_params:
                # Find the unprocessed parameter
                # Count parameter starts
                param_starts = []
                idx = 0
                while True:
                    idx = tool_text.find(self.parameter_prefix, idx)
                    if idx == -1:
                        break
                    param_starts.append(idx)
                    idx += len(self.parameter_prefix)

                if len(param_starts) > self.param_count:
                    # Process the next parameter
                    param_idx = param_starts[self.param_count]
                    param_start = param_idx + len(self.parameter_prefix)
                    remaining = tool_text[param_start:]

                    if ">" in remaining:
                        # We have the complete parameter name
                        name_end = remaining.find(">")
                        self.current_param_name = remaining[:name_end]

                        # Find the parameter value
                        value_start = param_start + name_end + 1
                        value_text = tool_text[value_start:]
                        if value_text.startswith("\n"):
                            value_text = value_text[1:]

                        # Find where this parameter ends
                        param_end_idx = value_text.find(self.parameter_end_token)
                        if param_end_idx != -1:
                            # Complete parameter found
                            param_value = value_text[:param_end_idx]
                            if param_value.endswith("\n"):
                                param_value = param_value[:-1]

                            # Build complete JSON fragment for this parameter
                            if self.param_count == 0:
                                json_fragment = (
                                    '"'
                                    + self.current_param_name
                                    + '": "'
                                    + json.dumps(param_value)[1:-1]
                                    + '"'
                                )
                            else:
                                json_fragment = (
                                    ', "'
                                    + self.current_param_name
                                    + '": "'
                                    + json.dumps(param_value)[1:-1]
                                    + '"'
                                )

                            self.param_count += 1

                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=json_fragment
                                        ),
                                    )
                                ]
                            )

            # Continue parameter value
            if self.in_param:
                if self.parameter_end_token in delta_text:
                    # End of parameter
                    end_idx = delta_text.find(self.parameter_end_token)
                    value_chunk = delta_text[:end_idx]

                    # Skip past > if at start
                    if not self.current_param_value and ">" in value_chunk:
                        gt_idx = value_chunk.find(">")
                        value_chunk = value_chunk[gt_idx + 1 :]

                    if not self.current_param_value and value_chunk.startswith("\n"):
                        value_chunk = value_chunk[1:]

                    # Calculate incremental JSON
                    full_value = self.current_param_value + value_chunk
                    prev_escaped = (
                        json.dumps(self.current_param_value)[1:-1]
                        if self.current_param_value
                        else ""
                    )
                    full_escaped = json.dumps(full_value)[1:-1]
                    delta_escaped = full_escaped[len(prev_escaped) :]

                    self.in_param = False
                    self.current_param_value = ""

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(
                                    arguments=delta_escaped + '"'
                                ),
                            )
                        ]
                    )
                else:
                    # Continue accumulating value
                    value_chunk = delta_text

                    # Handle first chunk after param name
                    if not self.current_param_value and ">" in value_chunk:
                        gt_idx = value_chunk.find(">")
                        value_chunk = value_chunk[gt_idx + 1 :]

                    if not self.current_param_value and value_chunk.startswith("\n"):
                        value_chunk = value_chunk[1:]

                    if value_chunk:
                        # Stream the escaped delta
                        prev_escaped = (
                            json.dumps(self.current_param_value)[1:-1]
                            if self.current_param_value
                            else ""
                        )
                        self.current_param_value += value_chunk
                        full_escaped = json.dumps(self.current_param_value)[1:-1]
                        delta_escaped = full_escaped[len(prev_escaped) :]

                        if delta_escaped:
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=delta_escaped
                                        ),
                                    )
                                ]
                            )

        return None
