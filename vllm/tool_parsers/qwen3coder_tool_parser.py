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
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class Qwen3CoderToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

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
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
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

        logger.info(
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

    def _get_arguments_config(
        self, func_name: str, tools: list[ChatCompletionToolsParam] | None
    ) -> dict:
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
        self, function_call_str: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

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

    def _backfill_unparsed_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest,
    ) -> None:
        """Parse all tool calls from current_text and ensure
        prev_tool_call_arr and streamed_args_for_tool are fully
        populated. Called at end-of-stream to catch tool calls
        that arrived too fast for incremental streaming."""
        try:
            function_calls = self._get_function_calls(current_text)
            tools = request.tools if request else None
            for i, fc_str in enumerate(function_calls):
                parsed = self._parse_xml_function_call(fc_str, tools)
                if not parsed:
                    continue
                name = parsed.function.name
                args_json = parsed.function.arguments
                args_dict = json.loads(args_json)
                # Update or add to prev_tool_call_arr by index
                while len(self.prev_tool_call_arr) <= i:
                    self.prev_tool_call_arr.append(
                        {"name": name, "arguments": {}}
                    )
                self.prev_tool_call_arr[i]["name"] = name
                self.prev_tool_call_arr[i]["arguments"] = args_dict
                # Ensure streamed_args_for_tool has entry
                while len(self.streamed_args_for_tool) <= i:
                    self.streamed_args_for_tool.append("")
                # Set the full args so remaining_call = ""
                self.streamed_args_for_tool[i] = args_json
        except Exception:
            logger.exception("Error in backfill_unparsed_tool_calls")

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
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, check if we still have pending work
        if not delta_text:
            # If we're mid-tool-call (header sent, function body pending),
            # don't short-circuit - fall through to process accumulated text
            if not (self.header_sent and self.in_function):
                # Check if this is an EOS token after all tool calls
                if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                    complete_calls = len(
                        self.tool_call_complete_regex.findall(current_text)
                    )
                    if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                        open_calls = current_text.count(
                            self.tool_call_start_token
                        ) - current_text.count(self.tool_call_end_token)
                        if open_calls == 0:
                            # Stream ending - ensure ALL tool calls
                            # are in prev_tool_call_arr even if we
                            # didn't get to stream them incrementally
                            self._backfill_unparsed_tool_calls(
                                current_text, request
                            )
                            return DeltaMessage(content="")
                    elif not self.is_tool_call_started and current_text:
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
                    return None
                # More tools to process - fall through immediately
                # instead of wasting a call on the advance step

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

                    # IMPORTANT: Add to prev_tool_call_arr immediately when
                    # we detect a tool call. This ensures
                    # finish_reason="tool_calls" even if parsing isn't complete.
                    # Use index-based check (not name-based) because
                    # the model can make multiple calls to the same tool.
                    while len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append(
                            {
                                "name": self.current_function_name,
                                "arguments": {},
                            }
                        )
                    # Always ensure streamed_args_for_tool has an
                    # entry for current_tool_index
                    while len(self.streamed_args_for_tool) <= self.current_tool_index:
                        self.streamed_args_for_tool.append("")

                    # If the complete tool call is already in
                    # tool_text, send header + args in one shot.
                    # This handles the case where the model sends
                    # everything in 1-2 chunks and there won't be
                    # another iteration to process the body.
                    if self.function_end_token in tool_text:
                        args_json = "{}"
                        fc_start = tool_text.find(
                            self.tool_call_prefix
                        ) + len(self.tool_call_prefix)
                        fc_end = tool_text.find(
                            self.function_end_token, fc_start
                        )
                        if fc_end != -1:
                            fc_content = tool_text[fc_start:fc_end]
                            try:
                                parsed = self._parse_xml_function_call(
                                    fc_content,
                                    request.tools if request else None,
                                )
                                if parsed:
                                    args_json = parsed.function.arguments
                                    # Update by index (not name) to
                                    # handle duplicate tool names
                                    idx = self.current_tool_index
                                    if idx < len(self.prev_tool_call_arr):
                                        self.prev_tool_call_arr[idx][
                                            "arguments"
                                        ] = json.loads(args_json)
                            except Exception:
                                logger.warning("Failed to parse tool call arguments in one-shot.", exc_info=True)
                        self.streamed_args_for_tool[
                            self.current_tool_index
                        ] = args_json
                        self.in_function = False
                        self.json_started = True
                        self.json_closed = True
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    id=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        name=self.current_function_name,
                                        arguments=args_json,
                                    ),
                                    type="function",
                                )
                            ]
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
            # If the complete function body is available, use
            # authoritative parse. Works regardless of json_started
            # state - handles both first-time and incremental cases.
            if self.function_end_token in tool_text:
                func_start = tool_text.find(
                    self.tool_call_prefix
                ) + len(self.tool_call_prefix)
                func_content_end = tool_text.find(
                    self.function_end_token, func_start
                )
                args_json = "{}"
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                            self.streaming_request.tools
                            if self.streaming_request
                            else None,
                        )
                        if parsed_tool:
                            args_json = parsed_tool.function.arguments
                            # Update by index (not name) to handle
                            # duplicate tool names correctly
                            idx = self.current_tool_index
                            if idx < len(self.prev_tool_call_arr):
                                self.prev_tool_call_arr[idx][
                                    "arguments"
                                ] = json.loads(args_json)
                    except Exception:
                        pass

                # Compute delta: only send what hasn't been sent yet
                already_sent = self.streamed_args_for_tool[
                    self.current_tool_index
                ]
                if already_sent and args_json.startswith(already_sent):
                    args_delta = args_json[len(already_sent):]
                else:
                    args_delta = args_json

                self.streamed_args_for_tool[
                    self.current_tool_index
                ] = args_json
                self.in_function = False
                self.json_started = True
                self.json_closed = True
                self.accumulated_params = {}

                if args_delta:
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(
                                    arguments=args_delta
                                ),
                            )
                        ]
                    )
                return None

            # Function body not complete yet - stream params
            # incrementally so the client sees progress.
            if not self.json_started:
                self.json_started = True

            # Get parameter config for type conversion
            param_config = self._get_arguments_config(
                self.current_function_name or "",
                self.streaming_request.tools
                if self.streaming_request
                else None,
            )

            # Find all parameter start positions in tool_text
            param_starts = []
            search_idx = 0
            while True:
                pos = tool_text.find(self.parameter_prefix, search_idx)
                if pos == -1:
                    break
                param_starts.append(pos)
                search_idx = pos + len(self.parameter_prefix)

            # Process all available complete parameters
            while self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break  # Parameter name not complete yet

                name_end = remaining.find(">")
                param_name = remaining[:name_end]

                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                # Find where this parameter ends
                param_end_idx = value_text.find(self.parameter_end_token)
                if param_end_idx == -1:
                    next_param_idx = value_text.find(
                        self.parameter_prefix
                    )
                    func_end_idx = value_text.find(
                        self.function_end_token
                    )
                    if next_param_idx != -1 and (
                        func_end_idx == -1
                        or next_param_idx < func_end_idx
                    ):
                        param_end_idx = next_param_idx
                    elif func_end_idx != -1:
                        param_end_idx = func_end_idx
                    elif self.tool_call_end_token in tool_text:
                        param_end_idx = len(value_text)
                    else:
                        break  # Still streaming, wait for more

                if param_end_idx == -1:
                    break

                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                # Store converted value (not raw string) so
                # json.dumps matches _parse_xml_function_call output
                converted_value = self._convert_param_value(
                    param_value,
                    param_name,
                    param_config,
                    self.current_function_name or "",
                )
                self.accumulated_params[param_name] = converted_value
                self.param_count += 1

            # Build partial args JSON from accumulated params
            # (without closing brace - that comes when function ends)
            if self.accumulated_params:
                args_so_far = json.dumps(
                    self.accumulated_params, ensure_ascii=False
                )[:-1]  # Strip closing }
            else:
                args_so_far = ""

            # Compute delta vs what was already sent
            already_sent = self.streamed_args_for_tool[
                self.current_tool_index
            ]
            if args_so_far and (
                not already_sent
                or args_so_far.startswith(already_sent)
            ):
                args_delta = args_so_far[len(already_sent):]
            else:
                args_delta = ""

            if args_delta:
                self.streamed_args_for_tool[
                    self.current_tool_index
                ] = args_so_far
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(
                                arguments=args_delta
                            ),
                        )
                    ]
                )

        return None
