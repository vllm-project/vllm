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


class DeepSeekV32ToolParser(ToolParser):
    """
    example tool call content:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.dsml_token: str = "｜DSML｜"
        self.dsml_start_check: str = "<" + self.dsml_token
        self.tool_call_start_token: str = "<｜DSML｜function_calls>"
        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
        self.invoke_end_token: str = "</｜DSML｜invoke>"
        self.parameter_prefix: str = "<｜DSML｜parameter name="
        self.parameter_end_token: str = "</｜DSML｜parameter>"

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
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def adjust_request(self, request):
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens
            # (<｜DSML｜function_calls>, </｜DSML｜function_calls>)
            # are not skippedduring decoding.
            # Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

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

    def _parse_invoke_params(self, invoke_str: str) -> dict | None:
        param_dict = dict()
        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = param_val
        return param_dict

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
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match
                ):
                    param_dict = self._parse_invoke_params(invoke_content)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name,
                                arguments=json.dumps(param_dict, ensure_ascii=False),
                            ),
                        )
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
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
        """Read all complete <parameter> pairs from tool_text and return a
        partial or complete JSON object string, diffed by the caller."""
        is_complete = self.invoke_end_token in tool_text

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

        param_pairs: list[tuple[str, Any]] = []
        for param_name, param_value in self.parameter_complete_regex.findall(tool_text):
            param_value = param_value.strip()
            param_type = "string"
            if param_name in param_config and isinstance(
                param_config[param_name], dict
            ):
                param_type = param_config[param_name].get("type", "string")
            param_pairs.append(
                (param_name, self._convert_param_value(param_value, param_type))
            )

        return build_partial_args_json(param_pairs, is_complete)

    def _finalize_tool_call(self, complete_args_json: str) -> None:
        """Mark the current tool call complete and persist final arguments."""
        if self.current_tool_index < len(self.prev_tool_call_arr):
            try:
                self.prev_tool_call_arr[self.current_tool_index]["arguments"] = (
                    json.loads(complete_args_json)
                )
            except json.JSONDecodeError:
                logger.warning(
                    "[V32_STREAMING] Failed to parse complete args JSON: %s",
                    complete_args_json,
                )
        self.json_closed = True
        self.in_function = False

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type."""
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        elif param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            # Try JSON parse first, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

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
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, check whether there is still work to do
        if not delta_text:
            # If we're mid-tool-call, fall through to the main processing
            # logic — current_text may contain parameters that haven't been
            # diffed yet (happens when stream_interval > 1 batches tokens)
            if self.is_tool_call_started and not self.json_closed:
                pass  # fall through
            elif delta_token_ids:
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
            invoke_ends = current_text.count(self.invoke_end_token)
            if invoke_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.json_closed = False

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            if self.dsml_token in delta_text:
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.dsml_start_check in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.dsml_start_check)
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
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    function_name_raw = tool_text[func_start:func_end]
                    self.current_function_name = self._extract_name(function_name_raw)
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    if len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append(
                            {"name": self.current_function_name, "arguments": {}}
                        )
                    if len(self.streamed_args_for_tool) <= self.current_tool_index:
                        self.streamed_args_for_tool.append("")

                    # Compute any args already visible in current_text right now
                    initial_args = self._compute_current_args_json(tool_text, request)
                    self.streamed_args_for_tool[self.current_tool_index] = initial_args

                    if self.invoke_end_token in tool_text and initial_args.endswith(
                        "}"
                    ):
                        self._finalize_tool_call(initial_args)

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

        self.streamed_args_for_tool[self.current_tool_index] = current_args

        if self.invoke_end_token in tool_text and current_args.endswith("}"):
            self._finalize_tool_call(current_args)
            logger.debug("[V32_STREAMING] Tool call completed")

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_index,
                    function=DeltaFunctionCall(arguments=args_delta),
                )
            ]
        )
