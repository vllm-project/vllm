# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
    )
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
    )
from vllm.entrypoints.openai.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
    partial_json_loads,
    )
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@ToolParserManager.register_module("apertus")
class ApertusToolParser(ToolParser):
    """
    Tool call parser for Apertus models.

    Extracts tool calls from the format:
    <|tools_prefix|>[{"function_name": {"arg1": "value1", ...}}, ...]<|tools_suffix|>

    Used when --enable-auto-tool-choice --tool-call-parser apertus are set.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        # Tokens for tool call delimiters
        self.tool_calls_prefix = "<|tools_prefix|>"
        self.tool_calls_suffix = "<|tools_suffix|>"

        # State for streaming
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        # Regex to extract tool calls block (suffix is optional for incomplete outputs)
        self.tool_call_regex = re.compile(
                rf"{re.escape(self.tool_calls_prefix)}(.*?)(?:{re.escape(self.tool_calls_suffix)}|$)",
                re.DOTALL,
                )

    def extract_tool_calls(
            self, model_output: str, request: ChatCompletionRequest
            ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response."""
        # Quick check before running regex
        if self.tool_calls_prefix not in model_output:
            return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                    )

        # Find tool calls block
        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                    )

        try:
            json_str = match.group(1).strip()
            tool_call_objects = json.loads(json_str)

            if not isinstance(tool_call_objects, list):
                tool_call_objects = [tool_call_objects]

            tool_calls = self._parse_tool_call_objects(tool_call_objects)

            return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=None
                    )

        except Exception:
            logger.exception("Error extracting tool call from response.")
            return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                    )

    def _parse_tool_call_objects(self, tool_call_objects: list[dict]) -> list[ToolCall]:
        """Parse tool call objects into ToolCall instances."""
        tool_calls: list[ToolCall] = []

        for obj in tool_call_objects:
            # Each object is {"function_name": {"arg1": "value1", ...}}
            if isinstance(obj, dict) and len(obj) == 1:
                function_name = next(iter(obj))
                arguments = obj[function_name]

                tool_calls.append(
                        ToolCall(
                                type="function",
                                function=FunctionCall(
                                        name=function_name,
                                        arguments=json.dumps(arguments, ensure_ascii=False),
                                        ),
                                )
                        )

        return tool_calls

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
        """Extract tool calls in streaming mode."""
        # Check if we're in a tool call block
        if self.tool_calls_prefix not in current_text:
            return DeltaMessage(content=delta_text)

        json_str = self._extract_json_str(current_text)

        try:
            tool_call_arr = self._parse_partial_json(json_str)

            if not tool_call_arr:
                return None

            # Starting a new tool in the array
            if len(tool_call_arr) > self.current_tool_id + 1:
                delta = self._finalize_previous_tool()
                self._start_new_tool(len(tool_call_arr))
                return delta

            current_tool_call = tool_call_arr[self.current_tool_id]

            # Send tool name if not sent yet
            if not self.current_tool_name_sent:
                return self._send_tool_name(current_tool_call)

            # Stream arguments
            delta = self._stream_arguments(current_tool_call, json_str)
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.debug("Error parsing streaming tool call, waiting for more tokens")
            return None

    def _extract_json_str(self, current_text: str) -> str:
        """Extract JSON string from the current text."""
        prefix_idx = current_text.find(self.tool_calls_prefix)
        start_idx = prefix_idx + len(self.tool_calls_prefix)

        # Check if suffix is present (complete tool call)
        suffix_idx = current_text.find(self.tool_calls_suffix, start_idx)
        if suffix_idx != -1:
            return current_text[start_idx:suffix_idx].strip()
        return current_text[start_idx:].strip()

    def _parse_partial_json(self, json_str: str) -> list[dict]:
        """Parse partial JSON with appropriate flags."""
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        tool_call_arr, _ = partial_json_loads(json_str, flags)

        if not isinstance(tool_call_arr, list):
            tool_call_arr = [tool_call_arr] if tool_call_arr else []

        return tool_call_arr

    def _finalize_previous_tool(self) -> DeltaMessage | None:
        """Finalize any remaining arguments from the previous tool."""
        if self.current_tool_id < 0:
            return None

        prev_tool = self.prev_tool_call_arr[self.current_tool_id]
        function_name = next(iter(prev_tool))
        arguments = prev_tool[function_name]
        args_json = json.dumps(arguments, ensure_ascii=False)
        sent = len(self.streamed_args_for_tool[self.current_tool_id])
        argument_diff = args_json[sent:]

        if not argument_diff:
            return None

        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
        return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=argument_diff).model_dump(
                                    exclude_none=True
                                    ),
                            )
                    ]
                )

    def _start_new_tool(self, array_length: int) -> None:
        """Start processing a new tool."""
        self.current_tool_id = array_length - 1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool.append("")

    def _send_tool_name(self, current_tool_call: dict) -> DeltaMessage | None:
        """Send the tool name if not sent yet."""
        if not isinstance(current_tool_call, dict) or len(current_tool_call) != 1:
            return None

        function_name = next(iter(current_tool_call))

        self.current_tool_name_sent = True
        return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=function_name).model_dump(
                                    exclude_none=True
                                    ),
                            )
                    ]
                )

    def _stream_arguments(
            self, current_tool_call: dict, json_str: str
            ) -> DeltaMessage | None:
        """Stream arguments for the current tool."""
        if not isinstance(current_tool_call, dict) or len(current_tool_call) != 1:
            return None

        function_name = next(iter(current_tool_call))
        arguments = current_tool_call[function_name]

        if not arguments:
            return None

        sent = len(self.streamed_args_for_tool[self.current_tool_id])
        args_json = json.dumps(arguments, ensure_ascii=False)

        argument_diff = self._calculate_argument_diff(
                function_name, args_json, json_str, sent
                )

        if not argument_diff:
            return None

        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
        return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=argument_diff).model_dump(
                                    exclude_none=True
                                    ),
                            )
                    ]
                )

    def _calculate_argument_diff(
            self, function_name: str, args_json: str, json_str: str, sent: int
            ) -> str | None:
        """Calculate the difference in arguments to stream."""
        is_complete_call = is_complete_json(json_str)

        if is_complete_call:
            return args_json[sent:]

        if not self.prev_tool_call_arr or self.current_tool_id >= len(
                self.prev_tool_call_arr
                ):
            return None

        prev_tool = self.prev_tool_call_arr[self.current_tool_id]
        prev_function_name = next(iter(prev_tool))

        if prev_function_name != function_name:
            return None

        prev_args = prev_tool[prev_function_name]
        prev_args_json = json.dumps(prev_args, ensure_ascii=False)

        if args_json == prev_args_json:
            return None

        prefix = find_common_prefix(prev_args_json, args_json)
        return prefix[sent:]