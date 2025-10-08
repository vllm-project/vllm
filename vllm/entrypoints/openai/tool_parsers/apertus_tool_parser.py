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

        # previous tool call '[{"calculate": {"x": 5}}]'
        self.prev_tool_call_arr: list[dict] = []

        # State for streaming
        self._reset_streaming_state()

        # Regex to extract tool calls block (suffix is optional for incomplete outputs)
        self.tool_call_regex = re.compile(
            rf"{re.escape(self.tool_calls_prefix)}(.*?)(?:{re.escape(self.tool_calls_suffix)}|$)",
            re.DOTALL,
        )

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

    def _reset_streaming_state(self):
        """Reset streaming state for a new request."""
        self._initialize_tool_state(-1)

    def _initialize_tool_state(self, tool_id: int):
        """
        Initialize or reset tool-specific state.

        Args:
            tool_id: The index position of the tool in the tool call array.
                     Use -1 to indicate no tool is being processed (initial state).
        """
        self.current_tool_id = tool_id
        self.current_tool_name_sent = False
        if tool_id == -1:
            self.streamed_args_for_tool: list[str] = []
        else:
            self.streamed_args_for_tool.append("")

    def _start_new_tool(self, array_length: int) -> None:
        """
        Start processing a new tool from the streaming array.

        Sets up state to process the tool at the last position in the array.
        The tool index is calculated as array_length - 1 since arrays are 0-indexed.

        Args:
            array_length: Current length of the tool_call_arr list.
        """
        self._initialize_tool_state(array_length - 1)

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
        # Reset state at the start of a new streaming session
        # (detected when previous_text is empty or doesn't contain tool prefix)
        if not previous_text or (
            self.tool_calls_prefix not in previous_text
            and self.tool_calls_prefix in current_text
        ):
            self._reset_streaming_state()

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
                self.prev_tool_call_arr = tool_call_arr
                return delta

            current_tool_call = tool_call_arr[self.current_tool_id]

            # Send tool name if not sent yet
            if not self.current_tool_name_sent:
                delta = self._send_tool_name(current_tool_call)
                self.prev_tool_call_arr = tool_call_arr
                return delta

            # Stream arguments
            delta = self._stream_arguments(current_tool_call, json_str)
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.debug("Error parsing streaming tool call, waiting for more tokens")
            return None

    @staticmethod
    def _parse_tool_call_objects(tool_call_objects: list[dict]) -> list[ToolCall]:
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
                        id=make_tool_call_id(),
                    )
                )

        return tool_calls

    def _extract_json_str(self, current_text: str) -> str:
        """
        Extract JSON string from the current text between tool call delimiters.

        Extracts the content between <|tools_prefix|> and <|tools_suffix|> tokens.
        If the suffix is not present (incomplete streaming),
        returns everything after the prefix.

        Args:
            current_text: The accumulated model output text.

        Returns:
            The extracted JSON string without the delimiter tokens.

        Example:
            >>> current_text = 'Some text <|tools_prefix|>
                [{"func": {"arg": "val"}}]<|tools_suffix|>'
            >>> self._extract_json_str(current_text)
            '[{"func": {"arg": "val"}}]'

            >>> current_text = 'Some text <|tools_prefix|>[{"func": {"arg":'
            >>> self._extract_json_str(current_text)
            '[{"func": {"arg":'
        """
        prefix_idx = current_text.find(self.tool_calls_prefix)
        start_idx = prefix_idx + len(self.tool_calls_prefix)

        # Check if suffix is present (complete tool call)
        suffix_idx = current_text.find(self.tool_calls_suffix, start_idx)
        if suffix_idx != -1:
            return current_text[start_idx:suffix_idx].strip()
        return current_text[start_idx:].strip()

    def _parse_partial_json(self, json_str: str) -> list[dict]:
        """
        Parse potentially incomplete JSON string into a list of tool call objects.

        Uses different parsing flags based on whether the tool name has been sent:
        - Before tool name is sent: Disallow incomplete strings (Allow.ALL & ~Allow.STR)
        - After tool name is sent: Allow all incomplete JSON (Allow.ALL)

        Always returns a list, even if the JSON represents a single object.

        Args:
            json_str: The JSON string to parse, which may be incomplete.

        Returns:
            A list of tool call dictionaries. Empty list if parsing fails.

        Example:
            >>> json_str = '[{"calculate": {"x": 5, "y":'
            >>> self._parse_partial_json(json_str)
            [{"calculate": {"x": 5}}]

            >>> json_str = '{"calculate": {"x": 5}}'
            >>> self._parse_partial_json(json_str)
            [{"calculate": {"x": 5}}]
        """
        # if current_tool_name_sent = False
        # Allows everything EXCEPT incomplete strings (~Allow.STR means "NOT Allow.STR")
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        tool_call_arr, _ = partial_json_loads(json_str, flags)

        if not isinstance(tool_call_arr, list):
            tool_call_arr = [tool_call_arr] if tool_call_arr else []

        return tool_call_arr

    def _finalize_previous_tool(self) -> DeltaMessage | None:
        """
        Finalize any remaining arguments
        from the previous tool before moving to the next one.

        When transitioning to a new tool in the array,
        this ensures all arguments
        from the current tool that haven't been
        streamed yet are sent in one final delta.

        This is necessary because when a new tool appears, we may have parsed additional
        arguments for the previous tool that weren't yet sent to the client.

        Returns:
            DeltaMessage with the remaining arguments, or None if:
            - No tool is currently being processed (current_tool_id < 0)
            - No previous tool call array exists
            - All arguments have already been streamed
        """
        if self.current_tool_id < 0:
            return None

        # Check if prev_tool_call_arr has been initialized and has the current tool
        if not self.prev_tool_call_arr or self.current_tool_id >= len(
            self.prev_tool_call_arr
        ):
            return None

        # Check if streamed_args_for_tool has the current tool
        if self.current_tool_id >= len(self.streamed_args_for_tool):
            return None

        prev_tool = self.prev_tool_call_arr[self.current_tool_id]
        # Using next(iter(prev_tool)) gets the first (and only) key
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

    def _send_tool_name(self, current_tool_call: dict) -> DeltaMessage | None:
        """
        Send the function name as the first delta for a new tool call.

        This is called once per tool to send the initial tool call information including
        the function name, tool ID, and type. After this, subsequent deltas will only
        contain arguments.

        The tool call format is expected to be: {"function_name": {"arg1": "val1", ...}}

        Args:
            current_tool_call: A dictionary with a single key (function name) and
                             value (arguments dict).

        Returns:
            DeltaMessage containing the tool name and metadata, or None if:
            - current_tool_call is not a dict
            - current_tool_call doesn't have exactly one key

        Example:
            >>> current_tool_call = {"calculate": {"x": 5}}
            >>> delta = self._send_tool_name(current_tool_call)
            # Returns DeltaMessage with function name="calculate" and a new tool ID
        """
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
        """
        Stream incremental argument updates for the current tool call.

        Called repeatedly during streaming to send new portions of the arguments
        as they become available. Tracks what has already been sent and only streams
        the difference.

        The tool call format is expected to be:
            {"function_name": {"arg1": "val1", ...}}

        Args:
            current_tool_call: A dictionary with a single key (function name) and
                             value (arguments dict).
            json_str: The raw JSON string being parsed (may be incomplete).

        Returns:
            DeltaMessage containing only the new argument portion, or None if:
            - current_tool_call is not a dict or doesn't have exactly one key
            - Arguments dict is empty
            - Current tool ID is out of bounds for streamed_args_for_tool
            - No new arguments to send (diff calculation returns None)

        Example:
            Iteration 1: arguments = {"x": 5}
                        -> Returns DeltaMessage(arguments='{"x": 5}')

            Iteration 2: arguments = {"x": 5, "y": 10}
                        -> Returns DeltaMessage(arguments=', "y": 10}')

            Iteration 3: arguments unchanged
                        -> Returns None
        """
        if not isinstance(current_tool_call, dict) or len(current_tool_call) != 1:
            return None

        function_name = next(iter(current_tool_call))
        arguments = current_tool_call[function_name]

        if not arguments:
            return None

        # Check if streamed_args_for_tool has the current tool
        if self.current_tool_id >= len(self.streamed_args_for_tool):
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
        """
        Calculate the new portion of arguments to stream to the client.

        Determines what part of the arguments JSON hasn't been sent yet by comparing
        the current arguments with the previous iteration's arguments.

        For complete JSON, returns everything after the already-sent portion.
        For incomplete JSON, finds the common prefix between previous and current
        arguments to determine what's safe to send.

        Args:
            function_name: The name of the current function being processed.
            args_json: Current arguments serialized as JSON string.
            json_str: The raw JSON string being parsed (may be incomplete).
            sent: Number of characters already sent to the client.

        Returns:
            The new portion of the arguments JSON to stream, or None if:
            - JSON is incomplete and function name has changed
            - No previous tool call exists to compare with
            - Arguments haven't changed since last iteration

        Example:
            Iteration 1: args_json =
                '{"x": 5'  -> sent = 0  -> returns '{"x": 5'
            Iteration 2: args_json =
                '{"x": 5, "y": 10}' -> sent = 7 -> returns ', "y": 10}'
        """
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
