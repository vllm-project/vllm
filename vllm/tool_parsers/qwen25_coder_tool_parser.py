# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool parser for Qwen2.5-Coder models.

Qwen2.5-Coder models output tool calls in <tools> tags with JSON content,
unlike Hermes format (<tool_call>) or Qwen3-Coder format (XML parameters).

Expected format (single tool):
<tools>
{"name": "function_name", "arguments": {"param1": "value1"}}
</tools>

Expected format (multiple tools):
<tools>
[{"name": "func1", "arguments": {...}}, {"name": "func2", "arguments": {...}}]
</tools>

The model follows this format with high compliance when given appropriate
few-shot examples in the chat template.
"""
import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
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
from vllm.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
    partial_json_loads,
)

logger = init_logger(__name__)


class Qwen25CoderToolParser(ToolParser):
    """
    Tool call parser for Qwen2.5-Coder models.

    Qwen2.5-Coder outputs tool calls within <tools></tools> tags containing
    JSON objects or arrays. This differs from:
    - Hermes format: uses <tool_call></tool_call> tags
    - Qwen3-Coder format: uses XML-style <function=name><parameter=...>

    This parser extracts JSON tool calls from <tools> tags and supports
    both single tool calls and parallel (array) tool calls.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        # Define the tag markers for Qwen2.5-Coder format
        self.tool_call_start_token: str = "<tools>"
        self.tool_call_end_token: str = "</tools>"

        # Regex to extract content between <tools> tags
        # Handles both complete and incomplete (streaming) cases
        self.tool_call_regex = re.compile(
            r"<tools>(.*?)</tools>|<tools>(.*)", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Get token IDs for start/end markers
        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(
            self.tool_call_end_token, add_special_tokens=False
        )

        # Create arrays for buffering partial tokens
        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_start_token_ids
        ]

        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_end_token_ids
        ]

        self.buffered_delta_text = ""

        logger.info(
            "vLLM Successfully initialized tool parser %s!", self.__class__.__name__
        )

    def _buffer_delta_text(self, delta_text: str) -> str:
        """
        Buffer tokens that might be part of <tools> or </tools> tags.

        When encountering tokens like <, tools, >, store them in a buffer.
        When the last token is encountered, return the buffer.
        If tokens don't form a valid sequence, return buffered + current.
        """
        if (
            delta_text in self.tool_call_start_token_array
            or delta_text in self.tool_call_end_token_array
        ):
            # Check if this completes a start or end token
            if (
                delta_text == self.tool_call_start_token_array[-1]
                or delta_text == self.tool_call_end_token_array[-1]
            ):
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def _parse_tool_calls_from_json(
        self, json_content: str
    ) -> list[ToolCall]:
        """
        Parse tool calls from JSON content.

        Supports both single object and array formats:
        - {"name": "func", "arguments": {...}}
        - [{"name": "func1", ...}, {"name": "func2", ...}]
        """
        tool_calls = []

        try:
            parsed = json.loads(json_content)

            # Handle array of tool calls
            if isinstance(parsed, list):
                raw_function_calls = parsed
            # Handle single tool call object
            elif isinstance(parsed, dict):
                raw_function_calls = [parsed]
            else:
                logger.warning(
                    "Unexpected JSON type in tool call: %s", type(parsed)
                )
                return []

            for function_call in raw_function_calls:
                if not isinstance(function_call, dict):
                    continue

                name = function_call.get("name")
                if not name:
                    continue

                # Support both "arguments" and "parameters" keys
                arguments = function_call.get(
                    "arguments", function_call.get("parameters", {})
                )

                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

        except json.JSONDecodeError:
            logger.exception("Failed to parse JSON from tool call content")

        return tool_calls

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request to not skip special tokens for tool parsing."""
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Don't skip special tokens as <tools> tags need to be parsed
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response.

        Looks for <tools>...</tools> tags and parses the JSON content within.
        """
        # Quick check to avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # Find all tool call blocks
            matches = self.tool_call_regex.findall(model_output)

            all_tool_calls = []
            for match in matches:
                # match is a tuple - get the non-empty group
                json_content = match[0] if match[0] else match[1]
                json_content = json_content.strip()

                if json_content:
                    tool_calls = self._parse_tool_calls_from_json(json_content)
                    all_tool_calls.extend(tool_calls)

            if all_tool_calls:
                # Extract content before the first tool call
                content_end = model_output.find(self.tool_call_start_token)
                content = model_output[:content_end].strip() if content_end > 0 else None

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=all_tool_calls,
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
        """
        Extract tool calls from streaming model output.

        Handles incremental parsing of <tools>JSON</tools> content.
        """
        # Apply buffering for partial tag tokens
        delta_text = self._buffer_delta_text(delta_text)

        # Adjust text to account for buffered content
        if (
            len(previous_text) >= len(self.buffered_delta_text)
            and previous_text[-len(self.buffered_delta_text):]
            == self.buffered_delta_text
        ):
            previous_text = previous_text[: -len(self.buffered_delta_text)]
            current_text = previous_text + delta_text

        # If no tool call started yet, check if we should stream content
        if self.tool_call_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            # Count start and end tags
            prev_tool_start_count = previous_text.count(self.tool_call_start_token)
            prev_tool_end_count = previous_text.count(self.tool_call_end_token)
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)

            # If we're outside tool calls (equal start/end), stream as content
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                return DeltaMessage(content=delta_text)

            # Get the current tool call JSON content
            tool_call_content = None

            if cur_tool_start_count > cur_tool_end_count:
                # We're inside a tool call - extract content after last <tools>
                last_start = current_text.rfind(self.tool_call_start_token)
                tool_call_content = current_text[
                    last_start + len(self.tool_call_start_token):
                ]
            elif self.tool_call_end_token in delta_text:
                # Tool call is being closed
                last_start = current_text.rfind(self.tool_call_start_token)
                last_end = current_text.rfind(self.tool_call_end_token)
                if last_start < last_end:
                    tool_call_content = current_text[
                        last_start + len(self.tool_call_start_token): last_end
                    ]

            if not tool_call_content:
                return None

            tool_call_content = tool_call_content.strip()

            # Use partial JSON parser for streaming
            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            try:
                tool_call_arr = []
                is_complete = []
                start_idx = 0

                while start_idx < len(tool_call_content):
                    try:
                        (obj, end_idx) = partial_json_loads(
                            tool_call_content[start_idx:], flags
                        )
                        is_complete.append(
                            is_complete_json(
                                tool_call_content[start_idx: start_idx + end_idx]
                            )
                        )
                        # Support both "arguments" and "parameters"
                        if "parameters" in obj and "arguments" not in obj:
                            obj["arguments"] = obj["parameters"]
                        tool_call_arr.append(obj)
                        # Move past this JSON object (handle comma/whitespace)
                        remaining = tool_call_content[start_idx + end_idx:].lstrip()
                        if remaining.startswith(","):
                            remaining = remaining[1:].lstrip()
                        start_idx = len(tool_call_content) - len(remaining)
                        if start_idx == len(tool_call_content) - len(
                            tool_call_content[start_idx + end_idx:]
                        ):
                            break
                    except partial_json_parser.core.exceptions.MalformedJSON:
                        break

            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("Not enough tokens to parse into JSON yet")
                return None

            if len(tool_call_arr) == 0:
                return None

            current_tool_call: dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # Starting a new tool call
            if len(tool_call_arr) > self.current_tool_id + 1:
                # Handle completion of previous tool if any
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        if argument_diff:
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=argument_diff
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            self.streamed_args_for_tool[self.current_tool_id] += (
                                argument_diff
                            )
                            self.prev_tool_call_arr = tool_call_arr
                            return delta
                    delta = None
                else:
                    delta = None

                # Start new tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on new tool %d", self.current_tool_id)
                self.prev_tool_call_arr = tool_call_arr
                return delta

            # Send tool name if not sent yet
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                    self.prev_tool_call_arr = tool_call_arr
                    return delta
                return None

            # Stream arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)

                    prev_arguments = None
                    if (
                        self.prev_tool_call_arr
                        and self.current_tool_id < len(self.prev_tool_call_arr)
                    ):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id
                        ].get("arguments")

                    argument_diff = None
                    if is_complete and self.current_tool_id < len(is_complete):
                        if is_complete[self.current_tool_id]:
                            argument_diff = cur_args_json[sent:]
                        elif prev_arguments:
                            prev_args_json = json.dumps(
                                prev_arguments, ensure_ascii=False
                            )
                            if cur_args_json != prev_args_json:
                                prefix = find_common_prefix(
                                    prev_args_json, cur_args_json
                                )
                                argument_diff = prefix[sent:]

                    if argument_diff:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff
                        )

                self.prev_tool_call_arr = tool_call_arr
                return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
