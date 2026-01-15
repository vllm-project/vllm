# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

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
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class FunctionGemmaToolParser(ToolParser):
    """
    Tool parser for Google's FunctionGemma model (google/functiongemma-270m-it).

    Handles the FunctionGemma function call format:
    <start_function_call>call:func_name{param:<escape>value<escape>}<end_function_call>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Streaming state
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        # FunctionGemma tokens
        self.tool_call_start_token: str = "<start_function_call>"
        self.tool_call_end_token: str = "<end_function_call>"

        # Regex patterns
        self.tool_call_regex = re.compile(
            r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
            r"|<start_function_call>call:(\w+)\{(.*)",
            re.DOTALL,
        )
        self.arg_regex = re.compile(
            r"(\w+):<escape>(.*?)<escape>",
            re.DOTALL,
        )

        if self.model_tokenizer:
            self.tool_call_start_token_ids = self.model_tokenizer.encode(
                self.tool_call_start_token, add_special_tokens=False
            )
            self.tool_call_end_token_ids = self.model_tokenizer.encode(
                self.tool_call_end_token, add_special_tokens=False
            )
        else:
            self.tool_call_start_token_ids = []
            self.tool_call_end_token_ids = []

        self.buffered_delta_text = ""

    def _parse_arguments(self, args_str: str) -> dict:
        """Parse FunctionGemma argument string into a dictionary."""
        arguments = {}
        if not args_str:
            return arguments

        matches = self.arg_regex.findall(args_str)
        for key, value in matches:
            try:
                parsed_value = json.loads(value)
                arguments[key] = parsed_value
            except json.JSONDecodeError:
                arguments[key] = value

        return arguments

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = self.tool_call_regex.findall(model_output)

            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []

            for match in matches:
                func_name = match[0] if match[0] else match[2]
                args_str = match[1] if match[1] else match[3]

                if not func_name:
                    continue

                arguments = self._parse_arguments(args_str)

                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

            if tool_calls:
                content_end = model_output.find(self.tool_call_start_token)
                content = (
                    model_output[:content_end].strip() if content_end > 0 else None
                )

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        except Exception:
            logger.exception("Error extracting tool calls from FunctionGemma response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Buffer incoming delta text to handle multi-token special sequences."""
        potential_start = "<start_function_call>"
        potential_end = "<end_function_call>"

        combined = self.buffered_delta_text + delta_text

        if combined.endswith(potential_start) or combined.endswith(potential_end):
            self.buffered_delta_text = ""
            return combined

        for tag in [potential_start, potential_end]:
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-(i):]
                    return combined[:-i]

        self.buffered_delta_text = ""
        return combined

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
        delta_text = self._buffer_delta_text(delta_text)
        current_text = previous_text + delta_text

        if self.tool_call_start_token not in current_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            start_count = current_text.count(self.tool_call_start_token)
            end_count = current_text.count(self.tool_call_end_token)
            prev_start_count = previous_text.count(self.tool_call_start_token)
            prev_end_count = previous_text.count(self.tool_call_end_token)

            if self.tool_call_start_token not in current_text:
                return DeltaMessage(content=delta_text)

            # Starting a new function call
            if start_count > prev_start_count and start_count > end_count:
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})
                logger.debug("Starting new tool call %d", self.current_tool_id)
                return None

            # In the middle of a function call
            if start_count > end_count:
                last_start = current_text.rfind(self.tool_call_start_token)
                partial_call = current_text[
                    last_start + len(self.tool_call_start_token) :
                ]

                if partial_call.startswith("call:"):
                    func_part = partial_call[5:]

                    if "{" in func_part:
                        func_name = func_part.split("{")[0]
                        args_part = (
                            func_part.split("{", 1)[1] if "{" in func_part else ""
                        )

                        if not self.current_tool_name_sent and func_name:
                            self.current_tool_name_sent = True
                            self.prev_tool_call_arr[self.current_tool_id] = {
                                "name": func_name,
                                "arguments": {},
                            }
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        type="function",
                                        id=make_tool_call_id(),
                                        function=DeltaFunctionCall(
                                            name=func_name
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )

                        if self.current_tool_name_sent and args_part:
                            current_args = self._parse_arguments(args_part)
                            if current_args:
                                current_args_json = json.dumps(
                                    current_args, ensure_ascii=False
                                )
                                prev_streamed = self.streamed_args_for_tool[
                                    self.current_tool_id
                                ]

                                if len(current_args_json) > len(prev_streamed):
                                    diff = current_args_json[len(prev_streamed) :]
                                    self.streamed_args_for_tool[
                                        self.current_tool_id
                                    ] = current_args_json
                                    self.prev_tool_call_arr[self.current_tool_id][
                                        "arguments"
                                    ] = current_args

                                    return DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                index=self.current_tool_id,
                                                function=DeltaFunctionCall(
                                                    arguments=diff
                                                ).model_dump(exclude_none=True),
                                            )
                                        ]
                                    )

                return None

            # Function call just ended
            if end_count > prev_end_count:
                if self.current_tool_id >= 0 and self.current_tool_id < len(
                    self.prev_tool_call_arr
                ):
                    all_calls = self.tool_call_regex.findall(current_text)
                    args = {}
                    if self.current_tool_id < len(all_calls):
                        match = all_calls[self.current_tool_id]
                        if match[0]:
                            args_str = match[1]
                            args = self._parse_arguments(args_str)
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = args

                    if args:
                        args_json = json.dumps(args, ensure_ascii=False)
                        prev_streamed = self.streamed_args_for_tool[
                            self.current_tool_id
                        ]
                        if len(args_json) > len(prev_streamed):
                            diff = args_json[len(prev_streamed) :]
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                args_json
                            )
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=diff
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                return None

            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        except Exception:
            logger.exception("Error in streaming tool call extraction")
            return None
