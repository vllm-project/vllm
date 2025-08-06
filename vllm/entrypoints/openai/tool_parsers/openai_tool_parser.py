# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence

from openai_harmony import Role, StreamableParser

from vllm.entrypoints.harmony_utils import (get_encoding,
                                            parse_output_into_messages)
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("openai")
class OpenAIToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.encoding = get_encoding()
        self.stream_parser: StreamableParser | None = None
        self.tool_calls_info: list[dict] = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError(
            "OpenAIToolParser requires token IDs and does not support "
            "text-based extraction.")

    def extract_tool_calls_from_ids(
            self, token_ids: Sequence[int],
            ) -> ExtractedToolCallInformation:
        parser = parse_output_into_messages(token_ids)
        tool_calls = []
        reasoning_content = None
        final_content = None

        if len(parser.messages) > 0:
            if parser.messages[0].channel == "analysis":
                reasoning_content = parser.messages[0].content[0].text
            elif parser.messages[0].channel == "final":
                final_content = parser.messages[0].content[0].text

        if len(parser.messages) > 1:
            for msg in parser.messages[1:]:
                if msg.recipient and msg.recipient.startswith("functions."):
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=msg.recipient.split("functions.")[1],
                                arguments=msg.content[0].text),
                        ))
                elif msg.channel == "final":
                    final_content = msg.content[0].text

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=final_content,
            reasoning_content=reasoning_content)

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
        if self.stream_parser is None:
            self.stream_parser = StreamableParser(self.encoding,
                                                  role=Role.ASSISTANT)
            self.tool_calls_info = []

        prev_recipient = self.stream_parser.current_recipient
        for token_id in delta_token_ids:
            self.stream_parser.process(token_id)

        delta_message = None
        if self.stream_parser.current_channel == "analysis":
            delta_message = DeltaMessage(
                reasoning_content=self.stream_parser.last_content_delta)
        elif self.stream_parser.current_channel == "final":
            delta_message = DeltaMessage(
                content=self.stream_parser.last_content_delta)
        elif (self.stream_parser.current_channel == "commentary"
              and self.stream_parser.current_recipient
              and self.stream_parser.current_recipient.startswith(
                  "functions.")):
            if self.stream_parser.current_recipient != prev_recipient:
                # New tool call
                tool_info = {
                    "name":
                    self.stream_parser.current_recipient.split("functions.")[1],
                    "args": "",
                }
                self.tool_calls_info.append(tool_info)
                delta_message = DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=len(self.tool_calls_info) - 1,
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_info["name"], arguments=""))
                ])
            elif self.stream_parser.last_content_delta:
                # Arguments for current tool call
                current_tool_index = len(self.tool_calls_info) - 1
                if current_tool_index >= 0:
                    self.tool_calls_info[current_tool_index][
                        "args"] += self.stream_parser.last_content_delta
                    delta_message = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=current_tool_index,
                            function=DeltaFunctionCall(
                                arguments=self.stream_parser.last_content_delta)
                        )
                    ])

        return delta_message
