# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.harmony_utils import (
    get_streamable_parser_for_assistant, parse_output_into_messages)
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer


@ToolParserManager.register_module("openai")
class OpenAIToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support text-based extraction."  # noqa: E501
            )

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
        stream_parser = get_streamable_parser_for_assistant()

        for token_id in previous_token_ids:
            stream_parser.process(token_id)

        # Count previously completed tool calls
        base_index = 0
        for msg in stream_parser.messages:
            if (msg.channel == "commentary" and msg.recipient
                    and msg.recipient.startswith("functions.")):
                base_index += 1

        prev_recipient = stream_parser.current_recipient
        for token_id in delta_token_ids:
            stream_parser.process(token_id)

        delta_message: DeltaMessage | None = None
        if stream_parser.current_channel == "analysis":
            if request.include_reasoning:
                delta_message = DeltaMessage(
                    reasoning_content=stream_parser.last_content_delta)
        elif stream_parser.current_channel == "final":
            delta_message = DeltaMessage(
                content=stream_parser.last_content_delta)
        elif (stream_parser.current_channel == "commentary"
              and stream_parser.current_recipient
              and stream_parser.current_recipient.startswith("functions.")):
            current_recipient = stream_parser.current_recipient
            if current_recipient != prev_recipient:
                tool_name = current_recipient.split("functions.")[1]
                delta_message = DeltaMessage(tool_calls=[
                    DeltaToolCall(index=base_index,
                                  type="function",
                                  function=DeltaFunctionCall(name=tool_name,
                                                             arguments=""))
                ])
            elif stream_parser.last_content_delta:
                delta_message = DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=base_index,
                        function=DeltaFunctionCall(
                            arguments=stream_parser.last_content_delta))
                ])

        return delta_message
