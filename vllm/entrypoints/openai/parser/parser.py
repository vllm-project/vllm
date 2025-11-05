# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Content,
    ResponseReasoningItem,
)

from vllm.entrypoints.openai.protocol import ResponseInputOutputItem, ResponsesRequest
from vllm.outputs import CompletionOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

logger = logging.getLogger(__name__)


class ResponseParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer,
        reasoning_parser: ReasoningParser,
        response_messages: list[ResponseInputOutputItem],
        request: ResponsesRequest,
        tool_parser_cls,
    ):
        self.response_messages: list[ResponseInputOutputItem] = (
            # TODO: initial messages may not be properly typed
            response_messages
        )
        self.num_init_messages = len(response_messages)
        self.tokens: list[int] = []
        self.tokenizer = tokenizer
        self.request = request

        # Initialize reasoning parser instance if provided
        self.reasoning_parser_instance = reasoning_parser(tokenizer)
        self.tool_parser_instance = tool_parser_cls(tokenizer)

    def process(self, output: CompletionOutput) -> "ResponseParser":
        reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
            output.text, request=None
        )
        if reasoning_content:
            # HACK
            self.response_messages.append(
                ResponseReasoningItem(
                    type="reasoning",
                    id="temp",
                    summary=[],
                    content=[
                        Content(
                            type="reasoning_text",
                            text=reasoning_content,
                        )
                    ],
                )
            )

        function_calls: list[ResponseFunctionToolCall] = []
        tool_call_info = self.tool_parser_instance.extract_tool_calls(
            content if content is not None else "",
            request=self.request,  # type: ignore
        )
        if tool_call_info is not None and tool_call_info.tools_called:
            # extract_tool_calls() returns a list of tool calls.
            function_calls.extend(
                ResponseFunctionToolCall(
                    id="fc_lol",
                    call_id="call_lol",
                    type="function_call",
                    status="completed",
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in tool_call_info.tool_calls
            )
            content = tool_call_info.content
            if content and content.strip() == "":
                content = None

        if content:
            self.response_messages.append(
                ResponseOutputMessage(
                    type="message",
                    id="lol",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text", text=content, annotations=[]
                        )
                    ],
                )
            )
        if len(function_calls) > 0:
            self.response_messages.extend(function_calls)

        return self


def get_streamable_parser_for_simple_context(
    *,
    tokenizer,
    reasoning_parser: ReasoningParser,
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
    tool_parser_cls,
) -> ResponseParser:
    """Factory function to create a ResponseParser with
    optional reasoning parser.

    Args:
        tokenizer: The tokenizer to use for decoding tokens
        reasoning_parser: Optional reasoning parser class (e.g., MiniMaxM2ReasoningParser)

    Returns:
        ResponseParser instance configured with the provided parser
    """
    return ResponseParser(
        tokenizer=tokenizer,
        reasoning_parser=reasoning_parser,
        response_messages=response_messages,
        request=request,
        tool_parser_cls=tool_parser_cls,
    )


# def render_parser_for_completion():


"""
TODO:
how to figure out which tokens are special tokens

system
tool
ai
"""
