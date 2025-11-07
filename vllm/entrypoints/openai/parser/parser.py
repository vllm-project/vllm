# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from vllm.entrypoints.chat_utils import CustomChatCompletionMessageParam
from vllm.entrypoints.openai.protocol import FunctionCall, ResponsesRequest
from vllm.outputs import CompletionOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

logger = logging.getLogger(__name__)


class StreamableParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer,
        reasoning_parser: ReasoningParser,
        chat_completion_messages: list[CustomChatCompletionMessageParam],
        request: ResponsesRequest,
        tool_parser_cls,  #: Callable[[AnyTokenizer], ToolParser]
    ):
        self.chat_completion_messages: list[CustomChatCompletionMessageParam] = (
            chat_completion_messages
        )
        self.tokens: list[int] = []
        self.tokenizer = tokenizer
        self.request = request

        # Initialize reasoning parser instance if provided
        self.reasoning_parser_instance = reasoning_parser(tokenizer)
        self.tool_parser_instance = tool_parser_cls(tokenizer)

        # start like this
        self.current_role = "assistant"
        # self.current_sentence = Sentence(
        #     author=Author(role=self.current_role), content=[]
        # )
        self.current_chat_completion_message = CustomChatCompletionMessageParam(
            role=self.current_role, content=[]
        )
        self.current_channel = "think"
        self.current_text = ""

    def render_for_completion(self):
        """TODO: Maybe this can be the chat template
        to help generate the initial prompt?"""
        pass

    def process(self, output: CompletionOutput) -> "StreamableParser":
        reasoning_content, content = (
            self.reasoning_parser_instance.extract_reasoning_content(
                output.text, request=None
            )
        )
        if reasoning_content:
            new_content = ResponseReasoningTextContent(
                text=reasoning_content, type="reasoning_text"
            )

            self.current_chat_completion_message["content"].append(new_content)

        function_calls = []
        tool_call_info = self.tool_parser_instance.extract_tool_calls(
            content if content is not None else "",
            request=self.request,  # type: ignore
        )
        if tool_call_info is not None and tool_call_info.tools_called:
            # extract_tool_calls() returns a list of tool calls.
            function_calls.extend(
                FunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in tool_call_info.tool_calls
            )
            content = tool_call_info.content

        if content:
            new_content = ChatCompletionContentPartTextParam(text=content, type="text")
            self.current_chat_completion_message["content"].append(new_content)
        if len(function_calls) > 0:
            self.current_chat_completion_message["content"].extend(function_calls)

        self.chat_completion_messages.append(self.current_chat_completion_message)
        # if len(function_calls) > 0:
        # TODO: add a tool call to the parser

        return self


def get_streamable_parser_for_simple_context(
    *,
    tokenizer,
    reasoning_parser: ReasoningParser,
    chat_completion_messages: list[CustomChatCompletionMessageParam],
    request: ResponsesRequest,
    tool_parser_cls,
) -> StreamableParser:
    """Factory function to create a StreamableParser with optional reasoning parser.

    Args:
        tokenizer: The tokenizer to use for decoding tokens
        reasoning_parser: Optional reasoning parser class (e.g., MiniMaxM2ReasoningParser)

    Returns:
        StreamableParser instance configured with the provided parser
    """
    return StreamableParser(
        tokenizer=tokenizer,
        reasoning_parser=reasoning_parser,
        chat_completion_messages=chat_completion_messages,
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
