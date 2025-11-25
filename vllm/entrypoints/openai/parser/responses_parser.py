# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

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


class ResponsesParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer,
        reasoning_parser: ReasoningParser,
        response_messages: list[ResponseInputOutputItem],
        request: ResponsesRequest,
    ):
        self.response_messages: list[ResponseInputOutputItem] = (
            # TODO: initial messages may not be properly typed
            response_messages
        )
        self.num_init_messages = len(response_messages)
        self.tokenizer = tokenizer
        self.request = request
        self.reasoning_parser_instance = reasoning_parser(tokenizer)

    def process(self, output: CompletionOutput) -> "ResponsesParser":
        reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
            output.text, request=None
        )
        if reasoning_content:
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

        return self


def get_responses_parser_for_simple_context(
    *,
    tokenizer,
    reasoning_parser: ReasoningParser,
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
) -> ResponsesParser:
    """Factory function to create a ResponsesParser with
    optional reasoning parser.

    Args:
        tokenizer: The tokenizer to use for decoding tokens
        reasoning_parser: Optional reasoning parser class (e.g., MiniMaxM2ReasoningParser)

    Returns:
        ResponsesParser instance configured with the provided parser
    """
    return ResponsesParser(
        tokenizer=tokenizer,
        reasoning_parser=reasoning_parser,
        response_messages=response_messages,
        request=request,
    )
