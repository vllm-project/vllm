# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable

from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Content,
    ResponseReasoningItem,
)

from vllm.entrypoints.openai.protocol import ResponseInputOutputItem, ResponsesRequest
from vllm.outputs import CompletionOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


class ResponsesParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer: AnyTokenizer,
        reasoning_parser_cls: Callable[[AnyTokenizer], ReasoningParser],
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

        self.reasoning_parser_instance = reasoning_parser_cls(tokenizer)

    def process(self, output: CompletionOutput) -> "ResponsesParser":
        reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
            output.text, request=self.request
        )
        if reasoning_content:
            self.response_messages.append(
                ResponseReasoningItem(
                    type="reasoning",
                    id=f"rs_{random_uuid()}",
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
                    id=f"msg_{random_uuid()}",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            annotations=[],  # TODO
                            type="output_text",
                            text=content,
                            logprobs=None,  # TODO
                        )
                    ],
                )
            )

        return self


def get_responses_parser_for_simple_context(
    *,
    tokenizer: AnyTokenizer,
    reasoning_parser_cls: Callable[[AnyTokenizer], ReasoningParser],
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
) -> ResponsesParser:
    """Factory function to create a ResponsesParser with
    optional reasoning parser.

    Returns:
        ResponsesParser instance configured with the provided parser
    """
    return ResponsesParser(
        tokenizer=tokenizer,
        reasoning_parser_cls=reasoning_parser_cls,
        response_messages=response_messages,
        request=request,
    )
