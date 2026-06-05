# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Any

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.entrypoints.serve.utils.constants import MCP_PREFIX
from vllm.outputs import CompletionOutput
from vllm.parser.abstract_parser import Parser
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


class ResponsesParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        parser_cls: type[Parser] | None,
        response_messages: list[ResponseInputOutputItem],
        request: ResponsesRequest,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
    ):
        self.response_messages: list[ResponseInputOutputItem] = (
            # TODO: initial messages may not be properly typed
            response_messages
        )
        self.num_init_messages = len(response_messages)
        self.tokenizer = tokenizer
        self.request = request

        self.parser_instance: Parser | None = None
        if parser_cls is not None:
            chat_template_kwargs = _effective_chat_template_kwargs(
                request,
                chat_template=chat_template,
                chat_template_content_format=chat_template_content_format,
            )

            self.parser_instance = parser_cls(
                tokenizer,
                tools=request.tools,
                chat_template_kwargs=chat_template_kwargs,
            )

        self.enable_auto_tools = enable_auto_tools
        self.tool_call_id_type = tool_call_id_type

        # Store the last finish_reason to determine response status
        self.finish_reason: str | None = None

    def process(self, output: CompletionOutput) -> "ResponsesParser":
        # Store the finish_reason from the output
        self.finish_reason = output.finish_reason

        if self.parser_instance is not None:
            output_items = self.parser_instance.extract_response_outputs(
                model_output=output.text,
                model_output_token_ids=output.token_ids,
                request=self.request,
                enable_auto_tools=self.enable_auto_tools,
                tool_call_id_type=self.tool_call_id_type,
            )
            self.response_messages.extend(output_items)
        else:
            # No parser configured, treat entire output as text content
            if output.text:
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
                                text=output.text,
                                logprobs=None,  # TODO
                            )
                        ],
                    )
                )

        return self

    def make_response_output_items_from_parsable_context(
        self,
    ) -> list[ResponseOutputItem]:
        """Given a list of sentences, construct ResponseOutput Items."""
        response_messages = self.response_messages[self.num_init_messages :]
        output_messages: list[ResponseOutputItem] = []
        for message in response_messages:
            if not isinstance(message, ResponseFunctionToolCallOutputItem):
                output_messages.append(message)
            else:
                if len(output_messages) == 0:
                    raise ValueError(
                        "Cannot have a FunctionToolCallOutput before FunctionToolCall."
                    )
                if isinstance(output_messages[-1], ResponseFunctionToolCall):
                    mcp_message = McpCall(
                        id=f"{MCP_PREFIX}{random_uuid()}",
                        arguments=output_messages[-1].arguments,
                        name=output_messages[-1].name,
                        server_label=output_messages[
                            -1
                        ].name,  # TODO: store the server label
                        type="mcp_call",
                        status="completed",
                        output=message.output,
                        # TODO: support error output
                    )
                    output_messages[-1] = mcp_message

        return output_messages


def get_responses_parser_for_simple_context(
    *,
    tokenizer: TokenizerLike,
    parser_cls: type[Parser] | None,
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
    chat_template: str | None,
    chat_template_content_format: ChatTemplateContentFormatOption,
    enable_auto_tools: bool = False,
    tool_call_id_type: str = "random",
) -> ResponsesParser:
    """Factory function to create a ResponsesParser with
    optional unified parser.

    Returns:
        ResponsesParser instance configured with the provided parser
    """
    return ResponsesParser(
        tokenizer=tokenizer,
        parser_cls=parser_cls,
        response_messages=response_messages,
        request=request,
        chat_template=chat_template,
        chat_template_content_format=chat_template_content_format,
        enable_auto_tools=enable_auto_tools,
        tool_call_id_type=tool_call_id_type,
    )


def _effective_chat_template_kwargs(
    request: ResponsesRequest,
    chat_template: str | None,
    chat_template_content_format: ChatTemplateContentFormatOption,
) -> dict[str, Any]:
    return request.build_chat_params(
        default_template=chat_template,
        default_template_content_format=chat_template_content_format,
    ).chat_template_kwargs
