# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

from vllm.entrypoints.generate.base.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser
from vllm.renderers.chat_utils import make_tool_call_id

logger = init_logger(__name__)


class ChatMessageFields(NamedTuple):
    reasoning: str | None
    content: str | None
    tool_calls: list[ToolCall]


def assemble_chat_message_fields(
    request: ChatCompletionRequest,
    reasoning: str | None,
    content: str | None,
    tool_calls: list[FunctionCall] | None,
    *,
    enable_auto_tools: bool,
    parser_cls: type[Parser] | None,
) -> ChatMessageFields:
    """Pick the assistant message fields for one parsed chat choice.

    Shared by `OpenAIServingChat` and batch chat derender so both surface the
    same `reasoning`, `content` and `tool_calls` for the same parser output.

    Args:
        request: Supplies `tools`, `tool_choice`, `include_reasoning` and
            `parallel_tool_calls`.
        reasoning: Reasoning from `Parser.parse`.
        content: Content from `Parser.parse` or the raw text without a parser.
        tool_calls: Tool calls from `Parser.parse`.
        enable_auto_tools: The server's `--enable-auto-tool-choice`.
        parser_cls: The server's parser class, if any.

    Returns:
        The fields for the response `ChatMessage`.

    """
    if not request.include_reasoning:
        reasoning = None

    tool_choice = request.tool_choice
    if (
        type(tool_choice) is ChatCompletionNamedToolChoiceParam
        or tool_choice == "required"
    ):
        return ChatMessageFields(
            reasoning, content or "", _to_tool_calls(tool_calls, request)
        )

    auto_tools = (
        enable_auto_tools
        and parser_cls is not None
        and parser_cls.tool_parser_cls is not None
    )
    if auto_tools and tool_choice == "auto":
        if request.tools:
            return ChatMessageFields(
                reasoning, content, _to_tool_calls(tool_calls, request)
            )
        logger.error(
            "Cannot determine if tools should be extracted. "
            "Returning a standard chat completion."
        )
    return ChatMessageFields(reasoning, content, [])


def _to_tool_calls(
    tool_calls: list[FunctionCall] | None, request: ChatCompletionRequest
) -> list[ToolCall]:
    return [
        ToolCall(id=tc.id or make_tool_call_id(), function=tc)
        for tc in maybe_filter_parallel_tool_calls(tool_calls or [], request)
    ]
