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

    if not keeps_tool_calls(
        request, enable_auto_tools=enable_auto_tools, parser_cls=parser_cls
    ):
        return ChatMessageFields(reasoning, content, [])
    if _forces_tool_calls(request):
        content = content or ""
    return ChatMessageFields(reasoning, content, _to_tool_calls(tool_calls, request))


def keeps_tool_calls(
    request: ChatCompletionRequest,
    *,
    enable_auto_tools: bool,
    parser_cls: type[Parser] | None,
) -> bool:
    """Whether a chat choice returns the tool calls its parser extracts.

    Shared by the non-streaming and streaming chat paths so both apply
    `tool_choice` the same way. Parsers don't all gate on it themselves:
    `HarmonyParser` ignores `"none"` and `null` is parsed like `"auto"`.

    Args:
        request: Supplies `tools` and `tool_choice`.
        enable_auto_tools: The server's `--enable-auto-tool-choice`.
        parser_cls: The server's parser class, if any.

    Returns:
        True for `"required"`, a named tool, or `"auto"` with tools and auto
        tool choice enabled.

    """
    if _forces_tool_calls(request):
        return True
    auto_tools = (
        enable_auto_tools
        and parser_cls is not None
        and parser_cls.tool_parser_cls is not None
    )
    if not auto_tools or request.tool_choice != "auto":
        return False
    if not request.tools:
        logger.error(
            "Cannot determine if tools should be extracted. "
            "Returning a standard chat completion."
        )
        return False
    return True


def _forces_tool_calls(request: ChatCompletionRequest) -> bool:
    return (
        type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
        or request.tool_choice == "required"
    )


def _to_tool_calls(
    tool_calls: list[FunctionCall] | None, request: ChatCompletionRequest
) -> list[ToolCall]:
    return [
        ToolCall(id=tc.id or make_tool_call_id(), function=tc)
        for tc in maybe_filter_parallel_tool_calls(tool_calls or [], request)
    ]
