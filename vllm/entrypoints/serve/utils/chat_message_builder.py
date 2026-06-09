# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import ToolCall
from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

logger = init_logger(__name__)


def build_chat_message(
    *,
    output_text: str,
    output_token_ids: Sequence[int],
    request: ChatCompletionRequest,
    parser: Parser | None,
    tool_parser: type[ToolParser] | None,
    use_harmony: bool,
    enable_auto_tools: bool,
    tokenizer: TokenizerLike,
    role: str,
    tool_call_id_type: str,
    history_tool_call_cnt: int,
) -> tuple[ChatMessage, bool, int]:
    """Build a ChatMessage from parsed output, handling harmony and all
    tool_choice modes.

    Single source of truth for output parsing shared by both the coupled chat
    path (chat_completion_full_generator) and /derender.

    Args:
        output_text: Decoded text of the model output.
        output_token_ids: Token IDs of the model output.
        request: The original (post adjust_request) ChatCompletionRequest.
        parser: Instantiated Parser for reasoning+tool extraction (non harmony).
        tool_parser: Tool parser class for harmony path (instantiated here).
        use_harmony: Whether the model uses the GPT OSS harmony protocol.
        enable_auto_tools: Whether auto tool choice is enabled server side.
        tokenizer: Tokenizer for tool extraction and ID type detection.
        role: Role for the output ChatMessage (usually "assistant").
        tool_call_id_type: ID generation type ("random" or "kimi_k2").
        history_tool_call_cnt: Running count of tool calls seen so far.

    Returns:
        Tuple of (message, auto_tools_called, updated_history_tool_call_cnt).
        auto_tools_called is True when auto or harmony tool calls were generated,
        signalling the finish_reason should be "tool_calls".
    """
    if use_harmony:
        if tokenizer is None:
            raise ValueError("Tokenizer not available when skip_tokenizer_init=True")
        reasoning, content, _ = parse_chat_output(output_token_ids)
        if not request.include_reasoning:
            reasoning = None

        auto_tools_called = False
        if tool_parser is not None:
            tp_instance = tool_parser(tokenizer, request.tools)
            tool_call_info = tp_instance.extract_tool_calls(
                "",
                request=request,
                token_ids=output_token_ids,  # type: ignore[call-arg]
            )
            content = tool_call_info.content
            auto_tools_called = tool_call_info.tools_called
            message = ChatMessage(
                role=role,
                reasoning=reasoning,
                content=content,
                tool_calls=tool_call_info.tool_calls,
            )
        else:
            message = ChatMessage(role=role, reasoning=reasoning, content=content)
        return message, auto_tools_called, history_tool_call_cnt

    # Non-harmony path.
    if parser is not None:
        reasoning, content, tool_calls = parser.parse(
            output_text,
            request,
            enable_auto_tools=enable_auto_tools,
        )
        if not request.include_reasoning:
            reasoning = None
    else:
        reasoning = None
        content = output_text
        tool_calls = []

    auto_tools_called = False
    if is_mistral_tokenizer(tokenizer):
        from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

        tool_call_class: type[ToolCall] = MistralToolCall
    else:
        tool_call_class = ToolCall

    # Re-derive for the derender path. _grammar_from_tool_parser is a
    # PrivateAttr and is excluded from JSON serialization, so it is always
    # False after a round trip. Trust it when set (chat path). Fall back to
    # checking the tool parser class and the grammar so that adjust_request
    # sets. Both are serializable.
    use_mistral_tool_parser = request._grammar_from_tool_parser or (
        is_mistral_tool_parser(tool_parser)
        and request.structured_outputs is not None
        and request.structured_outputs.grammar is not None
    )
    if use_mistral_tool_parser:
        from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

        tool_call_items = MistralToolParser.build_non_streaming_tool_calls(tool_calls)
        if tool_call_items:
            auto_tools_called = (
                request.tool_choice is None or request.tool_choice == "auto"
            )
        message = ChatMessage(
            role=role,
            reasoning=reasoning,
            content=content,
            tool_calls=tool_call_items,
        )

    elif (not enable_auto_tools or not tool_parser) and (
        not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
        and request.tool_choice != "required"
    ):
        message = ChatMessage(role=role, reasoning=reasoning, content=content)

    elif (
        request.tool_choice
        and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
    ):
        tool_call_class_items = []
        tool_calls = tool_calls or []
        for tc in tool_calls:
            if tc.id:
                tool_call_class_items.append(tool_call_class(id=tc.id, function=tc))
            else:
                if is_mistral_tokenizer(tokenizer):
                    tool_call_class_items.append(tool_call_class(function=tc))
                else:
                    generated_id = make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=tc.name,
                        idx=history_tool_call_cnt,
                    )
                    tool_call_class_items.append(
                        tool_call_class(id=generated_id, function=tc)
                    )
            history_tool_call_cnt += 1
        message = ChatMessage(
            role=role,
            reasoning=reasoning,
            content="",
            tool_calls=tool_call_class_items,
        )

    elif request.tool_choice and request.tool_choice == "required":
        tool_call_class_items = []
        tool_calls = tool_calls or []
        for tool_call in tool_calls:
            if tool_call.id:
                tool_call_class_items.append(
                    tool_call_class(id=tool_call.id, function=tool_call)
                )
            else:
                if is_mistral_tokenizer(tokenizer):
                    tool_call_class_items.append(tool_call_class(function=tool_call))
                else:
                    generated_id = make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=tool_call.name,
                        idx=history_tool_call_cnt,
                    )
                    tool_call_class_items.append(
                        tool_call_class(id=generated_id, function=tool_call)
                    )
            history_tool_call_cnt += 1
        message = ChatMessage(
            role=role,
            content="",
            tool_calls=tool_call_class_items,
            reasoning=reasoning,
        )

    elif not request.tool_choice or request.tool_choice == "none":
        message = ChatMessage(role=role, reasoning=reasoning, content=content)

    elif (
        request.tools
        and (request.tool_choice == "auto" or request.tool_choice is None)
        and enable_auto_tools
        and tool_parser
    ):
        auto_tools_called = tool_calls is not None and len(tool_calls) > 0
        if tool_calls:
            tool_call_items = []
            for tc in tool_calls:
                if tc.id:
                    tool_call_items.append(tool_call_class(id=tc.id, function=tc))
                else:
                    if is_mistral_tokenizer(tokenizer):
                        tool_call_items.append(tool_call_class(function=tc))
                    else:
                        generated_id = make_tool_call_id(
                            id_type=tool_call_id_type,
                            func_name=tc.name,
                            idx=history_tool_call_cnt,
                        )
                        tool_call_items.append(
                            tool_call_class(id=generated_id, function=tc)
                        )
                history_tool_call_cnt += 1
            message = ChatMessage(
                role=role,
                reasoning=reasoning,
                content=content,
                tool_calls=tool_call_items,
            )
        else:
            message = ChatMessage(role=role, reasoning=reasoning, content=content)

    else:
        logger.error(
            "Error in build_chat_message. Cannot determine if tools should be "
            "extracted. Returning a standard chat completion."
        )
        message = ChatMessage(role=role, reasoning=reasoning, content=content)

    return message, auto_tools_called, history_tool_call_cnt
