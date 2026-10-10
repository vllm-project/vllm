# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any, Literal

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.logger import init_logger

from .item_tracker import AsyncMultiModalItemTracker, MultiModalItemTracker
from .parse_chat_message_content import _parse_chat_message_content
from .types import ChatCompletionMessageParam, ConversationMessage

logger = init_logger(__name__)


# After resolving "auto"
ChatTemplateContentFormat = Literal["string", "openai"]


def _postprocess_messages(messages: list[ConversationMessage]) -> None:
    # per the Transformers docs & maintainers, tool call arguments in
    # assistant-role messages with tool_calls need to be dicts not JSON str -
    # this is how tool-use chat templates will expect them moving forwards
    # so, for messages that have tool_calls, parse the string (which we get
    # from openAI format) to dict
    for message in messages:
        if message["role"] == "assistant" and "tool_calls" in message:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            if len(tool_calls) == 0:
                # Drop empty tool_calls to keep templates on the normal assistant path.
                message.pop("tool_calls", None)
                continue

            for item in tool_calls:
                if not isinstance(item, dict):
                    raise VLLMValidationError(
                        "assistant tool_calls entries must be objects.",
                        parameter="tool_calls",
                    )

                function = item.get("function")
                if item.get("type", "function") != "function" or not isinstance(
                    function, dict
                ):
                    raise VLLMValidationError(
                        "chat completions only support assistant tool_calls "
                        "of type 'function'.",
                        parameter="tool_calls",
                    )

                # if arguments is None or empty string, set to {}
                if content := function.get("arguments"):
                    if isinstance(content, dict):
                        parsed = content
                    else:
                        if isinstance(content, str):
                            try:
                                parsed = json.loads(content)
                            except json.JSONDecodeError:
                                # A malformed `arguments` string lives in
                                # conversation history, so failing the request
                                # here would fail every subsequent turn too and
                                # leave the conversation unrecoverable. Coerce
                                # to an empty object so the turn can proceed.
                                logger.warning(
                                    "Tool call %r has arguments that are not valid "
                                    "JSON (%d chars); coercing to an empty object "
                                    "so the conversation can continue.",
                                    function.get("name"),
                                    len(content),
                                )
                                parsed = None
                        else:
                            parsed = content

                        if not isinstance(parsed, dict):
                            if parsed is not None:
                                # Valid JSON, but not an object (e.g. "[]",
                                # "42", "true").
                                # Chat templates require a mapping.
                                logger.warning(
                                    "Tool call %r arguments decoded to %s, not a "
                                    "JSON object; coercing to an empty object.",
                                    function.get("name"),
                                    type(parsed).__name__,
                                )
                            parsed = {}

                    function["arguments"] = parsed
                else:
                    function["arguments"] = {}


def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[
    list[ConversationMessage],
    MultiModalDataDict | None,
    MultiModalUUIDDict | None,
]:
    conversation: list[ConversationMessage] = []
    mm_tracker = MultiModalItemTracker(
        model_config,
        media_io_kwargs=media_io_kwargs,
    )

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            mm_tracker,
            content_format,
            interleave_strings=(
                content_format == "string"
                and model_config.multimodal_config is not None
                and model_config.multimodal_config.interleave_mm_strings
            ),
            mm_processor_kwargs=mm_processor_kwargs,
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    mm_data, mm_uuids = mm_tracker.resolve_items()

    return conversation, mm_data, mm_uuids


async def parse_chat_messages_async(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[
    list[ConversationMessage],
    MultiModalDataDict | None,
    MultiModalUUIDDict | None,
]:
    conversation: list[ConversationMessage] = []
    mm_tracker = AsyncMultiModalItemTracker(
        model_config,
        media_io_kwargs=media_io_kwargs,
    )

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            mm_tracker,
            content_format,
            interleave_strings=(
                content_format == "string"
                and model_config.multimodal_config is not None
                and model_config.multimodal_config.interleave_mm_strings
            ),
            mm_processor_kwargs=mm_processor_kwargs,
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    mm_data, mm_uuids = await mm_tracker.resolve_items()

    return conversation, mm_data, mm_uuids
