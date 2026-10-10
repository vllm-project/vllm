# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from collections.abc import Iterable
from functools import partial
from typing import Any, Final, Union, cast, get_args, get_origin

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from PIL import Image

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from .content_parser import (
    MODALITY_PLACEHOLDERS_MAP,
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    BaseMultiModalContentParser,
)
from .item_tracker import BaseMultiModalItemTracker
from .mm_parser_map import MM_PARSER_MAP, _ContentPart, _InputAudioParser
from .multimodal import _get_full_multimodal_text_prompt
from .parse_chat_message import ChatTemplateContentFormat
from .types import (
    ChatCompletionContentPartAudioEmbedsParam,
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartPromptEmbedsParam,
    ChatCompletionContentPartVideoEmbedsParam,
    ChatCompletionMessageParam,
    ConversationMessage,
    CustomChatCompletionContentPILImageParam,
    CustomChatCompletionContentSimpleAudioParam,
    CustomChatCompletionContentSimpleImageParam,
    CustomChatCompletionContentSimpleVideoParam,
    CustomChatCompletionContentToolReferenceParam,
    MultiModalEmbedsPayload,
)

logger = init_logger(__name__)


_RESERVED_PLACEHOLDER_IN_TEXT_ERROR: Final[str] = (
    "Text content may not contain the reserved placeholder {token!r}. "
    "This placeholder is used internally to mark `prompt_embeds` splice "
    "positions in the tokenized prompt."
)

_PROMPT_EMBEDS_MISSING_DATA_ERROR: Final[str] = (
    "prompt_embeds content part requires a non-empty `data` field "
    "with base64-encoded tensor bytes."
)


def _reject_reserved_placeholder_in_text(text: str, model_config: ModelConfig) -> None:
    """Reject user-supplied text parts that contains the reserved `prompt_embeds`
    placeholder sentinel.

    When the server accepts `prompt_embeds`, the placeholder token is
    registered as a single unsplittable special token on the tokenizer. Any
    user text that happens to contain the literal sequence would tokenize to
    the same ID and be mistaken for a splice point by the renderer, letting a
    caller move or inject splice positions via plain text content.
    """
    if model_config.enable_prompt_embeds and PROMPT_EMBEDS_PLACEHOLDER_TOKEN in text:
        raise VLLMValidationError(
            _RESERVED_PLACEHOLDER_IN_TEXT_ERROR.format(
                token=PROMPT_EMBEDS_PLACEHOLDER_TOKEN
            ),
            parameter="messages",
        )


PART_TYPES_TO_SKIP_NONE_CONTENT = (
    "text",
    "refusal",
)

# Content part types parsed as text rather than multimodal data.
TEXT_PART_TYPES = frozenset(
    {"text", "input_text", "output_text", "refusal", "thinking"}
)
# Content part types that carry no multimodal data.
_TEXT_CONTENT_PART_TYPES = TEXT_PART_TYPES | {"tool_reference"}
# Keys that mark a content part as multimodal, whatever its ``type``.
_MEDIA_CONTENT_PART_KEYS = frozenset(MM_PARSER_MAP) - _TEXT_CONTENT_PART_TYPES


def has_non_text_content(messages: Any) -> bool:
    """Whether any message in ``messages`` has a non-text content part.

    Only list content is inspected, so validated chat content that is a
    one-shot iterator is never consumed.
    """
    if not isinstance(messages, list):
        return False
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and (
                any(key in part for key in _MEDIA_CONTENT_PART_KEYS)
                or not isinstance(part_type := part.get("type", "text"), str)
                or part_type not in _TEXT_CONTENT_PART_TYPES
            ):
                return True
    return False


def _collect_known_content_part_fields() -> frozenset[str]:
    fields: set[str] = set()
    stack: list[Any] = [ChatCompletionContentPartParam]
    while stack:
        node = stack.pop()
        if get_origin(node) in (Union, types.UnionType):
            stack.extend(get_args(node))
        elif hasattr(node, "__required_keys__"):
            fields |= node.__required_keys__ | node.__optional_keys__
    return frozenset(fields)


_KNOWN_CONTENT_PART_FIELDS = _collect_known_content_part_fields()


def _collect_extra_fields(part: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in part.items() if k not in _KNOWN_CONTENT_PART_FIELDS}


def _parse_chat_message_content_part(
    part: ChatCompletionContentPartParam,
    mm_parser: BaseMultiModalContentParser,
    *,
    wrap_dicts: bool,
    interleave_strings: bool,
) -> _ContentPart | None:
    """Parses a single part of a conversation. If wrap_dicts is True,
    structured dictionary pieces for texts and images will be
    wrapped in dictionaries, i.e., {"type": "text", "text", ...} and
    {"type": "image"}, respectively. Otherwise multimodal data will be
    handled by mm_parser, and texts will be returned as strings to be joined
    with multimodal placeholders.
    """
    if isinstance(part, str):  # Handle plain text parts
        _reject_reserved_placeholder_in_text(part, mm_parser.model_config)
        if wrap_dicts:
            return {"type": "text", "text": part}
        return part
    # Handle structured dictionary parts
    part_type, content = _parse_chat_message_content_mm_part(part)
    # if part_type is text/refusal/image_url/audio_url/video_url/input_audio but
    # content is None, log a warning and skip
    if part_type in PART_TYPES_TO_SKIP_NONE_CONTENT and content is None:
        logger.warning(
            "Skipping multimodal part '%s' (type: '%s') "
            "with empty / unparsable content.",
            part,
            part_type,
        )
        return None

    if part_type in TEXT_PART_TYPES:
        str_content = cast(str, content)
        _reject_reserved_placeholder_in_text(str_content, mm_parser.model_config)
        if wrap_dicts:
            result: dict[str, Any] = {"type": "text", "text": str_content}
            result.update(_collect_extra_fields(cast(dict[str, Any], part)))
            return result
        else:
            return str_content

    # For media items, if a user has provided one, use it. Otherwise, insert
    # a placeholder empty uuid.
    uuid = part.get("uuid", None)
    if uuid is not None:
        uuid = str(uuid)

    modality = None
    if part_type == "image_pil":
        image_content = cast(Image.Image, content) if content is not None else None
        mm_parser.parse_image_pil(image_content, uuid)
        modality = "image"
    elif part_type in ("image_url", "input_image"):
        str_content = cast(str, content)
        mm_parser.parse_image(str_content, uuid)
        modality = "image"
    elif part_type == "image_embeds":
        content = (
            cast(MultiModalEmbedsPayload, content) if content is not None else None
        )
        mm_parser.parse_image_embeds(content, uuid)
        modality = "image"
    elif part_type == "audio_embeds":
        content = (
            cast(MultiModalEmbedsPayload, content) if content is not None else None
        )
        mm_parser.parse_audio_embeds(content, uuid)
        modality = "audio"
    elif part_type == "video_embeds":
        content = (
            cast(MultiModalEmbedsPayload, content) if content is not None else None
        )
        mm_parser.parse_video_embeds(content, uuid)
        modality = "video"
    elif part_type == "prompt_embeds":
        if not content:
            raise VLLMValidationError(
                _PROMPT_EMBEDS_MISSING_DATA_ERROR, parameter="prompt_embeds"
            )
        mm_parser.parse_prompt_embeds(cast(str, content))
        modality = "prompt_embeds"
    elif part_type == "audio_url":
        str_content = cast(str, content)
        mm_parser.parse_audio(str_content, uuid)
        modality = "audio"
    elif part_type == "input_audio":
        dict_content = cast(InputAudio, content)
        mm_parser.parse_input_audio(dict_content, uuid)
        modality = "audio"
    elif part_type == "video_url":
        str_content = cast(str, content)
        mm_parser.parse_video(str_content, uuid)
        modality = "video"
    elif part_type == "tool_reference":
        # Tool references are not multimodal data — they reference deferred
        # tools and are passed through as-is for the chat template to expand.
        if wrap_dicts:
            return {"type": "tool_reference", "name": cast(str, content)}
        return cast(str, content)
    else:
        supported = sorted(MM_PARSER_MAP.keys() | set(PART_TYPES_TO_SKIP_NONE_CONTENT))
        raise VLLMValidationError(
            f"Unsupported chat content part type: {part_type!r}. "
            f"Supported types: {', '.join(supported)}.",
            parameter="type",
            value=part_type,
        )

    if wrap_dicts:
        if modality == "prompt_embeds":
            # Chat templates don't know about the "prompt_embeds" modality,
            # emit the single sentinel token as text so the template renders
            # it inline. The renderer later expands it to N tokens post-tokenize.
            return {"type": "text", "text": PROMPT_EMBEDS_PLACEHOLDER_TOKEN}
        result = {"type": modality}
        result.update(_collect_extra_fields(cast(dict[str, Any], part)))
        return result
    if modality == "prompt_embeds":
        # Emit the renderer token inline regardless of `interleave_strings`,
        # prompt_embeds are spliced at the token offset so position matters.
        # Falling back to front-padding via `missing_placeholders` would
        # reorder them relative to surrounding text.
        return PROMPT_EMBEDS_PLACEHOLDER_TOKEN
    return MODALITY_PLACEHOLDERS_MAP[modality] if interleave_strings else None


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
    *,
    wrap_dicts: bool,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
    multimodal_content_part_separator="\n",
) -> list[ConversationMessage]:
    content = list[_ContentPart]()

    mm_parser = mm_tracker.create_parser(mm_processor_kwargs=mm_processor_kwargs)

    for part in parts:
        parse_res = _parse_chat_message_content_part(
            part,
            mm_parser,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings,
        )
        if parse_res:
            content.append(parse_res)

    if wrap_dicts:
        # Parsing wraps images and texts as interleaved dictionaries
        return [ConversationMessage(role=role, content=content)]  # type: ignore
    texts = cast(list[str], content)
    mm_placeholder_storage = mm_parser.mm_placeholder_storage()
    if mm_placeholder_storage:
        text_prompt = _get_full_multimodal_text_prompt(
            mm_placeholder_storage,
            texts,
            interleave_strings,
            multimodal_content_part_separator=multimodal_content_part_separator,
        )
    else:
        text_prompt = "\n".join(texts)

    return [ConversationMessage(role=role, content=text_prompt)]


# No need to validate using Pydantic again
_AssistantParser = partial(cast, ChatCompletionAssistantMessageParam)
_ToolParser = partial(cast, ChatCompletionToolMessageParam)


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_tracker: BaseMultiModalItemTracker,
    content_format: ChatTemplateContentFormat,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[ConversationMessage]:
    role = message["role"]
    content = message.get("content")
    reasoning = message.get("reasoning")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [ChatCompletionContentPartTextParam(type="text", text=content)]
    if role == "system":
        content = _strip_claude_code_billing_header(content)  # type: ignore[arg-type]
    result = _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        mm_tracker,
        wrap_dicts=(content_format == "openai"),
        interleave_strings=interleave_strings,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    for result_msg in result:
        if role == "assistant":
            parsed_msg = _AssistantParser(message)

            # The 'tool_calls' is not None check ensures compatibility.
            # It's needed only if downstream code doesn't strictly
            # follow the OpenAI spec.
            if "tool_calls" in parsed_msg and parsed_msg["tool_calls"] is not None:
                result_msg["tool_calls"] = list(parsed_msg["tool_calls"])
            # Include reasoning if present for interleaved thinking.
            if reasoning is not None:
                result_msg["reasoning"] = cast(str, reasoning)
                result_msg["reasoning_content"] = cast(
                    str, reasoning
                )  # keep compatibility
        elif role == "tool":
            parsed_msg = _ToolParser(message)
            if "tool_call_id" in parsed_msg:
                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]
            # Normalize tool message content from OpenAI array format to plain
            # string. Clients like Claude Code / Cursor send tool results as
            # [{"type": "text", "text": "..."}], but most chat templates only
            # handle string content for tool messages.
            # However, tool_reference items must be preserved as structured
            # dicts for the chat template to expand them.
            msg_content = result_msg.get("content")
            if isinstance(msg_content, list):
                has_non_text = any(
                    isinstance(item, dict) and item.get("type") != "text"
                    for item in msg_content
                )
                if has_non_text:
                    # Keep structured content (e.g., tool_reference)
                    result_msg["content"] = msg_content
                else:
                    texts = [
                        item.get("text", "")
                        for item in msg_content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    result_msg["content"] = "\n".join(texts) if texts else ""

        if "name" in message and isinstance(message["name"], str):
            result_msg["name"] = message["name"]

        if "task" in message and isinstance(message["task"], str):
            result_msg["task"] = message["task"]

        if role == "developer":
            result_msg["tools"] = message.get("tools", None)
    return result


def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam,
) -> tuple[str, _ContentPart]:
    """Parses a given multi-modal content part based on its type.

    Args:
        part: A dict containing the content part, with a potential 'type' field.

    Returns:
        A tuple (part_type, content) where:
        - part_type: Type of the part (e.g., 'text', 'image_url').
        - content: Parsed content (e.g., text, image URL).

    Raises:
        ValueError: If the 'type' field is missing and no direct URL is found.

    """
    assert isinstance(
        part, dict
    )  # This is needed to avoid mypy errors: part.get() from str
    part_type = part.get("type", None)
    uuid = part.get("uuid", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP and uuid is None:  # noqa: E501
        content = MM_PARSER_MAP[part_type](part)

        # Special case for 'image_url.detail'
        # We only support 'auto', which is the default
        if part_type == "image_url" and part.get("detail", "auto") != "auto":
            logger.warning(
                "'image_url.detail' is currently not supported and will be ignored."
            )

        return part_type, content

    # Handle missing 'type' but provided direct URL fields.
    # 'type' is required field by pydantic
    if part_type is None or uuid is not None:
        if "image_url" in part:
            image_params = cast(CustomChatCompletionContentSimpleImageParam, part)
            image_url = image_params.get("image_url", None)
            if isinstance(image_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                image_url = image_url.get("url", None)
            return "image_url", image_url
        if "image_pil" in part:
            # "image_pil" could be None if UUID is provided.
            image_params = cast(  # type: ignore
                CustomChatCompletionContentPILImageParam, part
            )
            image_pil = image_params.get("image_pil", None)
            return "image_pil", image_pil
        if "image_embeds" in part:
            # "image_embeds" could be None if UUID is provided.
            image_params = cast(  # type: ignore
                ChatCompletionContentPartImageEmbedsParam, part
            )
            image_embeds = image_params.get("image_embeds", None)
            return "image_embeds", image_embeds
        if "audio_embeds" in part:
            # "audio_embeds" could be None if UUID is provided.
            audio_params = cast(  # type: ignore[assignment]
                ChatCompletionContentPartAudioEmbedsParam, part
            )
            audio_embeds = audio_params.get("audio_embeds", None)
            return "audio_embeds", audio_embeds
        if "video_embeds" in part:
            # "video_embeds" could be None if UUID is provided.
            video_embeds_params = cast(  # type: ignore[assignment]
                ChatCompletionContentPartVideoEmbedsParam, part
            )
            video_embeds = video_embeds_params.get("video_embeds", None)
            return "video_embeds", video_embeds
        if "prompt_embeds" in part:
            prompt_embeds_params = cast(  # type: ignore[assignment]
                ChatCompletionContentPartPromptEmbedsParam, part
            )
            return "prompt_embeds", prompt_embeds_params.get("data", None)
        if "audio_url" in part:
            audio_params = cast(  # type: ignore[assignment]
                CustomChatCompletionContentSimpleAudioParam, part
            )
            audio_url = audio_params.get("audio_url", None)
            if isinstance(audio_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                audio_url = audio_url.get("url", None)
            return "audio_url", audio_url
        if "input_audio" in part:
            input_audio_params = _InputAudioParser(part).get("input_audio", None)
            return "input_audio", input_audio_params
        if "video_url" in part:
            video_params = cast(CustomChatCompletionContentSimpleVideoParam, part)
            video_url = video_params.get("video_url", None)
            if isinstance(video_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                video_url = video_url.get("url", None)
            return "video_url", video_url
        if "tool_reference" in part:
            tool_reference_params = cast(
                CustomChatCompletionContentToolReferenceParam, part
            )
            tool_reference = tool_reference_params.get("name", None)
            return "tool_reference", tool_reference
        # Raise an error if no 'type' or direct URL is found.
        raise VLLMValidationError(
            "Missing 'type' field in multimodal part.", parameter="type"
        )

    if not isinstance(part_type, str):
        raise VLLMValidationError(
            "Invalid 'type' field in multimodal part.", parameter="type"
        )
    return part_type, "unknown part_type content"


_CLAUDE_CODE_BILLING_HEADER = "x-anthropic-billing-header"


def _strip_claude_code_billing_header(
    parts: Iterable[ChatCompletionContentPartParam],
) -> list[ChatCompletionContentPartParam]:
    """Drop Claude Code's attribution header line from system text parts.

    Its ``cch`` value changes on every request, so a prompt that keeps it
    misses the prefix cache from the header on. Gateways that translate
    Anthropic requests to chat completions keep it; ``/v1/messages`` drops
    the same block.
    """
    out: list[ChatCompletionContentPartParam] = []
    for part in parts:
        if isinstance(part, str):
            text: object = part
        elif isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
        else:
            text = None
        if not (isinstance(text, str) and text.startswith(_CLAUDE_CODE_BILLING_HEADER)):
            out.append(part)
            continue
        rest = text.partition("\n")[2]
        if not rest:
            continue
        out.append(
            cast(ChatCompletionContentPartParam, rest)
            if isinstance(part, str)
            else cast(ChatCompletionContentPartParam, {**part, "text": rest})
        )
    return out
