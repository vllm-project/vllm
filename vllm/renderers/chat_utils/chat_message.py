# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import types
from collections import Counter
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Final, TypeAlias, Union, cast, get_args, get_origin

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionToolMessageParam,
)
from openai.types.responses import ResponseInputImageParam
from PIL import Image
from pydantic import TypeAdapter

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.logger import init_logger

from .content_parser import (
    MODALITY_PLACEHOLDERS_MAP,
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    BaseMultiModalContentParser,
    make_prompt_embeds_placeholder,
)
from .item_tracker import (
    AsyncMultiModalItemTracker,
    BaseMultiModalItemTracker,
    MultiModalItemTracker,
)
from .types import (
    ChatCompletionContentPartAudioEmbedsParam,
    ChatCompletionContentPartAudioParam,
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartPromptEmbedsParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoEmbedsParam,
    ChatCompletionContentPartVideoParam,
    ChatCompletionMessageParam,
    ChatTemplateContentFormat,
    ConversationMessage,
    CustomChatCompletionContentPILImageParam,
    CustomChatCompletionContentSimpleAudioParam,
    CustomChatCompletionContentSimpleImageParam,
    CustomChatCompletionContentSimpleVideoParam,
    CustomChatCompletionContentToolReferenceParam,
    CustomThinkCompletionContentParam,
    InputAudio,
    MultiModalEmbedsPayload,
    PILImage,
)

logger = init_logger(__name__)


_PROMPT_EMBEDS_MISSING_DATA_ERROR: Final[str] = (
    "prompt_embeds content part requires a non-empty `data` field "
    "with base64-encoded tensor bytes."
)

_RESERVED_PLACEHOLDER_IN_TEXT_ERROR: Final[str] = (
    "Text content may not contain the reserved placeholder {token!r}. "
    "This placeholder is used internally to mark `prompt_embeds` splice "
    "positions in the tokenized prompt."
)


# No need to validate using Pydantic again
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_AudioEmbedsParser = partial(cast, ChatCompletionContentPartAudioEmbedsParam)
_VideoEmbedsParser = partial(cast, ChatCompletionContentPartVideoEmbedsParam)
_PromptEmbedsParser = partial(cast, ChatCompletionContentPartPromptEmbedsParam)
_InputAudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
_RefusalParser = partial(cast, ChatCompletionContentPartRefusalParam)
_PILImageParser = partial(cast, CustomChatCompletionContentPILImageParam)
_ThinkParser = partial(cast, CustomThinkCompletionContentParam)
_AssistantParser = partial(cast, ChatCompletionAssistantMessageParam)
_ToolParser = partial(cast, ChatCompletionToolMessageParam)

# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ResponsesInputImageParser = TypeAdapter(ResponseInputImageParam).validate_python
ContentPart: TypeAlias = (
    MultiModalEmbedsPayload | dict[str, str] | InputAudio | PILImage
)

# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], ContentPart],
] = {
    "text": lambda part: _TextParser(part).get("text", None),
    "thinking": lambda part: _ThinkParser(part).get("thinking", None),
    "input_text": lambda part: _TextParser(part).get("text", None),
    "output_text": lambda part: _TextParser(part).get("text", None),
    "input_image": lambda part: _ResponsesInputImageParser(part).get("image_url", None),
    "image_url": lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "image_embeds": lambda part: _ImageEmbedsParser(part).get("image_embeds", None),
    "audio_embeds": lambda part: _AudioEmbedsParser(part).get("audio_embeds", None),
    "video_embeds": lambda part: _VideoEmbedsParser(part).get("video_embeds", None),
    "prompt_embeds": lambda part: _PromptEmbedsParser(part).get("data", None),
    "image_pil": lambda part: _PILImageParser(part).get("image_pil", None),
    "audio_url": lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    "input_audio": lambda part: _InputAudioParser(part).get("input_audio", None),
    "refusal": lambda part: _RefusalParser(part).get("refusal", None),
    "video_url": lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
    "tool_reference": lambda part: cast(
        CustomChatCompletionContentToolReferenceParam, part
    ).get("name", None),
}


def _get_interleaved_text_prompt(
    placeholder_storage: dict[str, list], texts: list[str]
) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            slots = placeholder_storage[elem]
            if not slots:
                raise VLLMValidationError(
                    f"Found more '{elem}' placeholders in input prompt than "
                    "actual multimodal data items."
                )
            texts[idx] = slots.pop(0)

    return "\n".join(texts)


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def get_full_multimodal_text_prompt(
    placeholder_storage: dict[str, list],
    texts: list[str],
    interleave_strings: bool,
    multimodal_content_part_separator: str = "\n",
) -> str:
    """Combine multimodal prompts for a multimodal language model."""
    # flatten storage to make it looks like
    # {
    #   "<|image|>": 2,
    #   "<|audio|>": 1
    # }
    placeholder_counts = Counter(
        [v for elem in placeholder_storage.values() for v in elem]
    )

    if interleave_strings:
        text_prompt = _get_interleaved_text_prompt(placeholder_storage, texts)
    else:
        text_prompt = "\n".join(texts)

    # Pass interleaved text further in case the user used image placeholders
    # himself, but forgot to disable the 'interleave_strings' flag

    # Look through the text prompt to check for missing placeholders
    missing_placeholders: list[str] = []
    for placeholder in placeholder_counts:
        # For any existing placeholder in the text prompt, we leave it as is
        placeholder_counts[placeholder] -= text_prompt.count(placeholder)

        if placeholder_counts[placeholder] < 0:
            logger.error(
                "Placeholder count is negative! "
                "Ensure that the 'interleave_strings' flag is disabled "
                "(current value: %s) "
                "when manually placing image placeholders.",
                interleave_strings,
            )
            logger.debug("Input prompt: %s", text_prompt)
            raise VLLMValidationError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items.",
                parameter="messages",
            )

        missing_placeholders.extend([placeholder] * placeholder_counts[placeholder])

    # NOTE: Default behaviour: we always add missing placeholders
    # at the front of the prompt, if interleave_strings=False
    if text_prompt:
        return multimodal_content_part_separator.join(
            missing_placeholders + [text_prompt]
        )
    else:
        return multimodal_content_part_separator.join(missing_placeholders)


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


def parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam,
) -> tuple[str, ContentPart]:
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
        if "audio_embeds" in part and part_type != "video_embeds":
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
    """Whether any message has non-text content or reasoning.

    Only list content is inspected, so validated chat content that is a
    one-shot iterator is never consumed.
    """
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if isinstance(msg, dict):
            for field in ("reasoning", "reasoning_content"):
                reasoning = msg.get(field)
                if (
                    isinstance(reasoning, dict)
                    and reasoning.get("type") == "prompt_embeds"
                ):
                    return True
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


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
    *,
    wrap_dicts: bool,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
    multimodal_content_part_separator="\n",
) -> list[ConversationMessage]:
    content = list[ContentPart]()

    mm_parser = mm_tracker.create_parser(mm_processor_kwargs=mm_processor_kwargs)

    for part in parts:
        parse_res = parse_chat_message_content_part(
            part,
            mm_parser,
            mm_tracker=mm_tracker,
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
        text_prompt = get_full_multimodal_text_prompt(
            mm_placeholder_storage,
            texts,
            interleave_strings,
            multimodal_content_part_separator=multimodal_content_part_separator,
        )
    else:
        text_prompt = "\n".join(texts)

    return [ConversationMessage(role=role, content=text_prompt)]


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


def parse_chat_message_content_part(
    part: ChatCompletionContentPartParam,
    mm_parser: BaseMultiModalContentParser,
    *,
    mm_tracker: BaseMultiModalItemTracker,
    wrap_dicts: bool,
    interleave_strings: bool,
) -> ContentPart | None:
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
    part_type, content = parse_chat_message_content_mm_part(part)
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
    prompt_embeds_placeholder = None
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
        audio_embeds = cast(MultiModalEmbedsPayload | None, part.get("audio_embeds"))
        mm_parser.parse_video_embeds(content, uuid, audio_embeds)
        modality = "video"
    elif part_type == "prompt_embeds":
        if not content:
            raise VLLMValidationError(
                _PROMPT_EMBEDS_MISSING_DATA_ERROR, parameter="prompt_embeds"
            )
        prompt_embeds_index = mm_tracker.get_num_items("prompt_embeds")
        mm_parser.parse_prompt_embeds(cast(str, content))
        prompt_embeds_placeholder = make_prompt_embeds_placeholder(prompt_embeds_index)
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
            # emit a unique sentinel as text so the template renders it inline.
            assert prompt_embeds_placeholder is not None
            return {"type": "text", "text": prompt_embeds_placeholder}
        result = {"type": modality}
        result.update(_collect_extra_fields(cast(dict[str, Any], part)))
        return result
    if modality == "prompt_embeds":
        # Emit the renderer token inline regardless of `interleave_strings`.
        # Falling back to front-padding via `missing_placeholders` would
        # reorder them relative to surrounding text.
        assert prompt_embeds_placeholder is not None
        return prompt_embeds_placeholder
    return MODALITY_PLACEHOLDERS_MAP[modality] if interleave_strings else None


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


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_tracker: BaseMultiModalItemTracker,
    content_format: ChatTemplateContentFormat,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[ConversationMessage]:
    role = message["role"]
    content = message.get("content")
    reasoning = message.get("reasoning")

    if isinstance(reasoning, dict) and reasoning.get("type") == "prompt_embeds":
        reasoning = parse_chat_message_content_parts(
            role,
            [cast(ChatCompletionContentPartParam, reasoning)],
            mm_tracker,
            wrap_dicts=False,
            interleave_strings=interleave_strings,
            mm_processor_kwargs=mm_processor_kwargs,
        )[0]["content"]

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [ChatCompletionContentPartTextParam(type="text", text=content)]
    if role == "system":
        content = _strip_claude_code_billing_header(content)  # type: ignore[arg-type]
    result = parse_chat_message_content_parts(
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
                                    "Tool call has arguments that are not valid "
                                    "JSON (%d chars); coercing to an empty object "
                                    "so the conversation can continue.",
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
                                    "Tool call arguments decoded to %s, not a "
                                    "JSON object; coercing to an empty object.",
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
        sub_messages = parse_chat_message_content(
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
        sub_messages = parse_chat_message_content(
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
