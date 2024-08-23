import asyncio
import codecs
from functools import lru_cache
from pathlib import Path
from typing import (Any, Awaitable, Iterable, List, Literal, Optional, Tuple,
                    Union)

# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
# yapf: enable
# pydantic needs the TypedDict from typing_extensions
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import Required, TypeAlias, TypedDict

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.utils import (async_get_and_parse_audio,
                                   async_get_and_parse_image)
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class AudioURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the audio or a data URL with base64 encoded audio data.
    """


class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]

    type: Required[Literal["audio_url"]]
    """The type of the content part."""


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, ChatCompletionContentPartAudioParam,
    CustomChatCompletionContentPartParam, ]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]


# TODO: Make fields ReadOnly once mypy supports it
class ConversationMessage(TypedDict):
    role: str
    content: str


class MultiModalItemTracker:
    """
    Tracks multi-model items in a given request and ensures that the number
    of multi-modal items in a given request does not exceed the configured
    maximum per prompt.
    """

    def __init__(self, model_config: ModelConfig):
        self._allowed_items = (model_config.multimodal_config.limit_per_prompt
                               if model_config.multimodal_config else {})
        self._consumed_items = {k: 0 for k in self._allowed_items}
        self._futures: List[Awaitable[MultiModalDataDict]] = []

    def add(self, modality: Literal["image", "audio"],
            mm_future: Awaitable[MultiModalDataDict]):
        allowed_count = self._allowed_items.get(modality, 1)
        existing_count = self._consumed_items.get(modality, 0)
        if existing_count >= allowed_count:
            raise ValueError(
                f"At most {allowed_count} {modality}s may be provided in one "
                "request.")

        self._consumed_items[modality] = existing_count + 1
        self._futures.append(mm_future)

    def all_mm_data(self) -> Optional[Awaitable[MultiModalDataDict]]:

        async def _combine(futures: List[Awaitable[MultiModalDataDict]]):
            mm_data: MultiModalDataDict = {}

            # Merge all the multi-modal items
            for single_mm_data in (await asyncio.gather(*futures)):
                for mm_key, mm_item in single_mm_data.items():
                    existing_item = mm_data.get(mm_key)
                    if not existing_item:
                        # Clone it if it's already a list so we can freely
                        # mutate it later.
                        item_to_insert = mm_item[:] if isinstance(
                            mm_item, list) else mm_item
                        mm_data[mm_key] = item_to_insert  # type: ignore
                    else:
                        if isinstance(existing_item, list):
                            result_list = existing_item
                        else:
                            result_list = [existing_item]
                            mm_data[mm_key] = result_list  # type: ignore

                        if isinstance(mm_item, list):
                            result_list.extend(mm_item)
                        else:
                            result_list.append(mm_item)

            return mm_data

        return _combine(self._futures) if self._futures else None


def load_chat_template(
        chat_template: Optional[Union[Path, str]]) -> Optional[str]:
    if chat_template is None:
        return None
    try:
        with open(chat_template, "r") as f:
            resolved_chat_template = f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            msg = (f"The supplied chat template ({chat_template}) "
                   f"looks like a file path, but it failed to be "
                   f"opened. Reason: {e}")
            raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        resolved_chat_template = codecs.decode(chat_template, "unicode_escape")

    logger.info("Using supplied chat template:\n%s", resolved_chat_template)
    return resolved_chat_template


@lru_cache(maxsize=None)
def _mm_token_str(model_config: ModelConfig, tokenizer: AnyTokenizer,
                  modality: Literal["image", "audio"]) -> Optional[str]:
    # TODO: Let user specify how to insert image tokens into prompt
    # (similar to chat template)
    model_type = model_config.hf_config.model_type
    if modality == "image":
        if model_type == "phi3_v":
            # Workaround since this token is not defined in the tokenizer
            return "<|image_1|>"
        if model_type == "minicpmv":
            return "(<image>./</image>)"
        if model_type in ("blip-2", "chatglm", "fuyu", "paligemma"):
            # These models do not use image tokens in the prompt
            return None
        if model_type.startswith("llava"):
            return tokenizer.decode(model_config.hf_config.image_token_index)
        if model_type in ("chameleon", "internvl_chat"):
            return "<image>"

        raise TypeError(f"Unknown model type: {model_type}")
    elif modality == "audio":
        if model_type == "ultravox":
            return "<|reserved_special_token_0|>"
        raise TypeError(f"Unknown model type: {model_type}")
    else:
        raise TypeError(f"Unknown modality: {modality}")


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(placeholders: List[str],
                                     text_prompt: str) -> str:
    """Combine multimodal prompts for a multimodal language model"""

    # Look through the text prompt so that we don't add placeholders for
    # items that already have them.
    prompt_fragments = [text_prompt]
    missing_placeholders: List[str] = []
    for placeholder in placeholders:
        for fragment_index, fragment in enumerate(prompt_fragments):
            index = fragment.find(placeholder)
            if index >= 0:
                # This part of the text prompt already has a placeholder.
                # Remove it from the text prompt fragments so that we don't
                # consider it for later placeholders.
                prompt_fragments.pop(fragment_index)
                before = fragment[:index]
                after = fragment[index + len(placeholder):]
                if before:
                    prompt_fragments.insert(fragment_index, before)
                if after:
                    prompt_fragments.insert(fragment_index + 1, after)
                break
        else:
            # The placeholder wasn't in any of the text prompts; we need
            # to include it.
            missing_placeholders.append(placeholder)

    # NOTE: For now we assume all model architectures use the same
    # placeholder + text prompt format. This may change in the future.
    return "\n".join(missing_placeholders + [text_prompt])


_TextParser = TypeAdapter(ChatCompletionContentPartTextParam)
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam)
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam)


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    mm_tracker: MultiModalItemTracker,
) -> List[ConversationMessage]:
    texts: List[str] = []
    mm_placeholders: List[str] = []

    for part in parts:
        part_type = part["type"]
        if part_type == "text":
            text = _TextParser.validate_python(part)["text"]
            texts.append(text)
        elif part_type == "image_url":
            image_url = _ImageParser.validate_python(part)["image_url"]

            if image_url.get("detail", "auto") != "auto":
                logger.warning(
                    "'image_url.detail' is currently not supported and "
                    "will be ignored.")

            mm_tracker.add("image",
                           async_get_and_parse_image(image_url["url"]))

            placeholder = _mm_token_str(model_config, tokenizer, "image")
            if placeholder:
                mm_placeholders.append(placeholder)
        elif part_type == "audio_url":
            audio_url = _AudioParser.validate_python(part)["audio_url"]
            mm_tracker.add("audio",
                           async_get_and_parse_audio(audio_url["url"]))

            placeholder = _mm_token_str(model_config, tokenizer, "audio")
            if placeholder:
                mm_placeholders.append(placeholder)
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)
    if mm_placeholders:
        text_prompt = _get_full_multimodal_text_prompt(mm_placeholders,
                                                       text_prompt)

    return [ConversationMessage(role=role, content=text_prompt)]


def _parse_chat_message_content(
        message: ChatCompletionMessageParam, model_config: ModelConfig,
        tokenizer: AnyTokenizer,
        mm_tracker: MultiModalItemTracker) -> List[ConversationMessage]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        return [ConversationMessage(role=role, content=content)]

    return _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        model_config,
        tokenizer,
        mm_tracker,
    )


def parse_chat_messages(
    messages: List[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> Tuple[List[ConversationMessage], Optional[Awaitable[MultiModalDataDict]]]:
    conversation: List[ConversationMessage] = []
    mm_tracker = MultiModalItemTracker(model_config)

    for msg in messages:
        sub_messages = _parse_chat_message_content(msg, model_config,
                                                   tokenizer, mm_tracker)

        conversation.extend(sub_messages)

    return conversation, mm_tracker.all_mm_data()


def apply_chat_template(
    tokenizer: AnyTokenizer,
    conversation: List[ConversationMessage],
    chat_template: Optional[str],
    *,
    tokenize: bool = False,  # Different from HF's default
    **kwargs: Any,
) -> str:
    if chat_template is None and tokenizer.chat_template is None:
        raise ValueError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one.")

    prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        chat_template=chat_template,
        tokenize=tokenize,
        **kwargs,
    )
    assert isinstance(prompt, str)

    return prompt
