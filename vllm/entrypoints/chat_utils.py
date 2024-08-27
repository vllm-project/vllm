import codecs
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    mm_futures: List[Awaitable[MultiModalDataDict]]


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
def _get_full_multimodal_text_prompt(placeholder_token_str: str,
                                     text_prompt: str) -> str:
    """Combine multimodal prompts for a multimodal language model"""

    # NOTE: For now we assume all model architectures use the same
    # placeholder + text prompt format. This may change in the future.
    return f"{placeholder_token_str}\n{text_prompt}"


_TextParser = TypeAdapter(ChatCompletionContentPartTextParam)
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam)
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam)


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> ChatMessageParseResult:
    texts: List[str] = []
    mm_futures: List[Awaitable[MultiModalDataDict]] = []
    modality: Literal["image", "audio"] = "image"

    for part in parts:
        part_type = part["type"]
        if part_type == "text":
            text = _TextParser.validate_python(part)["text"]
            texts.append(text)
        elif part_type == "image_url":
            modality = "image"
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple multimodal inputs is currently not supported.")

            image_url = _ImageParser.validate_python(part)["image_url"]

            if image_url.get("detail", "auto") != "auto":
                logger.warning(
                    "'image_url.detail' is currently not supported and "
                    "will be ignored.")

            image_future = async_get_and_parse_image(image_url["url"])
            mm_futures.append(image_future)
        elif part_type == "audio_url":
            modality = "audio"
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple multimodal inputs is currently not supported.")

            audio_url = _AudioParser.validate_python(part)["audio_url"]
            audio_future = async_get_and_parse_audio(audio_url["url"])
            mm_futures.append(audio_future)
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)

    if mm_futures:
        placeholder_token_str = _mm_token_str(model_config, tokenizer,
                                              modality)
        if placeholder_token_str is not None:
            if placeholder_token_str in text_prompt:
                logger.warning(
                    "Detected multi-modal token string in the text prompt. "
                    "Skipping prompt formatting.")
            else:
                text_prompt = _get_full_multimodal_text_prompt(
                    placeholder_token_str=placeholder_token_str,
                    text_prompt=text_prompt,
                )

    messages = [ConversationMessage(role=role, content=text_prompt)]

    return ChatMessageParseResult(messages=messages, mm_futures=mm_futures)


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> ChatMessageParseResult:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return ChatMessageParseResult(messages=[], mm_futures=[])
    if isinstance(content, str):
        messages = [ConversationMessage(role=role, content=content)]
        return ChatMessageParseResult(messages=messages, mm_futures=[])

    return _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        model_config,
        tokenizer,
    )


def parse_chat_messages(
    messages: List[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> Tuple[List[ConversationMessage], List[Awaitable[MultiModalDataDict]]]:
    conversation: List[ConversationMessage] = []
    mm_futures: List[Awaitable[MultiModalDataDict]] = []

    for msg in messages:
        parse_result = _parse_chat_message_content(msg, model_config,
                                                   tokenizer)

        conversation.extend(parse_result.messages)
        mm_futures.extend(parse_result.mm_futures)

    return conversation, mm_futures


def apply_chat_template(
    tokenizer: AnyTokenizer,
    conversation: List[ConversationMessage],
    chat_template: Optional[str],
    *,
    tokenize: bool = False,  # Different from HF's default
    **kwargs: Any,
) -> Union[str, List[int]]:
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
    return prompt
