import asyncio
import codecs
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import (Any, Awaitable, Dict, Generic, Iterable, List, Literal,
                    Mapping, Optional, Tuple, TypeVar, Union)

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
                                   async_get_and_parse_image,
                                   get_and_parse_audio, get_and_parse_image)
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


ModalityStr = Literal["image", "audio"]
_T = TypeVar("_T")


class BaseMultiModalItemTracker(ABC, Generic[_T]):
    """
    Tracks multi-modal items in a given request and ensures that the number
    of multi-modal items in a given request does not exceed the configured
    maximum per prompt.
    """

    def __init__(self, model_config: ModelConfig, tokenizer: AnyTokenizer):
        super().__init__()

        self._model_config = model_config
        self._tokenizer = tokenizer
        self._allowed_items = (model_config.multimodal_config.limit_per_prompt
                               if model_config.multimodal_config else {})
        self._consumed_items = {k: 0 for k in self._allowed_items}

        self._items: List[_T] = []

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_token_str(tokenizer: AnyTokenizer, token_index: int) -> str:
        return tokenizer.decode(token_index)

    def _placeholder_str(self, modality: ModalityStr,
                         current_count: int) -> Optional[str]:
        # TODO: Let user specify how to insert image tokens into prompt
        # (similar to chat template)
        hf_config = self._model_config.hf_config
        model_type = hf_config.model_type

        if modality == "image":
            if model_type == "phi3_v":
                # Workaround since this token is not defined in the tokenizer
                return f"<|image_{current_count}|>"
            if model_type == "minicpmv":
                return "(<image>./</image>)"
            if model_type in ("blip-2", "chatglm", "fuyu", "paligemma"):
                # These models do not use image tokens in the prompt
                return None
            if model_type.startswith("llava"):
                return self._cached_token_str(self._tokenizer,
                                              hf_config.image_token_index)
            if model_type in ("chameleon", "internvl_chat"):
                return "<image>"

            raise TypeError(f"Unknown model type: {model_type}")
        elif modality == "audio":
            if model_type == "ultravox":
                return "<|reserved_special_token_0|>"
            raise TypeError(f"Unknown model type: {model_type}")
        else:
            raise TypeError(f"Unknown modality: {modality}")

    @staticmethod
    def _combine(items: List[MultiModalDataDict]) -> MultiModalDataDict:
        mm_lists: Mapping[str, List[object]] = defaultdict(list)

        # Merge all the multi-modal items
        for single_mm_data in items:
            for mm_key, mm_item in single_mm_data.items():
                if isinstance(mm_item, list):
                    mm_lists[mm_key].extend(mm_item)
                else:
                    mm_lists[mm_key].append(mm_item)

        # Unpack any single item lists for models that don't expect multiple.
        return {
            mm_key: mm_list[0] if len(mm_list) == 1 else mm_list
            for mm_key, mm_list in mm_lists.items()
        }

    def add(self, modality: ModalityStr, item: _T) -> Optional[str]:
        """
        Add a multi-modal item to the current prompt and returns the
        placeholder string to use, if any.
        """
        allowed_count = self._allowed_items.get(modality, 1)
        current_count = self._consumed_items.get(modality, 0) + 1
        if current_count > allowed_count:
            raise ValueError(
                f"At most {allowed_count} {modality}(s) may be provided in "
                "one request.")

        self._consumed_items[modality] = current_count
        self._items.append(item)

        return self._placeholder_str(modality, current_count)

    @abstractmethod
    def create_parser(self) -> "BaseMultiModalContentParser":
        raise NotImplementedError


class MultiModalItemTracker(BaseMultiModalItemTracker[MultiModalDataDict]):

    def all_mm_data(self) -> Optional[MultiModalDataDict]:
        return self._combine(self._items) if self._items else None

    def create_parser(self) -> "BaseMultiModalContentParser":
        return MultiModalContentParser(self)


class AsyncMultiModalItemTracker(
        BaseMultiModalItemTracker[Awaitable[MultiModalDataDict]]):

    async def all_mm_data(self) -> Optional[MultiModalDataDict]:
        if self._items:
            items = await asyncio.gather(*self._items)
            return self._combine(items)

        return None

    def create_parser(self) -> "BaseMultiModalContentParser":
        return AsyncMultiModalContentParser(self)


class BaseMultiModalContentParser(ABC):

    def __init__(self) -> None:
        super().__init__()

        # multimodal placeholder_string : count
        self._placeholder_counts: Dict[str, int] = defaultdict(lambda: 0)

    def _add_placeholder(self, placeholder: Optional[str]):
        if placeholder:
            self._placeholder_counts[placeholder] += 1

    def mm_placeholder_counts(self) -> Dict[str, int]:
        return dict(self._placeholder_counts)

    @abstractmethod
    def parse_image(self, image_url: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_audio(self, audio_url: str) -> None:
        raise NotImplementedError


class MultiModalContentParser(BaseMultiModalContentParser):

    def __init__(self, tracker: MultiModalItemTracker) -> None:
        super().__init__()

        self._tracker = tracker

    def parse_image(self, image_url: str) -> None:
        image = get_and_parse_image(image_url)

        placeholder = self._tracker.add("image", image)
        self._add_placeholder(placeholder)

    def parse_audio(self, audio_url: str) -> None:
        audio = get_and_parse_audio(audio_url)

        placeholder = self._tracker.add("audio", audio)
        self._add_placeholder(placeholder)


class AsyncMultiModalContentParser(BaseMultiModalContentParser):

    def __init__(self, tracker: AsyncMultiModalItemTracker) -> None:
        super().__init__()

        self._tracker = tracker

    def parse_image(self, image_url: str) -> None:
        image_coro = async_get_and_parse_image(image_url)

        placeholder = self._tracker.add("image", image_coro)
        self._add_placeholder(placeholder)

    def parse_audio(self, audio_url: str) -> None:
        audio_coro = async_get_and_parse_audio(audio_url)

        placeholder = self._tracker.add("audio", audio_coro)
        self._add_placeholder(placeholder)


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


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(placeholder_counts: Dict[str, int],
                                     text_prompt: str) -> str:
    """Combine multimodal prompts for a multimodal language model."""

    # Look through the text prompt to check for missing placeholders
    missing_placeholders: List[str] = []
    for placeholder in placeholder_counts:

        # For any existing placeholder in the text prompt, we leave it as is
        placeholder_counts[placeholder] -= text_prompt.count(placeholder)

        if placeholder_counts[placeholder] < 0:
            raise ValueError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items.")

        missing_placeholders.extend([placeholder] *
                                    placeholder_counts[placeholder])

    # NOTE: For now we always add missing placeholders at the front of
    # the prompt. This may change to be customizable in the future.
    return "\n".join(missing_placeholders + [text_prompt])


_TextParser = TypeAdapter(ChatCompletionContentPartTextParam)
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam)
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam)


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
) -> List[ConversationMessage]:
    texts: List[str] = []

    mm_parser = mm_tracker.create_parser()

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

            mm_parser.parse_image(image_url["url"])
        elif part_type == "audio_url":
            audio_url = _AudioParser.validate_python(part)["audio_url"]

            mm_parser.parse_audio(audio_url["url"])
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)
    mm_placeholder_counts = mm_parser.mm_placeholder_counts()
    if mm_placeholder_counts:
        text_prompt = _get_full_multimodal_text_prompt(mm_placeholder_counts,
                                                       text_prompt)

    return [ConversationMessage(role=role, content=text_prompt)]


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_tracker: BaseMultiModalItemTracker,
) -> List[ConversationMessage]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        return [ConversationMessage(role=role, content=content)]

    return _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        mm_tracker,
    )


def parse_chat_messages(
    messages: List[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> Tuple[List[ConversationMessage], Optional[MultiModalDataDict]]:
    conversation: List[ConversationMessage] = []
    mm_tracker = MultiModalItemTracker(model_config, tokenizer)

    for msg in messages:
        sub_messages = _parse_chat_message_content(msg, mm_tracker)

        conversation.extend(sub_messages)

    return conversation, mm_tracker.all_mm_data()


def parse_chat_messages_futures(
    messages: List[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> Tuple[List[ConversationMessage], Awaitable[Optional[MultiModalDataDict]]]:
    conversation: List[ConversationMessage] = []
    mm_tracker = AsyncMultiModalItemTracker(model_config, tokenizer)

    for msg in messages:
        sub_messages = _parse_chat_message_content(msg, mm_tracker)

        conversation.extend(sub_messages)

    return conversation, mm_tracker.all_mm_data()


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
