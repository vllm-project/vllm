# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Iterable
from functools import cached_property, lru_cache, partial
from pathlib import Path
from typing import (Any, Callable, Generic, Literal, Optional, TypeVar, Union,
                    cast)

import jinja2.nodes
import transformers.utils.chat_template_utils as hf_chat_utils
# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartInputAudioParam)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import (ChatCompletionContentPartRefusalParam,
                               ChatCompletionContentPartTextParam)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
from openai.types.chat import (ChatCompletionMessageToolCallParam,
                               ChatCompletionToolMessageParam)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    InputAudio)
from openai.types.responses import ResponseInputImageParam
from openai_harmony import Message as OpenAIHarmonyMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter
# yapf: enable
from transformers import (PreTrainedTokenizer, PreTrainedTokenizerFast,
                          ProcessorMixin)
# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypeAlias, TypedDict

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.utils import MediaConnector
# yapf: disable
from vllm.transformers_utils.chat_templates import (
    get_chat_template_fallback_path)
# yapf: enable
from vllm.transformers_utils.processor import cached_get_processor
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)

MODALITY_PLACEHOLDERS_MAP = {
    "image": "<##IMAGE##>",
    "audio": "<##AUDIO##>",
    "video": "<##VIDEO##>",
}


class AudioURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the audio or a data URL with base64 encoded audio data.
    """


class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]

    type: Required[Literal["audio_url"]]
    """The type of the content part."""


class ChatCompletionContentPartImageEmbedsParam(TypedDict, total=False):
    image_embeds: Required[Union[str, dict[str, str]]]
    """
    The image embeddings. It can be either:
    - A single base64 string.
    - A dictionary where each value is a base64 string.
    """
    type: Required[Literal["image_embeds"]]
    """The type of the content part."""


class VideoURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the video or a data URL with base64 encoded video data.
    """


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content part."""


class PILImage(BaseModel):
    """
    A PIL.Image.Image object.
    """
    image_pil: Image.Image
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CustomChatCompletionContentPILImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a PIL image.

    Example:
    {
        "image_pil": ImageAsset('cherry_blossom').pil_image
    }
    """
    image_pil: Required[PILImage]


class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain image_url.
    This is supported by OpenAI API, although it is not documented.

    Example:
    {
        "image_url": "https://example.com/image.jpg"
    }
    """
    image_url: Required[str]


class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "audio_url": "https://example.com/audio.mp3"
    }
    """
    audio_url: Required[str]


class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "video_url": "https://example.com/video.mp4"
    }
    """
    video_url: Required[str]


class CustomThinkCompletionContentParam(TypedDict, total=False):
    """A Think Completion Content Param that accepts a plain text and a boolean.

    Example:
    {
        "thinking": "I am thinking about the answer",
        "closed": True,
        "type": "thinking"
    }
    """

    thinking: Required[str]
    """The thinking content."""

    closed: bool
    """Whether the thinking is closed."""

    type: Required[Literal["thinking"]]
    """The thinking type."""


ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, ChatCompletionContentPartAudioParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartVideoParam, ChatCompletionContentPartRefusalParam,
    CustomChatCompletionContentPILImageParam,
    CustomChatCompletionContentSimpleImageParam,
    ChatCompletionContentPartImageEmbedsParam,
    CustomChatCompletionContentSimpleAudioParam,
    CustomChatCompletionContentSimpleVideoParam, str,
    CustomThinkCompletionContentParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, list[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: Optional[str]
    """Tool call that this message is responding to."""

    tool_calls: Optional[Iterable[ChatCompletionMessageToolCallParam]]
    """The tool calls generated by the model, such as function calls."""


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam,
                                   OpenAIHarmonyMessage]


# TODO: Make fields ReadOnly once mypy supports it
class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    """The role of the message's author."""

    content: Union[Optional[str], list[dict[str, str]]]
    """The contents of the message"""

    tool_call_id: Optional[str]
    """Tool call that this message is responding to."""

    name: Optional[str]
    """The name of the function to call"""

    tool_calls: Optional[Iterable[ChatCompletionMessageToolCallParam]]
    """The tool calls generated by the model, such as function calls."""


# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

# Used internally
_ChatTemplateContentFormat = Literal["string", "openai"]


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (_is_var_access(node.node, varname)
                and isinstance(node.arg, jinja2.nodes.Const)
                and node.arg.value == key)

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: Optional[str] = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return (node.node is not None
                and _is_var_or_elems_access(node.node, varname, key))
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if (isinstance(node, jinja2.nodes.Getitem)
            and isinstance(node.arg, jinja2.nodes.Slice)):
        return _is_var_or_elems_access(node.node, varname, key)

    # yapf: disable
    return (
        _is_attr_access(node, varname, key) if key
        else _is_var_access(node, varname)
    ) # yapf: enable


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if _is_var_or_elems_access(rhs, related_varname):
                assert isinstance(lhs, jinja2.nodes.Name)
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname:
                    related_varnames.append(lhs.name)


# NOTE: The proper way to handle this is to build a CFG so that we can handle
# the scope in which each variable is defined, but that is too complicated
def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node):
    messages_varnames = [
        varname
        for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")
    ]

    # Search for {%- for message in messages -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in messages_varnames:
            if _is_var_or_elems_access(loop_iter, varname):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node):
    message_varnames = [
        varname for _, varname in _iter_nodes_assign_messages_item(root)
    ]

    # Search for {%- for content in message['content'] -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _try_extract_ast(chat_template: str) -> Optional[jinja2.nodes.Template]:
    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception:
        logger.exception("Error when compiling Jinja template")
        return None


@lru_cache(maxsize=32)
def _detect_content_format(
    chat_template: str,
    *,
    default: _ChatTemplateContentFormat,
) -> _ChatTemplateContentFormat:
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return default

    try:
        next(_iter_nodes_assign_content_item(jinja_ast))
    except StopIteration:
        return "string"
    except Exception:
        logger.exception("Error when parsing AST of Jinja template")
        return default
    else:
        return "openai"


def resolve_mistral_chat_template(
    chat_template: Optional[str],
    **kwargs: Any,
) -> Optional[str]:
    if chat_template is not None:
        logger.warning_once(
            "'chat_template' cannot be overridden for mistral tokenizer.")
    if "add_generation_prompt" in kwargs:
        logger.warning_once(
            "'add_generation_prompt' is not supported for mistral tokenizer, "
            "so it will be ignored.")
    if "continue_final_message" in kwargs:
        logger.warning_once(
            "'continue_final_message' is not supported for mistral tokenizer, "
            "so it will be ignored.")
    return None


def resolve_hf_chat_template(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    *,
    model_config: ModelConfig,
) -> Optional[str]:
    # 1st priority: The given chat template
    if chat_template is not None:
        return chat_template

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        try:
            processor = cached_get_processor(
                tokenizer.name_or_path,
                processor_cls=(PreTrainedTokenizer, PreTrainedTokenizerFast,
                               ProcessorMixin),
                trust_remote_code=model_config.trust_remote_code,
            )
            if isinstance(processor, ProcessorMixin) and \
                hasattr(processor, 'chat_template') and \
                processor.chat_template is not None:
                return processor.chat_template
        except Exception:
            logger.debug("Failed to load AutoProcessor chat template for %s", tokenizer.name_or_path, exc_info=True)  # noqa: E501

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug("Failed to load AutoTokenizer chat template for %s",
                     tokenizer.name_or_path, exc_info=True)

    # 4th priority: Predefined fallbacks
    path = get_chat_template_fallback_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=model_config.tokenizer,
    )
    if path is not None:
        logger.info("Loading chat template fallback for %s as there isn't one "
                    "defined on HF Hub.", tokenizer.name_or_path)
        chat_template = load_chat_template(path)
    else:
        logger.debug("There is no chat template fallback for %s",
                     tokenizer.name_or_path)

    return chat_template


def _resolve_chat_template_content_format(
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    tokenizer: AnyTokenizer,
    *,
    model_config: ModelConfig,
) -> _ChatTemplateContentFormat:
    if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        hf_chat_template = resolve_hf_chat_template(
            tokenizer,
            chat_template=chat_template,
            tools=tools,
            model_config=model_config,
        )
    else:
        hf_chat_template = None

    jinja_text = (hf_chat_template if isinstance(hf_chat_template, str)
                  else load_chat_template(chat_template, is_literal=True))

    detected_format = ("string" if jinja_text is None else
                       _detect_content_format(jinja_text, default="string"))

    return detected_format


@lru_cache
def _log_chat_template_content_format(
    chat_template: Optional[str],
    given_format: ChatTemplateContentFormatOption,
    detected_format: ChatTemplateContentFormatOption,
):
    logger.info(
        "Detected the chat template content format to be '%s'. "
        "You can set `--chat-template-content-format` to override this.",
        detected_format,
    )

    if given_format != "auto" and given_format != detected_format:
        logger.warning(
            "You specified `--chat-template-content-format %s` "
            "which is different from the detected format '%s'. "
            "If our automatic detection is incorrect, please consider "
            "opening a GitHub issue so that we can improve it: "
            "https://github.com/vllm-project/vllm/issues/new/choose",
            given_format,
            detected_format,
        )


def resolve_chat_template_content_format(
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    given_format: ChatTemplateContentFormatOption,
    tokenizer: AnyTokenizer,
    *,
    model_config: ModelConfig,
) -> _ChatTemplateContentFormat:
    if given_format != "auto":
        return given_format

    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        tokenizer,
        model_config=model_config,
    )

    _log_chat_template_content_format(
        chat_template,
        given_format=given_format,
        detected_format=detected_format,
    )

    return detected_format



ModalityStr = Literal["image", "audio", "video", "image_embeds"]
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

        self._items_by_modality = defaultdict[str, list[_T]](list)

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @cached_property
    def model_cls(self) -> type[SupportsMultiModal]:
        from vllm.model_executor.model_loader import get_model_cls
        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsMultiModal], model_cls)

    @property
    def allowed_local_media_path(self):
        return self._model_config.allowed_local_media_path

    @property
    def mm_registry(self):
        return MULTIMODAL_REGISTRY

    @cached_property
    def mm_processor(self):
        return self.mm_registry.create_processor(self.model_config)

    def add(self, modality: ModalityStr, item: _T) -> Optional[str]:
        """
        Add a multi-modal item to the current prompt and returns the
        placeholder string to use, if any.
        """
        input_modality = modality.replace("_embeds", "")
        num_items = len(self._items_by_modality[modality]) + 1

        self.mm_processor.validate_num_items(input_modality, num_items)

        self._items_by_modality[modality].append(item)

        return self.model_cls.get_placeholder_str(modality, num_items)

    @abstractmethod
    def create_parser(self) -> "BaseMultiModalContentParser":
        raise NotImplementedError


class MultiModalItemTracker(BaseMultiModalItemTracker[object]):

    def all_mm_data(self) -> Optional[MultiModalDataDict]:
        if not self._items_by_modality:
            return None
        mm_inputs = {}
        items_by_modality = dict(self._items_by_modality)
        if "image" in items_by_modality and "image_embeds" in items_by_modality:
            raise ValueError(\
                "Mixing raw image and embedding inputs is not allowed")

        if "image_embeds" in items_by_modality:
            image_embeds_lst = items_by_modality["image_embeds"]
            if len(image_embeds_lst) > 1:
                raise ValueError(\
                    "Only one message can have {'type': 'image_embeds'}")
            mm_inputs["image"] = image_embeds_lst[0]
        if "image" in items_by_modality:
            mm_inputs["image"] = items_by_modality["image"] # A list of images
        if "audio" in items_by_modality:
            mm_inputs["audio"] = items_by_modality["audio"] # A list of audios
        if "video" in items_by_modality:
            mm_inputs["video"] = items_by_modality["video"] # A list of videos
        return mm_inputs

    def create_parser(self) -> "BaseMultiModalContentParser":
        return MultiModalContentParser(self)


class AsyncMultiModalItemTracker(BaseMultiModalItemTracker[Awaitable[object]]):

    async def all_mm_data(self) -> Optional[MultiModalDataDict]:
        if not self._items_by_modality:
            return None
        mm_inputs = {}
        items_by_modality = {
                modality: await asyncio.gather(*items)
                for modality, items in self._items_by_modality.items()
            }

        if "image" in items_by_modality and "image_embeds" in items_by_modality:
            raise ValueError(
                "Mixing raw image and embedding inputs is not allowed")

        if "image_embeds" in items_by_modality:
            image_embeds_lst = items_by_modality["image_embeds"]
            if len(image_embeds_lst) > 1:
                raise ValueError(
                    "Only one message can have {'type': 'image_embeds'}")
            mm_inputs["image"] = image_embeds_lst[0]
        if "image" in items_by_modality:
            mm_inputs["image"] = items_by_modality["image"] # A list of images
        if "audio" in items_by_modality:
            mm_inputs["audio"] = items_by_modality["audio"] # A list of audios
        if "video" in items_by_modality:
            mm_inputs["video"] = items_by_modality["video"] # A list of videos
        return mm_inputs

    def create_parser(self) -> "BaseMultiModalContentParser":
        return AsyncMultiModalContentParser(self)


class BaseMultiModalContentParser(ABC):

    def __init__(self) -> None:
        super().__init__()

        # stores model placehodlers list with corresponding
        # general MM placeholder:
        # {
        #   "<##IMAGE##>": ["<image>", "<image>", "<image>"],
        #   "<##AUDIO##>": ["<audio>", "<audio>"]
        # }
        self._placeholder_storage: dict[str, list] = defaultdict(list)

    def _add_placeholder(self, modality: ModalityStr,
                         placeholder: Optional[str]):
        mod_placeholder = MODALITY_PLACEHOLDERS_MAP[modality]
        if placeholder:
            self._placeholder_storage[mod_placeholder].append(placeholder)

    def mm_placeholder_storage(self) -> dict[str, list]:
        return dict(self._placeholder_storage)

    @abstractmethod
    def parse_image(self, image_url: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_image_embeds(self,
                           image_embeds: Union[str, dict[str, str]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_image_pil(self, image_pil: Image.Image) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_audio(self, audio_url: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_input_audio(self, input_audio: InputAudio) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_video(self, video_url: str) -> None:
        raise NotImplementedError


class MultiModalContentParser(BaseMultiModalContentParser):

    def __init__(self, tracker: MultiModalItemTracker) -> None:
        super().__init__()

        self._tracker = tracker

        self._connector = MediaConnector(
            media_io_kwargs=self._tracker._model_config.media_io_kwargs,
            allowed_local_media_path=tracker.allowed_local_media_path,
        )

    def parse_image(self, image_url: str) -> None:
        image = self._connector.fetch_image(image_url)

        placeholder = self._tracker.add("image", image)
        self._add_placeholder("image", placeholder)

    def parse_image_embeds(self,
                           image_embeds: Union[str, dict[str, str]]) -> None:
        if isinstance(image_embeds, dict):
            embeds = {
                k: self._connector.fetch_image_embedding(v)
                for k, v in image_embeds.items()
            }
            placeholder = self._tracker.add("image_embeds", embeds)

        if isinstance(image_embeds, str):
            embedding = self._connector.fetch_image_embedding(image_embeds)
            placeholder = self._tracker.add("image_embeds", embedding)

        self._add_placeholder("image", placeholder)

    def parse_image_pil(self, image_pil: Image.Image) -> None:
        placeholder = self._tracker.add("image", image_pil)
        self._add_placeholder("image", placeholder)

    def parse_audio(self, audio_url: str) -> None:
        audio = self._connector.fetch_audio(audio_url)

        placeholder = self._tracker.add("audio", audio)
        self._add_placeholder("audio", placeholder)

    def parse_input_audio(self, input_audio: InputAudio) -> None:
        audio_data = input_audio.get("data", "")
        audio_format = input_audio.get("format", "")
        audio_url = f"data:audio/{audio_format};base64,{audio_data}"

        return self.parse_audio(audio_url)

    def parse_video(self, video_url: str) -> None:
        video = self._connector.fetch_video(video_url=video_url)

        placeholder = self._tracker.add("video", video)
        self._add_placeholder("video", placeholder)


class AsyncMultiModalContentParser(BaseMultiModalContentParser):

    def __init__(self, tracker: AsyncMultiModalItemTracker) -> None:
        super().__init__()

        self._tracker = tracker
        self._connector = MediaConnector(
            media_io_kwargs=self._tracker._model_config.media_io_kwargs,
            allowed_local_media_path=tracker.allowed_local_media_path
        )

    def parse_image(self, image_url: str) -> None:
        image_coro = self._connector.fetch_image_async(image_url)

        placeholder = self._tracker.add("image", image_coro)
        self._add_placeholder("image", placeholder)

    def parse_image_embeds(self,
                           image_embeds: Union[str, dict[str, str]]) -> None:
        future: asyncio.Future[Union[str, dict[str, str]]] = asyncio.Future()

        if isinstance(image_embeds, dict):
            embeds = {
                k: self._connector.fetch_image_embedding(v)
                for k, v in image_embeds.items()
            }
            future.set_result(embeds)

        if isinstance(image_embeds, str):
            embedding = self._connector.\
                fetch_image_embedding(image_embeds)
            future.set_result(embedding)

        placeholder = self._tracker.add("image_embeds", future)
        self._add_placeholder("image", placeholder)

    def parse_image_pil(self, image_pil: Image.Image) -> None:
        future: asyncio.Future[Image.Image] = asyncio.Future()
        future.set_result(image_pil)

        placeholder = self._tracker.add("image", future)
        self._add_placeholder("image", placeholder)

    def parse_audio(self, audio_url: str) -> None:
        audio_coro = self._connector.fetch_audio_async(audio_url)

        placeholder = self._tracker.add("audio", audio_coro)
        self._add_placeholder("audio", placeholder)

    def parse_input_audio(self, input_audio: InputAudio) -> None:
        audio_data = input_audio.get("data", "")
        audio_format = input_audio.get("format", "")
        audio_url = f"data:audio/{audio_format};base64,{audio_data}"

        return self.parse_audio(audio_url)

    def parse_video(self, video_url: str) -> None:
        video = self._connector.fetch_video_async(video_url=video_url)

        placeholder = self._tracker.add("video", video)
        self._add_placeholder("video", placeholder)


def validate_chat_template(chat_template: Optional[Union[Path, str]]):
    """Raises if the provided chat template appears invalid."""
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError(
            "the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if not any(c in chat_template
                   for c in JINJA_CHARS) and not Path(chat_template).exists():
            raise ValueError(
                f"The supplied chat template string ({chat_template}) "
                f"appears path-like, but doesn't exist!")

    else:
        raise TypeError(
            f"{type(chat_template)} is not a valid chat template type")


def _load_chat_template(
    chat_template: Optional[Union[Path, str]],
    *,
    is_literal: bool = False,
) -> Optional[str]:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError("chat_template is expected to be read directly "
                            "from its value")

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
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
        return _load_chat_template(chat_template, is_literal=True)


_cached_load_chat_template = lru_cache(_load_chat_template)


def load_chat_template(
    chat_template: Optional[Union[Path, str]],
    *,
    is_literal: bool = False,
) -> Optional[str]:
    return _cached_load_chat_template(chat_template, is_literal=is_literal)


def _get_interleaved_text_prompt(placeholder_storage: dict[str, list],
                                 texts: list[str]) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            texts[idx] = placeholder_storage[elem].pop(0)

    return "\n".join(texts)


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(placeholder_storage: dict[str, list],
                                     texts: list[str],
                                     interleave_strings: bool
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
                "when manually placing image placeholders.", interleave_strings
            )
            logger.debug("Input prompt: %s", text_prompt)
            raise ValueError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items.")

        missing_placeholders.extend([placeholder] *
                                    placeholder_counts[placeholder])

    # NOTE: Default behaviour: we always add missing placeholders
    # at the front of the prompt, if interleave_strings=False
    return "\n".join(missing_placeholders + [text_prompt])


# No need to validate using Pydantic again
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_InputAudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
_RefusalParser = partial(cast, ChatCompletionContentPartRefusalParam)
_PILImageParser = partial(cast, CustomChatCompletionContentPILImageParam)
_ThinkParser = partial(cast, CustomThinkCompletionContentParam)
# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ResponsesInputImageParser = TypeAdapter(
    ResponseInputImageParam).validate_python
_ContentPart: TypeAlias = Union[str, dict[str, str], InputAudio, PILImage]

# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], _ContentPart],
] = {
    "text":
    lambda part: _TextParser(part).get("text", None),
    "thinking":
    lambda part: _ThinkParser(part).get("thinking", None),
    "input_text":
    lambda part: _TextParser(part).get("text", None),
    "input_image":
    lambda part: _ResponsesInputImageParser(part).get("image_url", None),
    "image_url":
    lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "image_embeds":
    lambda part: _ImageEmbedsParser(part).get("image_embeds", None),
    "image_pil": lambda part: _PILImageParser(part).get("image_pil", None),
    "audio_url":
    lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    "input_audio":
    lambda part: _InputAudioParser(part).get("input_audio", None),
    "refusal":
    lambda part: _RefusalParser(part).get("refusal", None),
    "video_url":
    lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
}


def _parse_chat_message_content_mm_part(
        part: ChatCompletionContentPartParam) -> tuple[str, _ContentPart]:
    """
    Parses a given multi-modal content part based on its type.

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
        part, dict)  # This is needed to avoid mypy errors: part.get() from str
    part_type = part.get("type", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP:
        content = MM_PARSER_MAP[part_type](part)

        # Special case for 'image_url.detail'
        # We only support 'auto', which is the default
        if part_type == "image_url" and part.get("detail", "auto") != "auto":
            logger.warning("'image_url.detail' is currently not supported "
                           "and will be ignored.")

        return part_type, content

    # Handle missing 'type' but provided direct URL fields.
    # 'type' is required field by pydantic
    if part_type is None:
        if part.get("image_url") is not None:
            image_params = cast(CustomChatCompletionContentSimpleImageParam,
                                part)
            return "image_url", image_params.get("image_url", "")
        if part.get("audio_url") is not None:
            audio_params = cast(CustomChatCompletionContentSimpleAudioParam,
                                part)
            return "audio_url", audio_params.get("audio_url", "")
        if part.get("input_audio") is not None:
            input_audio_params = cast(dict[str, str], part)
            return "input_audio", input_audio_params
        if part.get("video_url") is not None:
            video_params = cast(CustomChatCompletionContentSimpleVideoParam,
                                part)
            return "video_url", video_params.get("video_url", "")
        # Raise an error if no 'type' or direct URL is found.
        raise ValueError("Missing 'type' field in multimodal part.")

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


VALID_MESSAGE_CONTENT_MM_PART_TYPES = ("text", "refusal", "image_url",
                                       "image_embeds", "image_pil",
                                       "audio_url", "input_audio", "video_url")


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
    *,
    wrap_dicts: bool,
    interleave_strings: bool,
) -> list[ConversationMessage]:
    content = list[_ContentPart]()

    mm_parser = mm_tracker.create_parser()

    for part in parts:
        parse_res = _parse_chat_message_content_part(
            part,
            mm_parser,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings
        )
        if parse_res:
            content.append(parse_res)

    if wrap_dicts:
        # Parsing wraps images and texts as interleaved dictionaries
        return [ConversationMessage(role=role,
                                    content=content)]  # type: ignore
    texts = cast(list[str], content)
    mm_placeholder_storage = mm_parser.mm_placeholder_storage()
    if mm_placeholder_storage:
        text_prompt = _get_full_multimodal_text_prompt(mm_placeholder_storage,
                                                       texts,
                                                       interleave_strings)
    else:
        text_prompt = "\n".join(texts)

    return [ConversationMessage(role=role, content=text_prompt)]


def _parse_chat_message_content_part(
    part: ChatCompletionContentPartParam,
    mm_parser: BaseMultiModalContentParser,
    *,
    wrap_dicts: bool,
    interleave_strings: bool,
) -> Optional[_ContentPart]:
    """Parses a single part of a conversation. If wrap_dicts is True,
    structured dictionary pieces for texts and images will be
    wrapped in dictionaries, i.e., {"type": "text", "text", ...} and
    {"type": "image"}, respectively. Otherwise multimodal data will be
    handled by mm_parser, and texts will be returned as strings to be joined
    with multimodal placeholders.
    """
    if isinstance(part, str):  # Handle plain text parts
        return part
    # Handle structured dictionary parts
    part_type, content = _parse_chat_message_content_mm_part(part)
    # if part_type is text/refusal/image_url/audio_url/video_url/input_audio but
    # content is None, log a warning and skip
    if part_type in VALID_MESSAGE_CONTENT_MM_PART_TYPES and content is None:
        logger.warning(
            "Skipping multimodal part '%s' (type: '%s') "
            "with empty / unparsable content.", part, part_type)
        return None

    if part_type in ("text", "input_text", "refusal", "thinking"):
        str_content = cast(str, content)
        if wrap_dicts:
            return {'type': 'text', 'text': str_content}
        else:
            return str_content

    modality = None
    if part_type == "image_pil":
        image_content = cast(Image.Image, content)
        mm_parser.parse_image_pil(image_content)
        modality = "image"
    elif part_type in ("image_url", "input_image"):
        str_content = cast(str, content)
        mm_parser.parse_image(str_content)
        modality = "image"
    elif part_type == "image_embeds":
        content = cast(Union[str, dict[str, str]], content)
        mm_parser.parse_image_embeds(content)
        modality = "image"
    elif part_type == "audio_url":
        str_content = cast(str, content)
        mm_parser.parse_audio(str_content)
        modality = "audio"
    elif part_type == "input_audio":
        dict_content = cast(InputAudio, content)
        mm_parser.parse_input_audio(dict_content)
        modality = "audio"
    elif part_type == "video_url":
        str_content = cast(str, content)
        mm_parser.parse_video(str_content)
        modality = "video"
    else:
        raise NotImplementedError(f"Unknown part type: {part_type}")

    return {'type': modality} if wrap_dicts else (
        MODALITY_PLACEHOLDERS_MAP[modality] if interleave_strings else None
    )


# No need to validate using Pydantic again
_AssistantParser = partial(cast, ChatCompletionAssistantMessageParam)
_ToolParser = partial(cast, ChatCompletionToolMessageParam)


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_tracker: BaseMultiModalItemTracker,
    content_format: _ChatTemplateContentFormat,
    interleave_strings: bool,
) -> list[ConversationMessage]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [
            ChatCompletionContentPartTextParam(type="text", text=content)
        ]
    result = _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        mm_tracker,
        wrap_dicts=(content_format == "openai"),
        interleave_strings=interleave_strings,
    )

    for result_msg in result:
        if role == 'assistant':
            parsed_msg = _AssistantParser(message)

            # The 'tool_calls' is not None check ensures compatibility.
            # It's needed only if downstream code doesn't strictly
            # follow the OpenAI spec.
            if ("tool_calls" in parsed_msg
                and parsed_msg["tool_calls"] is not None):
                result_msg["tool_calls"] = list(parsed_msg["tool_calls"])
        elif role == "tool":
            parsed_msg = _ToolParser(message)
            if "tool_call_id" in parsed_msg:
                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]

        if "name" in message and isinstance(message["name"], str):
            result_msg["name"] = message["name"]

    return result


def _postprocess_messages(messages: list[ConversationMessage]) -> None:
    # per the Transformers docs & maintainers, tool call arguments in
    # assistant-role messages with tool_calls need to be dicts not JSON str -
    # this is how tool-use chat templates will expect them moving forwards
    # so, for messages that have tool_calls, parse the string (which we get
    # from openAI format) to dict
    for message in messages:
        if (message["role"] == "assistant" and "tool_calls" in message
                and isinstance(message["tool_calls"], list)):

            for item in message["tool_calls"]:
                item["function"]["arguments"] = json.loads(
                    item["function"]["arguments"])


def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    content_format: _ChatTemplateContentFormat,
) -> tuple[list[ConversationMessage], Optional[MultiModalDataDict]]:
    conversation: list[ConversationMessage] = []
    mm_tracker = MultiModalItemTracker(model_config, tokenizer)

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            mm_tracker,
            content_format,
            interleave_strings=(
                content_format == "string"
                and model_config.multimodal_config is not None
                and model_config.multimodal_config.interleave_mm_strings
            )
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    return conversation, mm_tracker.all_mm_data()


def parse_chat_messages_futures(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    content_format: _ChatTemplateContentFormat,
) -> tuple[list[ConversationMessage], Awaitable[Optional[MultiModalDataDict]]]:
    conversation: list[ConversationMessage] = []
    mm_tracker = AsyncMultiModalItemTracker(model_config, tokenizer)

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            mm_tracker,
            content_format,
            interleave_strings=(
                content_format == "string"
                and model_config.multimodal_config is not None
                and model_config.multimodal_config.interleave_mm_strings
            )
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    return conversation, mm_tracker.all_mm_data()


def apply_hf_chat_template(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    conversation: list[ConversationMessage],
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    *,
    model_config: ModelConfig,
    tokenize: bool = False,  # Different from HF's default
    **kwargs: Any,
) -> str:
    hf_chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )

    if hf_chat_template is None:
        raise ValueError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one.")

    try:

        return tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=hf_chat_template,
            tokenize=tokenize,
            **kwargs,
        )

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:

        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `transformers` while applying chat template")
        raise ValueError(str(e)) from e

def apply_mistral_chat_template(
    tokenizer: MistralTokenizer,
    messages: list[ChatCompletionMessageParam],
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    **kwargs: Any,
) -> list[int]:
    from mistral_common.exceptions import MistralCommonException

    # The return value of resolve_mistral_chat_template is always None,
    # and we won't use it.
    resolve_mistral_chat_template(
        chat_template=chat_template,
        **kwargs,
    )

    try:
        return tokenizer.apply_chat_template(
            messages=messages,
            tools=tools,
            **kwargs,
        )
    # mistral-common uses assert statements to stop processing of input
    # if input does not comply with the expected format.
    # We convert those assertion errors to ValueErrors so they can be
    # are properly caught in the preprocessing_input step
    except (AssertionError, MistralCommonException) as e:
        raise ValueError(str(e)) from e

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:

        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `mistral_common` while applying chat "
            "template")
        raise ValueError(str(e)) from e

def random_tool_call_id() -> str:
    return f"chatcmpl-tool-{random_uuid()}"
