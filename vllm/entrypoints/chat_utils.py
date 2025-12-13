# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable
from functools import cached_property, lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast

import jinja2
import jinja2.ext
import jinja2.meta
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox
import transformers.utils.chat_template_utils as hf_chat_utils
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.responses import ResponseInputImageParam
from openai_harmony import Message as OpenAIHarmonyMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypedDict

from vllm import envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal.utils import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from vllm.tokenizers import TokenizerLike
from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils import random_uuid
from vllm.utils.collection_utils import is_list_of
from vllm.utils.func_utils import supports_kw
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import torch

    from vllm.tokenizers.mistral import MistralTokenizer
else:
    torch = LazyLoader("torch", globals(), "torch")

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
    image_embeds: str | dict[str, str] | None
    """
    The image embeddings. It can be either:
    - A single base64 string.
    - A dictionary where each value is a base64 string.
    """
    type: Required[Literal["image_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class ChatCompletionContentPartAudioEmbedsParam(TypedDict, total=False):
    audio_embeds: str | dict[str, str] | None
    """
    The audio embeddings. It can be either:
    - A single base64 string representing a serialized torch tensor.
    - A dictionary where each value is a base64 string.
    """
    type: Required[Literal["audio_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


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

    image_pil: PILImage | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain image_url.
    This is supported by OpenAI API, although it is not documented.

    Example:
    {
        "image_url": "https://example.com/image.jpg"
    }
    """

    image_url: str | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "audio_url": "https://example.com/audio.mp3"
    }
    """

    audio_url: str | None


class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "video_url": "https://example.com/video.mp4"
    }
    """

    video_url: str | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


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


ChatCompletionContentPartParam: TypeAlias = (
    OpenAIChatCompletionContentPartParam
    | ChatCompletionContentPartAudioParam
    | ChatCompletionContentPartInputAudioParam
    | ChatCompletionContentPartVideoParam
    | ChatCompletionContentPartRefusalParam
    | CustomChatCompletionContentPILImageParam
    | CustomChatCompletionContentSimpleImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartAudioEmbedsParam
    | CustomChatCompletionContentSimpleAudioParam
    | CustomChatCompletionContentSimpleVideoParam
    | str
    | CustomThinkCompletionContentParam
)


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    role: Required[str]
    """The role of the message's author."""

    content: str | list[ChatCompletionContentPartParam]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: str | None
    """Tool call that this message is responding to."""

    tool_calls: Iterable[ChatCompletionMessageToolCallParam] | None
    """The tool calls generated by the model, such as function calls."""

    reasoning: str | None
    """The reasoning content for interleaved thinking."""


ChatCompletionMessageParam: TypeAlias = (
    OpenAIChatCompletionMessageParam
    | CustomChatCompletionMessageParam
    | OpenAIHarmonyMessage
)


# TODO: Make fields ReadOnly once mypy supports it
class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    """The role of the message's author."""

    content: str | None | list[dict[str, str]]
    """The contents of the message"""

    tool_call_id: str | None
    """Tool call that this message is responding to."""

    name: str | None
    """The name of the function to call"""

    tool_calls: Iterable[ChatCompletionMessageToolCallParam] | None
    """The tool calls generated by the model, such as function calls."""

    reasoning: str | None
    """The reasoning content for interleaved thinking."""

    reasoning_content: str | None
    """Deprecated: The reasoning content for interleaved thinking."""


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
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str | None = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


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
        varname for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")
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


def _try_extract_ast(chat_template: str) -> jinja2.nodes.Template | None:
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
    chat_template: str | None,
    **kwargs: Any,
) -> str | None:
    if chat_template is not None or kwargs.get("chat_template_kwargs") is not None:
        raise ValueError(
            "'chat_template' or 'chat_template_kwargs' cannot be overridden "
            "for mistral tokenizer."
        )

    return None


_PROCESSOR_CHAT_TEMPLATES = dict[tuple[str, bool], str | None]()
"""
Used in `_try_get_processor_chat_template` to avoid calling
`cached_get_processor` again if the processor fails to be loaded.

This is needed because `lru_cache` does not cache when an exception happens.
"""


def _try_get_processor_chat_template(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model_config: ModelConfig,
) -> str | None:
    cache_key = (tokenizer.name_or_path, model_config.trust_remote_code)
    if cache_key in _PROCESSOR_CHAT_TEMPLATES:
        return _PROCESSOR_CHAT_TEMPLATES[cache_key]

    try:
        processor = cached_get_processor(
            tokenizer.name_or_path,
            processor_cls=(
                PreTrainedTokenizer,
                PreTrainedTokenizerFast,
                ProcessorMixin,
            ),
            trust_remote_code=model_config.trust_remote_code,
        )
        if (
            isinstance(processor, ProcessorMixin)
            and hasattr(processor, "chat_template")
            and (chat_template := processor.chat_template) is not None
        ):
            _PROCESSOR_CHAT_TEMPLATES[cache_key] = chat_template
            return chat_template
    except Exception:
        logger.debug(
            "Failed to load AutoProcessor chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    _PROCESSOR_CHAT_TEMPLATES[cache_key] = None
    return None


def resolve_hf_chat_template(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: ModelConfig,
) -> str | None:
    # 1st priority: The given chat template
    if chat_template is not None:
        return chat_template

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        chat_template = _try_get_processor_chat_template(tokenizer, model_config)
        if chat_template is not None:
            return chat_template

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug(
            "Failed to load AutoTokenizer chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    # 4th priority: Predefined fallbacks
    path = get_chat_template_fallback_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=model_config.tokenizer,
    )
    if path is not None:
        logger.info_once(
            "Loading chat template fallback for %s as there isn't one "
            "defined on HF Hub.",
            tokenizer.name_or_path,
        )
        chat_template = load_chat_template(path)
    else:
        logger.debug_once(
            "There is no chat template fallback for %s", tokenizer.name_or_path
        )

    return chat_template


def _resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    tokenizer: TokenizerLike | None,
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

    jinja_text = (
        hf_chat_template
        if isinstance(hf_chat_template, str)
        else load_chat_template(chat_template, is_literal=True)
    )

    detected_format = (
        "string"
        if jinja_text is None
        else _detect_content_format(jinja_text, default="string")
    )

    return detected_format


@lru_cache
def _log_chat_template_content_format(
    chat_template: str | None,
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
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: TokenizerLike | None,
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


ModalityStr = Literal["image", "audio", "video", "image_embeds", "audio_embeds"]
_T = TypeVar("_T")


def _extract_embeds(tensors: list[torch.Tensor]):
    if len(tensors) == 0:
        return tensors

    if len(tensors) == 1:
        tensors[0]._is_single_item = True  # type: ignore
        return tensors[0]  # To keep backwards compatibility for single item input

    first_shape = tensors[0].shape
    if all(t.shape == first_shape for t in tensors):
        return torch.stack(tensors)

    return tensors


def _get_embeds_data(items_by_modality: dict[str, list[Any]], modality: str):
    embeds_key = f"{modality}_embeds"
    embeds = items_by_modality[embeds_key]

    if len(embeds) == 0:
        return embeds
    if is_list_of(embeds, torch.Tensor):
        return _extract_embeds(embeds)
    if is_list_of(embeds, dict):
        if not embeds:
            return {}

        first_keys = set(embeds[0].keys())
        if any(set(item.keys()) != first_keys for item in embeds[1:]):
            raise ValueError(
                "All dictionaries in the list of embeddings must have the same keys."
            )

        return {k: _extract_embeds([item[k] for item in embeds]) for k in first_keys}

    return embeds


class BaseMultiModalItemTracker(ABC, Generic[_T]):
    """
    Tracks multi-modal items in a given request and ensures that the number
    of multi-modal items in a given request does not exceed the configured
    maximum per prompt.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self._model_config = model_config

        self._items_by_modality = defaultdict[str, list[_T | None]](list)
        self._uuids_by_modality = defaultdict[str, list[str | None]](list)

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
    def allowed_media_domains(self):
        return self._model_config.allowed_media_domains

    @property
    def mm_registry(self):
        return MULTIMODAL_REGISTRY

    @cached_property
    def mm_processor(self):
        return self.mm_registry.create_processor(self.model_config)

    def add(
        self,
        modality: ModalityStr,
        item: _T | None,
        uuid: str | None = None,
    ) -> str | None:
        """
        Add a multi-modal item to the current prompt and returns the
        placeholder string to use, if any.

        An optional uuid can be added which serves as a unique identifier of the
        media.
        """
        input_modality = modality.replace("_embeds", "")
        num_items = len(self._items_by_modality[modality]) + 1

        self.mm_processor.validate_num_items(input_modality, num_items)

        self._items_by_modality[modality].append(item)
        self._uuids_by_modality[modality].append(uuid)

        return self.model_cls.get_placeholder_str(modality, num_items)

    def all_mm_uuids(self) -> MultiModalUUIDDict | None:
        if not self._items_by_modality:
            return None

        uuids_by_modality = dict(self._uuids_by_modality)
        if "image" in uuids_by_modality and "image_embeds" in uuids_by_modality:
            raise ValueError("Mixing raw image and embedding inputs is not allowed")
        if "audio" in uuids_by_modality and "audio_embeds" in uuids_by_modality:
            raise ValueError("Mixing raw audio and embedding inputs is not allowed")

        mm_uuids = {}
        if "image_embeds" in uuids_by_modality:
            mm_uuids["image"] = uuids_by_modality["image_embeds"]
        if "image" in uuids_by_modality:
            mm_uuids["image"] = uuids_by_modality["image"]  # UUIDs of images
        if "audio_embeds" in uuids_by_modality:
            mm_uuids["audio"] = uuids_by_modality["audio_embeds"]
        if "audio" in uuids_by_modality:
            mm_uuids["audio"] = uuids_by_modality["audio"]  # UUIDs of audios
        if "video" in uuids_by_modality:
            mm_uuids["video"] = uuids_by_modality["video"]  # UUIDs of videos

        return mm_uuids

    @abstractmethod
    def create_parser(self) -> "BaseMultiModalContentParser":
        raise NotImplementedError


class MultiModalItemTracker(BaseMultiModalItemTracker[object]):
    def all_mm_data(self) -> MultiModalDataDict | None:
        if not self._items_by_modality:
            return None

        items_by_modality = dict(self._items_by_modality)
        if "image" in items_by_modality and "image_embeds" in items_by_modality:
            raise ValueError("Mixing raw image and embedding inputs is not allowed")
        if "audio" in items_by_modality and "audio_embeds" in items_by_modality:
            raise ValueError("Mixing raw audio and embedding inputs is not allowed")

        mm_inputs = {}
        if "image_embeds" in items_by_modality:
            mm_inputs["image"] = _get_embeds_data(items_by_modality, "image")
        if "image" in items_by_modality:
            mm_inputs["image"] = items_by_modality["image"]  # A list of images
        if "audio_embeds" in items_by_modality:
            mm_inputs["audio"] = _get_embeds_data(items_by_modality, "audio")
        if "audio" in items_by_modality:
            mm_inputs["audio"] = items_by_modality["audio"]  # A list of audios
        if "video" in items_by_modality:
            mm_inputs["video"] = items_by_modality["video"]  # A list of videos

        return mm_inputs

    def create_parser(self) -> "BaseMultiModalContentParser":
        return MultiModalContentParser(self)


class AsyncMultiModalItemTracker(BaseMultiModalItemTracker[Awaitable[object]]):
    async def all_mm_data(self) -> MultiModalDataDict | None:
        if not self._items_by_modality:
            return None

        coros_by_modality = {
            modality: [item or asyncio.sleep(0) for item in items]
            for modality, items in self._items_by_modality.items()
        }
        items_by_modality: dict[str, list[object | None]] = {
            modality: await asyncio.gather(*coros)
            for modality, coros in coros_by_modality.items()
        }
        if "image" in items_by_modality and "image_embeds" in items_by_modality:
            raise ValueError("Mixing raw image and embedding inputs is not allowed")
        if "audio" in items_by_modality and "audio_embeds" in items_by_modality:
            raise ValueError("Mixing raw audio and embedding inputs is not allowed")

        mm_inputs = {}
        if "image_embeds" in items_by_modality:
            mm_inputs["image"] = _get_embeds_data(items_by_modality, "image")
        if "image" in items_by_modality:
            mm_inputs["image"] = items_by_modality["image"]  # A list of images
        if "audio_embeds" in items_by_modality:
            mm_inputs["audio"] = _get_embeds_data(items_by_modality, "audio")
        if "audio" in items_by_modality:
            mm_inputs["audio"] = items_by_modality["audio"]  # A list of audios
        if "video" in items_by_modality:
            mm_inputs["video"] = items_by_modality["video"]  # A list of videos

        return mm_inputs

    def create_parser(self) -> "BaseMultiModalContentParser":
        return AsyncMultiModalContentParser(self)


class BaseMultiModalContentParser(ABC):
    def __init__(self) -> None:
        super().__init__()

        # stores model placeholders list with corresponding
        # general MM placeholder:
        # {
        #   "<##IMAGE##>": ["<image>", "<image>", "<image>"],
        #   "<##AUDIO##>": ["<audio>", "<audio>"]
        # }
        self._placeholder_storage: dict[str, list] = defaultdict(list)

    def _add_placeholder(self, modality: ModalityStr, placeholder: str | None):
        mod_placeholder = MODALITY_PLACEHOLDERS_MAP[modality]
        if placeholder:
            self._placeholder_storage[mod_placeholder].append(placeholder)

    def mm_placeholder_storage(self) -> dict[str, list]:
        return dict(self._placeholder_storage)

    @abstractmethod
    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_image_embeds(
        self,
        image_embeds: str | dict[str, str] | None,
        uuid: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_audio_embeds(
        self,
        audio_embeds: str | dict[str, str] | None,
        uuid: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        raise NotImplementedError


class MultiModalContentParser(BaseMultiModalContentParser):
    def __init__(self, tracker: MultiModalItemTracker) -> None:
        super().__init__()

        self._tracker = tracker
        multimodal_config = self._tracker.model_config.multimodal_config
        media_io_kwargs = getattr(multimodal_config, "media_io_kwargs", None)

        self._connector: MediaConnector = MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=media_io_kwargs,
            allowed_local_media_path=tracker.allowed_local_media_path,
            allowed_media_domains=tracker.allowed_media_domains,
        )

    @property
    def model_config(self) -> ModelConfig:
        return self._tracker.model_config

    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        image = self._connector.fetch_image(image_url) if image_url else None

        placeholder = self._tracker.add("image", image, uuid)
        self._add_placeholder("image", placeholder)

    def parse_image_embeds(
        self,
        image_embeds: str | dict[str, str] | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise ValueError(
                "You must set `--enable-mm-embeds` to input `image_embeds`"
            )

        if isinstance(image_embeds, dict):
            embeds = {
                k: self._connector.fetch_image_embedding(v)
                for k, v in image_embeds.items()
            }
            placeholder = self._tracker.add("image_embeds", embeds, uuid)

        if isinstance(image_embeds, str):
            embedding = self._connector.fetch_image_embedding(image_embeds)
            placeholder = self._tracker.add("image_embeds", embedding, uuid)

        if image_embeds is None:
            placeholder = self._tracker.add("image_embeds", None, uuid)

        self._add_placeholder("image", placeholder)

    def parse_audio_embeds(
        self,
        audio_embeds: str | dict[str, str] | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise ValueError(
                "You must set `--enable-mm-embeds` to input `audio_embeds`"
            )

        if isinstance(audio_embeds, dict):
            embeds = {
                k: self._connector.fetch_audio_embedding(v)
                for k, v in audio_embeds.items()
            }
            placeholder = self._tracker.add("audio_embeds", embeds, uuid)
        elif isinstance(audio_embeds, str):
            embedding = self._connector.fetch_audio_embedding(audio_embeds)
            placeholder = self._tracker.add("audio_embeds", embedding, uuid)
        else:
            placeholder = self._tracker.add("audio_embeds", None, uuid)

        self._add_placeholder("audio", placeholder)

    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None:
        placeholder = self._tracker.add("image", image_pil, uuid)
        self._add_placeholder("image", placeholder)

    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        audio = self._connector.fetch_audio(audio_url) if audio_url else None

        placeholder = self._tracker.add("audio", audio, uuid)
        self._add_placeholder("audio", placeholder)

    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None:
        if input_audio:
            audio_data = input_audio.get("data", "")
            audio_format = input_audio.get("format", "")
            if audio_data:
                audio_url = f"data:audio/{audio_format};base64,{audio_data}"
            else:
                # If a UUID is provided, audio data may be empty.
                audio_url = None
        else:
            audio_url = None

        return self.parse_audio(audio_url, uuid)

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        video = self._connector.fetch_video(video_url=video_url) if video_url else None

        placeholder = self._tracker.add("video", video, uuid)
        self._add_placeholder("video", placeholder)


class AsyncMultiModalContentParser(BaseMultiModalContentParser):
    def __init__(self, tracker: AsyncMultiModalItemTracker) -> None:
        super().__init__()

        self._tracker = tracker
        multimodal_config = self._tracker.model_config.multimodal_config
        media_io_kwargs = getattr(multimodal_config, "media_io_kwargs", None)
        self._connector: MediaConnector = MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=media_io_kwargs,
            allowed_local_media_path=tracker.allowed_local_media_path,
            allowed_media_domains=tracker.allowed_media_domains,
        )

    @property
    def model_config(self) -> ModelConfig:
        return self._tracker.model_config

    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        image_coro = self._connector.fetch_image_async(image_url) if image_url else None

        placeholder = self._tracker.add("image", image_coro, uuid)
        self._add_placeholder("image", placeholder)

    def parse_image_embeds(
        self,
        image_embeds: str | dict[str, str] | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise ValueError(
                "You must set `--enable-mm-embeds` to input `image_embeds`"
            )

        future: asyncio.Future[str | dict[str, str] | None] = asyncio.Future()

        if isinstance(image_embeds, dict):
            embeds = {
                k: self._connector.fetch_image_embedding(v)
                for k, v in image_embeds.items()
            }
            future.set_result(embeds)

        if isinstance(image_embeds, str):
            embedding = self._connector.fetch_image_embedding(image_embeds)
            future.set_result(embedding)

        if image_embeds is None:
            future.set_result(None)

        placeholder = self._tracker.add("image_embeds", future, uuid)
        self._add_placeholder("image", placeholder)

    def parse_audio_embeds(
        self,
        audio_embeds: str | dict[str, str] | None,
        uuid: str | None = None,
    ) -> None:
        mm_config = self.model_config.get_multimodal_config()
        if not mm_config.enable_mm_embeds:
            raise ValueError(
                "You must set `--enable-mm-embeds` to input `audio_embeds`"
            )

        logger.info(
            "🎵 Parsing audio_embeds: type=%s, uuid=%s, is_dict=%s, "
            "is_str=%s, is_none=%s",
            type(audio_embeds).__name__,
            uuid,
            isinstance(audio_embeds, dict),
            isinstance(audio_embeds, str),
            audio_embeds is None,
        )

        future: asyncio.Future[str | dict[str, str] | None] = asyncio.Future()

        if isinstance(audio_embeds, dict):
            logger.info(
                "🎵 Processing dict audio_embeds with %d entries",
                len(audio_embeds),
            )
            embeds = {
                k: self._connector.fetch_audio_embedding(v)
                for k, v in audio_embeds.items()
            }
            future.set_result(embeds)
            logger.info(
                "🎵 Successfully loaded %d audio embeddings from dict",
                len(embeds),
            )

        if isinstance(audio_embeds, str):
            base64_size = len(audio_embeds)
            logger.info(
                "🎵 Processing base64 audio_embeds: %d chars (%.2f KB)",
                base64_size,
                base64_size / 1024,
            )
            embedding = self._connector.fetch_audio_embedding(audio_embeds)
            future.set_result(embedding)
            logger.info(
                "🎵 Successfully loaded audio embedding tensor: shape=%s, dtype=%s",
                embedding.shape,
                embedding.dtype,
            )

        if audio_embeds is None:
            logger.info("🎵 Audio embeds is None (UUID-only reference)")
            future.set_result(None)

        placeholder = self._tracker.add("audio_embeds", future, uuid)
        self._add_placeholder("audio", placeholder)
        logger.info("🎵 Added audio_embeds placeholder with uuid=%s", uuid)

    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None:
        future: asyncio.Future[Image.Image | None] = asyncio.Future()
        if image_pil:
            future.set_result(image_pil)
        else:
            future.set_result(None)

        placeholder = self._tracker.add("image", future, uuid)
        self._add_placeholder("image", placeholder)

    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        audio_coro = self._connector.fetch_audio_async(audio_url) if audio_url else None

        placeholder = self._tracker.add("audio", audio_coro, uuid)
        self._add_placeholder("audio", placeholder)

    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None:
        if input_audio:
            audio_data = input_audio.get("data", "")
            audio_format = input_audio.get("format", "")
            if audio_data:
                audio_url = f"data:audio/{audio_format};base64,{audio_data}"
            else:
                # If a UUID is provided, audio data may be empty.
                audio_url = None
        else:
            audio_url = None

        return self.parse_audio(audio_url, uuid)

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        video = (
            self._connector.fetch_video_async(video_url=video_url)
            if video_url
            else None
        )

        placeholder = self._tracker.add("video", video, uuid)
        self._add_placeholder("video", placeholder)


def validate_chat_template(chat_template: Path | str | None):
    """Raises if the provided chat template appears invalid."""
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError("the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if (
            not any(c in chat_template for c in JINJA_CHARS)
            and not Path(chat_template).exists()
        ):
            # Try to find the template in the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            if not builtin_template_path.exists():
                raise ValueError(
                    f"The supplied chat template string ({chat_template}) "
                    f"appears path-like, but doesn't exist! "
                    f"Tried: {chat_template} and {builtin_template_path}"
                )

    else:
        raise TypeError(f"{type(chat_template)} is not a valid chat template type")


def _load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError(
                "chat_template is expected to be read directly from its value"
            )

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            # Try to load from the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            try:
                with open(builtin_template_path) as f:
                    return f.read()
            except OSError:
                msg = (
                    f"The supplied chat template ({chat_template}) "
                    f"looks like a file path, but it failed to be opened. "
                    f"Tried: {chat_template} and {builtin_template_path}. "
                    f"Reason: {e}"
                )
                raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return _load_chat_template(chat_template, is_literal=True)


_cached_load_chat_template = lru_cache(_load_chat_template)


def load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    return _cached_load_chat_template(chat_template, is_literal=is_literal)


def _get_interleaved_text_prompt(
    placeholder_storage: dict[str, list], texts: list[str]
) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            texts[idx] = placeholder_storage[elem].pop(0)

    return "\n".join(texts)


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(
    placeholder_storage: dict[str, list],
    texts: list[str],
    interleave_strings: bool,
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
            raise ValueError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items."
            )

        missing_placeholders.extend([placeholder] * placeholder_counts[placeholder])

    # NOTE: Default behaviour: we always add missing placeholders
    # at the front of the prompt, if interleave_strings=False
    return "\n".join(missing_placeholders + [text_prompt])


# No need to validate using Pydantic again
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_AudioEmbedsParser = partial(cast, ChatCompletionContentPartAudioEmbedsParam)
_InputAudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
_RefusalParser = partial(cast, ChatCompletionContentPartRefusalParam)
_PILImageParser = partial(cast, CustomChatCompletionContentPILImageParam)
_ThinkParser = partial(cast, CustomThinkCompletionContentParam)
# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ResponsesInputImageParser = TypeAdapter(ResponseInputImageParam).validate_python
_ContentPart: TypeAlias = str | dict[str, str] | InputAudio | PILImage

# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], _ContentPart],
] = {
    "text": lambda part: _TextParser(part).get("text", None),
    "thinking": lambda part: _ThinkParser(part).get("thinking", None),
    "input_text": lambda part: _TextParser(part).get("text", None),
    "output_text": lambda part: _TextParser(part).get("text", None),
    "input_image": lambda part: _ResponsesInputImageParser(part).get("image_url", None),
    "image_url": lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "image_embeds": lambda part: _ImageEmbedsParser(part).get("image_embeds", None),
    "audio_embeds": lambda part: _AudioEmbedsParser(part).get("audio_embeds", None),
    "image_pil": lambda part: _PILImageParser(part).get("image_pil", None),
    "audio_url": lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    "input_audio": lambda part: _InputAudioParser(part).get("input_audio", None),
    "refusal": lambda part: _RefusalParser(part).get("refusal", None),
    "video_url": lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
}


def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam,
) -> tuple[str, _ContentPart]:
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
        if part.get("input_audio") is not None:
            input_audio_params = cast(dict[str, str], part)
            return "input_audio", input_audio_params
        if "video_url" in part:
            video_params = cast(CustomChatCompletionContentSimpleVideoParam, part)
            video_url = video_params.get("video_url", None)
            if isinstance(video_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                video_url = video_url.get("url", None)
            return "video_url", video_url
        # Raise an error if no 'type' or direct URL is found.
        raise ValueError("Missing 'type' field in multimodal part.")

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


PART_TYPES_TO_SKIP_NONE_CONTENT = (
    "text",
    "refusal",
)


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
            mm_placeholder_storage, texts, interleave_strings
        )
    else:
        text_prompt = "\n".join(texts)

    return [ConversationMessage(role=role, content=text_prompt)]


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

    if part_type in ("text", "input_text", "output_text", "refusal", "thinking"):
        str_content = cast(str, content)
        if wrap_dicts:
            return {"type": "text", "text": str_content}
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
        content = cast(str | dict[str, str], content) if content is not None else None
        mm_parser.parse_image_embeds(content, uuid)
        modality = "image"
    elif part_type == "audio_embeds":
        content = cast(str | dict[str, str], content) if content is not None else None
        mm_parser.parse_audio_embeds(content, uuid)
        modality = "audio"
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
    else:
        raise NotImplementedError(f"Unknown part type: {part_type}")

    return (
        {"type": modality}
        if wrap_dicts
        else (MODALITY_PLACEHOLDERS_MAP[modality] if interleave_strings else None)
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
    reasoning = message.get("reasoning") or message.get("reasoning_content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [ChatCompletionContentPartTextParam(type="text", text=content)]
    result = _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        mm_tracker,
        wrap_dicts=(content_format == "openai"),
        interleave_strings=interleave_strings,
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
        if (
            message["role"] == "assistant"
            and "tool_calls" in message
            and isinstance(message["tool_calls"], list)
        ):
            for item in message["tool_calls"]:
                # if arguments is None or empty string, set to {}
                if content := item["function"].get("arguments"):
                    if not isinstance(content, (dict, list)):
                        item["function"]["arguments"] = json.loads(content)
                else:
                    item["function"]["arguments"] = {}


def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: _ChatTemplateContentFormat,
) -> tuple[
    list[ConversationMessage],
    MultiModalDataDict | None,
    MultiModalUUIDDict | None,
]:
    conversation: list[ConversationMessage] = []
    mm_tracker = MultiModalItemTracker(model_config)

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
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    return conversation, mm_tracker.all_mm_data(), mm_tracker.all_mm_uuids()


def parse_chat_messages_futures(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: _ChatTemplateContentFormat,
) -> tuple[
    list[ConversationMessage],
    Awaitable[MultiModalDataDict | None],
    MultiModalUUIDDict | None,
]:
    conversation: list[ConversationMessage] = []
    mm_tracker = AsyncMultiModalItemTracker(model_config)

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
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    return conversation, mm_tracker.all_mm_data(), mm_tracker.all_mm_uuids()


# adapted from https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/chat_template_utils.py#L398-L412
# only preserve the parse function used to resolve chat template kwargs
class AssistantTracker(jinja2.ext.Extension):
    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        call = self.call_method("_generation_support")
        call_block = jinja2.nodes.CallBlock(call, [], [], body)
        return call_block.set_lineno(lineno)


def _resolve_chat_template_kwargs(
    chat_template: str,
):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )
    parsed_content = env.parse(chat_template)
    template_vars = jinja2.meta.find_undeclared_variables(parsed_content)
    return template_vars


_cached_resolve_chat_template_kwargs = lru_cache(_resolve_chat_template_kwargs)


@lru_cache
def _get_hf_base_chat_template_params() -> frozenset[str]:
    # Get standard parameters from HuggingFace's base tokenizer class.
    # This dynamically extracts parameters from PreTrainedTokenizer's
    # apply_chat_template method, ensuring compatibility with tokenizers
    # that use **kwargs to receive standard parameters.

    # Read signature from HF's base class - the single source of truth
    base_sig = inspect.signature(PreTrainedTokenizer.apply_chat_template)
    # Exclude VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) placeholders
    return frozenset(
        p.name
        for p in base_sig.parameters.values()
        if p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )


def resolve_chat_template_kwargs(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    chat_template: str,
    chat_template_kwargs: dict[str, Any],
    raise_on_unexpected: bool = True,
) -> dict[str, Any]:
    # We exclude chat_template from kwargs here, because
    # chat template has been already resolved at this stage
    unexpected_vars = {"chat_template", "tokenize"}
    if raise_on_unexpected and (
        unexpected_in_kwargs := unexpected_vars & chat_template_kwargs.keys()
    ):
        raise ValueError(
            "Found unexpected chat template kwargs from request: "
            f"{unexpected_in_kwargs}"
        )

    fn_kw = {
        k
        for k in chat_template_kwargs
        if supports_kw(tokenizer.apply_chat_template, k, allow_var_kwargs=False)
    }
    template_vars = _cached_resolve_chat_template_kwargs(chat_template)

    # Allow standard HF parameters even if tokenizer uses **kwargs to receive them
    hf_base_params = _get_hf_base_chat_template_params()

    accept_vars = (fn_kw | template_vars | hf_base_params) - unexpected_vars
    return {k: v for k, v in chat_template_kwargs.items() if k in accept_vars}


def apply_hf_chat_template(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    conversation: list[ConversationMessage],
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: ModelConfig,
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
            "does not define one."
        )

    resolved_kwargs = resolve_chat_template_kwargs(
        tokenizer=tokenizer,
        chat_template=hf_chat_template,
        chat_template_kwargs=kwargs,
    )

    try:
        return tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=hf_chat_template,
            tokenize=False,
            **resolved_kwargs,
        )

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `transformers` while applying chat template"
        )
        raise ValueError(str(e)) from e


def apply_mistral_chat_template(
    tokenizer: "MistralTokenizer",
    messages: list[ChatCompletionMessageParam],
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
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
    # properly caught in the preprocessing_input step
    except (AssertionError, MistralCommonException) as e:
        raise ValueError(str(e)) from e

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `mistral_common` while applying chat template"
        )
        raise ValueError(str(e)) from e


def get_history_tool_calls_cnt(conversation: list[ConversationMessage]):
    idx = 0
    for msg in conversation:
        if msg["role"] == "assistant":
            tool_calls = msg.get("tool_calls")
            idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
    return idx


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    else:
        # by default return random
        return f"chatcmpl-tool-{random_uuid()}"
