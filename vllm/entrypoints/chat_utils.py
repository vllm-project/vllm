# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable
from functools import cached_property, lru_cache, partial
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
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

# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypedDict

from vllm import envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
    MultiModalSharedField,
    VisionChunk,
    VisionChunkImage,
    VisionChunkVideo,
)
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.utils import random_uuid
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import torch
    import transformers
else:
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)


def __getattr__(name: str):
    if name == "resolve_hf_chat_template":
        from vllm.renderers.hf import resolve_chat_template

        warnings.warn(
            "`vllm.entrypoints.chat_utils.resolve_hf_chat_template` has been moved to "
            "`vllm.renderers.hf.resolve_chat_template`. "
            "The old name will be removed in v0.16.",
            DeprecationWarning,
            stacklevel=2,
        )

        return resolve_chat_template

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ChatTemplateResolutionError(ValueError):
    """Raised when chat template resolution fails.

    This is a subclass of ValueError for backward compatibility with
    existing exception handlers.
    """


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

    tools: list[ChatCompletionFunctionToolParam] | None
    """The tools for developer role."""


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

    tools: list[ChatCompletionFunctionToolParam] | None
    """The tools for developer role."""


# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

# After resolving "auto"
ChatTemplateContentFormat = Literal["string", "openai"]


ModalityStr = Literal[
    "image", "audio", "video", "image_embeds", "audio_embeds", "vision_chunk"
]
_T = TypeVar("_T")


# Backward compatibility for single item input
class _BatchedSingleItemField(MultiModalSharedField):
    pass


def _detect_field(
    tensors: list[torch.Tensor],
    mm_processor: BaseMultiModalProcessor,
):
    first_item = tensors[0]
    hidden_size = mm_processor.info.ctx.model_config.get_inputs_embeds_size()

    if (
        len(tensors) == 1
        and first_item.ndim == 3
        and first_item.shape[0] == 1
        and first_item.shape[-1] == hidden_size
    ):
        logger.warning(
            "Batched multi-modal embedding inputs are deprecated for Chat API. "
            "Please pass a separate content part for each multi-modal item."
        )
        return _BatchedSingleItemField(batch_size=1)

    first_shape = first_item.shape
    if all(t.shape == first_shape for t in tensors):
        return MultiModalBatchedField()

    size_per_item = [len(tensor) for tensor in tensors]
    slice_idxs = [0, *accumulate(size_per_item)]
    slices = [
        (slice(slice_idxs[i], slice_idxs[i + 1]),) for i in range(len(size_per_item))
    ]
    return MultiModalFlatField(slices=slices)


def _merge_embeds(
    data_items: list[dict[str, "torch.Tensor"]],
    mm_processor: BaseMultiModalProcessor,
):
    if not data_items:
        return {}

    first_keys = set(data_items[0].keys())
    if any(set(item.keys()) != first_keys for item in data_items[1:]):
        raise ValueError(
            "All dictionaries in the list of embeddings must have the same keys."
        )

    fields = {
        key: _detect_field([item[key] for item in data_items], mm_processor)
        for key in first_keys
    }
    data_merged = {
        key: field._reduce_data([item[key] for item in data_items], pin_memory=False)
        for key, field in fields.items()
    }

    try:
        # TODO: Support per-request mm_processor_kwargs
        parsed_configs = mm_processor._get_mm_fields_config(
            transformers.BatchFeature(data_merged),
            {},
        )
        parsed_fields = {key: parsed_configs[key].field for key in first_keys}
        keys_to_update = [
            key
            for key in first_keys
            if (
                fields[key] != parsed_fields[key]
                and not isinstance(fields[key], _BatchedSingleItemField)
            )
        ]

        for key in keys_to_update:
            data_merged[key] = parsed_fields[key]._reduce_data(
                [item[key] for item in data_items], pin_memory=False
            )
    except Exception:
        logger.exception(
            "Error when parsing merged embeddings. "
            "Falling back to auto-detected fields."
        )

    return data_merged


def _get_embeds_data(
    modality: str,
    data_items: list[Any],
    mm_processor: BaseMultiModalProcessor,
):
    if len(data_items) == 0:
        return data_items

    if all(item is None for item in data_items):
        return data_items

    if is_list_of(data_items, torch.Tensor):
        embeds_key = f"{modality}_embeds"
        dict_items = [{embeds_key: item} for item in data_items]
        return _merge_embeds(dict_items, mm_processor)[embeds_key]

    if is_list_of(data_items, dict):
        return _merge_embeds(data_items, mm_processor)

    raise NotImplementedError(type(data_items))


class BaseMultiModalItemTracker(ABC, Generic[_T]):
    """
    Tracks multi-modal items in a given request and ensures that the number
    of multi-modal items in a given request does not exceed the configured
    maximum per prompt.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self._model_config = model_config

        self._items_by_modality = defaultdict[str, list[_T]](list)
        # Track original modality for each vision_chunk item (image or video)
        self._modality_order = defaultdict[str, list[str]](list)

    @cached_property
    def use_unified_vision_chunk_modality(self) -> bool:
        """Check if model uses unified vision_chunk modality for images/videos."""
        return getattr(self._model_config.hf_config, "use_unified_vision_chunk", False)

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

    def add(self, modality: ModalityStr, item: _T) -> str | None:
        """
        Add a multi-modal item to the current prompt and returns the
        placeholder string to use, if any.

        An optional uuid can be added which serves as a unique identifier of the
        media.
        """
        input_modality = modality.replace("_embeds", "")
        original_modality = modality
        use_vision_chunk = (
            self.use_unified_vision_chunk_modality
            and original_modality in ["video", "image"]
        )

        # If use_unified_vision_chunk_modality is enabled,
        # map image/video to vision_chunk
        if use_vision_chunk:
            # To avoid validation fail
            # because models with use_unified_vision_chunk_modality=True
            # will only accept vision_chunk modality.
            input_modality = "vision_chunk"
            num_items = len(self._items_by_modality[input_modality]) + 1
        else:
            num_items = len(self._items_by_modality[original_modality]) + 1

        mm_config = self.model_config.multimodal_config
        if (
            mm_config.enable_mm_embeds
            and mm_config.get_limit_per_prompt(input_modality) == 0
        ):
            # Skip validation: embeddings bypass limit when enable_mm_embeds=True
            pass
        else:
            self.mm_processor.info.validate_num_items(input_modality, num_items)

        # Track original modality for vision_chunk items
        if use_vision_chunk:
            self._items_by_modality[input_modality].append(item)  # type: ignore
            self._modality_order["vision_chunk"].append(original_modality)
        else:
            self._items_by_modality[original_modality].append(item)

        return self.model_cls.get_placeholder_str(modality, num_items)

    @abstractmethod
    def create_parser(self) -> "BaseMultiModalContentParser":
        raise NotImplementedError


def _resolve_vision_chunk_items(
    vision_chunk_items: list[tuple[object, str | None]],
    mm_processor: BaseMultiModalProcessor,
    vision_chunks_modality_order: list[str],
):
    # Process vision_chunk items - extract from (data, modality) tuples
    # and convert to VisionChunk types with proper UUID handling
    vision_chunks_uuids = [uuid for data, uuid in vision_chunk_items]

    assert len(vision_chunk_items) == len(vision_chunks_modality_order), (
        f"vision_chunk items ({len(vision_chunk_items)}) and "
        f"modality_order ({len(vision_chunks_modality_order)}) must have same length"
    )

    processed_chunks: list[VisionChunk] = []
    video_idx = 0
    for inner_modality, (data, uuid) in zip(
        vision_chunks_modality_order, vision_chunk_items
    ):
        if inner_modality == "image":
            # Cast data to proper type for image
            # Use .media (PIL.Image) directly to avoid redundant
            # bytes→PIL conversion in media_processor
            if hasattr(data, "media"):
                image_data = data.media  # type: ignore[union-attr]
                processed_chunks.append(
                    VisionChunkImage(type="image", image=image_data, uuid=uuid)
                )
            else:
                processed_chunks.append(data)  # type: ignore[arg-type]
        elif inner_modality == "video":
            # For video, we may need to split into chunks
            # if processor supports it
            # For now, just wrap as a video chunk placeholder
            if hasattr(mm_processor, "split_video_chunks") and data is not None:
                try:
                    video_uuid = uuid or random_uuid()
                    # video await result is (video_data, video_meta) tuple
                    if isinstance(data, tuple) and len(data) >= 1:
                        video_data = data[0]
                    else:
                        video_data = data
                    video_chunks = mm_processor.split_video_chunks(video_data)
                    for i, vc in enumerate(video_chunks):
                        processed_chunks.append(
                            VisionChunkVideo(
                                type="video_chunk",
                                video_chunk=vc["video_chunk"],
                                uuid=f"{video_uuid}-{i}",
                                video_idx=video_idx,
                                prompt=vc["prompt"],
                            )
                        )
                    video_idx += 1
                except Exception as e:
                    logger.warning("Failed to split video chunks: %s", e)
                    processed_chunks.append(data)  # type: ignore[arg-type]
            else:
                processed_chunks.append(data)  # type: ignore[arg-type]
    return processed_chunks, vision_chunks_uuids


def _resolve_items(
    items_by_modality: dict[str, list[tuple[object, str | None]]],
    mm_processor: BaseMultiModalProcessor,
    modality_order: dict[str, list[str]],
) -> tuple[MultiModalDataDict, MultiModalUUIDDict]:
    if "image" in items_by_modality and "image_embeds" in items_by_modality:
        raise ValueError("Mixing raw image and embedding inputs is not allowed")
    if "audio" in items_by_modality and "audio_embeds" in items_by_modality:
        raise ValueError("Mixing raw audio and embedding inputs is not allowed")

    mm_data = {}
    mm_uuids = {}
    if "image_embeds" in items_by_modality:
        mm_data["image"] = _get_embeds_data(
            "image",
            [data for data, uuid in items_by_modality["image_embeds"]],
            mm_processor,
        )
        mm_uuids["image"] = [uuid for data, uuid in items_by_modality["image_embeds"]]
    if "image" in items_by_modality:
        mm_data["image"] = [data for data, uuid in items_by_modality["image"]]
        mm_uuids["image"] = [uuid for data, uuid in items_by_modality["image"]]
    if "audio_embeds" in items_by_modality:
        mm_data["audio"] = _get_embeds_data(
            "audio",
            [data for data, uuid in items_by_modality["audio_embeds"]],
            mm_processor,
        )
        mm_uuids["audio"] = [uuid for data, uuid in items_by_modality["audio_embeds"]]
    if "audio" in items_by_modality:
        mm_data["audio"] = [data for data, uuid in items_by_modality["audio"]]
        mm_uuids["audio"] = [uuid for data, uuid in items_by_modality["audio"]]
    if "video" in items_by_modality:
        mm_data["video"] = [data for data, uuid in items_by_modality["video"]]
        mm_uuids["video"] = [uuid for data, uuid in items_by_modality["video"]]
    if "vision_chunk" in items_by_modality:
        # Process vision_chunk items - extract from (data, modality) tuples
        # and convert to VisionChunk types with proper UUID handling
        processed_chunks, vision_chunk_uuids = _resolve_vision_chunk_items(
            items_by_modality["vision_chunk"],
            mm_processor,
            modality_order.get("vision_chunk", []),
        )
        mm_data["vision_chunk"] = processed_chunks
        mm_uuids["vision_chunk"] = vision_chunk_uuids

    return mm_data, mm_uuids


class MultiModalItemTracker(BaseMultiModalItemTracker[tuple[object, str | None]]):
    def resolve_items(
        self,
    ) -> tuple[MultiModalDataDict | None, MultiModalUUIDDict | None]:
        if not self._items_by_modality:
            return None, None

        return _resolve_items(
            dict(self._items_by_modality), self.mm_processor, self._modality_order
        )

    def create_parser(self) -> "BaseMultiModalContentParser":
        return MultiModalContentParser(self)


class AsyncMultiModalItemTracker(
    BaseMultiModalItemTracker[Awaitable[tuple[object, str | None]]]
):
    async def resolve_items(
        self,
    ) -> tuple[MultiModalDataDict | None, MultiModalUUIDDict | None]:
        if not self._items_by_modality:
            return None, None

        resolved_items_by_modality = {
            modality: await asyncio.gather(*coros)
            for modality, coros in self._items_by_modality.items()
        }

        return _resolve_items(
            resolved_items_by_modality, self.mm_processor, self._modality_order
        )

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

        placeholder = self._tracker.add("image", (image, uuid))
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
            placeholder = self._tracker.add("image_embeds", (embeds, uuid))

        if isinstance(image_embeds, str):
            embedding = self._connector.fetch_image_embedding(image_embeds)
            placeholder = self._tracker.add("image_embeds", (embedding, uuid))

        if image_embeds is None:
            placeholder = self._tracker.add("image_embeds", (None, uuid))

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
            placeholder = self._tracker.add("audio_embeds", (embeds, uuid))
        elif isinstance(audio_embeds, str):
            embedding = self._connector.fetch_audio_embedding(audio_embeds)
            placeholder = self._tracker.add("audio_embeds", (embedding, uuid))
        else:
            placeholder = self._tracker.add("audio_embeds", (None, uuid))

        self._add_placeholder("audio", placeholder)

    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None:
        placeholder = self._tracker.add("image", (image_pil, uuid))
        self._add_placeholder("image", placeholder)

    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        audio = self._connector.fetch_audio(audio_url) if audio_url else None

        placeholder = self._tracker.add("audio", (audio, uuid))
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

        placeholder = self._tracker.add("video", (video, uuid))
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

    async def _image_with_uuid_async(self, image_url: str | None, uuid: str | None):
        image = (
            await self._connector.fetch_image_async(image_url) if image_url else None
        )
        return image, uuid

    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        coro = self._image_with_uuid_async(image_url, uuid)

        placeholder = self._tracker.add("image", coro)
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

        future = asyncio.Future[
            tuple[torch.Tensor | dict[str, torch.Tensor] | None, str | None]
        ]()

        if isinstance(image_embeds, dict):
            embeds = {
                k: self._connector.fetch_image_embedding(v)
                for k, v in image_embeds.items()
            }
            future.set_result((embeds, uuid))

        if isinstance(image_embeds, str):
            embedding = self._connector.fetch_image_embedding(image_embeds)
            future.set_result((embedding, uuid))

        if image_embeds is None:
            future.set_result((None, uuid))

        placeholder = self._tracker.add("image_embeds", future)
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

        future = asyncio.Future[
            tuple[torch.Tensor | dict[str, torch.Tensor] | None, str | None]
        ]()

        if isinstance(audio_embeds, dict):
            embeds = {
                k: self._connector.fetch_audio_embedding(v)
                for k, v in audio_embeds.items()
            }
            future.set_result((embeds, uuid))

        if isinstance(audio_embeds, str):
            embedding = self._connector.fetch_audio_embedding(audio_embeds)
            future.set_result((embedding, uuid))

        if audio_embeds is None:
            future.set_result((None, uuid))

        placeholder = self._tracker.add("audio_embeds", future)
        self._add_placeholder("audio", placeholder)

    def parse_image_pil(
        self,
        image_pil: Image.Image | None,
        uuid: str | None = None,
    ) -> None:
        future = asyncio.Future[tuple[Image.Image | None, str | None]]()
        if image_pil:
            future.set_result((image_pil, uuid))
        else:
            future.set_result((None, uuid))

        placeholder = self._tracker.add("image", future)
        self._add_placeholder("image", placeholder)

    async def _audio_with_uuid_async(self, audio_url: str | None, uuid: str | None):
        audio = (
            await self._connector.fetch_audio_async(audio_url) if audio_url else None
        )
        return audio, uuid

    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None:
        coro = self._audio_with_uuid_async(audio_url, uuid)

        placeholder = self._tracker.add("audio", coro)
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

    async def _video_with_uuid_async(self, video_url: str | None, uuid: str | None):
        video = (
            await self._connector.fetch_video_async(video_url) if video_url else None
        )
        return video, uuid

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        coro = self._video_with_uuid_async(video_url, uuid)

        placeholder = self._tracker.add("video", coro)
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
    if text_prompt:
        return "\n".join(missing_placeholders + [text_prompt])
    else:
        return "\n".join(missing_placeholders)


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
    content_format: ChatTemplateContentFormat,
    interleave_strings: bool,
) -> list[ConversationMessage]:
    role = message["role"]
    content = message.get("content")
    reasoning = message.get("reasoning")

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
                # if arguments is None or empty string, set to {}
                if content := item["function"].get("arguments"):
                    if not isinstance(content, (dict, list)):
                        item["function"]["arguments"] = json.loads(content)
                else:
                    item["function"]["arguments"] = {}


def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
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

    mm_data, mm_uuids = mm_tracker.resolve_items()

    return conversation, mm_data, mm_uuids


async def parse_chat_messages_async(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
) -> tuple[
    list[ConversationMessage],
    MultiModalDataDict | None,
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

    mm_data, mm_uuids = await mm_tracker.resolve_items()

    return conversation, mm_data, mm_uuids


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
