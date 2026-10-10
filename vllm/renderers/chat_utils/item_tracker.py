# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from functools import cached_property
from itertools import accumulate
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    TypeAlias,
    TypeVar,
    cast,
)

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
    MultiModalSharedField,
    VisionChunk,
    VisionChunkImage,
    VisionChunkVideo,
)
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.transformers_utils.processor import get_video_processor_cls_name
from vllm.utils import random_uuid
from vllm.utils.collection_utils import is_list_of, is_list_of_numbers
from vllm.utils.import_utils import LazyLoader

from .content_parser import (
    AsyncMultiModalContentParser,
    BaseMultiModalContentParser,
    MultiModalContentParser,
)
from .types import ModalityStr

_T = TypeVar("_T")
_AsyncMultiModalItem: TypeAlias = Callable[[], Awaitable[tuple[object, str | None]]]

if TYPE_CHECKING:
    import torch
    import transformers
else:
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)


_REQUIRE_MM_PROCESSOR_ERROR: Final[str] = (
    "Resolving modality {modality!r} requires a multimodal processor "
    "but none is available."
)


class BaseMultiModalItemTracker(ABC, Generic[_T]):
    """Tracks multi-modal items in a given request and ensures that the number
    of multi-modal items in a given request does not exceed the configured
    maximum per prompt.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__()

        self._model_config = model_config
        self._media_io_kwargs = media_io_kwargs

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
    def media_io_kwargs(self) -> dict[str, dict[str, Any]] | None:
        return self._media_io_kwargs or (
            self._model_config.multimodal_config.media_io_kwargs
            if self._model_config.multimodal_config
            else None
        )

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

    @property
    def video_processor_name(self) -> str | None:
        return get_video_processor_cls_name(self.model_config)

    def add(self, modality: ModalityStr, item: _T) -> str | None:
        """Add a multi-modal item to the current prompt and returns the
        placeholder string to use, if any.

        An optional uuid can be added which serves as a unique identifier of the
        media.

        Note:
            `prompt_embeds` bypass MM-processor validation because they are
            pre-computed embeddings that do not go through any HF processor, encoder,
            or model-specific placeholder logic. The corresponding placeholder string is
            managed by the parser via `_add_placeholder`, so we return None here.

        """
        add_info = self._validate_add(modality)
        if add_info is None:
            self._items_by_modality["prompt_embeds"].append(item)
            return None

        input_modality, original_modality, use_vision_chunk, num_items = add_info

        # Track original modality for vision_chunk items
        if use_vision_chunk:
            self._items_by_modality[input_modality].append(item)  # type: ignore
            self._modality_order["vision_chunk"].append(original_modality)
        else:
            self._items_by_modality[original_modality].append(item)

        return self.model_cls.get_placeholder_str(modality, num_items)

    def _validate_add(self, modality: ModalityStr) -> tuple[str, str, bool, int] | None:
        """Validate that one more item of the modality can be tracked."""
        if modality == "prompt_embeds":
            return None

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
            mm_config is not None
            and mm_config.enable_mm_embeds
            and mm_config.get_limit_per_prompt(input_modality) == 0
            and original_modality.endswith("_embeds")
        ):
            # Skip validation: embeddings bypass limit when enable_mm_embeds=True
            pass
        else:
            self.mm_processor.info.validate_num_items(input_modality, num_items)

        return input_modality, original_modality, use_vision_chunk, num_items

    @abstractmethod
    def create_parser(
        self, mm_processor_kwargs: dict[str, Any] | None = None
    ) -> "BaseMultiModalContentParser":
        raise NotImplementedError


class MultiModalItemTracker(BaseMultiModalItemTracker[tuple[object, str | None]]):
    def resolve_items(
        self,
    ) -> tuple[MultiModalDataDict | None, MultiModalUUIDDict | None]:
        if not self._items_by_modality:
            return None, None

        # Text-only models (`is_multimodal_model=False`) with inputs of
        # modality `prompt_embeds` have no MM processor since `prompt_embeds` are
        # pre-computed and require no processing, so we pass `None`.
        mm_processor = (
            self.mm_processor if self._model_config.is_multimodal_model else None
        )
        return _resolve_items(
            dict(self._items_by_modality),
            mm_processor,
            self._modality_order,
        )

    def create_parser(
        self, mm_processor_kwargs: dict[str, Any] | None = None
    ) -> "BaseMultiModalContentParser":
        return MultiModalContentParser(self, mm_processor_kwargs=mm_processor_kwargs)


class AsyncMultiModalItemTracker(BaseMultiModalItemTracker[_AsyncMultiModalItem]):
    async def resolve_items(
        self,
    ) -> tuple[MultiModalDataDict | None, MultiModalUUIDDict | None]:
        if not self._items_by_modality:
            return None, None

        # Fetch all modalities together. Each tracked item is already an
        # independent awaitable, and the async connector offloads blocking
        # decode work, so waiting for one modality before starting the next
        # needlessly adds their latency.
        # Keep the original group and item order when rebuilding the result.
        item_groups = list(self._items_by_modality.items())
        items = [item for _, group in item_groups for item in group]
        results = await asyncio.gather(
            *(item() for item in items), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                # Gathering with return_exceptions=True lets every task finish
                # (or itself fail) before we raise, instead of abandoning
                # still-in-flight fetches (real network/thread-pool work) the
                # moment the first one fails.
                raise result

        resolved_items_by_modality: dict[str, list[Any]] = {}
        result_idx = 0
        for modality, group in item_groups:
            next_result_idx = result_idx + len(group)
            resolved_items_by_modality[modality] = results[result_idx:next_result_idx]
            result_idx = next_result_idx

        mm_processor = (
            self.mm_processor if self._model_config.is_multimodal_model else None
        )
        return _resolve_items(
            resolved_items_by_modality,
            mm_processor,
            self._modality_order,
        )

    def create_parser(
        self, mm_processor_kwargs: dict[str, Any] | None = None
    ) -> "BaseMultiModalContentParser":
        return AsyncMultiModalContentParser(
            self, mm_processor_kwargs=mm_processor_kwargs
        )


def _resolve_items(
    items_by_modality: dict[str, list[tuple[object, str | None]]],
    mm_processor: BaseMultiModalProcessor | None,
    modality_order: dict[str, list[str]],
) -> tuple[MultiModalDataDict, MultiModalUUIDDict]:
    """Materialize the tracker's per-modality items into `mm_data` / `mm_uuids`.

    Note:
        `mm_processor` is `None` for text-only models (no registered HF
        processor) whose only modality is `prompt_embeds`. Every other
        modality requires a processor, enforced by the guard below.

    """
    if "image" in items_by_modality and "image_embeds" in items_by_modality:
        raise VLLMValidationError(
            "Mixing raw image and embedding inputs is not allowed",
            parameter="image_embeds",
        )
    if "audio" in items_by_modality and "audio_embeds" in items_by_modality:
        raise VLLMValidationError(
            "Mixing raw audio and embedding inputs is not allowed",
            parameter="audio_embeds",
        )
    if "video" in items_by_modality and "video_embeds" in items_by_modality:
        raise VLLMValidationError(
            "Mixing raw video and embedding inputs is not allowed",
            parameter="video_embeds",
        )
    # `prompt_embeds` bypasses HF MM processors. Every other modality requires one.
    processor_modalities = items_by_modality.keys() - {"prompt_embeds"}
    if processor_modalities and mm_processor is None:
        raise RuntimeError(
            _REQUIRE_MM_PROCESSOR_ERROR.format(modality=processor_modalities)
        )

    mm_data = {}
    mm_uuids = {}
    if "image_embeds" in items_by_modality:
        assert mm_processor is not None
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
        assert mm_processor is not None
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
    if "video_embeds" in items_by_modality:
        assert mm_processor is not None
        mm_data["video"] = _get_embeds_data(
            "video",
            [data for data, uuid in items_by_modality["video_embeds"]],
            mm_processor,
        )
        mm_uuids["video"] = [uuid for data, uuid in items_by_modality["video_embeds"]]
    if "vision_chunk" in items_by_modality:
        assert mm_processor is not None
        # Process vision_chunk items - extract from (data, modality) tuples
        # and convert to VisionChunk types with proper UUID handling
        processed_chunks, vision_chunk_uuids = _resolve_vision_chunk_items(
            items_by_modality["vision_chunk"],
            mm_processor,
            modality_order.get("vision_chunk", []),
        )
        mm_data["vision_chunk"] = processed_chunks
        mm_uuids["vision_chunk"] = vision_chunk_uuids
    if "prompt_embeds" in items_by_modality:
        mm_data["prompt_embeds"] = [
            data for data, _uuid in items_by_modality["prompt_embeds"]
        ]

    return mm_data, mm_uuids


# Backward compatibility for single item input
class _BatchedSingleItemField(MultiModalSharedField):
    pass


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
        metadata_fields = mm_processor.info.data_parser.placeholder_metadata_fields(
            modality
        )
        data_items = [
            {
                key: _parse_metadata_array(key, value, metadata_fields)
                if isinstance(value, list)
                else value
                for key, value in item.items()
            }
            for item in data_items
        ]
        return _merge_embeds(data_items, mm_processor)

    raise NotImplementedError(type(data_items))


def _parse_metadata_array(key: str, value: list, metadata_fields: set[str]):
    if key not in metadata_fields:
        raise VLLMValidationError(f"JSON arrays are only supported for metadata: {key}")
    if not is_list_of_numbers(value):
        raise VLLMValidationError(f"Metadata {key} must be a finite numeric array.")
    dtype = torch.float64 if any(isinstance(v, float) for v in value) else torch.int64
    try:
        return torch.tensor(value, dtype=dtype)
    except (ValueError, TypeError, OverflowError, RuntimeError) as error:
        raise VLLMValidationError(f"Invalid metadata array: {key}") from error


def _merge_embeds(
    data_items: list[dict[str, "torch.Tensor"]],
    mm_processor: BaseMultiModalProcessor,
):
    if not data_items:
        return {}

    first_keys = set(data_items[0].keys())
    if any(set(item.keys()) != first_keys for item in data_items[1:]):
        raise VLLMValidationError(
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
