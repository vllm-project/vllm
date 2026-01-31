# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import numpy as np
import torch
from typing_extensions import assert_never

from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import LazyLoader

from .audio import AudioResampler, AudioSpec, normalize_audio
from .inputs import (
    AudioItem,
    HfAudioItem,
    HfImageItem,
    HfVideoItem,
    ImageItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
    VisionChunk,
    VisionChunkImage,
    VisionChunkVideo,
)
from .media import MediaWithBytes

_T = TypeVar("_T")
_I = TypeVar("_I")

if TYPE_CHECKING:
    import PIL.Image as PILImage
else:
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


class ModalityDataItems(ABC, Generic[_T, _I]):
    """
    Represents data items for a modality in
    [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems].
    """

    def __init__(self, data: _T, modality: str) -> None:
        super().__init__()

        self.data: _T = data
        self.modality = modality

    def __repr__(self) -> str:
        return f"{type(self).__name__}(modality={self.modality!r}, len={len(self)})"

    def __len__(self) -> int:
        return self.get_count()

    def __getitem__(self, index: int) -> _I:
        return self.get(index)

    if TYPE_CHECKING:
        # Auto-generated
        def __iter__(self) -> Iterator[_I]: ...

    @abstractmethod
    def get_count(self) -> int:
        """Get the number of data items."""
        raise NotImplementedError

    @abstractmethod
    def get(self, index: int) -> _I:
        """Get a data item by its index."""
        raise NotImplementedError

    def get_all(self) -> list[_I]:
        """Get all data items."""
        return [self.get(idx) for idx in range(self.get_count())]

    def get_item_for_hash(self, index: int) -> object:
        return self.get(index)

    def get_all_items_for_hash(self) -> list[object]:
        return [self.get_item_for_hash(idx) for idx in range(self.get_count())]

    @abstractmethod
    def get_processor_data(self) -> Mapping[str, object]:
        """Get the data to pass to the HF processor."""
        raise NotImplementedError

    @abstractmethod
    def get_passthrough_data(self) -> Mapping[str, object]:
        """Get the data to pass directly to the model."""
        raise NotImplementedError


class ProcessorBatchItems(ModalityDataItems[Sequence[_T], _T]):
    """Base class for data items that are arranged in a list."""

    def _unwrap(self, item: _T | MediaWithBytes[_T]) -> _T:
        """Extract media from wrapper if present."""
        return item.media if isinstance(item, MediaWithBytes) else item

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> _T:
        return self._unwrap(self.data[index])

    def get_item_for_hash(self, index: int) -> _T | MediaWithBytes[_T]:
        # Return raw item for hashing (preserves original_bytes if present)
        return self.data[index]

    def get_processor_data(self) -> Mapping[str, object]:
        return {f"{self.modality}s": self.get_all()}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


def validate_embedding_ndim(
    tensor: torch.Tensor,
    modality: str,
    index: int | None = None,
) -> None:
    """Validate tensor ndim for multimodal embeddings.

    Single embeddings should be 2D (seq_len, hidden_size).
    Batched embeddings should be 3D (batch, seq_len, hidden_size).

    Args:
        tensor: The tensor to validate.
        modality: The modality name for error messages (e.g., "image", "audio").
        index: Optional index for list items, included in error messages.
    """
    if tensor.ndim < 2 or tensor.ndim > 3:
        idx_str = f" [{index}]" if index is not None else ""
        raise ValueError(
            f"{modality.capitalize()} embedding{idx_str} must be 2D "
            f"(seq_len, hidden_size) or 3D (batch, seq_len, hidden_size), "
            f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
        )


class EmbeddingItems(
    ModalityDataItems[torch.Tensor | list[torch.Tensor], torch.Tensor]
):
    """
    Base class for data items that are expressed as a batched embedding tensor,
    or a list of embedding tensors (one per item).
    """

    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        modality: str,
        expected_hidden_size: int | None = None,
    ) -> None:
        super().__init__(data, modality)

        # Validate ndim first (before hidden_size which depends on correct ndim)
        self._validate_ndim()

        # Validate hidden dimension if expected size is provided
        if expected_hidden_size is not None:
            self._validate_hidden_size(expected_hidden_size)

    def _validate_ndim(self) -> None:
        """Validate that embedding tensors have correct ndim (2D or 3D)."""
        if isinstance(self.data, torch.Tensor):
            validate_embedding_ndim(self.data, self.modality)
        else:
            # List of tensors: each should be 2D (seq_len, hidden_size)
            for idx, tensor in enumerate(self.data):
                if tensor.ndim != 2:
                    raise ValueError(
                        f"{self.modality.capitalize()} embedding [{idx}] must be "
                        f"2D (seq_len, hidden_size), got {tensor.ndim}D tensor "
                        f"with shape {tuple(tensor.shape)}"
                    )

    def _validate_hidden_size(self, expected_hidden_size: int) -> None:
        """Validate that embedding hidden dimension matches expected size.

        This validates hidden dimensions to prevent vulnerabilities: Embeddings
        with correct ndim but wrong hidden dimension could bypass initial
        checks and cause crashes during model inference when dimensions don't match.
        """
        if isinstance(self.data, torch.Tensor):
            # Batched tensor: shape is (batch, seq_len, hidden_size)
            actual_hidden_size = self.data.shape[-1]
            if actual_hidden_size != expected_hidden_size:
                raise ValueError(
                    f"{self.modality.capitalize()} embedding hidden dimension "
                    f"mismatch: got {actual_hidden_size}, but model expects "
                    f"{expected_hidden_size}. Embedding shape: {tuple(self.data.shape)}"
                )
        else:
            # List of tensors: each has shape (seq_len, hidden_size)
            for idx, tensor in enumerate(self.data):
                actual_hidden_size = tensor.shape[-1]
                if actual_hidden_size != expected_hidden_size:
                    raise ValueError(
                        f"{self.modality.capitalize()} embedding [{idx}] hidden "
                        f"dimension mismatch: got {actual_hidden_size}, but model "
                        f"expects {expected_hidden_size}. "
                        f"Embedding shape: {tuple(tensor.shape)}"
                    )

    def _unwrap(
        self, item: torch.Tensor | MediaWithBytes[torch.Tensor]
    ) -> torch.Tensor:
        """Extract media from wrapper if present."""
        return item.media if isinstance(item, MediaWithBytes) else item

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> torch.Tensor:
        return self._unwrap(self.data[index])

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {f"{self.modality}_embeds": self.data}

    def get_feature_size(self, item_idx: int) -> int:
        return len(self.get(item_idx))


class DictEmbeddingItems(
    ModalityDataItems[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
):
    """
    Base class for data items that are expressed as a dictionary of tensors.

    Usually, the dictionary keys correspond to the outputs of HF processor.
    """

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        modality: str,
        required_fields: set[str],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        from transformers.feature_extraction_utils import BatchFeature

        super().__init__(data, modality)

        missing_required_data_keys = required_fields - data.keys()
        if missing_required_data_keys:
            data_keys = set(data.keys())
            msg = (
                f"The data should contain the fields: {required_fields}, "
                f"but only found the following keys: {data_keys}"
            )
            raise ValueError(msg)

        fields_config = fields_factory(data)
        missing_required_fields = required_fields - fields_config.keys()
        if missing_required_fields:
            fields = set(fields_config.keys())
            msg = f"{required_fields=} should be a subset of {fields=}"
            raise ValueError(msg)

        self.fields_config = fields_config
        self.required_fields = required_fields

        self._kwargs = MultiModalKwargsItems.from_hf_inputs(
            BatchFeature(dict(data)),
            fields_config,
        )

    def get_count(self) -> int:
        return len(self._kwargs[self.modality])

    def get(self, index: int) -> Mapping[str, torch.Tensor]:
        return self._kwargs[self.modality][index].get_data()

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return self.data


class AudioProcessorItems(ProcessorBatchItems[HfAudioItem]):
    def __init__(self, data: Sequence[HfAudioItem] | None) -> None:
        if data is None:
            data = [None]
        super().__init__(data, "audio")

    def get_audio_length(self, item_idx: int) -> int:
        audio = self.get(item_idx)
        return len(audio)


class AudioEmbeddingItems(EmbeddingItems):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        expected_hidden_size: int | None = None,
    ) -> None:
        super().__init__(data, "audio", expected_hidden_size)


class ImageSize(NamedTuple):
    width: int
    height: int


class ImageProcessorItems(ProcessorBatchItems[HfImageItem]):
    def __init__(self, data: Sequence[HfImageItem] | None) -> None:
        if data is None:
            data = [None]
        super().__init__(data, "image")

    def get_image_size(self, item_idx: int) -> ImageSize:
        image = self.get(item_idx)

        if isinstance(image, PILImage.Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)


class ImageEmbeddingItems(EmbeddingItems):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        expected_hidden_size: int | None = None,
    ) -> None:
        super().__init__(data, "image", expected_hidden_size)


class VideoProcessorItems(ProcessorBatchItems[HfVideoItem]):
    def __init__(
        self,
        data: Sequence[HfVideoItem] | None,
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> None:
        if data is None:
            data = [None]
        super().__init__(data, "video")
        self.metadata = metadata

    def get_num_frames(self, item_idx: int) -> int:
        return len(self.get(item_idx))

    def get_frame_size(self, item_idx: int) -> ImageSize:
        image = self.get(item_idx)[0]  # Assume that the video isn't empty

        if isinstance(image, PILImage.Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)


class VideoEmbeddingItems(EmbeddingItems):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        expected_hidden_size: int | None = None,
    ) -> None:
        super().__init__(data, "video", expected_hidden_size)


class VisionChunkProcessorItems(ProcessorBatchItems[Any]):
    """Processor items for vision chunks (unified image and video chunks)."""

    def __init__(self, data: Sequence[Any]) -> None:
        super().__init__(data, "vision_chunk")


_D = TypeVar("_D", bound=ModalityDataItems[Any, Any])


class MultiModalDataItems(UserDict[str, ModalityDataItems[Any, Any]]):
    """
    As [`MultiModalDataDict`][vllm.multimodal.inputs.MultiModalDataDict], but
    normalized such that each entry corresponds to a list.
    """

    def get_count(self, modality: str, *, strict: bool = True) -> int:
        """
        Get the number of data items belonging to a modality.

        If `strict=False`, return `0` instead of raising [`KeyError`][]
        even if the modality is not found.
        """
        if modality not in self:
            if strict:
                available_modalities = set(self.keys())
                raise KeyError(
                    f"Modality {modality!r} not found. "
                    f"Available modalities: {available_modalities}"
                )

            return 0

        return self[modality].get_count()

    def get_all_counts(self) -> Mapping[str, int]:
        """Get the number of items belonging to each modality."""
        return {m: items.get_count() for m, items in self.items()}

    def get_items(
        self,
        modality: str,
        typ: type[_D] | tuple[type[_D], ...],
    ) -> _D:
        """
        Get the data items belonging to a modality,
        requiring that they belong to a certain type.
        """
        if modality not in self:
            available_modalities = set(self.keys())
            raise KeyError(
                f"Modality {modality!r} not found. "
                f"Available modalities: {available_modalities}"
            )

        items = self[modality]
        if not isinstance(items, typ):
            raise TypeError(
                f"Invalid type of data items for {modality=}. "
                f"Expected type: {typ}, but "
                f"found type: {type(items)}"
            )

        return items  # type: ignore[return-value]


ModalityDataParser: TypeAlias = Callable[
    [ModalityData[Any]], ModalityDataItems[Any, Any] | None
]


class MultiModalDataParser:
    """
    Parses [`MultiModalDataDict`][vllm.multimodal.inputs.MultiModalDataDict]
    into [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems].

    Args:
        target_sr (float, optional): Enables automatic resampling of audio
            items to the model's expected sampling rate.
        target_channels (int, optional): Target number of audio channels.
            If provided, normalizes audio to this many channels (e.g., 1 for mono).
            If None, audio channels are passed through unchanged.
        expected_hidden_size (int, optional): Expected hidden dimension for
            embedding inputs. If provided, validates that user-supplied
            embeddings have the correct hidden size to prevent crashes
            during model inference.
    """

    def __init__(
        self,
        *,
        target_sr: float | None = None,
        target_channels: int | None = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
        video_needs_metadata: bool = False,
        expected_hidden_size: int | None = None,
    ) -> None:
        super().__init__()

        self.audio_resampler = AudioResampler(
            target_sr=target_sr,
            method=audio_resample_method,
        )
        self.target_channels = target_channels
        self.video_needs_metadata = video_needs_metadata
        self.expected_hidden_size = expected_hidden_size

    @classmethod
    def is_embeddings(
        cls, data: object
    ) -> TypeGuard[torch.Tensor | list[torch.Tensor]]:
        if isinstance(data, torch.Tensor):
            return data.ndim == 3
        if is_list_of(data, torch.Tensor):
            return data[0].ndim == 2  # type: ignore[index]

        return False

    def _is_empty(self, data: object) -> TypeGuard[None]:
        if isinstance(data, list):
            return len(data) == 0
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return data.size == 0

        return False

    def _get_audio_with_sr(
        self,
        audio: AudioItem,
    ) -> tuple[np.ndarray, float | None]:
        if isinstance(audio, tuple):
            return audio
        if isinstance(audio, list):
            return np.array(audio), None
        if isinstance(audio, np.ndarray):
            return audio, None
        if isinstance(audio, torch.Tensor):
            return audio.numpy(), None

        assert_never(audio)

    def _get_video_with_metadata(
        self,
        video: VideoItem,
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        if isinstance(video, tuple):
            return video
        if isinstance(video, list):
            return np.array(video), None
        if isinstance(video, np.ndarray):
            return video, None
        if isinstance(video, torch.Tensor):
            return video.numpy(), None

        assert_never(video)

    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            return AudioProcessorItems(None)

        # also check single audio item with sampling rate
        if self._is_empty(data) or (
            isinstance(data, tuple) and self._is_empty(data[0])
        ):
            return None

        if self.is_embeddings(data):
            return AudioEmbeddingItems(data, self.expected_hidden_size)

        data_items: list[AudioItem]
        if (
            is_list_of(data, float)
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 1
            or isinstance(data, tuple)
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data  # type: ignore[assignment]

        new_audios = list[np.ndarray]()
        for data_item in data_items:
            audio, orig_sr = self._get_audio_with_sr(data_item)
            if orig_sr is None:
                new_audio = audio
            else:
                new_audio = self.audio_resampler.resample(audio, orig_sr=orig_sr)

            # Apply channel normalization if target_channels is set
            if self.target_channels is not None:
                spec = AudioSpec(target_channels=self.target_channels)
                new_audio = normalize_audio(new_audio, spec)

            new_audios.append(new_audio)

        return AudioProcessorItems(new_audios)

    def _parse_image_data(
        self,
        data: ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            return ImageProcessorItems(None)

        if self._is_empty(data):
            return None

        if self.is_embeddings(data):
            return ImageEmbeddingItems(data, self.expected_hidden_size)

        if (
            isinstance(data, (PILImage.Image, MediaWithBytes))
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 3
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return ImageProcessorItems(data_items)

    def _parse_video_data(
        self,
        data: ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            return VideoProcessorItems(None)

        if self._is_empty(data):
            return None

        if self.is_embeddings(data):
            return VideoEmbeddingItems(data, self.expected_hidden_size)

        data_items: list[VideoItem]
        if (
            is_list_of(data, PILImage.Image)
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 4
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        elif isinstance(data, tuple) and len(data) == 2:
            data_items = [data]
        else:
            data_items = data  # type: ignore[assignment]

        new_videos = list[tuple[np.ndarray, dict[str, Any] | None]]()
        metadata_lst: list[dict[str, Any] | None] = []
        for data_item in data_items:
            video, metadata = self._get_video_with_metadata(data_item)
            if self.video_needs_metadata:
                if metadata is None:
                    raise ValueError(
                        "Video metadata is required but not found in mm input. "
                        "Please check your video input in `multi_modal_data`"
                    )
                new_videos.append((video, metadata))
                metadata_lst.append(metadata)
            else:
                new_videos.append(video)

        if not self.video_needs_metadata:
            metadata = None

        return VideoProcessorItems(new_videos, metadata=metadata_lst)

    def _parse_vision_chunk_data(
        self,
        data: ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        """Parse vision chunk data (unified image and video chunks)."""
        if data is None or self._is_empty(data):
            return None
        if self.is_embeddings(data):
            raise ValueError("Do not support embedding data for vision_chunk right now")
        return VisionChunkProcessorItems(data)

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
            "vision_chunk": self._parse_vision_chunk_data,
        }

    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        subparsers = self._get_subparsers()

        mm_items = MultiModalDataItems()
        for k, v in mm_data.items():
            if k not in subparsers:
                raise ValueError(f"Unsupported modality: {k}")

            # ignore empty embedding data
            if (parsed_data := subparsers[k](v)) is not None:
                mm_items[k] = parsed_data

        return mm_items


class VisionChunkDataParser(MultiModalDataParser):
    """
    Parser for vision chunk data (unified image and video chunks).
    """
    def __init__(
        self,
        *,
        target_sr: float | None = None,
        target_channels: int | None = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
        video_needs_metadata: bool = False,
        expected_hidden_size: int | None = None,
    ) -> None:
        super().__init__(
            target_sr=target_sr,
            target_channels=target_channels,
            audio_resample_method=audio_resample_method,
            video_needs_metadata=video_needs_metadata,
            expected_hidden_size=expected_hidden_size,
        )
        assert not self.video_needs_metadata, (
            "VisionChunkDataParser does not support video metadata parsing yet."
        )

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "image": self._parse_vision_chunk_data,
            "video": self._parse_vision_chunk_data,
            "vision_chunk": self._parse_vision_chunk_data,
        }
    
    def _parse_image_data(
        self,
        data: ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            return VisionChunkProcessorItems(None)

        if self._is_empty(data):
            return None

        if self.is_embeddings(data):
            raise ValueError("Do not support embedding data for vision_chunk right now")

        if (
            isinstance(data, (PILImage.Image, MediaWithBytes))
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 3
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        data_items = [
            VisionChunkImage(
                type="image",
                images=item,
            )
            for item in data_items
        ]

        return VisionChunkProcessorItems(data_items)
    
    def _parse_video_data(
        self,
        data: ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None:
            return VisionChunkProcessorItems(None)

        if self._is_empty(data):
            return None

        if self.is_embeddings(data):
            raise ValueError("Do not support embedding data for vision_chunk right now")

        data_items: list[VideoItem]
        if (
            is_list_of(data, PILImage.Image)
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 4
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        elif isinstance(data, tuple) and len(data) == 2:
            data_items = [data]
        else:
            data_items = data  # type: ignore[assignment]

        data_items = [
            VisionChunkVideo(
                type="video",
                videos=item,
            )
            for item in data_items
        ]
        return VisionChunkProcessorItems(data_items)

        # new_videos = list[tuple[np.ndarray, dict[str, Any] | None]]()
        # metadata_lst: list[dict[str, Any] | None] = []
        # for data_item in data_items:
        #     video, metadata = self._get_video_with_metadata(data_item)
        #     if self.video_needs_metadata:
        #         if metadata is None:
        #             raise ValueError(
        #                 "Video metadata is required but not found in mm input. "
        #                 "Please check your video input in `multi_modal_data`"
        #             )
        #         new_videos.append((video, metadata))
        #         metadata_lst.append(metadata)
        #     else:
        #         new_videos.append(video)

        # if not self.video_needs_metadata:
        #     metadata = None

        # return VideoProcessorItems(new_videos, metadata=metadata_lst)

    def parse_mm_data(self, mm_data: MultiModalDataDict, modality_order: list[str] = []) -> MultiModalDataItems:
        mm_items = MultiModalDataItems()
        parsed_image = self._parse_image_data(mm_data.get("image"))
        parsed_video = self._parse_video_data(mm_data.get("video"))
        assert len(modality_order) == parsed_image.get_count() + parsed_video.get_count(), (
            "The length of modality_order should be equal to the total number of vision chunks."
        )

        for modality in modality_order:
            if modality == "image":
                if parsed_image is not None and parsed_image.get_count() > 0:
                    mm_items["vision_chunk"].data.append(
                        parsed_image.data.pop(0)
                    )
            elif modality == "video":
                if parsed_video is not None and parsed_video.get_count() > 0:
                    mm_items["vision_chunk"].data.append(
                        parsed_video.data.pop(0)
                    )
        return mm_items
