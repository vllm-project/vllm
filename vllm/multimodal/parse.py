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

from .audio import AudioResampler
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
)

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

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> _T:
        return self.data[index]

    def get_processor_data(self) -> Mapping[str, object]:
        return {f"{self.modality}s": self.data}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


class EmbeddingItems(
    ModalityDataItems[torch.Tensor | list[torch.Tensor], torch.Tensor]
):
    """
    Base class for data items that are expressed as a batched embedding tensor,
    or a list of embedding tensors (one per item).
    """

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> torch.Tensor:
        return self.data[index]

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
    def __init__(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        super().__init__(data, "audio")


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
    def __init__(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        super().__init__(data, "image")


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
    def __init__(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        super().__init__(data, "video")


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
    """

    def __init__(
        self,
        *,
        target_sr: float | None = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
        video_needs_metadata: bool = False,
    ) -> None:
        super().__init__()

        self.audio_resampler = AudioResampler(
            target_sr=target_sr,
            method=audio_resample_method,
        )
        self.video_needs_metadata = video_needs_metadata

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
            return AudioEmbeddingItems(data)

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
            return ImageEmbeddingItems(data)

        if (
            isinstance(data, PILImage.Image)
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
            return VideoEmbeddingItems(data)

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

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
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
