# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (TYPE_CHECKING, Any, Generic, Literal, NamedTuple, Optional,
                    TypeVar, Union)

import numpy as np
import torch
from typing_extensions import TypeAlias, TypeGuard, assert_never

from vllm.utils import LazyLoader, is_list_of

from .audio import AudioResampler
from .inputs import (AudioItem, HfAudioItem, HfImageItem, HfVideoItem,
                     ImageItem, ModalityData, MultiModalDataDict,
                     MultiModalFieldConfig, MultiModalKwargs, VideoItem)

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

        self.data = data
        self.modality = modality

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r}, "
                f"len={len(self)})")

    def __len__(self) -> int:
        return self.get_count()

    def __getitem__(self, index: int) -> _I:
        return self.get(index)

    if TYPE_CHECKING:
        # Auto-generated
        def __iter__(self) -> Iterator[_I]:
            ...

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


class EmbeddingItems(ModalityDataItems[Union[torch.Tensor, list[torch.Tensor]],
                                       torch.Tensor]):
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


class DictEmbeddingItems(ModalityDataItems[Mapping[str, torch.Tensor],
                                           Mapping[str, torch.Tensor]]):
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
            msg = (f"The data should contain the fields: {required_fields}, "
                   f"but only found the following keys: {data_keys}")
            raise ValueError(msg)

        fields_config = fields_factory(data)
        missing_required_fields = required_fields - fields_config.keys()
        if missing_required_fields:
            fields = set(fields_config.keys())
            msg = f"{required_fields=} should be a subset of {fields=}"
            raise ValueError(msg)

        self.fields_config = fields_config
        self.required_fields = required_fields

        self._kwargs = MultiModalKwargs.from_hf_inputs(
            BatchFeature(dict(data)),
            fields_config,
        )

    def get_count(self) -> int:
        return self._kwargs.get_item_count(self.modality)

    def get(self, index: int) -> Mapping[str, torch.Tensor]:
        return {
            k: v.data
            for k, v in self._kwargs.get_item(self.modality, index).items()
        }

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return self.data


class AudioProcessorItems(ProcessorBatchItems[HfAudioItem]):

    def __init__(self, data: Sequence[HfAudioItem]) -> None:
        super().__init__(data, "audio")

    def get_audio_length(self, item_idx: int) -> int:
        audio = self.get(item_idx)
        return len(audio)


class AudioEmbeddingItems(EmbeddingItems):

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
        super().__init__(data, "audio")


class ImageSize(NamedTuple):
    width: int
    height: int


class ImageProcessorItems(ProcessorBatchItems[HfImageItem]):

    def __init__(self, data: Sequence[HfImageItem]) -> None:
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

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
        super().__init__(data, "image")


class VideoProcessorItems(ProcessorBatchItems[HfVideoItem]):

    def __init__(self, data: Sequence[HfVideoItem]) -> None:
        super().__init__(data, "video")

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

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
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
                raise KeyError(f"Modality {modality!r} not found. "
                               f"Available modalities: {available_modalities}")

            return 0

        return self[modality].get_count()

    def get_all_counts(self) -> Mapping[str, int]:
        """Get the number of items belonging to each modality."""
        return {m: items.get_count() for m, items in self.items()}

    def get_items(
        self,
        modality: str,
        typ: Union[type[_D], tuple[type[_D], ...]],
    ) -> _D:
        """
        Get the data items belonging to a modality,
        requiring that they belong to a certain type.
        """
        if modality not in self:
            available_modalities = set(self.keys())
            raise KeyError(f"Modality {modality!r} not found. "
                           f"Available modalities: {available_modalities}")

        items = self[modality]
        if not isinstance(items, typ):
            raise TypeError(f"Invalid type of data items for {modality=}. "
                            f"Expected type: {typ}, but "
                            f"found type: {type(items)}")

        return items  # type: ignore[return-value]


ModalityDataParser: TypeAlias = Callable[[ModalityData[Any]],
                                         Optional[ModalityDataItems[Any, Any]]]


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
        target_sr: Optional[float] = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
    ) -> None:
        super().__init__()

        self.audio_resampler = AudioResampler(
            target_sr=target_sr,
            method=audio_resample_method,
        )

    def _is_embeddings(
            self, data: object
    ) -> TypeGuard[Union[torch.Tensor, list[torch.Tensor]]]:
        if isinstance(data, torch.Tensor):
            return data.ndim == 3
        if is_list_of(data, torch.Tensor):
            return data[0].ndim == 2

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
    ) -> tuple[np.ndarray, Optional[float]]:
        if isinstance(audio, tuple):
            return audio
        if isinstance(audio, list):
            return np.array(audio), None
        if isinstance(audio, np.ndarray):
            return audio, None
        if isinstance(audio, torch.Tensor):
            return audio.numpy(), None

        assert_never(audio)

    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        # also check single audio item with sampling rate
        if self._is_empty(data) or (isinstance(data, tuple)
                                    and self._is_empty(data[0])):
            return None

        if self._is_embeddings(data):
            return AudioEmbeddingItems(data)

        if (is_list_of(data, float)
                or isinstance(data,
                              (np.ndarray, torch.Tensor)) and data.ndim == 1
                or isinstance(data, tuple)):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        new_audios = list[np.ndarray]()
        for data_item in data_items:
            audio, orig_sr = self._get_audio_with_sr(data_item)
            if orig_sr is None:
                new_audio = audio
            else:
                new_audio = self.audio_resampler.resample(audio,
                                                          orig_sr=orig_sr)

            new_audios.append(new_audio)

        return AudioProcessorItems(new_audios)

    def _parse_image_data(
        self,
        data: ModalityData[ImageItem],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if self._is_empty(data):
            return None

        if self._is_embeddings(data):
            return ImageEmbeddingItems(data)

        if (isinstance(data, PILImage.Image)
                or isinstance(data,
                              (np.ndarray, torch.Tensor)) and data.ndim == 3):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return ImageProcessorItems(data_items)

    def _parse_video_data(
        self,
        data: ModalityData[VideoItem],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if self._is_empty(data):
            return None

        if self._is_embeddings(data):
            return VideoEmbeddingItems(data)

        if (is_list_of(data, PILImage.Image)
                or isinstance(data,
                              (np.ndarray, torch.Tensor)) and data.ndim == 4):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return VideoProcessorItems(data_items)

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
        }

    def parse_mm_data(self,
                      mm_data: MultiModalDataDict) -> MultiModalDataItems:
        subparsers = self._get_subparsers()

        mm_items = MultiModalDataItems()
        for k, v in mm_data.items():
            if k not in subparsers:
                raise ValueError(f"Unsupported modality: {k}")

            # ignore empty embedding data
            if (parsed_data := subparsers[k](v)) is not None:
                mm_items[k] = parsed_data

        return mm_items
