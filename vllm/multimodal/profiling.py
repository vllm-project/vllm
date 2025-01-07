from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from PIL import Image

from .inputs import MultiModalDataDict
from .processing import BaseProcessingInfo


@dataclass
class ProcessorInputs:
    """Keyword arguments to :meth:`BaseMultiModalProcessor`."""
    prompt_text: str
    mm_data: MultiModalDataDict
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)


_I = TypeVar("_I", bound=BaseProcessingInfo)


class BaseDummyDataBuilder(ABC, Generic[_I]):
    """
    Abstract base class that constructs the dummy data to profile
    multi-modal models.
    """

    def __init__(self, info: _I) -> None:
        super().__init__()

        self.info = info

    @abstractmethod
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """
        Build the input which, after processing, results in
        `self.info.get_mm_max_tokens_per_item()` placeholder tokens.
        """
        raise NotImplementedError

    def _get_dummy_audios(
        self,
        *,
        length: int,
        num_audios: int,
    ) -> list[npt.NDArray]:
        audio = np.zeros((length, ))
        return [audio] * num_audios

    def _get_dummy_images(
        self,
        *,
        width: int,
        height: int,
        num_images: int,
    ) -> list[Image.Image]:
        image = Image.new("RGB", (width, height), color=0)
        return [image] * num_images

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ) -> list[npt.NDArray]:
        video = np.zeros((num_frames, width, height, 3))
        return [video] * num_videos
