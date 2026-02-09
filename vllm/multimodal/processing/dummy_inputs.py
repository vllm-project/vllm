# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.logger import init_logger

from ..inputs import MultiModalDataDict
from ..parse import MultiModalDataItems
from .context import BaseProcessingInfo

_I = TypeVar("_I", bound=BaseProcessingInfo)

logger = init_logger(__name__)


@dataclass
class ProcessorInputs:
    """
    Represents the keyword arguments to
    [`vllm.multimodal.processing.BaseMultiModalProcessor.apply`][].
    """

    prompt: str | list[int]
    mm_items: MultiModalDataItems
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenization_kwargs: Mapping[str, object] = field(default_factory=dict)


class BaseDummyInputsBuilder(ABC, Generic[_I]):
    """
    Abstract base class that constructs the dummy data to profile
    multi-modal models.
    """

    def __init__(self, info: _I) -> None:
        super().__init__()

        self.info = info

    @abstractmethod
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        Build the text input corresponding to `mm_counts`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        """
        Build the multimodal input which, after processing, results in
        the maximum possible number of placeholder tokens.

        Args:
            seq_len: Sequence length
            mm_counts: Count of items per modality
            mm_options: Configurable options per modality (optional).
                       If None, use model defaults for backward compatibility.
                       If provided, models can use these to customize dummy
                       data generation.
        """
        raise NotImplementedError

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        """
        Build the input which, after processing, results in
        the maximum possible number of placeholder tokens.

        Args:
            seq_len: Sequence length
            mm_counts: Count of items per modality
            mm_options: Configurable options per modality (optional)
        """
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data, validate=False)

        tokenization_kwargs = {"truncation": False}

        return ProcessorInputs(
            prompt=dummy_text,
            mm_items=dummy_mm_items,
            tokenization_kwargs=tokenization_kwargs,
        )

    def _get_dummy_audios(
        self,
        *,
        length: int,
        num_audios: int,
        overrides: AudioDummyOptions | None = None,
    ) -> list[npt.NDArray]:
        if num_audios == 0:
            return []
        if overrides and overrides.length:
            if overrides.length > length:
                logger.warning(
                    "audio.length override (%d) exceeds model's "
                    "maximum length (%d), will be ignored",
                    overrides.length,
                    length,
                )
            length = min(length, overrides.length)
        audio = np.zeros((length,))
        return [audio] * num_audios

    def _get_dummy_images(
        self,
        *,
        width: int,
        height: int,
        num_images: int,
        overrides: ImageDummyOptions | None = None,
    ) -> list[Image.Image]:
        if num_images == 0:
            return []
        if overrides:
            if overrides.width:
                if overrides.width > width:
                    logger.warning(
                        "image.width override (%d) exceeds model's "
                        "maximum width (%d), will be ignored",
                        overrides.width,
                        width,
                    )
                width = min(width, overrides.width)
            if overrides.height:
                if overrides.height > height:
                    logger.warning(
                        "image.height override (%d) exceeds model's "
                        "maximum height (%d), will be ignored",
                        overrides.height,
                        height,
                    )
                height = min(height, overrides.height)
        image = Image.new("RGB", (width, height), color=255)
        return [image] * num_images

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides: VideoDummyOptions | None = None,
    ) -> list[npt.NDArray]:
        if num_videos == 0:
            return []
        if overrides:
            if overrides.num_frames:
                if overrides.num_frames > num_frames:
                    logger.warning(
                        "video.num_frames override (%d) exceeds model's "
                        "maximum number of frames (%d), will be ignored",
                        overrides.num_frames,
                        num_frames,
                    )
                num_frames = min(num_frames, overrides.num_frames)
            if overrides.width:
                if overrides.width > width:
                    logger.warning(
                        "video.width override (%d) exceeds model's "
                        "maximum width (%d), will be ignored",
                        overrides.width,
                        width,
                    )
                width = min(width, overrides.width)
            if overrides.height:
                if overrides.height > height:
                    logger.warning(
                        "video.height override (%d) exceeds model's "
                        "maximum height (%d), will be ignored",
                        overrides.height,
                        height,
                    )
                height = min(height, overrides.height)
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        return [video] * num_videos
