# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, NamedTuple, TypeVar, cast

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

from .inputs import (
    MultiModalDataDict,
    MultiModalEncDecInputs,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalPlaceholderDict,
)
from .processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
)

logger = init_logger(__name__)


@dataclass
class ProcessorInputs:
    """
    Represents the keyword arguments to
    [`vllm.multimodal.processing.BaseMultiModalProcessor.apply`][].
    """

    prompt: str | list[int]
    mm_data: MultiModalDataDict
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenization_kwargs: Mapping[str, object] = field(default_factory=dict)


class DummyEncoderData(NamedTuple):
    """Dummy data used for profiling."""

    prompt_token_ids: list[int]


class DummyDecoderData(NamedTuple):
    """Dummy data used for profiling."""

    prompt_token_ids: list[int]
    multi_modal_data: MultiModalKwargsItems
    multi_modal_placeholders: MultiModalPlaceholderDict


_I = TypeVar("_I", bound=BaseProcessingInfo)


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

        # Use the unified function for both legacy and configurable cases
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)

        tokenization_kwargs = {"truncation": False}

        return ProcessorInputs(
            prompt=dummy_text,
            mm_data=dummy_mm_data,
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


class MultiModalProfiler(Generic[_I]):
    """
    Contains code for running memory profiling for multi-modal models.
    """

    def __init__(
        self,
        processor: BaseMultiModalProcessor[_I],
    ) -> None:
        super().__init__()

        self.processor = processor

    @property
    def processing_info(self) -> BaseProcessingInfo:
        return self.processor.info

    @property
    def dummy_inputs(self) -> BaseDummyInputsBuilder[_I]:
        return self.processor.dummy_inputs

    def get_mm_limits(self) -> Mapping[str, int]:
        return self.processor.allowed_mm_limits

    def _get_dummy_mm_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalInputs:
        if mm_counts is None:
            mm_counts = self.get_mm_limits()

        factory = self.dummy_inputs
        processor_inputs = factory.get_dummy_processor_inputs(
            seq_len, mm_counts, mm_options
        )

        return self.processor.apply(
            prompt=processor_inputs.prompt,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
            tokenization_kwargs=processor_inputs.tokenization_kwargs,
        )

    def _get_mm_num_tokens(
        self,
        mm_inputs: MultiModalInputs,
        mm_embeddings_only: bool = True,
    ) -> Mapping[str, int]:
        placeholders_by_modality = mm_inputs["mm_placeholders"]

        return {
            modality: sum(
                item.get_num_embeds() if mm_embeddings_only else item.length
                for item in placeholders
            )
            for modality, placeholders in placeholders_by_modality.items()
        }

    def get_encoder_dummy_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> DummyEncoderData:
        mm_inputs = self._get_dummy_mm_inputs(seq_len, mm_counts, mm_options)
        mm_inputs = cast(MultiModalEncDecInputs, mm_inputs)

        # For encoder-decoder models, use encoder prompt token ids instead of
        # decoder prompt to construct dummy seq_data for encoder profiling.
        encoder_prompt_token_ids = mm_inputs["encoder_prompt_token_ids"]

        total_len = len(encoder_prompt_token_ids)

        processor = cast(EncDecMultiModalProcessor, self.processor)
        if processor.pad_dummy_encoder_prompt:
            num_tokens_to_pad = max(total_len, seq_len) - total_len
            encoder_prompt_token_ids.extend([0] * num_tokens_to_pad)

        return DummyEncoderData(encoder_prompt_token_ids)

    def get_decoder_dummy_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> DummyDecoderData:
        mm_inputs = self._get_dummy_mm_inputs(seq_len, mm_counts, mm_options)

        prompt_token_ids = mm_inputs["prompt_token_ids"]
        total_len = len(prompt_token_ids)

        if total_len < seq_len:
            prompt_token_ids.extend([0] * (seq_len - total_len))

        return DummyDecoderData(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data=mm_inputs["mm_kwargs"].require_data(),
            multi_modal_placeholders=mm_inputs["mm_placeholders"],
        )

    def _get_mm_max_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_embeddings_only: bool = True,
    ) -> Mapping[str, int]:
        if mm_counts is None:
            mm_counts = self.get_mm_limits()

        max_tokens_per_item = self.processing_info.get_mm_max_tokens_per_item(
            seq_len=seq_len,
            mm_counts=mm_counts,
        )
        if max_tokens_per_item is not None:
            return {
                modality: max_tokens
                for modality, max_tokens in max_tokens_per_item.items()
                if mm_counts.get(modality, 0) > 0
            }

        mm_inputs = self._get_dummy_mm_inputs(seq_len, mm_counts)
        return self._get_mm_num_tokens(mm_inputs, mm_embeddings_only=mm_embeddings_only)

    def get_mm_max_contiguous_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        """
        Returns the maximum length of the multimodal (image placeholders+text)
        tokens, including any break/text tokens in-between image embeddings.

        `<im_start> [IMG] [IMG] [IMG] <row_break> [IMG] [IMG] [IMG] <im_end>`
        Returns 9, even when the number of image embeddings is 6.

        This is important to take into account when profiling and
        initializing the encoder cache size.
        """
        return self._get_mm_max_tokens(seq_len, mm_counts, mm_embeddings_only=False)
