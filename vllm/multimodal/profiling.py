# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, NamedTuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from typing_extensions import TypeVar

from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.logger import init_logger

from .inputs import (
    MultiModalDataDict,
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


_Proc = TypeVar(
    "_Proc", covariant=True, bound=BaseProcessingInfo, default=BaseProcessingInfo
)


class BaseProfilingInfo(ABC, Generic[_Proc]):
    def __init__(self, processing_info: _Proc) -> None:
        super().__init__()

        self.processing_info = processing_info

        # Avoid recomputation
        self.supported_mm_limits = self.get_supported_mm_limits()
        self.allowed_mm_limits = self.get_allowed_mm_limits()

    @abstractmethod
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """
        Return the maximum supported number of items for each modality.

        A value of `None` means unlimited number of items.

        Omitting a modality from the returned dictionary means that
        it is not supported at all.
        """
        raise NotImplementedError

    def get_allowed_mm_limits(self) -> Mapping[str, int]:
        """Return the maximum allowed number of items for each modality."""
        supported_mm_limits = self.get_supported_mm_limits()
        mm_config = self.processing_info.ctx.get_mm_config()

        allowed_limits = dict[str, int]()
        for modality, supported_limit in supported_mm_limits.items():
            user_limit = mm_config.get_limit_per_prompt(modality)

            allowed_limits[modality] = (
                user_limit
                if supported_limit is None
                else min(user_limit, supported_limit)
            )

        return allowed_limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """
        Return the maximum number of tokens per item of for each modality.

        When `None` (the default) is returned, vLLM will generate dummy inputs
        (images/videos) at maximum possible sizes and process them to determine
        the maximum token count per modality.

        This approach works but can be very slow for certain models (e.g.,
        Qwen2.5-VL), leading to very long startup time. For better performance,
        each model can override this method to return pre-computed maximum token
        counts, avoiding the need for dummy input generation and processing.

        Note:
            The maximum number of tokens per item of each modality returned
            from this function should respect the model's maximum sequence
            length and the maximum number of items of each modality allowed,
            and agree with dummy inputs (images/videos) at maximum possible
            sizes.
        """
        return None


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


_Prof = TypeVar(
    "_Prof", covariant=True, bound=BaseProfilingInfo, default=BaseProfilingInfo
)


class BaseDummyInputsBuilder(ABC, Generic[_Proc, _Prof]):
    """
    Abstract base class that constructs the dummy data to profile
    multi-modal models.
    """

    def __init__(self, profiling_info: _Prof) -> None:
        super().__init__()

        self.profiling_info = profiling_info

    @property
    def processing_info(self) -> _Proc:
        return self.profiling_info.processing_info  # type: ignore[return-value]

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

    def _get_mm_max_tokens(
        self,
        processor: BaseMultiModalProcessor,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_embeddings_only: bool = True,
    ) -> Mapping[str, int]:
        if mm_counts is None:
            mm_counts = processor.profiling_info.allowed_mm_limits

        max_tokens_per_item = processor.profiling_info.get_mm_max_tokens_per_item(
            seq_len=seq_len,
            mm_counts=mm_counts,
        )
        if max_tokens_per_item is not None:
            return {
                modality: max_tokens
                for modality, max_tokens in max_tokens_per_item.items()
                if mm_counts.get(modality, 0) > 0
            }

        processor_inputs = self.get_dummy_processor_inputs(
            seq_len,
            mm_counts=mm_counts,
        )

        mm_inputs = processor.apply(
            prompt=processor_inputs.prompt,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
            tokenization_kwargs=processor_inputs.tokenization_kwargs,
        )
        return self._get_mm_num_tokens(mm_inputs, mm_embeddings_only=mm_embeddings_only)

    def get_mm_max_contiguous_tokens(
        self,
        processor: BaseMultiModalProcessor,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ):
        """
        Returns the maximum length of the multimodal (image placeholders+text)
        tokens, including any break/text tokens in-between image embeddings.

        `<im_start> [IMG] [IMG] [IMG] <row_break> [IMG] [IMG] [IMG] <im_end>`
        Returns 9, even when the number of image embeddings is 6.

        This is important to take into account when profiling and
        initializing the encoder cache size.
        """
        return self._get_mm_max_tokens(
            processor,
            seq_len,
            mm_counts,
            mm_embeddings_only=False,
        )

    def get_decoder_dummy_data(
        self,
        processor: BaseMultiModalProcessor,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> DummyDecoderData:
        if mm_counts is None:
            mm_counts = processor.profiling_info.allowed_mm_limits

        processor_inputs = self.get_dummy_processor_inputs(
            seq_len,
            mm_counts=mm_counts,
            mm_options=mm_options,
        )

        mm_inputs = processor.apply(
            prompt=processor_inputs.prompt,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
            tokenization_kwargs=processor_inputs.tokenization_kwargs,
        )

        prompt_token_ids = mm_inputs["prompt_token_ids"]
        total_len = len(prompt_token_ids)

        if total_len < seq_len:
            prompt_token_ids.extend([0] * (seq_len - total_len))

        return DummyDecoderData(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data=mm_inputs["mm_kwargs"].require_data(),
            multi_modal_placeholders=mm_inputs["mm_placeholders"],
        )


class BaseEncDecDummyInputsBuilder(BaseDummyInputsBuilder[_Proc, _Prof]):
    @property
    def pad_encoder_prompt(self) -> bool:
        return False

    def get_encoder_dummy_data(
        self,
        processor: EncDecMultiModalProcessor,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> DummyEncoderData:
        if mm_counts is None:
            mm_counts = processor.profiling_info.allowed_mm_limits

        processor_inputs = self.get_dummy_processor_inputs(
            seq_len,
            mm_counts=mm_counts,
            mm_options=mm_options,
        )

        mm_inputs = processor.apply(
            prompt=processor_inputs.prompt,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
            tokenization_kwargs=processor_inputs.tokenization_kwargs,
        )

        # For encoder-decoder models, use encoder prompt token ids instead of
        # decoder prompt to construct dummy seq_data for encoder profiling.
        encoder_prompt_token_ids = mm_inputs["encoder_prompt_token_ids"]

        if self.pad_encoder_prompt:
            total_len = len(encoder_prompt_token_ids)
            num_tokens_to_pad = max(total_len, seq_len) - total_len
            encoder_prompt_token_ids.extend([0] * num_tokens_to_pad)

        return DummyEncoderData(encoder_prompt_token_ids)
