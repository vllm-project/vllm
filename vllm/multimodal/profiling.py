# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from PIL import Image

import vllm.envs as envs
from vllm.inputs import DummyData
from vllm.logger import init_logger

from .inputs import MultiModalDataDict, MultiModalInputs
from .processing import BaseMultiModalProcessor, BaseProcessingInfo

logger = init_logger(__name__)


@dataclass
class ProcessorInputs:
    """
    Represents the keyword arguments to
    :meth:`vllm.multimodal.processing.BaseMultiModalProcessor.apply`.
    """
    prompt_text: str
    mm_data: MultiModalDataDict
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)


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
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """
        Build the input which, after processing, results in
        :code:`self.info.get_mm_max_tokens_per_item()` placeholder tokens.
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
        mm_config = self.processing_info.ctx.get_mm_config()
        mm_limit_per_prompt = mm_config.limit_per_prompt

        supported_mm_limits = self.processing_info.get_supported_mm_limits()

        mm_limits = {
            modality: mm_limit_per_prompt.get(modality, 1)
            for modality in supported_mm_limits
        }

        for modality, supported_limit in supported_mm_limits.items():
            limit = mm_limits[modality]
            if supported_limit is not None and supported_limit < limit:
                raise ValueError(
                    f"You set {modality}={limit} (or defaulted to 1) in "
                    f"`--limit-mm-per-prompt`, but this model only supports "
                    f"at most {supported_limit} {modality} items.")

        return mm_limits

    def _get_dummy_mm_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalInputs:
        factory = self.dummy_inputs
        processor_inputs = factory.get_dummy_processor_inputs(
            seq_len, mm_counts)

        return self.processor.apply(
            prompt=processor_inputs.prompt_text,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
        )

    def get_dummy_data(
        self,
        seq_len: int,
        is_encoder_data: bool = False,
    ) -> DummyData:
        # Avoid circular import
        from vllm.sequence import SequenceData

        mm_counts = self.get_mm_limits()

        info = self.processing_info
        mm_max_tokens_per_item = info.get_mm_max_tokens_per_item(
            seq_len, mm_counts)

        if mm_counts.keys() != mm_max_tokens_per_item.keys():
            raise AssertionError(
                "The keys returned by `get_supported_mm_limits` "
                f"({set(mm_counts.keys())}) should be the same as those "
                "returned by `get_mm_max_tokens_per_item` "
                f"({set(mm_max_tokens_per_item.keys())})")

        mm_inputs = self._get_dummy_mm_inputs(seq_len, mm_counts)
        placeholders_by_modality = mm_inputs["mm_placeholders"]
        # For encoder-decoder models, use encoder prompt token ids instead of
        # decoder prompt to construct dummy seq_data for encoder profiling.
        prompt_token_ids = (
            mm_inputs["prompt_token_ids"] if not is_encoder_data else
            mm_inputs["encoder_prompt_token_ids"])  # type: ignore

        total_placeholders_by_modality = {
            modality: sum(item["length"] for item in placeholders)
            for modality, placeholders in placeholders_by_modality.items()
        }
        expected_placeholders_by_modality = {
            modality: mm_max_tokens_per_item[modality] * mm_counts[modality]
            for modality in placeholders_by_modality
        }
        if total_placeholders_by_modality != expected_placeholders_by_modality:
            raise AssertionError(
                f"The processed dummy data has a total of "
                f"{total_placeholders_by_modality} placeholder tokens, which "
                f"is not the expected {expected_placeholders_by_modality} "
                "tokens.")

        total_len = len(prompt_token_ids)

        # V0 does not support chunked prefill.
        if (total_len > seq_len and not envs.VLLM_USE_V1) or is_encoder_data:
            if total_len > seq_len and not is_encoder_data:
                logger.warning(
                    "The context length (%d) of the model is too short "
                    "to hold the multi-modal embeddings in the worst case "
                    "(%d tokens in total, out of which %s are reserved for "
                    "multi-modal embeddings). This may cause certain "
                    "multi-modal inputs to fail during inference, even when "
                    "the input text is short. To avoid this, you should "
                    "increase `max_model_len`, reduce `max_num_seqs`, "
                    "and/or reduce `mm_counts`.", seq_len, total_len,
                    total_placeholders_by_modality)

            return DummyData(
                seq_data=SequenceData.from_prompt_token_counts(
                    (0, max(seq_len, total_len))),
                multi_modal_data=None,
                multi_modal_placeholders=None,
            )

        prompt_token_ids.extend([0] * (seq_len - len(prompt_token_ids)))

        return DummyData(
            seq_data=SequenceData.from_seqs(prompt_token_ids),
            multi_modal_data=mm_inputs["mm_kwargs"],
            multi_modal_placeholders=placeholders_by_modality,
        )
