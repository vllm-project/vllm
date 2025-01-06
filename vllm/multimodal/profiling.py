from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger

from .inputs import MultiModalDataDict

logger = init_logger(__name__)


@dataclass
class ProcessorInputs:
    """Keyword arguments to :meth:`BaseMultiModalProcessor`."""
    prompt_text: str
    mm_data: MultiModalDataDict
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)


class BaseProfilingInfo(ABC):
    """
    Abstract base class that provides the information necessary to profile
    multi-modal models.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__()

        self.ctx = ctx

    @abstractmethod
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        """
        Return the maximum supported number of items for each modality.

        A value of `None` means unlimited number of items.

        Omitting a modality from the returned dictionary means that
        it is not supported at all.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        """
        Get the maximum possible number of tokens per data item
        for each modality.

        The dictionary returned by this method should have the same
        keys as that returned by :meth:`get_supported_mm_limits`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """
        Build the multi-modal portion of the input which, after processing,
        results in `mm_max_tokens` in :meth:`get_mm_max_tokens_per_item`.
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

    def get_mm_limits(self) -> Mapping[str, int]:
        mm_config = self.ctx.get_mm_config()
        mm_limit_per_prompt = mm_config.limit_per_prompt

        supported_mm_limits = self.get_supported_mm_limits()

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
