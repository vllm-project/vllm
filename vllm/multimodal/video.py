from functools import lru_cache
from typing import List, Union

import numpy as np

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_video_processor
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import is_list_of

from .base import MultiModalData, MultiModalInputs
from .image import ImagePlugin

logger = init_logger(__name__)

cached_get_video_processor = lru_cache(get_video_processor)
cached_get_tokenizer = lru_cache(get_tokenizer)

VideoInput = Union[
    "np.ndarray",  # single video input
    List["np.ndarray"],
    # TODO: support more types
    # List[Image.Image], List[List[Image.Image]],
    # "torch.Tensor",
    # List["torch.Tensor"],
    # List[List["np.ndarrray"]],
    # List[List["torch.Tensor"]],
]


class VideoPlugin(ImagePlugin):
    """Plugin for video data."""

    def get_data_key(self) -> str:
        return "video"

    def _get_hf_video_processor(self, model_config: ModelConfig):
        return cached_get_video_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[object],
    ) -> MultiModalInputs:
        model_config = ctx.model_config

        # single video input as np.ndarray
        if isinstance(data, np.ndarray):
            video_processor = self._get_hf_video_processor(model_config)
            if video_processor is None:
                raise RuntimeError("No HuggingFace processor is available "
                                   "to process the image object")
            try:
                batch_data = video_processor(data, return_tensors="pt").data
            except Exception:
                logger.error("Failed to process image (%s)", data)
                raise

            return MultiModalInputs(batch_data)
        elif is_list_of(data, np.ndarray):
            raise NotImplementedError(
                "Multi video for a prompt is not supported yet")

        raise TypeError(f"Invalid video type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 4096
