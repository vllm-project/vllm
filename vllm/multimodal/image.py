from functools import lru_cache
from typing import Dict

import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_image_processor

from .base import MultiModalPlugin

logger = init_logger(__name__)

cached_get_image_processor = lru_cache(get_image_processor)


class ImagePlugin(MultiModalPlugin):

    def get_data_key(self) -> str:
        return "image"

    def _get_hf_image_processor(self, model_config: ModelConfig):
        return cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> Dict[str, torch.Tensor]:
        model_config = ctx.model_config
        if isinstance(data, Image.Image):
            image_processor = self._get_hf_image_processor(model_config)
            if image_processor is None:
                raise RuntimeError("No HuggingFace processor is available"
                                   "to process the image object")
            try:
                return image_processor.preprocess(data, return_tensors="pt") \
                    .to(model_config.dtype).data
            except Exception:
                logger.error("Failed to process image (%s)", data)
                raise

        raise TypeError(f"Invalid type for 'image': {type(data)}")
