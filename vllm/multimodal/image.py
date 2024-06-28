from functools import lru_cache
from typing import Dict, Tuple, Type

import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)

cached_get_image_processor = lru_cache(get_image_processor)


class ImageData(MultiModalData):
    """
    Contains a :class:`PIL.Image.Image` object. Requires that a HuggingFace
    processor is available to the model.
    """

    def __init__(self, image: Image.Image) -> None:
        # So that this class can be created inside the Image context manager
        image.load()
        self.image = image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(image={self.image})"


class ImagePlugin(MultiModalPlugin[ImageData]):

    def get_internal_data_type(self) -> Type[ImageData]:
        return ImageData

    def get_external_data_type(self) -> Tuple[str, Type[Image.Image]]:
        return ("image", Image.Image)

    def _get_hf_image_processor(self, model_config: ModelConfig):
        return cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)

    def _default_input_mapper(self, ctx: InputContext,
                              data: ImageData) -> Dict[str, torch.Tensor]:
        model_config = ctx.model_config
        image = data.image
        if isinstance(image, Image.Image):
            image_processor = self._get_hf_image_processor(model_config)
            if image_processor is None:
                raise RuntimeError("No HuggingFace processor is available"
                                   "to process the image object")
            try:
                return image_processor.preprocess(image, return_tensors="pt") \
                    .to(model_config.dtype).data
            except Exception:
                logger.error("Failed to process image (%s)", image)
                raise

        raise TypeError(f"Invalid image type: {type(image)}")
