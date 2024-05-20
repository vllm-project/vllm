from typing import Dict, Type

import numpy as np
import torch
from PIL import Image

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import cached_get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)


class ImagePixelData(MultiModalData):

    def __init__(self, image: Image.Image) -> None:
        # So that this class can be created inside the Image context manager
        image.load()

        self.image = image


class ImagePixelPlugin(MultiModalPlugin[ImagePixelData]):

    def get_data_type(self) -> Type[ImagePixelData]:
        return ImagePixelData

    def _get_hf_image_processor(self, model_config: ModelConfig,
                                vlm_config: VisionLanguageConfig):
        if vlm_config is None or vlm_config.image_processor is None:
            return None

        return cached_get_image_processor(
            vlm_config.image_processor,
            trust_remote_code=model_config.trust_remote_code,
            revision=vlm_config.image_processor_revision,
        )

    def _default_input_processor(
            self, data: ImagePixelData, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        image = data.image
        image_processor = self._get_hf_image_processor(model_config,
                                                       vlm_config)
        if image_processor is None:
            image_arr = np.array(image, copy=True)
            pixel_values = torch.as_tensor(image_arr) \
                .view(1, image.height, image.width, -1) \
                .permute((0, 3, 1, 2)) \
                .to(model_config.dtype)

            return {"pixel_values": pixel_values}

        try:
            return image_processor.preprocess(image) \
                .convert_to_tensors("pt").data
        except Exception:
            logger.error("Failed to process image (%s)", image)
            raise


class ImageFeatureData(MultiModalData):

    def __init__(self, image_features: torch.Tensor) -> None:
        self.image_features = image_features


class ImageFeaturePlugin(MultiModalPlugin[ImageFeatureData]):

    def get_data_type(self) -> Type[ImageFeatureData]:
        return ImageFeatureData

    def _default_input_processor(
            self, data: ImageFeatureData, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        image_features = data.image_features.to(model_config.dtype)

        return {"image_features": image_features}
