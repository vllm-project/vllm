"""The types of multi-modal data supported by vLLM."""
from abc import ABC, abstractmethod
from typing import  Dict

import numpy as np
import torch
from PIL import Image

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import cached_get_image_processor

logger = init_logger(__name__)


class MultiModalData(ABC):

    @abstractmethod
    def get_input_kwargs(
            self, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        """Returns a dictionary which are passed as keyword arguments to
        :meth:`torch.nn.Module.forward`.
        """
        raise NotImplementedError


class ImagePixelData(MultiModalData):

    def __init__(self, image: Image.Image) -> None:
        # So that this class can be created inside the Image context manager
        image.load()

        self.image = image

    def _get_image_processor(self, model_config: ModelConfig,
                             vlm_config: VisionLanguageConfig):
        if vlm_config is None or vlm_config.image_processor is None:
            return None

        return cached_get_image_processor(
            vlm_config.image_processor,
            trust_remote_code=model_config.trust_remote_code,
            revision=vlm_config.image_processor_revision,
        )

    def get_input_kwargs(
            self, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        # Temporary patch to make LLaVA-NeXT usable
        _, _, h, w = vlm_config.image_input_shape
        image = self.image.resize((w, h))

        image_processor = self._get_image_processor(model_config, vlm_config)
        if image_processor is None:
            image_arr = np.array(image, copy=True)
            pixel_values = torch.as_tensor(image_arr) \
                .view(1, image.height, image.width, -1) \
                .permute((0, 3, 1, 2)) \
                .to(model_config.dtype)

            return {"pixel_values": pixel_values}

        try:
            out_dict = image_processor.preprocess(image) \
                .convert_to_tensors("pt")
        except Exception:
            logger.error("Failed to process image (%s)", image)
            raise

        return {k: v.to(model_config.dtype) for k, v in out_dict.data.items()}


class ImageFeatureData(MultiModalData):

    def __init__(self, image_features: torch.Tensor) -> None:
        self.image_features = image_features

    def get_input_kwargs(
            self, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        image_features = self.image_features.to(model_config.dtype)

        return {"image_features": image_features}