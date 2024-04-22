from typing import Dict, Optional

import torch

from vllm.config import ModelConfig, VisionLanguageConfig

from .data import ImageFeatureData, ImagePixelData, MultiModalData
from .processor import (ImageFeatureProcessorRegistry,
                        ImagePixelProcessorRegistry, MultiModalProcessor)


class MultiModalRootRegistry:

    def __init__(self) -> None:
        self._image_pixel_registry = ImagePixelProcessorRegistry()
        self._image_feature_registry = ImageFeatureProcessorRegistry()

    def register_image_pixel_input(
            self,
            processor: Optional[MultiModalProcessor[ImagePixelData]] = None):
        return self._image_pixel_registry.register(processor)

    def register_image_feature_input(
            self,
            processor: Optional[MultiModalProcessor[ImageFeatureData]] = None):
        return self._image_feature_registry.register(processor)

    def process(self, data: MultiModalData, model_config: ModelConfig,
                vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        if isinstance(data, ImagePixelData):
            return self._image_pixel_registry \
                .process(model_cls, data, model_config, vlm_config)
        if isinstance(data, ImageFeatureData):
            return self._image_feature_registry\
                .process(model_cls, data, model_config, vlm_config)

        msg = f"Unknown multi-modal data type: {type(data)}"
        raise NotImplementedError(msg)


MM_REGISTRY = MultiModalRootRegistry()
"""The global registry for multi-modal data."""

__all__ = ["MM_REGISTRY"]
