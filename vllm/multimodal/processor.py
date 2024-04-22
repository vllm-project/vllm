from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Optional, Type, TypeVar

import numpy as np
import torch
from torch import nn

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import cached_get_image_processor

from .data import ImageFeatureData, ImagePixelData, MultiModalData

logger = init_logger(__name__)

D = TypeVar('D', bound=MultiModalData)
N = TypeVar('N', bound=Type[nn.Module])

MultiModalDataProcessor = Callable[[D, ModelConfig, VisionLanguageConfig],
                                   Dict[str, torch.Tensor]]
"""Returns a dictionary which are passed as keyword arguments to
:meth:`torch.nn.Module.forward`.
"""


class MultiModalProcessorBaseRegistry(ABC, Generic[D]):

    def __init__(self) -> None:
        self._processors: Dict[Type[nn.Module],
                               MultiModalDataProcessor[D]] = {}

    @abstractmethod
    def _default_processor(
            self, data: D, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def register(self, processor: Optional[MultiModalDataProcessor[D]] = None):

        def wrapper(model_cls: N) -> N:
            if model_cls in self._processors:
                logger.warning(
                    f"Model class {model_cls} is already registered to "
                    f"{type(self).__name__}.")

            self._processors[model_cls] = processor or self._default_processor

            return model_cls

        return wrapper

    def process(self, model_cls: Type[nn.Module], data: D,
                model_config: ModelConfig,
                vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        processor = self._processors.get(model_cls)
        if processor is None:
            raise KeyError(f"No processor in {type(self).__name__} is "
                           f"registered for model class {model_cls.__name__}")

        return processor(data, model_config, vlm_config)


class ImagePixelProcessorRegistry(
        MultiModalProcessorBaseRegistry[ImagePixelData]):

    def _get_hf_image_processor(self, model_config: ModelConfig,
                                vlm_config: VisionLanguageConfig):
        if vlm_config is None or vlm_config.image_processor is None:
            return None

        return cached_get_image_processor(
            vlm_config.image_processor,
            trust_remote_code=model_config.trust_remote_code,
            revision=vlm_config.image_processor_revision,
        )

    def _default_processor(
            self, data: ImagePixelData, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        # Temporary patch to make LLaVA-NeXT usable
        _, _, h, w = vlm_config.image_input_shape
        image = data.image.resize((w, h))

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
            out_dict = image_processor.preprocess(image) \
                .convert_to_tensors("pt")
        except Exception:
            logger.error("Failed to process image (%s)", image)
            raise

        return {k: v.to(model_config.dtype) for k, v in out_dict.data.items()}


class ImageFeatureProcessorRegistry(
        MultiModalProcessorBaseRegistry[ImageFeatureData]):

    def _default_processor(
            self, data: ImageFeatureData, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        image_features = data.image_features.to(model_config.dtype)

        return {"image_features": image_features}
