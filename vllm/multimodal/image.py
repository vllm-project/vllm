from typing import Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import cached_get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)


def _get_seq_data(seq_len: int,
                  vlm_config: VisionLanguageConfig) -> SequenceData:
    # NOTE: We assume that <image> token is repeated `image_feature_size` times
    # and then concatenated with the text prompt
    # TODO: Enable other ways of inserting the image into the prompt

    token_ids = [vlm_config.image_token_id] * vlm_config.image_feature_size
    token_ids += [0] * (seq_len - vlm_config.image_feature_size)

    return SequenceData(token_ids)


def _get_values(vlm_config: VisionLanguageConfig) -> torch.Tensor:
    if vlm_config.image_processor is None:
        values_dtype = torch.float16
    else:
        values_dtype = torch.uint8

    return torch.zeros(vlm_config.image_input_shape, dtype=values_dtype)


def get_dummy_image_data(
    seq_len: int,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    """Standard dummy data factory for image data (to be used in
    :meth:`vlm.multimodal.MultiModalRegistry.register_dummy_data`)."""
    seq_data = _get_seq_data(seq_len, vlm_config)
    values = _get_values(vlm_config)

    config_input_type = vlm_config.image_input_type
    ImageInputType = VisionLanguageConfig.ImageInputType

    fake_mm_data: MultiModalData
    if config_input_type == ImageInputType.PIXEL_VALUES:
        values_arr = values.squeeze(dim=0).permute((1, 2, 0)).numpy()
        image = Image.fromarray(values_arr, mode="RGB")
        fake_mm_data = ImagePixelData(image)
    elif config_input_type == ImageInputType.IMAGE_FEATURES:
        fake_mm_data = ImageFeatureData(values)
    else:
        raise NotImplementedError

    return seq_data, fake_mm_data


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
