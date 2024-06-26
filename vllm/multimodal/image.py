from typing import Dict, Tuple, Type

import torch
from PIL import Image

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import cached_get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)

IMAGE_TOKEN_ID = 32000
IMAGE_FEATURE_SIZE = 576
IMAGE_SHAPE = (336, 336)


# TODO: All the reference to `vlm_config` will be updated to `mm_config`.
# TODO: This file should also be scoped to mm.
def _get_dummy_seq_data(seq_len: int,
                        vlm_config: VisionLanguageConfig) -> SequenceData:
    assert seq_len >= IMAGE_FEATURE_SIZE, (
        f"`seq_len` should be at least {IMAGE_FEATURE_SIZE}.")
    token_ids = [IMAGE_TOKEN_ID] * IMAGE_FEATURE_SIZE
    token_ids += [0] * (seq_len - IMAGE_FEATURE_SIZE)
    return SequenceData(token_ids)


def _get_dummy_image(vlm_config: VisionLanguageConfig) -> Image.Image:
    return Image.new("RGB", IMAGE_SHAPE, color=(255, 255, 255))


def get_dummy_image_data(
    seq_len: int,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    """Standard dummy data factory for image data (to be used in
    :meth:`vlm.multimodal.MultiModalRegistry.register_dummy_data`)."""
    seq_data = _get_dummy_seq_data(seq_len, vlm_config)
    image = _get_dummy_image(vlm_config)

    return seq_data, ImageData(image)


class ImageData(MultiModalData):
    """An :class:``PIL.Image`` image. Requires that a HuggingFace
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

    def _default_input_processor(
            self, data: ImageData, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
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
