from functools import lru_cache
from typing import Dict, Optional, Type, Union

import torch
from PIL import Image
from transformers import CLIPVisionConfig

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)

_cached_get_image_processor = lru_cache(get_image_processor)


def get_clip_num_patches(hf_config: CLIPVisionConfig) -> int:
    image_size = hf_config.image_size
    patch_size = hf_config.patch_size

    assert image_size % patch_size == 0
    return image_size // patch_size


def get_clip_image_feature_size(hf_config: CLIPVisionConfig) -> int:
    num_patches = get_clip_num_patches(hf_config)
    return num_patches * num_patches


class DummyImageDataFactories:
    """
    Contains factories for dummy image data factories.

    See Also:
        :data:`vllm.inputs.registry.DummyDataFactory`
    """

    @classmethod
    def dummy_seq_data_for_clip(
        cls,
        hf_config: CLIPVisionConfig,
        seq_len: int,
        *,
        image_token_id: int,
        image_feature_size_override: Optional[int] = None,
    ):
        if image_feature_size_override is None:
            image_feature_size = get_clip_image_feature_size(hf_config)
        else:
            image_feature_size = image_feature_size_override

        token_ids = [image_token_id] * image_feature_size
        token_ids += [0] * (seq_len - image_feature_size)
        return SequenceData(token_ids)

    @classmethod
    def dummy_pixel_data_for_clip(
        cls,
        hf_config: CLIPVisionConfig,
        *,
        image_width_override: Optional[int] = None,
        image_height_override: Optional[int] = None,
    ):
        width = height = hf_config.image_size
        if image_width_override is not None:
            width = image_width_override
        if image_height_override is not None:
            height = image_height_override

        image = Image.new("RGB", (width, height), color=0)
        return ImagePixelData(image)

    @classmethod
    def dummy_feature_data_for_clip(
        cls,
        hf_config: CLIPVisionConfig,
        *,
        image_feature_size_override: Optional[int] = None,
    ):
        if image_feature_size_override is None:
            image_feature_size = get_clip_image_feature_size(hf_config)
        else:
            image_feature_size = image_feature_size_override

        values = torch.zeros((1, image_feature_size, hf_config.hidden_size),
                             dtype=torch.float16)
        return ImageFeatureData(values)


class ImagePixelData(MultiModalData):
    """
    The pixel data of an image. Can be one of:

    - :class:`PIL.Image.Image`: An image object. Requires that a HuggingFace
      processor is available to the model.
    - :class:`torch.Tensor`: The raw pixel data which is passed to the model
      without additional pre-processing.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor]) -> None:
        if isinstance(image, Image.Image):
            # So that this class can be created inside the Image context manager
            image.load()

        self.image = image

    def __repr__(self) -> str:
        image = self.image
        if isinstance(image, Image.Image):
            return f"{type(self).__name__}(image={image})"

        return (f"{type(self).__name__}(image=torch.Tensor(shape="
                f"{image.shape}, dtype={image.dtype}))")


class ImagePixelPlugin(MultiModalPlugin[ImagePixelData]):

    def get_data_type(self) -> Type[ImagePixelData]:
        return ImagePixelData

    def _get_hf_image_processor(self, model_config: ModelConfig):
        vlm_config = model_config.multimodal_config
        if vlm_config is None or vlm_config.image_processor is None:
            return None

        return _cached_get_image_processor(
            vlm_config.image_processor,
            trust_remote_code=model_config.trust_remote_code,
            revision=vlm_config.image_processor_revision,
        )

    def _default_input_mapper(self, ctx: InputContext,
                              data: ImagePixelData) -> Dict[str, torch.Tensor]:
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
        elif isinstance(image, torch.Tensor):
            pixel_values = image.to(model_config.dtype)

            return {"pixel_values": pixel_values}

        raise TypeError(f"Invalid image type: {type(image)}")


class ImageFeatureData(MultiModalData):
    """
    The feature vector of an image, passed directly to the model.

    This should be the output of the vision tower.
    """

    def __init__(self, image_features: torch.Tensor) -> None:
        self.image_features = image_features

    def __repr__(self) -> str:
        image_features = self.image_features

        return (f"{type(self).__name__}(image_features=torch.Tensor(shape="
                f"{image_features.shape}, dtype={image_features.dtype}))")


class ImageFeaturePlugin(MultiModalPlugin[ImageFeatureData]):

    def get_data_type(self) -> Type[ImageFeatureData]:
        return ImageFeatureData

    def _default_input_mapper(
            self, ctx: InputContext,
            data: ImageFeatureData) -> Dict[str, torch.Tensor]:
        model_config = ctx.model_config
        image_features = data.image_features.to(model_config.dtype)

        return {"image_features": image_features}
