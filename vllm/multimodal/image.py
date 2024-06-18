from functools import lru_cache
from typing import Dict, Optional, Tuple, Type, Union

import torch
from PIL import Image
from transformers import (CLIPVisionConfig, LlavaConfig, LlavaNextConfig,
                          PretrainedConfig)
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.inputs.registry import DummyDataFactory, InputContext
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)

_cached_get_image_processor = lru_cache(get_image_processor)


def _get_clip_num_patches(hf_config: CLIPVisionConfig) -> int:
    image_size = hf_config.image_size
    patch_size = hf_config.patch_size

    assert image_size % patch_size == 0
    return image_size // patch_size


def _get_clip_image_feature_size(hf_config: CLIPVisionConfig) -> int:
    num_patches = _get_clip_num_patches(hf_config)
    return num_patches * num_patches


def _get_llava_next_num_unpadded_features(
    height: int,
    width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> Tuple[int, int]:
    # Taken from: https://github.com/huggingface/text-generation-inference/blob/799a193b109662743bed1b18a09af1fdcd508c8b/server/text_generation_server/models/vlm_causal_lm.py#L111
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio: float = width / height
    current_aspect_ratio: float = current_width / current_height
    if aspect_ratio > current_aspect_ratio:
        new_height = (height * current_width) // width
        current_height = new_height
    else:
        new_width = (width * current_height) // height
        current_width = new_width

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)


def _get_llava_next_image_feature_size(
    hf_config: LlavaNextConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = _get_clip_num_patches(vision_config)
        base_feature_size = num_patches * num_patches

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(input_height, input_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_config.image_size,
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = _get_llava_next_num_unpadded_features(input_height, input_width,
                                                  num_patches,
                                                  num_patch_height,
                                                  num_patch_width)

        return unpadded_feature_size + newline_feature_size + base_feature_size

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


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
            image_feature_size = _get_clip_image_feature_size(hf_config)
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
            image_feature_size = _get_clip_image_feature_size(hf_config)
        else:
            image_feature_size = image_feature_size_override

        values = torch.zeros((1, image_feature_size, hf_config.hidden_size),
                             dtype=torch.float16)
        return ImageFeatureData(values)

    @classmethod
    def _dummy_data_for_llava(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
        hf_config: LlavaConfig,
        seq_len: int,
    ):
        vision_config = hf_config.vision_config

        if isinstance(vision_config, CLIPVisionConfig):
            seq_data = cls.dummy_seq_data_for_clip(
                vision_config,
                seq_len,
                image_token_id=hf_config.image_token_index,
            )

            image_input_type = multimodal_config.image_input_type
            ImageInputType = VisionLanguageConfig.ImageInputType
            multi_modal_data: MultiModalData
            if image_input_type == ImageInputType.PIXEL_VALUES:
                multi_modal_data = cls.dummy_pixel_data_for_clip(
                    vision_config)
            elif image_input_type == ImageInputType.IMAGE_FEATURES:
                multi_modal_data = cls.dummy_feature_data_for_clip(
                    vision_config)

            return seq_data, multi_modal_data

        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    @classmethod
    def _dummy_data_for_llava_next(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
        hf_config: LlavaNextConfig,
        seq_len: int,
    ):
        vision_config = hf_config.vision_config

        # Result in the max possible feature size
        dummy_height = dummy_width = 448
        image_feature_size = _get_llava_next_image_feature_size(
            hf_config, input_height=dummy_height, input_width=dummy_width)

        if isinstance(vision_config, CLIPVisionConfig):
            seq_data = cls.dummy_seq_data_for_clip(
                vision_config,
                seq_len,
                image_token_id=hf_config.image_token_index,
                image_feature_size_override=image_feature_size,
            )

            image_input_type = multimodal_config.image_input_type
            ImageInputType = VisionLanguageConfig.ImageInputType
            multi_modal_data: MultiModalData
            if image_input_type == ImageInputType.PIXEL_VALUES:
                multi_modal_data = cls.dummy_pixel_data_for_clip(
                    vision_config,
                    image_width_override=dummy_width,
                    image_height_override=dummy_height,
                )
            elif image_input_type == ImageInputType.IMAGE_FEATURES:
                multi_modal_data = cls.dummy_feature_data_for_clip(
                    vision_config,
                    image_feature_size_override=image_feature_size,
                )

            return seq_data, multi_modal_data

        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    @classmethod
    def for_model(
        cls,
        hf_config_type: Type[PretrainedConfig],
    ) -> DummyDataFactory:
        """
        Create an dummy image data factory for a model as identified
        by the config type.
        """
        if hf_config_type == LlavaConfig:
            return lambda ctx, seq_len: cls._dummy_data_for_llava(
                ctx.model_config,
                ctx.get_multimodal_config(),
                ctx.get_hf_config(LlavaConfig),
                seq_len=seq_len,
            )
        if hf_config_type == LlavaNextConfig:
            return lambda ctx, seq_len: cls._dummy_data_for_llava_next(
                ctx.model_config,
                ctx.get_multimodal_config(),
                ctx.get_hf_config(LlavaNextConfig),
                seq_len=seq_len,
            )

        msg = f"Unsupported model config: {type(hf_config_type)}"
        raise NotImplementedError(msg)


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
