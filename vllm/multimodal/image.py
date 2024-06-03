from typing import Dict, Optional, Tuple, Type, Union

import torch
from PIL import Image
from transformers import (CLIPVisionConfig, LlavaConfig, LlavaNextConfig,
                          PretrainedConfig)
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.inputs.registry import DummyDataFactories, DummyDataFactory
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import cached_get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)


def _get_dummy_seq_data(
    *,
    seq_len: int,
    image_token_id: int,
    image_feature_size: int,
) -> SequenceData:
    # NOTE: We assume that <image> token is repeated `image_feature_size` times
    # and then concatenated with the text prompt
    # TODO: Enable other ways of inserting the image into the prompt

    token_ids = [image_token_id] * image_feature_size
    token_ids += [0] * (seq_len - image_feature_size)

    return SequenceData(token_ids)


class DummyImageDataFactories:

    @classmethod
    def _get_clip_num_patches(
        cls,
        hf_config: CLIPVisionConfig,
    ) -> int:
        image_size = hf_config.image_size
        patch_size = hf_config.patch_size

        assert image_size % patch_size == 0
        return image_size // patch_size

    @classmethod
    def _dummy_data_for_clip(
        cls,
        seq_len: int,
        multimodal_config: VisionLanguageConfig,
        hf_config: CLIPVisionConfig,
        *,
        image_token_id: int,
        image_feature_size_override: Optional[int] = None,
    ):
        if image_feature_size_override is None:
            num_patches = cls._get_clip_num_patches(hf_config)
            image_feature_size = num_patches * num_patches
        else:
            image_feature_size = image_feature_size_override

        seq_data = _get_dummy_seq_data(
            seq_len=seq_len,
            image_token_id=image_token_id,
            image_feature_size=image_feature_size,
        )

        image_input_type = multimodal_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType
        multi_modal_data: MultiModalData
        if image_input_type == ImageInputType.PIXEL_VALUES:
            width = height = hf_config.image_size
            if multimodal_config.image_processor is None:
                values_dtype = torch.float16
            else:
                values_dtype = torch.uint8

            values = torch.zeros((1, 3, width, height), dtype=values_dtype)
            multi_modal_data = ImagePixelData(values)
        elif image_input_type == ImageInputType.IMAGE_FEATURES:
            depth = hf_config.hidden_size
            values_dtype = torch.float16

            values = torch.zeros((1, image_feature_size, depth),
                                 dtype=values_dtype)
            multi_modal_data = ImageFeatureData(values)

        return seq_data, multi_modal_data

    @classmethod
    def _dummy_data_for_llava(
        cls,
        seq_len: int,
        multimodal_config: VisionLanguageConfig,
        hf_config: LlavaConfig,
    ):
        vision_config = hf_config.vision_config

        if isinstance(vision_config, CLIPVisionConfig):
            return cls._dummy_data_for_clip(
                seq_len=seq_len,
                multimodal_config=multimodal_config,
                hf_config=vision_config,
                image_token_id=hf_config.image_token_index,
            )

        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    @classmethod
    def _get_llava_next_num_unpadded_features(
        cls,
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

    @classmethod
    def _dummy_data_for_llava_next(
        cls,
        seq_len: int,
        multimodal_config: VisionLanguageConfig,
        hf_config: LlavaNextConfig,
    ):
        vision_config = hf_config.vision_config

        if isinstance(vision_config, CLIPVisionConfig):
            num_patches = cls._get_clip_num_patches(vision_config)
            base_feature_size = num_patches * num_patches

            # Results in the max possible feature size
            dummy_height, dummy_width = 448, 448
            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                image_size=(dummy_height, dummy_width),
                grid_pinpoints=hf_config.image_grid_pinpoints,
                patch_size=vision_config.image_size,
            )

            (
                unpadded_feature_size,
                newline_feature_size,
            ) = cls._get_llava_next_num_unpadded_features(
                dummy_height,
                dummy_width,
                num_patches,
                num_patch_height,
                num_patch_width,
            )

            image_feature_size = unpadded_feature_size + newline_feature_size \
                + base_feature_size

            return cls._dummy_data_for_clip(
                seq_len=seq_len,
                multimodal_config=multimodal_config,
                hf_config=vision_config,
                image_token_id=hf_config.image_token_index,
                image_feature_size_override=image_feature_size,
            )

        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    @classmethod
    def for_model(
        cls,
        hf_config_type: Type[PretrainedConfig],
    ) -> DummyDataFactory:
        if hf_config_type == LlavaConfig:
            return DummyDataFactories.for_multimodal_hf(LlavaConfig) \
                (cls._dummy_data_for_llava)
        if hf_config_type == LlavaNextConfig:
            return DummyDataFactories.for_multimodal_hf(LlavaNextConfig) \
                (cls._dummy_data_for_llava_next)

        msg = f"Unsupported model config: {type(hf_config_type)}"
        raise NotImplementedError(msg)


class ImagePixelData(MultiModalData):
    """
    The pixel data of an image. Can be one of:

    - :class:``PIL.Image``: An image object. Requires that a HuggingFace
      processor is available to the model.
    - :class:``torch.Tensor``: The raw pixel data which is passed to the model
      without additional pre-processing.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor]) -> None:
        if isinstance(image, Image.Image):
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

        if isinstance(image, Image.Image):
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


class ImageFeaturePlugin(MultiModalPlugin[ImageFeatureData]):

    def get_data_type(self) -> Type[ImageFeatureData]:
        return ImageFeatureData

    def _default_input_processor(
            self, data: ImageFeatureData, model_config: ModelConfig,
            vlm_config: VisionLanguageConfig) -> Dict[str, torch.Tensor]:
        image_features = data.image_features.to(model_config.dtype)

        return {"image_features": image_features}
