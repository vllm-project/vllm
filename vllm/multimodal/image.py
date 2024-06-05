from functools import lru_cache
from typing import (TYPE_CHECKING, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union)

import torch
from PIL import Image
from transformers import (CLIPVisionConfig, LlavaConfig, LlavaNextConfig,
                          PretrainedConfig)
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.inputs.registry import DummyDataFactory, InputContext, InputProcessor
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import get_image_processor
from vllm.transformers_utils.tokenizer import get_tokenizer

from .base import MultiModalData, MultiModalPlugin

if TYPE_CHECKING:
    from vllm.inputs import LLMInputs
else:
    LLMInputs = dict

logger = init_logger(__name__)

_cached_get_tokenizer = lru_cache(get_tokenizer)
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


def _get_llava_next_image_feature_size(hf_config: LlavaNextConfig) -> int:
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = _get_clip_num_patches(vision_config)
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
        ) = _get_llava_next_num_unpadded_features(dummy_height, dummy_width,
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
    def _dummy_data_for_clip(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
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
        seq_data = SequenceData(token_ids)

        image_input_type = multimodal_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType
        multi_modal_data: MultiModalData
        if image_input_type == ImageInputType.PIXEL_VALUES:
            width = height = hf_config.image_size
            image = Image.new("RGB", (width, height), color=0)
            multi_modal_data = ImagePixelData(image)
        elif image_input_type == ImageInputType.IMAGE_FEATURES:
            depth = hf_config.hidden_size
            values = torch.zeros((1, image_feature_size, depth),
                                 dtype=torch.float16)
            multi_modal_data = ImageFeatureData(values)

        return seq_data, multi_modal_data

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
            return cls._dummy_data_for_clip(
                model_config,
                multimodal_config,
                vision_config,
                seq_len=seq_len,
                image_token_id=hf_config.image_token_index,
            )

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
        image_feature_size = _get_llava_next_image_feature_size(hf_config)

        if isinstance(vision_config, CLIPVisionConfig):
            return cls._dummy_data_for_clip(
                model_config,
                multimodal_config,
                vision_config,
                seq_len=seq_len,
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


_T = TypeVar("_T", str, int)


class ImageInputProcessors:
    """
    Contains factories for image input processors.

    See Also:
        :data:`vllm.inputs.registry.InputProcessor`
    """

    @classmethod
    def _repeat_and_pad_token(
        cls,
        token: _T,
        *,
        repeat_count: int = 1,
        pad_token_left: Optional[_T] = None,
        pad_token_right: Optional[_T] = None,
    ) -> List[_T]:
        replacement = [token] * repeat_count
        if pad_token_left is not None:
            replacement = [pad_token_left] + replacement
        if pad_token_right is not None:
            replacement = replacement + [pad_token_right]

        return replacement

    @classmethod
    def _repeat_and_pad_image_tokens(
        cls,
        model_config: ModelConfig,
        llm_inputs: LLMInputs,
        *,
        image_token_id: int,
        repeat_count: int = 1,
        pad_token_left: Optional[int] = None,
        pad_token_right: Optional[int] = None,
    ) -> LLMInputs:
        multi_modal_data = llm_inputs.get("multi_modal_data")
        if multi_modal_data is None:
            return llm_inputs

        tokenizer = _cached_get_tokenizer(model_config.tokenizer)
        image_token_str = tokenizer.decode(image_token_id)
        pad_token_str_left = (None if pad_token_left is None else
                              tokenizer.decode(pad_token_left))
        pad_token_str_right = (None if pad_token_right is None else
                               tokenizer.decode(pad_token_right))

        replacement_str = "".join(
            cls._repeat_and_pad_token(
                image_token_str,
                repeat_count=repeat_count,
                pad_token_left=pad_token_str_left,
                pad_token_right=pad_token_str_right,
            ))
        replacement_ids = cls._repeat_and_pad_token(
            image_token_id,
            repeat_count=repeat_count,
            pad_token_left=pad_token_left,
            pad_token_right=pad_token_right,
        )

        # To avoid invoking the tokenizer, we assume that the
        # image token is called "<image>"
        prompt = llm_inputs.get("prompt")
        if prompt is None:
            new_prompt = None
        else:
            # The image tokens are removed to be consistent with HuggingFace
            new_prompt = prompt.replace(image_token_str, replacement_str, 1)

        prompt_token_ids = llm_inputs["prompt_token_ids"]
        new_token_ids: List[int] = []
        for i, token in enumerate(prompt_token_ids):
            if token == image_token_id:
                new_token_ids.extend(replacement_ids)

                # No need to further scan the list since we only replace once
                new_token_ids.extend(prompt_token_ids[i + 1:])
                break
            else:
                new_token_ids.append(token)

        # NOTE: Create a defensive copy of the original inputs
        return LLMInputs(prompt_token_ids=new_token_ids,
                         prompt=new_prompt,
                         multi_modal_data=multi_modal_data)

    @classmethod
    def _input_processor_for_clip(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
        hf_config: CLIPVisionConfig,
        llm_inputs: LLMInputs,
        *,
        image_token_id: int,
        image_feature_size_override: Optional[int] = None,
    ):
        if image_feature_size_override is None:
            image_feature_size = _get_clip_image_feature_size(hf_config)
        else:
            image_feature_size = image_feature_size_override

        return cls._repeat_and_pad_image_tokens(
            model_config,
            llm_inputs,
            image_token_id=image_token_id,
            repeat_count=image_feature_size,
        )

    @classmethod
    def _input_processor_for_llava(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
        hf_config: LlavaConfig,
        llm_inputs: LLMInputs,
    ):
        vision_config = hf_config.vision_config

        if isinstance(vision_config, CLIPVisionConfig):
            return cls._input_processor_for_clip(
                model_config,
                multimodal_config,
                vision_config,
                llm_inputs,
                image_token_id=hf_config.image_token_index,
            )

        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    @classmethod
    def _input_processor_for_llava_next(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
        hf_config: LlavaNextConfig,
        llm_inputs: LLMInputs,
    ):
        vision_config = hf_config.vision_config
        image_feature_size = _get_llava_next_image_feature_size(hf_config)

        if isinstance(vision_config, CLIPVisionConfig):
            return cls._input_processor_for_clip(
                model_config,
                multimodal_config,
                vision_config,
                llm_inputs,
                image_token_id=hf_config.image_token_index,
                image_feature_size_override=image_feature_size,
            )

        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    @classmethod
    def for_model(
        cls,
        hf_config_type: Type[PretrainedConfig],
    ) -> InputProcessor:
        """
        Create an input processor for a model as identified
        by the config type.
        """
        if hf_config_type == LlavaConfig:
            return lambda ctx, llm_inputs: cls._input_processor_for_llava(
                ctx.model_config,
                ctx.get_multimodal_config(),
                ctx.get_hf_config(LlavaConfig),
                llm_inputs=llm_inputs,
            )
        if hf_config_type == LlavaNextConfig:
            return lambda ctx, llm_inputs: cls._input_processor_for_llava_next(
                ctx.model_config,
                ctx.get_multimodal_config(),
                ctx.get_hf_config(LlavaNextConfig),
                llm_inputs=llm_inputs,
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
        image_processor = self._get_hf_image_processor(model_config)

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

    def _default_input_mapper(
            self, ctx: InputContext,
            data: ImageFeatureData) -> Dict[str, torch.Tensor]:
        model_config = ctx.model_config
        image_features = data.image_features.to(model_config.dtype)

        return {"image_features": image_features}
