from functools import lru_cache
from typing import (TYPE_CHECKING, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union)

import torch
from PIL import Image
from transformers import CLIPVisionConfig, PreTrainedTokenizerBase

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.model_executor.models.clip import get_clip_image_feature_size
from vllm.transformers_utils.image_processor import get_image_processor
from vllm.transformers_utils.tokenizer import get_tokenizer

from .base import MultiModalData, MultiModalPlugin

if TYPE_CHECKING:
    from vllm.inputs import LLMInputs
else:
    LLMInputs = dict

logger = init_logger(__name__)

cached_get_image_processor = lru_cache(get_image_processor)
_cached_get_tokenizer = lru_cache(get_tokenizer)

_T = TypeVar("_T", str, int)


class ImageInputProcessors:
    """
    Contains factories for image input processors.

    See Also:
        :data:`vllm.inputs.registry.InputProcessor`
    """

    @classmethod
    def repeat_and_pad_token(
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
    def repeat_and_pad_image_tokens(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        *,
        image_token_id: int,
        repeat_count: int = 1,
        pad_token_left: Optional[int] = None,
        pad_token_right: Optional[int] = None,
    ) -> Tuple[Optional[str], List[int]]:
        if prompt is None:
            new_prompt = None
        else:
            image_token_str = tokenizer.decode(image_token_id)
            pad_token_str_left = (None if pad_token_left is None else
                                  tokenizer.decode(pad_token_left))
            pad_token_str_right = (None if pad_token_right is None else
                                   tokenizer.decode(pad_token_right))
            replacement_str = "".join(
                cls.repeat_and_pad_token(
                    image_token_str,
                    repeat_count=repeat_count,
                    pad_token_left=pad_token_str_left,
                    pad_token_right=pad_token_str_right,
                ))

            # The image tokens are removed to be consistent with HuggingFace
            new_prompt = prompt.replace(image_token_str, replacement_str, 1)

        new_token_ids: List[int] = []
        for i, token in enumerate(prompt_token_ids):
            if token == image_token_id:
                replacement_ids = cls.repeat_and_pad_token(
                    image_token_id,
                    repeat_count=repeat_count,
                    pad_token_left=pad_token_left,
                    pad_token_right=pad_token_right,
                )
                new_token_ids.extend(replacement_ids)

                # No need to further scan the list since we only replace once
                new_token_ids.extend(prompt_token_ids[i + 1:])
                break
            else:
                new_token_ids.append(token)

        return new_prompt, new_token_ids

    @classmethod
    def input_processor_for_clip(
        cls,
        model_config: ModelConfig,
        multimodal_config: VisionLanguageConfig,
        hf_config: CLIPVisionConfig,
        llm_inputs: LLMInputs,
        *,
        image_token_id: int,
        image_feature_size_override: Optional[int] = None,
    ):
        multi_modal_data = llm_inputs.get("multi_modal_data")
        if multi_modal_data is None or not isinstance(
                multi_modal_data, (ImagePixelData, ImageFeatureData)):
            return llm_inputs

        tokenizer = _cached_get_tokenizer(model_config.tokenizer)

        if image_feature_size_override is None:
            image_feature_size = get_clip_image_feature_size(hf_config)
        else:
            image_feature_size = image_feature_size_override

        new_prompt, new_token_ids = cls.repeat_and_pad_image_tokens(
            tokenizer,
            llm_inputs.get("prompt"),
            llm_inputs["prompt_token_ids"],
            image_token_id=image_token_id,
            repeat_count=image_feature_size,
        )

        # NOTE: Create a defensive copy of the original inputs
        return LLMInputs(prompt_token_ids=new_token_ids,
                         prompt=new_prompt,
                         multi_modal_data=multi_modal_data)


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

        return cached_get_image_processor(
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
