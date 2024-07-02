from functools import lru_cache
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_image_processor
from vllm.transformers_utils.tokenizer import get_tokenizer

from .base import MultiModalData, MultiModalInputs, MultiModalPlugin

logger = init_logger(__name__)

cached_get_image_processor = lru_cache(get_image_processor)
cached_get_tokenizer = lru_cache(get_tokenizer)

# Utilities for image input processors
_T = TypeVar("_T", str, int)


def repeat_and_pad_token(
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


def repeat_and_pad_image_tokens(
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
            repeat_and_pad_token(
                image_token_str,
                repeat_count=repeat_count,
                pad_token_left=pad_token_str_left,
                pad_token_right=pad_token_str_right,
            ))

        image_token_count = prompt.count(image_token_str)
        if image_token_count > 16:
            logger.warning(
                "Please follow the prompt format that is "
                "documented on HuggingFace which does not involve "
                "repeating %s tokens.", image_token_str)
        elif image_token_count > 1:
            logger.warning("Multiple image input is not supported yet, "
                           "so any extra image tokens will be treated "
                           "as plain text.")

        # The image tokens are removed to be consistent with HuggingFace
        new_prompt = prompt.replace(image_token_str, replacement_str, 1)

    new_token_ids: List[int] = []
    for i, token in enumerate(prompt_token_ids):
        if token == image_token_id:
            replacement_ids = repeat_and_pad_token(
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
                              data: ImagePixelData) -> MultiModalInputs:
        model_config = ctx.model_config
        image = data.image

        if isinstance(image, Image.Image):
            image_processor = self._get_hf_image_processor(model_config)
            if image_processor is None:
                raise RuntimeError("No HuggingFace processor is available"
                                   "to process the image object")
            try:
                batch_data = image_processor \
                    .preprocess(image, return_tensors="pt") \
                    .data
            except Exception:
                logger.error("Failed to process image (%s)", image)
                raise

            return MultiModalInputs(batch_data)
        elif isinstance(image, torch.Tensor):
            pixel_values = image

            return MultiModalInputs({"pixel_values": pixel_values})

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

    def _default_input_mapper(self, ctx: InputContext,
                              data: ImageFeatureData) -> MultiModalInputs:
        image_features = data.image_features

        return MultiModalInputs({"image_features": image_features})
