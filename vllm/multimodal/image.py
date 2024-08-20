from functools import lru_cache
from typing import List, Optional, Tuple, TypeVar, Union

import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_image_processor
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer
from vllm.utils import is_list_of

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
    tokenizer: AnyTokenizer,
    prompt: Optional[str],
    prompt_token_ids: List[int],
    *,
    image_token_id: int,
    repeat_count: Union[int, List[int]],
    pad_token_left: Optional[int] = None,
    pad_token_right: Optional[int] = None,
) -> Tuple[Optional[str], List[int]]:
    if isinstance(repeat_count, int):
        repeat_count = [repeat_count]

    if prompt is None:
        new_prompt = None
    else:
        image_token_str = tokenizer.decode(image_token_id)
        pad_token_str_left = (None if pad_token_left is None else
                              tokenizer.decode(pad_token_left))
        pad_token_str_right = (None if pad_token_right is None else
                               tokenizer.decode(pad_token_right))

        image_token_count = prompt.count(image_token_str)
        # This is an arbitrary number to distinguish between the two cases
        if image_token_count > 16:
            logger.warning(
                "Please follow the prompt format that is "
                "documented on HuggingFace which does not involve "
                "repeating %s tokens.", image_token_str)
        if image_token_count < len(repeat_count):
            logger.warning(
                "The number of image tokens in the prompt is less than "
                "the number of image inputs. Extra image tokens will be "
                "treated as plain text")
            repeat_count = repeat_count[:image_token_count]

        prompt_parts = prompt.split(image_token_str,
                                    maxsplit=len(repeat_count))
        new_prompt = ""
        for i in range(len(repeat_count)):
            replacement_str = "".join(
                repeat_and_pad_token(
                    image_token_str,
                    repeat_count=repeat_count[i],
                    pad_token_left=pad_token_str_left,
                    pad_token_right=pad_token_str_right,
                ))
            # The image tokens are removed to be consistent with HuggingFace
            new_prompt += prompt_parts[i] + replacement_str
        new_prompt += prompt_parts[-1]

    new_token_ids: List[int] = []
    image_token_idx = 0
    for i, token in enumerate(prompt_token_ids):
        if token == image_token_id:
            replacement_ids = repeat_and_pad_token(
                image_token_id,
                repeat_count=repeat_count[image_token_idx],
                pad_token_left=pad_token_left,
                pad_token_right=pad_token_right,
            )
            new_token_ids.extend(replacement_ids)
            image_token_idx += 1

            # No need to further scan the list since we replaced all tokens
            if image_token_idx >= len(repeat_count):
                new_token_ids.extend(prompt_token_ids[i + 1:])
                break
        else:
            new_token_ids.append(token)

    return new_prompt, new_token_ids


class ImagePlugin(MultiModalPlugin):
    """Plugin for image data."""

    def get_data_key(self) -> str:
        return "image"

    def _get_hf_image_processor(self, model_config: ModelConfig):
        return cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[object],
    ) -> MultiModalInputs:
        model_config = ctx.model_config

        # PIL image
        if isinstance(data, Image.Image) or is_list_of(data, Image.Image):
            image_processor = self._get_hf_image_processor(model_config)
            if image_processor is None:
                raise RuntimeError("No HuggingFace processor is available "
                                   "to process the image object")
            try:
                batch_data = image_processor \
                    .preprocess(data, return_tensors="pt") \
                    .data
            except Exception:
                logger.error("Failed to process image (%s)", data)
                raise

            return MultiModalInputs(batch_data)

        # Image embedding
        elif isinstance(data, torch.Tensor) or is_list_of(data, torch.Tensor):
            return MultiModalInputs({"image_embeds": data})

        raise TypeError(f"Invalid image type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000
