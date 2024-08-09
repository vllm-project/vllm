from functools import lru_cache
from typing import List, Optional, Tuple, TypeVar

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_image_processor
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import is_list_of

from .base import MultiModalInputs, MultiModalPlugin

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
        # This is an arbitrary number to distinguish between the two cases
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


class ImagePlugin(MultiModalPlugin):
    """Plugin for image data."""

    def get_data_key(self) -> str:
        return "image"

    def _get_hf_image_processor(self, model_config: ModelConfig):
        return cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
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
