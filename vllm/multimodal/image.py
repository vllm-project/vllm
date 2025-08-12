from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers.image_processing_base import BatchFeature

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.processor import get_image_processor
from vllm.utils import is_list_of

from .base import MultiModalData, MultiModalInputs, MultiModalPlugin

logger = init_logger(__name__)

cached_get_image_processor = lru_cache(get_image_processor)


class ImagePlugin(MultiModalPlugin):
    """Plugin for image data."""

    def get_data_key(self) -> str:
        return "image"

    def _get_hf_image_processor(
        self,
        model_config: ModelConfig,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}
        return cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            **mm_processor_kwargs)

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[object],
        **mm_processor_kwargs,
    ) -> MultiModalInputs:
        model_config = ctx.model_config

        # Processed by input processor
        if isinstance(data, BatchFeature):
            return MultiModalInputs(data.data)

        # PIL image
        if isinstance(data, Image.Image) or is_list_of(data, Image.Image):
            image_processor = self._get_hf_image_processor(
                model_config,
                mm_processor_kwargs,
            )

            if image_processor is None:
                raise RuntimeError("No HuggingFace processor is available "
                                   "to process the image object")
            try:
                # NOTE: It may make sense to forward the mm_processor_kwargs
                # here too. For now, to keep it simple, we only allow it be
                # used for the initialization call though, just in case the
                # signatures of the preprocessor initializer don't match
                # preprocess()
                batch_data = image_processor \
                    .preprocess(data, return_tensors="pt") \
                    .data
            except Exception:
                logger.error(
                    "Failed to process image (%s) with the default mapper. "
                    "This is most likely an edge-case with this model's image "
                    "processor in transformers (type: %s), and not vLLM.",
                    data,
                    type(image_processor).__name__)
                raise

            return MultiModalInputs(batch_data)

        # Image embedding
        elif isinstance(data, torch.Tensor) or is_list_of(data, torch.Tensor):
            return MultiModalInputs({"image_embeds": data})

        raise TypeError(f"Invalid image type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000
