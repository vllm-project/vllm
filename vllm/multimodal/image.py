import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from PIL import Image

from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.processor import get_image_processor
from vllm.utils import is_list_of

from .base import MediaIO, MultiModalPlugin
from .inputs import ImageItem, ModalityData, MultiModalKwargs

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

cached_get_image_processor = lru_cache(get_image_processor)


class ImagePlugin(MultiModalPlugin):
    """Plugin for image data."""

    def get_data_key(self) -> str:
        return "image"

    def _get_hf_image_processor(
        self,
        model_config: "ModelConfig",
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
        data: ModalityData[ImageItem],
        **mm_processor_kwargs,
    ) -> MultiModalKwargs:
        model_config = ctx.model_config

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

            return MultiModalKwargs(batch_data)

        # Image embedding
        elif isinstance(data, torch.Tensor) or is_list_of(data, torch.Tensor):
            return MultiModalKwargs({"image_embeds": data})

        raise TypeError(f"Invalid image type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000


def rescale_image_size(image: Image.Image,
                       size_factor: float,
                       transpose: int = -1) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    image = image.resize((new_width, new_height))
    if transpose >= 0:
        image = image.transpose(Image.Transpose(transpose))
    return image


class ImageMediaIO(MediaIO[Image.Image]):

    def __init__(self, *, image_mode: str = "RGB") -> None:
        super().__init__()

        self.image_mode = image_mode

    def load_bytes(self, data: bytes) -> Image.Image:
        image = Image.open(BytesIO(data))
        image.load()
        return image.convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> Image.Image:
        image = Image.open(filepath)
        image.load()
        return image.convert(self.image_mode)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "JPEG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')
