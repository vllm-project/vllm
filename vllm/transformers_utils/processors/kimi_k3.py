# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.tokenizers.hf import HfTokenizer


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for the image-only Kimi-K3 model.

    K3 exposes the standard ``image`` modality, so vLLM calls this processor
    with ``images=[PIL, ...]``. The underlying checkpoint image processor
    (``KimiK3VisionProcessor``) works on ``{"type": "image", "image": PIL}``
    media dicts, so this wrapper adapts bare PIL images into that shape before
    delegating to ``preprocess``.

    Text is only tokenized here; the single ``<|kimi_image_placeholder|>``
    token per image is expanded into the resolution-aware media block by the
    model's ``_get_prompt_updates`` on the vLLM side.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: object | list[object] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            medias = [{"type": "image", "image": image} for image in images]
            mm_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
            )
        else:
            mm_inputs = {}

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **mm_inputs},
            tensor_type=return_tensors,
        )
