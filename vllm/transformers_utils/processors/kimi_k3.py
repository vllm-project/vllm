# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.tokenizers.hf import HfTokenizer


def encode_k3_text_prompt(
    tokenizer: HfTokenizer,
    text: str,
    image_placeholder: str,
    *,
    add_special_tokens: bool = True,
) -> list[int]:
    """Tokenize a text prompt, encoding each image placeholder on its own.

    The placeholder is made of ordinary tokens. Encoded together with the
    surrounding text, its first and last tokens can merge with their
    neighbours, and the placeholder is then not found in the token ids.
    """
    placeholder_ids = tokenizer.encode(image_placeholder, add_special_tokens=False)
    token_ids: list[int] = []
    for i, segment in enumerate(text.split(image_placeholder)):
        if i > 0:
            token_ids += placeholder_ids
        if segment:
            token_ids += tokenizer.encode(
                segment, add_special_tokens=add_special_tokens
            )
    return token_ids


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for the image-only Kimi-K3 model.

    K3 exposes the standard `image` modality, so vLLM calls this processor
    with `images=[PIL, ...]`. The underlying checkpoint image processor
    (`KimiK3VisionProcessor`) works on `{"type": "image", "image": PIL}`
    media dicts, so this wrapper adapts bare PIL images into that shape before
    delegating to `preprocess`.

    Text is only tokenized here; the single `<|kimi_image_placeholder|>`
    token per image is expanded into the resolution-aware media block by the
    model's `_get_prompt_updates` on the vLLM side.
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
        do_rescale: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            medias: list[dict[str, Any]] = [
                {"type": "image", "image": image} for image in images
            ]
            mm_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
            )
            mm_inputs["image_sizes"] = [list(media["image"].size) for media in medias]
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
