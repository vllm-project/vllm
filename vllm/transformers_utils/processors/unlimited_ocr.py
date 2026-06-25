# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image processor for Unlimited-OCR (baidu/Unlimited-OCR)."""

import logging

from PIL import Image

from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

logger = logging.getLogger(__name__)


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    """DeepseekOCRProcessor variant for Unlimited-OCR.

    The only difference from the base processor is a multi-image safeguard:
    when more than one image is present, crop mode is automatically disabled.
    This mirrors the SGLang Unlimited-OCR restriction where the crop-enabled
    "gundam" mode is excluded from ``_MULTI_IMAGE_ALLOWED`` and would raise a
    ``ValueError`` for multi-image input.

    DeepSeek-OCR does *not* have this restriction because its ``max_crops=6``
    is small enough to be safe for multi-image use.
    """

    def tokenize_with_images(
        self,
        conversation: str,
        images: list[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        if len(images) > 1 and cropping:
            logger.warning(
                "Unlimited-OCR: crop mode is not supported for multi-image "
                "input (%d images). Falling back to cropping=False to match "
                "SGLang behaviour.",
                len(images),
            )
            cropping = False
        return super().tokenize_with_images(
            conversation, images, bos=bos, eos=eos, cropping=cropping
        )
