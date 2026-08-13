# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image processor for Unlimited-OCR (baidu/Unlimited-OCR)."""

from PIL import Image

from vllm.logger import init_logger
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

logger = init_logger(__name__)


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    """DeepseekOCRProcessor variant for Unlimited-OCR.

    The only behavioural difference from the base processor is a multi-image
    safeguard: when more than one image is present, crop ("gundam") mode is
    disabled.

    Because the effective crop flag then depends on *how many* images are in the
    request, the per-item processing output is no longer invariant of sibling
    images. ``UnlimitedOCRMultiModalProcessor`` accounts for this by bypassing
    the multimodal processing cache for multi-image requests (see its
    ``_cached_apply_hf_processor``), so the two paths stay consistent.

    DeepSeek-OCR does *not* have this restriction because its ``max_crops=6`` is
    small enough to be safe for multi-image use.
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
            logger.warning_once(
                "Unlimited-OCR: crop mode is not supported for multi-image "
                "input. Falling back to cropping=False."
            )
            cropping = False
        return super().tokenize_with_images(
            conversation, images, bos=bos, eos=eos, cropping=cropping
        )
