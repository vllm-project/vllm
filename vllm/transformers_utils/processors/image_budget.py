# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

import torch
from PIL import Image
from transformers.image_utils import get_image_size

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import HfImageItem
from vllm.multimodal.processing.context import is_profiling_media


def get_image_width_height(image: HfImageItem) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    height, width = get_image_size(image)
    return width, height


def validate_image_tensor_budget(
    images: Sequence[HfImageItem],
    view_pixels: Iterable[int],
    *,
    scratch_view_pixels: int | None = None,
) -> None:
    """Bound image output tensors and temporary tensors before processing.

    Stack-based processors retain a second copy of the output. Processors
    that write directly to the output pass the largest temporary view's size
    in pixels to account for its conversion and normalization buffers.
    """
    limit = envs.VLLM_MAX_MEDIA_PREPROCESSING_BYTES
    if limit <= 0 or is_profiling_media():
        return

    # ToTensor uses the default floating dtype for byte images. Count at least
    # four bytes per value for other supported PIL pixel types.
    bytes_per_value = max(4, torch.get_default_dtype().itemsize)
    output_bytes = 0
    scratch_bytes = 0
    for image, pixels in zip(images, view_pixels, strict=True):
        channels = len(image.getbands()) if isinstance(image, Image.Image) else 3
        bytes_per_pixel = channels * bytes_per_value
        output_bytes += pixels * bytes_per_pixel
        if scratch_view_pixels is None:
            scratch_bytes = output_bytes
        else:
            scratch_bytes = max(
                scratch_bytes, 2 * scratch_view_pixels * bytes_per_pixel
            )
        if output_bytes + scratch_bytes > limit:
            raise VLLMValidationError(
                "Image preprocessing exceeds "
                f"VLLM_MAX_MEDIA_PREPROCESSING_BYTES={limit}.",
                parameter="image",
            )
