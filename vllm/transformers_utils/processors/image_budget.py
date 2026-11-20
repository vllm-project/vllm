# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

import torch
from PIL import Image
from transformers.image_utils import get_image_size

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import HfImageItem


def get_image_width_height(image: HfImageItem) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    height, width = get_image_size(image)
    return width, height


def validate_image_tensor_budget(
    images: Sequence[HfImageItem], view_pixels: Iterable[int]
) -> None:
    """Bound the tensors and their stacked copies before processing images."""
    limit = envs.VLLM_MAX_MEDIA_PREPROCESSING_BYTES
    if limit <= 0:
        return

    # ToTensor uses the default floating dtype for byte images. Count at least
    # four bytes per value for other supported PIL pixel types.
    bytes_per_value = max(4, torch.get_default_dtype().itemsize)
    required_bytes = 0
    for image, pixels in zip(images, view_pixels, strict=True):
        channels = len(image.getbands()) if isinstance(image, Image.Image) else 3
        required_bytes += 2 * pixels * channels * bytes_per_value
        if required_bytes > limit:
            raise VLLMValidationError(
                "Image preprocessing exceeds "
                f"VLLM_MAX_MEDIA_PREPROCESSING_BYTES={limit}.",
                parameter="image",
            )
