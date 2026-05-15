# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

IMAGENET_MEAN = np.array([0.484375, 0.455078125, 0.40625], dtype=np.float32)
IMAGENET_STD = np.array([0.228515625, 0.2236328125, 0.224609375], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def to_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "OpenVLA image input must be a PIL image, numpy array, or torch tensor; "
            f"got {type(image)}"
        )

    if image.ndim != 3:
        raise ValueError(
            f"OpenVLA image input must have 3 dimensions, got shape {image.shape}"
        )

    if image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError(
            f"OpenVLA image input must have 1 or 3 channels, got shape {image.shape}"
        )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return Image.fromarray(image).convert("RGB")


def preprocess_openvla_image(image: Any, image_size: int) -> torch.Tensor:
    rgb_image = to_rgb_image(image)
    rgb_image = rgb_image.resize(
        (image_size, image_size),
        Image.Resampling.BICUBIC,
    )

    raw = np.asarray(rgb_image, dtype=np.float32) / 255.0
    dinov2_pixels = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    siglip_pixels = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    pixel_values = np.concatenate([dinov2_pixels, siglip_pixels], axis=0)
    return torch.from_numpy(pixel_values)


class OpenVLAProcessor:
    def __init__(self, *, tokenizer: Any, image_size: int) -> None:
        self.tokenizer = tokenizer
        self.image_size = image_size

    def __call__(
        self,
        text: str,
        images: Any | None = None,
        tok_kwargs: Mapping[str, object] | None = None,
    ) -> BatchFeature:
        tok_kwargs = tok_kwargs or {}
        prompt_ids = self.tokenizer.encode(text, **tok_kwargs)

        if images is None:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            images = [images]
        if len(images) == 0:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        pixel_values = torch.stack(
            [
                preprocess_openvla_image(image, image_size=self.image_size)
                for image in images
            ],
            dim=0,
        )
        return BatchFeature(
            dict(input_ids=[prompt_ids], pixel_values=pixel_values),
            tensor_type="pt",
        )
