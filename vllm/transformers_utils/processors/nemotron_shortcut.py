# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Pixtral processing adapted to Nemotron Shortcut's training transform."""

import cv2
import torch
from PIL import Image
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.models.pixtral import PixtralImageProcessor, PixtralProcessor
from transformers.processing_utils import MultiModalData


def get_resize_output_image_size(
    height: int, width: int, longest_edge: int, multiple: int
) -> tuple[int, int]:
    ratio = max(height / longest_edge, width / longest_edge, 1.0)
    height, width = round(height / ratio), round(width / ratio)
    return -(-height // multiple) * multiple, -(-width // multiple) * multiple


class NemotronShortcutImageProcessor(PixtralImageProcessor):
    size = {"longest_edge": 1036}
    patch_size = {"height": 28, "width": 28}
    resample = PILImageResampling.BOX

    def resize(self, image, size: SizeDict, patch_size: SizeDict, **kwargs):
        height, width = get_resize_output_image_size(
            image.shape[-2], image.shape[-1], size.longest_edge, patch_size.height
        )
        resized = cv2.resize(
            image.permute(1, 2, 0).cpu().numpy(),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        if resized.ndim == 2:
            resized = resized[..., None]
        return torch.from_numpy(resized).permute(2, 0, 1).to(image.device)

    def _preprocess(
        self,
        images,
        do_resize,
        size,
        resample,
        do_center_crop,
        crop_size,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        disable_grouping,
        return_tensors,
        patch_size=None,
        **kwargs,
    ):
        patch_size = SizeDict(
            **get_size_dict(patch_size or self.patch_size, default_to_square=True)
        )
        pixels, sizes = [], []
        for image in images:
            # Training rescales in FP32 before OpenCV area interpolation.
            if do_rescale:
                image = image.to(torch.float32) / (1.0 / rescale_factor)
            if do_resize:
                image = self.resize(image, size=size, patch_size=patch_size)
            if do_normalize:
                mean = torch.tensor(
                    image_mean, dtype=torch.float32, device=image.device
                )[:, None, None]
                std = torch.tensor(image_std, dtype=torch.float32, device=image.device)[
                    :, None, None
                ]
                image = (image - mean) / std
            pixels.append(image)
            sizes.append(image.shape[-2:])
        return BatchFeature(
            {
                "pixel_values": self._pad_for_batching(pixels, sizes),
                "image_sizes": sizes,
            },
            tensor_type=return_tensors,
        )

    def convert_to_rgb(self, image, **kwargs):
        """Composite transparent pixels on white, as in training.

        Depending on the transformers version, this receives a PIL image or a
        channels-first uint8 tensor that still has its alpha channel.
        """
        if isinstance(image, Image.Image):
            if image.mode == "RGB":
                return image
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.paste(image, (0, 0), image)
            return background.convert("RGB")
        channels_first = (
            kwargs.get("input_data_format", "channels_first") == "channels_first"
        )
        if (
            channels_first
            and isinstance(image, torch.Tensor)
            and image.ndim == 3
            and image.shape[0] == 4
        ):
            rgb, alpha = image[:3].float(), image[3:].float() / 255.0
            return (rgb * alpha + 255.0 * (1.0 - alpha)).round().to(image.dtype)
        return super().convert_to_rgb(image, **kwargs)


class NemotronShortcutProcessor(PixtralProcessor):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        if image_sizes is None:
            return MultiModalData()
        size = kwargs.get("size") or self.image_processor.size
        multiple = self.patch_size * self.spatial_merge_size
        counts = []
        for height, width in image_sizes:
            height, width = get_resize_output_image_size(
                height, width, size["longest_edge"], multiple
            )
            counts.append((width // multiple + 1) * (height // multiple))
        return MultiModalData(
            num_image_tokens=counts, num_image_patches=[1] * len(image_sizes)
        )
