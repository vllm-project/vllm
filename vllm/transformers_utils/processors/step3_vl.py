# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from vllm.platforms import current_platform


class GPUToTensor(torch.nn.Module):

    def forward(self, raw_image: Union[np.ndarray,
                                       Image.Image]) -> torch.Tensor:
        if isinstance(raw_image, Image.Image):
            return transforms.ToTensor()(raw_image)
        if raw_image.ndim == 2:
            raw_image = raw_image[:, :, None].repeat(3, -1)
        if current_platform.is_cuda():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        image_tensor = torch.from_numpy(raw_image).to(device)
        image_tensor = torch.permute(image_tensor, (2, 0, 1)).contiguous()
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32).div(255)
        return image_tensor


class Step3VisionProcessor:

    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose([
            GPUToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize(
                (size, size),
                interpolation=InterpolationMode.BICUBIC if interpolation_mode
                == "bicubic" else InterpolationMode.BILINEAR,
                antialias=True),
        ])

        self.patch_transform = transforms.Compose([
            GPUToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize(
                (patch_size, patch_size),
                interpolation=InterpolationMode.BICUBIC if interpolation_mode
                == "bicubic" else InterpolationMode.BILINEAR,
                antialias=True),
        ]) if patch_size is not None else None

    def __call__(self, image, is_patch=False):
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        else:
            return {"pixel_values": self.transform(image).unsqueeze(0)}
