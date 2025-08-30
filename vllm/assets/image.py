# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from .base import get_vllm_public_assets

VLM_IMAGES_DIR = "vision_model_images"

ImageAssetName = Literal["stop_sign", "cherry_blossom",
                         "2560px-Gfp-wisconsin-madison-the-nature-boardwalk",
                         "Grayscale_8bits_palette_sample_image",
                         "1280px-Venn_diagram_rgb", "RGBA_comp", "237-400x300",
                         "231-200x300", "27-500x500", "17-150x600",
                         "handelsblatt-preview", "paper-11"]


@dataclass(frozen=True)
class ImageAsset:
    name: ImageAssetName

    @property
    def pil_image(self, ext="jpg") -> Image.Image:

        image_path = get_vllm_public_assets(filename=f"{self.name}.{ext}",
                                            s3_prefix=VLM_IMAGES_DIR)
        return Image.open(image_path)

    @property
    def image_embeds(self) -> torch.Tensor:
        """
        Image embeddings, only used for testing purposes with llava 1.5.
        """
        image_path = get_vllm_public_assets(filename=f"{self.name}.pt",
                                            s3_prefix=VLM_IMAGES_DIR)
        return torch.load(image_path, map_location="cpu", weights_only=True)
