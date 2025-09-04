# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from .base import get_vllm_public_assets

VLM_IMAGES_DIR = "vision_model_images"

ImageAssetName = Literal["stop_sign", "cherry_blossom", "hato",
                         "2560px-Gfp-wisconsin-madison-the-nature-boardwalk",
                         "Grayscale_8bits_palette_sample_image",
                         "1280px-Venn_diagram_rgb", "RGBA_comp", "237-400x300",
                         "231-200x300", "27-500x500", "17-150x600",
                         "handelsblatt-preview", "paper-11"]


@dataclass(frozen=True)
class ImageAsset:
    name: ImageAssetName

    def get_path(self, ext: str) -> Path:
        """
        Return s3 path for given image.
        """
        return get_vllm_public_assets(filename=f"{self.name}.{ext}",
                                      s3_prefix=VLM_IMAGES_DIR)

    @property
    def pil_image(self, ext="jpg") -> Image.Image:

        image_path = self.get_path(ext)
        return Image.open(image_path)

    @property
    def image_embeds(self) -> torch.Tensor:
        """
        Image embeddings, only used for testing purposes with llava 1.5.
        """
        image_path = self.get_path('pt')
        return torch.load(image_path, map_location="cpu", weights_only=True)

    def read_bytes(self, ext: str) -> bytes:
        p = Path(self.get_path(ext))
        return p.read_bytes()
