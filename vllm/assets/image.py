# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

from .base import get_vllm_public_assets

VLM_IMAGES_DIR = "vision_model_images"


@dataclass(frozen=True)
class ImageAsset:
    name: Literal["stop_sign", "cherry_blossom"]

    @property
    def pil_image(self) -> Image.Image:
        image_path = get_vllm_public_assets(filename=f"{self.name}.jpg",
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
