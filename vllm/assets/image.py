from dataclasses import dataclass
from typing import Literal

from PIL import Image

from vllm.assets.base import get_vllm_public_assets

VLM_IMAGES_DIR = "vision_model_images"


@dataclass(frozen=True)
class ImageAsset:
    name: Literal["stop_sign", "cherry_blossom"]

    @property
    def pil_image(self) -> Image.Image:

        image_path = get_vllm_public_assets(
            f"{VLM_IMAGES_DIR}/{self.name}.jpg")
        return Image.open(image_path)
