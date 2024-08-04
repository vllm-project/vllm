from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from PIL import Image

from vllm.connections import global_http_connection
from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT

from .base import get_cache_dir


@lru_cache
def get_air_example_data_2_asset(filename: str) -> Image.Image:
    """
    Download and open an image from
    ``s3://air-example-data-2/vllm_opensource_llava/``.
    """
    image_directory = get_cache_dir() / "air-example-data-2"
    image_directory.mkdir(parents=True, exist_ok=True)

    image_path = image_directory / filename
    if not image_path.exists():
        base_url = "https://air-example-data-2.s3.us-west-2.amazonaws.com/vllm_opensource_llava"

        global_http_connection.download_file(f"{base_url}/{filename}",
                                             image_path,
                                             timeout=VLLM_IMAGE_FETCH_TIMEOUT)

    return Image.open(image_path)


@dataclass(frozen=True)
class ImageAsset:
    name: Literal["stop_sign", "cherry_blossom"]

    @property
    def pil_image(self) -> Image.Image:
        return get_air_example_data_2_asset(f"{self.name}.jpg")
