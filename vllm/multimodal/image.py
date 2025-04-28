# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from .base import MediaIO


def rescale_image_size(image: Image.Image,
                       size_factor: float,
                       transpose: int = -1) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    image = image.resize((new_width, new_height))
    if transpose >= 0:
        image = image.transpose(Image.Transpose(transpose))
    return image


class ImageMediaIO(MediaIO[Image.Image]):

    def __init__(self, *, image_mode: str = "RGB") -> None:
        super().__init__()

        self.image_mode = image_mode

    def load_bytes(self, data: bytes) -> Image.Image:
        image = Image.open(BytesIO(data))
        image.load()
        return image.convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> Image.Image:
        image = Image.open(filepath)
        image.load()
        return image.convert(self.image_mode)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "JPEG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):

    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        return torch.load(buffer, weights_only=True)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> torch.Tensor:
        return torch.load(filepath, weights_only=True)

    def encode_base64(self, media: torch.Tensor) -> str:
        return base64.b64encode(media.numpy()).decode('utf-8')
