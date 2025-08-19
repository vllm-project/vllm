# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from pathlib import Path
from typing import Union

import pybase64
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


def rgba_to_rgb(
    image: Image.Image,
    background_color: Union[tuple[int, int, int], list[int]] = (255, 255, 255)
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def convert_image_mode(image: Image.Image, to_mode: str):
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


class ImageMediaIO(MediaIO[Image.Image]):

    def __init__(self, image_mode: str = "RGB", **kwargs) -> None:
        super().__init__()

        self.image_mode = image_mode
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get('rgba_background_color', (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (isinstance(rgba_bg, tuple) and len(rgba_bg) == 3
                and all(isinstance(c, int) and 0 <= c <= 255
                        for c in rgba_bg)):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255].")
        self.rgba_background_color = rgba_bg

    def _convert_image_mode(self, image: Image.Image) -> Image.Image:
        """Convert image mode with custom background color."""
        if image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(image, self.image_mode)

    def load_bytes(self, data: bytes) -> Image.Image:
        image = Image.open(BytesIO(data))
        image.load()
        return self._convert_image_mode(image)

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> Image.Image:
        image = Image.open(filepath)
        image.load()
        return self._convert_image_mode(image)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "JPEG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode('utf-8')


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):

    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        return torch.load(buffer, weights_only=True)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        return torch.load(filepath, weights_only=True)

    def encode_base64(self, media: torch.Tensor) -> str:
        return pybase64.b64encode(media.numpy()).decode('utf-8')
