# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

from PIL import Image, ImageOps


def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    image = image.resize((new_width, new_height))
    if transpose >= 0:
        image = image.transpose(Image.Transpose(transpose))
    return image


def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize EXIF orientation so the pixel data matches visual display."""
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    return image


def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def _has_transparency(image: Image.Image) -> bool:
    """Detect whether an image carries transparency data (RGBA, LA, PA,
    or tRNS chunk in P/L/RGB PNGs)."""
    if image.mode in ("RGBA", "LA", "PA"):
        return True
    return "transparency" in getattr(image, "info", {})


def convert_image_mode(
    image: Image.Image,
    to_mode: str,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    if image.mode == to_mode:
        return image

    if to_mode == "RGB" and _has_transparency(image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return rgba_to_rgb(image, background_color)

    return image.convert(to_mode)
