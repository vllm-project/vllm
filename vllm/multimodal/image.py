# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os

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


def _smart_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Pick a composite background by sampling opaque border pixels.

    transparent images are usually authored over a solid background, and the
    image border is typically that background. Sampling opaque edge pixels and
    averaging their brightness lets us reconstruct the intended background -- light
    for dark-on-light figures (the common case, also the fixed-white default), dark
    for light-on-dark ones. Falls back to white when the border is fully transparent.
    """
    assert image.mode == "RGBA"
    width, height = image.size
    step_x = max(1, width // 20)
    step_y = max(1, height // 20)
    edge_pixels: list[tuple[int, int, int]] = []

    for x in range(0, width, step_x):
        for y in (0, height - 1):
            pixel = image.getpixel((x, y))
            if pixel[3] > 128:  # type: ignore[index]
                edge_pixels.append(pixel[:3])  # type: ignore[index]
    for y in range(0, height, step_y):
        for x in (0, width - 1):
            pixel = image.getpixel((x, y))
            if pixel[3] > 128:  # type: ignore[index]
                edge_pixels.append(pixel[:3])  # type: ignore[index]

    if not edge_pixels:
        return (255, 255, 255)
    avg_brightness = sum(sum(p) for p in edge_pixels) / (len(edge_pixels) * 3)
    return (32, 32, 32) if avg_brightness > 128 else (240, 240, 240)


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
        # Opt-in adaptive background: reconstruct the
        # intended background from opaque border pixels instead of always
        # compositing over white. Helps transparent figures authored on a dark
        # canvas; off by default to keep the fixed-background behavior.
        if os.environ.get("VLLM_SMART_IMAGE_RGB") == "1":
            background_color = _smart_background_color(image)
        return rgba_to_rgb(image, background_color)

    return image.convert(to_mode)
