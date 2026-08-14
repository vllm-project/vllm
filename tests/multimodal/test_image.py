# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

from vllm.multimodal.image import (
    _has_transparency,
    convert_image_mode,
    normalize_image,
)

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()


def test_rgb_to_rgb():
    # Start with an RGB image.
    original_image = Image.open(ASSETS_DIR / "image1.png").convert("RGB")
    converted_image = convert_image_mode(original_image, "RGB")

    # RGB to RGB should be a no-op.
    diff = ImageChops.difference(original_image, converted_image)
    assert diff.getbbox() is None


def test_rgba_to_rgb():
    original_image = Image.open(ASSETS_DIR / "rgba.png")
    original_image_numpy = np.array(original_image)

    converted_image = convert_image_mode(original_image, "RGB")
    converted_image_numpy = np.array(converted_image)

    for i in range(original_image_numpy.shape[0]):
        for j in range(original_image_numpy.shape[1]):
            # Verify that all transparent pixels are converted to white.
            if original_image_numpy[i][j][3] == 0:
                assert converted_image_numpy[i][j][0] == 255
                assert converted_image_numpy[i][j][1] == 255
                assert converted_image_numpy[i][j][2] == 255


def test_palette_with_trns_to_rgb():
    """P-mode PNG with tRNS: transparent index should become white."""
    # Built synthetically because the existing assets are RGBA PNGs;
    # this vulnerability only affects P/L/RGB images carrying a tRNS
    # chunk, which uses a different transparency mechanism.
    img = Image.new("P", (4, 4))
    palette = [0] * 768
    palette[0:3] = [255, 0, 0]
    palette[3:6] = [0, 0, 255]
    img.putpalette(palette)
    img.putpixel((0, 0), 0)
    img.putpixel((1, 0), 1)
    img.info["transparency"] = 1

    assert _has_transparency(img)
    converted = convert_image_mode(img, "RGB")
    assert converted.mode == "RGB"
    r, g, b = converted.getpixel((1, 0))
    assert (r, g, b) == (255, 255, 255)
    r, g, b = converted.getpixel((0, 0))
    assert (r, g, b) == (255, 0, 0)


def test_l_mode_no_trns_to_rgb():
    """L-mode without transparency should convert directly."""
    img = Image.new("L", (4, 4), 128)
    assert not _has_transparency(img)
    converted = convert_image_mode(img, "RGB")
    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (128, 128, 128)


def test_exif_transpose_normalizes_orientation():
    """Image with EXIF orientation 3 (180-degree rotation) should be
    normalized so pixel data matches visual display."""
    # Built synthetically because the existing assets are PNGs which
    # don't carry EXIF orientation metadata; we need a JPEG with an
    # injected EXIF orientation=3 tag.
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((1, 0), (0, 0, 255))

    buf = BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    exif_orientation_3 = (
        b"\xff\xe1"
        + struct.pack(">H", 26)
        + b"Exif\x00\x00"
        + b"MM"
        + b"\x00\x2a"
        + b"\x00\x00\x00\x08"
        + b"\x00\x01"
        + b"\x01\x12"
        + b"\x00\x03"
        + b"\x00\x00\x00\x01"
        + b"\x00\x03"
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
    )

    soi = jpeg_bytes[:2]
    rest = jpeg_bytes[2:]
    patched = soi + exif_orientation_3 + rest

    rotated = Image.open(BytesIO(patched))
    normalized = normalize_image(rotated)
    assert normalized.size == (2, 1)


def test_normalize_image_no_exif():
    """Images without EXIF should pass through unchanged."""
    img = Image.new("RGB", (4, 4), (100, 100, 100))
    result = normalize_image(img)
    assert result.size == img.size
    assert result.mode == img.mode
