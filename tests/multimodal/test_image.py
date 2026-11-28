# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import pybase64
import pytest
from PIL import Image, ImageChops

from vllm.multimodal.image import (
    _has_transparency,
    convert_image_mode,
    normalize_image,
)
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.parse import MultiModalDataParser

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


def _reopen(image: Image.Image, image_format: str, **save_kwargs) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, image_format, **save_kwargs)
    return Image.open(buffer)


def _rgb_png() -> Image.Image:
    image = Image.new("RGB", (2, 1), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    return _reopen(image, "PNG", transparency=(255, 0, 0))


def _gif() -> Image.Image:
    # The transparent index 0 and the opaque index 1 are both black.
    image = Image.new("P", (2, 1))
    image.putpalette([0] * 768)
    image.putpixel((1, 0), 1)
    return _reopen(image, "GIF", transparency=0)


def _rgba_png() -> Image.Image:
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 255, 255))
    return _reopen(image, "PNG")


@pytest.mark.parametrize(
    "make_image",
    [
        _rgb_png,
        lambda: _rgb_png().convert("RGB"),
        _gif,
        lambda: _gif().convert("RGB"),
        _rgba_png,
    ],
    ids=["rgb_png", "rgb_png_to_rgb", "gif", "gif_to_rgb", "rgba_png"],
)
def test_transparency_offline_matches_online(make_image):
    """A PIL image passed offline and the same image sent online as a PNG
    end up with the same pixels, matching Pillow's own RGBA conversion.

    After `convert("RGB")`, Pillow keeps the transparent color in `info`
    and saves it as a tRNS chunk, so every pixel of that color is
    transparent in both pipelines, including opaque GIF pixels that share it.
    """
    image = make_image()
    buffer = BytesIO()
    image.save(buffer, "PNG")
    url = f"data:image/png;base64,{pybase64.b64encode(buffer.getvalue()).decode()}"

    parser = MultiModalDataParser()
    offline = parser.parse_mm_data({"image": image})["image"].get(0)
    online = parser.parse_mm_data({"image": MediaConnector().fetch_image(url)})
    online = online["image"].get(0)
    np.testing.assert_array_equal(np.asarray(offline), np.asarray(online))

    expected = Image.alpha_composite(
        Image.new("RGBA", image.size, (255, 255, 255, 255)), image.convert("RGBA")
    ).convert("RGB")
    np.testing.assert_array_equal(np.asarray(online), np.asarray(expected))


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


def make_test_png_bytes(
    mode: str,
    orientation: int | None = None,
    *,
    include_metadata: bool = True,
) -> bytes:
    """Build an asymmetric rectangular PNG with optional EXIF and ICC metadata."""
    width, height = 6, 4
    if mode == "RGB":
        arr = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        img = Image.fromarray(arr, mode="RGB")
    elif mode == "RGBA":
        arr = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
        img = Image.fromarray(arr, mode="RGBA")
    elif mode == "L":
        arr = np.arange(height * width, dtype=np.uint8).reshape(height, width)
        img = Image.fromarray(arr, mode="L")
    elif mode == "P":
        arr = np.arange(height * width, dtype=np.uint8).reshape(height, width)
        img = Image.fromarray(arr, mode="P")
        palette = [(i * 7) % 256 for i in range(768)]
        img.putpalette(palette)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    save_kwargs: dict[str, bytes] = {}
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if include_metadata:
        exif[270] = "exif-description"
        save_kwargs["icc_profile"] = b"test-icc-profile"
    if len(exif):
        save_kwargs["exif"] = exif.tobytes()

    buf = BytesIO()
    img.save(buf, format="PNG", **save_kwargs)
    return buf.getvalue()


def test_normalize_image_default_preserves_caller_image():
    """Default normalize_image(in_place=False) must not mutate caller-owned images."""
    data = make_test_png_bytes("RGB", orientation=6)
    img = Image.open(BytesIO(data))
    img.load()

    orig_bytes = img.tobytes()
    orig_size = img.size
    orig_exif_bytes = img.getexif().tobytes()
    orig_info = dict(img.info)

    out = normalize_image(img)

    assert out is not img
    assert out.size == orig_size[::-1]
    assert img.size == orig_size
    assert img.tobytes() == orig_bytes
    assert img.getexif().tobytes() == orig_exif_bytes
    assert img.info == orig_info
    assert 0x0112 not in out.getexif()
    assert out.getexif().get(270) == "exif-description"

    # Without an Orientation tag, the default path still returns a detached copy.
    no_exif = Image.open(
        BytesIO(make_test_png_bytes("RGB", orientation=None, include_metadata=False))
    )
    no_exif.load()
    no_exif_bytes = no_exif.tobytes()
    out_copy = normalize_image(no_exif)
    assert out_copy is not no_exif
    out_copy.putpixel((0, 0), (255, 255, 255))
    assert no_exif.tobytes() == no_exif_bytes


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
@pytest.mark.parametrize("orientation", [None, *range(1, 9)])
def test_normalize_image_in_place_matches_copy(mode: str, orientation: int | None):
    """in_place=True reuses the image when no transform is needed and matches copy."""
    data = make_test_png_bytes(mode, orientation)
    img_ref = Image.open(BytesIO(data))
    img_in_place = Image.open(BytesIO(data))

    expected = normalize_image(img_ref)
    actual = normalize_image(img_in_place, in_place=True)

    assert (actual is img_in_place) == (orientation not in range(2, 9))
    assert actual.mode == expected.mode
    assert actual.size == expected.size
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(actual.convert("RGB")), np.asarray(expected.convert("RGB"))
    )
    if mode == "P":
        assert actual.getpalette() == expected.getpalette()
    assert dict(actual.getexif()) == dict(expected.getexif())
    assert actual.info == expected.info
    assert actual.info.get("icc_profile") == b"test-icc-profile"
    assert actual.getexif().get(270) == "exif-description"


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_normalize_image_in_place_without_exif(mode: str):
    """in_place=True handles EXIF-free images without copying or altering pixels."""
    data = make_test_png_bytes(mode, orientation=None, include_metadata=False)
    img_ref = Image.open(BytesIO(data))
    img_in_place = Image.open(BytesIO(data))

    expected = normalize_image(img_ref)
    actual = normalize_image(img_in_place, in_place=True)

    assert actual is img_in_place
    assert actual.mode == expected.mode
    assert actual.size == expected.size
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(actual.convert("RGB")), np.asarray(expected.convert("RGB"))
    )
    if mode == "P":
        assert actual.getpalette() == expected.getpalette()
    assert dict(actual.getexif()) == dict(expected.getexif()) == {}
    assert actual.info == expected.info


def test_normalize_image_in_place_exif_cleanup_failure_discards_rotation(monkeypatch):
    """If EXIF rewrite raises during transposition, in_place=True must not leave
    rotated pixels paired with the original Orientation tag."""
    data = make_test_png_bytes("RGB", orientation=6)
    img = Image.open(BytesIO(data))
    img.load()
    orig_size = img.size
    orig_bytes = img.tobytes()

    def _boom_tobytes(self):
        raise RuntimeError("EXIF serialize failed")

    monkeypatch.setattr(Image.Exif, "tobytes", _boom_tobytes)
    out = normalize_image(img, in_place=True)

    assert out is img
    assert out.size == orig_size
    assert out.tobytes() == orig_bytes
    assert out.getexif().get(0x0112) == 6
