# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import struct
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pybase64 as base64
import pytest
import torch
from PIL import Image, features
from PIL.BlpImagePlugin import BLPFormatError

from vllm.multimodal import image as image_utils
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.image import (
    ALLOWED_IMAGE_FORMATS,
    has_torchcodec_image_decoding,
    open_image,
)
from vllm.multimodal.media import ImageMediaIO, MediaConnector, MediaDecodedBudget
from vllm.multimodal.media.video import VideoMediaIO

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


def test_image_media_io_rgba_custom_background(tmp_path):
    """Test RGBA to RGB conversion with custom background colors."""
    # Create a simple RGBA image with transparent and opaque pixels
    rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))  # Red with full opacity

    # Make top-left quadrant transparent
    for i in range(5):
        for j in range(5):
            rgba_image.putpixel((i, j), (0, 0, 0, 0))  # Fully transparent

    # Save the test image to tmp_path
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Test 1: Default white background (backward compatibility)
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    default_numpy = np.array(converted_default)

    # Check transparent pixels are white
    assert default_numpy[0][0][0] == 255  # R
    assert default_numpy[0][0][1] == 255  # G
    assert default_numpy[0][0][2] == 255  # B
    # Check opaque pixels remain red
    assert default_numpy[5][5][0] == 255  # R
    assert default_numpy[5][5][1] == 0  # G
    assert default_numpy[5][5][2] == 0  # B

    # Test 2: Custom black background via kwargs
    image_io_black = ImageMediaIO(rgba_background_color=(0, 0, 0))
    converted_black = image_io_black.load_file(test_image_path)
    black_numpy = np.array(converted_black)

    # Check transparent pixels are black
    assert black_numpy[0][0][0] == 0  # R
    assert black_numpy[0][0][1] == 0  # G
    assert black_numpy[0][0][2] == 0  # B
    # Check opaque pixels remain red
    assert black_numpy[5][5][0] == 255  # R
    assert black_numpy[5][5][1] == 0  # G
    assert black_numpy[5][5][2] == 0  # B

    # Test 3: Custom blue background via kwargs (as list)
    image_io_blue = ImageMediaIO(rgba_background_color=[0, 0, 255])
    converted_blue = image_io_blue.load_file(test_image_path)
    blue_numpy = np.array(converted_blue)

    # Check transparent pixels are blue
    assert blue_numpy[0][0][0] == 0  # R
    assert blue_numpy[0][0][1] == 0  # G
    assert blue_numpy[0][0][2] == 255  # B

    # Test 4: Test with load_bytes method
    with open(test_image_path, "rb") as f:
        image_data = f.read()

    image_io_green = ImageMediaIO(rgba_background_color=(0, 255, 0))
    converted_green = image_io_green.load_bytes(image_data)
    green_numpy = np.array(converted_green)

    # Check transparent pixels are green
    assert green_numpy[0][0][0] == 0  # R
    assert green_numpy[0][0][1] == 255  # G
    assert green_numpy[0][0][2] == 0  # B


@pytest.mark.parametrize("background_color", [(255, 255, 255), (12, 34, 56)])
def test_image_media_io_rgb_png_transparency(tmp_path, background_color):
    image = Image.new("RGB", (2, 1), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    path = tmp_path / "rgb_transparency.png"
    image.save(path, transparency=(255, 0, 0))

    with Image.open(path) as source:
        reference = Image.alpha_composite(
            Image.new("RGBA", source.size, (*background_color, 255)),
            source.convert("RGBA"),
        ).convert("RGB")

    image_io = ImageMediaIO(rgba_background_color=background_color)
    converted = image_io.load_file(path)
    np.testing.assert_array_equal(np.asarray(converted), np.asarray(reference))


def test_image_media_io_no_mode_conversion(tmp_path):
    """image_mode=None skips conversion and preserves the original mode."""
    # RGBA image: opaque black pixel on a fully transparent background
    rgba_image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    rgba_image.putpixel((5, 5), (0, 0, 0, 255))
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Default behavior: RGBA is composited onto a white background
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    assert converted_default.media.mode == "RGB"
    assert converted_default.media.getpixel((0, 0)) == (255, 255, 255)
    assert converted_default.media.getpixel((5, 5)) == (0, 0, 0)

    # image_mode=None: original mode and alpha channel are preserved
    image_io_keep = ImageMediaIO(image_mode=None)
    converted_keep = image_io_keep.load_file(test_image_path)
    assert converted_keep.media.mode == "RGBA"
    assert converted_keep.media.getpixel((0, 0)) == (0, 0, 0, 0)
    assert converted_keep.media.getpixel((5, 5)) == (0, 0, 0, 255)


def test_image_media_io_rgba_background_color_validation():
    """Test that invalid rgba_background_color values are properly rejected."""
    # Test invalid types
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color="255,255,255")

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=255)

    # Test wrong number of elements
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255, 255, 255))

    # Test non-integer values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255.0, 255.0, 255.0))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, "255", 255))

    # Test out of range values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(256, 255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, -1, 255))

    # Test that valid values work
    ImageMediaIO(rgba_background_color=(0, 0, 0))  # Should not raise
    ImageMediaIO(rgba_background_color=[255, 255, 255])  # Should not raise
    ImageMediaIO(rgba_background_color=(128, 128, 128))  # Should not raise


def test_image_media_io_load_bytes(tmp_path):
    """Test load_bytes with valid and invalid image data."""
    # Save a valid RGB image to use as source bytes
    valid_image = Image.new("RGB", (8, 8), (100, 150, 200))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    valid_data = valid_path.read_bytes()

    # Test 1: Valid image bytes load successfully and are fully decoded
    image_io = ImageMediaIO()
    result = image_io.load_bytes(valid_data)

    # Check the returned media is a properly loaded image
    assert isinstance(result.media, Image.Image)
    assert result.media.size == (8, 8)
    assert result.media.getpixel((0, 0)) == (100, 150, 200)

    # Test 2: Garbage bytes raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"not an image")

    # Test 3: Truncated PNG header raises ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    # Test 4: Real PNG truncated mid-stream raises ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(valid_data[: len(valid_data) // 2])

    # Test 5: Empty bytes raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"")


def test_image_media_io_load_file(tmp_path):
    """Test load_file with valid and invalid image files."""
    # Save a valid RGB image to disk
    valid_image = Image.new("RGB", (4, 4), (10, 20, 30))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    # Test 1: Valid image file loads successfully and is fully decoded
    image_io = ImageMediaIO()
    result = image_io.load_file(valid_path)

    # Check the returned media is a properly loaded image
    assert isinstance(result.media, Image.Image)
    assert result.media.size == (4, 4)
    assert result.media.getpixel((0, 0)) == (10, 20, 30)

    # Test 2: File with garbage content raises ValueError
    bad_file = tmp_path / "bad.png"
    bad_file.write_bytes(b"this is not an image")

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(bad_file)

    # Test 3: File with truncated PNG header raises ValueError
    truncated_file = tmp_path / "truncated.png"
    truncated_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(truncated_file)

    # Test 4: Real PNG file truncated mid-stream raises ValueError
    valid_data = valid_path.read_bytes()
    truncated_real_file = tmp_path / "truncated_real.png"
    truncated_real_file.write_bytes(valid_data[: len(valid_data) // 2])

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(truncated_real_file)


def test_image_pixel_limit_respected():
    """A small image within the pixel limit loads successfully."""
    import vllm.envs as envs

    image = Image.new("RGB", (100, 100), (255, 0, 0))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    assert envs.VLLM_MAX_IMAGE_PIXELS >= 100 * 100

    image_io = ImageMediaIO()
    result = image_io.load_bytes(data)
    assert result.media.size == (100, 100)


def test_image_pixel_limit_rejected(monkeypatch):
    """An image exceeding the pixel limit is rejected before raster decode."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)

    image = Image.new("RGB", (20, 20), (0, 255, 0))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    image_io = ImageMediaIO()
    with pytest.raises(ValueError, match="exceed"):
        image_io.load_bytes(data)


def test_image_broken_png_rejected():
    """A PNG with a corrupted chunk structure surfaces as a client-side
    ValueError (Pillow raises SyntaxError, which is not an OSError)."""
    # Real-world request bytes: stray bytes between the IDAT CRC and the
    # IEND chunk shift the stream, so Pillow reads a bogus chunk header.
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000004000000040806000000a9f19e"
        "7e000000144944415478da63fccfc0c0c0c0c0c0c0c0c0c000000e0001ff02ab"
        "ce36890000000049454e44ae426082"
    )

    image_io = ImageMediaIO()
    with pytest.raises(ValueError, match="Failed to load image") as exc_info:
        image_io.load_bytes(data)
    assert isinstance(exc_info.value.__cause__, SyntaxError)


@pytest.mark.parametrize(
    ("data", "cause"),
    [
        # AVIF without a primary item: the AVIF decoder raises RuntimeError
        # from Image.open().
        pytest.param(
            base64.b64decode(
                "AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADrbWV0YQAAAAAAAAAh"
                "aGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAAAAAAAOAGl0bQAAAAAAAQAAAB5p"
                "bG9jAAAAAEQAAAEAAQAAAAEAAAETAAAAIwAAAChpaW5mAAAAAAABAAAAGmluZmUC"
                "AAAAAAEAAGF2MDFDb2xvcgAAAABqaXBycAAAAEtpcGNvAAAAFGlzcGUAAAAAAAAA"
                "AQAAAAEAAAAQcGl4aQAAAAADCAgIAAAADGF2MUOBAAwAAAAAE2NvbHJuY2x4AAEA"
                "DQAGgAAAABdpcG1hAAAAAAAAAAEAAQQBAoME"
            ),
            RuntimeError,
            marks=pytest.mark.skipif(
                not features.check("avif"), reason="Pillow built without AVIF"
            ),
        ),
        # BLP2 header with compression 0, which Pillow does not know:
        # BLPFormatError, a NotImplementedError and so a RuntimeError, from
        # image.load().
        (
            b"BLP2"
            + struct.pack("<i", 0)
            + bytes([1, 0, 0, 0])
            + struct.pack("<II", 1, 1)
            + bytes(128 + 1024),
            BLPFormatError,
        ),
    ],
    ids=["avif-no-primary-item", "blp-unknown-compression"],
)
def test_image_decoder_error_rejected(data, cause):
    """Decoder errors that are neither OSError nor SyntaxError surface as a
    client-side ValueError instead of HTTP 500 or 501."""
    image_io = ImageMediaIO()
    with pytest.raises(ValueError, match="Failed to load image") as exc_info:
        image_io.load_bytes(data)
    assert isinstance(exc_info.value.__cause__, cause)


def test_image_pixel_limit_disabled(monkeypatch):
    """Setting VLLM_MAX_IMAGE_PIXELS=0 disables the pixel limit."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0)

    image = Image.new("RGB", (1000, 1000), (0, 0, 255))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    image_io = ImageMediaIO()
    result = image_io.load_bytes(data)
    assert result.media.size == (1000, 1000)


def _png_bytes(width: int, height: int) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_request_decoded_budget_rejects_aggregate(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 600)
    budget = MediaDecodedBudget()
    image_io = ImageMediaIO(decoded_budget=budget)
    data = _png_bytes(10, 10)

    image_io.load_bytes(data)
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        image_io.load_bytes(data)


def test_request_decoded_budget_accepts_images_within_limit(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 800)
    budget = MediaDecodedBudget()
    image_io = ImageMediaIO(decoded_budget=budget)
    data = _png_bytes(10, 10)

    image_io.load_bytes(data)
    image_io.load_bytes(data)
    assert budget.used == 800


def test_request_decoded_budget_disabled(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 0)
    budget = MediaDecodedBudget()
    image_io = ImageMediaIO(decoded_budget=budget)
    data = _png_bytes(100, 100)

    image_io.load_bytes(data)
    image_io.load_bytes(data)
    assert budget.used == 0


def test_media_connector_enforces_request_decoded_budget(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 600)
    b64 = base64.b64encode(_png_bytes(10, 10)).decode()
    url = f"data:image/png;base64,{b64}"
    connector = MediaConnector(decoded_budget=MediaDecodedBudget())

    connector.fetch_image(url)
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        connector.fetch_image(url)


def test_image_media_io_rejects_postscript_payload():
    """EPS/PostScript bytes must not reach an external renderer."""
    import pybase64 as base64

    eps = b"""%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 10 10
%%EndComments
0 0 moveto 10 10 lineto stroke
showpage
"""
    image_io = ImageMediaIO()
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(eps)

    payload = base64.b64encode(eps).decode("ascii")
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_base64("image/png", payload)


def test_image_media_io_accepts_common_raster_formats(tmp_path):
    """Common Pillow-writable raster formats remain loadable."""
    image_io = ImageMediaIO()
    formats = (
        ("PNG", "ok.png"),
        ("JPEG", "ok.jpg"),
        ("WEBP", "ok.webp"),
        ("GIF", "ok.gif"),
        ("BMP", "ok.bmp"),
        ("TIFF", "ok.tiff"),
        ("PPM", "ok.ppm"),
        ("TGA", "ok.tga"),
        ("DDS", "ok.dds"),
        ("PCX", "ok.pcx"),
        ("SGI", "ok.sgi"),
        ("QOI", "ok.qoi"),
        ("JPEG2000", "ok.jp2"),
    )
    for fmt, name in formats:
        path = tmp_path / name
        Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format=fmt)
        result = image_io.load_bytes(path.read_bytes())
        assert result.media.size == (4, 4)


def test_allowed_image_formats_exclude_external_renderer_plugins():
    """Formats that shell out or use stubs stay off the allowlist."""
    denied = {"EPS", "WMF", "BUFR", "GRIB", "HDF5", "MPEG"}
    assert denied.isdisjoint(ALLOWED_IMAGE_FORMATS)


def test_kimi_fused_vision_rejects_postscript_payload():
    """Kimi fused vision must use the same image format allowlist."""
    from vllm.transformers_utils.processors.kimi_k25_vision_fused import _to_pil

    eps = b"""%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 10 10
%%EndComments
0 0 moveto 10 10 lineto stroke
showpage
"""
    with pytest.raises(Image.UnidentifiedImageError):
        _to_pil(eps)


def test_open_image_does_not_accept_format_override():
    """The allowlist helper must not expose a formats override."""
    assert "formats" not in inspect.signature(open_image).parameters


requires_torchcodec_images = pytest.mark.skipif(
    not has_torchcodec_image_decoding(),
    reason="requires torchcodec >= 0.16",
)


def _encode(image: Image.Image, fmt: str, **save_kwargs) -> bytes:
    buffer = BytesIO()
    image.save(buffer, fmt, **save_kwargs)
    return buffer.getvalue()


def _rgb_png_16_bit() -> bytes:
    """Pillow can't write 16-bit RGB PNGs, so build one by hand."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 2**16, (8, 8, 3), dtype=np.uint16).astype(">u2")
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload)
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 8, 8, 16, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _noisy_rgb_image(width: int = 48, height: int = 32) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _to_hwc(media: Image.Image | torch.Tensor) -> np.ndarray:
    if isinstance(media, torch.Tensor):
        return media.permute(1, 2, 0).numpy()
    return np.asarray(media)


@requires_torchcodec_images
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "AVIF"])
@pytest.mark.parametrize("mode", ["RGB", "L"])
@pytest.mark.parametrize("orientation", [None, *range(1, 9)])
def test_torchcodec_backend_matches_pil(fmt, mode, orientation):
    """The torchcodec backend must return the same pixels as Pillow, including
    grayscale-to-RGB conversion and EXIF orientation."""
    save_kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        save_kwargs["exif"] = exif.tobytes()
    data = _encode(_noisy_rgb_image().convert(mode), fmt, **save_kwargs)

    pil = ImageMediaIO().load_bytes(data)
    tc = ImageMediaIO(image_backend="torchcodec").load_bytes(data)

    assert isinstance(pil.media, Image.Image)
    assert isinstance(tc.media, torch.Tensor)
    assert tc.media.dtype == torch.uint8 and tc.media.shape[0] == 3
    np.testing.assert_array_equal(_to_hwc(tc.media), _to_hwc(pil.media))
    assert tc.io_config == pil.io_config


@requires_torchcodec_images
@pytest.mark.parametrize(
    ("data", "image_mode"),
    [
        (_encode(Image.new("RGBA", (8, 8), (255, 0, 0, 128)), "PNG"), "RGB"),
        (_encode(Image.new("RGB", (8, 8)), "PNG", transparency=(0, 0, 0)), "RGB"),
        (_encode(Image.new("CMYK", (8, 8), (0, 255, 0, 0)), "JPEG"), "RGB"),
        (_encode(Image.new("I;16", (8, 8), 1000), "PNG"), "RGB"),
        (_rgb_png_16_bit(), "RGB"),
        (_encode(Image.new("RGB", (8, 8)), "GIF"), "RGB"),
        (
            _encode(
                Image.new("RGB", (8, 8)),
                "WEBP",
                save_all=True,
                append_images=[Image.new("RGB", (8, 8), (255, 255, 255))],
            ),
            "RGB",
        ),
        (_encode(Image.new("RGB", (8, 8)), "JPEG"), None),
        (_encode(Image.new("RGB", (8, 8)), "JPEG"), "L"),
    ],
    ids=[
        "rgba",
        "rgb_png_trns",
        "cmyk",
        "16bit",
        "16bit_rgb",
        "gif",
        "animated_webp",
        "no_mode_conversion",
        "non_rgb_target",
    ],
)
def test_torchcodec_backend_uses_pil_when_outputs_would_differ(data, image_mode):
    result = ImageMediaIO(image_mode, image_backend="torchcodec").load_bytes(data)
    assert isinstance(result.media, Image.Image)
    expected = ImageMediaIO(image_mode).load_bytes(data)
    np.testing.assert_array_equal(np.asarray(result.media), np.asarray(expected.media))


@requires_torchcodec_images
def test_torchcodec_backend_checks_limits_before_decoding(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("torchcodec must not decode an over-limit image")

    import vllm.envs as envs

    monkeypatch.setattr(image_utils, "torchcodec_decode_image", fail)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)
    data = _encode(Image.new("RGB", (11, 10)), "JPEG")
    with pytest.raises(ValueError, match="exceed the maximum of 100 pixels"):
        ImageMediaIO(image_backend="torchcodec").load_bytes(data)


@requires_torchcodec_images
def test_torchcodec_backend_falls_back_to_pil_on_decode_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("JPEG decoding was disabled at build time")

    monkeypatch.setattr(image_utils, "torchcodec_decode_image", fail)
    data = _encode(_noisy_rgb_image(), "PNG")
    result = ImageMediaIO(image_backend="torchcodec").load_bytes(data)
    assert isinstance(result.media, Image.Image)

    # Corrupt input still surfaces as a client error.
    with pytest.raises(ValueError, match="Failed to load image"):
        ImageMediaIO(image_backend="torchcodec").load_bytes(data[: len(data) // 2])


def test_torchcodec_backend_unavailable_uses_pil(monkeypatch):
    monkeypatch.setattr(image_utils, "_HAS_TORCHCODEC_IMAGE_DECODING", False)
    image_io = ImageMediaIO(image_backend="torchcodec")
    assert image_io.image_backend == "pil"
    result = image_io.load_bytes(_encode(_noisy_rgb_image(), "JPEG"))
    assert isinstance(result.media, Image.Image)


def test_unknown_image_backend_rejected():
    with pytest.raises(ValueError, match="Unknown image_backend"):
        ImageMediaIO(image_backend="opencv")


@requires_torchcodec_images
def test_torchcodec_backend_hash_matches_pil():
    """The decoding backend must not change the multimodal cache key."""
    for mode in ["RGB", "L"]:
        data = _encode(_noisy_rgb_image().convert(mode), "JPEG")
        pil = ImageMediaIO().load_bytes(data)
        tc = ImageMediaIO(image_backend="torchcodec").load_bytes(data)
        assert MultiModalHasher.hash_kwargs(
            "blake3", image=tc
        ) == MultiModalHasher.hash_kwargs("blake3", image=pil)


@requires_torchcodec_images
def test_torchcodec_backend_jpeg_frame_list_video():
    frames = [_noisy_rgb_image(), _noisy_rgb_image().transpose(Image.FLIP_LEFT_RIGHT)]
    data = ",".join(
        base64.b64encode(_encode(frame, "JPEG")).decode() for frame in frames
    )

    def load(image_io):
        video_io = VideoMediaIO(image_io, num_frames=-1)
        frames, _ = video_io.load_base64("video/jpeg", data).media
        return frames

    expected = load(ImageMediaIO())
    actual = load(ImageMediaIO(image_backend="torchcodec"))
    assert actual.shape == (2, 32, 48, 3)
    np.testing.assert_array_equal(actual, expected)
