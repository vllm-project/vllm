# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.media import ImageMediaIO, LazyMedia

pytestmark = pytest.mark.cpu_test

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


def _png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _hash(item) -> str:
    return MultiModalHasher.hash_kwargs("blake3", image=item)


def test_load_bytes_lazy_defers_decode():
    """load_bytes_lazy opens only the header; pixels decode on first access."""
    data = _png_bytes(Image.new("RGB", (8, 8), (100, 150, 200)))
    image_io = ImageMediaIO()

    lazy = image_io.load_bytes_lazy(data)
    assert isinstance(lazy, LazyMedia)
    assert not lazy.is_decoded
    assert lazy.original_bytes == data
    # The header is readable without triggering pixel decoding
    assert lazy.header_image.size == (8, 8)
    assert lazy.header_image.mode == "RGB"
    assert not lazy.is_decoded

    decoded = lazy.media
    assert lazy.is_decoded
    assert decoded is lazy.media  # decode result is cached

    eager = image_io.load_bytes(data)
    assert lazy.io_config == eager.io_config
    assert decoded.mode == eager.media.mode
    assert np.array_equal(np.array(decoded), np.array(eager.media))


def test_load_bytes_lazy_preserves_mode_when_disabled():
    """image_mode=None keeps the original mode and leaves io_config None."""
    data = _png_bytes(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    lazy = ImageMediaIO(image_mode=None).load_bytes_lazy(data)

    assert lazy.io_config is None
    assert lazy.media.mode == "RGBA"


def test_load_bytes_lazy_error_timing():
    """Decode errors surface at first access, not at construction."""
    image_io = ImageMediaIO()

    # Unparsable header: deferred so decode errors surface at the use site
    lazy = image_io.load_bytes_lazy(b"not an image")
    assert lazy.header_image is None
    assert not lazy.is_decoded
    with pytest.raises(ValueError, match="Failed to load image"):
        _ = lazy.media

    # Truncated JPEG: the header still parses, pixel decoding fails
    buf = BytesIO()
    Image.new("RGB", (64, 64), (100, 150, 200)).save(buf, format="JPEG")
    data = buf.getvalue()[:-50]

    lazy = image_io.load_bytes_lazy(data)
    assert not lazy.is_decoded
    with pytest.raises(ValueError, match="Failed to load image"):
        _ = lazy.media


def test_load_bytes_lazy_rejects_oversized_image_eagerly(monkeypatch):
    """The VLLM_MAX_IMAGE_PIXELS check stays in the eager header phase."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)

    data = _png_bytes(Image.new("RGB", (20, 20), (0, 255, 0)))
    with pytest.raises(ValueError, match="exceed"):
        ImageMediaIO().load_bytes_lazy(data)


def test_load_bytes_lazy_hash_matches_eager_no_exif():
    """Same bytes, no EXIF: lazy and eager items hash identically, without
    triggering a decode."""
    data = _png_bytes(Image.new("RGB", (8, 8), (100, 150, 200)))
    image_io = ImageMediaIO()
    lazy = image_io.load_bytes_lazy(data)
    eager = image_io.load_bytes(data)

    assert lazy.io_config is None and eager.io_config is None
    assert _hash(lazy) == _hash(eager)
    assert not lazy.is_decoded


def test_load_bytes_lazy_hash_matches_eager_with_exif():
    """Same bytes, with EXIF: lazy and eager items hash identically."""
    image = Image.new("RGB", (8, 4), (10, 20, 30))
    exif = Image.Exif()
    exif[Image.ExifTags.Base.ImageID] = uuid.uuid4().bytes
    exif[Image.ExifTags.Base.Orientation] = 6
    buf = BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    data = buf.getvalue()

    image_io = ImageMediaIO()
    lazy = image_io.load_bytes_lazy(data)
    eager = image_io.load_bytes(data)

    assert lazy.io_config == eager.io_config
    assert _hash(lazy) == _hash(eager)
    assert not lazy.is_decoded
    # Decode equivalence, including the EXIF orientation transpose
    assert lazy.media.size == eager.media.size == (4, 8)
    assert np.array_equal(np.array(lazy.media), np.array(eager.media))


def test_load_bytes_lazy_hash_matches_eager_mode_conversion():
    """Non-RGB source: io_config is computed eagerly and hashes match the
    eager path exactly."""
    data = _png_bytes(Image.new("RGBA", (8, 8), (255, 0, 0, 128)))

    for background in ((255, 255, 255), (0, 0, 0)):
        image_io = ImageMediaIO(rgba_background_color=background)
        lazy = image_io.load_bytes_lazy(data)
        eager = image_io.load_bytes(data)

        assert lazy.io_config == eager.io_config is not None
        assert _hash(lazy) == _hash(eager)
        assert not lazy.is_decoded
        assert np.array_equal(np.array(lazy.media), np.array(eager.media))


def test_load_bytes_lazy_png_hash_does_not_decode(monkeypatch):
    """Hashing a lazy PNG must not trigger pixel decoding.

    PngImageFile.getexif() calls .load() when info lacks an "exif" key, so
    the hasher must only consult EXIF parsed at open time.
    """
    data = _png_bytes(Image.new("RGB", (8, 8), (1, 2, 3)))
    image_io = ImageMediaIO()
    eager = image_io.load_bytes(data)
    lazy = image_io.load_bytes_lazy(data)

    def fail_load(self):
        raise AssertionError("hashing triggered a pixel decode")

    monkeypatch.setattr(lazy.header_image, "load", fail_load)
    assert _hash(lazy) == _hash(eager)
    assert not lazy.is_decoded
