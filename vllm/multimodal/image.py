# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import struct
from typing import Any

import torch
from PIL import Image, ImageOps

from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

try:
    from torchcodec.decoders import decode_image as torchcodec_decode_image
except (ImportError, RuntimeError, OSError):
    # torchcodec may be missing, older than 0.16 (no image decoders), or
    # unable to load its native library.
    torchcodec_decode_image = PlaceholderModule("torchcodec").placeholder_attr(  # type: ignore[assignment]
        "decoders.decode_image"
    )
    _HAS_TORCHCODEC_IMAGE_DECODING = False
else:
    _HAS_TORCHCODEC_IMAGE_DECODING = True

logger = init_logger(__name__)

# Formats accepted by Image.open for untrusted multimodal media.
#
# Audited against Pillow's registered open plugins (python -m PIL / Image.OPEN):
# Allowed: native still-image / raster readers that decode in-process.
# Denied:
#   EPS — may invoke Ghostscript on untrusted PostScript
#   WMF — Windows GDI / stub-handler path
#   BUFR / GRIB / HDF5 — stub drivers requiring external handlers
#   MPEG — identify-only; not a decodable still image
ALLOWED_IMAGE_FORMATS: tuple[str, ...] = (
    "AVIF",
    "BLP",
    "BMP",
    "CUR",
    "DCX",
    "DDS",
    "DIB",
    "FITS",
    "FLI",
    "FTEX",
    "GBR",
    "GIF",
    "ICNS",
    "ICO",
    "IM",
    "IMT",
    "IPTC",
    "JPEG",
    "JPEG2000",
    "MCIDAS",
    "MSP",
    "PCD",
    "PCX",
    "PIXAR",
    "PNG",
    "PPM",
    "PSD",
    "QOI",
    "SGI",
    "SPIDER",
    "SUN",
    "TGA",
    "TIFF",
    "WEBP",
    "XBM",
    "XPM",
    "XVTHUMB",
)


def open_image(fp: Any) -> Image.Image:
    """Open image bytes/files with the serving format allowlist."""
    return Image.open(fp, formats=ALLOWED_IMAGE_FORMATS)


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
    if to_mode == "RGB" and _has_transparency(image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return rgba_to_rgb(image, background_color)

    if image.mode == to_mode:
        return image

    return image.convert(to_mode)


def has_torchcodec_image_decoding() -> bool:
    return _HAS_TORCHCODEC_IMAGE_DECODING


# The images torchcodec decodes to the same pixels as Pillow. It differs on
# alpha (dropped rather than composited onto the background color), CMYK,
# 16-bit samples and animated images, so those are left to Pillow.
_TORCHCODEC_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "AVIF"})
_TORCHCODEC_MODES = frozenset({"RGB", "L"})


def _is_16_bit_png(image: Image.Image, data: bytes) -> bool:
    # Pillow reports 16-bit RGB PNGs as "RGB" but reduces them to 8 bits
    # differently from torchcodec. The bit depth is byte 24 of the file, in the
    # IHDR chunk which the PNG spec requires to come first.
    return image.format == "PNG" and data[24] == 16


def _orientation_visible_to_torchcodec(image: Image.Image, data: bytes) -> bool:
    # torchcodec only reads the EXIF orientation from the first APP1 segment of
    # JPEGs and from eXIf chunks before the pixel data of PNGs. Pillow also
    # finds it in a later APP1 segment (e.g. after XMP), in XMP
    # tiff:Orientation, in "Raw profile type exif" PNG text chunks and in
    # chunks after the pixel data.
    if image.format == "JPEG":
        app1 = next((s for m, s in image.applist if m == "APP1"), None)
        if app1 is not None and not app1.startswith(b"Exif\0\0"):
            return False
    if b"tiff:Orientation" in image.info.get("xmp", b""):
        return False
    if image.format == "PNG":
        if "tiff:Orientation" in image.info.get("XML:com.adobe.xmp", ""):
            return False
        if "Raw profile type exif" in image.info:
            return False
        # Pillow only parses the chunks after IDAT when loading the image, so
        # they are not in `image.info` yet. Text chunks may hold XMP or EXIF.
        pos, seen_idat = 8, False
        while pos + 8 <= len(data):
            length, kind = struct.unpack_from(">I4s", data, pos)
            if kind == b"IDAT":
                seen_idat = True
            elif seen_idat and kind in (b"eXIf", b"tEXt", b"zTXt", b"iTXt"):
                return False
            pos += length + 12
    return True


def can_decode_with_torchcodec(image: Image.Image, data: bytes) -> bool:
    """Whether torchcodec decodes `data`, opened by Pillow as `image`, to the
    same pixels as `convert_image_mode(normalize_image(image), "RGB")`."""
    return (
        image.format in _TORCHCODEC_FORMATS
        and image.mode in _TORCHCODEC_MODES
        and not _has_transparency(image)
        and not getattr(image, "is_animated", False)
        and not _is_16_bit_png(image, data)
        and _orientation_visible_to_torchcodec(image, data)
    )


def decode_image_torchcodec(data: bytes) -> torch.Tensor | None:
    """Decode `data` to a CHW uint8 RGB tensor, applying the EXIF orientation
    like `normalize_image`. Returns None if torchcodec fails, so the caller
    can fall back to Pillow: it either decodes the image (e.g. a codec left out
    of a source build of torchcodec) or raises the usual client error."""
    try:
        return torchcodec_decode_image(data, mode="RGB")
    except Exception as e:
        logger.warning_once(
            "torchcodec failed to decode an image (%s); falling back to Pillow.",
            type(e).__name__,
        )
        return None
