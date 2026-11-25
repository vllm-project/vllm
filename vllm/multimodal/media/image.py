# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import torch
from PIL import Image, TiffImagePlugin, WebPImagePlugin

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import has_torchcodec_image_decoding
from vllm.utils.serial_utils import numpy2tensor, tensor2base64
from vllm.utils.sparse_utils import (
    check_sparse_tensor_invariants_threadsafe,
    load_torch_payload,
    safe_to_dense,
)

from ..image import (
    can_decode_with_torchcodec,
    convert_image_mode,
    decode_image_torchcodec,
    normalize_image,
    open_image,
)
from .base import MediaDecodedBudget, MediaIO, MediaWithBytes

logger = init_logger(__name__)

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0

# Image decoding backends selectable via
# `--media-io-kwargs '{"image": {"image_backend": ...}}'`.
IMAGE_BACKENDS = ("pil", "torchcodec")


def _webp_canvas_size(data: bytes) -> tuple[int, int] | None:
    header = data[:30]
    if WebPImagePlugin._accept(header) is not True:
        return None
    size = WebPImagePlugin._get_size(header)
    if size is None or min(size) <= 0:
        raise Image.UnidentifiedImageError("Invalid WebP canvas size")
    return size


def _tiff_directory_tags(
    data: bytes, directory: TiffImagePlugin.ImageFileDirectory_v2
) -> set[int]:
    endian = "<" if data[:2] == b"II" else ">"
    bigtiff = data[2:4] in (b"+\0", b"\0+")
    offset = directory.offset
    count_struct = struct.Struct(endian + ("Q" if bigtiff else "H"))
    entry_struct = struct.Struct(endian + ("H18x" if bigtiff else "H10x"))
    if offset is None or offset + count_struct.size > len(data):
        raise Image.UnidentifiedImageError("Invalid TIFF directory offset")
    (count,) = count_struct.unpack_from(data, offset)
    start = offset + count_struct.size
    end = start + count * entry_struct.size
    if end > len(data):
        raise Image.UnidentifiedImageError("Incomplete TIFF directory")

    # Pillow and libtiff must account for the same directory entries.
    tags = {tag for (tag,) in entry_struct.iter_unpack(memoryview(data)[start:end])}
    if len(tags) != count:
        raise Image.UnidentifiedImageError("Duplicate TIFF directory tags")
    return tags


def _decode_size(image: Image.Image, data: bytes) -> tuple[int, int]:
    size = image.size
    if isinstance(image, TiffImagePlugin.TiffImageFile):
        tags = image.tag_v2
        directory_tags = _tiff_directory_tags(data, tags)
        if not directory_tags.isdisjoint(
            (
                TiffImagePlugin.TILEWIDTH,
                TiffImagePlugin.TILELENGTH,
                TiffImagePlugin.TILEOFFSETS,
            )
        ):
            width = tags.get(TiffImagePlugin.TILEWIDTH)
            height = tags.get(TiffImagePlugin.TILELENGTH)
            if not (
                isinstance(width, int)
                and isinstance(height, int)
                and width > 0
                and height > 0
            ):
                raise Image.UnidentifiedImageError("Invalid TIFF tile size")
            # The native decoder reads a whole tile, including any edge padding.
            if width * height > size[0] * size[1]:
                return width, height
    return size


class ImageMediaIO(MediaIO[Image.Image | torch.Tensor]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(
        self,
        image_mode: str | None = "RGB",
        *,
        decoded_budget: MediaDecodedBudget | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Target mode for loaded images; `None` keeps the original mode
        # (i.e. no conversion, alpha channel is preserved as-is).
        self.image_mode = image_mode
        self.decoded_budget = decoded_budget

        # The torchcodec backend returns CHW uint8 tensors instead of PIL
        # images, for the images it supports.
        backend = kwargs.pop("image_backend", "pil")
        if backend not in IMAGE_BACKENDS:
            raise ValueError(
                f"Unknown image_backend {backend!r}. "
                f"Available backends: {list(IMAGE_BACKENDS)}"
            )
        if backend == "torchcodec" and not has_torchcodec_image_decoding():
            logger.warning_once(
                "image_backend='torchcodec' requires torchcodec >= 0.16; "
                "falling back to Pillow."
            )
            backend = "pil"
        self.image_backend = backend

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if self.image_mode is None:
            return image
        return convert_image_mode(image, self.image_mode, self.rgba_background_color)

    def _io_config(self, mode_converted: bool) -> dict[str, Any] | None:
        if not mode_converted:
            return None
        return {
            "image_mode": self.image_mode,
            "rgba_background_color": self.rgba_background_color,
        }

    def _charge_decode_size(self, width: int, height: int) -> None:
        pixels = width * height
        max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
        if max_pixels > 0 and pixels > max_pixels:
            raise ValueError(
                f"Image dimensions {width}x{height} ({pixels} pixels) exceed "
                f"the maximum of {max_pixels} pixels. Set "
                "VLLM_MAX_IMAGE_PIXELS to increase this limit."
            )
        if self.decoded_budget is not None:
            self.decoded_budget.charge(pixels * 4)

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image | torch.Tensor]:
        try:
            # Pillow allocates the WebP canvas during open.
            webp_size = _webp_canvas_size(data)
            if webp_size is not None:
                self._charge_decode_size(*webp_size)

            image = open_image(BytesIO(data))
            if webp_size is None:
                self._charge_decode_size(*_decode_size(image, data))
            elif image.size != webp_size:
                raise Image.UnidentifiedImageError("Inconsistent WebP canvas size")
            if (
                self.image_backend == "torchcodec"
                and self.image_mode == "RGB"
                and can_decode_with_torchcodec(image, data)
                and (tensor := decode_image_torchcodec(data)) is not None
            ):
                io_config = self._io_config(image.mode != self.image_mode)
                return MediaWithBytes(tensor, data, io_config)
            image = normalize_image(image)
            image.load()
            converted = self._convert_image_mode(image)
        # Pillow can raise SyntaxError for corrupt chunks and RuntimeError
        # for decoder failures.
        except (
            OSError,
            Image.UnidentifiedImageError,
            SyntaxError,
            RuntimeError,
            Image.DecompressionBombError,
        ) as e:
            raise ValueError(f"Failed to load image: {e}") from e

        return MediaWithBytes(converted, data, self._io_config(converted is not image))

    def load_base64(
        self, media_type: str, data: str
    ) -> MediaWithBytes[Image.Image | torch.Tensor]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image | torch.Tensor]:
        return self.load_bytes(filepath.read_bytes())

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def _load_pickled_torch(self, data: bytes) -> torch.Tensor:
        return load_torch_payload(data, parameter="image_embeds")

    def _load_numpy(self, data: bytes) -> torch.Tensor:
        with BytesIO(data) as buffer:
            return numpy2tensor(np.load(buffer))

    def load_bytes(self, data: bytes) -> torch.Tensor:
        if data[:6] == MAGIC_NUMPY_PREFIX:
            return self._load_numpy(data)

        return self._load_pickled_torch(data)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        if filepath.suffix == ".npy":
            return numpy2tensor(np.load(filepath))

        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(filepath, weights_only=True)
            return safe_to_dense(tensor, parameter="image_embeds")

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
