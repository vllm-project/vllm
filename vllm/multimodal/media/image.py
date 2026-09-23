# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import pybase64
import torch
from PIL import Image

import vllm.envs as envs
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import (
    check_sparse_tensor_invariants_threadsafe,
    safe_to_dense,
)

from ..image import (
    convert_image_mode,
    get_image_id_bytes,
    normalize_image,
    rgba_to_rgb,
)
from .base import DecodeSpec, MediaIO, MediaRef

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0


class ImageMediaIO(MediaIO[Image.Image]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self, image_mode: str | None = "RGB", **kwargs) -> None:
        super().__init__()

        # Target mode for loaded images; `None` keeps the original mode
        # (i.e. no conversion, alpha channel is preserved as-is).
        self.image_mode = image_mode
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

    def get_decode_spec(self) -> DecodeSpec:
        # The resolved values win: `rgba_background_color` also sits in
        # `kwargs`, in whatever shape the caller supplied it.
        return DecodeSpec(
            {
                **self.kwargs,
                "image_mode": self.image_mode,
                "rgba_background_color": self.rgba_background_color,
            }
        )

    def _convert_image_mode(self, image: Image.Image) -> Image.Image:
        """Convert image mode with custom background color."""
        if self.image_mode is None or image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(
                image, self.image_mode, self.rgba_background_color
            )

    def open_header(self, data: bytes) -> Image.Image:
        """Open only the image header, leaving the pixels undecoded.

        Mode, size and EXIF are readable from the header alone, so this is
        also where the max-pixels guard and the cache key's EXIF probe run.
        """
        try:
            image = Image.open(BytesIO(data))
            w, h = image.size
            max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
            if max_pixels > 0 and w * h > max_pixels:
                raise ValueError(
                    f"Image dimensions {w}x{h} ({w * h} pixels) exceed "
                    f"the maximum of {max_pixels} pixels. Set "
                    f"VLLM_MAX_IMAGE_PIXELS to increase this limit."
                )
        except (OSError, Image.UnidentifiedImageError) as e:
            raise ValueError(f"Failed to load image: {e}") from e

        return image

    def _exif_key(self, header_image: Image.Image) -> bytes | None:
        """The EXIF `ImageID` cache key of a header-opened image, if it has one.

        Key derivation must never rasterize pixels: `PngImageFile.getexif()`
        calls `.load()` to scan for a trailing eXIf chunk when `info` has
        none, and on Pillow >= 12 `getexif()` loads unconditionally. Per the
        PNG spec eXIf must precede IDAT, so compliant PNGs carry their EXIF in
        `info` at open time; a non-compliant trailing-eXIf PNG with an
        `ImageID` is not recognized here and keys off its bytes instead.
        """
        if header_image.format == "PNG" and "exif" not in header_image.info:
            return None
        return get_image_id_bytes(header_image)

    def _decode_pixels(self, image: Image.Image) -> Image.Image:
        """Rasterize a header-opened image, then normalize and convert it."""
        try:
            image = normalize_image(image)
            image.load()
            return self._convert_image_mode(image)
        except (OSError, Image.UnidentifiedImageError) as e:
            raise ValueError(f"Failed to load image: {e}") from e

    def _decode_from_bytes(self, data: bytes) -> Image.Image:
        return self._decode_pixels(self.open_header(data))

    def load_bytes(self, data: bytes) -> Image.Image:
        return self._decode_from_bytes(data)

    def load_bytes_ref(self, data: bytes) -> MediaRef[Image.Image]:
        """Eager header parse + lazy pixel decode.

        The header is parsed eagerly so oversized images still fail at fetch
        time and so the cache key can carry the EXIF `ImageID` without
        decoding. Unparsable headers and pixel decoding are deferred to first
        access, where decode errors surface.
        """
        spec = self.get_decode_spec()
        try:
            header_image = self.open_header(data)
        except ValueError as e:
            # The max-pixels ValueError has no __cause__ and stays eager;
            # header parse failures (wrapping OSError) are deferred.
            if e.__cause__ is None:
                raise
            return MediaRef(partial(self._decode_from_bytes, data), data, spec)
        return MediaRef(
            partial(self._decode_pixels, header_image),
            data,
            spec,
            key=self._exif_key(header_image),
        )

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> Image.Image:
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
        buffer = BytesIO(data)
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(buffer, weights_only=True)
            return safe_to_dense(tensor, parameter="image_embeds")

    def _load_numpy(self, data: bytes) -> torch.Tensor:
        with BytesIO(data) as buffer:
            return torch.from_numpy(np.load(buffer))

    def load_bytes(self, data: bytes) -> torch.Tensor:
        if data[:6] == MAGIC_NUMPY_PREFIX:
            return self._load_numpy(data)

        return self._load_pickled_torch(data)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        if filepath.suffix == ".npy":
            return torch.from_numpy(np.load(filepath))

        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(filepath, weights_only=True)
            return safe_to_dense(tensor, parameter="image_embeds")

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
