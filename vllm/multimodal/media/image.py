# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import torch
from PIL import Image

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import (
    check_sparse_tensor_invariants_threadsafe,
    safe_to_dense,
)

from ..image import (
    _has_transparency,
    convert_image_mode,
    normalize_image,
    open_image,
)
from .base import MediaDecodedBudget, MediaIO, MediaWithBytes

try:
    from torchcodec.decoders import decode_image as torchcodec_decode_image
except (ImportError, RuntimeError, OSError):
    # torchcodec may be missing, older than 0.16 (no image decoders), or
    # unable to load its native library.
    torchcodec_decode_image = None

logger = init_logger(__name__)

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0

# Image decoding backends selectable via
# `--media-io-kwargs '{"image": {"image_backend": ...}}'`.
IMAGE_BACKENDS = ("pil", "torchcodec")

# Images the torchcodec backend decodes; everything else goes through Pillow.
# The Pillow-only cases are those where torchcodec's output would differ from
# the Pillow path: alpha (torchcodec drops it rather than compositing it onto
# `rgba_background_color`), CMYK and 16-bit modes, and animated images.
_TORCHCODEC_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "AVIF"})
_TORCHCODEC_MODES = frozenset({"RGB", "L"})


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
        if backend == "torchcodec" and torchcodec_decode_image is None:
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

    def _can_decode_with_torchcodec(self, image: Image.Image) -> bool:
        return (
            self.image_backend == "torchcodec"
            and self.image_mode == "RGB"
            and image.format in _TORCHCODEC_FORMATS
            and image.mode in _TORCHCODEC_MODES
            and not _has_transparency(image)
            and not getattr(image, "is_animated", False)
        )

    def _decode_with_torchcodec(self, data: bytes) -> torch.Tensor | None:
        assert torchcodec_decode_image is not None
        try:
            # Like the Pillow path, torchcodec applies the EXIF orientation.
            return torchcodec_decode_image(data, mode="RGB")
        except Exception as e:
            # Let Pillow decode it instead: it either succeeds (e.g. a codec
            # left out of a source build of torchcodec) or raises the same
            # client error as with the Pillow backend.
            logger.warning_once(
                "torchcodec failed to decode an image (%s); falling back to Pillow.",
                type(e).__name__,
            )
            return None

    def _io_config(self, mode_converted: bool) -> dict[str, Any] | None:
        if not mode_converted:
            return None
        return {
            "image_mode": self.image_mode,
            "rgba_background_color": self.rgba_background_color,
        }

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image | torch.Tensor]:
        try:
            # Opening only parses the header, so the limits below are checked
            # before any pixel data is decoded.
            image = open_image(BytesIO(data))
            w, h = image.size
            max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
            if max_pixels > 0 and w * h > max_pixels:
                raise ValueError(
                    f"Image dimensions {w}x{h} ({w * h} pixels) exceed "
                    f"the maximum of {max_pixels} pixels. Set "
                    f"VLLM_MAX_IMAGE_PIXELS to increase this limit."
                )
            if self.decoded_budget is not None:
                self.decoded_budget.charge(w * h * 4)
            if self._can_decode_with_torchcodec(image):
                tensor = self._decode_with_torchcodec(data)
                if tensor is not None:
                    io_config = self._io_config(image.mode != self.image_mode)
                    return MediaWithBytes(tensor, data, io_config)
            image = normalize_image(image)
            image.load()
            converted = self._convert_image_mode(image)
        # Pillow raises SyntaxError (not OSError) for broken PNG chunks.
        except (OSError, Image.UnidentifiedImageError, SyntaxError) as e:
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
