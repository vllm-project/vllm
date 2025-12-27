# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal content hashing utilities.

This module provides hashing functionality for multimodal content (images,
tensors, etc.) used in cache key generation. It supports both high-performance
blake3 hashing and FIPS 140-3 compliant SHA-256 hashing.

FIPS Compliance:
    blake3 is not FIPS 140-3 approved. For environments requiring FIPS
    compliance (government, healthcare, finance), use one of these methods:

    1. Config (recommended): Set `use_fips_hashing=True` in MultiModalConfig
    2. Environment variable: Set VLLM_USE_FIPS_HASHING=1

Configuration:
    MultiModalConfig.use_fips_hashing:
        - None (default): Use environment variable or auto-detect
        - True: Force FIPS-compliant SHA-256 hashing
        - False: Use blake3 for faster hashing (if available)

Environment Variables:
    VLLM_USE_FIPS_HASHING: Set to "1", "true", or "yes" to enable
        FIPS-compliant SHA-256 hashing instead of blake3.
"""

import hashlib
import os
import pickle
import uuid
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image

from vllm.logger import init_logger

from .base import MediaWithBytes

logger = init_logger(__name__)

# blake3 is optional - not FIPS 140-3 approved
# In FIPS-constrained environments, blake3 may not be available or allowed
try:
    from blake3 import blake3 as _blake3

    _HAS_BLAKE3 = True
except ImportError:
    _blake3 = None
    _HAS_BLAKE3 = False


def _get_fips_hashing_default() -> bool:
    """Determine the default FIPS hashing setting from environment.

    Returns True if:
    - VLLM_USE_FIPS_HASHING environment variable is set to a truthy value
    - blake3 is not available (automatic fallback)

    Returns:
        bool: True if FIPS-compliant SHA-256 should be used, False for blake3.
    """
    fips_env = os.environ.get("VLLM_USE_FIPS_HASHING", "0")
    use_fips = fips_env.lower() in ("1", "true", "yes")

    if use_fips:
        logger.info("FIPS-compliant hashing enabled via VLLM_USE_FIPS_HASHING")
    elif not _HAS_BLAKE3:
        logger.info("blake3 not available, using FIPS-compliant SHA-256 hashing")

    return use_fips or not _HAS_BLAKE3


_USE_FIPS_HASHING = _get_fips_hashing_default()


def configure_fips_hashing(use_fips: bool | None) -> None:
    """Configure FIPS-compliant hashing based on MultiModalConfig.

    This function should be called during engine initialization when
    the MultiModalConfig is available.

    Args:
        use_fips: If True, force FIPS-compliant SHA-256 hashing.
            If False, use blake3 (if available).
            If None, use the environment variable or auto-detect.
    """
    global _USE_FIPS_HASHING

    if use_fips is None:
        # Use environment variable / auto-detection (already set at module load)
        return

    if use_fips:
        logger.info("FIPS-compliant hashing enabled via MultiModalConfig")
        _USE_FIPS_HASHING = True
    elif _HAS_BLAKE3:
        logger.info("Using blake3 hashing (configured via MultiModalConfig)")
        _USE_FIPS_HASHING = False
    else:
        logger.warning(
            "blake3 requested but not available, using FIPS-compliant SHA-256"
        )
        _USE_FIPS_HASHING = True


class _Blake3Hasher:
    """Wrapper for blake3 hasher with consistent interface."""

    def __init__(self):
        if _blake3 is None:
            raise RuntimeError("blake3 is not available")
        self._hasher = _blake3()

    def update(self, data: bytes | memoryview) -> None:
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


class _Sha256Hasher:
    """FIPS 140-3 compliant SHA-256 hasher with consistent interface.

    This provides the same interface as _Blake3Hasher but uses the
    FIPS-approved SHA-256 algorithm from hashlib.
    """

    def __init__(self):
        self._hasher = hashlib.sha256()

    def update(self, data: bytes | memoryview) -> None:
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def _create_hasher() -> _Blake3Hasher | _Sha256Hasher:
    """Create the appropriate hasher based on FIPS configuration.

    Returns:
        A hasher instance with update() and hexdigest() methods.
    """
    if _USE_FIPS_HASHING:
        return _Sha256Hasher()
    return _Blake3Hasher()


class MultiModalHasher:
    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]:
        # Simple cases
        if isinstance(obj, bytes | memoryview):
            return (obj,)
        if isinstance(obj, str):
            return (obj.encode("utf-8"),)
        if isinstance(obj, int | float):
            return (np.array(obj).tobytes(),)

        if isinstance(obj, Image.Image):
            exif = obj.getexif()
            if Image.ExifTags.Base.ImageID in exif and isinstance(
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return (exif[Image.ExifTags.Base.ImageID].bytes,)

            data = {"mode": obj.mode, "data": np.asarray(obj)}
            palette = obj.palette
            if palette is not None:
                data["palette"] = palette.palette
                if palette.rawmode is not None:
                    data["palette_rawmode"] = palette.rawmode

            return cls.iter_item_to_bytes("image", data)

        if isinstance(obj, MediaWithBytes) and isinstance(obj.media, Image.Image):
            exif = obj.media.getexif()
            if Image.ExifTags.Base.ImageID in exif and isinstance(
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return (exif[Image.ExifTags.Base.ImageID].bytes,)

            return cls.iter_item_to_bytes("image", obj.original_bytes)

        if isinstance(obj, torch.Tensor):
            tensor_obj: torch.Tensor = obj.cpu()
            tensor_dtype = tensor_obj.dtype
            tensor_shape = tensor_obj.shape

            # NumPy does not support bfloat16.
            # Workaround: View the tensor as a contiguous 1D array of bytes
            if tensor_dtype == torch.bfloat16:
                tensor_obj = tensor_obj.contiguous()
                tensor_obj = tensor_obj.view((tensor_obj.numel(),)).view(torch.uint8)

                return cls.iter_item_to_bytes(
                    "tensor",
                    {
                        "original_dtype": str(tensor_dtype),
                        "original_shape": tuple(tensor_shape),
                        "data": tensor_obj.numpy(),
                    },
                )
            return cls.iter_item_to_bytes("tensor", tensor_obj.numpy())
        if isinstance(obj, np.ndarray):
            # If the array is non-contiguous, we need to copy it first
            arr_data = (
                obj.view(np.uint8).data if obj.flags.c_contiguous else obj.tobytes()
            )
            return cls.iter_item_to_bytes(
                "ndarray",
                {
                    "dtype": obj.dtype.str,
                    "shape": obj.shape,
                    "data": arr_data,
                },
            )
        logger.warning(
            "No serialization method found for %s. Falling back to pickle.", type(obj)
        )

        return (pickle.dumps(obj),)

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[bytes | memoryview]:
        # Recursive cases
        if isinstance(obj, list | tuple):
            for i, elem in enumerate(obj):
                yield from cls.iter_item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.iter_item_to_bytes(f"{key}.{k}", v)
        else:
            yield key.encode("utf-8")
            yield from cls.serialize_item(obj)

    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str:
        hasher = _create_hasher()

        for k, v in kwargs.items():
            for bytes_ in cls.iter_item_to_bytes(k, v):
                hasher.update(bytes_)

        return hasher.hexdigest()
