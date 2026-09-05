# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import hashlib
import pickle
import uuid
from collections.abc import Callable, Iterable

import numpy as np
import torch
from PIL import Image

from vllm.config.multimodal import MMHasherAlgorithm
from vllm.logger import init_logger

from .media import MediaWithBytes

logger = init_logger(__name__)

# Framing for the digest input. The hash is built by feeding a stream of byte
# chunks to `hasher.update`, so the stream must be uniquely decodable: without
# an explicit length in front of every chunk and a tag in front of every
# container, distinct inputs serialize to identical bytes and share a cache
# entry. See `test_hash_collision_*` in tests/multimodal/test_hasher.py.
_LENGTH_BYTES = 8
_TAG_NONE = b"\x00"
_TAG_SEQUENCE = b"\x01"
_TAG_MAPPING = b"\x02"
_TAG_LEAF = b"\x03"


def _encode_length(value: int) -> bytes:
    return value.to_bytes(_LENGTH_BYTES, "little")


def _framed(chunk: bytes | memoryview) -> Iterable[bytes | memoryview]:
    """Yield *chunk* preceded by its size in bytes."""
    size = chunk.nbytes if isinstance(chunk, memoryview) else len(chunk)
    yield _encode_length(size)
    yield chunk


@functools.lru_cache(maxsize=3)
def _get_hasher_factory(
    algorithm: MMHasherAlgorithm,
) -> Callable[[], "hashlib._Hash"]:
    """
    Get the hasher factory based on the configured algorithm.

    Args:
        algorithm: Hash algorithm name (blake3, sha256, or sha512)

    Returns a callable that creates a new hasher instance.
    Supports blake3 (default), sha256, and sha512 for FIPS compliance.

    See: https://github.com/vllm-project/vllm/issues/18334
    """

    if algorithm == "blake3":
        from blake3 import blake3

        return blake3
    elif algorithm == "sha256":
        return hashlib.sha256
    elif algorithm == "sha512":
        return hashlib.sha512
    else:
        # This should never happen due to config validation
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


class MultiModalHasher:
    """Derives multi-modal cache keys.

    Every method here yields *framed* chunks: each chunk is preceded by its
    length and each container by its kind, so that the concatenation fed to the
    digest is uniquely decodable and distinct inputs cannot share a key.
    """

    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]:
        # Simple cases
        if isinstance(obj, (bytes, memoryview)):
            return _framed(obj)
        if isinstance(obj, str):
            return _framed(obj.encode("utf-8"))
        if isinstance(obj, (int, float)):
            return _framed(np.array(obj).tobytes())

        if isinstance(obj, Image.Image):
            exif = obj.getexif()
            if Image.ExifTags.Base.ImageID in exif and isinstance(
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return _framed(exif[Image.ExifTags.Base.ImageID].bytes)

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
                return _framed(exif[Image.ExifTags.Base.ImageID].bytes)

            if obj.io_config:
                return cls.iter_item_to_bytes(
                    "image",
                    {"io_config": obj.io_config, "data": obj.original_bytes},
                )
            return cls.iter_item_to_bytes("image", obj.original_bytes)

        if isinstance(obj, MediaWithBytes) and isinstance(obj.media, np.ndarray):
            frames = obj.media
            if frames.nbytes < len(obj.original_bytes):
                return cls.iter_item_to_bytes("video", frames)
            return cls.iter_item_to_bytes("video", obj.original_bytes)

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
            if obj.ndim == 0:
                arr_data = obj.item()
            elif obj.flags.c_contiguous:
                # Not valid for 0-D arrays
                arr_data = obj.view(np.uint8).data
            else:
                # If the array is non-contiguous, we need to copy it first
                arr_data = obj.tobytes()

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

        return _framed(pickle.dumps(obj))

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[bytes | memoryview]:
        """Yield the digest input for a single ``key``/``obj`` pair."""
        yield from _framed(key.encode("utf-8"))
        yield from cls.iter_value_to_bytes(obj)

    @classmethod
    def iter_value_to_bytes(
        cls,
        obj: object,
    ) -> Iterable[bytes | memoryview]:
        """Yield the digest input for a value, tagged by its container kind.

        Containers carry their kind and length so that a nested structure can
        never serialize to the same bytes as a differently shaped one (a list
        and a mapping keyed by stringified indices, for example).
        """
        if obj is None:
            yield _TAG_NONE
        elif isinstance(obj, (list, tuple)):
            yield _TAG_SEQUENCE
            yield _encode_length(len(obj))
            for elem in obj:
                yield from cls.iter_value_to_bytes(elem)
        elif isinstance(obj, dict):
            yield _TAG_MAPPING
            yield _encode_length(len(obj))
            for k, v in obj.items():
                yield from _framed(str(k).encode("utf-8"))
                yield from cls.iter_value_to_bytes(v)
        else:
            yield _TAG_LEAF
            yield from cls.serialize_item(obj)

    @classmethod
    def hash_kwargs(
        cls,
        algorithm: MMHasherAlgorithm,
        /,
        **kwargs: object,
    ) -> str:
        hasher_factory = _get_hasher_factory(algorithm)
        hasher = hasher_factory()

        for k, v in sorted(kwargs.items(), key=lambda kv: kv[0]):
            for bytes_ in cls.iter_item_to_bytes(k, v):
                hasher.update(bytes_)

        return hasher.hexdigest()
