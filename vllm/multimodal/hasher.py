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

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.utils import record_function_or_nullcontext

from .media import MediaWithBytes

logger = init_logger(__name__)


@functools.lru_cache(maxsize=3)
def _get_hasher_factory(algorithm: str) -> Callable[[], "hashlib._Hash"]:
    """
    Get the hasher factory based on the configured algorithm.

    Args:
        algorithm: Hash algorithm name (blake3, sha256, or sha512)

    Returns a callable that creates a new hasher instance.
    Supports blake3 (default), sha256, and sha512 for FIPS compliance.

    See: https://github.com/vllm-project/vllm/issues/18334
    """
    algorithm = algorithm.lower()

    if algorithm == "blake3":
        from blake3 import blake3

        return blake3
    elif algorithm == "sha256":
        return hashlib.sha256
    elif algorithm == "sha512":
        return hashlib.sha512
    else:
        # This should never happen due to env_with_choices validation
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


class MultiModalHasher:
    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]:
        # Simple cases
        if isinstance(obj, (bytes, memoryview)):
            with record_function_or_nullcontext("mm:hash:branch:bytes"):
                return (obj,)
        if isinstance(obj, str):
            with record_function_or_nullcontext("mm:hash:branch:str"):
                return (obj.encode("utf-8"),)
        if isinstance(obj, (int, float)):
            with record_function_or_nullcontext("mm:hash:branch:num"):
                return (np.array(obj).tobytes(),)

        if isinstance(obj, Image.Image):
            with record_function_or_nullcontext("mm:hash:branch:pil"):
                exif = obj.getexif()
                if Image.ExifTags.Base.ImageID in exif and isinstance(
                    exif[Image.ExifTags.Base.ImageID], uuid.UUID
                ):
                    return (exif[Image.ExifTags.Base.ImageID].bytes,)

                with record_function_or_nullcontext("mm:hash:asarray"):
                    arr = np.asarray(obj)
                data = {"mode": obj.mode, "data": arr}
                palette = obj.palette
                if palette is not None:
                    data["palette"] = palette.palette
                    if palette.rawmode is not None:
                        data["palette_rawmode"] = palette.rawmode

                return cls.iter_item_to_bytes("image", data)

        if isinstance(obj, MediaWithBytes) and isinstance(obj.media, Image.Image):
            with record_function_or_nullcontext("mm:hash:branch:mwb"):
                exif = obj.media.getexif()
                if Image.ExifTags.Base.ImageID in exif and isinstance(
                    exif[Image.ExifTags.Base.ImageID], uuid.UUID
                ):
                    return (exif[Image.ExifTags.Base.ImageID].bytes,)

                return cls.iter_item_to_bytes("image", obj.original_bytes)

        if isinstance(obj, torch.Tensor):
            with record_function_or_nullcontext("mm:hash:branch:tensor"):
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
            with record_function_or_nullcontext("mm:hash:branch:ndarray"):
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

        with record_function_or_nullcontext("mm:hash:branch:pickle_fallback"):
            logger.warning(
                "No serialization method found for %s. Falling back to pickle.",
                type(obj),
            )
            return (pickle.dumps(obj),)

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[bytes | memoryview]:
        if obj is None:
            yield key.encode("utf-8")
            return
        # Recursive cases
        if isinstance(obj, (list, tuple)):
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
        hasher_factory = _get_hasher_factory(envs.VLLM_MM_HASHER_ALGORITHM)
        hasher = hasher_factory()

        total_bytes = 0
        with record_function_or_nullcontext("mm:hash:digest"):
            for k, v in sorted(kwargs.items(), key=lambda kv: kv[0]):
                for bytes_ in cls.iter_item_to_bytes(k, v):
                    with record_function_or_nullcontext("mm:hash:update"):
                        hasher.update(bytes_)
                    total_bytes += len(bytes_)

        logger.info("[mm:hash] kwargs_keys=%s total_bytes=%d", sorted(kwargs.keys()), total_bytes)
        return hasher.hexdigest()
