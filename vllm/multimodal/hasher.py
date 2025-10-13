# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
import uuid
from collections.abc import Iterable

import numpy as np
import torch
from blake3 import blake3
from PIL import Image

from vllm.logger import init_logger

logger = init_logger(__name__)


class MultiModalHasher:
    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]:
        # Simple cases
        if isinstance(obj, (bytes, memoryview)):
            return (obj,)
        if isinstance(obj, str):
            return (obj.encode("utf-8"),)
        if isinstance(obj, (int, float)):
            return (np.array(obj).tobytes(),)

        if isinstance(obj, Image.Image):
            exif = obj.getexif()
            if Image.ExifTags.Base.ImageID in exif and isinstance(
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                # If the image has exif ImageID tag, use that
                return (exif[Image.ExifTags.Base.ImageID].bytes,)
            data = {"mode": obj.mode, "data": np.asarray(obj)}
            if obj.palette is not None:
                data["palette"] = obj.palette.palette
                if obj.palette.rawmode is not None:
                    data["palette_rawmode"] = obj.palette.rawmode
            return cls.iter_item_to_bytes("image", data)
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
        hasher = blake3()

        for k, v in kwargs.items():
            for bytes_ in cls.iter_item_to_bytes(k, v):
                hasher.update(bytes_)

        return hasher.hexdigest()
