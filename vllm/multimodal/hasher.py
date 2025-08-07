# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from collections.abc import Iterable, Mapping
from typing import Union

import numpy as np
import torch
from blake3 import blake3
from PIL import Image

from vllm.logger import init_logger
from vllm.multimodal.image import convert_image_mode

logger = init_logger(__name__)

MultiModalHashDict = Mapping[str, list[str]]
"""
A dictionary containing hashes for items in each modality.
"""


class MultiModalHasher:

    @classmethod
    def serialize_item(cls, obj: object) -> Union[bytes, memoryview]:
        # Simple cases
        if isinstance(obj, str):
            return obj.encode("utf-8")
        if isinstance(obj, (bytes, memoryview)):
            return obj
        if isinstance(obj, (int, float)):
            return np.array(obj).tobytes()

        if isinstance(obj, Image.Image):
            return cls.item_to_bytes(
                "image", np.asarray(convert_image_mode(obj, "RGBA")))
        if isinstance(obj, torch.Tensor):
            return cls.item_to_bytes("tensor", obj.numpy())
        if isinstance(obj, np.ndarray):
            # If the array is non-contiguous, we need to copy it first
            arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
            return cls.item_to_bytes("ndarray", {
                "dtype": obj.dtype.str,
                "shape": obj.shape,
                "data": arr_data,
            })

        logger.warning(
            "No serialization method found for %s. "
            "Falling back to pickle.", type(obj))

        return pickle.dumps(obj)

    @classmethod
    def item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> bytes:
        return b''.join(kb + vb for kb, vb in cls.iter_item_to_bytes(key, obj))

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[tuple[bytes, Union[bytes, memoryview]]]:
        # Recursive cases
        if isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
                yield from cls.iter_item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.iter_item_to_bytes(f"{key}.{k}", v)
        else:
            key_bytes = key.encode("utf-8")
            value_bytes = cls.serialize_item(obj)
            yield key_bytes, value_bytes

    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str:
        hasher = blake3()

        for k, v in kwargs.items():
            for k_bytes, v_bytes in cls.iter_item_to_bytes(k, v):
                hasher.update(k_bytes)
                hasher.update(v_bytes)

        return hasher.hexdigest()
