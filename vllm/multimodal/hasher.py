# SPDX-License-Identifier: Apache-2.0

import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from blake3 import blake3
from PIL import Image

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt

logger = init_logger(__name__)

MultiModalHashDict = Mapping[str, list[str]]
"""
A dictionary containing hashes for items in each modality.
"""


class MultiModalHasher:

    @classmethod
    def serialize_item(cls, obj: object) -> bytes:
        # Simple cases
        if isinstance(obj, str):
            return obj.encode("utf-8")
        if isinstance(obj, bytes):
            return obj
        if isinstance(obj, Image.Image):
            return obj.tobytes()

        # Convertible to NumPy arrays
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()
        if isinstance(obj, (int, float)):
            obj = np.array(obj)
        if isinstance(obj, np.ndarray):
            return obj.tobytes()

        logger.warning(
            "No serialization method found for %s. "
            "Falling back to pickle.", type(obj))

        return pickle.dumps(obj)

    @classmethod
    def item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[tuple[bytes, bytes]]:
        # Recursive cases
        if isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
                yield from cls.item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.item_to_bytes(f"{key}.{k}", v)
        else:
            key_bytes = cls.serialize_item(key)
            value_bytes = cls.serialize_item(obj)
            yield key_bytes, value_bytes

    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str:
        hasher = blake3()

        for k, v in kwargs.items():
            for k_bytes, v_bytes in cls.item_to_bytes(k, v):
                hasher.update(k_bytes)
                hasher.update(v_bytes)

        return hasher.hexdigest()

    @classmethod
    def hash_prompt_mm_data(
            cls, prompt: "TokensPrompt") -> Optional["MultiModalHashDict"]:
        """Hash multimodal data in the user input prompt if they exist."""

        if "multi_modal_data" not in prompt:
            return None

        mm_data = prompt["multi_modal_data"]
        if not mm_data:
            # mm_data can be None or an empty dict.
            return None

        mm_items = {
            modality: items if isinstance(items, list) else [items]
            for modality, items in mm_data.items()
        }

        mm_hashes = {
            modality: [cls.hash_kwargs(**{modality: item}) for item in items]
            for modality, items in mm_items.items()
        }

        return mm_hashes
