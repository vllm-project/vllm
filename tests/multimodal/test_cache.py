# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.cache import MultiModalCache, MultiModalCacheItemMetadata
from vllm.multimodal.inputs import (MultiModalFieldElem, MultiModalKwargs,
                                    MultiModalKwargsItem,
                                    MultiModalSharedField)


def _dummy_elem(modality: str, key: str, size: int):
    return MultiModalFieldElem(
        modality=modality,
        key=key,
        data=torch.empty((size, ), dtype=torch.int8),
        field=MultiModalSharedField(1),
    )


def _dummy_item(modality: str, size_by_key: dict[str, int]):
    return MultiModalKwargsItem.from_elems([
        _dummy_elem(modality, key, size) for key, size in size_by_key.items()
    ])


def _dummy_kw(size_by_key_modality: dict[str, dict[str, int]]):
    return MultiModalKwargs.from_items([
        _dummy_item(modality, size_by_key)
        for modality, size_by_key in size_by_key_modality.items()
    ])


# yapf: disable
@pytest.mark.parametrize(
    ("item", "expected_size"),
    [
        (_dummy_item("a", {"a1": 100}), 100),
        (_dummy_item("a", {"a1": 100, "a2": 110}), 210),
        (_dummy_kw({"a": {"a1": 100, "a2": 110}, "b": {"b1": 120, "b2": 130}}), 460),  # noqa: E501
    ],
)
# yapf: enable
def test_cache_item_size(item, expected_size):
    cache = MultiModalCache.get_lru_cache(2048, type(item))

    cache[""] = item
    assert cache.currsize == expected_size

    cache[""] = MultiModalCacheItemMetadata.wraps(item)
    assert cache.currsize == expected_size
