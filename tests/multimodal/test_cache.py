# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.multimodal.cache import (CachedMultiModalInputExchanger,
                                   MultiModalCache,
                                   MultiModalCacheItemMetadata)
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (MultiModalFieldElem, MultiModalKwargs,
                                    MultiModalKwargsItem,
                                    MultiModalSharedField)
from vllm.multimodal.registry import MultiModalRegistry


def _dummy_elem(
    modality: str,
    key: str,
    size: int,
    *,
    rng: Optional[np.random.RandomState] = None,
):
    if rng is None:
        data = torch.empty((size, ), dtype=torch.int8)
    else:
        data = torch.from_numpy(rng.randint(4, size=(size, ), dtype=np.int8))

    return MultiModalFieldElem(
        modality=modality,
        key=key,
        data=data,
        field=MultiModalSharedField(1),
    )


def _dummy_item(
    modality: str,
    size_by_key: dict[str, int],
    *,
    rng: Optional[np.random.RandomState] = None,
):
    return MultiModalKwargsItem.from_elems([
        _dummy_elem(modality, key, size, rng=rng)
        for key, size in size_by_key.items()
    ])


def _dummy_kw(
    size_by_key_modality: dict[str, dict[str, int]],
    *,
    rng: Optional[np.random.RandomState] = None,
):
    return MultiModalKwargs.from_items([
        _dummy_item(modality, size_by_key, rng=rng)
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


def _create_vllm_config(*, mm_processor_cache_gb: float):
    return VllmConfig(model_config=ModelConfig(
        mm_processor_cache_gb=mm_processor_cache_gb), )


@pytest.mark.parametrize("contains_call_count", [1, 2, 3])
def test_ipc_enable_disable_consistency(contains_call_count):
    cache_size_gb = 1 / (1 << 20)

    vllm_config_ipc_enabled = _create_vllm_config(
        mm_processor_cache_gb=cache_size_gb)
    vllm_config_ipc_disabled = _create_vllm_config(mm_processor_cache_gb=0)
    mm_registry = MultiModalRegistry()

    p0_ipc_enabled = CachedMultiModalInputExchanger.for_p0(
        vllm_config_ipc_enabled,
        mm_registry,
    )
    p1_ipc_enabled = CachedMultiModalInputExchanger.for_p1(
        vllm_config_ipc_enabled,
        mm_registry,
    )

    p0_ipc_disabled = CachedMultiModalInputExchanger.for_p0(
        vllm_config_ipc_disabled,
        mm_registry,
    )
    p1_ipc_disabled = CachedMultiModalInputExchanger.for_p1(
        vllm_config_ipc_disabled,
        mm_registry,
    )

    n_iter = 100
    item_capacity = 8
    item_size_gb = int(cache_size_gb / item_capacity)
    hit_rate = 0.5
    max_items_per_iter = 3

    rng = np.random.RandomState(0)
    all_items = [
        _dummy_item("item", {"key": item_size_gb}, rng=rng)
        for _ in range(int(item_capacity / hit_rate))
    ]
    all_hashes = [
        MultiModalHasher.hash_kwargs(item=item.require_data())
        for item in all_items
    ]

    for it in range(n_iter):
        num_items_to_select = rng.randint(0, max_items_per_iter)
        item_idxs_to_select = rng.choice(len(all_items), num_items_to_select)

        selected_items = [all_items[idx] for idx in item_idxs_to_select]
        selected_hashes = [all_hashes[idx] for idx in item_idxs_to_select]

        for _ in range(contains_call_count):
            p0_ipc_enabled.contains_items(selected_hashes)
        items_ipc_enabled = p0_ipc_enabled.get_and_update(
            selected_items,
            selected_hashes,
        )
        result_ipc_enabled = p1_ipc_enabled.get_and_update(
            items_ipc_enabled,
            selected_hashes,
        )

        for _ in range(contains_call_count):
            p1_ipc_enabled.contains_items(selected_hashes)
        items_ipc_disabled = p0_ipc_disabled.get_and_update(
            selected_items,
            selected_hashes,
        )
        result_ipc_disabled = p1_ipc_disabled.get_and_update(
            items_ipc_disabled,
            selected_hashes,
        )

        assert result_ipc_enabled == result_ipc_disabled, f"Failed at {it=}"
