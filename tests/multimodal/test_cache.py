# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    MultiModalCache,
    MultiModalProcessorCacheItem,
    MultiModalProcessorCacheItemMetadata,
    engine_receiver_cache_from_config,
    processor_cache_from_config,
)
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
)
from vllm.multimodal.processing import PromptInsertion

pytestmark = pytest.mark.cpu_test


def _dummy_elem(
    modality: str,
    key: str,
    size: int,
    *,
    rng: np.random.RandomState | None = None,
):
    if rng is None:
        data = torch.empty((size,), dtype=torch.int8)
    else:
        data = torch.from_numpy(rng.randint(4, size=(size,), dtype=np.int8))

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
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItem.from_elems(
        [_dummy_elem(modality, key, size, rng=rng) for key, size in size_by_key.items()]
    )


def _dummy_items(
    size_by_key_modality: dict[str, dict[str, int]],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItems.from_seq(
        [
            _dummy_item(modality, size_by_key, rng=rng)
            for modality, size_by_key in size_by_key_modality.items()
        ]
    )


@pytest.mark.parametrize(
    ("item", "expected_size"),
    [
        (_dummy_item("a", {"a1": 100}), 100),
        (_dummy_item("a", {"a1": 100, "a2": 110}), 210),
        (_dummy_items({"a": {"a1": 100, "a2": 110}, "b": {"b1": 120, "b2": 130}}), 460),  # noqa: E501
        (
            _dummy_items(
                {"a": {"a1": 100, "a2": 110}, "b": {"b1": 120, "b2": 130}}
            ).get_data(),
            460,
        ),  # noqa: E501
    ],
)
def test_cache_item_size(item, expected_size):
    cache = MultiModalCache.get_lru_cache(2048, type(item))

    cache[""] = item
    assert cache.currsize == expected_size

    prompt_update = PromptInsertion("dummy", "target", "insertion").resolve(0)

    cache[""] = MultiModalProcessorCacheItem(item, [prompt_update])
    assert cache.currsize == expected_size

    cache[""] = MultiModalProcessorCacheItemMetadata(item, [prompt_update])
    assert cache.currsize == expected_size


def _create_vllm_config(
    *,
    mm_processor_cache_gb: float,
    enable_ipc: bool,
):
    return VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_gb=mm_processor_cache_gb,
        ),
        parallel_config=ParallelConfig(data_parallel_size=1 if enable_ipc else 2),
    )


def _compare_caches(
    config_0: VllmConfig,
    config_1: VllmConfig,
    *,
    item_capacity: int = 8,
    hit_rate: float = 0.5,
    max_items_per_iter: int = 3,
    is_cached_calls_per_iter: int,
    n_iter: int = 100,
    seed: int = 0,
):
    cache_0_p0 = processor_cache_from_config(config_0, MULTIMODAL_REGISTRY)
    cache_0_p1 = engine_receiver_cache_from_config(config_0, MULTIMODAL_REGISTRY)
    cache_1_p0 = processor_cache_from_config(config_1, MULTIMODAL_REGISTRY)
    cache_1_p1 = engine_receiver_cache_from_config(config_1, MULTIMODAL_REGISTRY)

    cache_size_gb = max(
        config_0.model_config.multimodal_config.mm_processor_cache_gb,
        config_1.model_config.multimodal_config.mm_processor_cache_gb,
    )
    item_size_gb = int(cache_size_gb / item_capacity)

    rng = np.random.RandomState(seed)
    all_items = [
        _dummy_item("item", {"key": item_size_gb}, rng=rng)
        for _ in range(int(item_capacity / hit_rate))
    ]
    all_hashes = [
        MultiModalHasher.hash_kwargs(item=item.get_data()) for item in all_items
    ]

    # Should not be used since there is nothing to convert to text
    prompt_update = PromptInsertion("dummy", "target", "insertion")

    for it in range(n_iter):
        num_items_to_select = rng.randint(0, max_items_per_iter)
        item_idxs_to_select = rng.choice(len(all_items), num_items_to_select)

        selected_items = [all_items[idx] for idx in item_idxs_to_select]
        selected_hashes = [all_hashes[idx] for idx in item_idxs_to_select]

        if cache_0_p0 is None:
            cache_0_p0_out = selected_items
        else:
            for _ in range(is_cached_calls_per_iter):
                cache_0_p0.is_cached(selected_hashes)
            cache_0_p0_out = [
                item
                for item, _ in cache_0_p0.get_and_update(
                    [(item, prompt_update.content) for item in selected_items],
                    selected_hashes,
                )
            ]

        if cache_1_p0 is None:
            cache_1_p0_out = selected_items
        else:
            for _ in range(is_cached_calls_per_iter):
                cache_1_p0.is_cached(selected_hashes)
            cache_1_p0_out = [
                item
                for item, _ in cache_1_p0.get_and_update(
                    [(item, prompt_update.content) for item in selected_items],
                    selected_hashes,
                )
            ]

        if cache_0_p1 is None:
            cache_0_p1_out = cache_0_p0_out
        else:
            cache_0_p1_out = cache_0_p1.get_and_update(cache_0_p0_out, selected_hashes)

        if cache_1_p1 is None:
            cache_1_p1_out = cache_1_p0_out
        else:
            cache_1_p1_out = cache_1_p1.get_and_update(cache_1_p0_out, selected_hashes)

        assert cache_0_p1_out == cache_1_p1_out, f"Failed at {it=}"


@pytest.mark.parametrize("is_cached_calls_per_iter", [1, 2, 3])
def test_ipc_enable_disable_consistency(is_cached_calls_per_iter):
    cache_size_gb = 1 / (1 << 20)

    vllm_config_ipc_enabled = _create_vllm_config(
        mm_processor_cache_gb=cache_size_gb,
        enable_ipc=True,
    )
    vllm_config_ipc_disabled = _create_vllm_config(
        mm_processor_cache_gb=0,
        enable_ipc=False,
    )
    vllm_config_cache_disabled = _create_vllm_config(
        mm_processor_cache_gb=cache_size_gb,
        enable_ipc=True,
    )

    _compare_caches(
        vllm_config_ipc_enabled,
        vllm_config_ipc_disabled,
        is_cached_calls_per_iter=is_cached_calls_per_iter,
    )
    _compare_caches(
        vllm_config_ipc_disabled,
        vllm_config_cache_disabled,
        is_cached_calls_per_iter=is_cached_calls_per_iter,
    )
    _compare_caches(
        vllm_config_cache_disabled,
        vllm_config_ipc_enabled,
        is_cached_calls_per_iter=is_cached_calls_per_iter,
    )
