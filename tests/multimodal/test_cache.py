# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, RendererConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalCache,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheItem,
    MultiModalProcessorCacheItemMetadata,
    MultiModalProcessorSenderCache,
    MultiModalReceiverCache,
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
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
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes

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
        field=MultiModalSharedField(batch_size=1),
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

    cache[""] = item.get_data()
    assert cache.currsize == expected_size


def _create_vllm_config(
    *,
    mm_processor_cache_gb: float,
    enable_ipc: bool,
):
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=mm_processor_cache_gb,
    )

    return VllmConfig(
        model_config=model_config,
        renderer_config=RendererConfig(model_config=model_config),
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

    prompt_update = PromptInsertion("dummy", "target", "insertion").resolve(0)

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
                    [(item, [prompt_update]) for item in selected_items],
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
                    [(item, [prompt_update]) for item in selected_items],
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


def _run_test_cache_eviction_lru(
    p0_cache: BaseMultiModalProcessorCache,
    p1_cache: BaseMultiModalReceiverCache,
    base_item_size: int,
):
    request1_hashes = [
        "image_A",
        "image_B",
        "image_C",
    ]
    request1_items = {
        h: MultiModalKwargsItem.dummy(h, nbytes=2 * base_item_size)
        for h in request1_hashes
    }

    request2_hashes = ["image_D", "image_E", "image_A", "image_C"]
    request2_items = {
        h: MultiModalKwargsItem.dummy(h, nbytes=1 * base_item_size)
        for h in request2_hashes
    }

    ##########################
    # STEP 1: Request 1 send
    ##########################
    sender_is_cached_item_req1 = p0_cache.is_cached(request1_hashes)
    # Cache is empty
    assert sender_is_cached_item_req1 == [False, False, False]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request1_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 1 for P0 Cache
    ###########################
    item_tuple: MultiModalProcessorCacheInItem
    for i, h in enumerate(request1_hashes):
        # Use precomputed cache state
        is_cached = sender_is_cached_item_req1[i]
        item_tuple = (request1_items[h], []) if not is_cached else None
        print(f"Request 1: key={h} | cached={is_cached}")

        p0_cache.get_and_update_item(item_tuple, h)

    ###########################
    # Process request 1 for P1 Cache
    ###########################
    # Touch all mm hash for P1 Cache before process
    for mm_hash in request1_hashes:
        p1_cache.touch_receiver_cache_item(mm_hash)

    for h in request1_hashes:
        p1_cache.get_and_update_item(request1_items[h], h)

    expected_hashes = ["image_A", "image_B", "image_C"]
    assert list(p0_cache._cache.order) == expected_hashes

    ##########################
    # STEP 2: Request 2 send
    ##########################
    sender_is_cached_item_req2 = p0_cache.is_cached(request2_hashes)
    assert sender_is_cached_item_req2 == [False, False, True, True]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request2_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 2 for P0 Cache
    ###########################
    for i, h in enumerate(request2_hashes):
        # Use precomputed cache state again
        is_cached = sender_is_cached_item_req2[i]
        item_tuple = (request2_items[h], []) if not is_cached else None
        print(f"Request 2: key={h} | cached={is_cached}")

        p0_cache.get_and_update_item(item_tuple, h)

    ###########################
    # Process request 2 for P1 Cache
    ###########################

    # Touch all mm hash for P1 Cache before process
    for mm_hash in request2_hashes:
        p1_cache.touch_receiver_cache_item(mm_hash)

    for h in request2_hashes:
        p1_cache.get_and_update_item(request2_items[h], h)

    expected_hashes = ["image_D", "image_E", "image_A", "image_C"]
    assert list(p0_cache._cache.order) == expected_hashes


def test_cache_eviction_lru_cache():
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=6 / GiB_bytes,
    )
    sender_cache = MultiModalProcessorSenderCache(model_config)
    receiver_cache = MultiModalReceiverCache(model_config)

    _run_test_cache_eviction_lru(sender_cache, receiver_cache, base_item_size=1)


# This test verifies shared-memory cache eviction behavior across processor (p0)
# and receiver (p1) caches.
# Flow summary:
# 1. Request 1 adds images A, B, C — completely filling the cache.
# 2. Request 2 tries to add image_G and image_A, but image_G cannot be added because
#    cache is full and A is protected from eviction — cache remains unchanged.
# 3. Request 3 adds image_G, image_H, image_I and image_B
#    this time, image_A is evicted, freeing 5MB space
#    and image_G, image_H successfully fits,
#    image_B is protected from eviction then image_i cannot be added.
#    This proving normal eviction and reuse behavior.
def _run_test_cache_eviction_shm(
    p0_cache: BaseMultiModalProcessorCache,
    p1_cache: BaseMultiModalReceiverCache,
    base_item_size: int,
):
    request1_hashes = ["image_A", "image_B", "image_C"]
    request1_items = {
        h: MultiModalKwargsItem.dummy(h, nbytes=5 * base_item_size)
        for h in request1_hashes
    }
    request1_items_p0_result = []

    request2_hashes = ["image_G", "image_A"]
    request2_items = {
        h: MultiModalKwargsItem.dummy(
            h, nbytes=(5 if h in request1_hashes else 2) * base_item_size
        )
        for h in request2_hashes
    }
    request2_items_p0_result = []

    request3_hashes = ["image_G", "image_H", "image_I", "image_B"]
    request3_items = {
        h: MultiModalKwargsItem.dummy(
            h, nbytes=(5 if h in request1_hashes else 2) * base_item_size
        )
        for h in request3_hashes
    }
    request3_items_p0_result = []

    ##########################
    # STEP 1: Request 1 send
    # This will fill up the cache
    ##########################
    sender_is_cached_item_req1 = p0_cache.is_cached(request1_hashes)
    # Cache is empty
    assert sender_is_cached_item_req1 == [False, False, False]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request1_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 1 for P0 Cache
    ###########################
    item_tuple: MultiModalProcessorCacheInItem
    for i, h in enumerate(request1_hashes):
        # Use precomputed cache state
        is_cached = sender_is_cached_item_req1[i]
        item_tuple = (request1_items[h], []) if not is_cached else None
        print(f"Request 1: key={h} | cached={is_cached}")

        p0_result = p0_cache.get_and_update_item(item_tuple, h)
        # Only get mm item, ignore prompt update result
        request1_items_p0_result.append(p0_result[0])

    ###########################
    # Process request 1 for P1 Cache
    ###########################
    # Touch all mm hash for P1 Cache before process
    for mm_hash, mm_item in zip(request1_hashes, request1_items_p0_result):
        p1_cache.touch_receiver_cache_item(mm_hash, mm_item)

    for mm_hash, mm_item in zip(request1_hashes, request1_items_p0_result):
        p1_cache.get_and_update_item(mm_item, mm_hash)

    expected_hashes = ["image_A", "image_B", "image_C"]
    assert list(p0_cache._shm_cache.key_index.keys()) == expected_hashes

    ##########################
    # STEP 2: Request 2 send
    # There is no eviction because image_A is protected
    # No new item can add to cache
    ##########################
    sender_is_cached_item_req2 = p0_cache.is_cached(request2_hashes)
    assert sender_is_cached_item_req2 == [False, True]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request2_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 2 for P0 Cache
    ###########################
    for i, h in enumerate(request2_hashes):
        # Use precomputed cache state again
        is_cached = sender_is_cached_item_req2[i]
        item_tuple = (request2_items[h], []) if not is_cached else None
        print(f"Request 2: key={h} | cached={is_cached}")

        p0_result = p0_cache.get_and_update_item(item_tuple, h)
        # Only get mm item, ignore prompt update result
        request2_items_p0_result.append(p0_result[0])

    # image_A cannot be evict then
    # image_G will fail to allocate anyway and image_A still in cache
    assert p0_cache.is_cached(request2_hashes) == [False, True]

    ###########################
    # Process request 2 for P1 Cache
    ###########################

    # Touch all mm hash for P1 Cache before process
    for mm_hash, mm_item in zip(request2_hashes, request2_items_p0_result):
        p1_cache.touch_receiver_cache_item(mm_hash, mm_item)

    for mm_hash, mm_item in zip(request2_hashes, request2_items_p0_result):
        p1_cache.get_and_update_item(mm_item, mm_hash)

    # Prove that cache state is unchanged
    expected_hashes = ["image_A", "image_B", "image_C"]
    assert list(p0_cache._shm_cache.key_index.keys()) == expected_hashes

    ##########################
    # STEP 3: Request 3 send
    ##########################
    ##### Prove that cache eviction work normally
    sender_is_cached_item_req3 = p0_cache.is_cached(request3_hashes)
    assert sender_is_cached_item_req3 == [False, False, False, True]

    # Touch all mm hash for P0 Cache before process
    for mm_hash in request3_hashes:
        p0_cache.touch_sender_cache_item(mm_hash)

    ###########################
    # Process request 3 for P0 Cache
    ###########################
    for i, h in enumerate(request3_hashes):
        # Use precomputed cache state again
        is_cached = sender_is_cached_item_req3[i]
        item_tuple = (request3_items[h], []) if not is_cached else None
        print(f"Request 3: key={h} | cached={is_cached}")
        p0_result = p0_cache.get_and_update_item(item_tuple, h)
        # Only get mm item, ignore prompt update result
        request3_items_p0_result.append(p0_result[0])

    # image_A got evict and image_G add to cache
    # image_B is still protected
    # image_G, image_H fit but image_I cannot fit
    assert p0_cache.is_cached(request3_hashes) == [True, True, False, True]

    ###########################
    # Process request 3 for P1 Cache
    ###########################

    # Touch all mm hash for P1 Cache before process
    for mm_hash, mm_item in zip(request3_hashes, request3_items_p0_result):
        p1_cache.touch_receiver_cache_item(mm_hash, mm_item)

    for mm_hash, mm_item in zip(request3_hashes, request3_items_p0_result):
        p1_cache.get_and_update_item(mm_item, mm_hash)

    expected_hashes = ["image_B", "image_C", "image_G", "image_H"]
    assert list(p0_cache._shm_cache.key_index.keys()) == expected_hashes


def test_cache_eviction_shm_cache():
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_type="shm",
        mm_shm_cache_max_object_size_mb=6,
        mm_processor_cache_gb=15.2 * MiB_bytes / GiB_bytes,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        renderer_config=RendererConfig(model_config=model_config),
    )
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())

    _run_test_cache_eviction_shm(sender_cache, receiver_cache, base_item_size=MiB_bytes)
