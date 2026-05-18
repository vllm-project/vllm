# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, VllmConfig
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
)
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.processing import PromptInsertion
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes

pytestmark = pytest.mark.cpu_test


def _dummy_elem(
    size: int,
    *,
    rng: np.random.RandomState | None = None,
):
    if rng is None:
        data = torch.empty((size,), dtype=torch.int8)
    else:
        data = torch.from_numpy(rng.randint(4, size=(size,), dtype=np.int8))

    return MultiModalFieldElem(
        data=data,
        field=MultiModalSharedField(batch_size=1),
    )


def _dummy_item(
    size_by_key: dict[str, int],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItem(
        {key: _dummy_elem(size, rng=rng) for key, size in size_by_key.items()}
    )


def _dummy_items(
    size_by_key_modality: dict[str, dict[str, int]],
    *,
    rng: np.random.RandomState | None = None,
):
    return MultiModalKwargsItems(
        {
            modality: [_dummy_item(size_by_key, rng=rng)]
            for modality, size_by_key in size_by_key_modality.items()
        }
    )


@pytest.mark.parametrize(
    ("item", "expected_size"),
    [
        (_dummy_item({"a1": 100}), 100),
        (_dummy_item({"a1": 100, "a2": 110}), 210),
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
    cache_0_p0 = MULTIMODAL_REGISTRY.processor_cache_from_config(config_0)
    cache_0_p1 = MULTIMODAL_REGISTRY.engine_receiver_cache_from_config(config_0)
    cache_1_p0 = MULTIMODAL_REGISTRY.processor_cache_from_config(config_1)
    cache_1_p1 = MULTIMODAL_REGISTRY.engine_receiver_cache_from_config(config_1)

    cache_size_gb = max(
        config_0.model_config.multimodal_config.mm_processor_cache_gb,
        config_1.model_config.multimodal_config.mm_processor_cache_gb,
    )
    item_size_gb = int(cache_size_gb / item_capacity)

    rng = np.random.RandomState(seed)
    all_items = [
        _dummy_item({"key": item_size_gb}, rng=rng)
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
        h: MultiModalKwargsItem.dummy(nbytes=2 * base_item_size)
        for h in request1_hashes
    }

    request2_hashes = ["image_D", "image_E", "image_A", "image_C"]
    request2_items = {
        h: MultiModalKwargsItem.dummy(nbytes=1 * base_item_size)
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
        h: MultiModalKwargsItem.dummy(5 * base_item_size) for h in request1_hashes
    }
    request1_items_p0_result = []

    request2_hashes = ["image_G", "image_A"]
    request2_items = {
        h: MultiModalKwargsItem.dummy(
            (5 if h in request1_hashes else 2) * base_item_size
        )
        for h in request2_hashes
    }
    request2_items_p0_result = []

    request3_hashes = ["image_G", "image_H", "image_I", "image_B"]
    request3_items = {
        h: MultiModalKwargsItem.dummy(
            (5 if h in request1_hashes else 2) * base_item_size
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
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_type="shm",
            mm_shm_cache_max_object_size_mb=6,
            mm_processor_cache_gb=15.2 * MiB_bytes / GiB_bytes,
        ),
    )
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())

    _run_test_cache_eviction_shm(sender_cache, receiver_cache, base_item_size=MiB_bytes)


def test_processor_cache_shared_across_loras():
    """Test that processor cache uses mm_hash to share data across LoRAs."""
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=1,
    )
    receiver_cache = MultiModalReceiverCache(model_config)

    base_mm_hash = "image_hash_abc123"
    lora_a_identifier = f"12345:{base_mm_hash}"
    lora_b_identifier = f"67890:{base_mm_hash}"

    item_data = MultiModalKwargsItem.dummy(1024)

    feature_lora_a = MultiModalFeatureSpec(
        data=item_data,
        modality="image",
        identifier=lora_a_identifier,
        mm_position=PlaceholderRange(offset=0, length=100),
        mm_hash=base_mm_hash,
    )

    receiver_cache.get_and_update_features([feature_lora_a])
    assert base_mm_hash in receiver_cache._cache

    feature_lora_b = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=lora_b_identifier,
        mm_position=PlaceholderRange(offset=0, length=100),
        mm_hash=base_mm_hash,
    )

    receiver_cache.get_and_update_features([feature_lora_b])
    assert feature_lora_b.data == item_data


# Regression for vllm-project/vllm#42995:
#
# /sleep?level>=1 and /pause?clear_cache=true route through EngineCore on
# the engine side, which only clears the P1 receiver cache. The P0 sender
# cache lives in the API server process; if it's not also cleared, the
# next request that reuses a previously-seen mm_hash forwards
# mm_item=None to a P1 whose cache is empty, tripping
# `assert mm_item is not None` in MultiModalReceiverCache.get_and_update_item.
#
# AsyncLLM.sleep / AsyncLLM.pause_generation / LLMEngine.sleep must clear
# the P0 renderer cache before delegating to EngineCore.


def test_sleep_dual_clear_invariant_at_cache_layer():
    """If only P1 is cleared, sender->receiver path with cached mm_hash
    asserts. Clearing both sides restores the fresh-request behaviour."""
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=4096 / GiB_bytes,
    )
    sender = MultiModalProcessorSenderCache(model_config)
    receiver = MultiModalReceiverCache(model_config)

    mm_hash = "image_X"
    item = MultiModalKwargsItem.dummy(nbytes=128)
    prompt_updates: list = []

    # First request: miss on both.
    sender_out = sender.get_and_update_item((item, prompt_updates), mm_hash)
    assert sender_out == (item, prompt_updates)
    assert receiver.get_and_update_item(item, mm_hash) is item

    # Simulate EngineCore.reset_mm_cache running without a P0 clear.
    receiver.clear_cache()
    # P0 still believes the hash is cached on P1.
    assert sender.is_cached_item(mm_hash)

    # Second request: sender strips the payload because it thinks P1 has it.
    sender_out2 = sender.get_and_update_item((item, prompt_updates), mm_hash)
    assert sender_out2[0] is None, "sender should report a hit"

    # P1 has no entry -> the assertion that #42995 trips fires.
    with pytest.raises(AssertionError, match="Expected a cached item"):
        receiver.get_and_update_item(None, mm_hash)

    # Now do the symmetric clear that the fixed sleep/pause path performs.
    sender.clear_cache()
    receiver.clear_cache()

    # Both caches behave like a fresh boot.
    assert not sender.is_cached_item(mm_hash)
    assert sender.get_and_update_item((item, prompt_updates), mm_hash) == (
        item,
        prompt_updates,
    )
    assert receiver.get_and_update_item(item, mm_hash) is item


@pytest.mark.asyncio
async def test_async_llm_sleep_clears_p0_renderer_when_level_ge_1():
    """AsyncLLM.sleep with level>=1 must clear the P0 renderer cache
    before the engine-core sleep call (#42995)."""
    from unittest.mock import AsyncMock

    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.__new__(AsyncLLM)
    order: list[str] = []
    renderer = AsyncMock()
    renderer.clear_mm_cache_async = AsyncMock(
        side_effect=lambda: order.append("renderer.clear_mm_cache_async")
    )
    engine_core = AsyncMock()
    engine_core.sleep_async = AsyncMock(
        side_effect=lambda *_args, **_kwargs: order.append("engine_core.sleep_async")
    )
    engine.renderer = renderer
    engine.engine_core = engine_core
    engine.logger_manager = None

    await engine.sleep(level=1)
    assert order == ["renderer.clear_mm_cache_async", "engine_core.sleep_async"]
    renderer.clear_mm_cache_async.assert_awaited_once()
    engine_core.sleep_async.assert_awaited_once_with(1, "abort")

    # Level 0 is pause-only; EngineCore.sleep does not clear the P1
    # mm cache for level 0, so neither should AsyncLLM.sleep clear P0.
    order.clear()
    renderer.clear_mm_cache_async.reset_mock()
    engine_core.sleep_async.reset_mock()
    await engine.sleep(level=0)
    renderer.clear_mm_cache_async.assert_not_awaited()
    engine_core.sleep_async.assert_awaited_once_with(0, "abort")


@pytest.mark.asyncio
async def test_async_llm_pause_generation_clears_p0_renderer_when_clearing_cache():
    """AsyncLLM.pause_generation(clear_cache=True) goes through
    pause_scheduler(clear_cache=True) -> _reset_caches on the engine side,
    so the P0 sender must be cleared too (#42995)."""
    from unittest.mock import AsyncMock

    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.__new__(AsyncLLM)
    order: list[str] = []
    renderer = AsyncMock()
    renderer.clear_mm_cache_async = AsyncMock(
        side_effect=lambda: order.append("renderer.clear_mm_cache_async")
    )
    engine_core = AsyncMock()
    engine_core.pause_scheduler_async = AsyncMock(
        side_effect=lambda **_kwargs: order.append("engine_core.pause_scheduler_async")
    )
    engine.renderer = renderer
    engine.engine_core = engine_core

    await engine.pause_generation(mode="abort", clear_cache=True)
    assert order == [
        "renderer.clear_mm_cache_async",
        "engine_core.pause_scheduler_async",
    ]
    renderer.clear_mm_cache_async.assert_awaited_once()

    # clear_cache=False keeps the P0 cache intact.
    renderer.clear_mm_cache_async.reset_mock()
    await engine.pause_generation(mode="abort", clear_cache=False)
    renderer.clear_mm_cache_async.assert_not_awaited()


def test_llm_engine_sleep_clears_p0_renderer_when_level_ge_1():
    """LLMEngine.sleep with level>=1 must clear the P0 renderer cache
    before the engine-core sleep call (#42995)."""
    from unittest.mock import MagicMock

    from vllm.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    order: list[str] = []
    renderer = MagicMock()
    renderer.clear_mm_cache.side_effect = lambda: order.append(
        "renderer.clear_mm_cache"
    )
    engine_core = MagicMock()
    engine_core.sleep.side_effect = lambda *_args, **_kwargs: order.append(
        "engine_core.sleep"
    )
    engine.renderer = renderer
    engine.engine_core = engine_core
    engine.logger_manager = None

    engine.sleep(level=1)
    assert order == ["renderer.clear_mm_cache", "engine_core.sleep"]
    renderer.clear_mm_cache.assert_called_once()
    engine_core.sleep.assert_called_once_with(1, "abort")

    renderer.clear_mm_cache.reset_mock()
    engine_core.sleep.reset_mock()
    engine.sleep(level=0)
    renderer.clear_mm_cache.assert_not_called()
    engine_core.sleep.assert_called_once_with(0, "abort")
