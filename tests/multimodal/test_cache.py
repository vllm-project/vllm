# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import multiprocessing as mp
import threading
from contextlib import suppress
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm import PoolingParams
from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.cache import (
    BaseMultiModalReceiverCache,
    LruKeyReplicatedReceiverCache,
    LruKeyReplicatedSenderCache,
    MultiModalCacheMissError,
    MultiModalProcessorOnlyCache,
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
    engine_receiver_cache_from_config,
    processor_cache_from_config,
)
from vllm.multimodal.cache.base import (
    MultiModalCache,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheItem,
    MultiModalProcessorCacheItemMetadata,
)
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.processing import PromptInsertion
from vllm.outputs import PoolingRequestOutput
from vllm.renderers import renderer_from_config
from vllm.sampling_params import RequestOutputKind
from vllm.utils.async_utils import make_async
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from ..models.utils import build_model_context
from ..utils import create_new_process_for_each_test

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

    prompt_update = PromptInsertion("dummy", [0], [1]).resolve(0)

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
    cache_0_p0 = processor_cache_from_config(config_0)
    cache_0_p1 = engine_receiver_cache_from_config(config_0)
    cache_1_p0 = processor_cache_from_config(config_1)
    cache_1_p1 = engine_receiver_cache_from_config(config_1)

    assert config_0.model_config.multimodal_config is not None
    assert config_1.model_config.multimodal_config is not None
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
        MultiModalHasher.hash_kwargs("blake3", item=item.get_data())
        for item in all_items
    ]

    prompt_update = PromptInsertion("dummy", [0], [1]).resolve(0)

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


class _StubModelConfig:
    """Minimal model-config stand-in: the cache classes only call
    get_multimodal_config(), so this lets us construct them directly without a
    real model, keeping this unit test free of any Hugging Face / network lookup.
    """

    def __init__(self, mm_processor_cache_gb: float) -> None:
        self._mm_config = MultiModalConfig(mm_processor_cache_gb=mm_processor_cache_gb)

    def get_multimodal_config(self) -> MultiModalConfig:
        return self._mm_config


@pytest.mark.skip_global_cleanup
def test_oversized_item_is_served_uncached():
    """Items larger than the processor cache must not crash insert.

    cachetools.LRUCache raises ValueError("value too large") when a single
    item exceeds maxsize. Engine profiling uses a max-size dummy item, so this
    used to abort EngineCore startup (vllm-project/vllm#52835). Skip the
    insert and serve the item uncached instead.
    """
    # 1 KiB cache; a 4 KiB item cannot fit even if the cache is empty.
    model_config = _StubModelConfig(mm_processor_cache_gb=1024 / GiB_bytes)
    item = MultiModalKwargsItem.dummy(nbytes=4096)
    small = MultiModalKwargsItem.dummy(nbytes=64)

    p0_only = MultiModalProcessorOnlyCache(model_config)
    assert p0_only.get_and_update_item((item, []), "big")[0] is item
    assert not p0_only.is_cached_item("big")
    assert p0_only.get_and_update_item((small, []), "small")[0] is small
    assert p0_only.is_cached_item("small")

    p0 = LruKeyReplicatedSenderCache(model_config)
    p1 = LruKeyReplicatedReceiverCache(model_config)
    assert item is not None
    assert p0.get_and_update_item((item, []), "big")[0] is item
    assert not p0.is_cached_item("big")
    assert p1.get_and_update_item(item, "big") is item
    assert "big" not in p1._cache


def test_mm_cache_miss_raises_and_recovers():
    """A P0/P1 multimodal cache drift must be recoverable, not a hard crash.

    Reproduces the production failure: the P0 sender cache believes an item is
    cached on P1 (so it sends ``data=None``), but the P1 receiver cache does not
    have it. The receiver must raise ``MultiModalCacheMissError`` (carrying the
    hash) instead of asserting; ``P0.invalidate()`` must drop the stale shadow
    entry; and resending the item with data must repopulate both caches.

    Caches are built directly (not via the registry) so the test needs no real
    model or network.
    """
    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    p0 = LruKeyReplicatedSenderCache(model_config)
    p1 = LruKeyReplicatedReceiverCache(model_config)

    mm_hash = "image_A"
    item = MultiModalKwargsItem.dummy(nbytes=64)

    # Drift: the item reaches P1 with no data (P0 would have sent data=None on a
    # stale HIT) but P1 does not have it. Must raise a typed, retryable error
    # carrying the hash -- not assert.
    with pytest.raises(MultiModalCacheMissError) as exc_info:
        p1.get_and_update_item(None, mm_hash)
    assert exc_info.value.mm_hashes == [mm_hash]

    # Give P0 a (now stale) shadow entry for the hash.
    miss = p0.get_and_update_item((item, []), mm_hash)
    assert miss[0] is item  # MISS path returns the data to forward to P1
    assert p0.is_cached_item(mm_hash)
    # On the next request P0 short-circuits to data=None -- the drift bug when
    # P1 lacks the item.
    assert item is not None
    hit = p0.get_and_update_item((item, []), mm_hash)
    assert hit[0] is None

    # Recovery: drop the stale P0 entry so the client's resend-with-data takes
    # the MISS path again and repopulates P1.
    p0.invalidate(mm_hash)
    assert not p0.is_cached_item(mm_hash)

    assert item is not None
    resent = p0.get_and_update_item((item, []), mm_hash)
    assert resent[0] is item  # MISS again -> data is resent
    assert p1.get_and_update_item(item, mm_hash) == item  # P1 now caches it
    # A subsequent uuid-only request now succeeds on P1.
    assert p1.get_and_update_item(None, mm_hash) == item


def test_receiver_cache_replaces_stale_item_when_payload_resent():
    """A P0 miss can resend a different item under the same identity.

    Independent LRU eviction leaves P1 holding the old tensor. Substituting it
    for the new payload pairs the request's placeholders with the wrong item
    and kills EngineCore. Prefer the fresh payload and keep it for later hits.
    """
    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    p1 = LruKeyReplicatedReceiverCache(model_config)
    small = MultiModalKwargsItem.dummy(nbytes=64)
    large = MultiModalKwargsItem.dummy(nbytes=256)
    mm_hash = "shared-id"

    assert p1.get_and_update_item(small, mm_hash) is small
    assert p1.get_and_update_item(large, mm_hash) is large
    assert p1.get_and_update_item(None, mm_hash) is large


def test_receiver_cache_features_keep_resent_payload():
    """EngineCore updates features through get_and_update_features."""
    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    p1 = LruKeyReplicatedReceiverCache(model_config)
    small = MultiModalKwargsItem.dummy(nbytes=64)
    large = MultiModalKwargsItem.dummy(nbytes=256)
    mm_hash = "shared-id"

    def _feature(
        data: MultiModalKwargsItem | None, length: int
    ) -> MultiModalFeatureSpec:
        return MultiModalFeatureSpec(
            data=data,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=length),
            mm_hash=mm_hash,
        )

    seeded = p1.get_and_update_features([_feature(small, length=4)])
    assert seeded[0].data is small

    updated = p1.get_and_update_features([_feature(large, length=2048)])
    assert updated[0].data is large
    assert updated[0].mm_position.length == 2048


def test_mm_cache_miss_batches_all_drifted_hashes():
    """All hashes drifted within one request must surface in a single error.

    Otherwise a request with k drifted items needs k client retries (each resend
    un-shadows only the one reported hash). get_and_update_features collects every
    miss and raises once, so a single retry recovers the whole request -- while the
    non-drifted items in the same request are still ingested.
    """
    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    p1 = LruKeyReplicatedReceiverCache(model_config)
    item = MultiModalKwargsItem.dummy(nbytes=64)

    def _feature(
        mm_hash: str, data: MultiModalKwargsItem | None
    ) -> MultiModalFeatureSpec:
        return MultiModalFeatureSpec(
            data=data,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=1),
            mm_hash=mm_hash,
        )

    # A and C carry data; B, D, E arrive data=None (stale P0 shadow HITs) but P1
    # has none of them -- all three must be reported together, in order.
    features = [
        _feature("A", item),
        _feature("B", None),
        _feature("C", item),
        _feature("D", None),
        _feature("E", None),
    ]
    with pytest.raises(MultiModalCacheMissError) as exc_info:
        p1.get_and_update_features(features)
    assert exc_info.value.mm_hashes == ["B", "D", "E"]
    # The non-drifted items were still ingested into P1 before the raise.
    assert "A" in p1._cache and "C" in p1._cache


def test_shm_receiver_handles_prefix_covered_items(monkeypatch):
    """Regression test for the EngineCore crash with prefix caching + SHM cache.

    On a repeated request, prefix caching can fully cover an item's
    placeholder range, so strip_covered_mm_data() drops its payload: uncached
    items arrive at the worker with data=None, while SHM-cached items keep
    their address descriptor. Before the fix the SHM receiver asserted on the
    None item, terminating EngineCore on the second identical request
    (vllm-project/vllm#54994). The address item must still be resolved and
    acknowledged so the object stays reclaimable.
    """
    monkeypatch.setenv(
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", "test_shm_prefix_covered_items"
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1),
        model_config=_StubModelConfig(mm_processor_cache_gb=4 * MiB_bytes / GiB_bytes),
    )
    p0 = ShmObjectStoreSenderCache(vllm_config)
    p1 = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())

    def _feature(
        mm_hash: str, data: MultiModalKwargsItem | None
    ) -> MultiModalFeatureSpec:
        return MultiModalFeatureSpec(
            data=data,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=100),
            mm_hash=mm_hash,
        )

    mm_hash = "image_A"
    item = MultiModalKwargsItem.dummy(nbytes=1024)

    try:
        # Request 1 (sender miss): the payload is put into SHM; the worker
        # receives an address descriptor and resolves it.
        p0.touch_sender_cache_item(mm_hash)
        address_item, _ = p0.get_and_update_item((item, []), mm_hash)
        first = _feature(mm_hash, address_item)
        p1.get_and_update_features([first])
        assert first.data is not None
        assert torch.equal(first.data["dummy"].data, item["dummy"].data)

        # Request 2 (identical, fully prefix-covered): the sender hit takes
        # writer references (touch + get_cached) and returns the address
        # descriptor; the scheduler strips the covered uncached payload to
        # data=None.
        p0.touch_sender_cache_item(mm_hash)
        address_item_2, _ = p0.get_and_update_item(None, mm_hash)
        covered_uncached = _feature("image_B", None)
        covered_cached = _feature(mm_hash, address_item_2)

        # Pre-fix this asserted in touch_receiver_cache_item and terminated
        # EngineCore.
        p1.get_and_update_features([covered_uncached, covered_cached])

        assert covered_uncached.data is None
        # The address item is resolved to the cached payload.
        assert covered_cached.data is not None
        assert torch.equal(covered_cached.data["dummy"].data, item["dummy"].data)

        # The hit's writer references were acknowledged by the worker, so the
        # object remains reclaimable instead of permanently protected.
        p0._shm_cache.free_unused()
        assert not p0._shm_cache.is_cached(mm_hash)
    finally:
        p1._shm_cache.close()
        p0.close()


def _run_test_cache_eviction_lru(
    p0_cache: LruKeyReplicatedSenderCache,
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
    sender_cache = LruKeyReplicatedSenderCache(model_config)
    receiver_cache = LruKeyReplicatedReceiverCache(model_config)

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
    p0_cache: ShmObjectStoreSenderCache,
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


def test_shm_receiver_requires_sender_handle_for_current_cache_key():
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_type="shm",
            mm_shm_cache_max_object_size_mb=1,
            mm_processor_cache_gb=2 * MiB_bytes / GiB_bytes,
        ),
    )
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())
    mm_hash = "image_A"
    item = MultiModalKwargsItem.dummy(1024)

    try:
        sender_item, _ = sender_cache.get_and_update_item((item, []), mm_hash)
        assert sender_item is not None

        valid_feature = MultiModalFeatureSpec(
            data=sender_item,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=1),
            mm_hash=mm_hash,
        )
        receiver_cache.get_and_update_features([valid_feature])
        assert valid_feature.data == item

        forged_item = MultiModalKwargsItem(
            {
                "address": sender_item["address"],
                "monotonic_id": sender_item["monotonic_id"],
            }
        )
        forged_feature = MultiModalFeatureSpec(
            data=forged_item,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=1),
            mm_hash=mm_hash,
        )
        with pytest.raises(ValueError, match="SHM handle signature"):
            receiver_cache.get_and_update_features([forged_feature])

        wrong_key = "image_B"
        wrong_key_feature = MultiModalFeatureSpec(
            data=sender_item,
            modality="image",
            identifier=wrong_key,
            mm_position=PlaceholderRange(offset=0, length=1),
            mm_hash=wrong_key,
        )
        with pytest.raises(ValueError, match="SHM handle signature"):
            receiver_cache.get_and_update_features([wrong_key_feature])
    finally:
        receiver_cache._shm_cache.close()
        sender_cache.close()


def test_shm_sender_handle_survives_engine_msgpack_round_trip():
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_type="shm",
            mm_shm_cache_max_object_size_mb=1,
            mm_processor_cache_gb=2 * MiB_bytes / GiB_bytes,
        ),
    )
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())
    mm_hash = "image_A"
    item = MultiModalKwargsItem.dummy(1024)

    try:
        sender_item, _ = sender_cache.get_and_update_item((item, []), mm_hash)
        encoded = MsgpackEncoder().encode(sender_item)
        decoded = MsgpackDecoder(MultiModalKwargsItem).decode(encoded)
        feature = MultiModalFeatureSpec(
            data=decoded,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=1),
            mm_hash=mm_hash,
        )

        receiver_cache.get_and_update_features([feature])

        assert feature.data == item
    finally:
        receiver_cache._shm_cache.close()
        sender_cache.close()


def _small_shm_vllm_config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            mm_processor_cache_type="shm",
            mm_shm_cache_max_object_size_mb=1,
            mm_processor_cache_gb=2 * MiB_bytes / GiB_bytes,
        ),
    )


def _with_elem(item: MultiModalKwargsItem, key: str, data) -> MultiModalKwargsItem:
    elems = dict(item)
    elems[key] = MultiModalFieldElem(data=data, field=MultiModalBatchedField())
    return MultiModalKwargsItem(elems)


def test_shm_sender_validates_client_supplied_handles():
    sender_cache = ShmObjectStoreSenderCache(_small_shm_vllm_config())
    mm_hash = "image_A"

    try:
        handle, _ = sender_cache.get_and_update_item(
            (MultiModalKwargsItem.dummy(1024), []), mm_hash
        )
        assert handle is not None

        sender_cache.validate_input_item(handle, mm_hash)
        sender_cache.validate_input_item(MultiModalKwargsItem.dummy(16), mm_hash)

        signature = cast(list[int], handle["signature"].data)
        forged = [
            _with_elem(handle, "signature", [signature[0] ^ 1, *signature[1:]]),
            _with_elem(handle, "signature", [signature]),
            _with_elem(handle, "address", True),
            _with_elem(handle, "address", -1),
            _with_elem(handle, "monotonic_id", 2**32),
            MultiModalKwargsItem(
                {"address": handle["address"], "monotonic_id": handle["monotonic_id"]}
            ),
        ]
        for item in forged:
            with pytest.raises(ValueError, match="SHM handle signature"):
                sender_cache.validate_input_item(item, mm_hash)
        with pytest.raises(ValueError, match="SHM handle signature"):
            sender_cache.validate_input_item(handle, "image_B")

        sender_cache.clear_cache()
        with pytest.raises(ValueError, match="SHM handle signature"):
            sender_cache.validate_input_item(handle, mm_hash)
    finally:
        sender_cache.close()


def test_shm_receiver_rejects_out_of_range_handle_fields():
    vllm_config = _small_shm_vllm_config()
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())
    mm_hash = "image_A"

    try:
        handle, _ = sender_cache.get_and_update_item(
            (MultiModalKwargsItem.dummy(1024), []), mm_hash
        )
        assert handle is not None

        for key, value in [
            ("address", -1),
            ("address", 2**64),
            ("monotonic_id", -1),
            ("monotonic_id", 2**32),
        ]:
            with pytest.raises(ValueError, match="SHM handle signature"):
                receiver_cache.get_and_update_item(
                    _with_elem(handle, key, value), mm_hash
                )
    finally:
        receiver_cache._shm_cache.close()
        sender_cache.close()


def test_shm_handle_issued_before_cache_clear():
    vllm_config = _small_shm_vllm_config()
    sender_cache = ShmObjectStoreSenderCache(vllm_config)
    receiver_cache = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())
    mm_hash = "image_A"
    item = MultiModalKwargsItem.dummy(1024)

    def feature(data: MultiModalKwargsItem) -> MultiModalFeatureSpec:
        return MultiModalFeatureSpec(
            data=data,
            modality="image",
            identifier=mm_hash,
            mm_position=PlaceholderRange(offset=0, length=1),
            mm_hash=mm_hash,
        )

    try:
        stale_item, _ = sender_cache.get_and_update_item((item, []), mm_hash)
        assert stale_item is not None

        sender_cache.clear_cache()

        # A request still draining after the clear resolves to its own data.
        draining = feature(stale_item)
        receiver_cache.get_and_update_features([draining])
        assert draining.data == item

        sender_cache.get_and_update_item(
            (MultiModalKwargsItem.dummy(2048), []),
            mm_hash,
        )

        with pytest.raises(ValueError, match="modified or is invalid"):
            receiver_cache.get_and_update_features([feature(stale_item)])
    finally:
        receiver_cache._shm_cache.close()
        sender_cache.close()


def test_processor_cache_shared_across_loras():
    """Test that processor cache uses mm_hash to share data across LoRAs."""
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=1,
    )
    receiver_cache = LruKeyReplicatedReceiverCache(model_config)

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


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "release_error",
    [
        None,
        "requires a completed pause first",
        "requires all executor memory to be resident",
    ],
    ids=["released", "not-paused", "nonresident-memory"],
)
@pytest.mark.asyncio
async def test_release_kv_cache_resends_mm_payload(use_async, release_error):
    """Release must not leave a sender hit pointing at a cleared receiver."""
    ctx = build_model_context(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=1,
    )
    model_config = ctx.model_config

    sender = LruKeyReplicatedSenderCache(model_config)
    receiver = LruKeyReplicatedReceiverCache(model_config)
    item = _dummy_item({"pixel_values": 16})
    mm_hash = "image_A"
    payload, _ = sender.get_and_update_item((item, []), mm_hash)
    assert receiver.get_and_update_item(payload, mm_hash) is item
    assert sender.get_and_update_item(None, mm_hash)[0] is None

    renderer = renderer_from_config(VllmConfig(model_config=model_config))
    renderer._mm_processor_cache = sender

    def release():
        if release_error:
            raise RuntimeError(release_error)
        receiver.clear_cache()

    engine = SimpleNamespace(
        renderer=renderer,
        engine_core=SimpleNamespace(
            release_kv_cache_memory=release,
            release_kv_cache_memory_async=make_async(
                release,
                executor=renderer._executor,
            ),
        ),
        logger_manager=Mock(),
    )

    async def call_release():
        if use_async:
            await AsyncLLM.release_kv_cache_memory(engine)
        else:
            LLMEngine.release_kv_cache_memory(engine)

    if release_error:
        with pytest.raises(RuntimeError, match=release_error):
            await call_release()
        engine.logger_manager.record_sleep_state.assert_not_called()
    else:
        await call_release()
        engine.logger_manager.record_sleep_state.assert_called_once_with(1, 0)

    payload, _ = sender.get_and_update_item((item, []), mm_hash)
    assert payload is item
    assert receiver.get_and_update_item(payload, mm_hash) is item


@pytest.fixture
def mm_cache_renderer():
    model_config = ModelConfig(
        model="hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration",
        max_model_len=2048,
        mm_processor_cache_gb=1,
    )
    renderer = renderer_from_config(VllmConfig(model_config=model_config))
    prompt = {
        "prompt": "<|vision_start|><|image_pad|><|vision_end|>Describe.",
        "multi_modal_data": {"image": Image.new("RGB", (56, 56))},
    }
    try:
        yield renderer, prompt
    finally:
        renderer.shutdown()


@pytest.mark.asyncio
async def test_cache_drift_invalidation_waits_for_render_without_delaying_outputs(
    monkeypatch,
    mm_cache_renderer,
):
    """Apply drift invalidation between renders, including already queued work."""
    renderer, prompt = mm_cache_renderer
    primed = (await renderer.render_cmpl_async([prompt]))[0]
    mm_hash = primed["mm_hashes"]["image"][0]
    processor = renderer.get_mm_processor()
    get_missing = processor._get_cache_missing_items
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def hold_cache_hit(*args, **kwargs):
        result = get_missing(*args, **kwargs)
        assert result[0] == {"image": [True]}
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10)
        return result

    monkeypatch.setattr(processor, "_get_cache_missing_items", hold_cache_hit)
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    collectors = []
    for idx in range(2):
        request = EngineCoreRequest(
            request_id=str(idx),
            external_req_id=str(idx),
            prompt_token_ids=[1],
            mm_features=None,
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, str(idx))
        output_processor.add_request(request, prompt=None, queue=collector)
        collectors.append(collector)
    outputs: asyncio.Queue[EngineCoreOutputs] = asyncio.Queue()
    engine = SimpleNamespace(
        output_handler=None,
        engine_core=SimpleNamespace(get_output_async=outputs.get),
        output_processor=output_processor,
        log_stats=False,
        logger_manager=None,
        renderer=renderer,
    )
    AsyncLLM._run_output_handler(engine)
    render = asyncio.create_task(renderer.render_cmpl_async([prompt]))
    queued_render = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        # This render is queued before the output handler learns about the miss.
        queued_render = renderer._process_multimodal_async(
            renderer.get_tokenizer().encode(prompt["prompt"]),
            prompt["multi_modal_data"],
            mm_uuids=None,
            mm_processor_kwargs=None,
        )
        outputs.put_nowait(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="0",
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                        mm_cache_miss_hashes=[mm_hash],
                    )
                ]
            )
        )
        failed = await asyncio.wait_for(collectors[0].get(), timeout=5)
        assert isinstance(failed, PoolingRequestOutput)
        assert failed.error is not None
        assert failed.error.code == "multimodal_cache_miss"
        outputs.put_nowait(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="1",
                        new_token_ids=[],
                        finish_reason=FinishReason.STOP,
                        pooling_output=torch.tensor([1.0]),
                    )
                ]
            )
        )
        succeeded = await asyncio.wait_for(collectors[1].get(), timeout=5)
        assert isinstance(succeeded, PoolingRequestOutput)
        assert succeeded.error is None
        monkeypatch.setattr(processor, "_get_cache_missing_items", get_missing)
        release.set()
        assert (await render)[0]["mm_kwargs"]["image"][0] is None
        resent = await queued_render
        assert resent["mm_kwargs"]["image"][0] is not None
    finally:
        release.set()
        await asyncio.gather(render, return_exceptions=True)
        if queued_render is not None:
            await asyncio.gather(queued_render, return_exceptions=True)
        engine.output_handler.cancel()
        with suppress(asyncio.CancelledError):
            await engine.output_handler


@pytest.mark.asyncio
async def test_cache_miss_notifications_ignore_absent_and_coalesce_live_hashes(
    mm_cache_renderer,
):
    renderer, prompt = mm_cache_renderer
    renderer.queue_mm_cache_invalidations([f"absent-{i}" for i in range(1000)])
    assert not renderer._pending_mm_cache_invalidations

    primed = (await renderer.render_cmpl_async([prompt]))[0]
    mm_hash = primed["mm_hashes"]["image"][0]
    for _ in range(100):
        renderer.queue_mm_cache_invalidations([mm_hash, mm_hash])
    assert len(renderer._pending_mm_cache_invalidations) == 1
    resent = (await renderer.render_cmpl_async([prompt]))[0]
    assert resent["mm_kwargs"]["image"][0] is not None
    assert not renderer._pending_mm_cache_invalidations

    renderer.queue_mm_cache_invalidations([mm_hash])
    await renderer.clear_mm_cache_async()
    resent = (await renderer.render_cmpl_async([prompt]))[0]
    assert resent["mm_kwargs"]["image"][0] is not None
    renderer.queue_mm_cache_invalidations([mm_hash])
    resent = (await renderer.render_cmpl_async([prompt]))[0]
    assert resent["mm_kwargs"]["image"][0] is not None


@pytest.mark.asyncio
async def test_failed_cache_invalidation_preserves_pending_and_concurrent_misses(
    monkeypatch,
    mm_cache_renderer,
):
    renderer, _ = mm_cache_renderer
    cache = renderer.mm_processor_cache
    hashes = ["first", "second", "concurrent"]
    for mm_hash in hashes:
        cache.get_and_update_item((_dummy_item({"x": 8}), []), mm_hash)
    renderer.queue_mm_cache_invalidations(hashes[:2])
    entered = threading.Event()
    release = threading.Event()

    def failing_invalidate(mm_hash):
        entered.set()
        assert release.wait(timeout=10)
        raise RuntimeError("invalidate failed")

    with monkeypatch.context() as patch:
        patch.setattr(cache, "invalidate", failing_invalidate)
        drain = renderer._mm_executor.submit(renderer._drain_mm_cache_invalidations)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            # Enqueue must not wait for cache invalidation to finish.
            await asyncio.wait_for(
                asyncio.to_thread(renderer.queue_mm_cache_invalidations, hashes[2:]),
                timeout=5,
            )
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="invalidate failed"):
                await asyncio.wrap_future(drain)

    renderer._drain_mm_cache_invalidations()
    assert all(not cache.is_cached_item(mm_hash) for mm_hash in hashes)


def test_invalidate_evicts_across_processor_receiver_and_shm_caches(monkeypatch):
    """All multimodal processor/receiver caches must support targeted eviction
    when a poisoned entry is detected (#55546), including DP
    MultiModalProcessorOnlyCache and SHM ring-buffer re-insertion."""
    model_config = _StubModelConfig(mm_processor_cache_gb=4 * MiB_bytes / GiB_bytes)
    item_a = MultiModalKwargsItem.dummy(nbytes=512)
    item_b = MultiModalKwargsItem.dummy(nbytes=512)

    proc_only = MultiModalProcessorOnlyCache(model_config)
    proc_only.get_and_update_item((item_a, []), "bad")
    proc_only.get_and_update_item((item_b, []), "good")
    assert proc_only.is_cached_item("bad")
    proc_only.invalidate("bad")
    assert not proc_only.is_cached_item("bad")
    assert proc_only.is_cached_item("good")

    recv_lru = LruKeyReplicatedReceiverCache(model_config)
    recv_lru.get_and_update_item(item_a, "bad")
    recv_lru.get_and_update_item(item_b, "good")
    from vllm.v1.engine.core import EngineCore

    EngineCore._invalidate_mm_receiver_cache(
        SimpleNamespace(mm_receiver_cache=recv_lru),
        {
            0: EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="0",
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                        mm_cache_miss_hashes=["bad"],
                    )
                ]
            )
        },
    )
    with pytest.raises(MultiModalCacheMissError):
        recv_lru.get_and_update_item(None, "bad")
    assert recv_lru.get_and_update_item(None, "good") is item_b

    monkeypatch.setenv(
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", "test_shm_targeted_invalidate"
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1),
        model_config=model_config,
    )
    p0 = ShmObjectStoreSenderCache(vllm_config)
    p1 = ShmObjectStoreReceiverCache(vllm_config, mp.Lock())
    try:
        p0.touch_sender_cache_item("bad")
        addr1, _ = p0.get_and_update_item((item_a, []), "bad")
        feat1 = MultiModalFeatureSpec(
            data=addr1,
            modality="image",
            identifier="bad",
            mm_position=PlaceholderRange(offset=0, length=4),
            mm_hash="bad",
        )
        p1.get_and_update_features([feat1])
        assert p0.is_cached_item("bad")

        p0.invalidate("bad")
        assert not p0.is_cached_item("bad")

        # Re-inserting the same key after invalidation must succeed cleanly,
        # and freeing the old ring-buffer slot must not delete the new entry.
        p0.touch_sender_cache_item("bad")
        addr2, _ = p0.get_and_update_item((item_b, []), "bad")
        assert p0.is_cached_item("bad")
        p0._shm_cache.free_unused()
        assert p0.is_cached_item("bad")
        feat2 = MultiModalFeatureSpec(
            data=addr2,
            modality="image",
            identifier="bad",
            mm_position=PlaceholderRange(offset=0, length=4),
            mm_hash="bad",
        )
        p1.get_and_update_features([feat2])
    finally:
        p1._shm_cache.close()
        p0.close()


@pytest.mark.asyncio
async def test_mismatch_validation_error_invalidates_dp_processor_cache():
    """In DP internal-LB mode (MultiModalProcessorOnlyCache), a worker mismatch
    must invalidate the cached entry AND surface VLLMValidationError (HTTP 400)
    on only the bad request without overwriting error.code with cache_miss."""
    from vllm.exceptions import VLLMValidationError
    from vllm.outputs import RequestError

    model_config = ModelConfig(
        model="hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration",
        max_model_len=2048,
        mm_processor_cache_gb=1,
        mm_processor_cache_type="lru",
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(data_parallel_size=2),
    )
    renderer = renderer_from_config(vllm_config)
    assert isinstance(renderer.mm_processor_cache, MultiModalProcessorOnlyCache)

    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    collectors = []
    for idx in range(2):
        req = EngineCoreRequest(
            request_id=str(idx),
            external_req_id=str(idx),
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, str(idx))
        output_processor.add_request(req, prompt=None, queue=collector)
        collectors.append(collector)

    bad_hash = "poisoned-dp-hash"
    renderer.mm_processor_cache.get_and_update_item(
        (_dummy_item({"x": 16}), []), bad_hash
    )
    assert renderer.mm_processor_cache.is_cached_item(bad_hash)

    mismatch_msg = "Attempted to assign 138 = 138 multimodal tokens to 638 placeholders"
    outputs: asyncio.Queue[EngineCoreOutputs] = asyncio.Queue()
    engine = SimpleNamespace(
        output_handler=None,
        engine_core=SimpleNamespace(get_output_async=outputs.get),
        output_processor=output_processor,
        log_stats=False,
        logger_manager=None,
        renderer=renderer,
    )
    AsyncLLM._run_output_handler(engine)
    try:
        outputs.put_nowait(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="0",
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                        mm_cache_miss_hashes=[bad_hash],
                        error=RequestError(
                            code="validation_error", message=mismatch_msg
                        ),
                    ),
                    EngineCoreOutput(
                        request_id="1",
                        new_token_ids=[],
                        finish_reason=FinishReason.STOP,
                        pooling_output=torch.tensor([3.0, 4.0]),
                    ),
                ]
            )
        )
        with pytest.raises(VLLMValidationError, match=mismatch_msg):
            await asyncio.wait_for(collectors[0].get(), timeout=5)

        succeeded = await asyncio.wait_for(collectors[1].get(), timeout=5)
        assert isinstance(succeeded, PoolingRequestOutput)
        assert succeeded.error is None
        assert torch.equal(succeeded.outputs.data, torch.tensor([3.0, 4.0]))

        renderer._drain_mm_cache_invalidations()
        assert not renderer.mm_processor_cache.is_cached_item(bad_hash)
    finally:
        engine.output_handler.cancel()
        with suppress(asyncio.CancelledError):
            await engine.output_handler
        renderer.shutdown()


_SLEEP_VISION_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@create_new_process_for_each_test()
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="sleep mode regression requires a CUDA GPU",
)
def test_sleep_wake_preserves_mm_cache_consistency():
    """Regression for vllm-project/vllm#42995."""
    from vllm import LLM, SamplingParams
    from vllm.assets.image import ImageAsset
    from vllm.inputs import TextPrompt

    image = ImageAsset("stop_sign").pil_image
    prompt: TextPrompt = {
        "prompt": _SLEEP_VISION_PROMPT,
        "multi_modal_data": {"image": image},
    }
    sampling_params = SamplingParams(temperature=0, max_tokens=8)

    llm = LLM(
        model="Qwen/Qwen2-VL-2B-Instruct",
        enable_sleep_mode=True,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
    )

    llm.generate([prompt], sampling_params)
    llm.sleep(level=1)
    llm.wake_up()
    output2 = llm.generate([prompt], sampling_params)
    assert output2[0].outputs[0].text
