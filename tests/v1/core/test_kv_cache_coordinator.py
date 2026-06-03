# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool, CompactBlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCachePoolConfig,
    KVCacheTensor,
    MambaSpec,
    MemoryModel,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def make_full_attention_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float32,
    )


def make_sliding_window_spec(block_size: int = 16) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float32,
        sliding_window=128,
    )


def make_mamba_spec(block_size: int = 16) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
    )


def make_coordinator(config: KVCacheConfig, enable_caching: bool = False):
    return get_kv_cache_coordinator(
        kv_cache_config=config,
        max_model_len=128,
        max_num_batched_tokens=128,
        use_eagle=False,
        enable_caching=enable_caching,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
    )


def make_kv_cache_manager_config(
    block_size: int = 4, num_blocks: int = 3
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer"], make_full_attention_spec(block_size))
        ],
    )


def make_multi_pool_kv_cache_manager_config() -> KVCacheConfig:
    block_size = 4
    full_spec = make_full_attention_spec(block_size)
    mamba_spec = make_mamba_spec(block_size)
    return KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention_layer"], full_spec),
            KVCacheGroupSpec(["mamba_layer"], mamba_spec),
        ],
        pool_configs=(
            KVCachePoolConfig(
                pool_id=0,
                memory_model=MemoryModel.TOKEN_PROPORTIONAL,
                group_ids=(0,),
                num_blocks=3,
                accounting_page_size_bytes=full_spec.accounting_page_size_bytes,
                physical_page_size_bytes=full_spec.physical_page_size_bytes,
            ),
            KVCachePoolConfig(
                pool_id=1,
                memory_model=MemoryModel.REQUEST_CONSTANT,
                group_ids=(1,),
                num_blocks=2,
                accounting_page_size_bytes=mamba_spec.accounting_page_size_bytes,
                physical_page_size_bytes=mamba_spec.physical_page_size_bytes,
            ),
        ),
        group_to_pool_id=(0, 1),
    )


def make_request(request_id: str = "request", num_tokens: int = 4) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[0] * num_tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )


def test_attention_free_config_keeps_legacy_shared_pool_alias():
    coordinator = make_coordinator(
        KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])
    )

    assert coordinator.block_pool is coordinator._block_pools[0]
    assert coordinator._group_to_pool == ()
    assert coordinator.single_type_managers == ()
    assert coordinator.get_num_free_blocks_by_pool() == ()
    # The legacy coordinator still owns one shared BlockPool, but
    # attention-free configs have no KV cache pool metadata and no managers.
    assert (
        coordinator.get_num_blocks_to_allocate_by_pool(
            request_id="request",
            num_tokens=32,
            new_computed_blocks=(),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_tokens_main_model=32,
        )
        == ()
    )
    assert (
        coordinator.get_num_blocks_to_allocate(
            request_id="request",
            num_tokens=32,
            new_computed_blocks=(),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_tokens_main_model=32,
        )
        == 0
    )


def test_single_group_config_maps_manager_to_legacy_pool_alias():
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[KVCacheTensor(size=100, shared_by=["layer_0"])],
        kv_cache_groups=[KVCacheGroupSpec(["layer_0"], make_full_attention_spec())],
    )

    coordinator = make_coordinator(config, enable_caching=True)

    assert coordinator.block_pool is coordinator._block_pools[0]
    assert coordinator._group_to_pool == (coordinator.block_pool,)
    assert coordinator.single_type_managers[0].block_pool is coordinator.block_pool
    assert coordinator.get_num_free_blocks_by_pool() == (9,)
    assert coordinator.get_num_blocks_to_allocate_by_pool(
        request_id="request",
        num_tokens=32,
        new_computed_blocks=((),),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=32,
    ) == (2,)
    assert (
        coordinator.get_num_blocks_to_allocate(
            request_id="request",
            num_tokens=32,
            new_computed_blocks=((),),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_tokens_main_model=32,
        )
        == 2
    )


def test_multi_group_single_pool_config_maps_all_managers_to_legacy_pool_alias():
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=100, shared_by=["layer_0"]),
            KVCacheTensor(size=100, shared_by=["layer_1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_0"], make_full_attention_spec()),
            KVCacheGroupSpec(["layer_1"], make_sliding_window_spec()),
        ],
    )

    coordinator = make_coordinator(config)

    assert coordinator.block_pool is coordinator._block_pools[0]
    assert coordinator._group_to_pool == (
        coordinator.block_pool,
        coordinator.block_pool,
    )
    assert all(
        manager.block_pool is coordinator.block_pool
        for manager in coordinator.single_type_managers
    )
    assert coordinator.get_num_free_blocks_by_pool() == (9,)
    assert coordinator.get_num_blocks_to_allocate_by_pool(
        request_id="request",
        num_tokens=32,
        new_computed_blocks=((), ()),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=32,
    ) == (4,)
    assert (
        coordinator.get_num_blocks_to_allocate(
            request_id="request",
            num_tokens=32,
            new_computed_blocks=((), ()),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_tokens_main_model=32,
        )
        == 4
    )


def test_multi_pool_config_without_prefix_cache_builds_configured_pools():
    full_spec = make_full_attention_spec()
    mamba_spec = make_mamba_spec()
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_0"], full_spec),
            KVCacheGroupSpec(["layer_1"], mamba_spec),
        ],
        pool_configs=(
            KVCachePoolConfig(
                pool_id=0,
                memory_model=MemoryModel.TOKEN_PROPORTIONAL,
                group_ids=(0,),
                num_blocks=10,
                accounting_page_size_bytes=full_spec.accounting_page_size_bytes,
                physical_page_size_bytes=full_spec.physical_page_size_bytes,
            ),
            KVCachePoolConfig(
                pool_id=1,
                memory_model=MemoryModel.REQUEST_CONSTANT,
                group_ids=(1,),
                num_blocks=3,
                accounting_page_size_bytes=mamba_spec.accounting_page_size_bytes,
                physical_page_size_bytes=mamba_spec.physical_page_size_bytes,
            ),
        ),
        group_to_pool_id=(0, 1),
    )

    coordinator = make_coordinator(config, enable_caching=False)

    assert isinstance(coordinator._block_pools[0], BlockPool)
    assert isinstance(coordinator._block_pools[1], CompactBlockPool)
    assert coordinator.block_pool is coordinator._block_pools[0]
    assert coordinator._group_to_pool == coordinator._block_pools
    assert coordinator.get_num_free_blocks_by_pool() == (9, 2)
    # Full attention remains token-proportional: 32 tokens / block_size 16 = 2.
    # Request-constant Mamba uses one compact state block per request.
    assert coordinator.get_num_blocks_to_allocate_by_pool(
        request_id="request",
        num_tokens=32,
        new_computed_blocks=((), ()),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=32,
    ) == (2, 1)


def test_has_enough_free_blocks_by_pool_uses_pool_tuple():
    manager = KVCacheManager(
        kv_cache_config=make_kv_cache_manager_config(num_blocks=3),
        max_model_len=16,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_caching=False,
    )

    # num_blocks=3 includes the null sentinel, leaving two allocatable blocks.
    assert manager.coordinator.get_num_free_blocks_by_pool() == (2,)
    assert manager._has_enough_free_blocks_by_pool((2,))
    assert not manager._has_enough_free_blocks_by_pool((3,))


def test_allocate_slots_uses_pool_aware_accounting_path():
    manager = KVCacheManager(
        kv_cache_config=make_kv_cache_manager_config(num_blocks=3),
        max_model_len=16,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_caching=False,
    )
    request = make_request(num_tokens=4)
    manager.coordinator.get_num_blocks_to_allocate = Mock(
        side_effect=AssertionError("legacy scalar accounting should not be used")
    )
    get_num_blocks_to_allocate_by_pool = Mock(
        wraps=manager.coordinator.get_num_blocks_to_allocate_by_pool
    )
    manager.coordinator.get_num_blocks_to_allocate_by_pool = (
        get_num_blocks_to_allocate_by_pool
    )

    blocks = manager.allocate_slots(request, num_new_tokens=4)

    assert blocks is not None
    assert blocks.get_block_ids() == ([1],)
    manager.coordinator.get_num_blocks_to_allocate.assert_not_called()
    get_num_blocks_to_allocate_by_pool.assert_called_once()


def test_allocate_slots_checks_each_pool_independently():
    manager = KVCacheManager(
        kv_cache_config=make_multi_pool_kv_cache_manager_config(),
        max_model_len=16,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_caching=False,
    )

    blocks = manager.allocate_slots(make_request("first", num_tokens=4), 4)

    assert blocks is not None
    assert blocks.get_block_ids() == ([1], [1])
    assert manager.coordinator.get_num_free_blocks_by_pool() == (1, 0)
    assert manager.allocate_slots(make_request("second", num_tokens=4), 4) is None


def test_multi_pool_config_with_prefix_cache_is_rejected_until_pool_aware():
    full_spec = make_full_attention_spec()
    sliding_spec = make_sliding_window_spec()
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_0"], full_spec),
            KVCacheGroupSpec(["layer_1"], sliding_spec),
        ],
        pool_configs=(
            KVCachePoolConfig(
                pool_id=0,
                memory_model=MemoryModel.TOKEN_PROPORTIONAL,
                group_ids=(0,),
                num_blocks=10,
                accounting_page_size_bytes=full_spec.accounting_page_size_bytes,
                physical_page_size_bytes=full_spec.physical_page_size_bytes,
            ),
            KVCachePoolConfig(
                pool_id=1,
                memory_model=MemoryModel.TOKEN_PROPORTIONAL,
                group_ids=(1,),
                num_blocks=10,
                accounting_page_size_bytes=sliding_spec.accounting_page_size_bytes,
                physical_page_size_bytes=sliding_spec.physical_page_size_bytes,
            ),
        ),
        group_to_pool_id=(0, 1),
    )

    with pytest.raises(NotImplementedError, match="prefix caching is disabled"):
        make_coordinator(config, enable_caching=True)
