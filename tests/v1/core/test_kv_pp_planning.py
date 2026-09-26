# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for KV-PP (KV Pipeline Parallel / LayerSplit) KV cache planning."""

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    compute_kv_pp_placement_plan,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

pytestmark = pytest.mark.cpu_test


def _make_full_attn_spec(
    block_size: int = 16, num_kv_heads: int = 8, head_size: int = 128
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _make_mla_spec(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 576,
) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _make_mamba_spec(
    block_size: int = 16,
    d_state: int = 128,
    d_conv: int = 4,
    d_inner: int = 2048,
) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        d_state=d_state,
        d_conv=d_conv,
        d_inner=d_inner,
        dtype=torch.bfloat16,
    )


def test_kv_pp_placement_plan_uniform_layers():
    """Verify balanced layer partitioning across KV-PP ranks for uniform models."""
    num_layers = 32
    layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    spec = _make_full_attn_spec()
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)

    kv_pp_size = 4
    page_size = spec.page_size_bytes
    expected_scratch = 2 * page_size

    # Check plans for all 4 ranks
    for rank in range(kv_pp_size):
        plan = compute_kv_pp_placement_plan([group], kv_pp_size=kv_pp_size, rank=rank)
        assert plan.kv_pp_size == kv_pp_size
        assert plan.rank == rank
        assert len(plan.owned_layer_names) == 8
        assert plan.owned_layer_names == layer_names[rank * 8 : (rank + 1) * 8]
        assert plan.scratch_buffer_bytes == expected_scratch
        assert plan.bytes_for_owned_layers == 8 * page_size
        assert plan.bytes_for_local_draft_caches == 0
        assert plan.bytes_per_logical_block == 8 * page_size + expected_scratch

        for layer_name in plan.owned_layer_names:
            assert plan.is_layer_owned(layer_name)
            assert plan.get_layer_owner_rank(layer_name) == rank


def test_kv_pp_placement_plan_remainder_distribution():
    """Verify remainder layers are distributed contiguously across earlier ranks."""
    num_layers = 33
    layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    spec = _make_full_attn_spec()
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)

    kv_pp_size = 4
    # 33 / 4 -> rank 0 gets 9, ranks 1, 2, 3 get 8
    expected_counts = [9, 8, 8, 8]
    for rank in range(kv_pp_size):
        plan = compute_kv_pp_placement_plan([group], kv_pp_size=kv_pp_size, rank=rank)
        assert len(plan.owned_layer_names) == expected_counts[rank]


def test_kv_pp_placement_plan_with_draft_model():
    """Verify EAGLE draft caches remain rank-local and are excluded."""
    target_layer_names = [f"target.layers.{i}" for i in range(16)]
    draft_layer_names = [f"draft.layers.{i}" for i in range(2)]

    target_spec = _make_full_attn_spec(num_kv_heads=32, head_size=128)
    draft_spec = _make_full_attn_spec(num_kv_heads=8, head_size=128)

    target_group = KVCacheGroupSpec(
        layer_names=target_layer_names, kv_cache_spec=target_spec
    )
    draft_group = KVCacheGroupSpec(
        layer_names=draft_layer_names, kv_cache_spec=draft_spec, is_eagle_group=True
    )

    kv_pp_size = 2
    for rank in range(kv_pp_size):
        plan = compute_kv_pp_placement_plan(
            [target_group, draft_group], kv_pp_size=kv_pp_size, rank=rank
        )
        assert len(plan.owned_layer_names) == 8
        # Draft layers are not in target owned layers
        for draft_layer in draft_layer_names:
            assert draft_layer not in plan.owned_layer_names
        # Draft bytes are tracked locally
        expected_draft_bytes = 2 * draft_spec.page_size_bytes
        assert plan.bytes_for_local_draft_caches == expected_draft_bytes
        assert plan.scratch_buffer_bytes == 2 * target_spec.page_size_bytes
        assert plan.bytes_per_logical_block == (
            8 * target_spec.page_size_bytes
            + expected_draft_bytes
            + 2 * target_spec.page_size_bytes
        )


def test_kv_pp_bytes_per_block_and_capacity_scaling():
    """Verify logical KV block count scales with KV-PP."""
    num_layers = 32
    layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    spec = _make_full_attn_spec()
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
    page_size = spec.page_size_bytes

    # Standard (kv_pp_size=1)
    bytes_per_block_standard = _get_kv_cache_bytes_per_block([group], kv_pp_size=1)
    assert bytes_per_block_standard == 32 * page_size

    # KV-PP with 4 ranks: (32/4 = 8) + 2 scratch layers = 10 layers per logical block
    bytes_per_block_kvpp4 = _get_kv_cache_bytes_per_block([group], kv_pp_size=4)
    assert bytes_per_block_kvpp4 == 10 * page_size

    # KV-PP with 8 ranks: (32/8 = 4) + 2 scratch layers = 6 layers per logical block
    bytes_per_block_kvpp8 = _get_kv_cache_bytes_per_block([group], kv_pp_size=8)
    assert bytes_per_block_kvpp8 == 6 * page_size

    # Verify capacity increases significantly: 32/10 = 3.2x capacity gain for PP4
    available_memory = 32 * page_size * 1000
    blocks_std = available_memory // bytes_per_block_standard
    blocks_pp4 = available_memory // bytes_per_block_kvpp4
    blocks_pp8 = available_memory // bytes_per_block_kvpp8

    assert blocks_std == 1000
    assert blocks_pp4 == 3200
    assert blocks_pp8 == 5333


def test_kv_pp_get_kv_cache_config_from_groups():
    """Verify get_kv_cache_config_from_groups integrates KV-PP correctly."""
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            kv_pipeline_parallel_size=4,
        ),
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    num_layers = 32
    layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    spec = _make_full_attn_spec()
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)

    available_memory = 32 * spec.page_size_bytes * 1000
    config: KVCacheConfig = get_kv_cache_config_from_groups(
        vllm_config, [group], available_memory
    )

    assert config.is_kv_pp_enabled
    assert config.kv_pp_placement is not None
    assert config.kv_pp_placement.kv_pp_size == 4
    assert config.num_blocks == 3200
    # Rank 0 owns first 8 layers
    for i in range(8):
        assert config.is_layer_owned_by_rank(f"model.layers.{i}", rank=0)
    for i in range(8, 32):
        assert not config.is_layer_owned_by_rank(f"model.layers.{i}", rank=0)


def test_kv_pp_invalid_arguments():
    """Verify ValueError on invalid kv_pp_size or rank."""
    spec = _make_full_attn_spec()
    group = KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)

    with pytest.raises(ValueError, match="kv_pp_size must be positive"):
        compute_kv_pp_placement_plan([group], kv_pp_size=0, rank=0)

    with pytest.raises(ValueError, match="out of range"):
        compute_kv_pp_placement_plan([group], kv_pp_size=2, rank=2)
