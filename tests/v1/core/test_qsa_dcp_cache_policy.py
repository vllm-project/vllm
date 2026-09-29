# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.models.qwen4_exp.common.qsa_cache import QSACompressedKeyCache
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 784


class _IndependentSelectorWithoutGenericSlots(MLAAttentionSpec):
    @property
    def has_independent_slot_mapping(self) -> bool:
        return True

    @property
    def uses_slot_mapping(self) -> bool:
        return False


def _main_kv() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _selector(dcp_world_size: int) -> MLAAttentionSpec:
    block_size = BLOCK_SIZE * dcp_world_size
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=8,
        dcp_sharded=False,
        storage_block_size=block_size,
    )


def _qsa_selector(dcp_world_size: int) -> MLAAttentionSpec:
    cache = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=8,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_world_size)
    )
    selector = QSACompressedKeyCache.get_kv_cache_spec(cache, config)
    assert isinstance(selector, MLAAttentionSpec)
    return selector


def _config(dcp_world_size: int) -> VllmConfig:
    config = VllmConfig(
        model_config=ModelConfig(max_model_len=BLOCK_SIZE * 16),
        device_config=DeviceConfig(device="cpu"),
    )
    config.parallel_config.decode_context_parallel_size = dcp_world_size
    config.cache_config.kv_cache_layout = "BLHNC"
    return config


def test_replicated_qsa_caches_opt_out_of_dcp_sharding() -> None:
    ring = CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        dcp_sharded=False,
    )
    assert _main_kv().dcp_sharded
    assert not _qsa_selector(2).dcp_sharded
    assert not ring.dcp_sharded


@pytest.mark.parametrize("dcp_world_size", [2, 4])
@pytest.mark.parametrize("selector_first", [False, True])
def test_unmarked_replicated_cache_stays_separate(
    dcp_world_size: int, selector_first: bool
) -> None:
    main = _main_kv()
    selector = _selector(dcp_world_size)
    specs = (
        {"selector": selector, "main": main}
        if selector_first
        else {"main": main, "selector": selector}
    )
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs, dcp_world_size)
    group = UniformTypeKVCacheSpecs.from_specs(specs, dcp_world_size)
    assert group is None


@pytest.mark.parametrize("dcp_world_size", [2, 4])
@pytest.mark.parametrize("selector_first", [False, True])
def test_qsa_selector_shares_sharded_main_block_table(
    dcp_world_size: int, selector_first: bool
) -> None:
    config = _config(dcp_world_size)
    selector = _qsa_selector(dcp_world_size)
    specs = (
        {"selector": selector, "main": _main_kv()}
        if selector_first
        else {"main": _main_kv(), "selector": selector}
    )
    groups = get_kv_cache_groups(config, specs)

    assert len(groups) == 1
    group = groups[0]
    assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    assert group.kv_cache_spec.block_size == BLOCK_SIZE
    assert group.kv_cache_spec.dcp_sharded
    assert (
        group.kv_cache_spec.max_num_blocks_per_req(
            config, config.model_config.max_model_len
        )
        == 16 // dcp_world_size
    )
    scheduler_config = generate_scheduler_kv_cache_config(
        [KVCacheConfig(128, [], groups)]
    )
    scheduler_spec = scheduler_config.kv_cache_groups[0].kv_cache_spec
    assert scheduler_spec.dcp_sharded
    assert scheduler_spec.block_size == BLOCK_SIZE
    assert type(scheduler_spec) is FullAttentionSpec
    assert selector.storage_block_size == BLOCK_SIZE * dcp_world_size


def test_different_global_spans_stay_separate_in_grouping() -> None:
    config = _config(2)
    selector = replace(_qsa_selector(1), dcp_sharded=False)
    groups = get_kv_cache_groups(config, {"main": _main_kv(), "selector": selector})
    assert len(groups) == 2


@pytest.mark.parametrize("selector_first", [False, True])
def test_independent_selector_cannot_disable_main_slot_mapping(
    selector_first: bool,
) -> None:
    config = _config(2)
    selector = _IndependentSelectorWithoutGenericSlots(
        block_size=BLOCK_SIZE * 2,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=8,
        dcp_sharded=False,
        storage_block_size=BLOCK_SIZE * 2,
    )
    specs = (
        {"selector": selector, "main": _main_kv()}
        if selector_first
        else {"main": _main_kv(), "selector": selector}
    )
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    assert groups[0].kv_cache_spec.uses_slot_mapping


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_selector_storage_block_size_is_pinned_to_its_virtual_block(
    dcp_world_size: int,
) -> None:
    selector = _qsa_selector(dcp_world_size)
    assert (
        selector.storage_block_size
        == selector.block_size
        == (BLOCK_SIZE * dcp_world_size)
    )
    assert selector.dcp_sharded == (dcp_world_size == 1)


def test_dcp1_selector_can_share_a_group_with_main_kv() -> None:
    config = _config(1)
    selector = _qsa_selector(1)
    groups = get_kv_cache_groups(config, {"main": _main_kv(), "selector": selector})
    assert len(groups) == 1
    assert groups[0].layer_names == ["main", "selector"]
