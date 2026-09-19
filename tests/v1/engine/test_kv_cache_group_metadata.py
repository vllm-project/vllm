# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

pytestmark = pytest.mark.cpu_test

BASE_BLOCK_SIZE = 1536
DCP_WORLD_SIZE = 8


def _make_engine_core_with_dcp_manager() -> EngineCore:
    mla_spec = MLAAttentionSpec(
        block_size=BASE_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        block_size=BASE_BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla"], mla_spec),
            KVCacheGroupSpec(["mamba"], mamba_spec),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=BASE_BLOCK_SIZE * DCP_WORLD_SIZE * 2,
        scheduler_block_size=BASE_BLOCK_SIZE * DCP_WORLD_SIZE,
        hash_block_size=BASE_BLOCK_SIZE,
        dcp_world_size=DCP_WORLD_SIZE,
    )
    scheduler = SimpleNamespace(kv_cache_config=config, kv_cache_manager=manager)
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = scheduler
    return engine_core


def test_kv_cache_group_metadata_uses_effective_dcp_block_sizes() -> None:
    engine_core = _make_engine_core_with_dcp_manager()

    metadata = engine_core.get_kv_cache_group_metadata()

    # DCP scales attention blocks but does not scale replicated Mamba state.
    assert [item["block_size"] for item in metadata] == [12288, 1536]
    assert [item["kind"] for item in metadata] == ["mla_attention", "mamba"]

    managers = engine_core.scheduler.kv_cache_manager.coordinator.single_type_managers
    assert [item["block_size"] for item in metadata] == [
        manager.block_size for manager in managers
    ]
