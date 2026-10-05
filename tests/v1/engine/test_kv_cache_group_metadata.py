# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

BASE_BLOCK_SIZE = 1536
DCP_WORLD_SIZE = 8


def _make_engine_core(kv_cache_config: KVCacheConfig | None) -> EngineCore:
    """EngineCore whose managers report each group's spec block size."""
    groups = [] if kv_cache_config is None else kv_cache_config.kv_cache_groups
    managers = [SimpleNamespace(block_size=g.kv_cache_spec.block_size) for g in groups]
    kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=managers)
    )
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = SimpleNamespace(
        kv_cache_config=kv_cache_config, kv_cache_manager=kv_cache_manager
    )
    return engine_core


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


def test_get_kv_cache_group_metadata_no_config():
    engine_core = _make_engine_core(None)
    assert engine_core.get_kv_cache_group_metadata() == []


def test_get_kv_cache_group_metadata_full_attention():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=None,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0", "layer.1"], spec),
        ],
    )
    engine_core = _make_engine_core(kv_cache_config)

    metadata = engine_core.get_kv_cache_group_metadata()

    assert metadata == [
        {
            "group_idx": 0,
            "kind": "full_attention",
            "block_size": 16,
            "sliding_window": None,
            "attention_chunk_size": None,
            "layer_count": 2,
            "layer_names": ["layer.0", "layer.1"],
            "num_kv_heads": 8,
            "head_size": 128,
            "head_size_v": 128,
            "dtype": "bfloat16",
            "page_size_bytes": spec.page_size_bytes,
            "cache_dtype_str": None,
            "sink_len": None,
            "shapes": None,
            "dtypes": None,
            "mamba_type": None,
            "mamba_cache_mode": None,
            "layer_specs": None,
        }
    ]


def test_get_kv_cache_group_metadata_mla_attention():
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="fp8_ds_mla",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["kind"] == "mla_attention"
    assert group["layer_names"] == ["layer.0"]
    assert group["head_size_v"] == 576
    assert group["cache_dtype_str"] == "fp8_ds_mla"
    assert group["sink_len"] is None
    assert group["shapes"] is None
    assert group["mamba_type"] is None
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_chunked_local_attention():
    spec = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        attention_chunk_size=2048,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["kind"] == "chunked_local_attention"
    assert group["layer_names"] == ["layer.0"]
    assert group["attention_chunk_size"] == 2048
    assert group["sliding_window"] is None
    assert group["sink_len"] is None
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_sliding_window():
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=512,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["kind"] == "sliding_window"
    assert group["layer_names"] == ["layer.0"]
    assert group["sliding_window"] == 512
    assert group["attention_chunk_size"] is None
    assert group["head_size_v"] == 128
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_cross_attention():
    spec = CrossAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["kind"] == "cross_attention"
    assert group["layer_names"] == ["layer.0"]
    assert group["num_kv_heads"] == 8
    assert group["head_size"] == 128
    assert group["sliding_window"] is None
    assert group["attention_chunk_size"] is None
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_unhandled_spec_degrades_gracefully():
    """EncoderOnlyAttentionSpec has none of the union specific attributes
    (sliding_window, attention_chunk_size, cache_dtype_str, sink_len,
    shapes/dtypes/mamba_type). Serialization must not raise and should fall
    back to None for every attribute the spec doesn't define."""
    spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["kind"] == "encoder_only_attention"
    assert group["sliding_window"] is None
    assert group["attention_chunk_size"] is None
    assert group["cache_dtype_str"] is None
    assert group["sink_len"] is None
    assert group["shapes"] is None
    assert group["mamba_type"] is None
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_sink_full_attention():
    spec = SinkFullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
        sink_len=4,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["kind"] == "sink_full_attention"
    assert group["layer_names"] == ["layer.0"]
    assert group["sliding_window"] == 2048
    assert group["sink_len"] == 4
    assert group["head_size_v"] == 128
    assert group["cache_dtype_str"] is None
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_mamba():
    spec = MambaSpec(
        block_size=1,
        shapes=((16, 128), (5, 32, 32)),
        dtypes=(torch.float32, torch.bfloat16),
        num_speculative_blocks=0,
        mamba_cache_mode="none",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    assert group["layer_count"] == 1
    assert group["layer_names"] == ["layer.0"]
    assert group["num_kv_heads"] is None
    assert group["head_size"] is None
    assert group["head_size_v"] is None
    assert group["dtype"] is None
    assert group["page_size_bytes"] == spec.page_size_bytes
    assert group["attention_chunk_size"] is None
    assert group["cache_dtype_str"] is None
    assert group["sink_len"] is None
    assert group["shapes"] == [[16, 128], [5, 32, 32]]
    assert group["dtypes"] == ["float32", "bfloat16"]
    assert group["mamba_type"] == "mamba2"
    assert group["mamba_cache_mode"] == "none"
    assert group["layer_specs"] is None


def test_get_kv_cache_group_metadata_uniform_type():
    spec_a = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        attention_chunk_size=2048,
    )
    spec_b = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
        attention_chunk_size=2048,
    )
    spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer.0": spec_a, "layer.1": spec_b},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0", "layer.1"], spec)],
    )
    engine_core = _make_engine_core(kv_cache_config)

    (group,) = engine_core.get_kv_cache_group_metadata()

    # Both sub specs are chunked_local_attention, so the group level kind
    # collapses to that single kind. Per layer head_size differs, so the
    # group level attention fields are omitted in favor of `layer_specs`.
    assert group["kind"] == "chunked_local_attention"
    assert group["layer_count"] == 2
    assert group["layer_names"] == ["layer.0", "layer.1"]
    assert group["num_kv_heads"] is None
    assert group["head_size"] is None
    assert group["dtype"] is None

    assert group["layer_specs"] == [
        {
            "layer_names": ["layer.0"],
            "kind": "chunked_local_attention",
            "block_size": 16,
            "sliding_window": None,
            "attention_chunk_size": 2048,
            "num_kv_heads": 8,
            "head_size": 128,
            "head_size_v": None,
            "dtype": "float16",
            "page_size_bytes": spec_a.page_size_bytes,
            "cache_dtype_str": None,
            "sink_len": None,
            "shapes": None,
            "dtypes": None,
            "mamba_type": None,
            "mamba_cache_mode": None,
        },
        {
            "layer_names": ["layer.1"],
            "kind": "chunked_local_attention",
            "block_size": 16,
            "sliding_window": None,
            "attention_chunk_size": 2048,
            "num_kv_heads": 8,
            "head_size": 64,
            "head_size_v": None,
            "dtype": "float16",
            "page_size_bytes": spec_b.page_size_bytes,
            "cache_dtype_str": None,
            "sink_len": None,
            "shapes": None,
            "dtypes": None,
            "mamba_type": None,
            "mamba_cache_mode": None,
        },
    ]
