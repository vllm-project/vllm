# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCore.get_kv_cache_group_metadata().

Exercises the building of the data structure logic directly against fabricated
KVCacheConfig/KVCacheGroupSpec objects with no engine/model construction.
"""

import torch

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


class _FakeScheduler:
    def __init__(self, kv_cache_config: KVCacheConfig | None):
        self.kv_cache_config = kv_cache_config


class _FakeEngineCore:
    def __init__(self, kv_cache_config: KVCacheConfig | None):
        self.scheduler = _FakeScheduler(kv_cache_config)


def test_get_kv_cache_group_metadata_no_config():
    engine_core = _FakeEngineCore(kv_cache_config=None)
    assert EngineCore.get_kv_cache_group_metadata(engine_core) == []


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
    engine_core = _FakeEngineCore(kv_cache_config)

    metadata = EngineCore.get_kv_cache_group_metadata(engine_core)

    assert metadata == [
        {
            "group_id": 0,
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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
    engine_core = _FakeEngineCore(kv_cache_config)

    (group,) = EngineCore.get_kv_cache_group_metadata(engine_core)

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
