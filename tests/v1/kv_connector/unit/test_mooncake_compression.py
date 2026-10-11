# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration, cache-layout and memory-budget contracts for compression."""

import os
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import (
    config as compression_config_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import (
    cpp_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import (
    memory as compression_memory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import (
    pipeline as compression_pipeline,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.config import (
    MooncakeCompressionConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.layout import (
    ALIGNMENT,
    build_compression_layout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.memory import (
    build_memory_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.scores import (
    build_head_selection,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.tokens import (
    chunk_token_counts,
    chunk_window,
    suffix_token_counts,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _compression_total_chunks,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def _config(**overrides):
    return MooncakeCompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True, **overrides}}
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"compression": None},
        {"compression": {"enable-compression-pipeline": False, "unknown": 1}},
    ],
)
def test_disabled_compression_needs_no_gpu_or_dependencies(extra):
    config = MooncakeCompressionConfig.from_extra_config(extra)
    assert not config.available and config.transport == "raw"
    assert (
        build_memory_plan(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra)
            ),
            {},
        )
        is None
    )


def test_compression_defaults_and_pipeline_normalization():
    config = _config()
    assert config.transport == "compression"
    assert config.pipeline == ("transformer", "quantizer", "codec")
    assert (config.slot_count, config.logical_blocks_per_chunk) == (2, 1)
    assert _config(pipeline=["quantizer"]).pipeline == ("quantizer",)


@pytest.mark.parametrize(
    "extra, transport, slots, blocks",
    [
        ({"raw_aggregate": True}, "raw_aggregate", 2, 160),
        (
            {"raw_aggregate": {"slot_count": 3, "logical_blocks_per_chunk": 25}},
            "raw_aggregate",
            3,
            25,
        ),
        (
            {"raw_agg": {"slot_count": 1, "logical_blocks_per_chunk": 4}},
            "raw_aggregate",
            1,
            4,
        ),
        (
            {"raw_aggregate": {"slot_count": 2}, "raw_agg": {"slot_count": 7}},
            "raw_aggregate",
            2,
            160,
        ),
        (
            {
                "raw_aggregate": False,
                "compression": {"enable-compression-pipeline": True},
            },
            "compression",
            2,
            1,
        ),
    ],
)
def test_transport_selection(extra, transport, slots, blocks):
    config = MooncakeCompressionConfig.from_extra_config(extra)
    assert config.transport == transport
    assert (config.slot_count, config.logical_blocks_per_chunk) == (slots, blocks)
    if transport == "raw_aggregate":
        assert config.pipeline == ("aggregate",)
        assert not (
            config.uses_codec or config.uses_quantizer or config.uses_transformer
        )


@pytest.mark.parametrize(
    "compression, warning_count",
    [
        ({}, 0),
        ({"enable-compression-pipeline": False}, 0),
        ({"enable-compression-pipeline": True, "split_type": "head"}, 1),
    ],
)
def test_shadowed_compression_warns_once(
    monkeypatch, caplog_vllm, compression, warning_count
):
    monkeypatch.setattr(
        compression_config_module, "_RAW_AGGREGATE_WARNING_EMITTED", False
    )
    caplog_vllm.set_level("WARNING")
    for _ in range(2):
        MooncakeCompressionConfig.from_extra_config(
            {"raw_aggregate": True, "compression": compression}
        )
    warnings = [
        r.getMessage()
        for r in caplog_vllm.records
        if "raw_aggregate transport selected" in r.getMessage()
    ]
    assert len(warnings) == warning_count
    if warnings:
        assert "split_type" in warnings[0]


@pytest.mark.parametrize(
    "raw",
    [
        {"slot_count": 0},
        {"logical_blocks_per_chunk": 0},
        {"slot": 2},
        "yes",
    ],
)
def test_invalid_aggregate_config_fails_at_startup(raw):
    with pytest.raises(ValueError):
        MooncakeCompressionConfig.from_extra_config({"raw_aggregate": raw})


def test_layout_excludes_non_attention_state_and_aligns_head_sections():
    specs = {
        "model.layers.0.self_attn": _fa_spec(),
        "model.layers.1": MambaSpec(
            block_size=16, shapes=((4, 8),), dtypes=(torch.bfloat16,), num_heads=4
        ),
    }
    config = _config(split_type="head")
    layout = build_compression_layout(specs, config, {0: 1}, 2)
    assert {p.layer_name for p in layout.layer_parts} == {"model.layers.0.self_attn"}
    assert layout.scratch_stride % ALIGNMENT == layout.u8_base % ALIGNMENT == 0
    for part in layout.layer_parts:
        levels = (
            (config.low_key_num_levels, config.high_key_num_levels)
            if part.kv_part == 0
            else (config.low_value_num_levels, config.high_value_num_levels)
        )
        assert [(s.precision, s.head_count, s.num_levels) for s in part.sections] == [
            ("low", 1, levels[0]),
            ("high", 1, levels[1]),
        ]
        for section in part.sections:
            assert section.kv_part == part.kv_part
            assert (
                section.aux_min_offset % ALIGNMENT
                == section.aux_scale_offset % ALIGNMENT
                == 0
            )


def test_sparse_group_indices_are_compacted():
    specs = {f"model.layers.{i}.self_attn": _fa_spec() for i in range(2)}
    layout = build_compression_layout(
        specs,
        _config(pipeline=["aggregate"]),
        {0: 0, 1: 0},
        2,
        group_indices=dict(zip(specs, (2, 5))),
    )
    assert layout.index_rows_by_group == {2: 0, 5: 1}


@pytest.mark.parametrize(
    "stages",
    [
        ["transformer"],
        ["quantizer"],
        ["transformer", "quantizer"],
        [],
        ["aggregate"],
        ["transformer", "codec"],
        ["quantizer", "codec"],
        ["codec"],
        ["transformer", "quantizer", "codec"],
    ],
)
def test_memory_budget_probes_only_enabled_dependencies(monkeypatch, stages):
    from vllm.distributed import parallel_state
    from vllm.distributed.kv_transfer.kv_connector import utils
    from vllm.v1.worker import utils as worker_utils

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(utils, "get_current_attn_backends", lambda config: [])
    monkeypatch.setattr(
        utils,
        "get_attn_backends_and_specs_for_layers",
        lambda config, specs, backends: (backends, [None] * len(backends)),
    )
    monkeypatch.setattr(
        worker_utils, "select_common_block_size", lambda size, backends, specs: size
    )
    calls = []

    def probe(layout, device, sizes, warm_hadamard):
        calls.append((tuple(sizes), warm_hadamard))
        return 1024, (8192, 8192), 2048

    monkeypatch.setattr(compression_memory, "_probe_operators", probe)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "compression": {
                    "enable-compression-pipeline": True,
                    "pipeline": stages,
                    "slot_count": 2,
                }
            }
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=1)),
    )
    plan = build_memory_plan(config, {"model.layers.0.self_attn": _fa_spec()})
    assert plan is not None
    assert plan.reserved_bytes >= plan.metadata_bytes + 2 * plan.layout.arena_bytes
    if "codec" in stages:
        assert calls == [
            (
                plan.layout.packed_bytes_by_part,
                "transformer" in stages and not cpp_backend.enabled(),
            )
        ]
        assert plan.codec_workspace_bytes == 2048
        assert plan.retained_driver_bytes == 4096
        assert plan.layout.arena_bytes >= 8192
    else:
        assert not calls
        assert plan.codec_workspace_bytes == plan.retained_driver_bytes == 0
    assert bool(plan.layout.aux_arena_bytes) == ("quantizer" in stages)


def _fa_spec(num_kv_heads: int = 2) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=8,
        dtype=torch.bfloat16,
    )


def test_compression_chunk_window_alternates_kv_parts():
    assert _compression_total_chunks(0, 1) == 0
    assert _compression_total_chunks(1, 1) == 2
    assert _compression_total_chunks(5, 2) == 6
    assert _compression_total_chunks(5, 3) == 4

    assert chunk_window(0, 5, 2) == (0, 2, 0)
    assert chunk_window(1, 5, 2) == (0, 2, 1)
    assert chunk_window(2, 5, 2) == (2, 2, 0)
    assert chunk_window(4, 5, 2) == (4, 1, 0)
    assert chunk_window(5, 5, 2) == (4, 1, 1)

    # With lbc=1 every logical block gets its own K and V chunk.
    assert [chunk_window(chunk_id, 3, 1)[2] for chunk_id in range(6)] == [
        0,
        1,
        0,
        1,
        0,
        1,
    ]


def test_suffix_token_counts():
    counts = suffix_token_counts(1000, [[0, 1, 2], [7, 8, 9, 10]], {0: 16, 1: 32})
    assert counts == {0: 1000 - (63 - 3) * 16, 1: 1000 - (32 - 4) * 32}
    # A full prefix hit has no transferred blocks and therefore no suffix.
    assert suffix_token_counts(1000, [], {0: 16}) == {0: 0}


def test_chunk_token_counts():
    valid_tokens = {3: 1371}
    block_sizes = {3: 784}
    assert chunk_token_counts(valid_tokens, block_sizes, 0, 1) == {3: 784}
    assert chunk_token_counts(valid_tokens, block_sizes, 1, 1) == {3: 587}
    assert chunk_token_counts(valid_tokens, block_sizes, 0, 2) == {3: 1371}


def test_head_selection_layer_split_needs_no_scores():
    specs = {
        "model.layers.3.self_attn": _fa_spec(),
        "model.layers.7.self_attn": _fa_spec(),
    }
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "layer",
            }
        }
    )
    selection = build_head_selection(specs, config, tp_rank=1, tp_size=2)
    assert selection.low_counts == {3: 0, 7: 0}
    assert selection.local_orders == {3: (0, 1), 7: (0, 1)}


def test_head_selection_uses_scores_and_tp_slice():
    specs = {
        "model.layers.3.self_attn": _fa_spec(),
        "model.layers.7.self_attn": _fa_spec(),
    }
    scores: NDArray[np.float32] = np.ones((64, 4), dtype=np.float32)
    scores[3, 3] = 0.0  # layer 3, global head 3 -> rank 1 local head 1
    scores[7, 2] = 0.0  # layer 7, global head 2 -> rank 1 local head 0
    config = MooncakeCompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "head",
                "hybrid_ratio": 0.2,
            }
        }
    )
    selection = build_head_selection(specs, config, tp_rank=1, tp_size=2, scores=scores)
    assert selection.low_counts == {3: 1, 7: 1}
    assert selection.local_orders == {3: (1, 0), 7: (0, 1)}


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot_count", 0),
        ("logical_blocks_per_chunk", 0),
        ("hybrid_ratio", -0.1),
        ("hybrid_ratio", 1.1),
        ("axis_key", "unknown"),
        ("axis_value", "unknown"),
        ("codec", "unknown"),
        ("transformer", "unknown"),
        ("pipeline", ["unknown"]),
        ("pipeline", ["codec", "codec"]),
        ("pipeline", ["codec", "quantizer"]),
        ("pipeline", ["quantizer", "transformer"]),
        ("high_key_num_levels", 1),
        ("high_key_num_levels", 257),
        ("low_value_num_levels", 512),
        ("low_key_num_levels", 2.5),
    ],
)
def test_unsupported_compression_config_fails_at_startup(field, value):
    with pytest.raises((ValueError, TypeError)):
        MooncakeCompressionConfig.from_extra_config(
            {"compression": {"enable-compression-pipeline": True, field: value}}
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float8_e4m3fn])
def test_unsupported_cache_dtype_fails_before_allocation(dtype):
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=dtype)
    config = MooncakeCompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True, "split_type": "layer"}}
    )
    with pytest.raises(ValueError, match="BF16"):
        build_compression_layout({"model.layers.0.self_attn": spec}, config, {}, 1)


@pytest.mark.parametrize(
    "geometry",
    [
        {"tokens_per_state": Fraction(1, 2)},
        {"num_head_slots": 1, "state_content_bytes": 64},
        {"num_head_slots": 2},
        {"state_content_bytes": 64},
    ],
)
def test_packed_or_pooled_bf16_cache_rejected_before_allocation(geometry):
    spec = replace(_fa_spec(), **geometry)
    with pytest.raises(ValueError, match="unsupported cache geometry"):
        build_compression_layout(
            {"model.layers.0.self_attn": spec}, MooncakeCompressionConfig(), {}, 1
        )


def test_unknown_explicit_backend_does_not_silently_use_python(monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpo")
    with pytest.raises(ValueError, match="Unknown KVSERVE_COMPRESSION_BACKEND"):
        cpp_backend.enabled()


@pytest.mark.parametrize("tp_rank", range(8))
def test_head_selection_maps_replicated_kv_heads(tp_rank):
    config = MooncakeCompressionConfig(
        enabled=True, split_type="head", hybrid_ratio=0.5
    )
    selection = build_head_selection(
        {"model.layers.0.self_attn": _fa_spec(num_kv_heads=1)},
        config,
        tp_rank,
        8,
        scores=np.arange(4, dtype=np.float32).reshape(1, 4),
    )
    assert selection.local_orders == {0: (0,)}
    assert selection.low_counts == {0: int(tp_rank // 2 < 2)}


@pytest.mark.parametrize(
    "scores", [np.ones(3), np.full((1, 4), np.nan), np.ones((1, 3))]
)
def test_head_selection_rejects_invalid_score_shape_and_values(scores):
    config = MooncakeCompressionConfig(enabled=True, split_type="head")
    with pytest.raises(ValueError, match="Head score"):
        build_head_selection(
            {"model.layers.0.self_attn": _fa_spec()}, config, 0, 2, scores
        )


def test_cpp_jit_uses_worker_device_and_respects_explicit_arch(monkeypatch):
    device = torch.device("cuda", 3)
    capability = Mock(return_value=(12, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(cpp_backend, "load", Mock())
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cpp_backend, "_module", None)
    cpp_backend.load_kvs_compress(device)
    capability.assert_called_once_with(device)

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.0"
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0;12.0")
    monkeypatch.setattr(cpp_backend, "_module", None)
    cpp_backend.load_kvs_compress(device)
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0;12.0"


def test_cpp_cache_binding_loads_backend_and_keeps_asymmetric_kv_views(monkeypatch):
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpp")
    load = Mock()
    monkeypatch.setattr(cpp_backend, "load_kvs_compress", load)
    parts = tuple(
        SimpleNamespace(
            layer_name="layer",
            kv_part=i,
            head_size=size,
            dtype=torch.bfloat16,
            num_heads=2,
            block_size=1,
            sections=(),
        )
        for i, size in enumerate((8, 16))
    )
    device = torch.device("cuda", 3)
    pipeline = compression_pipeline.CompressionPipeline(
        SimpleNamespace(
            layer_parts=parts,
            fa_group_indices={0},
            parts=lambda kv: tuple(part for part in parts if part.kv_part == kv),
        ),
        "LBHNC",
        1,
        [],
        None,
        SimpleNamespace(device=device),
        {},
        {},
        stages=("quantizer",),
    )
    cache = torch.arange(96, dtype=torch.bfloat16).reshape(2, 2, 1, 24)
    pipeline.bind_kv_caches({"layer": cache})
    load.assert_called_once_with(device)
    for part, expected in zip(parts, (cache[..., :8], cache[..., 8:])):
        view = pipeline._cache_part(part)
        assert view.data_ptr() == expected.data_ptr()
        torch.testing.assert_close(view, expected)
    with pytest.raises(ValueError, match="Unsupported compression cache"):
        pipeline.bind_kv_caches({"layer": cache.squeeze(2)})

    # Cache rebinding must discard metadata containing old cache pointers.
    build = Mock(side_effect=[torch.tensor([1]), torch.tensor([2])])
    monkeypatch.setattr(cpp_backend, "build_meta", build)
    pipeline.slots.append(SimpleNamespace())
    original = pipeline._cpp_meta(0, 0)
    assert pipeline._cpp_meta(0, 0) is original
    assert build.call_count == 1
    pipeline.bind_kv_caches({"layer": cache.clone()})
    assert pipeline._cpp_meta(0, 0) is not original
    assert build.call_count == 2


def test_mixed_fa_block_sizes_are_rejected_before_gpu_allocation():
    config = MooncakeCompressionConfig(enabled=True)
    specs = {
        "model.layers.0.self_attn": _fa_spec(),
        "model.layers.1.self_attn": FullAttentionSpec(
            block_size=32, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
        ),
    }
    with pytest.raises(ValueError, match="common logical block size"):
        build_compression_layout(specs, config, {}, 2)


@pytest.mark.parametrize("cache_layout", ["NHD", "LBNHC"])
def test_pipeline_rejects_unsupported_cache_layout(cache_layout):
    with pytest.raises(ValueError, match="LBHNC/HND KV cache layout"):
        compression_pipeline.CompressionPipeline(
            SimpleNamespace(), cache_layout, 1, [], None, SimpleNamespace(), {}, {}
        )
